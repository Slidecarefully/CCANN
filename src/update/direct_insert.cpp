// 引入 aligned_file_reader.h：抽象对齐 I/O 与 IORequest，供 SSD/PM 统一读写。
#include "aligned_file_reader.h"
// 引入 libcuckoo/cuckoohash_map.hh：并发哈希表，用于 id2loc、tags 等共享映射。
#include "libcuckoo/cuckoohash_map.hh"
// 引入 ssd_index.h：SSDIndex 的核心类声明与索引布局辅助函数。
#include "ssd_index.h"
// 引入 algorithm：排序、min/max 等通用算法。
#include <algorithm>
// 引入 filesystem：文件存在性、复制、大小等文件系统操作。
#include <filesystem>
// 引入 malloc.h：底层内存分配相关接口。
#include <malloc.h>
// 引入 future：异步任务/线程池相关支持。
#include <future>

// 引入 timer.h：项目内性能计时宏和 Timer。
#include "timer.h"
// 引入 tsl/robin_map.h：高性能 robin hash map，用于候选坐标映射。
#include "tsl/robin_map.h"
// 引入 utils.h：项目通用辅助函数、常量和二进制 I/O 工具。
#include "utils.h"
// 引入 v2/journal.h：CCANN-J baseline 的 journal 数据结构与提交逻辑。
#include "v2/journal.h"
// 引入 v2/page_cache.h：用户态 page cache 及 page 引用管理。
#include "v2/page_cache.h"
// 引入 chrono：标准时间与持续时间工具。
#include <chrono>
// 引入 cmath：浮点数学函数。
#include <cmath>
// 引入 cstdint：固定宽度整数类型。
#include <cstdint>
// 引入 limits：数值边界值，例如 max() 作为哨兵。
#include <limits>
// 引入 omp.h：OpenMP 并行循环与线程编号。
#include <omp.h>
// 引入 tuple：tuple/pair 相关辅助。
#include <tuple>
// 引入 boost/crc.hpp：调试/校验写入内容时使用 CRC32。
#include <boost/crc.hpp>

// 引入 libpmem.h：PMDK libpmem：pmem_memcpy、persist 等 PM 原语。
#include "libpmem.h"
// 引入 linux_aligned_file_reader.h：Linux 下 O_DIRECT、io_uring 与 DAX mmap 的具体 reader。
#include "linux_aligned_file_reader.h"
// 引入 sys/syscall.h：Linux 系统调用编号接口。
#include <sys/syscall.h>
// 引入 unistd.h：pread/pwrite/sleep/close 等 POSIX 接口。
#include <unistd.h>

// 进入 ccann 命名空间，下面实现的函数都属于 CCANN 核心索引模块。
namespace ccann {

// 定义小块 PM 拷贝策略：不立即 drain/flush，并倾向 non-temporal 写；真正的持久化顺序由后续显式 flush/barrier 控制。
#define PMEM_TRANSFER_CACHE (PMEM_F_MEM_NODRAIN | PMEM_F_MEM_NOFLUSH | PMEM_F_MEM_NONTEMPORAL)
// 定义大块 PM 拷贝策略：允许 temporal cache 行为，但同样把 drain/flush 交给调用方统一控制。
#define PMEM_TRANSFER_LARGE (PMEM_F_MEM_NODRAIN | PMEM_F_MEM_NOFLUSH | PMEM_F_MEM_TEMPORAL)
// 定义普通 PM 拷贝策略，只禁止立即 drain；用于 location table 这类稍后统一建立持久化顺序的数据。
#define PMEM_TRANSFER (PMEM_F_MEM_NODRAIN)


  // ============================================================================
  // search_phase
  // 插入的搜索阶段：先为新向量分配稳定的逻辑 ID，并生成 PQ 压缩码；随后按当前 search_mode 在已有图上搜索候选节点，收集候选的精确坐标，最后通过
  // prune_neighbors 得到真正要连接的新邻居集合。这个阶段只决定“新点应该连到谁”，还没有把图结构正式发布到持久化索引。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<typename T, typename TagT>
  uint32_t SSDIndex<T, TagT>::search_phase(const T *point, tsl::robin_set<uint32_t> *deletion_set,
                                           std::vector<Neighbor> &exp_node_info,
                                           tsl::robin_map<uint32_t, T *> &coord_map, std::vector<uint32_t> &new_nhood,
                                           std::vector<uint64_t> &page_ref, std::vector<uint8_t> &out_pq_coords) {
    // 用断言检查这里依赖的内部不变量；失败说明索引布局或并发状态已与预期不一致。
    assert(reader != nullptr);
    // 取得当前线程的 I/O 上下文；SSD 路径通常对应 io_uring/AIO，PM 路径仍复用统一接口。
    void *ctx = reader->get_ctx();
    // 初始化计时器 read_t，后续用于把该阶段开销计入性能 breakdown。
    ANN_INIT_TIMING(read_t);
    // 初始化计时器 update_t，后续用于把该阶段开销计入性能 breakdown。
    ANN_INIT_TIMING(update_t);
    // 原子取得下一个逻辑 vector ID，并推进 cur_id；这个 ID 从现在起标识本次插入。
    uint32_t target_id = cur_id++;
    // write PQ.

    // save target PQ vector into memory data.
    // 把新向量编码成 PQ code；完整坐标会进 graph node，而这份紧凑 code 供后续搜索快速估距。
    std::vector<uint8_t> pq_coords = deflate_vector(point);
    // chunk is dim
    // 开始记录下面这段逻辑的耗时，便于论文中的阶段级性能分解。
    ANN_START_TIMING(update_PQ_vec_time, update_t);
    // 按 target_id 计算该向量在连续 PQ byte array 中的起始位置。
    uint64_t pq_offset = target_id * n_chunks;
    {
      // PQ vector 使用共享 std::vector 存储，扩容/写入时用互斥锁避免多个插入线程同时 resize 导致地址失效。
      static std::mutex pq_mu;
      // 进入 PQ 数组的短临界区，确保 resize 与 memcpy 期间容器状态稳定。
      std::lock_guard<std::mutex> lock(pq_mu);
      if (this->data.size() < pq_offset + n_chunks) {
        // 输出运行状态/统计信息，便于观察实验流程和后台线程进度。
        LOG(INFO) << "Resizing PQ data storage from " << this->data.size() << " to "
                  << (1.5 * this->data.size() + n_chunks);
        while (this->data.size() < pq_offset + n_chunks) {
          // 扩容内存中的 PQ code 数组，确保新 target_id 对应的 n_chunks 字节有可写空间。
          this->data.resize(1.5 * this->data.size());
        }
      }
      // 把刚生成的 PQ code 写入 DRAM PQ 数组，搜索阶段立即就可以通过新 ID 使用它。
      memcpy(this->data.data() + pq_offset, pq_coords.data(), n_chunks);
    }
    // 结束当前阶段计时并把耗时累计到统计项。
    ANN_END_TIMING(update_PQ_vec_time, update_t);

    // l_index is candidate size.
    // 提前为候选节点的 ID→完整坐标映射预留容量，降低搜索阶段 rehash 开销。
    coord_map.reserve(2 * this->l_index);

    // 开始记录下面这段逻辑的耗时，便于论文中的阶段级性能分解。
    ANN_START_TIMING(search_graph_time, read_t);
    // this->do_beam_search(point, 0, l_index, beam_width, exp_node_info, &coord_map, nullptr, deletion_set, false,
    //                      &page_ref);
    // 创建本次搜索的统计对象，记录 I/O、比较次数等指标。
    QueryStats stats;
    // LOG(INFO) << "Starting pipe search for insert.";

    // NOTE: 10 is hardcoded mem_L for insert search, refer to CCANN configuration.

    void (SSDIndex<T, TagT>::*search_func)(
        const T *, uint32_t, uint32_t, const uint32_t, std::vector<Neighbor> &, tsl::robin_map<uint32_t, T *> *,
        QueryStats *, tsl::robin_set<uint32_t> * /* tags */, bool, std::vector<uint64_t> *, uint32_t) = nullptr;

    if (this->search_mode == BEAM_SEARCH) {
      // 把统一的函数指针绑定到当前配置选择的搜索实现，后面用同一套参数调用。
      search_func = &SSDIndex<T, TagT>::do_beam_search;
    } else if (this->search_mode == PIPE_SEARCH) {
      // 把统一的函数指针绑定到当前配置选择的搜索实现，后面用同一套参数调用。
      search_func = &SSDIndex<T, TagT>::do_pipe_search;
    } else if (this->search_mode == PARA_SEARCH) {
#ifdef ANN_LARGE
      // 把统一的函数指针绑定到当前配置选择的搜索实现，后面用同一套参数调用。
      search_func = &SSDIndex<T, TagT>::do_para_search_sync;
#else
      // 把统一的函数指针绑定到当前配置选择的搜索实现，后面用同一套参数调用。
      search_func = &SSDIndex<T, TagT>::do_para_search;
#endif
    } else {
      // 记录不可恢复的内部状态错误，下面通常会终止当前流程。
      LOG(ERROR) << "Invalid search mode: " << this->search_mode;
      crash();
    }

    if (this->mem_index_ != nullptr) {
      // this->do_pipe_search(point, 10, l_index, beam_width, exp_node_info, &coord_map, &stats, deletion_set, false);
      // 执行选定的图搜索；输出候选 Neighbor、候选坐标以及搜索期间引用的 page。
      (this->*search_func)(point, 10, l_index, beam_width, exp_node_info, &coord_map, &stats, deletion_set, false,
                           &page_ref, l_index);
    } else {
      // this->do_pipe_search(point, 0, l_index, beam_width, exp_node_info, &coord_map, &stats, deletion_set, false);
      // 执行选定的图搜索；输出候选 Neighbor、候选坐标以及搜索期间引用的 page。
      (this->*search_func)(point, 0, l_index, beam_width, exp_node_info, &coord_map, &stats, deletion_set, false,
                           &page_ref, l_index);
    }
    // 结束当前阶段计时并把耗时累计到统计项。
    ANN_END_TIMING(search_graph_time, read_t);

    // 初始化计时器 prune_search_neighbors_t，后续用于把该阶段开销计入性能 breakdown。
    ANN_INIT_TIMING(prune_search_neighbors_t);
    // 开始记录下面这段逻辑的耗时，便于论文中的阶段级性能分解。
    ANN_START_TIMING(prune_search_neighbors_time, prune_search_neighbors_t);
    // 对搜索阶段找到的候选做最终剪枝，产出 insert phase 要建立反向连接的 new_nhood。
    prune_neighbors(coord_map, exp_node_info, new_nhood);
    // 结束当前阶段计时并把耗时累计到统计项。
    ANN_END_TIMING(prune_search_neighbors_time, prune_search_neighbors_t);

    // 把 PQ code 的所有权移交给调用方，避免复制；insert phase 最终会把它交给后台 commit。
    out_pq_coords = std::move(pq_coords);
    // 返回这个新向量的稳定逻辑 ID；物理 loc 后续可以变化，但 ID 不变。
    return target_id;
  }


  // ============================================================================
  // insert_phase_pm
  // PM/DAX 上的 Soft Insert 主路径。核心顺序是：为目标节点和受影响邻居分配新的物理 loc → 读取旧邻居版本 → 在新 loc 构造目标节点和邻居的新版本 →
  // 先持久化目标向量/Tag，再发布 target 的 ID→loc → 持久化邻居新版本，再切换邻居的 ID→loc。这样 location table
  // 充当“可见性开关”，避免读者看到半写入节点。PQ 与部分 metadata 更新被推迟到后台 commit 线程。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<typename T, typename TagT>
  uint32_t SSDIndex<T, TagT>::insert_phase_pm(const T *point, const TagT &tag, uint32_t target_id,
                                              std::vector<Neighbor> &exp_node_info,
                                              tsl::robin_map<uint32_t, T *> &coord_map,
                                              std::vector<uint32_t> &new_nhood, std::vector<uint64_t> &page_ref,
                                              std::vector<uint8_t> &in_pq_coords) {
    // 取得当前线程的 I/O 上下文；SSD 路径通常对应 io_uring/AIO，PM 路径仍复用统一接口。
    void *ctx = reader->get_ctx();
    // 从 QueryBuffer 池取得线程私有 scratch buffer，避免插入期间反复分配大块临时内存。
    QueryBuffer<T> *read_data = this->pop_query_buf(nullptr);
    this->is_index_inserttable = true;

    // 初始化计时器 update_t，后续用于把该阶段开销计入性能 breakdown。
    ANN_INIT_TIMING(update_t);
    // 初始化计时器 update_neighbor_t，后续用于把该阶段开销计入性能 breakdown。
    ANN_INIT_TIMING(update_neighbor_t);
    // 初始化计时器 read_nodes_t，后续用于把该阶段开销计入性能 breakdown。
    ANN_INIT_TIMING(read_nodes_t);
    // 初始化计时器 prune_neighbor_t，后续用于把该阶段开销计入性能 breakdown。
    ANN_INIT_TIMING(prune_neighbor_t);
    // 初始化计时器 journal_t，后续用于把该阶段开销计入性能 breakdown。
    ANN_INIT_TIMING(journal_t);

    // 记录哪些新目标 page 在修改前必须先读旧内容；allocator 会结合已有 page 利用率填充它。
    std::set<uint64_t> pages_need_to_read;

    // 登记一个正在执行的 insert phase；ACC 会读取这个计数判断 CPU 是否过载。
    this->insert_thread_count_++;

// 编译期开关：启用 baseline 的 in-place/identity-location 更新；默认 Soft Insert 则走下面的 out-of-place location
// allocator。
#ifdef IN_PLACE_RECORD_UPDATE
    // baseline in-place 模式下手工收集将被更新节点的现有 loc。
    std::vector<uint64_t> locs;
    for (auto &nbr : new_nhood) {
      // 取得邻居当前版本的物理 loc，in-place 模式直接复用它。
      locs.emplace_back(id2loc(nbr));
      pages_need_to_read.insert(node_sector_no(nbr));
    }
    // in-place/identity 模式让新 target 的初始 loc 等于 target_id。
    locs.push_back(target_id);
    pages_need_to_read.insert(loc_sector_no(target_id));
    // 更新 DRAM 中的并发 ID→loc 映射，让后续 reader 能按逻辑 ID 找到当前版本。
    id2loc_.insert_or_assign(target_id, target_id);

    // update loc2id, target_id <-> target_id.
    cur_loc++;  // for target ID, atomic update.
    set_loc2id(target_id, target_id);
#else
    // 为每个受影响邻居的新版本以及最后的 target 节点一次性分配 out-of-place loc；locs[i] 对应 new_nhood[i]，末尾 loc 对应 target。
    auto locs = this->alloc_loc(new_nhood.size() + 1, page_ref, pages_need_to_read);
#endif

    // 准备寻找本批新 loc 中的最大值，以确定 graph 文件至少要映射到哪里。
    uint64_t max_loc = 0;
    uint64_t extend_fsize = 0;
    void *faddr = nullptr;
    // 读取当前 graph 文件长度，用来判断新分配 loc 是否要求扩展 DAX 映射。
    uint64_t fsize = reader->file_size();
    for (auto loc : locs) {
      if (loc > max_loc)
        max_loc = loc;
    }
    // 把 node slot 的 loc 换算成所在 4 KiB sector/page 编号。
    extend_fsize = loc_sector_no(max_loc) * SECTOR_LEN + SECTOR_LEN;
    extend_fsize = extend_fsize > fsize ? extend_fsize : fsize;

    // 取得/扩展 DAX mmap 区域，使后续可以通过普通指针直接访问 PM 文件。
    faddr = reader->get_dax(extend_fsize, false);

    // LOG(INFO) << "mmaped addr: " << faddr << " len: " << extend_fsize << " for loc up to " << max_loc;

    // 收集所有将被写入的目标 page，并自动去重；一个 page 可能容纳多个新 node slot。
    std::set<uint64_t> pages_to_rmw_set;
    for (auto &loc : locs) {
      pages_to_rmw_set.insert(loc_sector_no(loc));
    }
    std::vector<IORequest> pages_to_rmw;
    // ordered because of std::set
    for (auto &page_no : pages_to_rmw_set) {
      // 把 page 转成 IORequest 形式，主要用于统一锁页与后续批处理。
      pages_to_rmw.push_back(IORequest(page_no * SECTOR_LEN, size_per_io, nullptr, 0, 0));
    }
    // lock the target and the neighbor ids (ensure that sector_no does not change).
    // 按 page 加锁，防止另一个 updater 在本次读-改-写期间同时修改相同物理页。
    auto pages_locked = v2::lockReqs(this->page_lock_table, pages_to_rmw);
    // 锁住 target 和受影响邻居的逻辑 ID，保证 graph edge 与 version switch 的并发一致性。
    lock_vec(vec_lock_table, target_id, new_nhood);

    // re-read the candidate pages (mostly in the cache).
    // 建立 page_id→可修改 buffer 地址的映射；旧邻居页指向 DRAM read buffer，新分配页可直接指向 DAX PM。
    std::unordered_map<uint32_t, char *> page_buf_map;

    // 使用 QueryBuffer 的 update_buf 承载旧邻居 page 的读入副本。
    auto &update_buf = read_data->update_buf;
    // 分别收集需要读取、写回和按 4 KiB 聚合的 I/O 请求，后面统一批处理。
    std::vector<IORequest> reads, writes, writes_4k;
    // 为可选 CCANN-J baseline 准备 journal entries；正常 Soft Insert 不会使用它们。
    std::vector<v2::journal_entry<T>> journal_entries;
    // 暂存邻居新版本需要 flush 的 PM 地址区间，等全部邻居构造完后统一 flush。
    std::vector<FlushRequest> flush_requests;
    // 用断言检查这里依赖的内部不变量；失败说明索引布局或并发状态已与预期不一致。
    assert(new_nhood.size() < MAX_N_EDGES);

    // read old pages for out-of-place update
    // 逐个处理 target 将要连接的旧邻居：为每个邻居生成包含 target 反向边的新版本。
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      // 为一个旧邻居所在 page 构造整页 read request，把旧 graph node 读到线程私有 update_buf。
      IORequest req(node_sector_no(new_nhood[i]) * SECTOR_LEN, size_per_io, update_buf + i * size_per_io, 0, 0);
      // 把该旧页加入批量读集合，稍后一次 reader->read 提交。
      reads.push_back(req);
      // 根据逻辑 ID 的当前 id2loc 映射取得该节点所在 sector。
      page_buf_map[node_sector_no(new_nhood[i])] = update_buf + i * size_per_io;

      // (static_cast<char *>(faddr) + node_sector_no(new_nhood[i]) * SECTOR_LEN);
      // NOTE: random PM I/O is super slow, use block read instead.
    }
    // read new pages for RMW (might be in-place update).
    for (uint32_t i = new_nhood.size(); i < new_nhood.size() + pages_to_rmw.size(); ++i) {
      auto off = pages_to_rmw[i - new_nhood.size()].offset;
      // writes_4k.push_back(req);
      // LOG(INFO) << off / SECTOR_LEN;
      uint64_t page = off / SECTOR_LEN;
      // if (pages_need_to_read.find(page) != pages_need_to_read.end()) {
      //   // need to read this page first.
      //   reads.push_back(req);
      // }

      // check if page_buf_map exsits
      // 检查目标写 page 是否已经因为读取旧邻居而有 DRAM 副本。
      auto res = page_buf_map.find(page);
      if (res == page_buf_map.end()) {
        // 如果该 page 不需要旧内容，直接让 page_buf_map 指向 DAX PM 地址，后续可原地构造新 slot。
        page_buf_map[page] = (static_cast<char *>(faddr) + off);
      } else {
        // already read
        // 已有 DRAM 副本时复用它作为 RMW buffer；之后需要把整页写回 PM。
        char *addr = res->second;
        IORequest req(off, size_per_io, addr, 0, 0);
        // 把这个被 DRAM 修改的 4 KiB page 记入写回集合。
        writes_4k.push_back(req);
      }

      // update_buf + i * size_per_io;
      // page_buf_map_pm[off / SECTOR_LEN] = (static_cast<char *>(faddr) + off);
      // static_cast<char *>(reader->mmap(req, true));
    }

    // generate continuous writes from 4k writes.
    // dummy one.
    if (!writes_4k.empty()) {
      // 追加哨兵请求，简化下面把相邻 4 KiB page 合并成连续大写请求的边界处理。
      writes_4k.push_back(IORequest(std::numeric_limits<uint64_t>::max(), 0, nullptr, 0, 0));
      uint64_t start_idx = 0;
      uint64_t cur_off = writes_4k[0].offset;
      uint32_t i;

      for (i = 1; i < writes_4k.size() - 1; ++i) {
        if (writes_4k[i].offset != cur_off + size_per_io) {
          writes.push_back(
              IORequest(writes_4k[start_idx].offset, size_per_io * (i - start_idx), writes_4k[start_idx].buf, 0, 0));
          start_idx = i;
        }
        cur_off = writes_4k[i].offset;
      }

// the last one
// TODO: for PM, only needs one single barrier to be written (figure it out).
#ifdef CC_ANN
      // merge all writes except the last one.
      writes.push_back(
          IORequest(writes_4k[start_idx].offset, size_per_io * (i - start_idx), writes_4k[start_idx].buf, 0, 0));
#else
      if (writes_4k[i].offset != cur_off + size_per_io) {
        writes.push_back(
            IORequest(writes_4k[start_idx].offset, size_per_io * (i - start_idx), writes_4k[start_idx].buf, 0, 0));
      }
#endif
      writes_4k.pop_back();
    }

    // 开始记录下面这段逻辑的耗时，便于论文中的阶段级性能分解。
    ANN_START_TIMING(read_nodes_time, read_nodes_t);
    // #ifdef DIRECT_READ_CC
    // 批量读入所有旧邻居 page；接下来所有 prune 和新版本构造都基于这些稳定副本。
    reader->read(reads, ctx);
    // #else
    //     reader->read_alloc(reads, ctx, &page_ref);
    // #endif
    // 结束当前阶段计时并把耗时累计到统计项。
    ANN_END_TIMING(read_nodes_time, read_nodes_t);

    // update the target node.
    // 开始记录下面这段逻辑的耗时，便于论文中的阶段级性能分解。
    ANN_START_TIMING(update_graph_time, update_t);

    // 把 node slot 的 loc 换算成所在 4 KiB sector/page 编号。
    auto target_sector = loc_sector_no(locs[new_nhood.size()]);
    // 在一个 sector/page 基址内，根据 loc 计算具体 graph-node slot 的起始指针。
    auto node_buf = offset_to_loc(page_buf_map[target_sector], locs[new_nhood.size()]);
    // 把 raw node bytes 包装成 DiskNode 运行时视图；coords/nbrs 只是指向底层 buffer 的指针，不发生节点整体拷贝。
    DiskNode<T> target_node(target_id, offset_to_node_coords(node_buf), offset_to_node_nhood(node_buf));
    // 设置 target 的实际邻居数，等于搜索+prune 后的 new_nhood 大小。
    target_node.nnbrs = new_nhood.size();
    *(target_node.nbrs - 1) = target_node.nnbrs;

    // LOG(INFO) << "Inserting node " << target_id << " at loc " << locs[new_nhood.size()] << " in Sector "
    //           << target_sector << " (" << target_sector * SECTOR_LEN << ") with " << target_node.nnbrs << "
    //           neighbors.";

    // only store ids? that's good.
    // but where to find the real coordinates?
    // where is the PQ compressed vector data saved?
    // 把新向量完整坐标写进 target graph-node slot。
    memcpy(target_node.coords, point, data_dim * sizeof(T));
    // 把逻辑邻居 ID 列表写入 target graph-node slot；这里存的是 ID，不是 loc。
    memcpy(target_node.nbrs, new_nhood.data(), new_nhood.size() * sizeof(uint32_t));

    // assert(reader->check_addr_in_pm(node_buf) == true);
    // 判断当前 target slot 是直接位于 DAX PM，还是只存在于 DRAM RMW buffer；后者需要额外 shadow copy 到真正 PM 地址。
    if (!reader->check_addr_in_pm(node_buf)) {
      // the buffer is in memory
      // shadow copy to PMem
      // 计算 target sector 在 DAX 映射中的真实 PM 基址。
      char *pm_sec = (static_cast<char *>(faddr) + target_sector * SECTOR_LEN);
      // 在一个 sector/page 基址内，根据 loc 计算具体 graph-node slot 的起始指针。
      char *pm_node = offset_to_loc(pm_sec, locs[new_nhood.size()]);
      // 把 raw node bytes 包装成 DiskNode 运行时视图；coords/nbrs 只是指向底层 buffer 的指针，不发生节点整体拷贝。
      DiskNode<T> target_node_pm(target_id, offset_to_node_coords(pm_node), offset_to_node_nhood(pm_node));
      target_node_pm.nnbrs = new_nhood.size();
      *(target_node_pm.nbrs - 1) = target_node.nnbrs;  // write to buf
      // 把数据复制到 PM 映射地址；这里使用的 flags 把 flush/drain 时机交给后续统一的 ordered-persistence 步骤。
      pmem_memcpy(target_node_pm.coords, point, data_dim * sizeof(T), PMEM_TRANSFER_CACHE);
      // 把数据复制到 PM 映射地址；这里使用的 flags 把 flush/drain 时机交给后续统一的 ordered-persistence 步骤。
      pmem_memcpy(target_node_pm.nbrs, new_nhood.data(), new_nhood.size() * sizeof(uint32_t), PMEM_TRANSFER_CACHE);
      node_buf = pm_node;
    }

    // 计算本次实际有效 node payload 长度，用于只 flush 坐标和已使用的邻居部分，而不是整个 max_node_len slot。
    auto node_len = data_dim * sizeof(T) + target_node.nnbrs * sizeof(uint32_t);
    // 对刚修改的 PM cache lines 发出 flush，先把数据推向 persistence domain；真正顺序由后续 sfence/barrier 确认。
    reader->flush_dax(node_buf, node_len);

    // Step 1. Update Tags in PM
    if (this->enable_tags) {
      // 确保 tags 文件的 DAX 映射至少覆盖 target_id 对应元素，并按 sector 对齐扩展。
      auto tag_size = ROUND_UP((target_id + 1) * sizeof(TagT), SECTOR_LEN);
      // file system allows all zero
      auto tag_dax = tags_writer->get_dax(tag_size, false);
      // Tag table 按 ID 索引，因此 tag 的 byte offset 就是 target_id*sizeof(TagT)。
      auto target_tag_offset = target_id * sizeof(TagT);
      // 使用 PMDK 的 persist copy：复制后直接保证这段数据达到持久化语义。
      pmem_memcpy_persist((char *) tag_dax + target_tag_offset, &tag, sizeof(TagT));
      tags_writer->put_dax();
    }

    // Target Vector|Tags -> ID2LOC
    // 执行 PM persistence barrier（实现里是 sfence），保证 barrier 之前的 flush/store 在之后的可见性切换前完成。
    reader->barrier_dax();

    // ANN_END_TIMING(update_graph_time, update_t);
    // 同步更新 DRAM tag map，让查询/删除逻辑立即能从 ID 解析用户 tag。
    tags.insert_or_assign(target_id, tag);

    // update the neighbors
    // 开始记录下面这段逻辑的耗时，便于论文中的阶段级性能分解。
    ANN_START_TIMING(update_neighbor_time, update_neighbor_t);
    // 逐个处理 target 将要连接的旧邻居：为每个邻居生成包含 target 反向边的新版本。
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      // 根据逻辑 ID 的当前 id2loc 映射取得该节点所在 sector。
      auto r_sector = node_sector_no(new_nhood[i]);
      if (page_buf_map.find(r_sector) == page_buf_map.end()) {
        // 记录不可恢复的内部状态错误，下面通常会终止当前流程。
        LOG(ERROR) << new_nhood[i] << " "
                   << "Sector " << r_sector << " not found in page_buf_map";
        exit(-1);
      }
      // LOG(INFO) << r_sector << " " << node_sector_no(new_nhood[i]) << " "
      //           << (page_buf_map.find(r_sector) == page_buf_map.end());
      // 用逻辑 ID→loc→page 内偏移定位旧 graph node 的 raw bytes。
      auto r_node_buf = offset_to_node(page_buf_map[r_sector], new_nhood[i]);
      // 把 raw node bytes 包装成 DiskNode 运行时视图；coords/nbrs 只是指向底层 buffer 的指针，不发生节点整体拷贝。
      DiskNode<T> r_nbr_node(new_nhood[i], offset_to_node_coords(r_node_buf), offset_to_node_nhood(r_node_buf));
      // 准备可变邻居列表，容量预留为旧 degree+1，因为首先要尝试追加 target_id。
      std::vector<uint32_t> nhood(r_nbr_node.nnbrs + 1);

      // 用断言检查这里依赖的内部不变量；失败说明索引布局或并发状态已与预期不一致。
      assert(reader->check_addr_in_pm(r_node_buf) == false);
      // 用断言检查这里依赖的内部不变量；失败说明索引布局或并发状态已与预期不一致。
      assert(reader->check_addr_in_pm(r_nbr_node.nbrs) == false);
      // 用断言检查这里依赖的内部不变量；失败说明索引布局或并发状态已与预期不一致。
      assert(reader->check_addr_in_pm(r_nbr_node.nbrs + r_nbr_node.nnbrs) == false);

      // assign the original neighbors
      // 复制旧邻居节点的 neighbor IDs 到临时 vector，后续 prune 不会直接破坏旧版本。
      nhood.assign(r_nbr_node.nbrs, r_nbr_node.nbrs + r_nbr_node.nnbrs);
      // add the new target neighbor
      // 加入指向新 target 的反向边；如果超出最大出度，下面再 prune。
      nhood.emplace_back(target_id);  // attention: we do not reuse IDs.

      // 用断言检查这里依赖的内部不变量；失败说明索引布局或并发状态已与预期不一致。
      assert(reader->check_addr_in_pm(nhood.data()) == false);

      // 只有追加 target 后超过 graph 最大出度 R 时才触发 pruning，避免无必要计算。
      if (nhood.size() > this->range) {  // prune neighbors
// 选择 delta pruning：利用目标点和邻居的 PQ 距离减少完整 prune 的计算量。
#ifdef DELTA_PRUNING
        auto &thread_pq_buf = read_data->aligned_pq_coord_scratch;
        std::vector<float> tgt_dists(nhood.size(), 0.0f), nbr_dists(nhood.size(), 0.0f);

        // TODO: do we really need to compute all distance?
        // TODO: Key: can we only calculate part of the distances?
        // TODO: batch this computation?

        // 开始记录下面这段逻辑的耗时，便于论文中的阶段级性能分解。
        ANN_START_TIMING(prune_neighbor_time, prune_neighbor_t);
        // 批量用 PQ compressed vectors 估算距离，为 pruning/候选筛选提供低成本距离信息。
        compute_pq_dists(target_id, nhood.data(), tgt_dists.data(), (_u32) nhood.size(), thread_pq_buf);
        // 批量用 PQ compressed vectors 估算距离，为 pruning/候选筛选提供低成本距离信息。
        compute_pq_dists(r_nbr_node.id, nhood.data(), nbr_dists.data(), (_u32) nhood.size(), thread_pq_buf);
        // 结束当前阶段计时并把耗时累计到统计项。
        ANN_END_TIMING(prune_neighbor_time, prune_neighbor_t);

        std::vector<TriangleNeighbor> tri_pool(nhood.size());

        for (uint32_t k = 0; k < nhood.size(); k++) {
          tri_pool[k].id = nhood[k];
          tri_pool[k].tgt_dis = tgt_dists[k];
          tri_pool[k].distance = nbr_dists[k];
        }
        // 按 pruning 所需的距离顺序排列候选，便于后续快速判断遮蔽/淘汰关系。
        std::sort(tri_pool.begin(), tri_pool.end());

        int tgt_idx = -1;
        for (int k = 0; k < (int) nhood.size(); ++k) {
          if (tri_pool[k].id == target_id) {
            tgt_idx = k;
            break;
          }
        }
        if (unlikely(tgt_idx == -1)) {
          // 记录不可恢复的内部状态错误，下面通常会终止当前流程。
          LOG(ERROR) << "Target ID " << target_id << " not found in tri_pool";
          exit(-1);
        }
        // 执行基于 PQ 距离的 graph pruning，把候选邻居压到索引允许的最大出度范围内。
        this->delta_prune_neighbors_pq(tri_pool, nhood, thread_pq_buf, tgt_idx);
// 选择 batch pruning：分批计算 PQ 距离，尝试尽早确定只需淘汰一个候选。
#elif BATCH_PRUNING
        auto &thread_pq_buf = read_data->aligned_pq_coord_scratch;
        std::vector<float> tgt_dists(nhood.size(), 0.0f), nbr_dists(nhood.size(), 0.0f);
        std::vector<float> tgt_dists_batch(PRUNE_BATCH_SIZE, 0.0f), nbr_dists_batch(PRUNE_BATCH_SIZE, 0.0f);

        bool pruned = false;
        float tgt_nbr_dis = 0;
        // 批量用 PQ compressed vectors 估算距离，为 pruning/候选筛选提供低成本距离信息。
        compute_pq_dists(target_id, &r_nbr_node.id, &tgt_nbr_dis, 1, thread_pq_buf);

        for (size_t k = 0; k < nhood.size(); k += PRUNE_BATCH_SIZE) {
          size_t bsize = std::min((size_t) PRUNE_BATCH_SIZE, nhood.size() - k);
          // 开始记录下面这段逻辑的耗时，便于论文中的阶段级性能分解。
          ANN_START_TIMING(prune_neighbor_time, prune_neighbor_t);
          // 批量用 PQ compressed vectors 估算距离，为 pruning/候选筛选提供低成本距离信息。
          compute_pq_dists(target_id, nhood.data() + k, tgt_dists_batch.data(), (_u32) bsize, thread_pq_buf);
          // 批量用 PQ compressed vectors 估算距离，为 pruning/候选筛选提供低成本距离信息。
          compute_pq_dists(r_nbr_node.id, nhood.data() + k, nbr_dists_batch.data(), (_u32) bsize, thread_pq_buf);
          // 结束当前阶段计时并把耗时累计到统计项。
          ANN_END_TIMING(prune_neighbor_time, prune_neighbor_t);

          std::vector<TriangleNeighbor> tri_pool(PRUNE_BATCH_SIZE);

          for (size_t j = 0; j < PRUNE_BATCH_SIZE; j++) {
            tri_pool[j].id = nhood[k + j];
            tri_pool[j].tgt_dis = tgt_dists_batch[j];
            tri_pool[j].distance = nbr_dists_batch[j];
          }
          // 按 pruning 所需的距离顺序排列候选，便于后续快速判断遮蔽/淘汰关系。
          std::sort(tri_pool.begin(), tri_pool.end());

          int to_evict = -1;
          // 执行基于 PQ 距离的 graph pruning，把候选邻居压到索引允许的最大出度范围内。
          pruned = this->fast_delta_prune_neighbors_pq(tri_pool, to_evict, tgt_nbr_dis);
          if (to_evict != -1) {
            if ((uint32_t) to_evict != this->range) {
              // 快速路径确定应淘汰的旧候选后，直接从 nhood 删除对应位置。
              nhood.erase(nhood.begin() + k + to_evict);
            } else {
              // remove target node
              // 若 pruning 判断新 target 本身不应进入该邻居的列表，就移除刚追加在末尾的 target_id。
              nhood.pop_back();
            }
            break;
          }

          // assign to the full buffer
          for (size_t j = 0; j < bsize; ++j) {
            tgt_dists[k + j] = tgt_dists_batch[j];
            nbr_dists[k + j] = nbr_dists_batch[j];
          }
        }

        if (!pruned) {
          // full prune
          std::vector<TriangleNeighbor> tri_pool(nhood.size());

          for (uint32_t k = 0; k < nhood.size(); k++) {
            tri_pool[k].id = nhood[k];
            tri_pool[k].tgt_dis = tgt_dists[k];
            tri_pool[k].distance = nbr_dists[k];
          }
          // 按 pruning 所需的距离顺序排列候选，便于后续快速判断遮蔽/淘汰关系。
          std::sort(tri_pool.begin(), tri_pool.end());

          int tgt_idx = -1;
          for (int k = 0; k < (int) nhood.size(); ++k) {
            if (tri_pool[k].id == target_id) {
              tgt_idx = k;
              break;
            }
          }
          if (unlikely(tgt_idx == -1)) {
            // 记录不可恢复的内部状态错误，下面通常会终止当前流程。
            LOG(ERROR) << "Target ID " << target_id << " not found in tri_pool";
            exit(-1);
          }
          // 执行基于 PQ 距离的 graph pruning，把候选邻居压到索引允许的最大出度范围内。
          this->slow_delta_prune_neighbors_pq(tri_pool, nhood, thread_pq_buf, tgt_idx);
        }

#else
        std::vector<float> dists(nhood.size(), 0.0f);
        std::vector<Neighbor> pool(nhood.size());
        auto &thread_pq_buf = read_data->aligned_pq_coord_scratch;
        // 批量用 PQ compressed vectors 估算距离，为 pruning/候选筛选提供低成本距离信息。
        compute_pq_dists(r_nbr_node.id, nhood.data(), dists.data(), (_u32) nhood.size(), thread_pq_buf);
        for (uint32_t k = 0; k < nhood.size(); k++) {
          pool[k].id = nhood[k];
          pool[k].distance = dists[k];
        }
        nhood.clear();
        // 按该元素类型定义的距离顺序排序，为后续 top-k/prune 决策准备有序候选。
        std::sort(pool.begin(), pool.end());
        // 执行基于 PQ 距离的 graph pruning，把候选邻居压到索引允许的最大出度范围内。
        this->prune_neighbors_pq(pool, nhood, thread_pq_buf);
#endif
      }

      // 把 node slot 的 loc 换算成所在 4 KiB sector/page 编号。
      auto w_sector = loc_sector_no(locs[i]);
      // 在一个 sector/page 基址内，根据 loc 计算具体 graph-node slot 的起始指针。
      auto w_node_buf = offset_to_loc(page_buf_map[w_sector], locs[i]);
      // 把 raw node bytes 包装成 DiskNode 运行时视图；coords/nbrs 只是指向底层 buffer 的指针，不发生节点整体拷贝。
      DiskNode<T> w_nbr_node(new_nhood[i], offset_to_node_coords(w_node_buf), offset_to_node_nhood(w_node_buf));
      // 把 prune 后的 degree 写入邻居新版本的运行时视图。
      w_nbr_node.nnbrs = (_u32) nhood.size();
      *(w_nbr_node.nbrs - 1) = (_u32) nhood.size();  // write to buf
      // 把数据复制到 PM 映射地址；这里使用的 flags 把 flush/drain 时机交给后续统一的 ordered-persistence 步骤。
      pmem_memcpy(w_nbr_node.coords, r_nbr_node.coords, data_dim * sizeof(T), PMEM_TRANSFER_CACHE);
      // 把数据复制到 PM 映射地址；这里使用的 flags 把 flush/drain 时机交给后续统一的 ordered-persistence 步骤。
      pmem_memcpy(w_nbr_node.nbrs, nhood.data(), w_nbr_node.nnbrs * sizeof(uint32_t), PMEM_TRANSFER_CACHE);

      if (!reader->check_addr_in_pm(w_node_buf)) {
        // the buffer is in memory
        // shadow copy to PMem
        // 计算 target sector 在 DAX 映射中的真实 PM 基址。
        char *pm_sec = (static_cast<char *>(faddr) + w_sector * SECTOR_LEN);
        // 在一个 sector/page 基址内，根据 loc 计算具体 graph-node slot 的起始指针。
        char *pm_node = offset_to_loc(pm_sec, locs[i]);
        // 把 raw node bytes 包装成 DiskNode 运行时视图；coords/nbrs 只是指向底层 buffer 的指针，不发生节点整体拷贝。
        DiskNode<T> w_nbr_node_pm(new_nhood[i], offset_to_node_coords(pm_node), offset_to_node_nhood(pm_node));
        w_nbr_node_pm.nnbrs = (_u32) nhood.size();
        *(w_nbr_node_pm.nbrs - 1) = (_u32) nhood.size();  // write to buf
        // 把数据复制到 PM 映射地址；这里使用的 flags 把 flush/drain 时机交给后续统一的 ordered-persistence 步骤。
        pmem_memcpy(w_nbr_node_pm.coords, r_nbr_node.coords, data_dim * sizeof(T), PMEM_TRANSFER_CACHE);
        // 把数据复制到 PM 映射地址；这里使用的 flags 把 flush/drain 时机交给后续统一的 ordered-persistence 步骤。
        pmem_memcpy(w_nbr_node_pm.nbrs, nhood.data(), w_nbr_node_pm.nnbrs * sizeof(uint32_t), PMEM_TRANSFER_CACHE);
        w_node_buf = pm_node;
      }
      // assert(reader->check_addr_in_pm(w_node_buf) == true);

      // 计算本次实际有效 node payload 长度，用于只 flush 坐标和已使用的邻居部分，而不是整个 max_node_len slot。
      auto node_len = data_dim * sizeof(T) + w_nbr_node.nnbrs * sizeof(uint32_t);
      // 暂存邻居新版本需要 flush 的 PM 地址区间，等全部邻居构造完后统一 flush。
      flush_requests.push_back(FlushRequest(w_node_buf, node_len));
    }

    // 结束本次 DAX 映射读侧临界区，允许旧映射在 RCU 安全点后被回收。
    reader->put_dax();

    // 结束当前阶段计时并把耗时累计到统计项。
    ANN_END_TIMING(update_neighbor_time, update_neighbor_t);

    // 初始化计时器 update_meta_t，后续用于把该阶段开销计入性能 breakdown。
    ANN_INIT_TIMING(update_meta_t);
    std::vector<uint64_t> write_page_ref;
    // reader->wbc_write(writes, ctx, &write_page_ref);

    // NOTE: File System provides atomic writes, ensuring that fallocate with zero populates.
    // 开始记录下面这段逻辑的耗时，便于论文中的阶段级性能分解。
    ANN_START_TIMING(update_metadata_time, update_meta_t);
    // Step 2. Update ID to Location Mapping in PM and DRAM
// 只有 out-of-place Soft Insert 才需要显式发布新的 ID→loc 映射。
#ifndef IN_PLACE_RECORD_UPDATE
    // Update id2loc PMem mapping to make Target Vector|Tags Persistent.
    // 确保持久化 location table 的 DAX 映射覆盖新 target ID。
    auto id2loc_size = ROUND_UP((target_id + 1) * sizeof(uint32_t), SECTOR_LEN);
    auto id2loc_dax = id2loc_writer->get_dax(id2loc_size, false);
    // location table 是 uint32_t 数组，因此用 target_id 直接计算对应 loc 槽位的 byte offset。
    auto target_id_offset = target_id * sizeof(uint32_t);
    // 把数据复制到 PM 映射地址；这里使用的 flags 把 flush/drain 时机交给后续统一的 ordered-persistence 步骤。
    pmem_memcpy((char *) id2loc_dax + target_id_offset, &locs[new_nhood.size()], sizeof(uint32_t), PMEM_TRANSFER);

    // update locs
    // no concurrency issue for target_id (as it can be only inserted).
    // 更新 DRAM 中的并发 ID→loc 映射，让后续 reader 能按逻辑 ID 找到当前版本。
    id2loc_.insert_or_assign(target_id, locs[new_nhood.size()]);

    // Neighbors -> ID2LOC
    // batch flush caches
    // 暂存邻居新版本需要 flush 的 PM 地址区间，等全部邻居构造完后统一 flush。
    for (auto &flush_req : flush_requests) {
      // 对刚修改的 PM cache lines 发出 flush，先把数据推向 persistence domain；真正顺序由后续 sfence/barrier 确认。
      reader->flush_dax(flush_req.buf, flush_req.len);
    }
    // 执行 PM persistence barrier（实现里是 sfence），保证 barrier 之前的 flush/store 在之后的可见性切换前完成。
    reader->barrier_dax();

// 启用细粒度并发：依赖并发 id2loc map 的原子更新，缩小传统全局/粗粒度锁的范围。
#ifdef FINE_GRAINED_CONCURRENCY
    // We do not need to lock idx_lock_table here, as id2loc_ is concurrent.
    // id2loc_ is already a concurrent hash map.
    // NOTE:
    // We need to ensure the reader-side consistency.
    // just use find_fn to make reader side being atomic.
    std::vector<uint64_t> orig_locs;
    // 逐个处理 target 将要连接的旧邻居：为每个邻居生成包含 target 反向边的新版本。
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      // 保存邻居切换前的旧 loc，稍后 allocator 才能把旧 slot 回收。
      orig_locs.emplace_back(id2loc(new_nhood[i]));
      // Atomically update DRAM id2loc and send (id, loc) to background
      // insert_commit_thread for asynchronous PM id2loc table update.
      // Sequence number (id) ensures ordering via the commit priority queue.
      // 原子切换 DRAM 中该邻居的 ID→新 loc，同时把需要持久化的 (ID,loc) 推给后台 commit 路径。
      id2loc_insert_or_assign(new_nhood[i], (_u32) locs[i]);
    }

    // NOTE:
    // Delay update allocator
    // i.e., loc2id is not updated immediately after id2loc update.
    // 把 target 也追加到 ID 列表，使 allocator 更新函数能用与 locs 相同的顺序一次处理“邻居们 + target”。
    new_nhood.push_back(target_id);
    // 同步 allocator 的反向 loc→id/page_layout：释放旧 loc，并把新 loc 标记给对应 ID。
    erase_and_set_loc(orig_locs, locs, new_nhood);
#else
    auto locked = lock_idx(idx_lock_table, target_id, new_nhood);
    auto page_locked = lock_page_idx(page_idx_lock_table, target_id, new_nhood);
    std::vector<uint64_t> orig_locs;
    // 逐个处理 target 将要连接的旧邻居：为每个邻居生成包含 target 反向边的新版本。
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      // 保存邻居切换前的旧 loc，稍后 allocator 才能把旧 slot 回收。
      orig_locs.emplace_back(id2loc(new_nhood[i]));
      // 更新 DRAM 中的并发 ID→loc 映射，让后续 reader 能按逻辑 ID 找到当前版本。
      id2loc_.insert_or_assign(new_nhood[i], locs[i]);

      // update PM id2loc
      auto id_offset = new_nhood[i] * sizeof(uint32_t);
      // 把数据复制到 PM 映射地址；这里使用的 flags 把 flush/drain 时机交给后续统一的 ordered-persistence 步骤。
      pmem_memcpy((char *) id2loc_dax + id_offset, &locs[i], sizeof(uint32_t), PMEM_TRANSFER);
    }

    // with lock, for simple concurrency with alloc_loc.
    // Only for convenience, note that locs[new_nhood.size()] -> target.
    // 把 target 也追加到 ID 列表，使 allocator 更新函数能用与 locs 相同的顺序一次处理“邻居们 + target”。
    new_nhood.push_back(target_id);
    // 同步 allocator 的反向 loc→id/page_layout：释放旧 loc，并把新 loc 标记给对应 ID。
    erase_and_set_loc(orig_locs, locs, new_nhood);
    unlock_page_idx(page_idx_lock_table, page_locked);
    unlock_idx(idx_lock_table, locked);
#endif
    id2loc_writer->put_dax();
#endif

    // 结束当前阶段计时并把耗时累计到统计项。
    ANN_END_TIMING(update_metadata_time, update_meta_t);

    T *commit_point = nullptr;

    if (this->mem_index_ != nullptr) {
      std::random_device rd;  // Will be used to obtain a seed for the random number engine
      auto x = rd();
      std::mt19937 generator((unsigned) x);
      std::uniform_real_distribution<float> distribution(0, 1);

      if (distribution(generator) < 0.01) {
        // update DRAM index
        // 若抽样命中新向量进入小型 DRAM index，就复制一份坐标交给后台 commit 使用，避免原 point 生命周期结束后悬空。
        commit_point = new T[data_dim];
        // 复制这一段连续内存数据；源和目标的布局在此处已经按 ID/loc 关系确定。
        memcpy(commit_point, point, data_dim * sizeof(T));
      }
    }

    // Step 3. Update PQ Compressed Vector, this can be done in background
    // Step 4. Update in memory graph if possible
    // 封装后台提交任务：携带 target 的 PQ code、逻辑 ID，以及可选的小内存索引坐标副本。
    auto commit_task = new CommitTask{
        .pq_coords = std::move(in_pq_coords),
        .target_id = target_id,
        .point = commit_point,
    };

    // 把提交任务放入 commit queue；前台 Soft Insert 到这里无需等待 PQ/ISS 收尾即可返回。
    commit_tasks.push(commit_task);

    // 锁住 target 和受影响邻居的逻辑 ID，保证 graph edge 与 version switch 的并发一致性。
    unlock_vec(vec_lock_table, target_id, new_nhood);

    // commit writes (in the background thread.)
    if (!writes.empty()) {
      // std::cout << "Flushing " << writes.size() + 1 << " writes to PMem." << std::endl;
      //   reader->write(writes, ctx);
      // writes.push_back(IORequest(target_sector * SECTOR_LEN, size_per_io, nullptr, 0, 0));
      // for (auto &req : writes) {
      //   char *pm_addr = static_cast<char *>(faddr) + req.offset;
      //   boost::crc_32_type result;
      //   result.process_bytes(pm_addr, req.len);
      //   auto cksum = result.checksum();
      //   // std::cout << "Write to sector " << req.offset / SECTOR_LEN << " len " << req.len << " cksum " << cksum
      //   //           << std::endl;
      // }
    }
    // 结束当前阶段计时并把耗时累计到统计项。
    ANN_END_TIMING(update_graph_time, update_t);

    // 按 page 加锁，防止另一个 updater 在本次读-改-写期间同时修改相同物理页。
    v2::unlockReqs(this->page_lock_table, pages_locked);

// 未启用直接读取路径时，需要显式维护 page cache/reference 生命周期。
#ifndef DIRECT_READ_CC
    if (search_mode == BEAM_SEARCH)
      reader->deref(&page_ref, ctx);
#endif

    // 当前 insert phase 结束，减少活动插入线程计数。
    this->insert_thread_count_--;

    // 本次操作结束，把 QueryBuffer 归还池中供后续请求复用。
    this->push_query_buf(read_data);
    // 返回这个新向量的稳定逻辑 ID；物理 loc 后续可以变化，但 ID 不变。
    return target_id;
  }


  // ============================================================================
  // insert_phase
  // 传统块 I/O / 非 PM（以及部分 baseline）插入路径。整体图更新逻辑和 PM 路径相似，但这里围绕 page read-modify-write、write-
  // back cache、可选 journal 来组织 I/O，不依赖 DAX 上细粒度的 ordered persistence。该函数主要作为
  // SSD/OdinANN/CCANN-J 等路径的实现基础。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<typename T, typename TagT>
  uint32_t SSDIndex<T, TagT>::insert_phase(const T *point, const TagT &tag, uint32_t target_id,
                                           std::vector<Neighbor> &exp_node_info,
                                           tsl::robin_map<uint32_t, T *> &coord_map, std::vector<uint32_t> &new_nhood,
                                           std::vector<uint64_t> &page_ref, std::vector<uint8_t> &in_pq_coords) {
    // 取得当前线程的 I/O 上下文；SSD 路径通常对应 io_uring/AIO，PM 路径仍复用统一接口。
    void *ctx = reader->get_ctx();
    // 从 QueryBuffer 池取得线程私有 scratch buffer，避免插入期间反复分配大块临时内存。
    QueryBuffer<T> *read_data = this->pop_query_buf(nullptr);

    this->is_index_inserttable = true;

    // 初始化计时器 update_t，后续用于把该阶段开销计入性能 breakdown。
    ANN_INIT_TIMING(update_t);
    // 初始化计时器 update_neighbor_t，后续用于把该阶段开销计入性能 breakdown。
    ANN_INIT_TIMING(update_neighbor_t);
    // 初始化计时器 read_nodes_t，后续用于把该阶段开销计入性能 breakdown。
    ANN_INIT_TIMING(read_nodes_t);
    // 初始化计时器 prune_neighbor_t，后续用于把该阶段开销计入性能 breakdown。
    ANN_INIT_TIMING(prune_neighbor_t);
    // 初始化计时器 journal_t，后续用于把该阶段开销计入性能 breakdown。
    ANN_INIT_TIMING(journal_t);
    std::set<uint64_t> pages_need_to_read;

    // 登记一个正在执行的 insert phase；ACC 会读取这个计数判断 CPU 是否过载。
    this->insert_thread_count_++;

// 编译期开关：启用 baseline 的 in-place/identity-location 更新；默认 Soft Insert 则走下面的 out-of-place location
// allocator。
#ifdef IN_PLACE_RECORD_UPDATE
    std::vector<uint64_t> locs;
    for (auto &nbr : new_nhood) {
      locs.emplace_back(id2loc(nbr));
      pages_need_to_read.insert(node_sector_no(nbr));
    }
    locs.push_back(target_id);
    pages_need_to_read.insert(loc_sector_no(target_id));
    // 更新 DRAM 中的并发 ID→loc 映射，让后续 reader 能按逻辑 ID 找到当前版本。
    id2loc_.insert_or_assign(target_id, target_id);

    // update loc2id, target_id <-> target_id.
    cur_loc++;  // for target ID, atomic update.
    set_loc2id(target_id, target_id);
#else
    // 为 target 与受影响邻居的新版本分配 out-of-place slots；普通块 I/O 路径同样沿用 ID/loc 解耦。
    auto locs = this->alloc_loc(new_nhood.size() + 1, page_ref, pages_need_to_read);
#endif

    // 收集所有新 slots 所在 page，后续做 page 级 read-modify-write。
    std::set<uint64_t> pages_to_rmw_set;
    for (auto &loc : locs) {
      pages_to_rmw_set.insert(loc_sector_no(loc));
    }
    std::vector<IORequest> pages_to_rmw;
    // ordered because of std::set
    for (auto &page_no : pages_to_rmw_set) {
      pages_to_rmw.push_back(IORequest(page_no * SECTOR_LEN, size_per_io, nullptr, 0, 0));
    }
    // lock the target and the neighbor ids (ensure that sector_no does not change).
    auto pages_locked = v2::lockReqs(this->page_lock_table, pages_to_rmw);
    lock_vec(vec_lock_table, target_id, new_nhood);

    // re-read the candidate pages (mostly in the cache).
    std::unordered_map<uint32_t, char *> page_buf_map;

    auto &update_buf = read_data->update_buf;
    std::vector<IORequest> reads, writes_4k, writes;
    std::vector<v2::journal_entry<T>> journal_entries;

    // 用断言检查这里依赖的内部不变量；失败说明索引布局或并发状态已与预期不一致。
    assert(new_nhood.size() < MAX_N_EDGES);
    // read old pages for out-of-place update
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      reads.push_back(
          IORequest(node_sector_no(new_nhood[i]) * SECTOR_LEN, size_per_io, update_buf + i * size_per_io, 0, 0));
      // 根据逻辑 ID 的当前 id2loc 映射取得该节点所在 sector。
      page_buf_map[node_sector_no(new_nhood[i])] = update_buf + i * size_per_io;
    }

    // read new pages for RMW (might be in-place update).
    for (uint32_t i = new_nhood.size(); i < new_nhood.size() + pages_to_rmw.size(); ++i) {
      auto off = pages_to_rmw[i - new_nhood.size()].offset;
      writes_4k.push_back(IORequest(off, size_per_io, update_buf + i * size_per_io, 0, 0));
      // LOG(INFO) << off / SECTOR_LEN;
      uint64_t page = off / SECTOR_LEN;
      if (pages_need_to_read.find(page) != pages_need_to_read.end()) {
        // need to read this page first.
        reads.push_back(IORequest(off, size_per_io, update_buf + i * size_per_io, 0, 0));
      }
      page_buf_map[off / SECTOR_LEN] = update_buf + i * size_per_io;
    }

    // generate continuous writes from 4k writes.
    // dummy one.
    writes_4k.push_back(IORequest(std::numeric_limits<uint64_t>::max(), 0, nullptr, 0, 0));
    uint64_t start_idx = 0;
    uint64_t cur_off = writes_4k[0].offset;
    uint32_t i;

    for (i = 1; i < writes_4k.size() - 1; ++i) {
      if (writes_4k[i].offset != cur_off + size_per_io) {
        writes.push_back(
            IORequest(writes_4k[start_idx].offset, size_per_io * (i - start_idx), writes_4k[start_idx].buf, 0, 0));
        start_idx = i;
      }
      cur_off = writes_4k[i].offset;
    }

// the last one
// TODO: for PM, only needs one single barrier to be written (figure it out).
#ifdef CC_ANN
    // merge all writes except the last one.
    writes.push_back(
        IORequest(writes_4k[start_idx].offset, size_per_io * (i - start_idx), writes_4k[start_idx].buf, 0, 0));
#else
    if (writes_4k[i].offset != cur_off + size_per_io) {
      writes.push_back(
          IORequest(writes_4k[start_idx].offset, size_per_io * (i - start_idx), writes_4k[start_idx].buf, 0, 0));
    }
#endif
    writes_4k.pop_back();

    std::vector<uint64_t> read_page_ref;
    // 开始记录下面这段逻辑的耗时，便于论文中的阶段级性能分解。
    ANN_START_TIMING(read_nodes_time, read_nodes_t);
#ifdef DIRECT_READ_CC
    reader->read(reads, ctx);
#else
    // 通过 page cache/read allocator 读取旧页，并记录引用，减少重复磁盘 I/O。
    reader->read_alloc(reads, ctx, &read_page_ref);
#endif
    // 结束当前阶段计时并把耗时累计到统计项。
    ANN_END_TIMING(read_nodes_time, read_nodes_t);

    // update the target node.
    // 开始记录下面这段逻辑的耗时，便于论文中的阶段级性能分解。
    ANN_START_TIMING(update_graph_time, update_t);
    // 把 node slot 的 loc 换算成所在 4 KiB sector/page 编号。
    auto sector = loc_sector_no(locs[new_nhood.size()]);
    // 在一个 sector/page 基址内，根据 loc 计算具体 graph-node slot 的起始指针。
    auto node_buf = offset_to_loc(page_buf_map[sector], locs[new_nhood.size()]);
    // 把 raw node bytes 包装成 DiskNode 运行时视图；coords/nbrs 只是指向底层 buffer 的指针，不发生节点整体拷贝。
    DiskNode<T> target_node(target_id, offset_to_node_coords(node_buf), offset_to_node_nhood(node_buf));
    // 复制这一段连续内存数据；源和目标的布局在此处已经按 ID/loc 关系确定。
    memcpy(target_node.coords, point, data_dim * sizeof(T));
    target_node.nnbrs = new_nhood.size();
    *(target_node.nbrs - 1) = target_node.nnbrs;
    // only store ids? that's good.
    // but where to find the real coordinates?
    // where is the PQ compressed vector data saved?
    // 复制这一段连续内存数据；源和目标的布局在此处已经按 ID/loc 关系确定。
    memcpy(target_node.nbrs, new_nhood.data(), new_nhood.size() * sizeof(uint32_t));
    tags.insert_or_assign(target_id, tag);
    auto node_len = data_dim * sizeof(T) + target_node.nnbrs * sizeof(uint32_t);
// 若构建 CCANN-J baseline，则额外生成/提交 journal 记录；Soft Insert 主设计本身不依赖 journal。
#ifdef J_ANN
    auto jhead = v2::journal_entry_head{locs[new_nhood.size()], target_id, node_len, data_dim, target_node.nnbrs};
    auto jentry = v2::journal_entry<T>{jhead, target_node.coords, target_node.nbrs};
    journal_entries.push_back(jentry);
#endif

    // LOG(INFO) << "Target Node at " << locs[new_nhood.size()] << " in Sector " << sector << " (" << sector *
    // SECTOR_LEN
    //           << ") with " << target_node.nnbrs << " neighbors.";

    // update the neighbors
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      // 根据逻辑 ID 的当前 id2loc 映射取得该节点所在 sector。
      auto r_sector = node_sector_no(new_nhood[i]);
      if (page_buf_map.find(r_sector) == page_buf_map.end()) {
        // 记录不可恢复的内部状态错误，下面通常会终止当前流程。
        LOG(ERROR) << new_nhood[i] << " "
                   << "Sector " << r_sector << " not found in page_buf_map";
        exit(-1);
      }
      // 用逻辑 ID→loc→page 内偏移定位旧 graph node 的 raw bytes。
      auto r_node_buf = offset_to_node(page_buf_map[r_sector], new_nhood[i]);
      // 把 raw node bytes 包装成 DiskNode 运行时视图；coords/nbrs 只是指向底层 buffer 的指针，不发生节点整体拷贝。
      DiskNode<T> r_nbr_node(new_nhood[i], offset_to_node_coords(r_node_buf), offset_to_node_nhood(r_node_buf));
      std::vector<uint32_t> nhood(r_nbr_node.nnbrs + 1);
      // assign the original neighbors
      nhood.assign(r_nbr_node.nbrs, r_nbr_node.nbrs + r_nbr_node.nnbrs);
      // add the new target neighbor
      nhood.emplace_back(target_id);  // attention: we do not reuse IDs.

      // LOG(INFO) << "Original Neighbor Node (" << (new_nhood[i]) << ") at " << id2loc(new_nhood[i]) << " in Sector "
      //           << r_sector << " (" << r_sector * SECTOR_LEN << ")";
      if (nhood.size() > this->range) {  // prune neighbors
// 选择 delta pruning：利用目标点和邻居的 PQ 距离减少完整 prune 的计算量。
#ifdef DELTA_PRUNING
        auto &thread_pq_buf = read_data->aligned_pq_coord_scratch;
        std::vector<float> tgt_dists(nhood.size(), 0.0f), nbr_dists(nhood.size(), 0.0f);
        // 批量用 PQ compressed vectors 估算距离，为 pruning/候选筛选提供低成本距离信息。
        compute_pq_dists(target_id, nhood.data(), tgt_dists.data(), (_u32) nhood.size(), thread_pq_buf);
        // 批量用 PQ compressed vectors 估算距离，为 pruning/候选筛选提供低成本距离信息。
        compute_pq_dists(r_nbr_node.id, nhood.data(), nbr_dists.data(), (_u32) nhood.size(), thread_pq_buf);
        std::vector<TriangleNeighbor> tri_pool(nhood.size());

        for (uint32_t k = 0; k < nhood.size(); k++) {
          tri_pool[k].id = nhood[k];
          tri_pool[k].tgt_dis = tgt_dists[k];
          tri_pool[k].distance = nbr_dists[k];
        }
        // 按该元素类型定义的距离顺序排序，为后续 top-k/prune 决策准备有序候选。
        std::sort(tri_pool.begin(), tri_pool.end());

        int tgt_idx = -1;
        for (int k = 0; k < (int) nhood.size(); ++k) {
          if (tri_pool[k].id == target_id) {
            tgt_idx = k;
            break;
          }
        }
        if (unlikely(tgt_idx == -1)) {
          // 记录不可恢复的内部状态错误，下面通常会终止当前流程。
          LOG(ERROR) << "Target ID " << target_id << " not found in tri_pool";
          exit(-1);
        }
        // 执行基于 PQ 距离的 graph pruning，把候选邻居压到索引允许的最大出度范围内。
        this->delta_prune_neighbors_pq(tri_pool, nhood, thread_pq_buf, tgt_idx);
// 选择 batch pruning：分批计算 PQ 距离，尝试尽早确定只需淘汰一个候选。
#elif BATCH_PRUNING
        auto &thread_pq_buf = read_data->aligned_pq_coord_scratch;
        std::vector<float> tgt_dists(nhood.size(), 0.0f), nbr_dists(nhood.size(), 0.0f);
        std::vector<float> tgt_dists_batch(PRUNE_BATCH_SIZE, 0.0f), nbr_dists_batch(PRUNE_BATCH_SIZE, 0.0f);

        bool pruned = false;
        float tgt_nbr_dis = 0;
        // 批量用 PQ compressed vectors 估算距离，为 pruning/候选筛选提供低成本距离信息。
        compute_pq_dists(target_id, &r_nbr_node.id, &tgt_nbr_dis, 1, thread_pq_buf);

        for (size_t k = 0; k < nhood.size(); k += PRUNE_BATCH_SIZE) {
          size_t bsize = std::min((size_t) PRUNE_BATCH_SIZE, nhood.size() - k);
          // 批量用 PQ compressed vectors 估算距离，为 pruning/候选筛选提供低成本距离信息。
          compute_pq_dists(target_id, nhood.data() + k, tgt_dists_batch.data(), (_u32) bsize, thread_pq_buf);
          // 批量用 PQ compressed vectors 估算距离，为 pruning/候选筛选提供低成本距离信息。
          compute_pq_dists(r_nbr_node.id, nhood.data() + k, nbr_dists_batch.data(), (_u32) bsize, thread_pq_buf);

          std::vector<TriangleNeighbor> tri_pool(PRUNE_BATCH_SIZE);

          for (size_t j = 0; j < PRUNE_BATCH_SIZE; j++) {
            tri_pool[j].id = nhood[k + j];
            tri_pool[j].tgt_dis = tgt_dists_batch[j];
            tri_pool[j].distance = nbr_dists_batch[j];
          }
          // 按该元素类型定义的距离顺序排序，为后续 top-k/prune 决策准备有序候选。
          std::sort(tri_pool.begin(), tri_pool.end());

          int to_evict = -1;
          // 执行基于 PQ 距离的 graph pruning，把候选邻居压到索引允许的最大出度范围内。
          pruned = this->fast_delta_prune_neighbors_pq(tri_pool, to_evict, tgt_nbr_dis);
          if (to_evict != -1) {
            if ((uint32_t) to_evict != this->range) {
              nhood.erase(nhood.begin() + k + to_evict);
            } else {
              // remove target node
              nhood.pop_back();
            }
            break;
          }

          // assign to the full buffer
          for (size_t j = 0; j < bsize; ++j) {
            tgt_dists[k + j] = tgt_dists_batch[j];
            nbr_dists[k + j] = nbr_dists_batch[j];
          }
        }

        if (!pruned) {
          // full prune
          std::vector<TriangleNeighbor> tri_pool(nhood.size());

          for (uint32_t k = 0; k < nhood.size(); k++) {
            tri_pool[k].id = nhood[k];
            tri_pool[k].tgt_dis = tgt_dists[k];
            tri_pool[k].distance = nbr_dists[k];
          }
          // 按该元素类型定义的距离顺序排序，为后续 top-k/prune 决策准备有序候选。
          std::sort(tri_pool.begin(), tri_pool.end());

          int tgt_idx = -1;
          for (int k = 0; k < (int) nhood.size(); ++k) {
            if (tri_pool[k].id == target_id) {
              tgt_idx = k;
              break;
            }
          }
          if (unlikely(tgt_idx == -1)) {
            // 记录不可恢复的内部状态错误，下面通常会终止当前流程。
            LOG(ERROR) << "Target ID " << target_id << " not found in tri_pool";
            exit(-1);
          }
          // 执行基于 PQ 距离的 graph pruning，把候选邻居压到索引允许的最大出度范围内。
          this->slow_delta_prune_neighbors_pq(tri_pool, nhood, thread_pq_buf, tgt_idx);
        }
#else
        std::vector<float> dists(nhood.size(), 0.0f);
        std::vector<Neighbor> pool(nhood.size());
        auto &thread_pq_buf = read_data->aligned_pq_coord_scratch;
        // 批量用 PQ compressed vectors 估算距离，为 pruning/候选筛选提供低成本距离信息。
        compute_pq_dists(r_nbr_node.id, nhood.data(), dists.data(), (_u32) nhood.size(), thread_pq_buf);
        for (uint32_t k = 0; k < nhood.size(); k++) {
          pool[k].id = nhood[k];
          pool[k].distance = dists[k];
        }
        nhood.clear();
        // 按该元素类型定义的距离顺序排序，为后续 top-k/prune 决策准备有序候选。
        std::sort(pool.begin(), pool.end());
        // 执行基于 PQ 距离的 graph pruning，把候选邻居压到索引允许的最大出度范围内。
        this->prune_neighbors_pq(pool, nhood, thread_pq_buf);
#endif
      }

      // 把 node slot 的 loc 换算成所在 4 KiB sector/page 编号。
      auto w_sector = loc_sector_no(locs[i]);
      // 在一个 sector/page 基址内，根据 loc 计算具体 graph-node slot 的起始指针。
      auto w_node_buf = offset_to_loc(page_buf_map[w_sector], locs[i]);
      // 把 raw node bytes 包装成 DiskNode 运行时视图；coords/nbrs 只是指向底层 buffer 的指针，不发生节点整体拷贝。
      DiskNode<T> w_nbr_node(new_nhood[i], offset_to_node_coords(w_node_buf), offset_to_node_nhood(w_node_buf));
      w_nbr_node.nnbrs = (_u32) nhood.size();
      *(w_nbr_node.nbrs - 1) = (_u32) nhood.size();  // write to buf
      // 复制这一段连续内存数据；源和目标的布局在此处已经按 ID/loc 关系确定。
      memcpy(w_nbr_node.coords, r_nbr_node.coords, data_dim * sizeof(T));
      // 复制这一段连续内存数据；源和目标的布局在此处已经按 ID/loc 关系确定。
      memcpy(w_nbr_node.nbrs, nhood.data(), w_nbr_node.nnbrs * sizeof(uint32_t));
// 若构建 CCANN-J baseline，则额外生成/提交 journal 记录；Soft Insert 主设计本身不依赖 journal。
#ifdef J_ANN
      auto node_len = data_dim * sizeof(T) + w_nbr_node.nnbrs * sizeof(uint32_t);
      auto jhead = v2::journal_entry_head{locs[i], new_nhood[i], node_len, data_dim, w_nbr_node.nnbrs};
      auto jentry = v2::journal_entry<T>{jhead, w_nbr_node.coords, w_nbr_node.nbrs};
      journal_entries.push_back(jentry);
#endif
      // LOG(INFO) << "New Neighbor Node (" << new_nhood[i] << ") at " << locs[i] << " in Sector "
      //           << w_sector << " (" << w_sector * SECTOR_LEN << ")";
    }

    std::vector<uint64_t> write_page_ref;

// 未启用直接读取路径时，需要显式维护 page cache/reference 生命周期。
#ifndef DIRECT_READ_CC
    // 把修改后的 pages 交给 write-back cache 路径，并返回需要稍后 deref 的 page 引用。
    reader->wbc_write(writes, ctx, &write_page_ref);
#endif

// 只有 out-of-place Soft Insert 才需要显式发布新的 ID→loc 映射。
#ifndef IN_PLACE_RECORD_UPDATE
    // update locs
    // no concurrency issue for target_id (as it can be only inserted).
    // 更新 DRAM 中的并发 ID→loc 映射，让后续 reader 能按逻辑 ID 找到当前版本。
    id2loc_.insert_or_assign(target_id, locs[new_nhood.size()]);
    auto locked = lock_idx(idx_lock_table, target_id, new_nhood);
    auto page_locked = lock_page_idx(page_idx_lock_table, target_id, new_nhood);
    std::vector<uint64_t> orig_locs;
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      orig_locs.emplace_back(id2loc(new_nhood[i]));
      // 更新 DRAM 中的并发 ID→loc 映射，让后续 reader 能按逻辑 ID 找到当前版本。
      id2loc_.insert_or_assign(new_nhood[i], locs[i]);
    }

    // with lock, for simple concurrency with alloc_loc.
    // Only for convenience, note that locs[new_nhood.size()] -> target.
    new_nhood.push_back(target_id);
    // 同步 allocator 的反向 loc→id/page_layout：释放旧 loc，并把新 loc 标记给对应 ID。
    erase_and_set_loc(orig_locs, locs, new_nhood);
    unlock_page_idx(page_idx_lock_table, page_locked);
    unlock_idx(idx_lock_table, locked);
    // LOG(INFO) << "ID " << target_id << " Target loc " << id2loc(target_id);
#endif

    unlock_vec(vec_lock_table, target_id, new_nhood);

    // commit writes (in the background thread.)
// 启用后台 I/O 线程时，前台只封装 BgTask，不同步等待块写完成。
#ifdef BG_IO_THREAD
    if (!page_ref.empty()) {
      auto bg_task = new BgTask{
          .thread_data = read_data,
          .writes = std::move(writes),
          .pages_to_unlock = std::move(pages_locked),
          .pages_to_deref = std::move(write_page_ref),
      };
      bg_io_tasks.push(bg_task);
      bg_io_tasks.push_notify_all();
    } else {
      v2::unlockReqs(this->page_lock_table, pages_locked);
    }
    reader->deref(&page_ref, ctx);
#else
    // 结束当前阶段计时并把耗时累计到统计项。
    ANN_END_TIMING(update_graph_time, update_t);

    // generate journal writes
    // copy all entries to a continuous buffer

// 若构建 CCANN-J baseline，则额外生成/提交 journal 记录；Soft Insert 主设计本身不依赖 journal。
#ifdef J_ANN
    auto journal = (v2::Journal<TagT> *) this->get_cur_journal_instance();
    // 开始记录下面这段逻辑的耗时，便于论文中的阶段级性能分解。
    ANN_START_TIMING(journal_time, journal_t);
    // CCANN-J baseline 先持久化 journal 与 commit marker，再允许数据页写回。
    journal->append_and_commit_journal(journal_entries);
    // 结束当前阶段计时并把耗时累计到统计项。
    ANN_END_TIMING(journal_time, journal_t);
#endif
    // std::cout << "Flushing " << writes.size() << " writes to PMem." << std::endl;

    // 开始记录下面这段逻辑的耗时，便于论文中的阶段级性能分解。
    ANN_START_TIMING(update_graph_time, update_t);
    // 同步把聚合后的 page writes 发到底层存储；这是非 PM 路径真正落盘的关键点。
    reader->write(writes, ctx);
    // 结束当前阶段计时并把耗时累计到统计项。
    ANN_END_TIMING(update_graph_time, update_t);
    // for (auto &req : writes) {
    //   void *faddr = reader->get_dax(1 * 1024L * 1024L * 1024L, false);
    //   char *pm_addr = static_cast<char *>(faddr) + req.offset;
    //   boost::crc_32_type result;
    //   result.process_bytes(pm_addr, req.len);
    //   auto cksum = result.checksum();
    //   // std::cout << "Write to sector " << req.offset / SECTOR_LEN << " len " << req.len << " cksum " << cksum
    //   //           << std::endl;
    // }
// 若构建 CCANN-J baseline，则额外生成/提交 journal 记录；Soft Insert 主设计本身不依赖 journal。
#ifdef J_ANN
    // the following part seems can be done asynchronously.
    // ensure the updates are persistent,
    // before clearing the journal.
    // 开始记录下面这段逻辑的耗时，便于论文中的阶段级性能分解。
    ANN_START_TIMING(journal_time, journal_t);
    reader->sync();
    // commit journal. How?
    // 确认 graph 更新持久化后清理 journal，结束该事务的 write-ahead logging 生命周期。
    journal->clear_journal();
    // 结束当前阶段计时并把耗时累计到统计项。
    ANN_END_TIMING(journal_time, journal_t);
#endif

    v2::unlockReqs(this->page_lock_table, pages_locked);
    reader->deref(&write_page_ref, ctx);

// 未启用直接读取路径时，需要显式维护 page cache/reference 生命周期。
#ifndef DIRECT_READ_CC
    if (search_mode == BEAM_SEARCH)
      reader->deref(&page_ref, ctx);
#endif

    reader->deref(&read_page_ref, ctx);

    // 当前 insert phase 结束，减少活动插入线程计数。
    this->insert_thread_count_--;
    // 本次操作结束，把 QueryBuffer 归还池中供后续请求复用。
    this->push_query_buf(read_data);
    // 普通同步写路径在完整插入结束后增加当前有效点数。
    num_points++;
#endif
    // 返回这个新向量的稳定逻辑 ID；物理 loc 后续可以变化，但 ID 不变。
    return target_id;
  }


  // ============================================================================
  // async_insert_in_place
  // 异步插入入口：搜索阶段仍在调用线程中完成，因为它决定候选和邻居；随后根据 ACC 观察到的 search/insert/compute 线程总负载，决定把 insert
  // phase 提交到后台线程池，还是退化为当前线程同步执行。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::async_insert_in_place(const T *point, const TagT &tag,
                                               tsl::robin_set<uint32_t> *deletion_set) {
    // 保存 search phase 展开的候选节点及精确距离，后续 pruning/insert phase 会复用。
    std::vector<Neighbor> exp_node_info;
    // 保存候选 ID→完整向量坐标，避免 insert phase pruning 时再次从存储读取这些向量。
    tsl::robin_map<uint32_t, T *> coord_map;
    // 承载 prune 后的新 target 邻居 ID 列表。
    std::vector<uint32_t> new_nhood;
    // 记录搜索阶段持有的 page-cache 引用，insert 完成后统一释放。
    std::vector<uint64_t> page_ref;
    // 接收 search_phase 生成的 target PQ code，稍后转交 insert/commit 路径。
    std::vector<uint8_t> out_pq_coords;
    uint32_t target_id;

    // 初始化计时器 search_t，后续用于把该阶段开销计入性能 breakdown。
    ANN_INIT_TIMING(search_t);

    // 开始记录下面这段逻辑的耗时，便于论文中的阶段级性能分解。
    ANN_START_TIMING(search_phase_time, search_t);
    // 同步完成图搜索与邻居选择，并取得为本次插入分配的逻辑 target_id。
    target_id = search_phase(point, deletion_set, exp_node_info, coord_map, new_nhood, page_ref, out_pq_coords);
    // 结束当前阶段计时并把耗时累计到统计项。
    ANN_END_TIMING(search_phase_time, search_t);

    uint32_t (SSDIndex<T, TagT>::*func)(const T *, const TagT &, uint32_t, std::vector<Neighbor> &,
                                        tsl::robin_map<uint32_t, T *> &, std::vector<uint32_t> &,
                                        std::vector<uint64_t> &, std::vector<uint8_t> &) = nullptr;
    // 运行时根据索引是否使用 PM/DAX 选择 ordered Soft Insert 路径或普通块 I/O 路径。
    if (this->on_pm) {
      func = &SSDIndex<T, TagT>::insert_phase_pm;
    } else {
      func = &SSDIndex<T, TagT>::insert_phase;
    }

// 若构建 CCANN-J baseline，则额外生成/提交 journal 记录；Soft Insert 主设计本身不依赖 journal。
#ifdef J_ANN
    func = &SSDIndex<T, TagT>::insert_phase;
#endif

    // 读取当前活动 search 线程数，作为 ACC 负载估计的一部分。
    auto search_threads = this->search_thread_count_.load();
    // 读取当前活动 insert phase 数。
    auto insert_threads = this->insert_thread_count_.load();
    // 读取 PNE 并行计算 worker 数。
    auto calc_threads = this->calc_thread_count_.load();
    // 默认允许把 insert phase 异步化；下面在 CPU 饱和时可能改为同步。
    bool should_async = true;

    // 当逻辑并发量达到约两倍硬件核数时认为资源趋于饱和，ACC 开始限制额外异步任务。
    if (search_threads + insert_threads + calc_threads >= this->num_cpus * 2) {
      if (calc_threads <= search_threads) {  // calc thread is decreased significantly
        // do not submission, fall back to synchronous insert
        // 计算 worker 已被显著压缩时不再继续堆后台 insert，改由当前线程同步完成以避免过度并发。
        should_async = false;
      }
    }

// 禁用 ACC 时强制保持异步插入，不再根据 CPU 压力降级。
#ifdef NO_ACC_OPT
    should_async = true;
#endif

    if (search_threads + insert_threads + calc_threads > this->peak_cpus) {
      // 记录运行期间观察到的最大活动线程总量，用于统计 ACC 的资源占用。
      this->peak_cpus = search_threads + insert_threads + calc_threads;
    }

    if (should_async) {
// 选择 BS thread-pool 实现来异步执行 insert phase。
#ifdef USE_BS_THREAD_POOL
      // 把 insert phase 闭包提交到线程池；各搜索结果通过 move 捕获转移所有权，避免大容器复制。
      insert_pool->detach_task([this, point, tag, target_id, exp_node_info = std::move(exp_node_info),
                                coord_map = std::move(coord_map), new_nhood = std::move(new_nhood),
                                // 把 PQ code 的所有权移交给调用方，避免复制；insert phase 最终会把它交给后台 commit。
                                page_ref = std::move(page_ref), out_pq_coords = std::move(out_pq_coords),
                                func]() mutable {
        // 初始化计时器 insert_t，后续用于把该阶段开销计入性能 breakdown。
        ANN_INIT_TIMING(insert_t);
        // 开始记录下面这段逻辑的耗时，便于论文中的阶段级性能分解。
        ANN_START_TIMING(insert_phase_time, insert_t);
        // 调用选定的 insert phase 实现，真正构造/发布 graph node 新版本。
        (this->*func)(point, tag, target_id, exp_node_info, coord_map, new_nhood, page_ref, out_pq_coords);
        // 结束当前阶段计时并把耗时累计到统计项。
        ANN_END_TIMING(insert_phase_time, insert_t);
      });
// 选择项目的轻量线程池实现来异步执行 insert phase。
#elif USE_SMALL_THREAD_POOL
      // 把 insert phase 闭包提交到线程池；各搜索结果通过 move 捕获转移所有权，避免大容器复制。
      insert_pool->submit([this, point, tag, target_id, exp_node_info = std::move(exp_node_info),
                           coord_map = std::move(coord_map), new_nhood = std::move(new_nhood),
                           // 把 PQ code 的所有权移交给调用方，避免复制；insert phase 最终会把它交给后台 commit。
                           page_ref = std::move(page_ref), out_pq_coords = std::move(out_pq_coords), func]() mutable {
        // 初始化计时器 insert_t，后续用于把该阶段开销计入性能 breakdown。
        ANN_INIT_TIMING(insert_t);
        // 开始记录下面这段逻辑的耗时，便于论文中的阶段级性能分解。
        ANN_START_TIMING(insert_phase_time, insert_t);
        // 调用选定的 insert phase 实现，真正构造/发布 graph node 新版本。
        (this->*func)(point, tag, target_id, exp_node_info, coord_map, new_nhood, page_ref, out_pq_coords);
        // 结束当前阶段计时并把耗时累计到统计项。
        ANN_END_TIMING(insert_phase_time, insert_t);
      });
#endif
    } else {
      // 初始化计时器 insert_t，后续用于把该阶段开销计入性能 breakdown。
      ANN_INIT_TIMING(insert_t);
      // 开始记录下面这段逻辑的耗时，便于论文中的阶段级性能分解。
      ANN_START_TIMING(insert_phase_time, insert_t);
      // 调用选定的 insert phase 实现，真正构造/发布 graph node 新版本。
      (this->*func)(point, tag, target_id, exp_node_info, coord_map, new_nhood, page_ref, out_pq_coords);
      // 结束当前阶段计时并把耗时累计到统计项。
      ANN_END_TIMING(insert_phase_time, insert_t);
    }

    // 返回这个新向量的稳定逻辑 ID；物理 loc 后续可以变化，但 ID 不变。
    return target_id;
  }


  // ============================================================================
  // synchronize_insertions
  // 等待异步插入线程池中的任务全部完成，用于析构、merge 或显式同步点，确保后续操作不会与尚未结束的 insert phase 并发。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::synchronize_insertions() {
// 选择 BS thread-pool 实现来异步执行 insert phase。
#ifdef USE_BS_THREAD_POOL
    // 阻塞直到线程池当前所有插入任务完成。
    insert_pool->wait();
// 选择项目的轻量线程池实现来异步执行 insert phase。
#elif USE_SMALL_THREAD_POOL
    // 阻塞直到线程池当前所有插入任务完成。
    insert_pool->wait_all();
#endif
  }


  // ============================================================================
  // insert_in_place
  // 同步插入入口：依次执行 search_phase 和 insert_phase。根据索引是否驻留 PM 选择 insert_phase_pm 或普通
  // insert_phase；某些 baseline 宏会强制走普通路径。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::insert_in_place(const T *point, const TagT &tag, tsl::robin_set<uint32_t> *deletion_set) {
    // 保存 search phase 展开的候选节点及精确距离，后续 pruning/insert phase 会复用。
    std::vector<Neighbor> exp_node_info;
    // 保存候选 ID→完整向量坐标，避免 insert phase pruning 时再次从存储读取这些向量。
    tsl::robin_map<uint32_t, T *> coord_map;
    // 承载 prune 后的新 target 邻居 ID 列表。
    std::vector<uint32_t> new_nhood;
    // 记录搜索阶段持有的 page-cache 引用，insert 完成后统一释放。
    std::vector<uint64_t> page_ref;
    // 接收 search_phase 生成的 target PQ code，稍后转交 insert/commit 路径。
    std::vector<uint8_t> out_pq_coords;
    uint32_t target_id;

    // 初始化计时器 search_t，后续用于把该阶段开销计入性能 breakdown。
    ANN_INIT_TIMING(search_t);

    // 开始记录下面这段逻辑的耗时，便于论文中的阶段级性能分解。
    ANN_START_TIMING(search_phase_time, search_t);
    // 同步完成图搜索与邻居选择，并取得为本次插入分配的逻辑 target_id。
    target_id = search_phase(point, deletion_set, exp_node_info, coord_map, new_nhood, page_ref, out_pq_coords);
    // 结束当前阶段计时并把耗时累计到统计项。
    ANN_END_TIMING(search_phase_time, search_t);

    uint32_t (SSDIndex<T, TagT>::*func)(const T *, const TagT &, uint32_t, std::vector<Neighbor> &,
                                        tsl::robin_map<uint32_t, T *> &, std::vector<uint32_t> &,
                                        std::vector<uint64_t> &, std::vector<uint8_t> &) = nullptr;
    // 运行时根据索引是否使用 PM/DAX 选择 ordered Soft Insert 路径或普通块 I/O 路径。
    if (this->on_pm) {
      func = &SSDIndex<T, TagT>::insert_phase_pm;
    } else {
      func = &SSDIndex<T, TagT>::insert_phase;
    }

#ifdef ODIN_ANN
    func = &SSDIndex<T, TagT>::insert_phase;
#endif

// 若构建 CCANN-J baseline，则额外生成/提交 journal 记录；Soft Insert 主设计本身不依赖 journal。
#ifdef J_ANN
    func = &SSDIndex<T, TagT>::insert_phase;
#endif

    // 初始化计时器 insert_t，后续用于把该阶段开销计入性能 breakdown。
    ANN_INIT_TIMING(insert_t);
    // 开始记录下面这段逻辑的耗时，便于论文中的阶段级性能分解。
    ANN_START_TIMING(insert_phase_time, insert_t);
    // 调用选定的 insert phase 实现，真正构造/发布 graph node 新版本。
    target_id = (this->*func)(point, tag, target_id, exp_node_info, coord_map, new_nhood, page_ref, out_pq_coords);
    // 结束当前阶段计时并把耗时累计到统计项。
    ANN_END_TIMING(insert_phase_time, insert_t);
    // 返回这个新向量的稳定逻辑 ID；物理 loc 后续可以变化，但 ID 不变。
    return target_id;
  }


  // ============================================================================
  // bg_io_thread
  // 后台块 I/O 线程：不断从 bg_io_tasks 中取出批量写任务，完成实际写回后释放 page lock、解除 page 引用并归还 QueryBuffer。它把慢的写
  // I/O 从前台插入关键路径中移走。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<class T, class TagT>
  void SSDIndex<T, TagT>::bg_io_thread() {
    // 取得当前线程的 I/O 上下文；SSD 路径通常对应 io_uring/AIO，PM 路径仍复用统一接口。
    auto ctx = reader->get_ctx();
    auto timer = ccann::Timer();
    uint64_t n_tasks = 0;

    while (true) {
      // 尝试从无锁/并发任务队列取一个后台写任务。
      auto task = bg_io_tasks.pop();
      // 队列暂时为空时进入等待循环，避免持续忙等占用 CPU。
      while (task == nullptr) {
        // 睡眠等待生产者 push 后的通知，再重新尝试 pop。
        this->bg_io_tasks.wait_for_push_notify();
        task = bg_io_tasks.pop();
      }

      // 执行这个任务携带的批量存储写请求。
      reader->write(task->writes, ctx);
      // 写回完成后才释放相应 page lock，保证别的 updater 不会抢先复用修改中的页。
      v2::unlockReqs(this->page_lock_table, task->pages_to_unlock);
      // 释放该任务持有的 page-cache 引用。
      reader->deref(&task->pages_to_deref, ctx);
      // 本次操作结束，把 QueryBuffer 归还池中供后续请求复用。
      this->push_query_buf(task->thread_data);
      // 任务资源和引用均处理完成，释放 BgTask 对象。
      delete task;
      ++n_tasks;

      if (timer.elapsed() >= 5000000) {
        // 输出运行状态/统计信息，便于观察实验流程和后台线程进度。
        LOG(INFO) << "Processed " << n_tasks << " tasks, throughput: " << (double) n_tasks * 1e6 / timer.elapsed()
                  << " tasks/sec.";
        timer.reset();
        n_tasks = 0;
      }
    }
  }

// ISS 每累计一批 commit task 再尝试推进连续 checkpoint，减少频繁写 super block 的开销。
#define COMMIT_INTERVAL 1000


  // ============================================================================
  // insert_commit_thread
  // 后台 commit/ISS 线程：持久化延后的 PQ code，批量刷新邻居的 ID→loc 更新，并用最小堆按 target_id 整理乱序完成的插入。只有从 ckpt_id
  // 开始形成连续完成前缀时才推进 checkpoint，并把新的 checkpoint 写入 super block，从而缩小 crash recovery 的检查范围。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<class T, class TagT>
  void SSDIndex<T, TagT>::insert_commit_thread() {
    auto timer = ccann::Timer();
    uint64_t n_tasks = 0;
    // 用最小堆保存已经完成后台提交的 target_id；插入线程可能乱序结束，因此不能直接把最大 ID 当 checkpoint。
    std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>> commit_queue;

    // 定义推进 ISS checkpoint 的局部函数：只消费从当前 ckpt_id 开始连续出现的完成 ID。
    auto process_commit_queue = [this, &commit_queue]() {
      // 读取当前已确认的连续持久化前缀起点。
      auto cur_ckpt_id = this->ckpt_id.load();
      uint32_t smallest_commit_id = 0;
      bool ckpt = false;
      while (!commit_queue.empty()) {
        // 查看所有乱序完成任务中最小的 ID，判断它是否正好接在 checkpoint 后面。
        smallest_commit_id = commit_queue.top();
        // 读取当前已确认的连续持久化前缀起点。
        if (cur_ckpt_id == smallest_commit_id) {
          commit_queue.pop();
          // 读取当前已确认的连续持久化前缀起点。
          cur_ckpt_id = smallest_commit_id + 1;
          ckpt = true;
        } else {
          // not continuous
          break;
        }
      }
// 只有启用 ISS 时才维护/持久化 checkpoint 与相关恢复优化状态。
#ifndef NO_ISS
      if (ckpt) {
        // 先在 DRAM 更新 ISS checkpoint，表示更早的 ID 已形成连续完成前缀。
        this->ckpt_id.store(cur_ckpt_id);
        // ensure all previous writes are persistent
        // 执行 PM persistence barrier（实现里是 sfence），保证 barrier 之前的 flush/store 在之后的可见性切换前完成。
        reader->barrier_dax();
        // update current checkpoint id
        // so we do not check these data during next recovery
        // 取得/扩展 DAX mmap 区域，使后续可以通过普通指针直接访问 PM 文件。
        auto index_addr = reader->get_dax(SECTOR_LEN, false);
        // 当前实现复用 super block 起始 4 字节保存 checkpoint/计数语义。
        auto npts_ofs = 0;
        // 使用 PMDK 的 persist copy：复制后直接保证这段数据达到持久化语义。
        pmem_memcpy_persist((char *) index_addr + npts_ofs, &cur_ckpt_id, sizeof(uint32_t));
        // 结束本次 DAX 映射读侧临界区，允许旧映射在 RCU 安全点后被回收。
        reader->put_dax();
      }
#endif
    };

    while (true) {
      // 从前台插入提交队列取得一个 CommitTask。
      auto task = commit_tasks.pop();
      while (task == nullptr) {
        // 没有 commit task 时睡眠等待生产者通知。
        this->commit_tasks.wait_for_push_notify();
        // 从前台插入提交队列取得一个 CommitTask。
        task = commit_tasks.pop();
      }

      // 析构路径发送 terminate sentinel；收到后退出主循环，但之后仍会尝试处理堆里剩余 checkpoint。
      if (task->terminate) {
        // 输出运行状态/统计信息，便于观察实验流程和后台线程进度。
        LOG(INFO) << "Commit thread received terminate signal.";
        delete task;
        break;
      }

      // 若 search_phase 抽样保留了完整向量，则这里负责处理小型 DRAM index 的延后更新资源。
      if (task->point) {
        ccann::Parameters paras;
        paras.Set<unsigned>("R", 32);
        paras.Set<unsigned>("L", 64);
        paras.Set<unsigned>("C", 750);
        paras.Set<float>("alpha", 1.2);

        // TODO: fix distribution
        // mem_index_->insert_point(task->point, paras, task->target_id);
        // LOG(INFO) << "Inserted point " << task->target_id << " into in-memory index.";
        // 当前内存索引插入代码被注释掉，因此至少释放前台为 commit 复制的向量坐标。
        delete[] task->point;
      }

      auto pq_coords = task->pq_coords.data();
      auto target_id = task->target_id;

// 只有启用 ISS 时才维护/持久化 checkpoint 与相关恢复优化状态。
#ifndef NO_ISS
// 非超大索引模式下，PQ compressed vectors 作为独立 PM 文件直接维护。
#ifndef ANN_LARGE
      // 一个 PQ code 的持久化长度就是 CommitTask 中 pq_coords 的字节数。
      auto pq_bytes_per_vector = task->pq_coords.size() * sizeof(uint8_t);
      // 一个 PQ code 的持久化长度就是 CommitTask 中 pq_coords 的字节数。
      auto pq_size = ROUND_UP((target_id + 1) * pq_bytes_per_vector, SECTOR_LEN);
      auto pq_addr = this->pq_compressed_writer->get_dax(pq_size, false);
      // 按 target_id 计算该向量在连续 PQ byte array 中的起始位置。
      auto pq_offset = target_id * pq_bytes_per_vector;
      // 使用 PMDK 的 persist copy：复制后直接保证这段数据达到持久化语义。
      pmem_memcpy_persist((char *) pq_addr + pq_offset, pq_coords, pq_bytes_per_vector);
      this->pq_compressed_writer->put_dax();
#endif
#endif

      // Drain pending background PM id2loc updates (from id2loc_insert_or_assign).
      // Sequence number (target_id / id) ensures ordering via the commit priority queue.
      {
        auto null_pair = std::make_pair(kInvalidID, kInvalidID);
        // 取出细粒度并发路径延后持久化的邻居 (ID,new_loc) 更新。
        auto entry = id2loc_pm_queue.pop();
        if (entry != null_pair) {
          // 把当前能取到的 id2loc 更新聚成一个 batch，减少反复 mmap/put_dax 成本。
          std::vector<std::pair<uint32_t, uint32_t>> id2loc_batch;
          // 跟踪 batch 中最大 ID，以确定 location table 至少需要映射到多大。
          uint32_t max_id = 0;
          do {
            id2loc_batch.push_back(entry);
            if (entry.first > max_id)
              max_id = entry.first;
            // 取出细粒度并发路径延后持久化的邻居 (ID,new_loc) 更新。
            entry = id2loc_pm_queue.pop();
          } while (entry != null_pair);

          auto id2loc_size = ROUND_UP((max_id + 1) * sizeof(uint32_t), SECTOR_LEN);
          auto id2loc_dax = id2loc_writer->get_dax(id2loc_size, false);
          // 逐个把后台累积的邻居新 loc 写入持久化 location table。
          for (auto &[id, loc] : id2loc_batch) {
            auto id_offset = id * sizeof(uint32_t);
            // 把数据复制到 PM 映射地址；这里使用的 flags 把 flush/drain 时机交给后续统一的 ordered-persistence 步骤。
            pmem_memcpy((char *) id2loc_dax + id_offset, &loc, sizeof(uint32_t), PMEM_TRANSFER);
          }
          id2loc_writer->put_dax();
        }
      }

      // 该 target 的后台 PQ/id2loc 收尾已完成，把 target_id 放入最小堆等待形成连续 checkpoint。
      commit_queue.push(target_id);
      // commit in batch
      // 每处理约 1000 个任务再批量尝试推进 checkpoint，降低 super block 持久化频率。
      if (n_tasks != 0 && n_tasks % COMMIT_INTERVAL == 0) {
        // 定义推进 ISS checkpoint 的局部函数：只消费从当前 ckpt_id 开始连续出现的完成 ID。
        process_commit_queue();
      }

      delete task;
      ++n_tasks;

      if (timer.elapsed() >= 5000000) {
        // 输出运行状态/统计信息，便于观察实验流程和后台线程进度。
        LOG(INFO) << "Processed " << n_tasks << " tasks, throughput: " << (double) n_tasks * 1e6 / timer.elapsed()
                  << " tasks/sec.";
        timer.reset();
        n_tasks = 0;
      }
    }

    // 输出运行状态/统计信息，便于观察实验流程和后台线程进度。
    LOG(INFO) << "Processing remaining commit " << commit_queue.size() << " tasks...";
    // 定义推进 ISS checkpoint 的局部函数：只消费从当前 ckpt_id 开始连续出现的完成 ID。
    process_commit_queue();
    // 终止时若仍有无法形成连续前缀的完成 ID，说明存在缺口，当前实现把它视为异常。
    if (!commit_queue.empty()) {
      // 输出运行状态/统计信息，便于观察实验流程和后台线程进度。
      LOG(INFO) << "Commit queue not empty after termination!";
      crash();
    }
    // 输出运行状态/统计信息，便于观察实验流程和后台线程进度。
    LOG(INFO) << "Commit thread terminated.";
  }

  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template class SSDIndex<float>;
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template class SSDIndex<_s8>;
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template class SSDIndex<_u8>;
}  // namespace ccann
