// 引入 aligned_file_reader.h：抽象对齐 I/O 与 IORequest，供 SSD/PM 统一读写。
#include "aligned_file_reader.h"
// 引入 libcuckoo/cuckoohash_map.hh：并发哈希表，用于 id2loc、tags 等共享映射。
#include "libcuckoo/cuckoohash_map.hh"
// 引入 ssd_index.h：SSDIndex 的核心类声明与索引布局辅助函数。
#include "ssd_index.h"
// 引入 malloc.h：底层内存分配相关接口。
#include <malloc.h>
// 引入 algorithm：排序、min/max 等通用算法。
#include <algorithm>
// 引入 filesystem：文件存在性、复制、大小等文件系统操作。
#include <filesystem>

// 引入 omp.h：OpenMP 并行循环与线程编号。
#include <omp.h>
// 引入 chrono：标准时间与持续时间工具。
#include <chrono>
// 引入 cmath：浮点数学函数。
#include <cmath>
// 引入 cstdint：固定宽度整数类型。
#include <cstdint>
// 引入 limits：数值边界值，例如 max() 作为哨兵。
#include <limits>
// 引入 tuple：tuple/pair 相关辅助。
#include <tuple>
// 引入 timer.h：项目内性能计时宏和 Timer。
#include "timer.h"
// 引入 tsl/robin_map.h：高性能 robin hash map，用于候选坐标映射。
#include "tsl/robin_map.h"
// 引入 utils.h：项目通用辅助函数、常量和二进制 I/O 工具。
#include "utils.h"
// 引入 v2/page_cache.h：用户态 page cache 及 page 引用管理。
#include "v2/page_cache.h"

// 引入 unistd.h：pread/pwrite/sleep/close 等 POSIX 接口。
#include <unistd.h>
// 引入 sys/syscall.h：Linux 系统调用编号接口。
#include <sys/syscall.h>
// 引入 linux_aligned_file_reader.h：Linux 下 O_DIRECT、io_uring 与 DAX mmap 的具体 reader。
#include "linux_aligned_file_reader.h"

// 进入 ccann 命名空间，下面实现的函数都属于 CCANN 核心索引模块。
namespace ccann {
// 规定一次 delete-merge 批处理的 sector 数，用较大的顺序 I/O 摊薄随机访问成本。
#define SECTORS_PER_MERGE 65536

  // ============================================================================
  // merge_deletes
  // 删除合并的主体。第一遍扫描现有 graph slots：给未删除节点重新分配紧凑的新
  // ID，同时为已删除节点保存过滤后的邻居集合；第二遍扫描存活节点：遇到被删邻居时把其邻居“绕接”进来，再按 range 做 prune，随后把重编号后的 graph
  // node、PQ 和 tag 写到新索引。最后重建 DRAM 中的 tags/id2loc/page_layout，并写新的 metadata。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::merge_deletes(const std::string &in_path_prefix, const std::string &out_path_prefix,
                                        const std::vector<TagT> &deleted_nodes,
                                        const tsl::robin_set<TagT> &deleted_nodes_set, uint32_t nthreads,
                                        const uint32_t &n_sampled_nbrs) {
    if (nthreads == 0) {
      nthreads = this->max_nthreads;
    }
    // 初始化计时器 merge_t，后续用于把该阶段开销计入性能 breakdown。
    ANN_INIT_TIMING(merge_t);

    // 开始记录下面这段逻辑的耗时，便于论文中的阶段级性能分解。
    ANN_START_TIMING(merge_time, merge_t);
    // 取得当前线程的 I/O 上下文；SSD 路径通常对应 io_uring/AIO，PM 路径仍复用统一接口。
    void *ctx = reader->get_ctx();

    // FIXME?
    // Must wait for persistence, not for successfully submitted IOs.
    // merge 前先等所有后台写任务排空，避免一边扫描旧图一边还有插入在修改物理页。
    while (!bg_io_tasks.empty()) {
      sleep(5);  // simple way to wait for background IO thread.
    }
    // 构造新 graph 文件路径；删除合并采用重写新索引而不是就地压缩旧文件。
    std::string disk_index_out = out_path_prefix + "_disk.index";
    // Note that the index is immutable currently.
    // Step 1: populate neighborhoods, allocate IDs.
    // 建立 old_id→new_id 重编号表，删除后把存活节点压成连续 ID 空间。
    libcuckoo::cuckoohash_map<uint32_t, uint32_t> id_map;                       // old_id -> new_id
    // 缓存每个已删除节点过滤后的邻居，用于第二遍扫描时把跨越 tombstone 的边重新接起来。
    libcuckoo::cuckoohash_map<uint32_t, std::vector<uint32_t>> deleted_nhoods;  // id -> nhood
    // 并行第一遍扫描时用原子计数器为每个存活节点分配唯一的新 ID。
    std::atomic<uint64_t> new_npoints = 0;
    Timer delete_timer;

    char *rbuf = nullptr, *wbuf = nullptr;
    // 分配大块 sector 对齐读 buffer，merge 通过顺序批量 I/O 扫描旧 graph。
    alloc_aligned((void **) &rbuf, SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);
    // 分配双窗口写 buffer，一边填下一批重建节点，一边可写回上一批。
    alloc_aligned((void **) &wbuf, 2 * SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);  // sliding window buffer.
    // 根据当前物理分配前沿 cur_loc 计算需要扫描多少 graph sectors。
    uint64_t n_sectors = (cur_loc + nnodes_per_sector - 1) / nnodes_per_sector;
    // LOG(INFO) << "Cur loc: " << cur_loc.load() << ", cur ID: " << cur_id << ", n_sectors: " << n_sectors
    //           << ", nnodes_per_sector: " << nnodes_per_sector;

    // 第一遍只按 128 sectors 小批扫描，避免长时间占用资源阻塞并发搜索。
    constexpr int SECTORS_PER_POPULATE = 128;             // small to avoid blocking search threads.
    // 第一遍最多使用 4 个线程，限制 populate 阶段对 CPU 和 PM 带宽的冲击。
    uint32_t populate_nthreads = std::min(nthreads, 4u);  // restrict the flow.

    for (uint64_t in_sector = 0; in_sector < n_sectors; in_sector += SECTORS_PER_POPULATE) {
      uint64_t st_sector = in_sector, ed_sector = std::min(in_sector + SECTORS_PER_POPULATE, n_sectors);
      uint64_t loc_st = st_sector * nnodes_per_sector, loc_ed = std::min(cur_loc.load(), ed_sector * nnodes_per_sector);
      uint64_t n_sectors_to_read = ed_sector - st_sector;
      std::vector<IORequest> read_reqs;
      // 把当前连续 sector 区间合并成一个顺序 read request，减少随机 I/O 次数。
      read_reqs.push_back(IORequest(loc_sector_no(loc_st) * SECTOR_LEN, n_sectors_to_read * size_per_io, rbuf, 0, 0));
      // 一次读入当前扫描窗口，随后各 OpenMP worker 只在内存 buffer 上解析节点。
      reader->read(read_reqs, ctx, false);

// 第一遍最多使用 4 个线程，限制 populate 阶段对 CPU 和 PM 带宽的冲击。
#pragma omp parallel for num_threads(populate_nthreads)
      for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
        // populate nhood.
        // 从 allocator 的反向映射判断这个物理 slot 当前属于哪个逻辑 ID；空洞返回 kInvalidID。
        uint64_t id = loc2id(loc);
        // 跳过未被任何活跃 ID 引用的空 slot。
        if (id == kInvalidID) {
          continue;
        }

        // 把内部逻辑 ID 转回用户 tag；删除集合按 tag 维护。
        uint64_t tag = id2tag(id);
        // 如果该 tag 没被 tombstone 删除，就把它保留在新索引中。
        if (deleted_nodes_set.find(tag) == deleted_nodes_set.end()) {  // 2. not deleted, alloc ID.
          // allocate ID.
          // 原子分配紧凑的新 ID；并行扫描不同 loc 时不会重复。
          uint64_t new_id = new_npoints.fetch_add(1);
          // 记录旧 ID 到新 ID 的对应关系，第二遍重写 neighbor IDs 时必须使用。
          id_map.insert(id, new_id);
          continue;
        }

        // 3. deleted, populate nhoods.
        auto page_rbuf = rbuf + (loc / nnodes_per_sector - st_sector) * SECTOR_LEN;
        // 在一个 sector/page 基址内，根据 loc 计算具体 graph-node slot 的起始指针。
        auto node_rbuf = offset_to_loc(page_rbuf, loc);
        // 把 raw node bytes 包装成 DiskNode 运行时视图；coords/nbrs 只是指向底层 buffer 的指针，不发生节点整体拷贝。
        DiskNode<T> node(id, offset_to_node_coords(node_rbuf), offset_to_node_nhood(node_rbuf));
        std::vector<uint32_t> nhood;
        for (uint32_t i = 0; i < node.nnbrs; ++i) {
          uint32_t nbr_tag = id2tag(node.nbrs[i]);
          // 构造 deleted node 的替代邻居时，只保留仍然存活的邻居。
          if (deleted_nodes_set.find(nbr_tag) == deleted_nodes_set.end()) {
            nhood.push_back(node.nbrs[i]);  // filtered neighborhoods.
          }
        }
        // sample for less space consumption.
        if (nhood.size() > n_sampled_nbrs) {
          // std::shuffle(nhood.begin(), nhood.end(), std::default_random_engine());
          // 为控制内存占用，只保留最多 n_sampled_nbrs 个邻居；当前顺序假设更前面的候选更近。
          nhood.resize(n_sampled_nbrs);  // nearest.
        }
        // 缓存这个已删除节点的可用邻居，供第二遍把指向它的边替换成绕接边。
        deleted_nhoods.insert(id, nhood);
      }
    }
    // LOG(INFO) << "Finished populating neighborhoods, totally elapsed: " << delete_timer.elapsed() / 1e3
    //           << "ms, new npoints: " << new_npoints.load() << " " << "id_map size: " << id_map.size();

    // Step 2: prune neighbors, populate PQ and tags.
    // 以 O_DIRECT 打开新 graph 文件，第二遍将重建节点顺序写入这里。
    int fd = open(disk_index_out.c_str(), O_DIRECT | O_LARGEFILE | O_RDWR | O_CREAT, 0755);
    const uint64_t kVecInWBuf = 2 * SECTORS_PER_MERGE * nnodes_per_sector;
    uint64_t wb_id = 0;
    std::atomic<uint64_t> n_used_id = 0;
    // 定义滑动写窗口的 flush 函数，把已填满的一段 wbuf 顺序写入新 graph 文件。
    auto write_back = [&]() {
      // write one buffer.
      uint64_t buf_id = (wb_id % kVecInWBuf) / (nnodes_per_sector * SECTORS_PER_MERGE);
      auto b = wbuf + buf_id * SECTORS_PER_MERGE * SECTOR_LEN;
      std::vector<IORequest> write_reqs;
      uint64_t id_delta = std::min((uint64_t) SECTORS_PER_MERGE * nnodes_per_sector, n_used_id - wb_id);
      write_reqs.push_back(IORequest(loc_sector_no(wb_id) * SECTOR_LEN,
                                     ROUND_UP(id_delta, nnodes_per_sector) / nnodes_per_sector * size_per_io, b, 0, 0));
      reader->write_fd(fd, write_reqs, ctx);
      wb_id += id_delta;
      // LOG(INFO) << "Write back " << wb_id << "/" << n_used_id << " IDs.";
    };

    // 为新索引预分配紧凑 PQ 数组，新 ID 决定每个 code 的新位置。
    std::vector<uint8_t> pq_coords(new_npoints * n_chunks, 0);
    // 为新索引预分配按新 ID 排列的 tag 数组。
    std::vector<TagT> new_tags(new_npoints);

    for (uint64_t in_sector = 0; in_sector < n_sectors; in_sector += SECTORS_PER_MERGE) {
      uint64_t st_sector = in_sector, ed_sector = std::min(in_sector + SECTORS_PER_MERGE, n_sectors);
      uint64_t loc_st = st_sector * nnodes_per_sector, loc_ed = std::min(cur_loc.load(), ed_sector * nnodes_per_sector);
      uint64_t n_sectors_to_read = ed_sector - st_sector;
      std::vector<IORequest> read_reqs;
      // 把当前连续 sector 区间合并成一个顺序 read request，减少随机 I/O 次数。
      read_reqs.push_back(IORequest(loc_sector_no(loc_st) * SECTOR_LEN, n_sectors_to_read * size_per_io, rbuf, 0, 0));
      // 一次读入当前扫描窗口，随后各 OpenMP worker 只在内存 buffer 上解析节点。
      reader->read(read_reqs, ctx, false);  // read in fd

#pragma omp parallel for num_threads(nthreads)
      for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
        // 从 allocator 的反向映射判断这个物理 slot 当前属于哪个逻辑 ID；空洞返回 kInvalidID。
        uint64_t id = loc2id(loc);
        // 跳过未被任何活跃 ID 引用的空 slot。
        if (id == kInvalidID) {
          continue;
        }

        // 把内部逻辑 ID 转回用户 tag；删除集合按 tag 维护。
        uint64_t tag = id2tag(id);
        // 第二遍只重建存活节点；tombstone 节点本身不再写入新 graph。
        if (deleted_nodes_set.find(tag) != deleted_nodes_set.end()) {  // deleted.
          continue;
        }

        auto page_rbuf = rbuf + (loc / nnodes_per_sector - st_sector) * SECTOR_LEN;
        // 在一个 sector/page 基址内，根据 loc 计算具体 graph-node slot 的起始指针。
        auto loc_rbuf = offset_to_loc(page_rbuf, loc);
        // 把 raw node bytes 包装成 DiskNode 运行时视图；coords/nbrs 只是指向底层 buffer 的指针，不发生节点整体拷贝。
        DiskNode<T> node(id, offset_to_node_coords(loc_rbuf), offset_to_node_nhood(loc_rbuf));
        // prune neighbors.
        // 用集合构造候选邻居，既去重又方便把被删节点替换为它的邻居。
        std::unordered_set<uint32_t> nhood_set;
        for (uint32_t i = 0; i < node.nnbrs; ++i) {
          uint32_t nbr_tag = id2tag(node.nbrs[i]);
          if (deleted_nodes_set.find(nbr_tag) != deleted_nodes_set.end()) {
            // deleted, insert neighbors.
            // 取出被删邻居在第一遍保存的存活邻居集合，准备做一跳绕接。
            const auto &nhoods = deleted_nhoods.find(node.nbrs[i]);
            // 把被删节点的邻居并入当前节点，尽量保持删除前的图连通性。
            nhood_set.insert(nhoods.begin(), nhoods.end());
          } else {
            nhood_set.insert(node.nbrs[i]);
            // LOG(INFO) << id << " insert " << node.nbrs[i];
          }
        }
        // 绕接过程中可能重新引入自己，显式移除 self-loop。
        nhood_set.erase(id);  // remove self.
        std::vector<uint32_t> nhood(nhood_set.begin(), nhood_set.end());

        // 候选邻居过多时按 graph 的最大出度 R 重新 prune。
        if (nhood.size() > this->range) {
          std::vector<float> dists(nhood.size(), 0.0f);
          std::vector<Neighbor> pool(nhood.size());
          auto &thread_pq_buf = thread_pq_bufs[omp_get_thread_num()];
          // 批量用 PQ compressed vectors 估算距离，为 pruning/候选筛选提供低成本距离信息。
          compute_pq_dists(id, nhood.data(), dists.data(), (_u32) nhood.size(), thread_pq_buf);

          for (uint32_t k = 0; k < nhood.size(); k++) {
            pool[k].id = nhood[k];
            pool[k].distance = dists[k];
          }
          // 按该元素类型定义的距离顺序排序，为后续 top-k/prune 决策准备有序候选。
          std::sort(pool.begin(), pool.end());
          if (pool.size() > this->maxc) {
            pool.resize(this->maxc);
          }
          nhood.clear();
          // 执行基于 PQ 距离的 graph pruning，把候选邻居压到索引允许的最大出度范围内。
          this->prune_neighbors_pq(pool, nhood, thread_pq_buf);
        }

        // map to new IDs.
        for (auto &nbr : nhood) {
          // graph 重编号后，所有 neighbor ID 都必须从旧 ID 转成新 ID。
          nbr = id_map.find(nbr);
        }

        // write neighbors.
        // 取得当前存活节点在新索引中的逻辑 ID/写入顺序。
        uint64_t new_id = id_map.find(id);
        uint64_t off = new_id % kVecInWBuf;
        auto page_wbuf = wbuf + (off / nnodes_per_sector) * SECTOR_LEN;
        // 在一个 sector/page 基址内，根据 loc 计算具体 graph-node slot 的起始指针。
        auto loc_wbuf = offset_to_loc(page_wbuf, off);
        // 把 raw node bytes 包装成 DiskNode 运行时视图；coords/nbrs 只是指向底层 buffer 的指针，不发生节点整体拷贝。
        DiskNode<T> w_node(new_id, offset_to_node_coords(loc_wbuf), offset_to_node_nhood(loc_wbuf));
        // 完整坐标不因删除而变化，直接从旧 node 复制到新 node。
        memcpy(w_node.coords, node.coords, data_dim * sizeof(T));
        w_node.nnbrs = nhood.size();
        *(w_node.nbrs - 1) = w_node.nnbrs;
        // 写入已经过滤、绕接、prune 且完成重编号的新 neighbor IDs。
        memcpy(w_node.nbrs, nhood.data(), w_node.nnbrs * sizeof(uint32_t));
        ++n_used_id;
        // copy PQ and tags.
        // 把旧 ID 对应的 PQ code 搬到 new_id 对应位置，保持 metadata 与 graph ID 空间一致。
        memcpy(pq_coords.data() + new_id * n_chunks, this->data.data() + id * n_chunks, n_chunks);
        // 把用户 tag 同步重排到 new_id 位置。
        new_tags[new_id] = id2tag(id);
      }

      // LOG(INFO) << "Processed " << ed_sector << "/" << n_sectors << " sectors, n_used_id: " << n_used_id << ".";
      if (n_used_id - wb_id >= SECTORS_PER_MERGE * nnodes_per_sector) {
        write_back();
      }
    }

    // 主循环结束后把未填满一个大窗口的尾部节点也写回。
    while (wb_id < n_used_id) {
      write_back();
    }
    // LOG(INFO) << "Write nhoods finished, totally elapsed " << delete_timer.elapsed() / 1e3 << "ms.";

    uint32_t medoid = this->medoids[0];
    // 如果原入口 medoid 被删除，沿保存的邻居关系选择一个存活节点作为新入口。
    while (deleted_nodes_set.find(id2tag(medoid)) != deleted_nodes_set.end()) {
      // 输出运行状态/统计信息，便于观察实验流程和后台线程进度。
      LOG(INFO) << "Medoid deleted. Choosing another start node.";
      // 取出被删邻居在第一遍保存的存活邻居集合，准备做一跳绕接。
      const auto &nhoods = deleted_nhoods.find(medoid);
      medoid = nhoods[0];
    }
    close(fd);
    // free buf
    aligned_free((void *) rbuf);
    aligned_free((void *) wbuf);

    // set metadata, PQ and tags.
    // 进入 metadata/运行时状态切换临界区，避免其他线程看到半切换的新索引状态。
    merge_lock.lock();  // unlock in reload().
    // TODO: do we need support delete?
    // metadata.
    this->num_points = new_npoints;
    this->medoids[0] = id_map.find(medoid);
    // PQ.
    this->data = std::move(pq_coords);
    // tags.
    // 丢弃旧 ID 空间对应的 DRAM tag map，下面按 new_id 重建。
    tags.clear();
    // 清空旧 location table；新合并文件采用紧凑 identity layout。
    id2loc_.clear();
    // 清空 allocator 的旧 page occupancy 视图，随后按新连续 layout 重建。
    page_layout.clear();
#pragma omp parallel for num_threads(nthreads)
    for (size_t i = 0; i < new_tags.size(); ++i) {
      tags.insert_or_assign(i, new_tags[i]);
      // TODO(gh): use partition data to init id2loc_ and page_layout.
      // 更新 DRAM 中的并发 ID→loc 映射，让后续 reader 能按逻辑 ID 找到当前版本。
      id2loc_.insert_or_assign(i, i);
      // 同步建立反向 loc→id/page_layout，占用关系同样是 identity mapping。
      set_loc2id(i, i);
    }

    // 把新 num_points、medoid、tags、PQ 等配套 metadata 写出，使新 graph 文件成为完整索引。
    this->write_metadata_and_pq(in_path_prefix, out_path_prefix, new_npoints, id_map.find(medoid), &new_tags);
    // LOG(INFO) << "Write metadata and PQ finished, totally elapsed " << delete_timer.elapsed() / 1e3 << "ms.";
    // LOG(INFO) << "Write metadata finished, totally elapsed " << delete_timer.elapsed() / 1e3 << "ms.";
    merge_lock.unlock();

    // 结束当前阶段计时并把耗时累计到统计项。
    ANN_END_TIMING(merge_time, merge_t);
  }


  // ============================================================================
  // merge
  // 轻量增量 merge：不重写整张图，只把 checkpoint 之后新增部分的 metadata/PQ/tag/id2loc 同步到文件，并把 ckpt_id 推进到当前
  // num_points。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::merge(const std::string &in_path_prefix, const std::string &out_path_prefix) {
    // 初始化计时器 merge_t，后续用于把该阶段开销计入性能 breakdown。
    ANN_INIT_TIMING(merge_t);

    // 开始记录下面这段逻辑的耗时，便于论文中的阶段级性能分解。
    ANN_START_TIMING(merge_time, merge_t);

    merge_lock.lock();  // unlock in reload().
    // 仅同步 checkpoint 之后的新增后缀，不重新扫描/重写整个 graph。
    this->write_metadata_and_pq_incremental(in_path_prefix, out_path_prefix, this->ckpt_id.load(),
                                            this->num_points - this->ckpt_id.load());
    // 增量写出成功后把 merge/checkpoint 前沿推进到当前有效点数。
    this->ckpt_id = this->num_points;
    merge_lock.unlock();

    // 结束当前阶段计时并把耗时累计到统计项。
    ANN_END_TIMING(merge_time, merge_t);
  }


  // ============================================================================
  // write_metadata_and_pq_incremental
  // 增量写出索引元数据和新增后缀。先计算新 graph 文件尺寸并写 super-block metadata，再只遍历 last_id 之后的新 ID，把 tags、PQ
  // code 和 id2loc 逐项写到对应文件偏移。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::write_metadata_and_pq_incremental(const std::string &in_path_prefix,
                                                            const std::string &out_path_prefix, unsigned long last_id,
                                                            unsigned num_new_points) {
    auto new_npoints = last_id + num_new_points;
    // 按 super block 1 sector + graph slots 所需 sectors 计算新 graph 文件的逻辑长度。
    uint64_t file_size = SECTOR_LEN + ROUND_UP(new_npoints, nnodes_per_sector) / nnodes_per_sector * SECTOR_LEN;
    // 用 uint64_t 向量按固定顺序手工序列化 super-block metadata；代码中没有 Metadata struct。
    std::vector<uint64_t> output_metadata;
    auto new_medoid = this->medoids[0];

    // metadata 第一个主要字段记录新索引中的节点数量。
    output_metadata.push_back(new_npoints);
    // 记录原始向量维度，loader 需要它解释每个 graph node 的 coords 长度。
    output_metadata.push_back((uint64_t) this->data_dim);

    // 记录新的图入口 medoid ID。
    output_metadata.push_back(new_medoid);  // mapped medoid
    // 记录固定 graph-node slot 的最大字节长度。
    output_metadata.push_back(this->max_node_len);
    // 记录每个 4 KiB sector 可容纳多少个 graph node slots。
    output_metadata.push_back(nnodes_per_sector);
    // 记录 frozen point 数量，兼容动态 DiskANN 的入口/冻结节点语义。
    output_metadata.push_back(this->num_frozen_points);
    // 记录 frozen point 的位置。
    output_metadata.push_back(this->frozen_location);
    // 把计算出的新 graph 文件大小写入 metadata，供后续一致性/布局处理使用。
    output_metadata.push_back(file_size);
    // LOG(INFO) << "New metadata: " << "num points: " << new_npoints << " data dim: " << this->data_dim
    //           << " medoid: " << new_medoid << " max node len: " << this->max_node_len;
    // LOG(INFO) << "Nnodes per sector: " << nnodes_per_sector << " num frozen points: " << this->num_frozen_points
    //           << " frozen location: " << this->frozen_location << " file size: " << file_size / 1024 / 1024 << "MB";

    std::string disk_index_out = out_path_prefix + "_disk.index";
    // 把 metadata vector 通过项目的 binary format 写到 _disk.index 开头。
    ccann::save_bin<uint64_t>(disk_index_out, output_metadata.data(), output_metadata.size(), 1, 0);
    // 把 graph 文件截断/扩展到与 metadata 描述一致的最终长度。
    std::ignore = truncate(disk_index_out.c_str(), file_size);

    // Step 3. Write tags and PQ.
    std::ofstream writer;
    // 打开 tags 文件，增量路径将在每个 ID 对应的固定偏移更新 tag。
    open_file_to_write(writer, in_path_prefix + "_disk.index.tags");
    for (auto i = last_id + 1; i < new_npoints; ++i) {
      TagT tag = id2tag(i);

      // 直接定位到 ID=i 的 tag 槽位；这里 ID 就是数组下标。
      writer.seekp(i * sizeof(TagT), writer.beg);
      // 写入一个 TagT，不改变其他 ID 的 tags。
      writer.write((char *) (&tag), sizeof(TagT));
    }
    writer.flush();
    writer.close();

    // write PQ pivots.
    // std::string pq_out = out_path_prefix + "_pq_compressed.bin";
    // 打开 PQ compressed 文件，按 ID×n_chunks 的 byte offset 更新 code。
    open_file_to_write(writer, in_path_prefix + "_pq_compressed.bin");
    for (auto i = last_id + 1; i < new_npoints; ++i) {
      // 定位到 ID=i 的 PQ code 起点。
      writer.seekp(i * n_chunks, writer.beg);
      // 从 DRAM PQ 数组复制该 ID 的 n_chunks 字节到持久化文件。
      writer.write((char *) (this->data.data() + i * n_chunks), n_chunks);
    }
    writer.flush();
    writer.close();

    // Step 4: write id2loc
    // 打开持久化 location table 文件，按 ID 写入 uint32_t loc。
    open_file_to_write(writer, in_path_prefix + "_disk.index.id2loc");
    for (auto i = last_id + 1; i < new_npoints; ++i) {
      // 读取当前 DRAM location map 中 ID=i 的最新物理 loc。
      auto loc = id2loc(i);
      // Location Table 的 key 隐含在数组下标 i，文件里只存对应 uint32_t loc value。
      writer.seekp(i * sizeof(uint32_t), writer.beg);
      writer.write((char *) (&loc), sizeof(uint32_t));
    }
    writer.flush();
    writer.close();

    // ccann::save_bin<uint8_t>(pq_out, this->data.data(), new_npoints, n_chunks);
    // if (in_path_prefix != out_path_prefix) {
    //   std::filesystem::copy(in_path_prefix + "_pq_pivots.bin", out_path_prefix + "_pq_pivots.bin",
    //                         std::filesystem::copy_options::overwrite_existing);
    // }
  }


  // ============================================================================
  // write_metadata_and_pq
  // 完整写出 metadata、全部 tags 和全部 PQ compressed vectors；当输出前缀不同，还复制 PQ
  // pivots/codebook。用于删除合并后生成一个自洽的新索引版本。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::write_metadata_and_pq(const std::string &in_path_prefix, const std::string &out_path_prefix,
                                                const uint64_t &new_npoints, const uint64_t &new_medoid,
                                                std::vector<TagT> *new_tags) {
    // 按 super block 1 sector + graph slots 所需 sectors 计算新 graph 文件的逻辑长度。
    uint64_t file_size = SECTOR_LEN + ROUND_UP(new_npoints, nnodes_per_sector) / nnodes_per_sector * SECTOR_LEN;
    // 用 uint64_t 向量按固定顺序手工序列化 super-block metadata；代码中没有 Metadata struct。
    std::vector<uint64_t> output_metadata;
    // metadata 第一个主要字段记录新索引中的节点数量。
    output_metadata.push_back(new_npoints);
    // 记录原始向量维度，loader 需要它解释每个 graph node 的 coords 长度。
    output_metadata.push_back((uint64_t) this->data_dim);

    // 记录新的图入口 medoid ID。
    output_metadata.push_back(new_medoid);  // mapped medoid
    // 记录固定 graph-node slot 的最大字节长度。
    output_metadata.push_back(this->max_node_len);
    // 记录每个 4 KiB sector 可容纳多少个 graph node slots。
    output_metadata.push_back(nnodes_per_sector);
    // 记录 frozen point 数量，兼容动态 DiskANN 的入口/冻结节点语义。
    output_metadata.push_back(this->num_frozen_points);
    // 记录 frozen point 的位置。
    output_metadata.push_back(this->frozen_location);
    // 把计算出的新 graph 文件大小写入 metadata，供后续一致性/布局处理使用。
    output_metadata.push_back(file_size);
    // 输出运行状态/统计信息，便于观察实验流程和后台线程进度。
    LOG(INFO) << "New metadata: " << "num points: " << new_npoints << " data dim: " << this->data_dim
              << " medoid: " << new_medoid << " max node len: " << this->max_node_len;
    // 输出运行状态/统计信息，便于观察实验流程和后台线程进度。
    LOG(INFO) << "Nnodes per sector: " << nnodes_per_sector << " num frozen points: " << this->num_frozen_points
              << " frozen location: " << this->frozen_location << " file size: " << file_size / 1024 / 1024 << "MB";

    std::string disk_index_out = out_path_prefix + "_disk.index";
    // 把 metadata vector 通过项目的 binary format 写到 _disk.index 开头。
    ccann::save_bin<uint64_t>(disk_index_out, output_metadata.data(), output_metadata.size(), 1, 0);
    // 把 graph 文件截断/扩展到与 metadata 描述一致的最终长度。
    std::ignore = truncate(disk_index_out.c_str(), file_size);

    // Step 3. Write tags and PQ.
    std::vector<TagT> tags_vec;
    if (new_tags == nullptr) {
      tags_vec.resize(new_npoints);
      for (uint64_t i = 0; i < new_npoints; ++i) {
        tags_vec[i] = id2tag(i);
      }
      new_tags = &tags_vec;
    }
    // 完整重写 tags 文件，使 tag 顺序与新的连续 ID 空间一致。
    ccann::save_bin<TagT>(out_path_prefix + "_disk.index.tags", new_tags->data(), new_npoints, 1, 0);

    // write PQ pivots.
    std::string pq_out = out_path_prefix + "_pq_compressed.bin";
    // 完整重写所有 PQ compressed codes。
    ccann::save_bin<uint8_t>(pq_out, this->data.data(), new_npoints, n_chunks);

    if (in_path_prefix != out_path_prefix) {
      // PQ codebook/pivots 与 ID 重编号无关，输出前缀不同情况下直接复制即可。
      std::filesystem::copy(in_path_prefix + "_pq_pivots.bin", out_path_prefix + "_pq_pivots.bin",
                            // PQ codebook/pivots 与 ID 重编号无关，输出前缀不同情况下直接复制即可。
                            std::filesystem::copy_options::overwrite_existing);
    }
  }


  // ============================================================================
  // reload
  // merge 后重新打开 graph 文件并重置运行时分配游标。cur_id 指向下一个逻辑 ID，cur_loc 对齐到下一个 page 的可分配边界，同时清空旧
  // empty_pages 状态。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::reload(const char *index_prefix, uint32_t num_threads) {
    std::string iprefix = std::string(index_prefix);
    std::string pq_compressed_vectors = iprefix + "_pq_compressed.bin";
    std::string disk_index_file = iprefix + "_disk.index";
    this->_disk_index_file = disk_index_file;
    this->max_nthreads = num_threads;

    // 关闭 merge 前的旧 graph 文件句柄。
    reader->close();
    // 重新打开新的/已合并 graph 文件，使后续查询读新版本。
    reader->open(disk_index_file, true, false);

    // 输出运行状态/统计信息，便于观察实验流程和后台线程进度。
    LOG(INFO) << "Reloading, num_points " << this->num_points << " n_chunks: " << this->n_chunks;
    // 紧凑 merge 后 ID 和 loc 都连续占用 [0,num_points)，因此两个分配前沿先从 num_points 开始。
    this->cur_id = this->cur_loc = this->num_points;
    if (this->num_points % nnodes_per_sector != 0) {
      // 把物理 loc 分配前沿对齐到下一个完整 page，避免新插入跨越当前尾页布局假设。
      this->cur_loc += nnodes_per_sector - (num_points % nnodes_per_sector);
    }

    // 旧索引的空页信息已失效，merge 后必须清空 allocator 的 free-page 队列。
    while (!this->empty_pages.empty()) {
      this->empty_pages.pop();
    }
    // reload 完成后释放 merge 锁，允许查询/插入重新使用索引。
    merge_lock.unlock();
    // 当前函数工作已经完成，直接返回调用方。
    return;
  }

  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template class SSDIndex<float>;
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template class SSDIndex<_s8>;
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template class SSDIndex<_u8>;
}  // namespace ccann
