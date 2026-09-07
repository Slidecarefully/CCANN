// ============================================================================
// 文件逻辑导读
// 这部分实现 PipeSearch：把多个 graph-node 读取请求保持为 in-flight I/O，主线程不断
// poll 已完成请求、展开节点、再补发新的读取，试图重叠 I/O 与搜索计算。
// 它与 PNE 的关键区别是：PipeSearch 主要并行/流水化 I/O，而 PNE 进一步把邻居 PQ
// 距离计算本身放到独立计算线程中。文件中的锁顺序注释也非常关键，它解释了动态更新
// 场景下为什么异步流水容易与写线程形成死锁，以及 CCANN 的 fine-grained 路径如何规避。
// ============================================================================
#include "aligned_file_reader.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "neighbor.h"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#include <map>
#include <tbb/parallel_for.h>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include "timer.h"
#include "tsl/robin_set.h"
#include "utils.h"
#include "v2/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>

// 进入编译期开关 USE_AIO 对应的实现分支；同一算法可按硬件/实验配置切换不同路径。
#ifndef USE_AIO
#include "liburing.h"
// 结束这一组编译期条件分支。
#endif

// 进入 CCANN 相关命名空间，后续定义均属于索引实现。
namespace ccann {
  struct io_t {
    Neighbor nbr;
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    unsigned page_id;
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    unsigned loc;
    // 构造/保存一次底层 I/O 描述，其中包含对齐读范围、目标 buffer 以及可选 useful byte range。
    IORequest *read_req;
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    bool operator>(const io_t &rhs) const {
      // 把当前阶段得到的结果/状态返回给上层调用者。
      return nbr.distance > rhs.nbr.distance;
    }

    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    bool operator<(const io_t &rhs) const {
      // 把当前阶段得到的结果/状态返回给上层调用者。
      return nbr.distance < rhs.nbr.distance;
    }

    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    bool finished() {
      // 把当前阶段得到的结果/状态返回给上层调用者。
      return read_req->finished;
    }
  };

  // 模板参数 T 表示向量坐标类型，TagT 表示用户可见标签类型。
  template<typename T, typename TagT>
// ---------------------------------------------------------------------------
// do_pipe_search：PipeSearch 内部实现。
// on_flight_ios 保存已经提交但尚未消费的 graph-node read；主循环反复 poll 完成 I/O、
// 展开完成节点、再发新 I/O，从而用流水化方式隐藏设备访问延迟。
// ---------------------------------------------------------------------------
  void SSDIndex<T, TagT>::do_pipe_search(const T *query1, uint32_t mem_L, uint32_t l_search, const uint32_t beam_width,
                                         std::vector<Neighbor> &expanded_nodes_info,
                                         tsl::robin_map<uint32_t, T *> *coord_map, QueryStats *stats,
                                         tsl::robin_set<uint32_t> *exclude_nodes /* tags */, bool dyn_search_l,
                                         // 维护本次搜索持有的 page-cache 引用，避免搜索过程中缓存页被提前回收。
                                         std::vector<uint64_t> *passthrough_page_ref, uint32_t k_search) {
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    uint32_t original_l_search = l_search;
    // 从 QueryBuffer 池取得本查询专用的对齐 scratch 空间，并把 query1 拷入对齐查询缓冲。
    QueryBuffer<T> *query_buf = pop_query_buf(query1);
    // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
    ANN_INIT_TIMING(populate_t);
// 进入编译期开关 USE_AIO 对应的实现分支；同一算法可按硬件/实验配置切换不同路径。
#ifdef USE_AIO
    // 取得当前线程对应的底层 I/O 上下文，后续 read/send/poll 都复用它。
    void *ctx = reader->get_ctx();
// 进入上述编译期开关的备用实现分支。
#else
    // 取得当前线程的 io_uring 上下文，并请求 SQPOLL；Pipe/PNE 用它减少提交 I/O 时的系统调用开销。
    void *ctx = reader->get_ctx(IORING_SETUP_SQPOLL);  // use SQ polling only for pipe search.
// 结束这一组编译期条件分支。
#endif

    // 登记当前活跃搜索线程；ACC 会把搜索、插入和计算 worker 的总并发作为资源压力信号。
    this->search_thread_count_++;

    // 根据当前搜索状态/编译配置决定是否执行这一分支。
    if (beam_width > MAX_N_SECTOR_READS) {
      // 遇到违反索引不变量的状态时记录错误；后续通常直接终止以避免继续使用损坏状态。
      LOG(ERROR) << "Beamwidth can not be higher than MAX_N_SECTOR_READS";
      // 当前状态无法安全恢复，主动终止而不是返回可能错误的搜索结果。
      crash();
    }

    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    // 后续距离计算统一使用对齐后的 query 指针，以满足 SIMD/AVX 距离函数的对齐需求。
    const T *query = query_buf->aligned_query_T;

    // reset query
    // 只清空本次查询的运行状态（索引、visited 等），保留已分配的大块 scratch 内存复用。
    query_buf->reset();

    // pointers to buffers for data
    // 取得 DRAM 中的完整向量 scratch；从 PM/SSD 读出的 coords 会复制到这里再做 exact distance。
    T *data_buf = query_buf->coord_scratch;
    // 提前把 DRAM coordinate scratch 拉近 CPU cache，降低随后 exact-distance 写入/读取的冷启动开销。
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    // 取得页/sector 读取缓冲；graph node 的原始字节先进入这里，再按 loc 定位 node。
    char *sector_scratch = query_buf->sector_scratch;

    // query <-> neighbor list
    // 取得距离结果 scratch；PQ 批量距离或临时精确距离结果都复用这块对齐内存。
    float *dist_scratch = query_buf->aligned_dist_scratch;
    // 取得 PQ code 聚合 scratch，用来把离散 ID 对应的压缩码整理成连续布局后批量查表。
    _u8 *pq_coord_scratch = query_buf->aligned_pq_coord_scratch;

    Timer query_timer;
    // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
    std::vector<Neighbor> retset(mem_L + l_search * 10);
    std::vector<unsigned int> inserts;
    // 维护已经发现/处理过的 logical ID 集合，避免图中多条边导致重复工作。
    auto &visited = *(query_buf->visited);
    // 维护候选池当前有效元素个数；容器实际 capacity 可能更大。
    unsigned cur_list_size = 0;

    // re-naming `expanded_nodes_info` to not change rest of the code
    // 把调用者提供的 expanded_nodes_info 作为“真正展开节点”的结果容器；这里记录 exact distance。
    std::vector<Neighbor> &full_retset = expanded_nodes_info;
    // 预留容量以减少搜索热路径中的动态扩容/内存搬迁。
    full_retset.reserve(l_search * 10);

    // query <-> PQ chunk centers distances
    // 取得 PQ 查表距离缓冲；它保存 query 每个 PQ chunk 到各 centroid 的距离表。
    float *pq_dists = query_buf->aligned_pqtable_dist_scratch;

// 进入编译期开关 OVERLAP_INIT 对应的实现分支；同一算法可按硬件/实验配置切换不同路径。
#ifndef OVERLAP_INIT
    // 预计算 query 到 PQ codebook centroid 的查找表，后续大量候选只需按压缩码查表求近似距离。
    pq_table.populate_chunk_distances(query, pq_dists);  // overlap with the first I/O.
// 结束这一组编译期条件分支。
#endif

    // lambda to batch compute query<-> node distances in PQ space
// ---- 构造 PQ 批量距离 helper：把 ID -> PQ code -> ADC distance 封装起来 ----
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    auto compute_pq_dists = [this, pq_dists](const unsigned *ids, const _u64 n_ids, float *dists_out,
                                             // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
                                             _u8 *pq_coord_scratch) {
      // 按 logical ID 从内存 PQ 数据 this->data 中收集压缩码，形成连续批次，便于后续 SIMD/查表计算。
      ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks, pq_coord_scratch);
      // 使用预计算的 query->centroid 距离表，对这一批 PQ code 做 ADC 查表并输出近似距离。
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists, dists_out);
    };

// ---- 精确距离阶段：完整 coords 已到 DRAM 后，计算 query<->node exact distance ----
    // 这个 helper 负责 expanded node 的完整向量精确距离，并把结果放入 full_retset；early-exit 也基于这些精确距离判断。
    auto compute_exact_dists_and_push = [&](const char *node_buf, const unsigned id) -> float {
      // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
      ANN_INIT_TIMING(compute_t);

      // 开始统计这一逻辑阶段的耗时。
      ANN_START_TIMING(calc_exact_dist_time, compute_t);
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      T *node_fp_coords_copy = data_buf;
      // Graph node 的首字段就是 coords，因此从 node_buf 起点复制完整向量到 DRAM scratch，再进行精确距离计算。
      memcpy(node_fp_coords_copy, node_buf, data_dim * sizeof(T));
      // 在 CPU 上对 query 与 DRAM scratch 中的完整向量做 exact distance；底层可走 AVX-512/AVX2。
      float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);

      // NOTE: add coord_map
      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (coord_map != nullptr) {
        coord_map->insert(std::make_pair(id, node_fp_coords_copy));
      }

      // LOG(INFO) << "Expanding node " << id << " distance " << cur_expanded_dist;
      // 把“确实读过完整 vector 并计算 exact distance”的节点加入最终展开集合。
      full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
      // 结束该阶段计时，并把耗时累计进对应统计项。
      ANN_END_TIMING(calc_exact_dist_time, compute_t);

      // 把当前阶段得到的结果/状态返回给上层调用者。
      return cur_expanded_dist;
    };
    // LOG(INFO) << "Start pipe search with mem_L: " << mem_L << ", l_search: " << l_search
    //           << ", beam_width: " << beam_width;
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    uint64_t n_computes = 0;
// ---- 邻居展开阶段：去重邻居、批量 PQ 距离、筛选后插回 retset ----
    // nk 记录本轮新邻居插入后最靠前的候选位置，用来决定搜索指针是否需要回退。
    auto compute_and_push_nbrs = [&](const char *node_buf, unsigned &nk) {
      // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
      ANN_INIT_TIMING(expand_t);

      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      unsigned *node_nbrs = offset_to_node_nhood(node_buf);
      // 读取 neighbor 数量，并把指针前移到第一个 neighbor ID，后续可直接遍历。
      unsigned nnbrs = *(node_nbrs++);
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      unsigned nbors_cand_size = 0;
      // 逐个扫描当前 graph node 的邻居 logical ID。
      for (unsigned m = 0; m < nnbrs; ++m) {
        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if (visited.find(node_nbrs[m]) == visited.end()) {
          node_nbrs[nbors_cand_size++] = node_nbrs[m];
          // 维护已经发现/处理过的 logical ID 集合，避免图中多条边导致重复工作。
          visited.insert(node_nbrs[m]);
        }
      }

      n_computes += nbors_cand_size;
      // 开始统计这一逻辑阶段的耗时。
      ANN_START_TIMING(expand_neighbors_time, expand_t);
      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (nbors_cand_size) {
        // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
        auto adjust_size = nbors_cand_size;
        // auto cpu1_st = std::chrono::high_resolution_clock::now();
        // 3400ns
        // 只对尚未 visited 的邻居批量算 PQ 距离，减少重复计算。
        compute_pq_dists(node_nbrs, nbors_cand_size, dist_scratch, pq_coord_scratch);
        // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
        auto not_insert = 0;
        // 逐个扫描当前 graph node 的邻居 logical ID。
        for (unsigned m = 0; m < nbors_cand_size; ++m) {
          const int nbor_id = node_nbrs[m];
          const float nbor_dist = dist_scratch[m];
          // 根据当前搜索状态/编译配置决定是否执行这一分支。
          if (stats != nullptr) {
            stats->n_cmps++;
          }
          // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
          ANN_ADD_STAT(ncalc_for_expanding_neighbors, 1);
          // 根据当前搜索状态/编译配置决定是否执行这一分支。
          if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search)) {
            ++not_insert;
            // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
            ANN_ADD_STAT(ncalc_for_useless_neighbors, 1);
            continue;
          }
          // 构造候选记录：logical ID + 当前距离 + 可展开标记。
          Neighbor nn(nbor_id, nbor_dist, true);
          // Return position in sorted list where nn inserted
          // 把新候选按距离插入有序 retset，并返回插入位置；这可能让搜索指针 k 回退到更优候选。
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);  // may be overflow in retset...
          // 根据当前搜索状态/编译配置决定是否执行这一分支。
          if (cur_list_size < l_search) {
            ++cur_list_size;
            // 根据当前搜索状态/编译配置决定是否执行这一分支。
            if (unlikely(cur_list_size >= retset.size())) {
              // 调整容器有效容量，确保后续按索引写入不会越界。
              retset.resize(2 * cur_list_size);
            }
          }
          // nk logs the best position in the retset that was updated due to
          // neighbors of n.
          // 根据当前搜索状态/编译配置决定是否执行这一分支。
          if (r < nk)
            // nk 记录本轮新邻居插入后最靠前的候选位置，用来决定搜索指针是否需要回退。
            nk = r;
        }
        // auto cpu1_ed = std::chrono::high_resolution_clock::now();
        // stats->cpu_us1 += std::chrono::duration_cast<std::chrono::microseconds>(cpu1_ed - cpu1_st).count();
        // LOG(INFO) << "Computed and pushed " << (nbors_cand_size - not_insert) << " / " << nbors_cand_size
      }
      // 结束该阶段计时，并把耗时累计进对应统计项。
      ANN_END_TIMING(expand_neighbors_time, expand_t);
    };

    // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
    auto add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids, float *dists) {
      // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
      for (_u64 i = 0; i < n_ids; ++i) {
        // 维护候选池当前有效元素个数；容器实际 capacity 可能更大。
        retset[cur_list_size++] = Neighbor(node_ids[i], dists[i], true);
        // 把入口候选标记为已发现，防止后续从不同边再次重复加入候选池。
        visited.insert(node_ids[i]);
      }
    };

    // stats.
    // 初始化或更新本次查询的统计字段，便于拆分 I/O 与 CPU 开销。
    stats->io_us = 0;
    // 初始化或更新本次查询的统计字段，便于拆分 I/O 与 CPU 开销。
    stats->io_us1 = 0;
    // 初始化或更新本次查询的统计字段，便于拆分 I/O 与 CPU 开销。
    stats->cpu_us = 0;
    // 初始化或更新本次查询的统计字段，便于拆分 I/O 与 CPU 开销。
    stats->cpu_us1 = 0;
    // 初始化或更新本次查询的统计字段，便于拆分 I/O 与 CPU 开销。
    stats->cpu_us2 = 0;
    // search in in-memory index.

// 进入编译期开关 DYN_PIPE_WIDTH 对应的实现分支；同一算法可按硬件/实验配置切换不同路径。
#ifdef DYN_PIPE_WIDTH
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    int64_t cur_beam_width = 4;  // before converge.
// 进入上述编译期开关的备用实现分支。
#else
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    int64_t cur_beam_width = beam_width;  // before converge.
// 结束这一组编译期条件分支。
#endif
    std::vector<unsigned> mem_tags(mem_L);
    std::vector<float> mem_dists(mem_L);

    // 开始统计这一逻辑阶段的耗时。
    ANN_START_TIMING(populate_pq_dists_time, populate_t);
// 进入编译期开关 OVERLAP_INIT 对应的实现分支；同一算法可按硬件/实验配置切换不同路径。
#ifdef OVERLAP_INIT
// ---- 选择搜索入口：优先使用小型 DRAM index；否则从 graph medoid 开始 ----
    // 根据当前搜索状态/编译配置决定是否执行这一分支。
    if (mem_L) {
      // 用小型内存索引快速产生 mem_L 个入口候选；后续仍会在主图上继续搜索。
      mem_index_->search_with_tags_fast(query, mem_L, mem_tags.data(), mem_dists.data());
      // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
      add_to_retset(mem_tags.data(), std::min((unsigned) mem_L, (unsigned) l_search), mem_dists.data());
    } else {
      // cannot overlap.
      // 生成 PQ 查找表；nt 版本用于和其他初始化/首批 I/O 重叠，减少启动阶段串行开销。
      pq_table.populate_chunk_distances_nt(query, pq_dists);
      compute_pq_dists(&medoids[0], 1, dist_scratch, pq_coord_scratch);
      // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
      add_to_retset(&medoids[0], 1, dist_scratch);
    }
// 进入上述编译期开关的备用实现分支。
#else
// ---- 选择搜索入口：优先使用小型 DRAM index；否则从 graph medoid 开始 ----
    // 根据当前搜索状态/编译配置决定是否执行这一分支。
    if (mem_L) {
      // 用小型内存索引快速产生 mem_L 个入口候选；后续仍会在主图上继续搜索。
      mem_index_->search_with_tags_fast(query, mem_L, mem_tags.data(), mem_dists.data());
      compute_pq_dists(mem_tags.data(), mem_L, dist_scratch, pq_coord_scratch);
      // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
      add_to_retset(mem_tags.data(), std::min((_u64) mem_L, l_search), dist_scratch);
    } else {
      compute_pq_dists(&medoids[0], 1, dist_scratch, pq_coord_scratch);
      // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
      add_to_retset(&medoids[0], 1, dist_scratch);
    }
    // 把当前有效候选按距离排序，使 retset 前部始终代表优先扩展的最近候选。
    std::sort(retset.begin(), retset.begin() + cur_list_size);
// 结束这一组编译期条件分支。
#endif
    // 结束该阶段计时，并把耗时累计进对应统计项。
    ANN_END_TIMING(populate_pq_dists_time, populate_t);

// ---- PipeSearch 流水状态：从这里开始 I/O 提交和节点展开解耦 ----
    // 维护已提交但尚未消费的 graph-node I/O FIFO；PipeSearch 的流水深度由它的 size 控制。
    std::queue<io_t> on_flight_ios;

// ---- graph-node 读取 helper：ID -> loc -> page/byte-range -> QueryBuffer ----
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    auto send_read_req = [&](Neighbor &item) -> bool {
      item.flag = false;

      // NOTE: here needs special attention:
      //
      // The correct version is:
      //
      // Insert Thread:         Search Thread:
      //
      // wrLock(id-a)           rdLock(id-a)
      // wrLock(id-b)           rdLock(id-b)
      //
      // unLock(id-a)
      // unLock(id-b)
      //
      // However, if use Pipe style:
      //
      // Search Thread (without interrupt):
      //
      // rdLock(id-a)
      // unLock(id-a)
      //
      // rdLock(id-b)
      // unLock(id-b)
      //
      // However, the case is that
      //
      // Search Thread (with interrupt):
      //
      // rdLock(id-b)
      // still proc
      // rdLock(id-a) <-- deadlock
      //
      // unLock(id-a)
      // unLock(id-b)
      //
      // So, we should keep the lock id order.

      // LOG(INFO) << "Locking node " << item.id << " in thread " << syscall(SYS_gettid);
      // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
      ANN_INIT_TIMING(send_best_t);
      // lock the corresponding page.
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      uint32_t pid;
      // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
      uint64_t &cur_buf_idx = query_buf->sector_idx;
      // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
      auto buf = sector_scratch + cur_buf_idx * size_per_io;
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      auto &req = query_buf->reqs[cur_buf_idx];
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      unsigned loc = 0;
// 进入编译期开关 FINE_GRAINED_CONCURRENCY 对应的实现分支；同一算法可按硬件/实验配置切换不同路径。
#ifdef FINE_GRAINED_CONCURRENCY
      // This is CCANN-improved
      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (this->on_pm) {
        loc = id2loc_func(item.id, [&](uint32_t &loc) {
          // 由 loc 计算 graph node 所在的 4KB sector/page 编号。
          pid = loc_sector_no(loc);
          // u_loc_offset(loc) 给出该 node 在整个 graph 文件中的精确 byte 起点，用于 PM byte-range read。
          req = IORequest(static_cast<_u64>(pid) * SECTOR_LEN, size_per_io, buf, u_loc_offset(loc), max_node_len);

          // 开始统计这一逻辑阶段的耗时。
          ANN_START_TIMING(send_best_node_time, send_best_t);
          // 提交一次 graph-node 读取；在 PM+DAX 路径中会变成 byte-range prefetch+copy，在非 PM 路径中走异步 I/O。
          reader->send_io(req, ctx, false);
          // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
          ANN_ADD_STAT(send_best_node_number, 1);
          // 结束该阶段计时，并把耗时累计进对应统计项。
          ANN_END_TIMING(send_best_node_time, send_best_t);
        });
      } else {
        // Because of long I/O latency of SSD, we cannot support high concurrency.
        // 遇到违反索引不变量的状态时记录错误；后续通常直接终止以避免继续使用损坏状态。
        LOG(ERROR) << "Fine grained concurrency is only supported for PM index.";
        // 当前状态无法安全恢复，主动终止而不是返回可能错误的搜索结果。
        crash();
      }
// 进入上述编译期开关的备用实现分支。
#else
      // 读取候选 logical ID 当前对应的 physical loc，后续所有 page/offset 计算都以这个版本为准。
      loc = id2loc(item.id);
      // 由 loc 计算 graph node 所在的 4KB sector/page 编号。
      pid = loc_sector_no(loc);
      this->lock_idx(idx_lock_table, item.id, std::vector<uint32_t>(), true);
      // u_loc_offset(loc) 给出该 node 在整个 graph 文件中的精确 byte 起点，用于 PM byte-range read。
      req = IORequest(static_cast<_u64>(pid) * SECTOR_LEN, size_per_io, buf, u_loc_offset(loc), max_node_len);

      // 开始统计这一逻辑阶段的耗时。
      ANN_START_TIMING(send_best_node_time, send_best_t);
      // 发起不重新分配 buffer 的读取；PipeSearch 复用 QueryBuffer 中预留的 sector scratch。
      reader->send_read_no_alloc(req, ctx);
      // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
      ANN_ADD_STAT(send_best_node_number, 1);
      // 结束该阶段计时，并把耗时累计进对应统计项。
      ANN_END_TIMING(send_best_node_time, send_best_t);

// 进入编译期开关 PIPE_PM_READS 对应的实现分支；同一算法可按硬件/实验配置切换不同路径。
#ifndef PIPE_PM_READS
      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (this->on_pm) {
        // for PM index, unlock immediately.
        this->unlock_idx(idx_lock_table, item.id);
      }
// 结束这一组编译期条件分支。
#endif

// 结束这一组编译期条件分支。
#endif
      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (passthrough_page_ref != nullptr)
        // 维护本次搜索持有的 page-cache 引用，避免搜索过程中缓存页被提前回收。
        passthrough_page_ref->push_back((static_cast<_u64>(pid) * SECTOR_LEN) / SECTOR_LEN);
      on_flight_ios.push(io_t{item, pid, loc, &req});
      cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;

      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (stats != nullptr) {
        stats->n_ios++;
      }
      // 把当前阶段得到的结果/状态返回给上层调用者。
      return true;
    };

    // id_buf_map: id -> buf
    std::unordered_map<unsigned, char *> id_buf_map;
// ---- 完成队列处理：收割已完成 I/O/计算并把结果重新推进搜索状态 ----
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    auto poll_all = [&]() -> std::pair<int, int> {
      // poll once.
      // 非阻塞收割底层已经完成的所有 I/O，并把对应 IORequest.finished 置为 true。
      reader->poll_all(ctx);
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      unsigned n_in = 0, n_out = 0;

      // there exists on flight ios.
      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (!on_flight_ios.empty()) {
        // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
        ANN_ADD_STAT(poll_number, 1);
      }

      // 进入以“仍有未完成搜索工作”为条件的循环，直到候选或 in-flight 工作全部收敛。
      while (!on_flight_ios.empty() && on_flight_ios.front().finished()) {
        io_t &io = on_flight_ios.front();
        id_buf_map.insert(std::make_pair(io.nbr.id, offset_to_loc((char *) io.read_req->buf, io.loc)));
        // 维护候选池当前有效元素个数；容器实际 capacity 可能更大。
        io.nbr.distance <= retset[cur_list_size - 1].distance ? ++n_in : ++n_out;

// 进入编译期开关 PIPE_PM_READS 对应的实现分支；同一算法可按硬件/实验配置切换不同路径。
#ifndef PIPE_PM_READS
        // unlock the corresponding page.
        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if (!this->on_pm) {
          // enable async only for SSD index.
          this->unlock_idx(idx_lock_table, io.nbr.id);
        }
// 进入上述编译期开关的备用实现分支。
#else
        this->unlock_idx(idx_lock_table, io.nbr.id);
// 结束这一组编译期条件分支。
#endif

        // 该读请求对应节点已经完成展开，移出 in-flight I/O 队列。
        on_flight_ios.pop();
        // LOG(INFO) << "Unlocked node " << io.nbr.id << " in thread " << syscall(SYS_gettid);
      }

      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (n_in + n_out > 0) {
        // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
        ANN_ADD_STAT(poll_hit_number, 1);
      }
      // 把当前阶段得到的结果/状态返回给上层调用者。
      return std::make_pair(n_in, n_out);
    };

// ---- 流水补充策略：从当前最近的未展开候选中继续补发读请求 ----
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    auto send_best_read_req = [&](uint32_t n) -> bool {
      // auto io_st = std::chrono::high_resolution_clock::now();
      // retset is ordered (from small to large).
      // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
      unsigned n_sent = 0, marker = 0;
      // 进入以“仍有未完成搜索工作”为条件的循环，直到候选或 in-flight 工作全部收敛。
      while (marker < cur_list_size && n_sent < n) {
        // 进入以“仍有未完成搜索工作”为条件的循环，直到候选或 in-flight 工作全部收敛。
        while (marker < cur_list_size /* pool size */ &&
               // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
               (retset[marker].flag == false /* on flight */ ||
                // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
                id_buf_map.find(retset[marker].id) != id_buf_map.end() /* already read */)) {
          // find the first not sent and not already read neighbor.
          // 标记该候选已经被选入本轮展开，避免下一轮再次发起相同 graph-node 读取。
          retset[marker].flag = false;  // even out the id_buf_map cost to O(1)
          ++marker;
        }
        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if (marker >= cur_list_size) {
          break;  // nothing to send.
        }

        // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
        auto id = retset[marker].id;

        // ensure that on_flight ios are also ordered
        // on_flight_ios reflects the order of sending IOs.
        // NOTE: Ensure Lock Order.
        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if (!on_flight_ios.empty() &&
            on_flight_ios.back().nbr.id > id) {  // if the last sent IO has id > current id, then wait.
          break;
        }

        // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
        n_sent += send_read_req(retset[marker]);
      }
      // auto io_ed = std::chrono::high_resolution_clock::now();
      // stats->io_us += std::chrono::duration_cast<std::chrono::microseconds>(io_ed - io_st).count();
      // 把当前阶段得到的结果/状态返回给上层调用者。
      return n_sent != 0;  // nothing to send.
    };

// ---- 候选选择核心：结合 exact distance/当前 retset 判断下一步是否继续扩展 ----
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    auto calc_best_node = [&](int &expand_retries) -> int {  // if converged.
      // auto cpu_st = std::chrono::high_resolution_clock::now();
      // 维护候选池当前有效元素个数；容器实际 capacity 可能更大。
      unsigned marker = 0, nk = cur_list_size, first_unvisited_eager = cur_list_size;
      /* calculate one from "already read" */
      // 在线性候选区间中寻找满足条件的候选；L 通常只有几十到几百，因此这一扫描成本可控。
      for (marker = 0; marker < cur_list_size; ++marker) {
        // NOTE: after compute_exact_dists_and_push, the retset is changed, so marker should be changed.
        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if (!retset[marker].visited) {
          // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
          auto it = id_buf_map.find(retset[marker].id);
          // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
          bool res = it != id_buf_map.end();
          // 根据当前搜索状态/编译配置决定是否执行这一分支。
          if (res) {
            // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
            auto [id, buf] = *it;
            // 标记该候选已经被选入本轮展开，避免下一轮再次发起相同 graph-node 读取。
            retset[marker].flag = false;  // even out the id_buf_map cost to O(1)
            // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
            retset[marker].visited = true;
            // Do not expand too far nodes.
            // 这个 helper 负责 expanded node 的完整向量精确距离，并把结果放入 full_retset；early-exit 也基于这些精确距离判断。
            compute_exact_dists_and_push(buf, id);
            compute_and_push_nbrs(buf, nk);
            break;
          }
        }
      }

      /* guess the first unvisited vector (eager) */
      // 在线性候选区间中寻找满足条件的候选；L 通常只有几十到几百，因此这一扫描成本可控。
      for (unsigned i = 0; i < cur_list_size; ++i) {
        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if (!retset[i].visited && retset[i].flag /* not on-fly */
            // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
            && id_buf_map.find(retset[i].id) == id_buf_map.end() /* not already read */) {
          // 维护已经发现/处理过的 logical ID 集合，避免图中多条边导致重复工作。
          first_unvisited_eager = i;
          break;
        }
      }
      // 把当前阶段得到的结果/状态返回给上层调用者。
      return first_unvisited_eager;
      // auto cpu_ed = std::chrono::high_resolution_clock::now();
      // stats->cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(cpu_ed - cpu_st).count();
    };

// ---- 查找 retset 中最靠前的尚未真正展开候选，用于判断搜索是否收敛 ----
    // 维护已经发现/处理过的 logical ID 集合，避免图中多条边导致重复工作。
    auto get_first_unvisited = [&]() -> int {
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      int ret = -1;
      // 在线性候选区间中寻找满足条件的候选；L 通常只有几十到几百，因此这一扫描成本可控。
      for (unsigned i = 0; i < cur_list_size; ++i) {
        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if (!retset[i].visited) {
          ret = i;
          break;
        }
      }
      // 把当前阶段得到的结果/状态返回给上层调用者。
      return ret;
    };

    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    auto print_state = [&]() {
      LOG(INFO) << "cur_list_size: " << cur_list_size;
      // 在线性候选区间中寻找满足条件的候选；L 通常只有几十到几百，因此这一扫描成本可控。
      for (unsigned i = 0; i < cur_list_size; ++i) {
        // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
        LOG(INFO) << "retset[" << i << "]: " << retset[i].id << ", " << retset[i].distance << ", " << retset[i].flag
                  // 维护已经发现/处理过的 logical ID 集合，避免图中多条边导致重复工作。
                  << ", " << retset[i].visited << ", " << (id_buf_map.find(retset[i].id) != id_buf_map.end());
      }
      LOG(INFO) << "On flight IOs: " << on_flight_ios.size();
      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (on_flight_ios.size() != 0) {
        // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
        auto &io = on_flight_ios.front();
        LOG(INFO) << "on_flight_io: " << io.nbr.id << ", " << io.nbr.distance << ", " << io.nbr.flag << ", "
                  << io.page_id << ", " << io.loc << ", " << io.finished();
      }
      usleep(500);
    };

    std::ignore = print_state;

    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    auto cpu2_st = std::chrono::high_resolution_clock::now();
    send_best_read_req(cur_beam_width - on_flight_ios.size());
    // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
    unsigned marker = 0, max_marker = 0;
// 进入编译期开关 OVERLAP_INIT 对应的实现分支；同一算法可按硬件/实验配置切换不同路径。
#ifdef OVERLAP_INIT
    // 根据当前搜索状态/编译配置决定是否执行这一分支。
    if (likely(mem_L != 0)) {
      // 生成 PQ 查找表；nt 版本用于和其他初始化/首批 I/O 重叠，减少启动阶段串行开销。
      pq_table.populate_chunk_distances_nt(query, pq_dists);  // overlap with the first I/O.
      compute_pq_dists(mem_tags.data(), mem_L, dist_scratch, pq_coord_scratch);
      // 在线性候选区间中寻找满足条件的候选；L 通常只有几十到几百，因此这一扫描成本可控。
      for (unsigned i = 0; i < cur_list_size; ++i) {
        // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
        retset[i].distance = dist_scratch[i];
      }
      // 把当前有效候选按距离排序，使 retset 前部始终代表优先扩展的最近候选。
      std::sort(retset.begin(), retset.begin() + cur_list_size);
    }
// 结束这一组编译期条件分支。
#endif

// 进入编译期开关 STATIC_POLICY 对应的实现分支；同一算法可按硬件/实验配置切换不同路径。
#ifndef STATIC_POLICY
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    int cur_n_in = 0, cur_tot = 0;
// 结束这一组编译期条件分支。
#endif
    // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
    ANN_INIT_TIMING(poll_t);
    // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
    ANN_INIT_TIMING(calc_best_t);

    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    int expand_retries = 0;
    // LOG(INFO) << "Start One Query Pipe Search: beam_width=" << beam_width;
// ---- PipeSearch 主事件循环：poll I/O -> 展开完成节点 -> 补发下一批 I/O ----
    // 进入以“仍有未完成搜索工作”为条件的循环，直到候选或 in-flight 工作全部收敛。
    while (get_first_unvisited() != -1) {
      // poll to heap (best-effort) -> calc best from heap (skip if heap is empty) -> send IO (if can send) -> ...
      // auto io1_st = std::chrono::high_resolution_clock::now();
      // 开始统计这一逻辑阶段的耗时。
      ANN_START_TIMING(poll_all_time, poll_t);
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      auto [n_in, n_out] = poll_all();
      // 结束该阶段计时，并把耗时累计进对应统计项。
      ANN_END_TIMING(poll_all_time, poll_t);
      std::ignore = n_in;
      std::ignore = n_out;

      // n_in: number of nodes that can improve the retset.
      // n_out: number of nodes that can not improve the retset.

// 进入编译期开关 DYN_PIPE_WIDTH 对应的实现分支；同一算法可按硬件/实验配置切换不同路径。
#ifdef DYN_PIPE_WIDTH
// 进入编译期开关 STATIC_POLICY 对应的实现分支；同一算法可按硬件/实验配置切换不同路径。
#ifdef STATIC_POLICY
      constexpr int kBeamWidths[] = {4, 4, 8, 8, 16, 16, 24, 24, 32};
      // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
      cur_beam_width = kBeamWidths[std::min(max_marker / 5, 8u)];
// 进入上述编译期开关的备用实现分支。
#else
      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (max_marker >= 5 && n_in + n_out > 0) {
        cur_n_in += n_in;
        cur_tot += n_in + n_out;
        // LOG(INFO) << "Current beam width: " << cur_beam_width << " n_in: " << cur_n_in << " n_out: " << cur_tot;
        // converged, tune beam width.
        constexpr double kWasteThreshold = 0.1;  // 0.1 * 10
        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if ((cur_tot - cur_n_in) * 1.0 / cur_tot <= kWasteThreshold) {
          cur_beam_width = cur_beam_width + 1;
          cur_beam_width = std::max(cur_beam_width, 4l);
          cur_beam_width = std::min((int64_t) beam_width, cur_beam_width);
        }
      }
// 结束这一组编译期条件分支。
#endif
// 结束这一组编译期条件分支。
#endif

      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if ((int64_t) on_flight_ios.size() < cur_beam_width) {
// 进入编译期开关 NAIVE_PIPE 对应的实现分支；同一算法可按硬件/实验配置切换不同路径。
#ifdef NAIVE_PIPE
        send_best_read_req(cur_beam_width - on_flight_ios.size());
// 进入上述编译期开关的备用实现分支。
#else
        send_best_read_req(1);
// 结束这一组编译期条件分支。
#endif
      }

      // auto io1_ed = std::chrono::high_resolution_clock::now();
      // stats->io_us1 += std::chrono::duration_cast<std::chrono::microseconds>(io1_ed - io1_st).count();
      // 开始统计这一逻辑阶段的耗时。
      ANN_START_TIMING(calc_best_node_time, calc_best_t);
      // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
      marker = calc_best_node(expand_retries);
      // 结束该阶段计时，并把耗时累计进对应统计项。
      ANN_END_TIMING(calc_best_node_time, calc_best_t);
      // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
      ANN_ADD_STAT(calc_best_node_number, 1);
      // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
      max_marker = std::max(max_marker, marker);
    }

    // LOG(INFO) << "Pipe search expanded distribution:";
    // for (auto &insert : inserts) {
    //   LOG(INFO) << "Insert expanded neighbors: " << insert;
    // }

    // 用断言检查内部不变量，帮助发现 buffer、ID/loc 或容量关系被破坏的情况。
    assert(on_flight_ios.size() == 0);

    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    auto cpu2_ed = std::chrono::high_resolution_clock::now();
    // 初始化或更新本次查询的统计字段，便于拆分 I/O 与 CPU 开销。
    stats->cpu_us2 = std::chrono::duration_cast<std::chrono::microseconds>(cpu2_ed - cpu2_st).count();
    // 初始化或更新本次查询的统计字段，便于拆分 I/O 与 CPU 开销。
    stats->cpu_us = n_computes;

// ---- 搜索结束：按 exact distance 重排真正展开过的节点，然后生成最终 top-k ----
    // 按 Neighbor 的距离顺序排序，保证最近候选位于容器前部。
    std::sort(full_retset.begin(), full_retset.end(),
              // 把当前阶段得到的结果/状态返回给上层调用者。
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    // 搜索结束，撤销本次查询对活跃搜索线程计数的占用。
    this->search_thread_count_--;
    push_query_buf(query_buf);

    // 根据当前搜索状态/编译配置决定是否执行这一分支。
    if (stats != nullptr) {
      // 初始化或更新本次查询的统计字段，便于拆分 I/O 与 CPU 开销。
      stats->total_us = (double) query_timer.elapsed();
    }
  }

  // 模板参数 T 表示向量坐标类型，TagT 表示用户可见标签类型。
  template<typename T, typename TagT>
// ---------------------------------------------------------------------------
// pipe_search：对外包装层。执行 do_pipe_search 后，把内部 ID/距离结果转成用户可见 tag。
// ---------------------------------------------------------------------------
  size_t SSDIndex<T, TagT>::pipe_search(const T *query1, const _u64 k_search, const _u32 mem_L, const _u64 l_search,
                                        TagT *res_tags, float *distances, const _u64 beam_width, QueryStats *stats,
                                        tsl::robin_set<uint32_t> *deleted_nodes, bool dyn_search_l) {
    std::shared_lock lk(merge_lock);
    std::vector<Neighbor> expanded_nodes_info;
    // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
    ANN_INIT_TIMING(search_t);
    // 开始统计这一逻辑阶段的耗时。
    ANN_START_TIMING(search_graph_time, search_t);
    this->do_pipe_search(query1, mem_L, l_search, beam_width, expanded_nodes_info, nullptr, stats, deleted_nodes,
                         dyn_search_l);
    // 结束该阶段计时，并把耗时累计进对应统计项。
    ANN_END_TIMING(search_graph_time, search_t);
    // copy k_search values
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    _u64 t = 0;
    // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
    for (_u64 i = 0; i < expanded_nodes_info.size() && t < k_search && i < l_search; i++) {
      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (i > 0 && expanded_nodes_info[i].id == expanded_nodes_info[i - 1].id) {
        continue;  // deduplicate.
      }
      res_tags[t] = id2tag(expanded_nodes_info[i].id);
      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (distances != nullptr) {
        // 保存当前候选的距离值，后续候选排序和剪枝都以它为依据。
        distances[t] = expanded_nodes_info[i].distance;
      }
      t++;
    }

    // 把当前阶段得到的结果/状态返回给上层调用者。
    return t;
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace ccann
