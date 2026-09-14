// ============================================================================
// 文件逻辑导读
// 这是论文 Algorithm 3 / §5.2 PNE（Parallel Node Expansion）的核心实现。
// 主搜索线程维护按 PQ 近似距离排序的 candidate pool（retset），选择下一节点并读取
// graph node，在主线程计算该节点的 exact distance；随后把“该节点所有未访问邻居的
// PQ 距离计算”提交给 thread-local AsyncRing。完成结果经 poll_all() 合并回 retset。
//
// PM 路径的 node read 本身是同步 byte-range copy；真正与主线程并行的是 neighbor PQ
// computation。非 PM 路径仍可通过统一 reader 接口发 I/O，但本文件的 CCANN 目标配置
// 是 DAX PM。
//
// 文件还实现 §5.3 ACC 和 §5.2 early exit。do_para_search_sync() 保留相同状态机，
// 但把 neighbor PQ computation 放回主线程执行，主要供 ANN_LARGE 配置使用。
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

#ifndef USE_AIO
#include "liburing.h"
#endif

#include "async_comp.h"
#include "circular_buffer.h"

// 关键编译选项：
// USE_AIO 选择 reader 的 AIO 上下文，否则使用 io_uring；FINE_GRAINED_CONCURRENCY 启用细粒度并发路径；
// EARLY_EXIT 启用基于 exact-distance 稳定性的提前停止；NO_ACC_OPT 关闭 ACC 动态 worker 调节；
// ANN_LARGE 在外层入口选择同步 neighbor-PQ 计算变体。

// 进入 CCANN 相关命名空间，后续定义均属于索引实现。
namespace compctx {
  // 每个调用搜索的 OS 线程拥有一个长期复用的计算 ring，避免查询间争用提交队列。
  static thread_local std::unique_ptr<AsyncRing<>> comp_engine = nullptr;
}

// 进入 CCANN 相关命名空间，后续定义均属于索引实现。
namespace ccann {
  struct io_t {
    // 一个 graph-node read 及其候选信息。当前主路径直接使用 QueryBuffer::reqs；
    // 此结构保留给按请求完成度排序/轮询的实现。
    Neighbor nbr;
    // 物理 page 和 node loc，用于从 page buffer 定位 graph node。
    unsigned page_id;
    unsigned loc;
    // 构造/保存一次底层 I/O 描述，其中包含对齐读范围、目标 buffer 以及可选 useful byte range。
    IORequest *read_req;
    bool operator>(const io_t &rhs) const {
      return nbr.distance > rhs.nbr.distance;
    }

    bool operator<(const io_t &rhs) const {
      return nbr.distance < rhs.nbr.distance;
    }

    bool finished() {
      return read_req->finished;
    }
  };

  struct comp_t {
    // 一个已提交的 neighbor-expansion 任务。后四个字段都指向当前 QueryBuffer 中
    // 为该任务预留的槽位，必须等 finished() 后才能复用。
    Neighbor nbr;
    unsigned nnbrs;
    unsigned *node_nbrs;
    float *nbr_dists;
    // 复用 IORequest::finished 作为计算完成令牌；该对象不发起存储 I/O。
    IORequest *comp_req;
    bool finished() {
      return comp_req->finished;
    }
  };

// early-exit 的统计辅助函数只在对应实验配置中编译。
#ifdef EARLY_EXIT
  inline float mean(const std::deque<float> &vals) {
    if (vals.empty())
      return 0.0f;
    float sum = std::accumulate(vals.begin(), vals.end(), 0.0f);
    return sum / vals.size();
  }

  inline float variance(const std::deque<float> &vals, float mean_val) {
    if (vals.size() <= 1)
      return 0.0f;
    float accum = 0.0f;
    for (float v : vals) {
      float diff = v - mean_val;
      accum += diff * diff;
    }
    return accum / (vals.size() - 1);  // 无偏估计
  }

#endif

// calc_best_node() 用负值把 early-exit 状态与正常候选下标放在同一返回值中。
#define NO_EARLY_STOP_FLAG (0)
#define EARLY_STOP_FLAG (-1)
// 已检测到稳定但 exact 结果还不足 k_search：停止发新任务，先排空在途任务。
#define LIKELY_EARLY_STOP_FLAG (-2)

  // 模板参数 T 表示向量坐标类型，TagT 表示用户可见标签类型。
  template<typename T, typename TagT>
// ---------------------------------------------------------------------------
// get_comp_engine：为当前搜索线程懒创建 thread_local AsyncRing。
// 每个首次执行搜索的 OS 线程默认创建 4 个计算 worker。ACC 只激活/停用该 ring
// 内的 worker（范围 1..4）；calc_thread_count_ 统计所有 ring 的活动 worker 总数。
// ---------------------------------------------------------------------------
  AsyncRing<> *SSDIndex<T, TagT>::get_comp_engine() {
    if (unlikely(compctx::comp_engine == nullptr)) {
      compctx::comp_engine = std::make_unique<AsyncRing<>>(
          4, [this](size_t active_workers) { this->calc_thread_count_ -= active_workers; });
      this->calc_thread_count_ += 4;
    }
    return compctx::comp_engine.get();
  }

  // 模板参数 T 表示向量坐标类型，TagT 表示用户可见标签类型。
  template<typename T, typename TagT>
// ---------------------------------------------------------------------------
// do_para_search：PNE 多核搜索主实现。
// 搜索线程做 candidate selection、PM node read 和 exact distance；邻居 PQ distance
// 提交到 AsyncRing。on_flight_comps 对应 Algorithm 3 的 Q，poll_all() 负责收割。
// ---------------------------------------------------------------------------
  void SSDIndex<T, TagT>::do_para_search(const T *query1, uint32_t mem_L, uint32_t l_search, const uint32_t beam_width,
                                         std::vector<Neighbor> &expanded_nodes_info,
                                         tsl::robin_map<uint32_t, T *> *coord_map, QueryStats *stats,
                                         tsl::robin_set<uint32_t> *exclude_nodes /* tags */, bool dyn_search_l,
                                         // 维护本次搜索持有的 page-cache 引用，避免搜索过程中缓存页被提前回收。
                                         std::vector<uint64_t> *passthrough_page_ref, uint32_t k_search) {
    uint32_t original_l_search = l_search;
    // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
    ANN_INIT_TIMING(populate_t);
#ifdef USE_AIO
    // 取得当前线程对应的底层 I/O 上下文，后续 read/send/poll 都复用它。
    void *ctx = reader->get_ctx();
#else
    // 取得当前线程的 io_uring 上下文，并请求 SQPOLL；Pipe/PNE 用它减少提交 I/O 时的系统调用开销。
    void *ctx = reader->get_ctx(IORING_SETUP_SQPOLL);  // use SQ polling only for pipe search.
#endif
    // 取得当前搜索线程独享的 PNE 计算引擎；后续 neighbor-PQ 任务从这里异步提交和 poll。
    auto comp_ring = get_comp_engine();

    // 登记当前活跃搜索调用。ACC 使用活动 search、insert phase 和 PNE worker 的
    // 数量作为 CPU 压力近似值，并不读取操作系统的 CPU utilization 百分比。
    this->search_thread_count_++;

    auto search_threads = this->search_thread_count_.load();
    auto insert_threads = this->insert_thread_count_.load();
    auto calc_threads = this->calc_thread_count_.load();

#ifdef NO_ACC_OPT
    // 实验开关：保留默认 4 个 worker，不执行 ACC 动态调节。
#else
    auto threshold = this->num_cpus;

    if (this->is_index_inserttable) {
      threshold = this->num_cpus * 2;
    }

    // LOG(INFO) << "threshold for para search dynamic calc thread adjustment: " << threshold;
    // 达到阈值时先减少本线程 ring 的一个计算 worker；低于阈值时逐次恢复，最多 4 个。
    // 索引尚未发生插入时阈值为 num_cpus；进入可插入状态后使用 2*num_cpus。
    if (search_threads + insert_threads + calc_threads >= threshold) {
      if (comp_ring->activated_worker_count() > 1) {
        // 异步停用一个 worker；已提交任务仍会完成，主搜索线程无需在这里等待。
        // ACC 借此避免高并发下搜索/插入被计算线程挤占 CPU。
        comp_ring->remove_worker(true);
        this->calc_thread_count_--;
      }
    } else {
      if (comp_ring->activated_worker_count() < 4) {
        // 系统仍有 CPU 余量时恢复一个 PNE worker，提高邻居并行计算能力。
        comp_ring->add_worker();
        this->calc_thread_count_++;
      }
    }
#endif

    if (search_threads + insert_threads + calc_threads > this->peak_cpus) {
      this->peak_cpus = search_threads + insert_threads + calc_threads;
    }

    if (beam_width > MAX_N_COMPUTES) {
      // 遇到违反索引不变量的状态时记录错误；后续通常直接终止以避免继续使用损坏状态。
      LOG(ERROR) << "Beamwidth can not be higher than MAX_N_COMPUTES";
      // 当前状态无法安全恢复，主动终止而不是返回可能错误的搜索结果。
      crash();
    }

    // 从 QueryBuffer 池取得本查询专用的对齐 scratch 空间，并把 query1 拷入对齐查询缓冲。
    QueryBuffer<T> *query_buf = pop_query_buf(query1);

    // 后续距离计算统一使用对齐后的 query 指针，以满足 SIMD/AVX 距离函数的对齐需求。
    const T *query = query_buf->aligned_query_T;

    // 只清空本次查询的运行状态（索引、visited 等），保留已分配的大块 scratch 内存复用。
    query_buf->reset();

    // 取得 DRAM 中的完整向量 scratch；从 PM/SSD 读出的 coords 会复制到这里再做 exact distance。
    T *data_buf = query_buf->coord_scratch;
    // 提前把 DRAM coordinate scratch 拉近 CPU cache，降低随后 exact-distance 写入/读取的冷启动开销。
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // 取得页/sector 读取缓冲；graph node 的原始字节先进入这里，再按 loc 定位 node。
    char *sector_scratch = query_buf->sector_scratch;

    // 取得距离结果 scratch；PQ 批量距离或临时精确距离结果都复用这块对齐内存。
    float *dist_scratch = query_buf->aligned_dist_scratch;
    // 取得 PQ code 聚合 scratch，用来把离散 ID 对应的压缩码整理成连续布局后批量查表。
    _u8 *pq_coord_scratch = query_buf->aligned_pq_coord_scratch;

    Timer query_timer;
    // retset 对应 Algorithm 3 的 C：按 PQ approximate distance 排序，最多保留 L 个。
    std::vector<Neighbor> retset(mem_L + l_search * 10);
    SlidingWindow recentQ(10);
    std::priority_queue<float, std::vector<float>, std::greater<float>> smallestQ;

    std::vector<unsigned int> inserts;
    // 维护已经发现/处理过的 logical ID 集合，避免图中多条边导致重复工作。
    auto &visited = *(query_buf->visited);
    // 维护候选池当前有效元素个数；容器实际 capacity 可能更大。
    unsigned cur_list_size = 0;

    // full_retset 对应 Algorithm 3 的 E：只有读到完整向量并算过 exact distance 的
    // 节点才进入这里。插入调用还会利用这些坐标做 target-neighbor pruning。
    std::vector<Neighbor> &full_retset = expanded_nodes_info;
    // 预留容量以减少搜索热路径中的动态扩容/内存搬迁。
    full_retset.reserve(l_search * 10);

    // 取得 PQ 查表距离缓冲；它保存 query 每个 PQ chunk 到各 centroid 的距离表。
    float *pq_dists = query_buf->aligned_pqtable_dist_scratch;

#ifndef OVERLAP_INIT
    // 预计算 query 到 PQ codebook centroid 的查找表，后续大量候选只需按压缩码查表求近似距离。
    pq_table.populate_chunk_distances(query, pq_dists);  // overlap with the first I/O.
#endif

// ---- 构造 PQ 批量距离 helper：把 ID -> PQ code -> ADC distance 封装起来 ----
    auto compute_pq_dists = [this, pq_dists, query_buf](const unsigned *ids, const _u64 n_ids, float *dists_out,
                                                        _u8 *pq_coord_scratch) {
      // 按 logical ID 从内存 PQ 数据 this->data 中收集压缩码，形成连续批次，便于后续 SIMD/查表计算。
      ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks, pq_coord_scratch);
      // 使用预计算的 query->centroid 距离表，对这一批 PQ code 做 ADC 查表并输出近似距离。
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists, dists_out);
    };

#ifdef EARLY_EXIT
    float prev_median = std::numeric_limits<float>::infinity();
    float alpha = 0;                    // 旧版阈值策略遗留；当前有效判据不读取 alpha
    float alpha_min = 0;                // 与 alpha 一样仅供下方注释掉的策略参考
    float alpha_max = 0.3;
    float tau_stable = 0.05;            // 中位数变化率和历史标准差均低于 5%
    float tau_volatile = 0.1;           // is_volatile 当前未参与控制流
    std::deque<float> median_history;  // 存储最近N个窗口中位数
    unsigned median_window = 5;        // 可调，用于检测趋势稳定性
#endif

    auto push_nbrs = [&](unsigned *nbrs, unsigned nnbrs, float *dist_scratch, unsigned &n_in, unsigned &n_out) {
      // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
      ANN_INIT_TIMING(compute_t);

      // 开始统计这一逻辑阶段的耗时。
      ANN_START_TIMING(expand_neighbors_time, compute_t);
      // 逐个扫描当前 graph node 的邻居 logical ID。
      for (unsigned m = 0; m < nnbrs; ++m) {
        const int nbor_id = nbrs[m];
        const float nbor_dist = dist_scratch[m];
        if (stats != nullptr) {
          stats->n_cmps++;
        }
        // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
        ANN_ADD_STAT(ncalc_for_expanding_neighbors, 1);
        if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search)) {
          // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
          ANN_ADD_STAT(ncalc_for_useless_neighbors, 1);
          n_out++;
          continue;
        }
        n_in++;
        // 构造候选记录：logical ID + 当前距离 + 可展开标记。
        Neighbor nn(nbor_id, nbor_dist, true);
        // 把新候选按距离插入有序 retset，并返回插入位置；这可能让搜索指针 k 回退到更优候选。
        auto r = InsertIntoPool(retset.data(), cur_list_size, nn);  // may be overflow in retset...
        if (cur_list_size < l_search) {
          ++cur_list_size;
          if (unlikely(cur_list_size >= retset.size())) {
            // 调整容器有效容量，确保后续按索引写入不会越界。
            retset.resize(2 * cur_list_size);
          }
        }
      }
      // 结束该阶段计时，并把耗时累计进对应统计项。
      ANN_END_TIMING(expand_neighbors_time, compute_t);
    };

    // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
    auto add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids, float *dists) {
      for (_u64 i = 0; i < n_ids; ++i) {
        // 维护候选池当前有效元素个数；容器实际 capacity 可能更大。
        retset[cur_list_size++] = Neighbor(node_ids[i], dists[i], true);
        // 把入口候选标记为已发现，防止后续从不同边再次重复加入候选池。
        visited.insert(node_ids[i]);
      }
    };

    // 清零本次查询的分项统计；项目调用路径会为 para search 提供有效 stats。
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
    // 先从可选的小型 DRAM index 取得入口；没有内存索引时使用 graph medoid。

#ifdef DYN_PIPE_WIDTH
    int64_t cur_beam_width = 4;  // before converge.
#else
    int64_t cur_beam_width = beam_width;  // before converge.
#endif

    std::vector<unsigned> mem_tags(mem_L);
    std::vector<float> mem_dists(mem_L);

    // 开始统计这一逻辑阶段的耗时。
    ANN_START_TIMING(populate_pq_dists_time, populate_t);
#ifdef OVERLAP_INIT
// ---- 选择搜索入口：优先使用小型 DRAM index；否则从 graph medoid 开始 ----
    if (mem_L) {
      // 用小型内存索引快速产生 mem_L 个入口候选；后续仍会在主图上继续搜索。
      mem_index_->search_with_tags_fast(query, mem_L, mem_tags.data(), mem_dists.data());
      // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
      add_to_retset(mem_tags.data(), std::min((unsigned) mem_L, (unsigned) l_search), mem_dists.data());
      // 生成 PQ 查找表；nt 版本用于和其他初始化/首批 I/O 重叠，减少启动阶段串行开销。
      pq_table.populate_chunk_distances_nt(query, pq_dists);  // overlap with the first I/O.
      compute_pq_dists(mem_tags.data(), mem_L, dist_scratch, pq_coord_scratch);
      // 把当前有效候选按距离排序，使 retset 前部始终代表优先扩展的最近候选。
      std::sort(retset.begin(), retset.begin() + cur_list_size);
    } else {
      // 只有一个 medoid 入口，先建立 PQ 查找表，再计算该入口的距离，无法与内存索引搜索重叠。
      // 生成 PQ 查找表；nt 版本用于和其他初始化/首批 I/O 重叠，减少启动阶段串行开销。
      pq_table.populate_chunk_distances_nt(query, pq_dists);
      compute_pq_dists(&medoids[0], 1, dist_scratch, pq_coord_scratch);
      // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
      add_to_retset(&medoids[0], 1, dist_scratch);
    }
#else
// ---- 选择搜索入口：优先使用小型 DRAM index；否则从 graph medoid 开始 ----
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
#endif
    // 结束该阶段计时，并把耗时累计进对应统计项。
    ANN_END_TIMING(populate_pq_dists_time, populate_t);

// ---- PNE 流水状态：从这里开始 neighbor-PQ computation 与主搜索线程解耦 ----
    // 记录已经提交给计算线程、但结果尚未并回 retset 的邻居 PQ 任务。
    std::queue<comp_t> on_flight_comps;
    // 记录 logical ID 到本查询 node buffer 的映射，供当前 expansion 生命周期内定位已读节点。
    std::unordered_map<unsigned, char *> id_buf_map;

// ---- graph-node 读取 helper：ID -> loc -> page/byte-range -> QueryBuffer ----
    auto send_read_req = [&](Neighbor &item) -> char * {
      // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
      ANN_INIT_TIMING(send_best_t);
      // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
      ANN_INIT_TIMING(read_best_t);
      // 开始统计这一逻辑阶段的耗时。
      ANN_START_TIMING(do_read_best_node_time, read_best_t);
      // 取得当前 ring slot，并在读取期间稳定该 ID 对应的 location/page。
      uint32_t pid;
      // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
      uint64_t &cur_buf_idx = query_buf->sector_idx;
      // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
      auto buf = sector_scratch + cur_buf_idx * size_per_io;
      auto &req = query_buf->reqs[cur_buf_idx];
      auto loc = 0;
#ifdef FINE_GRAINED_CONCURRENCY
      if (this->on_pm) {
        loc = id2loc_func(item.id, [&](uint32_t &loc) {
          // 由 loc 计算 graph node 所在的 4KB sector/page 编号。
          pid = loc_sector_no(loc);
          // u_loc_offset(loc) 给出该 node 在整个 graph 文件中的精确 byte 起点，用于 PM byte-range read。
          req = IORequest(static_cast<_u64>(pid) * SECTOR_LEN, size_per_io, buf, u_loc_offset(loc), max_node_len);
          // PM reader 在 send_io() 内同步完成 byte-range prefetch+copy；非 PM reader
          // 才可能把请求真正留在 I/O 队列中。
          reader->send_io(req, ctx, false);
          if (passthrough_page_ref != nullptr)
            // 维护本次搜索持有的 page-cache 引用，避免搜索过程中缓存页被提前回收。
            passthrough_page_ref->push_back((static_cast<_u64>(pid) * SECTOR_LEN) / SECTOR_LEN);
        });
        // 用断言检查内部不变量，帮助发现 buffer、ID/loc 或容量关系被破坏的情况。
        assert(req.finished == true);
        // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
        ANN_ADD_STAT(send_best_node_number, 1);
        // PM read 已同步完成，直接登记该 ID 在返回 buffer 中的 node 起点。
        id_buf_map.insert(std::make_pair(item.id, offset_to_loc((char *) req.buf, loc)));
      } else {
        // 该细粒度路径依赖 PM 的同步 byte-range read；普通 SSD 配置不支持这一分支。
        // 遇到违反索引不变量的状态时记录错误；后续通常直接终止以避免继续使用损坏状态。
        LOG(ERROR) << "Fine grained concurrency is only supported for PM index.";
        // 当前状态无法安全恢复，主动终止而不是返回可能错误的搜索结果。
        crash();
      }
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
      // PM reader 在 send_io() 内同步完成 byte-range prefetch+copy；非 PM reader
      // 才可能把请求真正留在 I/O 队列中。
      reader->send_io(req, ctx, false);
      // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
      ANN_ADD_STAT(send_best_node_number, 1);
      // 结束该阶段计时，并把耗时累计进对应统计项。
      ANN_END_TIMING(send_best_node_time, send_best_t);
      if (passthrough_page_ref != nullptr)
        // 维护本次搜索持有的 page-cache 引用，避免搜索过程中缓存页被提前回收。
        passthrough_page_ref->push_back((static_cast<_u64>(pid) * SECTOR_LEN) / SECTOR_LEN);

      // send_io 返回后登记 node 起点；PM 路径此时已完成 copy。
      id_buf_map.insert(std::make_pair(item.id, offset_to_loc((char *) req.buf, loc)));

      // node bytes 已进入查询私有 buffer，可以立即释放用于稳定 id2loc 的读锁。
      this->unlock_idx(idx_lock_table, item.id);
#endif
      cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;
      if (stats != nullptr) {
        stats->n_ios++;
      }

      // 结束该阶段计时，并把耗时累计进对应统计项。
      ANN_END_TIMING(do_read_best_node_time, read_best_t);

      return offset_to_loc((char *) req.buf, loc);
    };

// ---- PNE expand helper：筛选未访问邻居，并把 PQ 计算提交给 AsyncRing ----
    auto send_compute_req = [&](Neighbor &item, char *node_buf, bool sync = false) -> bool {
      // closure 捕获本槽位的邻居与 scratch，供 AsyncRing worker 批量计算 PQ 距离。
      uint64_t &cur_comp_idx = query_buf->comp_idx;
      auto dist_buf = (float *) (((_u8 *) dist_scratch) + cur_comp_idx * 512 * sizeof(float));
      auto pq_buf = pq_coord_scratch + cur_comp_idx * 32768 * 32 * sizeof(_u8);
      auto &comp_req = query_buf->comp_reqs[cur_comp_idx];
      // 初始化当前槽位的计算完成令牌；worker 完成 neighbor-PQ 计算后置 finished。
      comp_req = IORequest();  // dummy init
      comp_req.finished = false;

      unsigned *node_nbrs = offset_to_node_nhood(node_buf);
      // 读取 neighbor 数量，并把指针前移到第一个 neighbor ID，后续可直接遍历。
      unsigned nnbrs = *(node_nbrs++);
      unsigned nbors_cand_size = 0;

      // 原地压紧尚未访问的 neighbor IDs。node_buf 位于查询 scratch，而不是持久化
      // graph，因此覆盖已经过滤掉的 slot 不会修改索引。
      for (unsigned m = 0; m < nnbrs; ++m) {
        if (visited.find(node_nbrs[m]) == visited.end()) {
          node_nbrs[nbors_cand_size++] = node_nbrs[m];
          // 维护已经发现/处理过的 logical ID 集合，避免图中多条边导致重复工作。
          visited.insert(node_nbrs[m]);
        }
      }

      auto compute_fn = [this, &compute_pq_dists, node_nbrs, nbors_cand_size, dist_buf, pq_buf,
                         cur_comp_idx]() -> uint64_t {
        // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
        ANN_INIT_TIMING(compute_t);
        // 开始统计这一逻辑阶段的耗时。
        ANN_START_TIMING(expand_neighbors_time, compute_t);
        compute_pq_dists(node_nbrs, nbors_cand_size, dist_buf, pq_buf);
        // 结束该阶段计时，并把耗时累计进对应统计项。
        ANN_END_TIMING(expand_neighbors_time, compute_t);
        return cur_comp_idx;
      };

      cur_comp_idx = (cur_comp_idx + 1) % MAX_N_COMPUTES;

      if (sync) {
        compute_fn();
        // 把对应逻辑计算请求标记完成，使 on_flight_comps 前端任务可以被安全消费。
        comp_req.finished = true;
      } else {
        // 把邻居 PQ 距离计算 closure 提交给 AsyncRing worker，使主搜索线程不必同步等待计算完成。
        comp_ring->send_for_comp(compute_fn);
      }

      on_flight_comps.push(comp_t{item, nbors_cand_size, node_nbrs, dist_buf, &comp_req});

      return true;
    };

// ---- 查找 retset 中最靠前的尚未真正展开候选，用于判断搜索是否收敛 ----
    // 维护已经发现/处理过的 logical ID 集合，避免图中多条边导致重复工作。
    auto get_first_unvisited = [&]() -> int {
      int ret = -1;
      // 在线性候选区间中寻找满足条件的候选；L 通常只有几十到几百，因此这一扫描成本可控。
      for (unsigned i = 0; i < cur_list_size; ++i) {
        if (!retset[i].visited) {
          ret = i;
          break;
        }
      }
      return ret;
    };

// ---- 完成队列处理：收割计算结果并把 neighbor candidates 合并回 C ----
    auto poll_all = [&]() -> std::pair<int, int> {
      // 每轮非阻塞收割当前已经完成的 AsyncRing completion。
      unsigned n_in = 0, n_out = 0;

      // 有在途计算时记录一次 poll；变量名沿用了早期 I/O pipeline 的命名。
      if (!on_flight_comps.empty()) {
        // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
        ANN_ADD_STAT(poll_number, 1);
      }

      // 非阻塞收割当前已经完成的 PNE 计算任务；未完成任务继续留在 worker/队列中。
      auto r = comp_ring->poll_all();
      for (auto &cres : r) {
        // find the corresponding comp_t
        auto cur_comp_idx = (uint64_t) cres.result;
        auto &comp_req = query_buf->comp_reqs[cur_comp_idx];
        // 把对应逻辑计算请求标记完成，使 on_flight_comps 前端任务可以被安全消费。
        comp_req.finished = true;
      }

      // AsyncRing 可以乱序完成；on_flight_comps 按提交顺序消费，避免较早任务仍
      // 引用 scratch slot 时过早复用。已完成但不在队首的任务会暂时等待。
      while (!on_flight_comps.empty() && on_flight_comps.front().finished()) {
        comp_t &comp = on_flight_comps.front();
        unsigned nnbrs = comp.nnbrs;
        unsigned *node_nbrs = comp.node_nbrs;
        float *nbr_dists = comp.nbr_dists;

        push_nbrs(node_nbrs, nnbrs, nbr_dists, n_in, n_out);
        // 该计算任务的 neighbor distances 已经并入候选池，移出 in-flight FIFO。
        on_flight_comps.pop();
      }

      if (n_in + n_out > 0) {
        // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
        ANN_ADD_STAT(poll_hit_number, 1);
      }
      return std::make_pair(n_in, n_out);
    };

    auto cpu2_st = std::chrono::high_resolution_clock::now();
    // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
    int marker = 0, max_marker = 0;

#ifndef STATIC_POLICY
    int cur_n_in = 0, cur_tot = 0;
#endif
    // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
    ANN_INIT_TIMING(poll_t);
    // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
    ANN_INIT_TIMING(calc_best_t);

    // LOG(INFO) << "Start One Query Para Search: beam_width=" << beam_width;
    int expand_retries = 0;
    bool sent = false;
    // 维护已经发现/处理过的 logical ID 集合，避免图中多条边导致重复工作。
    int first_unvisited = 0;
    bool wait_for_flight = false;
    bool early_stop = false;

// ---- 精确距离阶段：完整 coords 已到 DRAM 后，计算 query<->node exact distance ----
    // 这个 helper 负责 expanded node 的完整向量精确距离，并把结果放入 full_retset；early-exit 也基于这些精确距离判断。
    auto compute_exact_dists_and_push = [&](Neighbor &item, const char *node_buf,
                                            const unsigned id) -> std::pair<float, int> {
      // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
      ANN_INIT_TIMING(compute_t);

      // 开始统计这一逻辑阶段的耗时。
      ANN_START_TIMING(calc_exact_dist_time, compute_t);
      T *node_fp_coords_copy = data_buf;
      // Graph node 的首字段就是 coords，因此从 node_buf 起点复制完整向量到 DRAM scratch，再进行精确距离计算。
      memcpy(node_fp_coords_copy, node_buf, data_dim * sizeof(T));

      // 在 CPU 上对 query 与 DRAM scratch 中的完整向量做 exact distance；底层可走 AVX-512/AVX2。
      auto cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
      int stop_flag = NO_EARLY_STOP_FLAG;
      // 插入的 search phase 需要保留完整坐标，普通 top-k 查询传入 nullptr。
      if (coord_map != nullptr) {
        coord_map->insert(std::make_pair(id, node_fp_coords_copy));
      }

      // LOG(INFO) << "Expanding node " << id << " distance " << cur_expanded_dist;
      // 只有确实读过完整 vector 并计算 exact distance 的节点才进入 E。
      full_retset.push_back(Neighbor(id, cur_expanded_dist, true));

#ifdef EARLY_EXIT
      // 论文称在探索 L/2 后开始观察稳定窗口；代码严格使用 max_marker > L/2。
      // 论文还建议只在 L>=128 时启用，但这里没有运行时门槛：只要编译了
      // EARLY_EXIT，任何 L 都会进入该逻辑，实验配置需自行保证 L 的取值。
      int start_thresh = l_search / 2;
      if (max_marker > start_thresh) {
        // uint32_t E = std::min((uint32_t) (l_search - start_thresh), 2 * k_search);

        // if (smallestQ.size() < E) {
        //   smallestQ.push(cur_expanded_dist);
        // } else {
        //   if (cur_expanded_dist < smallestQ.top()) {
        //     smallestQ.pop();
        //     smallestQ.push(cur_expanded_dist);
        //   }
        // }
        // // +inf
        // auto R_q = std::numeric_limits<float>::infinity();
        // if (smallestQ.size() == E) {
        //   R_q = smallestQ.top();
        // }

        // recentQ 保存最近 10 个 exact distance。实现先取窗口中位数，再观察
        // 最近 5 个窗口中位数的变化率和标准差，而非直接计算这 10 个值的方差。
        recentQ.push(cur_expanded_dist);
        if (recentQ.filled()) {
          float med = recentQ.median();
          float delta_med = fabs(med - prev_median) / (prev_median + 1e-6);

          median_history.push_back(med);
          if (median_history.size() > median_window)
            median_history.pop_front();

          float meanM = mean(median_history);
          float varM = variance(median_history, meanM);
          float stdM = std::sqrt(varM);

          // 稳定条件：窗口中位数变化 <5%，且最近五个窗口中位数的标准差
          // 小于其均值的 5%。这是论文“variation and variance <5%”的具体实现。
          bool is_stable = (delta_med < tau_stable) && (stdM < tau_stable * meanM);
          bool is_volatile = (delta_med > tau_volatile) || (stdM > tau_volatile * meanM);

          prev_median = med;
          // 连续窗口的中位数变化足够小且窗口内波动足够低时，判定 exact distance 已稳定。
          if (is_stable) {
            // LOG(INFO) << "Search stable detected. Median: " << med << ", Delta: " << delta_med << ", Std: " << stdM
            //           << ", R_q: " << R_q << ", R_q * (1 - alpha): " << R_q * (1 - alpha);
            // alpha = std::min(alpha_max, (float) (alpha + 0.1));  // 更激进
            // LOG(INFO) << "Search stable detected. Increasing alpha to " << alpha;
            // current worst distance in retset
            // only do this when we have enough candidates, and the median is stable
            // if (med >= R_q * (1 - alpha)) {
            if (full_retset.size() > k_search) {
              // can terminate early
              // 满足收敛条件且已有足够结果，允许直接结束搜索，减少后续无效展开。
              stop_flag = EARLY_STOP_FLAG;
            } else {
              // likely early stop, do not send new computations
              // 接近收敛但还不立即返回：停止继续发新计算，让已经 in-flight 的工作自然排空。
              stop_flag = LIKELY_EARLY_STOP_FLAG;
            }
            // }
          }
          // else {
          //   alpha = std::max(alpha_min, (float) (alpha - 0.1));  // 更保守
          //   // LOG(INFO) << "Search volatile detected. Decreasing alpha to " << alpha;
          // }
        }
      }
#endif

      // 结束该阶段计时，并把耗时累计进对应统计项。
      ANN_END_TIMING(calc_exact_dist_time, compute_t);

      return std::make_pair(cur_expanded_dist, stop_flag);
    };

// ---- 候选选择核心：结合 exact distance/当前 retset 判断下一步是否继续扩展 ----
    auto calc_best_node = [&](int &expand_retries, bool &sent) -> int {  // if converged.
      // auto cpu_st = std::chrono::high_resolution_clock::now();
      // 维护候选池当前有效元素个数；容器实际 capacity 可能更大。
      unsigned marker = 0, nk = cur_list_size, first_unvisited_eager = cur_list_size;
      /* calculate one from "already read" */
      // 在线性候选区间中寻找满足条件的候选；L 通常只有几十到几百，因此这一扫描成本可控。
      for (marker = 0; marker < cur_list_size; ++marker) {
        // exact distance 会追加 E，并可能触发 early exit；neighbor PQ 结果稍后还会
        // 改写 C，因此每次只选择一个候选，下一轮重新从有序 retset 扫描。
        if (!retset[marker].visited) {
          // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
          auto id = retset[marker].id;

          if ((int64_t) on_flight_comps.size() < cur_beam_width) {
            // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
            auto buf = send_read_req(retset[marker]);
            // LOG(INFO) << "Exploring marker @ " << marker;
            // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
            retset[marker].visited = true;
            // 这个 helper 负责 expanded node 的完整向量精确距离，并把结果放入 full_retset；early-exit 也基于这些精确距离判断。
            auto [exact_dist, stop_flag] = compute_exact_dists_and_push(retset[marker], buf, id);
            if (stop_flag != NO_EARLY_STOP_FLAG) {
              // LOG(INFO) << "Early stop at expand retries " << expand_retries;
              return stop_flag;
            }
            // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
            send_compute_req(retset[marker], buf, false);
            sent = true;
          }
          break;
        }
      }

      /* guess the first unvisited vector (eager) */
      // 在线性候选区间中寻找满足条件的候选；L 通常只有几十到几百，因此这一扫描成本可控。
      for (unsigned i = marker; i < cur_list_size; ++i) {
        if (!retset[i].visited) {
          // 维护已经发现/处理过的 logical ID 集合，避免图中多条边导致重复工作。
          first_unvisited_eager = i;
          break;
        }
      }
      return first_unvisited_eager;
      // auto cpu_ed = std::chrono::high_resolution_clock::now();
      // stats->cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(cpu_ed - cpu_st).count();
    };

// ---- Algorithm 3 主循环：收割 Q -> 更新 C -> 选 top-1 -> 读/算 exact -> 提交 expand ----
    // 只要候选池仍有未访问节点，或 Q 中仍有待收割的 expansion，就继续 Algorithm 3。
    while (((first_unvisited = get_first_unvisited()) != -1) || !on_flight_comps.empty()) {
      sent = false;
      // 开始统计这一逻辑阶段的耗时。
      ANN_START_TIMING(poll_all_time, poll_t);
      auto [n_in, n_out] = poll_all();
      // 结束该阶段计时，并把耗时累计进对应统计项。
      ANN_END_TIMING(poll_all_time, poll_t);
      std::ignore = n_in;
      std::ignore = n_out;

      expand_retries++;

      if (wait_for_flight && n_in + n_out > 0) {
        if (n_in == 0) {
          // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
          ANN_ADD_STAT(para_useless_wait_for_flight_number, 1);
        }
        wait_for_flight = false;
      }

      if (n_in + n_out == 0 && first_unvisited == -1) {
        // LOG(INFO) << "Waiting for on flight computations to finish. On flight comps: " << on_flight_comps.size();
        wait_for_flight = true;
        // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
        ANN_ADD_STAT(para_wait_for_flight_number, 1);
        sched_yield();
        continue;
      }

      // if (n_in + n_out > 0) {
      //   if (n_in > 0) {
      //     LOG(INFO) << "Expand retries " << expand_retries << " got n_in " << n_in << " n_out " << n_out
      //               << " max_marker " << max_marker;
      //   } else {
      //     LOG(INFO) << "Expand retries " << expand_retries << " got 0 n_ins ";
      //   }
      // }

      if (early_stop) {
        // tag all remaining as visited.
        // 在线性候选区间中寻找满足条件的候选；L 通常只有几十到几百，因此这一扫描成本可控。
        for (unsigned i = 0; i < cur_list_size; ++i) {
          // 维护已经发现/处理过的 logical ID 集合，避免图中多条边导致重复工作。
          retset[i].visited = true;
        }
      } else {
        // 开始统计这一逻辑阶段的耗时。
        ANN_START_TIMING(calc_best_node_time, calc_best_t);
        // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
        marker = calc_best_node(expand_retries, sent);
        // 结束该阶段计时，并把耗时累计进对应统计项。
        ANN_END_TIMING(calc_best_node_time, calc_best_t);
        // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
        ANN_ADD_STAT(calc_best_node_number, 1);
        // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
        max_marker = std::max(max_marker, marker);

        if (n_in + n_out == 0 && sent == false && first_unvisited != -1) {
          // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
          ANN_ADD_STAT(para_busy_wait_number, 1);
          // LOG(INFO) << "Busy wait due to small beam width.";
          cur_beam_width = cur_beam_width + 1;
          cur_beam_width = std::max(cur_beam_width, 4l);
          cur_beam_width = std::min((int64_t) beam_width, cur_beam_width);
        }

        if (marker == EARLY_STOP_FLAG) {
          // LOG(INFO) << "early stop at expand retries " << expand_retries;
          // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
          ANN_ADD_STAT(para_early_exit_number, 1);
          early_stop = true;
        // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
        } else if (marker == LIKELY_EARLY_STOP_FLAG) {
          // LOG(INFO) << "likely early stop at expand retries " << expand_retries;
          // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
          ANN_ADD_STAT(para_likely_early_exit_number, 1);
          // do nothing, wait for retset to be full.
        }
      }
    }

    // LOG(INFO) << "Pipe search expanded distribution: " << expand_retries;

    // for (auto &insert : inserts) {
    //   LOG(INFO) << "Insert expanded neighbors: " << insert;
    // }
    // LOG(INFO) << "Remaining on flight comps: " << on_flight_comps.size();

    // 用断言检查内部不变量，帮助发现 buffer、ID/loc 或容量关系被破坏的情况。
    assert(on_flight_comps.size() == 0);

    auto cpu2_ed = std::chrono::high_resolution_clock::now();
    // 初始化或更新本次查询的统计字段，便于拆分 I/O 与 CPU 开销。
    stats->cpu_us2 = std::chrono::duration_cast<std::chrono::microseconds>(cpu2_ed - cpu2_st).count();
    // stats->cpu_us = n_computes;

// ---- 搜索结束：按 exact distance 重排真正展开过的节点，然后生成最终 top-k ----
    // 按 Neighbor 的距离顺序排序，保证最近候选位于容器前部。
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    // 搜索结束，撤销本次查询对活跃搜索线程计数的占用。
    this->search_thread_count_--;
    push_query_buf(query_buf);

    if (stats != nullptr) {
      // 初始化或更新本次查询的统计字段，便于拆分 I/O 与 CPU 开销。
      stats->total_us = (double) query_timer.elapsed();
    }
  }

  // 模板参数 T 表示向量坐标类型，TagT 表示用户可见标签类型。
  template<typename T, typename TagT>
// ---------------------------------------------------------------------------
// do_para_search_sync：ANN_LARGE 使用的同步 neighbor-expansion 变体。
// 它复用 C/E/Q、PM byte-range read 和 early-exit 状态机，但 send_compute_req(..., true)
// 直接在搜索线程计算 PQ distances，不创建/使用 AsyncRing worker。因此它保持 PNE 的
// 调度结构，却不具备普通 do_para_search() 的多核计算并行性。
// ---------------------------------------------------------------------------
  void SSDIndex<T, TagT>::do_para_search_sync(const T *query1, uint32_t mem_L, uint32_t l_search,
                                              const uint32_t beam_width, std::vector<Neighbor> &expanded_nodes_info,
                                              tsl::robin_map<uint32_t, T *> *coord_map, QueryStats *stats,
                                              tsl::robin_set<uint32_t> *exclude_nodes /* tags */, bool dyn_search_l,
                                              // 维护本次搜索持有的 page-cache 引用，避免搜索过程中缓存页被提前回收。
                                              std::vector<uint64_t> *passthrough_page_ref, uint32_t k_search) {
    uint32_t original_l_search = l_search;
    // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
    ANN_INIT_TIMING(populate_t);
#ifdef USE_AIO
    // 取得当前线程对应的底层 I/O 上下文，后续 read/send/poll 都复用它。
    void *ctx = reader->get_ctx();
#else
    // 取得当前线程的 io_uring 上下文，并请求 SQPOLL；Pipe/PNE 用它减少提交 I/O 时的系统调用开销。
    void *ctx = reader->get_ctx(IORING_SETUP_SQPOLL);  // use SQ polling only for pipe search.
#endif

    // 登记当前活跃搜索线程；ACC 会把搜索、插入和计算 worker 的总并发作为资源压力信号。
    this->search_thread_count_++;
    auto search_threads = this->search_thread_count_.load();
    auto insert_threads = this->insert_thread_count_.load();
    auto calc_threads = this->calc_thread_count_.load();
    
    if (search_threads + insert_threads + calc_threads > this->peak_cpus) {
      this->peak_cpus = search_threads + insert_threads + calc_threads;
    }

    if (beam_width > MAX_N_COMPUTES) {
      // 遇到违反索引不变量的状态时记录错误；后续通常直接终止以避免继续使用损坏状态。
      LOG(ERROR) << "Beamwidth can not be higher than MAX_N_COMPUTES";
      // 当前状态无法安全恢复，主动终止而不是返回可能错误的搜索结果。
      crash();
    }

    // 从 QueryBuffer 池取得本查询专用的对齐 scratch 空间，并把 query1 拷入对齐查询缓冲。
    QueryBuffer<T> *query_buf = pop_query_buf(query1);

    // 后续距离计算统一使用对齐后的 query 指针，以满足 SIMD/AVX 距离函数的对齐需求。
    const T *query = query_buf->aligned_query_T;

    // 只清空本次查询的运行状态（索引、visited 等），保留已分配的大块 scratch 内存复用。
    query_buf->reset();

    // 取得 DRAM 中的完整向量 scratch；从 PM/SSD 读出的 coords 会复制到这里再做 exact distance。
    T *data_buf = query_buf->coord_scratch;
    // 提前把 DRAM coordinate scratch 拉近 CPU cache，降低随后 exact-distance 写入/读取的冷启动开销。
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // 取得页/sector 读取缓冲；graph node 的原始字节先进入这里，再按 loc 定位 node。
    char *sector_scratch = query_buf->sector_scratch;

    // 取得距离结果 scratch；PQ 批量距离或临时精确距离结果都复用这块对齐内存。
    float *dist_scratch = query_buf->aligned_dist_scratch;
    // 取得 PQ code 聚合 scratch，用来把离散 ID 对应的压缩码整理成连续布局后批量查表。
    _u8 *pq_coord_scratch = query_buf->aligned_pq_coord_scratch;

    Timer query_timer;
    // retset 对应 Algorithm 3 的 C：按 PQ approximate distance 排序，最多保留 L 个。
    std::vector<Neighbor> retset(mem_L + l_search * 10);
    SlidingWindow recentQ(10);
    std::priority_queue<float, std::vector<float>, std::greater<float>> smallestQ;

    std::vector<unsigned int> inserts;
    // 维护已经发现/处理过的 logical ID 集合，避免图中多条边导致重复工作。
    auto &visited = *(query_buf->visited);
    // 维护候选池当前有效元素个数；容器实际 capacity 可能更大。
    unsigned cur_list_size = 0;

    // full_retset 对应 Algorithm 3 的 E：只记录读到完整向量并算过 exact distance 的节点。
    std::vector<Neighbor> &full_retset = expanded_nodes_info;
    // 预留容量以减少搜索热路径中的动态扩容/内存搬迁。
    full_retset.reserve(l_search * 10);

    // 取得 PQ 查表距离缓冲；它保存 query 每个 PQ chunk 到各 centroid 的距离表。
    float *pq_dists = query_buf->aligned_pqtable_dist_scratch;

#ifndef OVERLAP_INIT
    // 预计算 query 到 PQ codebook centroid 的查找表，后续大量候选只需按压缩码查表求近似距离。
    pq_table.populate_chunk_distances(query, pq_dists);  // overlap with the first I/O.
#endif

// ---- 构造 PQ 批量距离 helper：把 ID -> PQ code -> ADC distance 封装起来 ----
    auto compute_pq_dists = [this, pq_dists, query_buf](const unsigned *ids, const _u64 n_ids, float *dists_out,
                                                        _u8 *pq_coord_scratch) {
      // 按 logical ID 从内存 PQ 数据 this->data 中收集压缩码，形成连续批次，便于后续 SIMD/查表计算。
      ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks, pq_coord_scratch);
      // 使用预计算的 query->centroid 距离表，对这一批 PQ code 做 ADC 查表并输出近似距离。
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists, dists_out);
    };

#ifdef EARLY_EXIT
    float prev_median = std::numeric_limits<float>::infinity();
    float alpha = 0;                    // 旧版阈值策略遗留；当前有效判据不读取 alpha
    float alpha_min = 0;                // 与 alpha 一样仅供下方注释掉的策略参考
    float alpha_max = 0.3;
    float tau_stable = 0.05;            // 中位数变化率和历史标准差均低于 5%
    float tau_volatile = 0.1;           // is_volatile 当前未参与控制流
    std::deque<float> median_history;  // 存储最近N个窗口中位数
    unsigned median_window = 5;        // 可调，用于检测趋势稳定性
#endif

    auto push_nbrs = [&](unsigned *nbrs, unsigned nnbrs, float *dist_scratch, unsigned &n_in, unsigned &n_out) {
      // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
      ANN_INIT_TIMING(compute_t);

      // 开始统计这一逻辑阶段的耗时。
      ANN_START_TIMING(expand_neighbors_time, compute_t);
      // 逐个扫描当前 graph node 的邻居 logical ID。
      for (unsigned m = 0; m < nnbrs; ++m) {
        const int nbor_id = nbrs[m];
        const float nbor_dist = dist_scratch[m];
        if (stats != nullptr) {
          stats->n_cmps++;
        }
        // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
        ANN_ADD_STAT(ncalc_for_expanding_neighbors, 1);
        if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search)) {
          // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
          ANN_ADD_STAT(ncalc_for_useless_neighbors, 1);
          n_out++;
          continue;
        }
        n_in++;
        // 构造候选记录：logical ID + 当前距离 + 可展开标记。
        Neighbor nn(nbor_id, nbor_dist, true);
        // 把新候选按距离插入有序 retset，并返回插入位置；这可能让搜索指针 k 回退到更优候选。
        auto r = InsertIntoPool(retset.data(), cur_list_size, nn);  // may be overflow in retset...
        if (cur_list_size < l_search) {
          ++cur_list_size;
          if (unlikely(cur_list_size >= retset.size())) {
            // 调整容器有效容量，确保后续按索引写入不会越界。
            retset.resize(2 * cur_list_size);
          }
        }
      }
      // 结束该阶段计时，并把耗时累计进对应统计项。
      ANN_END_TIMING(expand_neighbors_time, compute_t);
    };

    // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
    auto add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids, float *dists) {
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

#ifdef DYN_PIPE_WIDTH
    int64_t cur_beam_width = 4;  // before converge.
#else
    int64_t cur_beam_width = beam_width;  // before converge.
#endif

    std::vector<unsigned> mem_tags(mem_L);
    std::vector<float> mem_dists(mem_L);

    // 开始统计这一逻辑阶段的耗时。
    ANN_START_TIMING(populate_pq_dists_time, populate_t);
#ifdef OVERLAP_INIT
// ---- 选择搜索入口：优先使用小型 DRAM index；否则从 graph medoid 开始 ----
    if (mem_L) {
      // 用小型内存索引快速产生 mem_L 个入口候选；后续仍会在主图上继续搜索。
      mem_index_->search_with_tags_fast(query, mem_L, mem_tags.data(), mem_dists.data());
      // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
      add_to_retset(mem_tags.data(), std::min((unsigned) mem_L, (unsigned) l_search), mem_dists.data());
      // 生成 PQ 查找表；nt 版本用于和其他初始化/首批 I/O 重叠，减少启动阶段串行开销。
      pq_table.populate_chunk_distances_nt(query, pq_dists);  // overlap with the first I/O.
      compute_pq_dists(mem_tags.data(), mem_L, dist_scratch, pq_coord_scratch);
      // 把当前有效候选按距离排序，使 retset 前部始终代表优先扩展的最近候选。
      std::sort(retset.begin(), retset.begin() + cur_list_size);
    } else {
      // 只有一个 medoid 入口，先建立 PQ 查找表，再计算该入口的距离，无法与内存索引搜索重叠。
      // 生成 PQ 查找表；nt 版本用于和其他初始化/首批 I/O 重叠，减少启动阶段串行开销。
      pq_table.populate_chunk_distances_nt(query, pq_dists);
      compute_pq_dists(&medoids[0], 1, dist_scratch, pq_coord_scratch);
      // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
      add_to_retset(&medoids[0], 1, dist_scratch);
    }
#else
// ---- 选择搜索入口：优先使用小型 DRAM index；否则从 graph medoid 开始 ----
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
#endif
    // 结束该阶段计时，并把耗时累计进对应统计项。
    ANN_END_TIMING(populate_pq_dists_time, populate_t);

// ---- 同步变体复用 Algorithm 3 的 Q 数据结构，但 neighbor-PQ computation 在提交点已完成 ----
    // 保存尚未并回 retset 的 expansion；这些任务的完成令牌在入队前已经置位。
    std::queue<comp_t> on_flight_comps;
    // 记录 logical ID 到本查询 node buffer 的映射，供当前 expansion 生命周期内定位已读节点。
    std::unordered_map<unsigned, char *> id_buf_map;

// ---- graph-node 读取 helper：ID -> loc -> page/byte-range -> QueryBuffer ----
    auto send_read_req = [&](Neighbor &item) -> char * {
      // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
      ANN_INIT_TIMING(send_best_t);
      // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
      ANN_INIT_TIMING(read_best_t);
      // 开始统计这一逻辑阶段的耗时。
      ANN_START_TIMING(do_read_best_node_time, read_best_t);
      // 取得当前 ring slot，并在读取期间稳定该 ID 对应的 location/page。
      uint32_t pid;
      // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
      uint64_t &cur_buf_idx = query_buf->sector_idx;
      // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
      auto buf = sector_scratch + cur_buf_idx * size_per_io;
      auto &req = query_buf->reqs[cur_buf_idx];
      auto loc = 0;
#ifdef FINE_GRAINED_CONCURRENCY
      if (this->on_pm) {
        loc = id2loc_func(item.id, [&](uint32_t &loc) {
          // 由 loc 计算 graph node 所在的 4KB sector/page 编号。
          pid = loc_sector_no(loc);
          // u_loc_offset(loc) 给出该 node 在整个 graph 文件中的精确 byte 起点，用于 PM byte-range read。
          req = IORequest(static_cast<_u64>(pid) * SECTOR_LEN, size_per_io, buf, u_loc_offset(loc), max_node_len);
          // PM reader 在 send_io() 内同步完成 byte-range prefetch+copy；非 PM reader
          // 才可能把请求真正留在 I/O 队列中。
          reader->send_io(req, ctx, false);
          if (passthrough_page_ref != nullptr)
            // 维护本次搜索持有的 page-cache 引用，避免搜索过程中缓存页被提前回收。
            passthrough_page_ref->push_back((static_cast<_u64>(pid) * SECTOR_LEN) / SECTOR_LEN);
        });
        // 用断言检查内部不变量，帮助发现 buffer、ID/loc 或容量关系被破坏的情况。
        assert(req.finished == true);
        // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
        ANN_ADD_STAT(send_best_node_number, 1);
        // PM read 已同步完成，直接登记该 ID 在返回 buffer 中的 node 起点。
        id_buf_map.insert(std::make_pair(item.id, offset_to_loc((char *) req.buf, loc)));
      } else {
        // 该细粒度路径依赖 PM 的同步 byte-range read；普通 SSD 配置不支持这一分支。
        // 遇到违反索引不变量的状态时记录错误；后续通常直接终止以避免继续使用损坏状态。
        LOG(ERROR) << "Fine grained concurrency is only supported for PM index.";
        // 当前状态无法安全恢复，主动终止而不是返回可能错误的搜索结果。
        crash();
      }
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
      // PM reader 在 send_io() 内同步完成 byte-range prefetch+copy；非 PM reader
      // 才可能把请求真正留在 I/O 队列中。
      reader->send_io(req, ctx, false);
      // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
      ANN_ADD_STAT(send_best_node_number, 1);
      // 结束该阶段计时，并把耗时累计进对应统计项。
      ANN_END_TIMING(send_best_node_time, send_best_t);
      if (passthrough_page_ref != nullptr)
        // 维护本次搜索持有的 page-cache 引用，避免搜索过程中缓存页被提前回收。
        passthrough_page_ref->push_back((static_cast<_u64>(pid) * SECTOR_LEN) / SECTOR_LEN);

      // send_io 返回后登记 node 起点；PM 路径此时已完成 copy。
      id_buf_map.insert(std::make_pair(item.id, offset_to_loc((char *) req.buf, loc)));

      // node bytes 已进入查询私有 buffer，可以立即释放用于稳定 id2loc 的读锁。
      this->unlock_idx(idx_lock_table, item.id);
#endif
      cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;
      if (stats != nullptr) {
        stats->n_ios++;
      }

      // 结束该阶段计时，并把耗时累计进对应统计项。
      ANN_END_TIMING(do_read_best_node_time, read_best_t);

      return offset_to_loc((char *) req.buf, loc);
    };

// ---- 同步 expand helper：筛选未访问邻居，并在当前线程计算 PQ distances ----
    auto send_compute_req = [&](Neighbor &item, char *node_buf, bool sync = false) -> bool {
      // 同步变体沿用 closure 组织计算，但在当前线程直接调用，不提交 AsyncRing。
      uint64_t &cur_comp_idx = query_buf->comp_idx;
      auto dist_buf = (float *) (((_u8 *) dist_scratch) + cur_comp_idx * 512 * sizeof(float));
      auto pq_buf = pq_coord_scratch + cur_comp_idx * 32768 * 32 * sizeof(_u8);
      auto &comp_req = query_buf->comp_reqs[cur_comp_idx];
      // 同步变体仍复用相同队列结构；这里初始化槽位的计算完成令牌。
      comp_req = IORequest();  // dummy init
      comp_req.finished = false;

      unsigned *node_nbrs = offset_to_node_nhood(node_buf);
      // 读取 neighbor 数量，并把指针前移到第一个 neighbor ID，后续可直接遍历。
      unsigned nnbrs = *(node_nbrs++);
      unsigned nbors_cand_size = 0;

      // 原地压紧尚未访问的 neighbor IDs。node_buf 位于查询 scratch，而不是持久化
      // graph，因此覆盖已经过滤掉的 slot 不会修改索引。
      for (unsigned m = 0; m < nnbrs; ++m) {
        if (visited.find(node_nbrs[m]) == visited.end()) {
          node_nbrs[nbors_cand_size++] = node_nbrs[m];
          // 维护已经发现/处理过的 logical ID 集合，避免图中多条边导致重复工作。
          visited.insert(node_nbrs[m]);
        }
      }

      auto compute_fn = [this, &compute_pq_dists, node_nbrs, nbors_cand_size, dist_buf, pq_buf,
                         cur_comp_idx]() -> uint64_t {
        // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
        ANN_INIT_TIMING(compute_t);
        // 开始统计这一逻辑阶段的耗时。
        ANN_START_TIMING(expand_neighbors_time, compute_t);
        compute_pq_dists(node_nbrs, nbors_cand_size, dist_buf, pq_buf);
        // 结束该阶段计时，并把耗时累计进对应统计项。
        ANN_END_TIMING(expand_neighbors_time, compute_t);
        return cur_comp_idx;
      };

      cur_comp_idx = (cur_comp_idx + 1) % MAX_N_COMPUTES;

      compute_fn();
      // 把对应逻辑计算请求标记完成，使 on_flight_comps 前端任务可以被安全消费。
      comp_req.finished = true;

      on_flight_comps.push(comp_t{item, nbors_cand_size, node_nbrs, dist_buf, &comp_req});

      return true;
    };

// ---- 查找 retset 中最靠前的尚未真正展开候选，用于判断搜索是否收敛 ----
    // 维护已经发现/处理过的 logical ID 集合，避免图中多条边导致重复工作。
    auto get_first_unvisited = [&]() -> int {
      int ret = -1;
      // 在线性候选区间中寻找满足条件的候选；L 通常只有几十到几百，因此这一扫描成本可控。
      for (unsigned i = 0; i < cur_list_size; ++i) {
        if (!retset[i].visited) {
          ret = i;
          break;
        }
      }
      return ret;
    };

// ---- 同步变体仍沿用完成队列，把计算结果统一合并回 C ----
    auto poll_all = [&]() -> std::pair<int, int> {
      // 同步计算在提交时已完成，这里统一把 completion 合并回候选池。
      unsigned n_in = 0, n_out = 0;

      // 变量名保留自 pipeline 版本；该分支没有真正的异步计算 I/O。
      if (!on_flight_comps.empty()) {
        // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
        ANN_ADD_STAT(poll_number, 1);
      }

      // 同步计算已在提交处完成，因此按 FIFO 立即收割所有已完成 expansion。
      while (!on_flight_comps.empty() && on_flight_comps.front().finished()) {
        comp_t &comp = on_flight_comps.front();
        unsigned nnbrs = comp.nnbrs;
        unsigned *node_nbrs = comp.node_nbrs;
        float *nbr_dists = comp.nbr_dists;

        push_nbrs(node_nbrs, nnbrs, nbr_dists, n_in, n_out);
        // 该计算任务的 neighbor distances 已经并入候选池，移出 in-flight FIFO。
        on_flight_comps.pop();
      }

      if (n_in + n_out > 0) {
        // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
        ANN_ADD_STAT(poll_hit_number, 1);
      }
      return std::make_pair(n_in, n_out);
    };

    auto cpu2_st = std::chrono::high_resolution_clock::now();
    // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
    int marker = 0, max_marker = 0;

#ifndef STATIC_POLICY
    int cur_n_in = 0, cur_tot = 0;
#endif
    // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
    ANN_INIT_TIMING(poll_t);
    // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
    ANN_INIT_TIMING(calc_best_t);

    // LOG(INFO) << "Start One Query Para Search: beam_width=" << beam_width;
    int expand_retries = 0;
    bool sent = false;
    // 维护已经发现/处理过的 logical ID 集合，避免图中多条边导致重复工作。
    int first_unvisited = 0;
    bool wait_for_flight = false;
    bool early_stop = false;

// ---- 精确距离阶段：完整 coords 已到 DRAM 后，计算 query<->node exact distance ----
    // 这个 helper 负责 expanded node 的完整向量精确距离，并把结果放入 full_retset；early-exit 也基于这些精确距离判断。
    auto compute_exact_dists_and_push = [&](Neighbor &item, const char *node_buf,
                                            const unsigned id) -> std::pair<float, int> {
      // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
      ANN_INIT_TIMING(compute_t);

      // 开始统计这一逻辑阶段的耗时。
      ANN_START_TIMING(calc_exact_dist_time, compute_t);
      T *node_fp_coords_copy = data_buf;
      // Graph node 的首字段就是 coords，因此从 node_buf 起点复制完整向量到 DRAM scratch，再进行精确距离计算。
      memcpy(node_fp_coords_copy, node_buf, data_dim * sizeof(T));

      // 在 CPU 上对 query 与 DRAM scratch 中的完整向量做 exact distance；底层可走 AVX-512/AVX2。
      auto cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
      int stop_flag = NO_EARLY_STOP_FLAG;
      // 插入搜索需要保留完整坐标，普通 top-k 查询传入 nullptr。
      if (coord_map != nullptr) {
        coord_map->insert(std::make_pair(id, node_fp_coords_copy));
      }

      // LOG(INFO) << "Expanding node " << id << " distance " << cur_expanded_dist;
      // 只有确实读过完整 vector 并计算 exact distance 的节点才进入 E。
      full_retset.push_back(Neighbor(id, cur_expanded_dist, true));

#ifdef EARLY_EXIT
      // 与异步版相同：编译 EARLY_EXIT 后没有额外检查 L>=128。
      int start_thresh = l_search / 2;
      if (max_marker > start_thresh) {
        // uint32_t E = std::min((uint32_t) (l_search - start_thresh), 2 * k_search);

        // if (smallestQ.size() < E) {
        //   smallestQ.push(cur_expanded_dist);
        // } else {
        //   if (cur_expanded_dist < smallestQ.top()) {
        //     smallestQ.pop();
        //     smallestQ.push(cur_expanded_dist);
        //   }
        // }
        // // +inf
        // auto R_q = std::numeric_limits<float>::infinity();
        // if (smallestQ.size() == E) {
        //   R_q = smallestQ.top();
        // }

        // 最近 10 个 exact distance 形成窗口，再观察最近 5 个窗口中位数。
        recentQ.push(cur_expanded_dist);
        if (recentQ.filled()) {
          float med = recentQ.median();
          float delta_med = fabs(med - prev_median) / (prev_median + 1e-6);

          median_history.push_back(med);
          if (median_history.size() > median_window)
            median_history.pop_front();

          float meanM = mean(median_history);
          float varM = variance(median_history, meanM);
          float stdM = std::sqrt(varM);

          // 使用窗口中位数的相对变化和历史标准差共同判断稳定。
          bool is_stable = (delta_med < tau_stable) && (stdM < tau_stable * meanM);
          bool is_volatile = (delta_med > tau_volatile) || (stdM > tau_volatile * meanM);

          prev_median = med;
          // 连续窗口的中位数变化足够小且窗口内波动足够低时，判定 exact distance 已稳定。
          if (is_stable) {
            // LOG(INFO) << "Search stable detected. Median: " << med << ", Delta: " << delta_med << ", Std: " << stdM
            //           << ", R_q: " << R_q << ", R_q * (1 - alpha): " << R_q * (1 - alpha);
            // alpha = std::min(alpha_max, (float) (alpha + 0.1));  // 更激进
            // LOG(INFO) << "Search stable detected. Increasing alpha to " << alpha;
            // current worst distance in retset
            // only do this when we have enough candidates, and the median is stable
            // if (med >= R_q * (1 - alpha)) {
            if (full_retset.size() > k_search) {
              // can terminate early
              // 满足收敛条件且已有足够结果，允许直接结束搜索，减少后续无效展开。
              stop_flag = EARLY_STOP_FLAG;
            } else {
              // likely early stop, do not send new computations
              // 接近收敛但还不立即返回：停止继续发新计算，让已经 in-flight 的工作自然排空。
              stop_flag = LIKELY_EARLY_STOP_FLAG;
            }
            // }
          }
          // else {
          //   alpha = std::max(alpha_min, (float) (alpha - 0.1));  // 更保守
          //   // LOG(INFO) << "Search volatile detected. Decreasing alpha to " << alpha;
          // }
        }
      }
#endif

      // 结束该阶段计时，并把耗时累计进对应统计项。
      ANN_END_TIMING(calc_exact_dist_time, compute_t);

      return std::make_pair(cur_expanded_dist, stop_flag);
    };

// ---- 候选选择核心：结合 exact distance/当前 retset 判断下一步是否继续扩展 ----
    auto calc_best_node = [&](int &expand_retries, bool &sent) -> int {  // if converged.
      // auto cpu_st = std::chrono::high_resolution_clock::now();
      // 维护候选池当前有效元素个数；容器实际 capacity 可能更大。
      unsigned marker = 0, nk = cur_list_size, first_unvisited_eager = cur_list_size;
      /* calculate one from "already read" */
      // 在线性候选区间中寻找满足条件的候选；L 通常只有几十到几百，因此这一扫描成本可控。
      for (marker = 0; marker < cur_list_size; ++marker) {
        // 每次处理一个候选，下一轮基于已经更新的有序 retset 重新选择。
        if (!retset[marker].visited) {
          // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
          auto id = retset[marker].id;

          if ((int64_t) on_flight_comps.size() < cur_beam_width) {
            // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
            auto buf = send_read_req(retset[marker]);
            // LOG(INFO) << "Exploring marker @ " << marker;
            // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
            retset[marker].visited = true;
            // 这个 helper 负责 expanded node 的完整向量精确距离，并把结果放入 full_retset；early-exit 也基于这些精确距离判断。
            auto [exact_dist, stop_flag] = compute_exact_dists_and_push(retset[marker], buf, id);
            if (stop_flag != NO_EARLY_STOP_FLAG) {
              // LOG(INFO) << "Early stop at expand retries " << expand_retries;
              return stop_flag;
            }
            // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
            send_compute_req(retset[marker], buf, true);
            sent = true;
          }
          break;
        }
      }

      /* guess the first unvisited vector (eager) */
      // 在线性候选区间中寻找满足条件的候选；L 通常只有几十到几百，因此这一扫描成本可控。
      for (unsigned i = marker; i < cur_list_size; ++i) {
        if (!retset[i].visited) {
          // 维护已经发现/处理过的 logical ID 集合，避免图中多条边导致重复工作。
          first_unvisited_eager = i;
          break;
        }
      }
      return first_unvisited_eager;
      // auto cpu_ed = std::chrono::high_resolution_clock::now();
      // stats->cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(cpu_ed - cpu_st).count();
    };

// ---- 同步变体的 Algorithm 3 状态循环；Q 中任务在提交时已经完成计算 ----
    // 只要候选池仍有未访问节点，或 Q 中仍有待收割的 expansion，就继续 Algorithm 3。
    while (((first_unvisited = get_first_unvisited()) != -1) || !on_flight_comps.empty()) {
      sent = false;
      // 开始统计这一逻辑阶段的耗时。
      ANN_START_TIMING(poll_all_time, poll_t);
      auto [n_in, n_out] = poll_all();
      // 结束该阶段计时，并把耗时累计进对应统计项。
      ANN_END_TIMING(poll_all_time, poll_t);
      std::ignore = n_in;
      std::ignore = n_out;

      expand_retries++;

      if (wait_for_flight && n_in + n_out > 0) {
        if (n_in == 0) {
          // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
          ANN_ADD_STAT(para_useless_wait_for_flight_number, 1);
        }
        wait_for_flight = false;
      }

      if (n_in + n_out == 0 && first_unvisited == -1) {
        // LOG(INFO) << "Waiting for on flight computations to finish. On flight comps: " << on_flight_comps.size();
        wait_for_flight = true;
        // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
        ANN_ADD_STAT(para_wait_for_flight_number, 1);
        sched_yield();
        continue;
      }

      // if (n_in + n_out > 0) {
      //   if (n_in > 0) {
      //     LOG(INFO) << "Expand retries " << expand_retries << " got n_in " << n_in << " n_out " << n_out
      //               << " max_marker " << max_marker;
      //   } else {
      //     LOG(INFO) << "Expand retries " << expand_retries << " got 0 n_ins ";
      //   }
      // }

      if (early_stop) {
        // tag all remaining as visited.
        // 在线性候选区间中寻找满足条件的候选；L 通常只有几十到几百，因此这一扫描成本可控。
        for (unsigned i = 0; i < cur_list_size; ++i) {
          // 维护已经发现/处理过的 logical ID 集合，避免图中多条边导致重复工作。
          retset[i].visited = true;
        }
      } else {
        // 开始统计这一逻辑阶段的耗时。
        ANN_START_TIMING(calc_best_node_time, calc_best_t);
        // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
        marker = calc_best_node(expand_retries, sent);
        // 结束该阶段计时，并把耗时累计进对应统计项。
        ANN_END_TIMING(calc_best_node_time, calc_best_t);
        // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
        ANN_ADD_STAT(calc_best_node_number, 1);
        // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
        max_marker = std::max(max_marker, marker);

        if (n_in + n_out == 0 && sent == false && first_unvisited != -1) {
          // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
          ANN_ADD_STAT(para_busy_wait_number, 1);
          // LOG(INFO) << "Busy wait due to small beam width.";
          cur_beam_width = cur_beam_width + 1;
          cur_beam_width = std::max(cur_beam_width, 4l);
          cur_beam_width = std::min((int64_t) beam_width, cur_beam_width);
        }

        if (marker == EARLY_STOP_FLAG) {
          // LOG(INFO) << "early stop at expand retries " << expand_retries;
          // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
          ANN_ADD_STAT(para_early_exit_number, 1);
          early_stop = true;
        // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
        } else if (marker == LIKELY_EARLY_STOP_FLAG) {
          // LOG(INFO) << "likely early stop at expand retries " << expand_retries;
          // 累加实验统计计数，用于论文中的性能 breakdown/无效工作分析。
          ANN_ADD_STAT(para_likely_early_exit_number, 1);
          // do nothing, wait for retset to be full.
        }
      }
    }

    // LOG(INFO) << "Pipe search expanded distribution: " << expand_retries;

    // for (auto &insert : inserts) {
    //   LOG(INFO) << "Insert expanded neighbors: " << insert;
    // }
    // LOG(INFO) << "Remaining on flight comps: " << on_flight_comps.size();

    // 用断言检查内部不变量，帮助发现 buffer、ID/loc 或容量关系被破坏的情况。
    assert(on_flight_comps.size() == 0);

    auto cpu2_ed = std::chrono::high_resolution_clock::now();
    // 初始化或更新本次查询的统计字段，便于拆分 I/O 与 CPU 开销。
    stats->cpu_us2 = std::chrono::duration_cast<std::chrono::microseconds>(cpu2_ed - cpu2_st).count();
    // stats->cpu_us = n_computes;

// ---- 搜索结束：按 exact distance 重排真正展开过的节点，然后生成最终 top-k ----
    // 按 Neighbor 的距离顺序排序，保证最近候选位于容器前部。
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    // 搜索结束，撤销本次查询对活跃搜索线程计数的占用。
    this->search_thread_count_--;
    push_query_buf(query_buf);

    if (stats != nullptr) {
      // 初始化或更新本次查询的统计字段，便于拆分 I/O 与 CPU 开销。
      stats->total_us = (double) query_timer.elapsed();
    }
  }

  // 模板参数 T 表示向量坐标类型，TagT 表示用户可见标签类型。
  template<typename T, typename TagT>
// ---------------------------------------------------------------------------
// para_search：对外查询包装层。普通配置执行多核 PNE，ANN_LARGE 执行同步变体；
// 最后把按 exact distance 排序的 logical IDs 去重并转换成用户 TagT。
// ---------------------------------------------------------------------------
  size_t SSDIndex<T, TagT>::para_search(const T *query1, const _u64 k_search, const _u32 mem_L, const _u64 l_search,
                                        TagT *res_tags, float *distances, const _u64 beam_width, QueryStats *stats,
                                        tsl::robin_set<uint32_t> *deleted_nodes, bool dyn_search_l) {
    std::shared_lock lk(merge_lock);
    std::vector<Neighbor> expanded_nodes_info;
#ifdef ANN_LARGE
    this->do_para_search_sync(query1, mem_L, l_search, beam_width, expanded_nodes_info, nullptr, stats, deleted_nodes,
                              dyn_search_l, nullptr, k_search);
#else
    this->do_para_search(query1, mem_L, l_search, beam_width, expanded_nodes_info, nullptr, stats, deleted_nodes,
                         dyn_search_l, nullptr, k_search);
#endif
    // full_retset 已按 exact distance 排序；去掉相邻重复 ID，再输出前 k_search 个 tag。
    _u64 t = 0;
    for (_u64 i = 0; i < expanded_nodes_info.size() && t < k_search && i < l_search; i++) {
      if (i > 0 && expanded_nodes_info[i].id == expanded_nodes_info[i - 1].id) {
        continue;  // deduplicate.
      }
      res_tags[t] = id2tag(expanded_nodes_info[i].id);
      if (distances != nullptr) {
        // 保存当前候选的距离值，后续候选排序和剪枝都以它为依据。
        distances[t] = expanded_nodes_info[i].distance;
      }
      t++;
    }

    return t;
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace ccann
