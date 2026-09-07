// ============================================================================
// 文件逻辑导读
// 这部分实现传统 beam search 以及对外的 beam_search 包装函数。
// 主线可以概括为：准备 QueryBuffer/PQ 查找表 -> 选择入口候选 -> 每轮从 retset
// 取最多 beam_width 个尚未展开的节点 -> 通过 ID->loc 找到当前有效物理版本 ->
// 读取 graph node -> 用完整 coords 做精确距离 -> 对邻居用 PQ 距离批量估计 ->
// 将有竞争力的新候选插回 retset -> 收敛后按精确距离整理 expanded_nodes_info。
// 这里尤其要区分：retset 中大量使用 PQ 近似距离，而 full_retset/expanded_nodes_info
// 保存的是实际被展开节点基于完整向量计算出的精确距离。
// ============================================================================
#include "aligned_file_reader.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#include <filesystem>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include "timer.h"
#include "tsl/robin_map.h"
#include "utils.h"
#include "v2/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

// 进入 CCANN 相关命名空间，后续定义均属于索引实现。
namespace ccann {
  // 模板参数 T 表示向量坐标类型，TagT 表示用户可见标签类型。
  template<typename T, typename TagT>
// ---------------------------------------------------------------------------
// do_beam_search：传统 beam search 的内部实现。
// 输入 query1；输出 expanded_nodes_info（被真正展开过的节点及其精确距离）。
// retset 是按“搜索阶段距离”维护的候选池；每轮取最多 beam_width 个 flag=true 的候选，
// 读取其 graph node 后再扩展邻居。对于邻居只先算 PQ 近似距离，只有节点真的被展开时
// 才读取完整 coords 并计算 exact distance。
// ---------------------------------------------------------------------------
  void SSDIndex<T, TagT>::do_beam_search(const T *query1, uint32_t mem_L, uint32_t l_search, const uint32_t beam_width,
                                         std::vector<Neighbor> &expanded_nodes_info,
                                         tsl::robin_map<uint32_t, T *> *coord_map, QueryStats *stats,
                                         tsl::robin_set<uint32_t> *exclude_nodes /* tags */, bool dyn_search_l,
                                         // 维护本次搜索持有的 page-cache 引用，避免搜索过程中缓存页被提前回收。
                                         std::vector<uint64_t> *passthrough_page_ref, uint32_t k_search) {
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    uint32_t original_l_search = l_search;
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    auto diskSearchBegin = std::chrono::high_resolution_clock::now();
    // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
    ANN_INIT_TIMING(populate_t);

    // 登记当前活跃搜索线程；ACC 会把搜索、插入和计算 worker 的总并发作为资源压力信号。
    this->search_thread_count_++;

    // 从 QueryBuffer 池取得本查询的线程私有 scratch buffer，避免搜索热路径反复分配内存。
    auto query_buf = pop_query_buf(query1);
    // 取得当前线程对应的底层 I/O 上下文，后续 read/send/poll 都复用它。
    void *ctx = reader->get_ctx();

    // 后续距离计算统一使用对齐后的 query 指针，以满足 SIMD/AVX 距离函数的对齐需求。
    const T *query = query_buf->aligned_query_T;

    // reset query
    // 只清空本次查询的运行状态（索引、visited 等），保留已分配的大块 scratch 内存复用。
    query_buf->reset();

    // pointers to buffers for data
    // 取得 DRAM 中的完整向量 scratch；从 PM/SSD 读出的 coords 会复制到这里再做 exact distance。
    T *data_buf = query_buf->coord_scratch;
    // 推进完整向量 scratch 的槽位索引，保证不同 expanded node 的 coords 不互相覆盖。
    _u64 &data_buf_idx = query_buf->coord_idx;
    // 提前把 DRAM coordinate scratch 拉近 CPU cache，降低随后 exact-distance 写入/读取的冷启动开销。
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    // 取得页/sector 读取缓冲；graph node 的原始字节先进入这里，再按 loc 定位 node。
    char *sector_scratch = query_buf->sector_scratch;
    // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
    _u64 &sector_scratch_idx = query_buf->sector_idx;

    // query <-> PQ chunk centers distances
    // pq_dists is grouped by chunk
    // 取得 PQ 查表距离缓冲；它保存 query 每个 PQ chunk 到各 centroid 的距离表。
    float *pq_dists = query_buf->aligned_pqtable_dist_scratch;
    // pq_dists[i][j] means distance of chunk i of query to center j
    // size in [n_chunks, 256]
    // 开始统计这一逻辑阶段的耗时。
    ANN_START_TIMING(populate_pq_dists_time, populate_t);
    // 预计算 query 到 PQ codebook centroid 的查找表，后续大量候选只需按压缩码查表求近似距离。
    pq_table.populate_chunk_distances(query, pq_dists);
    // 结束该阶段计时，并把耗时累计进对应统计项。
    ANN_END_TIMING(populate_pq_dists_time, populate_t);

    // query <-> neighbor list
    // 取得距离结果 scratch；PQ 批量距离或临时精确距离结果都复用这块对齐内存。
    float *dist_scratch = query_buf->aligned_dist_scratch;
    // 取得 PQ code 聚合 scratch，用来把离散 ID 对应的压缩码整理成连续布局后批量查表。
    _u8 *pq_coord_scratch = query_buf->aligned_pq_coord_scratch;

    // lambda to batch compute query<-> node distances in PQ space
// ---- 构造 PQ 批量距离 helper：后面所有邻居筛选都走这一条近似距离路径 ----
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    auto compute_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids, const _u64 n_ids, float *dists_out) {
      // this->data stores the compressed PQ vectors
      // aggregate_coords copies the compressed vectors of ids into pq_coord_scratch
      // 按 logical ID 从内存 PQ 数据 this->data 中收集压缩码，形成连续批次，便于后续 SIMD/查表计算。
      ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks, pq_coord_scratch);
      // 使用预计算的 query->centroid 距离表，对这一批 PQ code 做 ADC 查表并输出近似距离。
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists, dists_out);
    };

    Timer query_timer, io_timer, cpu_timer;
    // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
    std::vector<Neighbor> retset;
    // 调整容器有效容量，确保后续按索引写入不会越界。
    retset.resize(mem_L + 10 * l_search);
    // 维护已经发现/处理过的 logical ID 集合，避免图中多条边导致重复工作。
    tsl::robin_set<_u64> visited(4096);

    // re-naming `expanded_nodes_info` to not change rest of the code
    // 把调用者提供的 expanded_nodes_info 作为“真正展开节点”的结果容器；这里记录 exact distance。
    std::vector<Neighbor> &full_retset = expanded_nodes_info;
    // 预留容量以减少搜索热路径中的动态扩容/内存搬迁。
    full_retset.reserve(10 * l_search);

    // get the entry point
    // 选择持久化图的 medoid 作为默认入口点；若有小型内存索引则会用更好的入口候选替代。
    _u32 best_medoid = medoids[0];

    // 维护候选池当前有效元素个数；容器实际 capacity 可能更大。
    unsigned cur_list_size = 0;
// ---- 初始化候选池 helper：批量算入口候选 PQ 距离并登记 visited ----
    // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
    auto compute_and_add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids) {
      compute_dists(node_ids, n_ids, dist_scratch);
      // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
      for (_u64 i = 0; i < n_ids; ++i) {
        // 维护候选池当前有效元素个数；容器实际 capacity 可能更大。
        retset[cur_list_size].id = node_ids[i];
        // 维护候选池当前有效元素个数；容器实际 capacity 可能更大。
        retset[cur_list_size].distance = dist_scratch[i];
        // 维护候选池当前有效元素个数；容器实际 capacity 可能更大。
        retset[cur_list_size++].flag = true;
        // 把入口候选标记为已发现，防止后续从不同边再次重复加入候选池。
        visited.insert(node_ids[i]);
      }
    };

// ---- 选择搜索入口：优先使用小型 DRAM index；否则从 graph medoid 开始 ----
    // 根据当前搜索状态/编译配置决定是否执行这一分支。
    if (mem_L) {
      std::vector<unsigned> mem_tags(mem_L);
      std::vector<float> mem_dists(mem_L);
      // 先在小型 DRAM 索引中搜索入口候选，减少从单一 medoid 开始时的图遍历距离。
      mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data());
      // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
      compute_and_add_to_retset(mem_tags.data(), std::min((unsigned) mem_L, (unsigned) l_search));
    } else {
      // Do not use optimized start point.
      // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
      compute_and_add_to_retset(&best_medoid, 1);
    }

    // 把当前有效候选按距离排序，使 retset 前部始终代表优先扩展的最近候选。
    std::sort(retset.begin(), retset.begin() + cur_list_size);

    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    unsigned cmps = 0;
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    unsigned hops = 0;
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    unsigned num_ios = 0;
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    unsigned k = 0;

    // cleared every iteration
    // 操作当前一轮准备读取/展开的 beam frontier。
    std::vector<unsigned> frontier;
    using fnhood_t = std::tuple<unsigned, unsigned, char *>;
    // 操作当前一轮准备读取/展开的 beam frontier。
    std::vector<fnhood_t> frontier_nhoods;
    // 操作当前一轮准备读取/展开的 beam frontier。
    std::vector<IORequest> frontier_read_reqs;
    std::vector<uint32_t> vec_rdlocks;

    // 维护本次搜索持有的 page-cache 引用，避免搜索过程中缓存页被提前回收。
    std::vector<uint64_t> new_page_ref{};
    // 维护本次搜索持有的 page-cache 引用，避免搜索过程中缓存页被提前回收。
    std::vector<uint64_t> &page_ref = passthrough_page_ref ? *passthrough_page_ref : new_page_ref;

// ---- 主 beam-search 循环：不断选择 beam、读 graph node、扩邻居，直到没有更优未展开候选 ----
    // 进入以“仍有未完成搜索工作”为条件的循环，直到候选或 in-flight 工作全部收敛。
    while (k < cur_list_size) {
      // 维护候选池当前有效元素个数；容器实际 capacity 可能更大。
      auto nk = cur_list_size;
      // clear iteration state
      // 清空上一轮 beam 的节点 ID；本轮会重新从 retset 选择尚未展开的最近候选。
      frontier.clear();
      // 清空上一轮“候选 ID/loc/读取缓冲”的关联，避免复用过期 node buffer。
      frontier_nhoods.clear();
      // 清空上一轮 I/O request 列表，为本轮 frontier 重新构造读请求。
      frontier_read_reqs.clear();
      // 清除上一轮的逻辑内容但尽量保留已分配容量，供下一轮复用。
      vec_rdlocks.clear();
      // 从 sector scratch 的第一个槽重新放置本轮读回的 page，上一轮内容不再需要。
      sector_scratch_idx = 0;
      // find new beam
      // WAS: _u64 marker = k - 1;
      // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
      _u32 marker = k;
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      _u32 num_seen = 0;

      // get top-beam_width nearest nodes from retset that are flagged
      // 进入以“仍有未完成搜索工作”为条件的循环，直到候选或 in-flight 工作全部收敛。
      while (marker < cur_list_size && frontier.size() < beam_width && num_seen < beam_width) {
        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if (retset[marker].flag) {
          num_seen++;
          // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
          frontier.push_back(retset[marker].id);
          // 标记该候选已经被选入本轮展开，避免下一轮再次发起相同 graph-node 读取。
          retset[marker].flag = false;
        }
        marker++;
      }

      // read nhoods of frontier ids
      std::vector<uint32_t> locked;
      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (!frontier.empty()) {
        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if (stats != nullptr)
          stats->n_hops++;
        // 操作当前一轮准备读取/展开的 beam frontier。
        locked = this->lock_idx(idx_lock_table, kInvalidID, frontier, true);
        // read nhoods of frontier
        // 逐个处理本轮 frontier 中已经选出的待展开节点。
        for (_u64 i = 0; i < frontier.size(); i++) {
          // 操作当前一轮准备读取/展开的 beam frontier。
          uint32_t id = frontier[i];
          // 把稳定 logical ID 翻译成当前有效 physical loc；Soft Insert 后 ID 不变但 loc 可能已切换到新版本。
          uint32_t loc = this->id2loc(id);
          // 把 loc 对应页号转换为文件/PM 映射中的页起始 byte offset。
          uint64_t offset = loc_sector_no(loc) * SECTOR_LEN;
          // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
          auto sector_buf = sector_scratch + sector_scratch_idx * size_per_io;
          // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
          fnhood_t fnhood = std::make_tuple(id, loc, sector_buf);
          sector_scratch_idx++;
          // 操作当前一轮准备读取/展开的 beam frontier。
          frontier_nhoods.push_back(fnhood);
          // u_loc_offset(loc) 给出该 node 在整个 graph 文件中的精确 byte 起点，用于 PM byte-range read。
          frontier_read_reqs.emplace_back(IORequest(offset, size_per_io, sector_buf, u_loc_offset(loc), max_node_len));
          // 根据当前搜索状态/编译配置决定是否执行这一分支。
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
          num_ios++;
        }
        // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
        ANN_INIT_TIMING(read_best_t);
        io_timer.reset();

        // 开始统计这一逻辑阶段的耗时。
        ANN_START_TIMING(do_read_best_node_time, read_best_t);
// 进入编译期开关 DIRECT_READ_CC 对应的实现分支；同一算法可按硬件/实验配置切换不同路径。
#ifdef DIRECT_READ_CC
        // 批量执行本轮 frontier 的读取；PM 路径可以利用 request 的 useful range 做细粒度复制。
        reader->read(frontier_read_reqs, ctx);
        // 逐个处理本轮 frontier 中已经选出的待展开节点。
        for (auto &req : frontier_read_reqs) {
          // 维护本次搜索持有的 page-cache 引用，避免搜索过程中缓存页被提前回收。
          page_ref.push_back(req.offset / SECTOR_LEN);
        }
// 进入上述编译期开关的备用实现分支。
#else
        // 通过带 page-cache/引用跟踪的读路径获取 frontier pages，并记录 page_ref 供后续释放。
        reader->read_alloc(frontier_read_reqs, ctx, &page_ref);
// 结束这一组编译期条件分支。
#endif
        // 结束该阶段计时，并把耗时累计进对应统计项。
        ANN_END_TIMING(do_read_best_node_time, read_best_t);

        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if (stats != nullptr) {
          // 初始化或更新本次查询的统计字段，便于拆分 I/O 与 CPU 开销。
          stats->io_us += (double) io_timer.elapsed();
        }
        this->unlock_idx(idx_lock_table, locked);
      }
      // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
      ANN_INIT_TIMING(expand_t);
      // 声明该阶段的性能计时器；只用于实验统计，不改变搜索语义。
      ANN_INIT_TIMING(compute_t);
      // 逐个处理本轮 frontier 中已经选出的待展开节点。
      for (auto &frontier_nhood : frontier_nhoods) {
        // 操作当前一轮准备读取/展开的 beam frontier。
        auto [id, loc, sector_buf] = frontier_nhood;
        // 根据 loc 在已经读入的 sector buffer 中定位目标 graph-node slot。
        char *node_disk_buf = offset_to_loc(sector_buf, loc);
        // 跳过 coords 区域，定位 graph node 的邻接表头；第一个 uint32 是 nnbrs，后面才是 neighbor IDs。
        unsigned *node_buf = offset_to_node_nhood(node_disk_buf);
        // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
        _u64 nnbrs = (_u64) (*node_buf);
        // 取得 graph node 开头的完整向量 coords；这是 exact distance 使用的未压缩向量。
        T *node_fp_coords = offset_to_node_coords(node_disk_buf);
        // 用断言检查内部不变量，帮助发现 buffer、ID/loc 或容量关系被破坏的情况。
        assert(data_buf_idx < MAX_N_CMPS);

        // 推进完整向量 scratch 的槽位索引，保证不同 expanded node 的 coords 不互相覆盖。
        T *node_fp_coords_copy = data_buf + (data_buf_idx * aligned_dim);
        // 推进完整向量 scratch 的槽位索引，保证不同 expanded node 的 coords 不互相覆盖。
        data_buf_idx++;
        // 只复制真实 data_dim 个坐标到 DRAM scratch；对齐后的尾部通常由 scratch 初始化策略保证可安全用于 aligned_dim 计算。
        memcpy(node_fp_coords_copy, node_fp_coords, data_dim * sizeof(T));
        // 开始统计这一逻辑阶段的耗时。
        ANN_START_TIMING(calc_exact_dist_time, compute_t);
        // 在 CPU 上对 query 与 DRAM scratch 中的完整向量做 exact distance；底层可走 AVX-512/AVX2。
        float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
        // 结束该阶段计时，并把耗时累计进对应统计项。
        ANN_END_TIMING(calc_exact_dist_time, compute_t);

        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if (coord_map != nullptr) {
          coord_map->insert(std::make_pair(id, node_fp_coords_copy));
        }
        // 把“确实读过完整 vector 并计算 exact distance”的节点加入最终展开集合。
        full_retset.push_back(Neighbor(id, cur_expanded_dist, true));

        // 跳过邻接表头的 nnbrs 字段，node_nbrs 现在指向连续的 neighbor logical IDs。
        unsigned *node_nbrs = (node_buf + 1);

        // compute node_nbrs <-> query dist in PQ space
        cpu_timer.reset();
        // 开始统计这一逻辑阶段的耗时。
        ANN_START_TIMING(expand_neighbors_time, expand_t);
        // 对当前节点的所有邻居批量计算 PQ 近似距离；这里不读取每个邻居的完整 coords。
        compute_dists(node_nbrs, nnbrs, dist_scratch);
        // 结束该阶段计时，并把耗时累计进对应统计项。
        ANN_END_TIMING(expand_neighbors_time, expand_t);
        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if (stats != nullptr) {
          // 初始化或更新本次查询的统计字段，便于拆分 I/O 与 CPU 开销。
          stats->n_cmps += (double) nnbrs;
          // 初始化或更新本次查询的统计字段，便于拆分 I/O 与 CPU 开销。
          stats->cpu_us += (double) cpu_timer.elapsed();
        }

        cpu_timer.reset();
        // process prefetch-ed nhood
        // 逐个扫描当前 graph node 的邻居 logical ID。
        for (_u64 m = 0; m < nnbrs; ++m) {
          // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
          unsigned id = node_nbrs[m];
          // 根据当前搜索状态/编译配置决定是否执行这一分支。
          if (unlikely(id > this->cur_id)) {
            // 遇到违反索引不变量的状态时记录错误；后续通常直接终止以避免继续使用损坏状态。
            LOG(ERROR) << "ID is larger than current ID, " << id << " vs " << this->cur_id;
            // 当前状态无法安全恢复，主动终止而不是返回可能错误的搜索结果。
            crash();
          }
          // 根据当前搜索状态/编译配置决定是否执行这一分支。
          if (visited.find(id) != visited.end()) {
            // the node has been visited
            continue;
          } else {
            // 立即记录该 logical ID 已发现；即使它稍后因距离太差被丢弃，也避免重复 PQ 计算。
            visited.insert(id);
            cmps++;
            // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
            float dist = dist_scratch[m];
            // 根据当前搜索状态/编译配置决定是否执行这一分支。
            if (stats != nullptr) {
              stats->n_cmps++;
            }
            // 根据当前搜索状态/编译配置决定是否执行这一分支。
            if (dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search))
              continue;
            // 构造候选记录：logical ID + 当前距离 + 可展开标记。
            Neighbor nn(id, dist, true);
            // variable search_L for deleted nodes.
            // Return position in sorted list where nn inserted.

            // 把新候选按距离插入有序 retset，并返回插入位置；这可能让搜索指针 k 回退到更优候选。
            auto r = InsertIntoPool(retset.data(), cur_list_size, nn);

            // 根据当前搜索状态/编译配置决定是否执行这一分支。
            if (cur_list_size < l_search) {
              ++cur_list_size;
              // 根据当前搜索状态/编译配置决定是否执行这一分支。
              if (unlikely(cur_list_size >= retset.size())) {
                // 调整容器有效容量，确保后续按索引写入不会越界。
                retset.resize(2 * cur_list_size);
              }
            }

            // 根据当前搜索状态/编译配置决定是否执行这一分支。
            if (r < nk)
              // nk 记录本轮新邻居插入后最靠前的候选位置，用来决定搜索指针是否需要回退。
              nk = r;  // nk logs the best position in the retset that was
                       // updated due to neighbors of n.
          }
        }

        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if (dyn_search_l) {
          // TODO(gh): contention still exists in id2tag(x)
          // O(n), but it is not slow as L is typically smaller than 300.
          // l_search monotonically increases to handle deleted nodes.
          // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
          _u32 tot = 0, cur = 0;
          // 在线性候选区间中寻找满足条件的候选；L 通常只有几十到几百，因此这一扫描成本可控。
          for (cur = 0; cur < cur_list_size; ++cur) {
            // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
            uint32_t tag = id2tag(retset[cur].id);
            // 根据当前搜索状态/编译配置决定是否执行这一分支。
            if (exclude_nodes->find(tag) == exclude_nodes->end()) {
              ++tot;
              // 根据当前搜索状态/编译配置决定是否执行这一分支。
              if (tot == original_l_search) {
                break;
              }
            }
          }
          // cur is the stopped index (cur + 1 is the length it should be)
          l_search = std::max(original_l_search, cur + 1);
        }

        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if (stats != nullptr) {
          // 初始化或更新本次查询的统计字段，便于拆分 I/O 与 CPU 开销。
          stats->cpu_us += (double) cpu_timer.elapsed();
        }
      }

      // update best inserted position
      //

      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (nk <= k)
        // nk 记录本轮新邻居插入后最靠前的候选位置，用来决定搜索指针是否需要回退。
        k = nk;  // k is the best position in retset updated in this round.
      // 进入前述条件不成立时的备用路径。
      else
        ++k;

      hops++;
      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (stats != nullptr && stats->n_current_used != 0) {
        // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
        auto diskSearchEnd = std::chrono::high_resolution_clock::now();
        double elapsedSeconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(diskSearchEnd - diskSearchBegin).count();
        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if (elapsedSeconds >= stats->n_current_used)
          break;
      }
    }
    // re-sort by distance
// ---- 搜索结束：按 exact distance 重排真正展开过的节点，然后生成最终 top-k ----
    // 按 Neighbor 的距离顺序排序，保证最近候选位于容器前部。
    std::sort(full_retset.begin(), full_retset.end(),
              // 把当前阶段得到的结果/状态返回给上层调用者。
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

// 进入编译期开关 DIRECT_READ_CC 对应的实现分支；同一算法可按硬件/实验配置切换不同路径。
#ifndef DIRECT_READ_CC
    // 根据当前搜索状态/编译配置决定是否执行这一分支。
    if (passthrough_page_ref == nullptr) {
      // 释放本次搜索通过 page cache 持有的页引用，允许缓存后续回收这些页。
      reader->deref(&page_ref, ctx);
    }
// 结束这一组编译期条件分支。
#endif

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
// beam_search：对外包装层。调用 do_beam_search 得到 expanded_nodes_info，随后按精确距离
// 排序/过滤，并把内部 logical ID 转成对外 TagT 返回给调用者。
// ---------------------------------------------------------------------------
  size_t SSDIndex<T, TagT>::beam_search(const T *query, const _u64 k_search, const _u32 mem_L, const _u64 l_search,
                                        TagT *res_tags, float *distances, const _u64 beam_width, QueryStats *stats,
                                        tsl::robin_set<uint32_t> *deleted_nodes, bool dyn_search_l) {
    // iterate to fixed point
    std::shared_lock lk(merge_lock);
    std::vector<Neighbor> expanded_nodes_info;
    this->do_beam_search(query, mem_L, (_u32) l_search, (_u32) beam_width, expanded_nodes_info, nullptr, stats,
                         deleted_nodes, dyn_search_l);
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    _u64 res_count = 0;
    // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
    for (uint32_t i = 0; i < l_search && res_count < k_search && i < expanded_nodes_info.size(); i++) {
      res_tags[res_count] = id2tag(expanded_nodes_info[i].id);
      // 保存当前候选的距离值，后续候选排序和剪枝都以它为依据。
      distances[res_count] = expanded_nodes_info[i].distance;
      res_count++;
    }
    // 把当前阶段得到的结果/状态返回给上层调用者。
    return res_count;
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace ccann
