// ============================================================================
// 文件逻辑导读
// 这部分包含两块逻辑：load_page_layout() 重建/加载 page -> slot occupant 的内存映射；
// page_search() 则以“页”为 I/O/探索单位搜索。与普通 beam search 逐 node 读不同，
// page_search 会利用 page_layout：一旦某个 4KB 页被读入，就顺便检查该页内多个 node，
// 尽可能摊薄一次 PM/SSD 读的成本。page_visited 用来防止同一页被重复读取。
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
#include "tsl/robin_set.h"
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
// load_page_layout：建立运行时的 ID->loc 与 page->slot occupant 关系。
// 若存在 partition 文件，则按其中的分区布局恢复；否则使用初始等值映射 ID==loc，
// 并按 nnodes_per_sector 顺序填充 PageArr。page_layout 本身是 DRAM 运行时结构。
// ---------------------------------------------------------------------------
  void SSDIndex<T, TagT>::load_page_layout(const std::string &index_prefix, const _u64 nnodes_per_sector,
                                           const _u64 num_points) {
    // 定位可选的预计算 partition 文件；它记录每个 page 中实际放了哪些 logical IDs。
    std::string partition_file = index_prefix + "_partition.bin.aligned";
    // 根据当前搜索状态/编译配置决定是否执行这一分支。
    if (std::filesystem::exists(partition_file)) {
      LOG(INFO) << "Loading partition file " << partition_file;
      std::ifstream part(partition_file);
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      _u64 C, partition_nums, nd;
      part.read((char *) &C, sizeof(_u64));
      part.read((char *) &partition_nums, sizeof(_u64));
      part.read((char *) &nd, sizeof(_u64));
      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (nnodes_per_sector && num_points && (C != nnodes_per_sector)) {
        // 遇到违反索引不变量的状态时记录错误；后续通常直接终止以避免继续使用损坏状态。
        LOG(ERROR) << "partition information not correct.";
        // 当前状态无法安全恢复，主动终止而不是返回可能错误的搜索结果。
        exit(-1);
      }
      LOG(INFO) << "Partition meta: C: " << C << " partition_nums: " << partition_nums;

      // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
      uint64_t page_offset = loc_sector_no(0);
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      auto st = std::chrono::high_resolution_clock::now();

      constexpr uint64_t n_parts_per_read = 1024 * 1024;
      std::vector<unsigned> part_buf(n_parts_per_read * (1 + nnodes_per_sector));
      // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
      for (uint64_t p = 0; p < partition_nums; p += n_parts_per_read) {
        // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
        uint64_t nxt_p = std::min(p + n_parts_per_read, partition_nums);
        part.read((char *) part_buf.data(), sizeof(unsigned) * n_parts_per_read * (1 + nnodes_per_sector));
#pragma omp parallel for schedule(dynamic)
        // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
        for (uint64_t i = p; i < nxt_p; ++i) {
          // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
          uint32_t s = part_buf[(i - p) * (1 + nnodes_per_sector)];
          // PageArr 是 page 内 slot->logical ID 的运行时数组；page_search 读到一页后可据此枚举页内其他 node。
          PageArr tmp_arr;
          memcpy(tmp_arr.data(), part_buf.data() + (i - p) * (1 + nnodes_per_sector) + 1,
                 sizeof(unsigned) * nnodes_per_sector);
          // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
          for (uint32_t j = 0; j < s; ++j) {
            // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
            uint64_t loc = i * nnodes_per_sector + j;
            // 根据 partition 中的 slot occupant 重建 DRAM ID->loc 映射。
            id2loc_.insert_or_assign(tmp_arr[j], loc);
          }
          // 记录该 graph page 每个 slot 的 occupant ID，供 allocator 与 page_search 使用。
          this->page_layout.insert(page_offset + i, tmp_arr);
        }
      }
      // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
      this->cur_loc = partition_nums * nnodes_per_sector;  // aligned.

      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      auto et = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "Page layout loaded in " << std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count()
                << " ms";
    } else {
      LOG(INFO) << partition_file << " does not exist, use equal partition mapping";
// use equal mapping for id2loc and page_layout.
// 进入编译期开关 NO_MAPPING 对应的实现分支；同一算法可按硬件/实验配置切换不同路径。
#ifndef NO_MAPPING
#pragma omp parallel for
      // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
      for (size_t i = 0; i < this->num_points; ++i) {
        // 无 partition 时采用初始等值布局：logical ID i 默认存放在 physical loc i。
        id2loc_.insert_or_assign(i, i);
      }

      // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
      uint64_t page_offset = loc_sector_no(0);
      // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
      uint64_t num_sectors = (num_points + nnodes_per_sector - 1) / nnodes_per_sector;
#pragma omp parallel for
      // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
      for (size_t i = 0; i < num_sectors; ++i) {
        // PageArr 是 page 内 slot->logical ID 的运行时数组；page_search 读到一页后可据此枚举页内其他 node。
        PageArr tmp_arr;
        // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
        for (uint32_t j = 0; j < nnodes_per_sector; ++j) {
          // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
          uint64_t id = i * nnodes_per_sector + j;
          tmp_arr[j] = id < num_points ? id : kInvalidID;  // fill with kInvalidID if out of bounds.
        }
        // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
        for (uint32_t j = nnodes_per_sector; j < tmp_arr.size(); ++j) {
          tmp_arr[j] = kInvalidID;
        }
        this->page_layout.insert(i + page_offset, tmp_arr);
      }
      // 把分配前沿放到现有最后一个 loc 之后，后续再按 page 边界向上对齐。
      this->cur_loc = num_points;
      // aligned.
      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (num_points % nnodes_per_sector != 0) {
        // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
        cur_loc += nnodes_per_sector - (num_points % nnodes_per_sector);
      }
// 结束这一组编译期条件分支。
#endif
    }
    LOG(INFO) << "Cur location: " << this->cur_loc;
    LOG(INFO) << "Page layout loaded.";
  }

  // 模板参数 T 表示向量坐标类型，TagT 表示用户可见标签类型。
  template<typename T, typename TagT>
// ---------------------------------------------------------------------------
// page_search：以页为粒度的搜索实现。
// 一个候选 ID 先映射到 page，若该页未访问则读取整页；页到达后，结合 PageArr 一次处理
// 页中的多个 node。这样可以利用一次 4KB I/O 携带的其他节点，减少随机读次数。
// ---------------------------------------------------------------------------
  size_t SSDIndex<T, TagT>::page_search(const T *query1, const _u64 k_search, const _u32 mem_L, const _u64 l_search,
                                        TagT *res_tags, float *distances, const _u64 beam_width, QueryStats *stats) {
    // 从 QueryBuffer 池取得本查询专用的对齐 scratch 空间，并把 query1 拷入对齐查询缓冲。
    QueryBuffer<T> *query_buf = pop_query_buf(query1);
    // 取得当前线程对应的底层 I/O 上下文，后续 read/send/poll 都复用它。
    void *ctx = reader->get_ctx();

    // 根据当前搜索状态/编译配置决定是否执行这一分支。
    if (beam_width > MAX_N_SECTOR_READS) {
      // 遇到违反索引不变量的状态时记录错误；后续通常直接终止以避免继续使用损坏状态。
      LOG(ERROR) << "Beamwidth can not be higher than MAX_N_SECTOR_READS";
      // 当前状态无法安全恢复，主动终止而不是返回可能错误的搜索结果。
      crash();
    }
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
    // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
    _u64 &sector_scratch_idx = query_buf->sector_idx;

    // query <-> PQ chunk centers distances
    // 取得 PQ 查表距离缓冲；它保存 query 每个 PQ chunk 到各 centroid 的距离表。
    float *pq_dists = query_buf->aligned_pqtable_dist_scratch;
    // 预计算 query 到 PQ codebook centroid 的查找表，后续大量候选只需按压缩码查表求近似距离。
    pq_table.populate_chunk_distances(query, pq_dists);

    // query <-> neighbor list
    // 取得距离结果 scratch；PQ 批量距离或临时精确距离结果都复用这块对齐内存。
    float *dist_scratch = query_buf->aligned_dist_scratch;
    // 取得 PQ code 聚合 scratch，用来把离散 ID 对应的压缩码整理成连续布局后批量查表。
    _u8 *pq_coord_scratch = query_buf->aligned_pq_coord_scratch;

    Timer query_timer, io_timer, cpu_timer;
    // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
    std::vector<Neighbor> retset(4096);
    // 维护已经发现/处理过的 logical ID 集合，避免图中多条边导致重复工作。
    tsl::robin_set<_u64> &visited = *(query_buf->visited);
    // 维护已经处理过的 page ID，page_search 用它避免重复读取同一页。
    tsl::robin_set<unsigned> &page_visited = *(query_buf->page_visited);
    // 维护候选池当前有效元素个数；容器实际 capacity 可能更大。
    unsigned cur_list_size = 0;

    // 操作最终“已精确展开”集合，而不是仅有 PQ 近似距离的候选池。
    std::vector<Neighbor> full_retset;
    // 预留容量以减少搜索热路径中的动态扩容/内存搬迁。
    full_retset.reserve(4096);
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    _u32 best_medoid = 0;

    // lambda to batch compute query<-> node distances in PQ space
// ---- 构造 PQ 批量距离 helper：把 ID -> PQ code -> ADC distance 封装起来 ----
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    auto compute_pq_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids, const _u64 n_ids,
                                                               // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
                                                               float *dists_out) {
      // 按 logical ID 从内存 PQ 数据 this->data 中收集压缩码，形成连续批次，便于后续 SIMD/查表计算。
      ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks, pq_coord_scratch);
      // 使用预计算的 query->centroid 距离表，对这一批 PQ code 做 ADC 查表并输出近似距离。
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists, dists_out);
    };

// ---- 精确距离阶段：完整 coords 已到 DRAM 后，计算 query<->node exact distance ----
    // 这个 helper 负责 expanded node 的完整向量精确距离，并把结果放入 full_retset；early-exit 也基于这些精确距离判断。
    auto compute_exact_dists_and_push = [&](const char *node_buf, const unsigned id) -> float {
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      T *node_fp_coords_copy = data_buf;
      // Graph node 的首字段就是 coords，因此从 node_buf 起点复制完整向量到 DRAM scratch，再进行精确距离计算。
      memcpy(node_fp_coords_copy, node_buf, data_dim * sizeof(T));
      // 在 CPU 上对 query 与 DRAM scratch 中的完整向量做 exact distance；底层可走 AVX-512/AVX2。
      float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
      // 把“确实读过完整 vector 并计算 exact distance”的节点加入最终展开集合。
      full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
      // 把当前阶段得到的结果/状态返回给上层调用者。
      return cur_expanded_dist;
    };

// ---- 邻居展开阶段：去重邻居、批量 PQ 距离、筛选后插回 retset ----
    // nk 记录本轮新邻居插入后最靠前的候选位置，用来决定搜索指针是否需要回退。
    auto compute_and_push_nbrs = [&](const char *node_buf, unsigned &nk) {
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
      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (nbors_cand_size) {
        // 只对尚未 visited 的邻居批量算 PQ 距离，减少重复计算。
        compute_pq_dists(node_nbrs, nbors_cand_size, dist_scratch);
        // 逐个扫描当前 graph node 的邻居 logical ID。
        for (unsigned m = 0; m < nbors_cand_size; ++m) {
          const int nbor_id = node_nbrs[m];
          const float nbor_dist = dist_scratch[m];
          // 根据当前搜索状态/编译配置决定是否执行这一分支。
          if (stats != nullptr) {
            stats->n_cmps++;
          }
          // 根据当前搜索状态/编译配置决定是否执行这一分支。
          if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search))
            continue;
          // 构造候选记录：logical ID + 当前距离 + 可展开标记。
          Neighbor nn(nbor_id, nbor_dist, true);
          // Return position in sorted list where nn inserted
          // 把新候选按距离插入有序 retset，并返回插入位置；这可能让搜索指针 k 回退到更优候选。
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);  // may be overflow in retset...
          // 根据当前搜索状态/编译配置决定是否执行这一分支。
          if (cur_list_size < l_search)
            ++cur_list_size;
          // nk logs the best position in the retset that was updated due to
          // neighbors of n.
          // 根据当前搜索状态/编译配置决定是否执行这一分支。
          if (r < nk)
            // nk 记录本轮新邻居插入后最靠前的候选位置，用来决定搜索指针是否需要回退。
            nk = r;
        }
      }
    };

// ---- 初始化候选池 helper：批量算入口候选 PQ 距离并登记 visited ----
    // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
    auto compute_and_add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids) {
      compute_pq_dists(node_ids, n_ids, dist_scratch);
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

    // stats.
    // 初始化或更新本次查询的统计字段，便于拆分 I/O 与 CPU 开销。
    stats->io_us = 0;
    // 初始化或更新本次查询的统计字段，便于拆分 I/O 与 CPU 开销。
    stats->cpu_us = 0;
    // search in in-memory index.
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
      // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
      compute_and_add_to_retset(&best_medoid, 1);
    }

    // 把当前有效候选按距离排序，使 retset 前部始终代表优先扩展的最近候选。
    std::sort(retset.begin(), retset.begin() + cur_list_size);

    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    unsigned num_ios = 0;
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    unsigned k = 0;

    // cleared every iteration
    // 操作当前一轮准备读取/展开的 beam frontier。
    std::vector<unsigned> frontier;
    // 预留容量以减少搜索热路径中的动态扩容/内存搬迁。
    frontier.reserve(2 * beam_width);
    // PageArr 是 page 内 slot->logical ID 的运行时数组；page_search 读到一页后可据此枚举页内其他 node。
    using page_fnhood_t = std::tuple<unsigned, unsigned, PageArr, char *>;  // <node_id, page_id, page_layout, page_buf>
    // 操作当前一轮准备读取/展开的 beam frontier。
    std::vector<page_fnhood_t> frontier_nhoods;
    // 预留容量以减少搜索热路径中的动态扩容/内存搬迁。
    frontier_nhoods.reserve(2 * beam_width);
    // 操作当前一轮准备读取/展开的 beam frontier。
    std::vector<IORequest> frontier_read_reqs;
    // 预留容量以减少搜索热路径中的动态扩容/内存搬迁。
    frontier_read_reqs.reserve(2 * beam_width);

    // PageArr 是 page 内 slot->logical ID 的运行时数组；page_search 读到一页后可据此枚举页内其他 node。
    using io_ss_t = std::tuple<unsigned, unsigned, PageArr>;  // <node_id, page_id, page_layout>
    std::vector<io_ss_t> last_io_snapshot;
    // 预留容量以减少搜索热路径中的动态扩容/内存搬迁。
    last_io_snapshot.reserve(2 * beam_width);

    std::vector<char> last_pages(SECTOR_LEN * beam_width * 2);

    // search on disk.
// ---- 主 beam-search 循环：不断选择 beam、读 graph node、扩邻居，直到没有更优未展开候选 ----
    // 进入以“仍有未完成搜索工作”为条件的循环，直到候选或 in-flight 工作全部收敛。
    while (k < cur_list_size) {
      // 维护候选池当前有效元素个数；容器实际 capacity 可能更大。
      unsigned nk = cur_list_size;
      // clear iteration state
      // 清空上一轮 beam 的节点 ID；本轮会重新从 retset 选择尚未展开的最近候选。
      frontier.clear();
      // 清空上一轮“候选 ID/loc/读取缓冲”的关联，避免复用过期 node buffer。
      frontier_nhoods.clear();
      // 清空上一轮 I/O request 列表，为本轮 frontier 重新构造读请求。
      frontier_read_reqs.clear();
      // 从 sector scratch 的第一个槽重新放置本轮读回的 page，上一轮内容不再需要。
      sector_scratch_idx = 0;
      // find new beam
      // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
      _u32 marker = k;
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      _u32 num_seen = 0;

      // distribute cache and disk-read nodes
      // 100 us
      // 进入以“仍有未完成搜索工作”为条件的循环，直到候选或 in-flight 工作全部收敛。
      while (marker < cur_list_size && frontier.size() < beam_width && num_seen < beam_width) {
        // 把候选 ID 经 ID->loc->page 映射到其 graph page；page_search 以 page 去重而不是仅以 node 去重。
        const unsigned pid = id2page(retset[marker].id);
        // 检查这一 4KB page 是否已经被处理；同页多个候选可共享一次读取。
        if (page_visited.find(pid) == page_visited.end() && retset[marker].flag) {
          num_seen++;
          // disable nhood cache.
          // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
          frontier.push_back(retset[marker].id);
          // 维护已经处理过的 page ID，page_search 用它避免重复读取同一页。
          page_visited.insert(pid);
          // 标记该候选已经被选入本轮展开，避免下一轮再次发起相同 graph-node 读取。
          retset[marker].flag = false;
        }
        marker++;
      }

      // read nhoods of frontier ids
      std::vector<uint32_t> locked, page_locked;
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      int n_ios = 0;
      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (!frontier.empty()) {
        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if (stats != nullptr)
          stats->n_hops++;

        // 操作当前一轮准备读取/展开的 beam frontier。
        locked = this->lock_idx(idx_lock_table, kInvalidID, frontier, true);
        // 操作当前一轮准备读取/展开的 beam frontier。
        page_locked = this->lock_page_idx(page_idx_lock_table, kInvalidID, frontier, true);

        // 逐个处理本轮 frontier 中已经选出的待展开节点。
        for (_u64 i = 0; i < frontier.size(); i++) {
          // 操作当前一轮准备读取/展开的 beam frontier。
          auto id = frontier[i];
          // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
          uint64_t page_id = id2page(id);
          // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
          auto buf = sector_scratch + sector_scratch_idx * size_per_io;
          // PageArr 是 page 内 slot->logical ID 的运行时数组；page_search 读到一页后可据此枚举页内其他 node。
          PageArr layout;
          // 根据当前搜索状态/编译配置决定是否执行这一分支。
          if (unlikely(!page_layout.find(page_id, layout))) {
            // 遇到违反索引不变量的状态时记录错误；后续通常直接终止以避免继续使用损坏状态。
            LOG(ERROR) << "Page layout not found for page " << page_id;
            // 当前状态无法安全恢复，主动终止而不是返回可能错误的搜索结果。
            crash();
          }
          page_fnhood_t fnhood = std::make_tuple(id, page_id, layout, buf);
          sector_scratch_idx++;
          // 操作当前一轮准备读取/展开的 beam frontier。
          frontier_nhoods.push_back(fnhood);
          // read the page to the temporary buffer
          // 操作当前一轮准备读取/展开的 beam frontier。
          frontier_read_reqs.emplace_back(
              // 构造/保存一次底层 I/O 描述，其中包含对齐读范围、目标 buffer 以及可选 useful byte range。
              IORequest(page_id * SECTOR_LEN, size_per_io, buf, page_id * SECTOR_LEN, size_per_io));
          // 根据当前搜索状态/编译配置决定是否执行这一分支。
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
          num_ios++;
        }

        // 操作当前一轮准备读取/展开的 beam frontier。
        n_ios = reader->send_read_no_alloc(frontier_read_reqs, ctx);
      }

      // compute remaining nodes in the pages that are fetched in the previous
      // round
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      auto cpu1_st = std::chrono::high_resolution_clock::now();
      // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
      for (size_t i = 0; i < last_io_snapshot.size(); ++i) {
        // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
        auto &[last_io_id, pid, page_layout] = last_io_snapshot[i];
        // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
        char *sector_buf = last_pages.data() + i * SECTOR_LEN;

        // minus one for the vector that is computed previously
        std::vector<std::pair<float, const char *>> vis_cand;
        // 预留容量以减少搜索热路径中的动态扩容/内存搬迁。
        vis_cand.reserve(nnodes_per_sector);

        // compute exact distances of the vectors within the page
        // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
        for (unsigned j = 0; j < nnodes_per_sector; ++j) {
          const unsigned id = page_layout[j];
          // 根据当前搜索状态/编译配置决定是否执行这一分支。
          if (id == last_io_id || id == kAllocatedID || id == kInvalidID) {
            continue;
          }
          // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
          const char *node_buf = sector_buf + j * max_node_len;
          // 这个 helper 负责 expanded node 的完整向量精确距离，并把结果放入 full_retset；early-exit 也基于这些精确距离判断。
          float dist = compute_exact_dists_and_push(node_buf, id);
          vis_cand.emplace_back(dist, node_buf);
        }
        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if (vis_cand.size() > 0) {
          // 按 Neighbor 的距离顺序排序，保证最近候选位于容器前部。
          std::sort(vis_cand.begin(), vis_cand.end());
        }

        // compute PQ distances for neighbours of the vectors in the page
        // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
        for (unsigned j = 0; j < vis_cand.size(); ++j) {
          compute_and_push_nbrs(vis_cand[j].second, nk);
        }
      }
      // 清除上一轮的逻辑内容但尽量保留已分配容量，供下一轮复用。
      last_io_snapshot.clear();
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      auto cpu1_ed = std::chrono::high_resolution_clock::now();
      // 初始化或更新本次查询的统计字段，便于拆分 I/O 与 CPU 开销。
      stats->cpu_us1 += std::chrono::duration_cast<std::chrono::microseconds>(cpu1_ed - cpu1_st).count();

      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      auto io_time_st = std::chrono::high_resolution_clock::now();
      // get last submitted io results, blocking
      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (!frontier.empty()) {
        // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
        for (int i = 0; i < n_ios; ++i) {
          // 阻塞等待至少一个 I/O 完成；page_search 在真正需要某批 page 时才进入等待。
          reader->poll_wait(ctx);
        }
        this->unlock_page_idx(page_idx_lock_table, page_locked);
        this->unlock_idx(idx_lock_table, locked);
      }
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      auto io_time_ed = std::chrono::high_resolution_clock::now();
      // 初始化或更新本次查询的统计字段，便于拆分 I/O 与 CPU 开销。
      stats->io_us += std::chrono::duration_cast<std::chrono::microseconds>(io_time_ed - io_time_st).count();

      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      auto cpu_st = std::chrono::high_resolution_clock::now();
      // compute only the desired vectors in the pages - one for each page
      // postpone remaining vectors to the next round
      // 逐个处理本轮 frontier 中已经选出的待展开节点。
      for (auto &[id, pid, layout, sector_buf] : frontier_nhoods) {
        // fill in the last_io_ids() and last_pages() with neighbor buffers.
        memcpy(last_pages.data() + last_io_snapshot.size() * SECTOR_LEN, sector_buf, SECTOR_LEN);
        last_io_snapshot.emplace_back(std::make_tuple(id, pid, layout));

        // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
        for (unsigned j = 0; j < nnodes_per_sector; ++j) {
          // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
          unsigned cur_id = layout[j];
          // 根据当前搜索状态/编译配置决定是否执行这一分支。
          if (cur_id == id) {
            // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
            char *node_buf = sector_buf + j * max_node_len;
            // 这个 helper 负责 expanded node 的完整向量精确距离，并把结果放入 full_retset；early-exit 也基于这些精确距离判断。
            compute_exact_dists_and_push(node_buf, id);
            compute_and_push_nbrs(node_buf, nk);
          }
        }
      }
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      auto cpu_ed = std::chrono::high_resolution_clock::now();
      // 初始化或更新本次查询的统计字段，便于拆分 I/O 与 CPU 开销。
      stats->cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(cpu_ed - cpu_st).count();

      // update best inserted position
      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (nk <= k)
        // nk 记录本轮新邻居插入后最靠前的候选位置，用来决定搜索指针是否需要回退。
        k = nk;  // k is the best position in retset updated in this round.
      // 进入前述条件不成立时的备用路径。
      else
        ++k;
    }

// ---- 搜索结束：按 exact distance 重排真正展开过的节点，然后生成最终 top-k ----
    // 按 Neighbor 的距离顺序排序，保证最近候选位于容器前部。
    std::sort(full_retset.begin(), full_retset.end(),
              // 把当前阶段得到的结果/状态返回给上层调用者。
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    // copy k_search values
    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    _u64 t = 0;
    // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
    for (_u64 i = 0; i < full_retset.size() && t < k_search; i++) {
      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
        continue;
      }
      // 操作最终“已精确展开”集合，而不是仅有 PQ 近似距离的候选池。
      res_tags[t] = id2tag(full_retset[i].id);
      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (distances != nullptr) {
        // 操作最终“已精确展开”集合，而不是仅有 PQ 近似距离的候选池。
        distances[t] = full_retset[i].distance;
      }
      t++;
    }

    push_query_buf(query_buf);

    // 根据当前搜索状态/编译配置决定是否执行这一分支。
    if (stats != nullptr) {
      // 初始化或更新本次查询的统计字段，便于拆分 I/O 与 CPU 开销。
      stats->total_us = (double) query_timer.elapsed();
    }
    // 把当前阶段得到的结果/状态返回给上层调用者。
    return t;
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace ccann
