// ============================================================================
// 文件逻辑导读
// 这部分实现多查询的 coroutine-style / cooperative search。
// 每个 CoroDataOne 保存“一条查询”的完整搜索状态和 scratch buffer；同一线程上最多
// 维护 kMaxCoroPerThread 条查询。代码先为每条查询发出一批异步 I/O，然后轮询完成状态，
// 哪条查询的 I/O 先完成就先展开哪条，从而在单线程内把多条查询的 I/O 等待互相遮蔽。
// 注意这里的“协程”并不是 C++20 coroutine 语法，而是显式保存状态 + 手动调度。
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
// coro_search：同一线程并发推进 N 条查询。
// 每条查询拥有独立 CoroDataOne 状态；外层调度循环不断 poll I/O，完成则 explore_frontier，
// 随后立刻 issue_next_io_batch。这样一条查询等待设备时，线程可以服务其他查询。
// ---------------------------------------------------------------------------
  size_t SSDIndex<T, TagT>::coro_search(T **queries, const _u64 k_search, const _u32 mem_L, const _u64 l_search,
                                        TagT **res_tags, float **res_dists, const _u64 beam_width, int N) {
    // beam search with intra-thread parallelism.
    // 限制单线程同时保留的查询状态数，防止每条查询的大 scratch buffer 让线程本地内存无限膨胀。
    static constexpr int kMaxCoroPerThread = 8;
    static constexpr int kMaxVectorDim = 512;
    // 一个 CoroDataOne 就是一条查询的“手动协程帧”：buffer、候选池、visited、当前 k 等状态全部保存在这里。
    struct alignas(SECTOR_LEN) CoroDataOne {
      // buffer.
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      char sectors[SECTOR_LEN * 128];
      T query[kMaxVectorDim];
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      _u8 pq_coord_scratch[32768 * 32];
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      float pq_dists[32768];
      T data_buf[ROUND_UP(1024 * kMaxVectorDim, 256)];
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      float dist_scratch[512];
      // 推进完整向量 scratch 的槽位索引，保证不同 expanded node 的 coords 不互相覆盖。
      _u64 data_buf_idx;
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      _u64 sector_idx;

      // search state.
      // 操作最终“已精确展开”集合，而不是仅有 PQ 近似距离的候选池。
      std::vector<Neighbor> full_retset;
      // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
      std::vector<Neighbor> retset;
      // 维护已经发现/处理过的 logical ID 集合，避免图中多条边导致重复工作。
      tsl::robin_set<_u64> visited;

      // 操作当前一轮准备读取/展开的 beam frontier。
      std::vector<unsigned> frontier;
      using fnhood_t = std::tuple<unsigned, unsigned, char *>;
      // 操作当前一轮准备读取/展开的 beam frontier。
      std::vector<fnhood_t> frontier_nhoods;
      // 操作当前一轮准备读取/展开的 beam frontier。
      std::vector<IORequest> frontier_read_reqs;

      SSDIndex<T> *parent;
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      unsigned cur_list_size, cmps, k;

      void compute_dists(const unsigned *ids, const _u64 n_ids, float *dists_out) {
        // 这里的 n_chunks 是每个向量的 PQ 压缩码长度/分块数，用于定位压缩向量。
        ::aggregate_coords(ids, n_ids, parent->data.data(), parent->n_chunks, pq_coord_scratch);
        // 这里的 n_chunks 是每个向量的 PQ 压缩码长度/分块数，用于定位压缩向量。
        ::pq_dist_lookup(pq_coord_scratch, n_ids, parent->n_chunks, pq_dists, dists_out);
      };

      void print() {
        // 操作最终“已精确展开”集合，而不是仅有 PQ 近似距离的候选池。
        LOG(INFO) << "Full retset size " << full_retset.size() << " retset size: " << retset.size()
                  // 维护已经发现/处理过的 logical ID 集合，避免图中多条边导致重复工作。
                  << " visited size: " << visited.size() << " frontier size: " << frontier.size()
                  // 操作当前一轮准备读取/展开的 beam frontier。
                  << " frontier nhood size: " << frontier_nhoods.size()
                  // 操作当前一轮准备读取/展开的 beam frontier。
                  << " frontier read reqs size: " << frontier_read_reqs.size();
      }

      void reset() {
        // 推进完整向量 scratch 的槽位索引，保证不同 expanded node 的 coords 不互相覆盖。
        data_buf_idx = 0;
        // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
        sector_idx = 0;
        // 清除上一轮的逻辑内容但尽量保留已分配容量，供下一轮复用。
        visited.clear();  // does not deallocate memory.
        // 调整容器有效容量，确保后续按索引写入不会越界。
        retset.resize(4096);
        // 清除上一轮的逻辑内容但尽量保留已分配容量，供下一轮复用。
        retset.clear();
        // 清除上一轮的逻辑内容但尽量保留已分配容量，供下一轮复用。
        full_retset.clear();
        // 维护候选池当前有效元素个数；容器实际 capacity 可能更大。
        cur_list_size = cmps = k = 0;
      }

      // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
      void compute_and_add_to_retset(const unsigned *node_ids, const _u64 n_ids) {
        compute_dists(node_ids, n_ids, dist_scratch);
        // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
        for (_u64 i = 0; i < n_ids; ++i) {
          // 维护候选池当前有效元素个数；容器实际 capacity 可能更大。
          auto &item = retset[cur_list_size];
          item.id = node_ids[i];
          // 保存当前候选的距离值，后续候选排序和剪枝都以它为依据。
          item.distance = dist_scratch[i];
          item.flag = true;
          cur_list_size++;
          // 把入口候选标记为已发现，防止后续从不同边再次重复加入候选池。
          visited.insert(node_ids[i]);
        }
      };

// 这一成员函数把“选择候选”和“提交 I/O”封装成一次可暂停的协程步骤。
      // 为当前查询从 retset 选下一批 beam，并异步提交 graph page 读取；调用后立即返回给外层调度器。
      void issue_next_io_batch(const _u64 beam_width, void *ctx) {
        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if (search_ends()) {
          return;
        }
        // clear iteration state
        // 清空上一轮 beam 的节点 ID；本轮会重新从 retset 选择尚未展开的最近候选。
        frontier.clear();
        // 清空上一轮“候选 ID/loc/读取缓冲”的关联，避免复用过期 node buffer。
        frontier_nhoods.clear();
        // 清空上一轮 I/O request 列表，为本轮 frontier 重新构造读请求。
        frontier_read_reqs.clear();
        // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
        sector_idx = 0;

        // marker 指向候选池中当前扫描位置，用它寻找下一批尚未展开的最近节点。
        _u32 marker = k;
        // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
        _u32 num_seen = 0;
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
          // 逐个处理本轮 frontier 中已经选出的待展开节点。
          for (_u64 i = 0; i < frontier.size(); i++) {
            // 操作当前一轮准备读取/展开的 beam frontier。
            uint32_t loc = frontier[i];
            // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
            uint64_t offset = parent->loc_sector_no(loc) * SECTOR_LEN;
            // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
            auto sector_buf = sectors + sector_idx * parent->size_per_io;
            // 计算或保存 graph page/sector 相关位置，用于把 logical loc 翻译成实际读取地址。
            fnhood_t fnhood = std::make_tuple(loc, loc, sector_buf);
            sector_idx++;
            // 操作当前一轮准备读取/展开的 beam frontier。
            frontier_nhoods.push_back(fnhood);
            // 操作当前一轮准备读取/展开的 beam frontier。
            frontier_read_reqs.emplace_back(IORequest(offset, parent->size_per_io, sector_buf, 0, 0));
          }
          // 操作当前一轮准备读取/展开的 beam frontier。
          parent->reader->send_io(frontier_read_reqs, ctx, false);
        }
      }

// 这一成员函数只检查 I/O 完成度，不推进搜索状态，因此适合被外层轮询调度。
      // 轮询本查询上一批 I/O 是否全部完成；未完成就让出执行机会给其他查询。
      bool io_finished(void *ctx) {
        // 推进/收割一次底层异步 I/O 完成事件，用于 cooperative 多查询调度。
        parent->reader->poll(ctx);
        // 逐个处理本轮 frontier 中已经选出的待展开节点。
        for (auto &req : frontier_read_reqs) {
          // 根据当前搜索状态/编译配置决定是否执行这一分支。
          if (!req.finished) {
            // 把当前阶段得到的结果/状态返回给上层调用者。
            return false;
          }
        }
        // 把当前阶段得到的结果/状态返回给上层调用者。
        return true;
      }

// 这一成员函数只消费已经完成的 I/O，不负责等待设备。
      // 消费已经读回的 graph nodes：算 exact distance、扩展邻居 PQ distance，并更新本查询 retset。
      void explore_frontier(uint64_t l_search) {
        // 维护候选池当前有效元素个数；容器实际 capacity 可能更大。
        auto nk = cur_list_size;

        // 逐个处理本轮 frontier 中已经选出的待展开节点。
        for (auto &frontier_nhood : frontier_nhoods) {
          // 操作当前一轮准备读取/展开的 beam frontier。
          auto [id, loc, sector_buf] = frontier_nhood;
          // 根据 loc 在已经读入的 sector buffer 中定位目标 graph-node slot。
          char *node_disk_buf = parent->offset_to_loc(sector_buf, loc);
          // 跳过 coords 区域，定位 graph node 的邻接表头；第一个 uint32 是 nnbrs，后面才是 neighbor IDs。
          unsigned *node_buf = parent->offset_to_node_nhood(node_disk_buf);
          // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
          _u64 nnbrs = (_u64) (*node_buf);
          // 取得 graph node 开头的完整向量 coords；这是 exact distance 使用的未压缩向量。
          T *node_fp_coords = parent->offset_to_node_coords(node_disk_buf);

          // 推进完整向量 scratch 的槽位索引，保证不同 expanded node 的 coords 不互相覆盖。
          T *node_fp_coords_copy = data_buf + (data_buf_idx * parent->aligned_dim);
          // 推进完整向量 scratch 的槽位索引，保证不同 expanded node 的 coords 不互相覆盖。
          data_buf_idx++;
          memcpy(node_fp_coords_copy, node_fp_coords, parent->data_dim * sizeof(T));
          // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
          float cur_expanded_dist =
              parent->dist_cmp->compare(query, node_fp_coords_copy, (unsigned) parent->aligned_dim);

          // 构造候选记录：logical ID + 当前距离 + 可展开标记。
          Neighbor n(id, cur_expanded_dist, true);
          // 操作最终“已精确展开”集合，而不是仅有 PQ 近似距离的候选池。
          full_retset.push_back(n);

          // 跳过邻接表头的 nnbrs 字段，node_nbrs 现在指向连续的 neighbor logical IDs。
          unsigned *node_nbrs = (node_buf + 1);
          // compute node_nbrs <-> query dist in PQ space
          // 对当前节点的所有邻居批量计算 PQ 近似距离；这里不读取每个邻居的完整 coords。
          compute_dists(node_nbrs, nnbrs, dist_scratch);

          // process prefetch-ed nhood
          // 逐个扫描当前 graph node 的邻居 logical ID。
          for (_u64 m = 0; m < nnbrs; ++m) {
            // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
            unsigned id = node_nbrs[m];
            // 根据当前搜索状态/编译配置决定是否执行这一分支。
            if (visited.find(id) != visited.end()) {
              continue;
            } else {
              // 立即记录该 logical ID 已发现；即使它稍后因距离太差被丢弃，也避免重复 PQ 计算。
              visited.insert(id);
              cmps++;
              // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
              float dist = dist_scratch[m];
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
              }

              // 根据当前搜索状态/编译配置决定是否执行这一分支。
              if (r < nk)
                // nk 记录本轮新邻居插入后最靠前的候选位置，用来决定搜索指针是否需要回退。
                nk = r;
            }
          }
        }

        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if (nk <= k)
          // nk 记录本轮新邻居插入后最靠前的候选位置，用来决定搜索指针是否需要回退。
          k = nk;  // k is the best position in retset updated in this round.
        // 进入前述条件不成立时的备用路径。
        else
          ++k;
      }

      // 判断这一条查询是否已没有尚待展开的候选；外层只有所有查询都结束才退出。
      bool search_ends() {
        // this->print();
        // 把当前阶段得到的结果/状态返回给上层调用者。
        return k >= cur_list_size;
      }
    };

    struct alignas(4096) CoroData {
      CoroDataOne data[kMaxCoroPerThread];
      CoroData(SSDIndex<T> *parent) {
        // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
        for (int i = 0; i < kMaxCoroPerThread; ++i) {
          data[i].parent = parent;
        }
      }
    };

    // 每个 OS 线程持有一份可复用的多查询状态块，避免每批请求反复构造大数组。
    static __thread CoroData *data;
    // 根据当前搜索状态/编译配置决定是否执行这一分支。
    if (unlikely(data == nullptr)) {
      data = new CoroData(this);
    }

    // 根据当前搜索状态/编译配置决定是否执行这一分支。
    if (unlikely(N > kMaxCoroPerThread)) {
      // 遇到违反索引不变量的状态时记录错误；后续通常直接终止以避免继续使用损坏状态。
      LOG(ERROR) << "N > kMaxCoroPerThread";
      // 当前状态无法安全恢复，主动终止而不是返回可能错误的搜索结果。
      exit(-1);
    }
    // 根据当前搜索状态/编译配置决定是否执行这一分支。
    if (unlikely(data_is_normalized)) {
      LOG(INFO) << "Unsupported yet";
      // 当前状态无法安全恢复，主动终止而不是返回可能错误的搜索结果。
      exit(-1);
    }

    // do not use the thread data's buf.
    // 仅借用线程级 QueryBuffer/IO 上下文；每条并发查询自己的搜索状态放在 CoroDataOne 中。
    QueryBuffer<T> *thread_data = pop_query_buf(queries[0]);
    // 取得当前线程对应的底层 I/O 上下文，后续 read/send/poll 都复用它。
    void *ctx = reader->get_ctx();
    // lambda to batch compute query<-> node distances in PQ space

// ---- 为每条并发查询分别初始化 query/PQ table/候选池状态 ----
    // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
    for (int v = 0; v < N; ++v) {
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      auto &coro_data = data->data[v];
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      auto &query1 = queries[v];
      memcpy(coro_data.query, query1, this->data_dim * sizeof(T));

      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      auto &query = coro_data.query;

      // pointers to buffers for data
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      T *data_buf = coro_data.data_buf;
      // 提前把 DRAM coordinate scratch 拉近 CPU cache，降低随后 exact-distance 写入/读取的冷启动开销。
      _mm_prefetch((char *) data_buf, _MM_HINT_T1);

      // query <-> PQ chunk centers distances
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      float *pq_dists = coro_data.pq_dists;
      // 预计算 query 到 PQ codebook centroid 的查找表，后续大量候选只需按压缩码查表求近似距离。
      pq_table.populate_chunk_distances(query, pq_dists);

      coro_data.reset();

      // 选择持久化图的 medoid 作为默认入口点；若有小型内存索引则会用更好的入口候选替代。
      _u32 best_medoid = medoids[0];

// ---- 选择搜索入口：优先使用小型 DRAM index；否则从 graph medoid 开始 ----
      // 根据当前搜索状态/编译配置决定是否执行这一分支。
      if (mem_L) {
        std::vector<unsigned> mem_tags(mem_L);
        std::vector<float> mem_dists(mem_L);
        // 先在小型 DRAM 索引中搜索入口候选，减少从单一 medoid 开始时的图遍历距离。
        mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data());
        // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
        coro_data.compute_and_add_to_retset(mem_tags.data(), std::min((unsigned) mem_L, (unsigned) l_search));
      } else {
        // Do not use optimized start point.
        // 操作有序候选池 retset；它决定后续优先扩展哪些 logical IDs。
        coro_data.compute_and_add_to_retset(&best_medoid, 1);
      }
      // 按 Neighbor 的距离顺序排序，保证最近候选位于容器前部。
      std::sort(coro_data.retset.begin(), coro_data.retset.begin() + coro_data.cur_list_size);
    }

    // SEARCH!
    // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
    for (int i = 0; i < N; ++i) {
      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      auto &coro_data = data->data[i];
      // 消费后立刻补发下一批 I/O，保持该查询的流水不断。
      coro_data.issue_next_io_batch(beam_width, ctx);
    }

    // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
    bool all_finished = false;
// ---- cooperative 调度循环：依次检查 N 条查询，让已经完成 I/O 的查询先向前推进 ----
    // 进入以“仍有未完成搜索工作”为条件的循环，直到候选或 in-flight 工作全部收敛。
    while (!all_finished) {
      // 每轮先乐观假定所有查询都完成；只要发现一条仍在推进就重新置 false。
      all_finished = true;
      // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
      for (int i = 0; i < N; ++i) {
        // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
        auto &coro_data = data->data[i];
        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if (!coro_data.search_ends()) {
          all_finished = false;
          // 如果这条查询的 I/O 还没完成，立即跳到下一条查询，从而用其他查询遮蔽等待时间。
          if (!coro_data.io_finished(ctx)) {
            continue;
          }
          // LOG(INFO) << "Full retset size: " << coro_data.full_retset.size();
          // 当前查询 I/O 已到达，消费这批 graph nodes 并更新其候选集合。
          coro_data.explore_frontier(l_search);
          // 消费后立刻补发下一批 I/O，保持该查询的流水不断。
          coro_data.issue_next_io_batch(beam_width, ctx);
        }
      }
    }

// ---- 为每条并发查询分别初始化 query/PQ table/候选池状态 ----
    // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
    for (int v = 0; v < N; ++v) {
      // re-sort by distance
      // 操作最终“已精确展开”集合，而不是仅有 PQ 近似距离的候选池。
      auto &full_retset = data->data[v].full_retset;
// ---- 搜索结束：按 exact distance 重排真正展开过的节点，然后生成最终 top-k ----
      // 按 Neighbor 的距离顺序排序，保证最近候选位于容器前部。
      std::sort(full_retset.begin(), full_retset.end(),
                // 把当前阶段得到的结果/状态返回给上层调用者。
                [](const Neighbor &left, const Neighbor &right) { return left < right; });

      // 初始化这一阶段需要的局部状态；其生命周期限定在当前查询/当前循环中。
      _u64 t = 0;
      // 遍历当前批次/容器中的元素，并在同一逻辑阶段完成处理。
      for (_u64 i = 0; i < full_retset.size() && t < k_search; i++) {
        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
          continue;  // deduplicate.
        }
        // 操作最终“已精确展开”集合，而不是仅有 PQ 近似距离的候选池。
        res_tags[v][t] = full_retset[i].id;  // use ID to replace tags
        // 根据当前搜索状态/编译配置决定是否执行这一分支。
        if (res_dists[v] != nullptr) {
          // 操作最终“已精确展开”集合，而不是仅有 PQ 近似距离的候选池。
          res_dists[v][t] = full_retset[i].distance;
        }
        t++;
      }
    }

    this->push_query_buf(thread_data);
    // 把当前阶段得到的结果/状态返回给上层调用者。
    return 0;
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace ccann
