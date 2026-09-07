// 引入 neighbor.h：Neighbor/NeighborTag 等搜索候选结构。
#include "neighbor.h"
// 引入 timer.h：项目内性能计时宏和 Timer。
#include "timer.h"
// 引入 tsl/robin_set.h：高性能 hash set，用于 visited/deletion set。
#include "tsl/robin_set.h"
// 引入 utils.h：项目通用辅助函数、常量和二进制 I/O 工具。
#include "utils.h"
// 引入 v2/dynamic_index.h：DynamicSSDIndex 对外动态索引接口声明。
#include "v2/dynamic_index.h"
// 引入 csignal：信号处理接口。
#include <csignal>
// 引入 cstdint：固定宽度整数类型。
#include <cstdint>
// 引入 mutex：互斥锁/lock_guard。
#include <mutex>
// 引入 vector：动态数组容器。
#include <vector>

// 引入 algorithm：排序、min/max 等通用算法。
#include <algorithm>
// 引入 filesystem：文件存在性、复制、大小等文件系统操作。
#include <filesystem>
// 引入 cassert：assert 断言。
#include <cassert>
// 引入 cmath：浮点数学函数。
#include <cmath>
// 引入 cstdio：C 标准 I/O。
#include <cstdio>
// 引入 ctime：时间相关 C 接口。
#include <ctime>
// 引入 omp.h：OpenMP 并行循环与线程编号。
#include <omp.h>
// 引入 shared_mutex：读写锁，用于 search/delete/merge 并发控制。
#include <shared_mutex>
// 引入 string：字符串类型。
#include <string>
// 引入 sys/mman.h：mmap/munmap 内存映射接口。
#include <sys/mman.h>
// 引入 libpmem.h：PMDK libpmem：pmem_memcpy、persist 等 PM 原语。
#include <libpmem.h>

// 引入 fcntl.h：open flags，例如 O_DIRECT/O_CREAT。
#include <fcntl.h>
// 引入 sys/stat.h：文件状态/stat 接口。
#include <sys/stat.h>
// 引入 time.h：POSIX 时间接口。
#include <time.h>

// 引入 aux_utils.h：项目辅助工具函数。
#include "aux_utils.h"
// 引入 ssd_index.h：SSDIndex 的核心类声明与索引布局辅助函数。
#include "ssd_index.h"
// 引入 parameters.h：索引构建/搜索参数容器。
#include "parameters.h"

// 引入 linux_aligned_file_reader.h：Linux 下 O_DIRECT、io_uring 与 DAX mmap 的具体 reader。
#include "linux_aligned_file_reader.h"

// 进入 ccann 命名空间，下面实现的函数都属于 CCANN 核心索引模块。
namespace ccann {
// 并行文件复制时每个逻辑块的目标粒度设为 64 MiB。
#define BLOCK_SIZE (64 * 1024 * 1024)  // 64 MiB
// 限制并行文件复制最多使用 32 个线程，避免为了复制文件无限扩张线程数。
#define MAX_THREADS 32

  // ============================================================================
  // copy_block
  // 按指定文件区间做分块复制。每个线程只负责 [offset, offset+size) 的一段，通过 pread/pwrite 保持显式偏移，并累加全局 progress。
  // ============================================================================
  void copy_block(const std::string &src, const std::string &dst, size_t offset, size_t size,
                  std::atomic<size_t> &progress) {
    // 以只读方式打开源文件；每个 worker 独立持有 fd，避免共享 seek position。
    int fd_src = open(src.c_str(), O_RDONLY);
    // 打开/创建目标文件；pwrite 使用显式 offset，因此多个线程可以安全写不同区间。
    int fd_dst = open(dst.c_str(), O_WRONLY | O_CREAT, 0666);
    if (fd_src < 0 || fd_dst < 0)
      // 当前函数工作已经完成，直接返回调用方。
      return;

    // 给当前复制线程分配固定大小临时 buffer；注释写 1 MiB，但实际表达式是 16 MiB。
    std::vector<char> buffer(16 * 1024 * 1024);  // 1 MiB buffer
    size_t copied = 0;
    // 循环处理该线程负责的文件区间，直到本段全部复制完成。
    while (copied < size) {
      size_t to_read = std::min(buffer.size(), size - copied);
      // 从源文件的绝对 offset 读取，不改变共享文件位置语义。
      ssize_t r = pread(fd_src, buffer.data(), to_read, offset + copied);
      if (r <= 0)
        break;
      // 把刚读取的数据写到目标文件相同绝对 offset，线程之间无需串行 seek。
      pwrite(fd_dst, buffer.data(), r, offset + copied);
      copied += r;
      // 原子累加全局已复制字节数，供进度线程计算百分比。
      progress += r;
    }

    close(fd_src);
    close(fd_dst);
  }


  // ============================================================================
  // parallel_copy_file
  // 并行复制大索引文件：小文件直接 copy_file；大文件按线程数切成不重叠区间，由 copy_block 并行复制，同时另起一个进度线程周期性打印完成比例。
  // ============================================================================
  void parallel_copy_file(const std::string &src, const std::string &dst) {
    auto time_start = std::chrono::high_resolution_clock::now();
    // 先删除目标文件（如果存在）
    // 先删除可能存在的旧目标，避免旧文件尾部或旧 metadata 残留。
    if (std::filesystem::exists(dst)) {
      std::filesystem::remove(dst);
    }

    // 取得源文件总长度，用于决定串行还是并行复制以及切分区间。
    size_t filesize = std::filesystem::file_size(src);

    // 小于等于 64 MiB 时线程切分收益有限，直接调用标准库复制。
    if (filesize <= BLOCK_SIZE) {
      std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);
      std::cout << src << " copied (small file)\n";
      // 当前函数工作已经完成，直接返回调用方。
      return;
    }

    std::atomic<size_t> progress(0);
    // 线程数取“最多 32”和“按 64 MiB 至少需要的块数”中的较小值。
    size_t num_threads = std::min(static_cast<size_t>(MAX_THREADS), (filesize + BLOCK_SIZE - 1) / BLOCK_SIZE);
    std::vector<std::thread> threads;

    for (size_t i = 0; i < num_threads; ++i) {
      size_t offset = i * filesize / num_threads;
      size_t end = (i + 1) * filesize / num_threads;
      size_t sz = end - offset;
      // 启动 worker 复制自己负责的连续区间；各区间按文件比例切分且互不重叠。
      threads.emplace_back(copy_block, src, dst, offset, sz, std::ref(progress));
    }

    // 打印进度线程
    // 独立进度线程只读原子 progress，不参与实际 I/O。
    std::thread progress_thread([&]() {
      while (progress < filesize) {
        {
          double pct = 100.0 * progress / filesize;
          std::cout << "\rCopying " << src << ": " << pct << "% " << std::flush;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    });

    for (auto &t : threads)
      // 等待所有复制 worker 完成，保证目标文件数据已全部写入。
      t.join();
    // 独立进度线程只读原子 progress，不参与实际 I/O。
    progress_thread.join();

    {
      std::cout << "\rCopying " << src << ": 100% done." << std::endl;
    }
    auto time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = time_end - time_start;
    std::cout << "Total copy time: " << diff.count() << " seconds.\n";
  }


  // ============================================================================
  // copy_index
  // 复制一个 CCANN 索引前缀对应的相关文件。graph、PQ pivots、PQ compressed、partition、id2loc 走统一的
  // copy_if_exists；tags 单独处理，以便源不存在时删除目标侧旧 tags，避免残留。
  // ============================================================================
  void copy_index(const std::string &prefix_in, const std::string &prefix_out) {
    // 定义统一辅助函数：源组件存在才复制，允许某些可选索引文件缺失。
    auto copy_if_exists = [&](const std::string &src_file, const std::string &dst_file) {
      if (std::filesystem::exists(src_file)) {
        parallel_copy_file(src_file, dst_file);
      }
    };

    std::cout << "Copying disk index from " << prefix_in << " to " << prefix_out << "\n";

    // 复制核心 graph 文件（super block + graph data）。
    copy_if_exists(prefix_in + "_disk.index", prefix_out + "_disk.index");

    if (std::filesystem::exists(prefix_in + "_disk.index.tags")) {
      // tags 文件较小，直接用标准 copy_file 完整覆盖。
      std::filesystem::copy_file(prefix_in + "_disk.index.tags", prefix_out + "_disk.index.tags",
                                 std::filesystem::copy_options::overwrite_existing);
    } else if (std::filesystem::exists(prefix_out + "_disk.index.tags")) {
      // 若源索引没有 tags，则删除目标残留 tags，防止 shadow 索引混入旧标签。
      std::filesystem::remove(prefix_out + "_disk.index.tags");
    }

    // 复制 PQ codebook/pivots。
    copy_if_exists(prefix_in + "_pq_pivots.bin", prefix_out + "_pq_pivots.bin");
    // 复制按 ID 排列的 PQ compressed vectors。
    copy_if_exists(prefix_in + "_pq_compressed.bin", prefix_out + "_pq_compressed.bin");
    copy_if_exists(prefix_in + "_partition.bin.aligned", prefix_out + "_partition.bin.aligned");
    // 复制持久化 Location Table。
    copy_if_exists(prefix_in + "_disk.index.id2loc", prefix_out + "_disk.index.id2loc");
  }


  // ============================================================================
  // build_id2loc_mapping
  // 为旧/初始索引补建 ID→loc 文件。初始静态布局中采用 identity mapping，即 id2loc[id] = id；按 4 KiB sector 批量生成
  // uint32_t 数组并顺序写入 .id2loc。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::build_id2loc_mapping(std::string &index_file, std::string &id2loc_file) {
    // 打开 _disk.index 的 super block，先读取节点数量以确定 location table 长度。
    std::ifstream index_metadata(index_file, std::ios::binary);
    _u32 nr, nc;
    _u64 npts;
    // 读取 save_bin/header 中的 32 位字段；这里保留与 loader 相同的元数据解析顺序。
    READ_U32(index_metadata, nr);
    // 读取 save_bin/header 中的 32 位字段；这里保留与 loader 相同的元数据解析顺序。
    READ_U32(index_metadata, nc);
    // 读取实际 graph node 数量 npts，用它生成 id2loc[0..npts-1]。
    READ_U64(index_metadata, npts);
    index_metadata.close();

    // 创建二进制 .id2loc 文件。
    std::ofstream id2loc_writer(id2loc_file, std::ios::binary);
    // 输出运行状态/统计信息，便于观察实验流程和后台线程进度。
    LOG(INFO) << "Build ID to Location mapping with " << npts << " points.";
    // 把 npts 个 uint32_t loc 向上取整成 4 KiB sectors，保证文件按 sector 写。
    auto sector_num = DIV_ROUND_UP(npts * sizeof(uint32_t), SECTOR_LEN);
    std::vector<IORequest> writes;

    // 一个 4 KiB 栈上数组，恰好容纳 SECTOR_LEN/4 个 location entries。
    uint32_t sector_buf[SECTOR_LEN / sizeof(uint32_t)];

    for (uint64_t s = 0; s < sector_num; ++s) {
      // 每个 sector 先清零，尾部不足一个完整 sector 的未使用 entries 保持 0。
      memset((void *) sector_buf, 0, SECTOR_LEN);
      for (uint64_t i = 0; i < SECTOR_LEN / sizeof(uint32_t); ++i) {
        // 把 sector 内下标换算成全局 vector ID。
        uint64_t global_idx = s * (SECTOR_LEN / sizeof(uint32_t)) + i;
        if (global_idx < npts) {
          // 初始静态布局采用 identity mapping：ID i 的物理 loc 也是 i。
          sector_buf[i] = global_idx;
        } else {
          break;
        }
      }
      IORequest req;
      req.offset = s * SECTOR_LEN;
      req.len = SECTOR_LEN;
      req.buf = (void *) sector_buf;
      writes.push_back(req);
      for (auto &req : writes) {
        // 顺序写出完整 4 KiB location-table sector。
        id2loc_writer.write((char *) req.buf, req.len);
      }
      writes.clear();
    }

    id2loc_writer.flush();

    // 输出运行状态/统计信息，便于观察实验流程和后台线程进度。
    LOG(INFO) << "Built ID to Location mapping.";
    id2loc_writer.close();
  }


  // ============================================================================
  // DynamicSSDIndex_ctor
  // DynamicSSDIndex 构造流程：校验索引文件 → 读取参数并创建各类 reader/writer → 构造底层 SSDIndex → 必要时补建 id2loc →
  // 非只读模式下复制 shadow 索引避免污染原文件 → 设置 search mode → load 底层索引 → 可选加载小型内存索引。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<typename T, typename TagT>
  DynamicSSDIndex<T, TagT>::DynamicSSDIndex(Parameters &parameters, const std::string disk_prefix_in,
                                            const std::string disk_prefix_out, Distance<T> *dist,
                                            ccann::Metric dist_metric, int search_mode, bool use_mem_index,
                                            bool read_only, int cpu_bound) {
    // check if file exists.
    // 构造动态索引前先确认核心 graph 文件存在，否则无法恢复 graph layout。
    if (!std::filesystem::exists(disk_prefix_in + "_disk.index")) {
      // 记录不可恢复的内部状态错误，下面通常会终止当前流程。
      LOG(ERROR) << "Disk index file does not exist: " << disk_prefix_in << "_disk.index";
      exit(-1);
    }
    // 若用户要求小型 DRAM index 加速入口搜索，则对应 _mem.index 也必须存在。
    if (use_mem_index && !std::filesystem::exists(disk_prefix_in + "_mem.index")) {
      // 记录不可恢复的内部状态错误，下面通常会终止当前流程。
      LOG(ERROR) << "In-memory index file does not exist: " << disk_prefix_in << "_mem.index";
      exit(-1);
    }

    this->active_del[0] = true;
    this->active_del[1] = false;
    this->_dist_metric = dist_metric;

    // 把外层 Parameters 中的磁盘索引参数复制到底层 SSDIndex 使用的参数对象。
    _paras_disk.Set<unsigned>("L", parameters.Get<unsigned>("L_disk"));
    // 把外层 Parameters 中的磁盘索引参数复制到底层 SSDIndex 使用的参数对象。
    _paras_disk.Set<unsigned>("R", parameters.Get<unsigned>("R_disk"));
    // 把外层 Parameters 中的磁盘索引参数复制到底层 SSDIndex 使用的参数对象。
    _paras_disk.Set<unsigned>("C", parameters.Get<unsigned>("C"));
    // 把外层 Parameters 中的磁盘索引参数复制到底层 SSDIndex 使用的参数对象。
    _paras_disk.Set<float>("alpha", parameters.Get<float>("alpha_disk"));
    // 把外层 Parameters 中的磁盘索引参数复制到底层 SSDIndex 使用的参数对象。
    _paras_disk.Set<unsigned>("beamwidth", parameters.Get<unsigned>("beamwidth"));
    // 把外层 Parameters 中的磁盘索引参数复制到底层 SSDIndex 使用的参数对象。
    _paras_disk.Set<bool>("saturate_graph", 0);

    // 保存底层搜索/更新允许使用的线程数。
    _num_threads = parameters.Get<_u32>("num_threads");
    // 保存 beam width，控制一次图搜索并行/批量探索的节点数。
    _beamwidth = parameters.Get<uint32_t>("beamwidth");

    _disk_index_prefix_in = disk_prefix_in;
    _disk_index_prefix_out = disk_prefix_out;
    _dist_comp = dist;

    // 创建 graph 文件 reader；Linux 实现同时支持 O_DIRECT/io_uring 与 PM DAX。
    reader.reset(new LinuxAlignedFileReader());
    // 为 PQ compressed 文件创建独立 reader/writer 和 DAX 映射状态。
    pq_compressed_writer.reset(new LinuxAlignedFileReader());
    // 为 tags 文件创建独立 writer/DAX 映射。
    tags_writer.reset(new LinuxAlignedFileReader());
    // 为持久化 Location Table 创建独立 writer/DAX 映射。
    id2loc_writer.reset(new LinuxAlignedFileReader());

    // 构造真正执行 graph search/insert 的底层 SSDIndex，并把上述文件访问器注入进去。
    _disk_index = new ccann::SSDIndex<T, TagT>(this->_dist_metric, reader, pq_compressed_writer, tags_writer,
                                                 id2loc_writer, false, false, &_paras_disk);
// 若构建 CCANN-J baseline，则额外生成/提交 journal 记录；Soft Insert 主设计本身不依赖 journal。
#ifdef J_ANN
    _disk_index->journals = new v2::Journal<TagT> *[N_JOURNAL];
    for (int i = 0; i < N_JOURNAL; i++) {
      _disk_index->journals[i] = new v2::Journal<TagT>(disk_prefix_out + "_journal" + std::to_string(i) + ".log");
    }
#endif

    std::string id2loc_file(disk_prefix_in + "_disk.index.id2loc");
    std::string index_file(disk_prefix_in + "_disk.index");
    // 旧索引若没有单独的 location table，则现场补建 identity mapping。
    if (!file_exists(id2loc_file)) {
      // 生成 .id2loc，使后续动态 out-of-place update 可以通过 ID→loc 解耦逻辑身份和物理位置。
      build_id2loc_mapping(index_file, id2loc_file);
    }

// 默认保护原始索引：动态更新前会复制 shadow 版本，避免实验直接改坏输入索引。
#ifndef NO_POLLUTE_ORIGINAL
    // 只读模式直接使用原索引，不创建 shadow 副本。
    if (read_only) {
      LOG(WARNING)
          << "Read-only mode is enabled. The original index files will not be modified during dynamic updates.";
    } else {
      std::string disk_index_prefix_shadow = _disk_index_prefix_in + "_shadow";
      // 复制出 shadow 索引，后续动态更新落到副本而不是输入数据。
      copy_index(_disk_index_prefix_in, disk_index_prefix_shadow);
      // 输出运行状态/统计信息，便于观察实验流程和后台线程进度。
      LOG(INFO) << "Copy disk index file to " << disk_index_prefix_shadow << "_disk.index";
      _disk_index_prefix_in = disk_index_prefix_shadow;
    }
#endif

    if (search_mode == BEAM_SEARCH || search_mode == PAGE_SEARCH || search_mode == PIPE_SEARCH ||
        search_mode == PARA_SEARCH) {
      // 保存对外 DynamicSSDIndex 的搜索模式。
      this->search_mode = search_mode;
      // 同步设置底层 SSDIndex，确保真正执行的 search path 与外层一致。
      _disk_index->search_mode = search_mode;
    } else {
      // 记录不可恢复的内部状态错误，下面通常会终止当前流程。
      LOG(ERROR) << "Invalid search mode: " << search_mode
                 << ". Must be one of BEAM_SEARCH, PAGE_SEARCH, or PIPE_SEARCH.";
      exit(-1);
    }
    bool use_page_search = (search_mode == PAGE_SEARCH);
    // 加载 graph metadata、PQ、id2loc、tags，并在 PM 场景建立 DAX 映射和恢复 allocator。
    int res = _disk_index->load(_disk_index_prefix_in.c_str(), _num_threads, true, use_page_search, cpu_bound);
    if (res != 0) {
      // 输出运行状态/统计信息，便于观察实验流程和后台线程进度。
      LOG(INFO) << "Failed to load disk index in DynamicSSDIndex constructor";
      exit(-1);
    }

    this->_use_mem_index = use_mem_index;
    if (use_mem_index) {
      std::string mem_index_path = disk_prefix_in + "_mem.index";  // use the original one.
      // 输出运行状态/统计信息，便于观察实验流程和后台线程进度。
      LOG(INFO) << "Use static in-memory index for acceleration, path: " << mem_index_path;
      // 加载小型 DRAM navigation index，用少量内存换取更好的图搜索入口。
      _disk_index->load_mem_index(this->_dist_metric, _disk_index->data_dim, mem_index_path);
    }
  }


  // ============================================================================
  // show_statistics
  // 把统计展示请求直接转发给底层 SSDIndex。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::show_statistics() {
    _disk_index->statistics();
  }


  // ============================================================================
  // reset_statistics
  // 清空底层 SSDIndex 累积的性能统计计数器。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::reset_statistics() {
    _disk_index->reset_statistics();
  }


  // ============================================================================
  // DynamicSSDIndex_dtor
  // 析构时先输出统计并等待所有异步插入结束，再向 commit queue 放入 terminate 任务并 join commit
  // thread，保证后台持久化线程正常退出后对象才销毁。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<typename T, typename TagT>
  DynamicSSDIndex<T, TagT>::~DynamicSSDIndex() {
    // put in destructor code
    // 销毁前输出最终统计，避免进程退出后丢失实验指标。
    _disk_index->statistics();

    // 输出运行状态/统计信息，便于观察实验流程和后台线程进度。
    LOG(INFO) << "Waiting for all async threads to finish...";
    if (_disk_index->insert_pool != nullptr) {
      // 先等待前台提交到 insert thread pool 的任务全部结束，否则 commit/reader 仍可能被使用。
      _disk_index->synchronize_insertions();

#ifdef USE_SMALL_THREAD_POOL
      // _disk_index is not automatically deleted.
      delete _disk_index->insert_pool.get();
#endif
    }

    // 构造 terminate sentinel CommitTask；不携带 PQ/point，只用 terminate=true 通知后台线程退出。
    auto final_task = new typename SSDIndex<T, TagT>::CommitTask{
        .pq_coords = std::vector<uint8_t>(), .target_id = 0, .terminate = true, .point = nullptr};
    // 把终止任务放入 commit queue。
    _disk_index->commit_tasks.push(final_task);
    // 唤醒可能阻塞等待 queue 的 commit thread，让它及时看到 terminate sentinel。
    _disk_index->commit_tasks.push_notify_all();
    // 等待 commit thread 完成剩余 checkpoint 收尾并真正退出。
    if (_disk_index->commit_thread_->joinable()) {
      // 输出运行状态/统计信息，便于观察实验流程和后台线程进度。
      LOG(INFO) << "Joining commit thread...";
      // 等待 commit thread 完成剩余 checkpoint 收尾并真正退出。
      _disk_index->commit_thread_->join();
      delete _disk_index->commit_thread_;
    }
    // 输出运行状态/统计信息，便于观察实验流程和后台线程进度。
    LOG(INFO) << "Done";
  }


  // ============================================================================
  // insert
  // 对外插入 API。持有 merge 的共享锁以阻止 merge 与 insert 交叉执行，然后根据编译选项选择异步或同步插入入口，并返回新分配的逻辑 target_id。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<typename T, typename TagT>
  int DynamicSSDIndex<T, TagT>::insert(const T *point, const TagT &tag) {
    // insert 只获取 merge 锁的共享模式：多个 insert 可并发，但 final_merge 的独占锁会阻止它们进入。
    std::shared_lock<std::shared_timed_mutex> lock(_merge_lock);  // prevent merge during insert
    // journal->append(v2::TxType::kInsert, tag);
    // 取得当前活动 tombstone set，search_phase 会避开这些已逻辑删除的节点。
    auto *deletion_set = &deletion_sets[active_delete_set];
    int target_id = 0;

// 编译为异步插入模式时，对外 insert API 走 async_insert_in_place。
#ifdef ASYNC_INSERTION
    // 调用 CCANN 的异步插入入口：同步搜索后尽量把 insert phase 后台化。
    target_id = _disk_index->async_insert_in_place(point, tag, deletion_set);
#else
    // 同步模式下在当前线程完成搜索和 graph 更新。
    target_id = _disk_index->insert_in_place(point, tag, deletion_set);
#endif

    // 返回这个新向量的稳定逻辑 ID；物理 loc 后续可以变化，但 ID 不变。
    return target_id;
  }


  // ============================================================================
  // search
  // 对外搜索 API。根据 search_mode 调用底层 beam/page/pipe/para search，先收集较大的候选 tag+distance 列表，再在
  // delete_lock 保护下过滤 tombstone，直到返回 K 个未删除结果。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::search(const T *query, const uint64_t K, const uint32_t mem_L, const uint64_t search_L,
                                        const uint32_t beam_width, TagT *tags, float *distances, QueryStats *stats,
                                        bool dyn_search_l) {
    // 先准备较大的内部结果缓冲区，底层搜索可能返回多于最终 K 的候选以便过滤 tombstone。
    std::vector<TagT> result_tags(4096);
    // 为内部候选的距离准备与 tags 同步的数组。
    std::vector<float> result_distances(4096);
    // 查询结束后会再次用当前活动 tombstone set 过滤逻辑删除结果。
    auto *deletion_set = &deletion_sets[active_delete_set];
    size_t n = 0;
    if (search_mode == BEAM_SEARCH) {
      // 选择经典 beam search 路径。
      n = _disk_index->beam_search(query, search_L, mem_L, search_L, result_tags.data(), result_distances.data(),
                                   beam_width, stats, deletion_set, dyn_search_l);
    } else if (search_mode == PAGE_SEARCH) {
      // 选择 page-aware search 路径。
      n = _disk_index->page_search(query, search_L, mem_L, search_L, result_tags.data(), result_distances.data(),
                                   beam_width, stats);
    } else if (search_mode == PIPE_SEARCH) {
      // 选择 PipeSearch 路径。
      n = _disk_index->pipe_search(query, search_L, mem_L, search_L, result_tags.data(), result_distances.data(),
                                   beam_width, stats);
    } else if (search_mode == PARA_SEARCH) {
      // 选择 CCANN PNE/parallel search 路径。
      n = _disk_index->para_search(query, search_L, mem_L, search_L, result_tags.data(), result_distances.data(),
                                   beam_width, stats);
    } else {
      // 记录不可恢复的内部状态错误，下面通常会终止当前流程。
      LOG(ERROR) << "Invalid search mode: " << search_mode;
      exit(-1);
    }
    std::vector<NeighborTag<TagT>> best_vec;
    for (size_t i = 0; i < n; i++) {
      // 把底层并行数组结果包装成 NeighborTag，便于后面统一过滤删除项。
      best_vec.emplace_back(result_tags[i], result_distances[i]);
    }
    // 以共享模式锁住 deletion set，保证过滤期间 lazy_delete/save_del_set 不改变集合结构。
    std::shared_lock<std::shared_timed_mutex> lock(delete_lock);
    size_t pos = 0;

    for (auto iter : best_vec) {
      // 只把当前不在 tombstone set 中的 tag 返回给用户。
      if (deletion_set->find(iter.tag) == deletion_set->end()) {
        tags[pos] = iter.tag;
        distances[pos] = iter.dist;
        pos++;
      }
      // 已经收集到请求的 K 个有效结果后立即返回，不再扫描剩余候选。
      if (pos == K) {
        // 当前函数工作已经完成，直接返回调用方。
        return;
      }
    }
    // LOG(INFO) << "Failed to find enough tags after " << i + 1 << " attempts";
  }


  // ============================================================================
  // lazy_delete
  // 惰性删除：不立即修改 graph，只把 tag 放入当前活动 deletion set 和 deleted_tags 列表。查询时会过滤这些 tag，真正的图结构回收留给后续
  // merge。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::lazy_delete(const TagT &tag) {
    // 删除集合是可变共享状态，使用独占锁序列化 lazy_delete 与集合切换。
    std::unique_lock<std::shared_timed_mutex> lock(delete_lock);
    // journal->append(v2::TxType::kDelete, tag);

    // 防御性检查当前选中的 deletion buffer 是否确实处于可接收新删除的状态。
    if (active_del[active_delete_set].load() == false) {
      // 记录不可恢复的内部状态错误，下面通常会终止当前流程。
      LOG(ERROR) << "Active deletion set indicated as _deletion_set_" << active_delete_set
                 << " but it cannot accept deletions";
    }

    // if not deleted, then buffer the deletion.
    // 避免同一个 tag 重复进入 tombstone set 和 deleted_tags 列表。
    if (deletion_sets[active_delete_set].find(tag) == deletion_sets[active_delete_set].end()) {
      // 把 tag 标记为逻辑删除；之后 search 会过滤它。
      deletion_sets[active_delete_set].insert(tag);
      // 同时保留有序/可遍历的删除 tag 列表，merge_deletes 后续需要批量处理。
      deleted_tags[active_delete_set].push_back(tag);
    }
  }


  // ============================================================================
  // save_del_set
  // 双缓冲 deletion set 切换。先清空下一组并标记为可接收删除，再把 active_delete_set 切过去，旧组冻结下来供 merge 处理，从而允许删除记录与
  // merge 解耦。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::save_del_set() {
    // 在两个 deletion buffers 之间切换：nxt_idx 是即将接收新删除的空集合，cur_idx 是将被冻结给 merge 的集合。
    int nxt_idx = 1 - active_delete_set, cur_idx = active_delete_set;
    std::unique_lock<std::shared_timed_mutex> lock(delete_lock);
    // 确保下一 buffer 没有上一次 merge 残留 tombstones。
    deletion_sets[nxt_idx].clear();
    // 同步清空下一 buffer 的删除列表表示。
    deleted_tags[nxt_idx].clear();
    bool expected_active = false;
#ifndef ODIN_ANN_IMMEDIATE_NO_CC
    // 原子把下一 buffer 标记为 active，防止状态机意外从错误状态切换。
    if (active_del[nxt_idx].compare_exchange_strong(expected_active, true)) {
      // 输出运行状态/统计信息，便于观察实验流程和后台线程进度。
      LOG(INFO) << "Cleared deletion set " << nxt_idx << " - ready to accept new points";
    } else {
      // 输出运行状态/统计信息，便于观察实验流程和后台线程进度。
      LOG(INFO) << "Failed to clear deletion set " << nxt_idx;
    }
#endif
    // 正式切换生产者写入目标；之后新的 lazy_delete 都进入 nxt_idx。
    active_delete_set = nxt_idx;
    // 旧 buffer 停止接收新删除，因此 merge 可以把它当成稳定快照处理。
    active_del[cur_idx].store(false);
  }


  // ============================================================================
  // final_merge
  // 对外最终 merge：获取 merge 独占锁，先切换 deletion set，让待处理删除集合稳定下来，再执行 merge_deletes 或 baseline
  // merge。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::final_merge(const uint32_t &nthreads, const uint32_t &n_sampled_nbrs) {
    // 取得 merge 独占锁，等待所有持共享锁的 insert 退出，并阻止新的 insert 进入。
    std::unique_lock<std::shared_timed_mutex> lock(_merge_lock);  // only one merge at a time
    // _disk_index_in -> _disk_index_out
    // 先切换 deletion 双缓冲，把这次 merge 要处理的 tombstone 集合冻结下来。
    save_del_set();
    ccann::Timer timer;
    // 执行实际 graph merge/rebuild。
    merge(nthreads, n_sampled_nbrs);

    // TODO(gh): do we really need to reload disk index?
    // std::swap(_disk_index_prefix_in, _disk_index_prefix_out);
    // _disk_index->reload(_disk_index_prefix_in.c_str(), _num_threads);
#ifndef ODIN_ANN_IMMEDIATE_NO_CC
    // 输出运行状态/统计信息，便于观察实验流程和后台线程进度。
    LOG(INFO) << "Merge time : " << timer.elapsed() / 1000 << " ms";
#endif
  }


  // ============================================================================
  // merge
  // 根据编译模式选择普通增量 merge 或带删除重建的 merge_deletes。
  // ============================================================================
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::merge(const uint32_t &nthreads, const uint32_t &n_sampled_nbrs) {
// OdinANN immediate baseline 下使用不带 CCANN crash-consistency 删除重建逻辑的 merge。
#ifdef ODIN_ANN_IMMEDIATE_NO_CC
    // baseline 模式只调用底层增量 merge。
    _disk_index->merge(_disk_index_prefix_in, _disk_index_prefix_out);
#else
    // CCANN 正常路径用被冻结的删除列表和集合重建 graph，消除 tombstones 并回收空间。
    _disk_index->merge_deletes(_disk_index_prefix_in, _disk_index_prefix_out, deleted_tags[1 - active_delete_set],
                               deletion_sets[1 - active_delete_set], nthreads, n_sampled_nbrs);
#endif
  }

  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template class DynamicSSDIndex<float>;
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template class DynamicSSDIndex<uint8_t>;
  // 模板参数 T 表示向量坐标类型，TagT 表示用户侧 tag 类型；同一套逻辑可实例化为 float/int8/uint8。
  template class DynamicSSDIndex<int8_t>;
}  // namespace ccann
