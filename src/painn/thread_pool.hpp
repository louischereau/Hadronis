#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

struct PaiNNUpdateScratch {
  std::vector<float> u;
  std::vector<float> v;
  std::vector<float> stacked;
  std::vector<float> hidden;
  std::vector<float> gates;

  explicit PaiNNUpdateScratch(int dim)
      : u(3u * static_cast<std::size_t>(dim)),
        v(3u * static_cast<std::size_t>(dim)),
        stacked(static_cast<std::size_t>(dim)),
        hidden(static_cast<std::size_t>(dim)),
        gates(3u * static_cast<std::size_t>(dim)) {}
};

struct PaiNNMessageScratch {
  std::vector<float> phi;    // [F]
  std::vector<float> mix;    // [3F]
  std::vector<float> filter; // [3F]
  std::vector<float> ds_sum; // [F]
  std::vector<float> dv_sum; // [3F]

  explicit PaiNNMessageScratch(int dim)
      : phi(static_cast<std::size_t>(dim)),
        mix(3u * static_cast<std::size_t>(dim)),
        filter(3u * static_cast<std::size_t>(dim)),
        ds_sum(static_cast<std::size_t>(dim)),
        dv_sum(3u * static_cast<std::size_t>(dim)) {}
};

template <typename Scratch> class PersistentThreadPool {
  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  std::vector<std::thread> workers_;
  std::vector<Scratch> scratch_pool_;
  std::function<void(int, Scratch &)> job_;
  std::atomic<int> next_item_{0};
  int end_item_ = 0;
  std::size_t generation_ = 0;
  std::size_t completed_workers_ = 0;
  bool main_thread_done_ = false;
  bool stop_ = false;

  void worker_loop(std::size_t scratch_index) {
    std::size_t observed_generation = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      cv_.wait(lock,
               [&]() { return stop_ || generation_ != observed_generation; });

      if (stop_) {
        return;
      }

      observed_generation = generation_;
      const int local_end = end_item_;
      auto local_job = job_;
      lock.unlock();

      Scratch &scratch = scratch_pool_[scratch_index];
      for (;;) {
        const int item = next_item_.fetch_add(1, std::memory_order_relaxed);
        if (item >= local_end) {
          break;
        }
        local_job(item, scratch);
      }

      lock.lock();
      ++completed_workers_;
      if (completed_workers_ == workers_.size() && main_thread_done_) {
        done_cv_.notify_one();
      }
    }
  }

public:
  PersistentThreadPool(const PersistentThreadPool &) = delete;
  PersistentThreadPool &operator=(const PersistentThreadPool &) = delete;
  PersistentThreadPool(PersistentThreadPool &&) = delete;
  PersistentThreadPool &operator=(PersistentThreadPool &&) = delete;

  template <typename ScratchFactory>
  PersistentThreadPool(unsigned worker_count, ScratchFactory &&make_scratch) {
    if (worker_count <= 1) {
      return;
    }

    scratch_pool_.reserve(worker_count - 1);
    workers_.reserve(worker_count - 1);
    for (unsigned worker = 1; worker < worker_count; ++worker) {
      scratch_pool_.push_back(make_scratch());
      workers_.emplace_back(
          [this, worker_index = worker - 1]() { worker_loop(worker_index); });
    }
  }

  ~PersistentThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      ++generation_;
    }
    cv_.notify_all();

    for (auto &worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  template <typename Fn> void run(int n_items, Fn &&fn, Scratch &main_scratch) {
    if (n_items <= 1 || workers_.empty()) {
      for (int item = 0; item < n_items; ++item) {
        fn(item, main_scratch);
      }
      return;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      job_ = std::forward<Fn>(fn);
      end_item_ = n_items;
      completed_workers_ = 0;
      main_thread_done_ = false;
      next_item_.store(0, std::memory_order_relaxed);
      ++generation_;
    }
    cv_.notify_all();

    for (;;) {
      const int item = next_item_.fetch_add(1, std::memory_order_relaxed);
      if (item >= n_items) {
        break;
      }
      fn(item, main_scratch);
    }

    std::unique_lock<std::mutex> lock(mutex_);
    main_thread_done_ = true;
    if (completed_workers_ != workers_.size()) {
      done_cv_.wait(lock,
                    [&]() { return completed_workers_ == workers_.size(); });
    }
    job_ = {};
  }
};
