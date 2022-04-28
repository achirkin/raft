/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <optional>

#include <common/benchmark.hpp>
#include <raft/spatial/knn/knn.cuh>

#if defined RAFT_NN_COMPILED
#include <raft/spatial/knn/specializations.hpp>
#endif

#include <raft/random/rng.cuh>
#include <raft/sparse/detail/utils.h>

#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <rmm/mr/host/new_delete_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>

namespace raft::bench::spatial {

struct params {
  /** Size of the dataset. */
  size_t n_samples;
  /** Number of dimensions in the dataset. */
  size_t n_dims;
  /** The batch size -- number of KNN searches. */
  size_t n_probes;
  /** Number of nearest neighbours to find for every probe. */
  size_t k;
};

auto operator<<(std::ostream& os, const params& p) -> std::ostream&
{
  os << p.n_samples << "#" << p.n_dims << "#" << p.n_probes << "#" << p.k;
  return os;
}

enum class TransferStrategy { NO_COPY, COPY_PLAIN, COPY_PINNED, MAP_PINNED, MANAGED };

auto operator<<(std::ostream& os, const TransferStrategy& ts) -> std::ostream&
{
  switch (ts) {
    case TransferStrategy::NO_COPY: os << "NO_COPY"; break;
    case TransferStrategy::COPY_PLAIN: os << "COPY_PLAIN"; break;
    case TransferStrategy::COPY_PINNED: os << "COPY_PINNED"; break;
    case TransferStrategy::MAP_PINNED: os << "MAP_PINNED"; break;
    case TransferStrategy::MANAGED: os << "MANAGED"; break;
    default: os << "UNKNOWN";
  }
  return os;
}

struct device_resource {
 public:
  explicit device_resource(bool managed) : managed_(managed)
  {
    if (managed_) {
      res_ = new rmm::mr::managed_memory_resource();
    } else {
      res_ = rmm::mr::get_current_device_resource();
    }
  }

  ~device_resource()
  {
    if (managed_) { delete res_; }
  }

  [[nodiscard]] auto get() const -> rmm::mr::device_memory_resource* { return res_; }

 private:
  const bool managed_;
  rmm::mr::device_memory_resource* res_;
};

template <typename T>
struct host_uvector {
  host_uvector(size_t n, bool pinned) : n_(n)
  {
    if (pinned) {
      res_ = new rmm::mr::pinned_memory_resource();
    } else {
      res_ = new rmm::mr::new_delete_resource();
    }
    arr_ = static_cast<T*>(res_->allocate(n_ * sizeof(T)));
  }

  ~host_uvector() noexcept
  {
    res_->deallocate(arr_, n_ * sizeof(T));
    delete res_;
  }

  auto data() -> T* { return arr_; }
  [[nodiscard]] auto size() const -> size_t { return n_; }

 private:
  rmm::mr::host_memory_resource* res_;
  size_t n_;
  T* arr_;
};

struct brute_force_knn {
  template <typename ValT, typename IdxT>
  static void run(const raft::handle_t& handle,
                  const params& params,
                  const ValT* data,
                  const ValT* search_items,
                  ValT* out_dists,
                  IdxT* out_idxs)
  {
    std::vector<ValT*> input{const_cast<ValT*>(data)};
    std::vector<size_t> sizes{params.n_samples};
    raft::spatial::knn::brute_force_knn<IdxT, ValT, size_t>(handle,
                                                            input,
                                                            sizes,
                                                            params.n_dims,
                                                            const_cast<ValT*>(search_items),
                                                            params.n_probes,
                                                            out_idxs,
                                                            out_dists,
                                                            params.k);
  }
};

template <typename ValT, typename IdxT, typename ImplT>
struct knn : public fixture {
  explicit knn(const params& p, const TransferStrategy& strategy)
    : params_(p),
      strategy_(strategy),
      dev_mem_res_(strategy == TransferStrategy::MANAGED),
      data_host_(p.n_samples * p.n_dims),
      search_items_(p.n_probes * p.n_dims, stream),
      out_dists_(p.n_probes * p.k, stream),
      out_idxs_(p.n_probes * p.k, stream)
  {
    raft::random::RngState state{42};
    raft::random::uniform(
      state, search_items_.data(), search_items_.size(), ValT(-1.0), ValT(1.0), stream);
    rmm::device_uvector<ValT> d(data_host_.size(), stream);
    raft::random::uniform(state, d.data(), d.size(), ValT(-1.0), ValT(1.0), stream);
    copy(data_host_.data(), d.data(), data_host_.size(), stream);
  }

  void run_benchmark(::benchmark::State& state) override
  {
    using_pool_memory_res default_resource;

    try {
      std::ostringstream label_stream;
      label_stream << params_ << "#" << strategy_;
      state.SetLabel(label_stream.str());
      raft::handle_t handle(stream);

      // benchmark loop
      for (auto _ : state) {
        // managed or plain device memory initialized anew every time
        rmm::device_uvector<ValT> data(data_host_.size(), stream, dev_mem_res_.get());
        ValT* data_ptr         = data.data();
        size_t allocation_size = data_host_.size() * sizeof(ValT);

        // Non-benchmarked part: using different methods to copy the data if necessary
        switch (strategy_) {
          case TransferStrategy::NO_COPY:  // copy data to GPU before starting the timer.
            copy(data_ptr, data_host_.data(), data_host_.size(), stream);
            break;
          case TransferStrategy::COPY_PINNED:
            RAFT_CUDA_TRY(
              cudaHostRegister(data_host_.data(), allocation_size, cudaHostRegisterDefault));
            break;
          case TransferStrategy::MAP_PINNED:
            RAFT_CUDA_TRY(
              cudaHostRegister(data_host_.data(), allocation_size, cudaHostRegisterMapped));
            RAFT_CUDA_TRY(cudaHostGetDevicePointer(&data_ptr, data_host_.data(), 0));
            break;
          case TransferStrategy::MANAGED:  // sic! using std::memcpy rather than cuda copy
            CUDA_CHECK(cudaMemAdvise(
              data_ptr, allocation_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
            CUDA_CHECK(cudaMemAdvise(
              data_ptr, allocation_size, cudaMemAdviseSetAccessedBy, handle.get_device()));
            std::memcpy(data_ptr, data_host_.data(), allocation_size);
            break;
          default: break;
        }

        flush_L2_cache();
        {
          // Timer synchronizes the stream, so all prior gpu work should be done before it sets off.
          cuda_event_timer timer(state, stream);
          switch (strategy_) {
            case TransferStrategy::COPY_PLAIN:
            case TransferStrategy::COPY_PINNED:
              copy(data_ptr, data_host_.data(), data_host_.size(), stream);
            default: break;
          }
          ImplT::run(
            handle, params_, data_ptr, search_items_.data(), out_dists_.data(), out_idxs_.data());
        }

        switch (strategy_) {
          case TransferStrategy::COPY_PINNED:
          case TransferStrategy::MAP_PINNED:
            RAFT_CUDA_TRY(cudaHostUnregister(data_host_.data()));
            break;
          default: break;
        }
      }
    } catch (raft::exception& e) {
      state.SkipWithError(e.what());
    } catch (std::bad_alloc& e) {
      state.SkipWithError(e.what());
    }
  }

 private:
  const params params_;
  const TransferStrategy strategy_;
  device_resource dev_mem_res_;

  std::vector<ValT> data_host_;
  rmm::device_uvector<ValT> search_items_;
  rmm::device_uvector<ValT> out_dists_;
  rmm::device_uvector<IdxT> out_idxs_;
};

const std::vector<params> kInputs{{20000000, 128, 1000, 32}, {40000000, 128, 1000, 32}};

const std::vector<TransferStrategy> kStrategies{TransferStrategy::NO_COPY,
                                                TransferStrategy::COPY_PLAIN,
                                                TransferStrategy::COPY_PINNED,
                                                TransferStrategy::MAP_PINNED,
                                                TransferStrategy::MANAGED};

#define KNN_REGISTER(ValT, IdxT, ImplT)                                         \
  namespace BENCHMARK_PRIVATE_NAME(knn)                                         \
  {                                                                             \
    using KNN = knn<ValT, IdxT, ImplT>;                                         \
    RAFT_BENCH_REGISTER(KNN, #ValT "/" #IdxT "/" #ImplT, kInputs, kStrategies); \
  }

KNN_REGISTER(float, int64_t, brute_force_knn);

}  // namespace raft::bench::spatial
