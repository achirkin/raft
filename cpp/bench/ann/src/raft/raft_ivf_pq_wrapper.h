/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#pragma once

#include "../common/ann_types.hpp"
#include "raft_ann_bench_utils.h"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft_runtime/neighbors/ivf_pq.hpp>
#include <raft_runtime/neighbors/refine.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <type_traits>

namespace raft::bench::ann {

template <typename T, typename IdxT>
class RaftIvfPQ : public ANN<T> {
 public:
  using typename ANN<T>::AnnSearchParam;
  using ANN<T>::dim_;

  struct SearchParam : public AnnSearchParam {
    raft::neighbors::ivf_pq::search_params pq_param;
  };

  using BuildParam = raft::neighbors::ivf_pq::index_params;

  RaftIvfPQ(Metric metric, int dim, const BuildParam& param, float refine_ratio);

  void build(const T* dataset, size_t nrow, cudaStream_t stream) final;

  void set_search_param(const AnnSearchParam& param) override;
  void set_search_dataset(const T* dataset, size_t nrow) override;

  // TODO: if the number of results is less than k, the remaining elements of 'neighbors'
  // will be filled with (size_t)-1
  void search(const T* queries,
              int batch_size,
              int k,
              size_t* neighbors,
              float* distances,
              cudaStream_t stream = 0) const override;

  // to enable dataset access from GPU memory
  AlgoProperty get_property() const override
  {
    AlgoProperty property;
    property.dataset_memory_type      = MemoryType::HostMmap;
    property.query_memory_type        = MemoryType::Device;
    property.need_dataset_when_search = false;  // actually it is only used during refinement
    return property;
  }
  void save(const std::string& file) const override;
  void load(const std::string&) override;
  void simulate_use(cudaStream_t stream = 0) const override;

 private:
  mutable raft::random::Rng rng_{1234ULL};
  raft::device_resources handle_;
  BuildParam index_params_;
  raft::neighbors::ivf_pq::search_params search_params_;
  std::optional<raft::neighbors::ivf_pq::index<IdxT>> index_;
  int device_;
  int dimension_;
  float refine_ratio_ = 1.0;
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> mr_;
  raft::device_matrix_view<const T, IdxT> dataset_;
};
template <typename T, typename IdxT>
RaftIvfPQ<T, IdxT>::RaftIvfPQ(Metric metric, int dim, const BuildParam& param, float refine_ratio)
  : ANN<T>(metric, dim),
    index_params_(param),
    dimension_(dim),
    refine_ratio_(refine_ratio),
    mr_(rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull)
{
  index_params_.metric = parse_metric_type(metric);
  RAFT_CUDA_TRY(cudaGetDevice(&device_));
}

template <typename T, typename IdxT>
void RaftIvfPQ<T, IdxT>::save(const std::string& file) const
{
  raft::runtime::neighbors::ivf_pq::serialize(handle_, file, *index_);
}

template <typename T, typename IdxT>
void RaftIvfPQ<T, IdxT>::load(const std::string& file)
{
  auto index_tmp = raft::neighbors::ivf_pq::index<IdxT>(handle_, index_params_, dimension_);
  raft::runtime::neighbors::ivf_pq::deserialize(handle_, file, &index_tmp);
  index_.emplace(std::move(index_tmp));
  return;
}

template <typename T, typename IdxT>
void RaftIvfPQ<T, IdxT>::build(const T* dataset, size_t nrow, cudaStream_t)
{
  auto dataset_v = raft::make_device_matrix_view<const T, IdxT>(dataset, IdxT(nrow), dim_);

  index_.emplace(raft::runtime::neighbors::ivf_pq::build(handle_, index_params_, dataset_v));
  return;
}

template <typename T, typename IdxT>
void RaftIvfPQ<T, IdxT>::set_search_param(const AnnSearchParam& param)
{
  auto search_param = dynamic_cast<const SearchParam&>(param);
  search_params_    = search_param.pq_param;
  assert(search_params_.n_probes <= index_params_.n_lists);
}

template <typename T, typename IdxT>
void RaftIvfPQ<T, IdxT>::set_search_dataset(const T* dataset, size_t nrow)
{
  dataset_ = raft::make_device_matrix_view<const T, IdxT>(dataset, nrow, index_->dim());
}

template <typename T, typename IdxT>
void RaftIvfPQ<T, IdxT>::search(const T* queries,
                                int batch_size,
                                int k,
                                size_t* neighbors,
                                float* distances,
                                cudaStream_t stream) const
{
  if (refine_ratio_ > 1.0f) {
    uint32_t k0 = static_cast<uint32_t>(refine_ratio_ * k);
    auto queries_v =
      raft::make_device_matrix_view<const T, IdxT>(queries, batch_size, index_->dim());
    auto distances_tmp = raft::make_device_matrix<float, IdxT>(handle_, batch_size, k0);
    auto candidates    = raft::make_device_matrix<IdxT, IdxT>(handle_, batch_size, k0);

    raft::runtime::neighbors::ivf_pq::search(
      handle_, search_params_, *index_, queries_v, candidates.view(), distances_tmp.view());

    if (get_property().dataset_memory_type == MemoryType::Device) {
      auto queries_v =
        raft::make_device_matrix_view<const T, IdxT>(queries, batch_size, index_->dim());
      auto neighbors_v = raft::make_device_matrix_view<IdxT, IdxT>((IdxT*)neighbors, batch_size, k);
      auto distances_v = raft::make_device_matrix_view<float, IdxT>(distances, batch_size, k);

      raft::runtime::neighbors::refine(handle_,
                                       dataset_,
                                       queries_v,
                                       candidates.view(),
                                       neighbors_v,
                                       distances_v,
                                       index_->metric());
    } else {
      auto queries_host    = raft::make_host_matrix<T, IdxT>(batch_size, index_->dim());
      auto candidates_host = raft::make_host_matrix<IdxT, IdxT>(batch_size, k0);
      auto neighbors_host  = raft::make_host_matrix<IdxT, IdxT>(batch_size, k);
      auto distances_host  = raft::make_host_matrix<float, IdxT>(batch_size, k);

      raft::copy(queries_host.data_handle(),
                 queries,
                 queries_host.size(),
                 resource::get_cuda_stream(handle_));
      raft::copy(candidates_host.data_handle(),
                 candidates.data_handle(),
                 candidates_host.size(),
                 resource::get_cuda_stream(handle_));

      auto dataset_v = raft::make_host_matrix_view<const T, IdxT>(
        dataset_.data_handle(), batch_size, index_->dim());

      raft::runtime::neighbors::refine(handle_,
                                       dataset_v,
                                       queries_host.view(),
                                       candidates_host.view(),
                                       neighbors_host.view(),
                                       distances_host.view(),
                                       index_->metric());

      raft::copy(neighbors,
                 (size_t*)neighbors_host.data_handle(),
                 neighbors_host.size(),
                 resource::get_cuda_stream(handle_));
      raft::copy(distances,
                 distances_host.data_handle(),
                 distances_host.size(),
                 resource::get_cuda_stream(handle_));
    }
  } else {
    auto queries_v =
      raft::make_device_matrix_view<const T, IdxT>(queries, batch_size, index_->dim());
    auto neighbors_v = raft::make_device_matrix_view<IdxT, IdxT>((IdxT*)neighbors, batch_size, k);
    auto distances_v = raft::make_device_matrix_view<float, IdxT>(distances, batch_size, k);

    raft::runtime::neighbors::ivf_pq::search(
      handle_, search_params_, *index_, queries_v, neighbors_v, distances_v);
  }
  resource::sync_stream(handle_);
  return;
}

template <typename T, typename IdxT>
void RaftIvfPQ<T, IdxT>::simulate_use(cudaStream_t stream) const
{
  /*
    The goal of this function is to probe a certain number of cluster randomly.
    Assuming the cluster data is in managed memory, this should mimic the real-world state of the
    index after some use. The point is that in a subsequent search we expect to see some UVM/L2 hits
    and misses, to give us an idea of how much the performance should degrade depending on the
    oversubscription ratio.
  */
  if (!index_.has_value()) { return; }
  auto& index = index_.value();
  size_t free_mem, total_mem;
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_mem, &total_mem));
  size_t avg_mem_per_cluster =
    raft::make_mdspan<uint8_t, size_t>(
      nullptr,
      neighbors::ivf_pq::list_spec<size_t, IdxT>(index.pq_bits(), index.pq_dim(), false)
        .make_list_extents(div_rounding_up_safe<size_t>(index.size(), index.n_lists())))
      .size();
  size_t total_probes = div_rounding_up_safe(total_mem, avg_mem_per_cluster);
  // set the number of queries to cover the whole GPU mem two times.
  size_t num_queries = 2 * div_rounding_up_safe<size_t>(total_probes, search_params_.n_probes);
  RAFT_LOG_INFO("Simulating index use (n_queries = %u, n_probes = %u)...",
                num_queries,
                search_params_.n_probes);
  int k = 1;
  raft::device_resources res;
  rmm::device_uvector<uint32_t> cluster_ids(num_queries, stream);
  auto queries = raft::make_device_matrix<float, uint32_t>(res, num_queries, index.dim());
  rng_.uniformInt<uint32_t>(cluster_ids.data(), num_queries, 0, index.n_lists(), stream);
  RAFT_CUDA_TRY(cudaMemcpy2DAsync(queries.data_handle(),
                                  sizeof(float) * index.dim(),
                                  index.centers().data_handle(),
                                  sizeof(float) * index.dim_ext(),
                                  sizeof(float) * index.dim(),
                                  num_queries,
                                  cudaMemcpyDefault,
                                  stream));
  auto neighbors = raft::make_device_matrix<IdxT, uint32_t>(res, num_queries, k);
  auto distances = raft::make_device_matrix<float, uint32_t>(res, num_queries, k);
  runtime::neighbors::ivf_pq::search(
    res, search_params_, index, queries.view(), neighbors.view(), distances.view());
}

}  // namespace raft::bench::ann
