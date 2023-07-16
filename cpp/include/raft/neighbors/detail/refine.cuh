/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/neighbors/detail/ivf_flat_build.cuh>
#include <raft/neighbors/detail/ivf_flat_interleaved_scan.cuh>
#include <raft/neighbors/detail/ivf_flat_search.cuh>
#include <raft/spatial/knn/detail/ann_utils.cuh>

#include <cstdlib>
#include <omp.h>

#include <thrust/sequence.h>

namespace raft::neighbors::detail {

/** Checks whether the input data extents are compatible. */
template <typename ExtentsT>
void check_input(ExtentsT dataset,
                 ExtentsT queries,
                 ExtentsT candidates,
                 ExtentsT indices,
                 ExtentsT distances,
                 distance::DistanceType metric)
{
  auto n_queries = queries.extent(0);
  auto k         = distances.extent(1);

  RAFT_EXPECTS(k <= raft::matrix::detail::select::warpsort::kMaxCapacity,
               "k must be lest than topk::kMaxCapacity (%d).",
               raft::matrix::detail::select::warpsort::kMaxCapacity);

  RAFT_EXPECTS(indices.extent(0) == n_queries && distances.extent(0) == n_queries &&
                 candidates.extent(0) == n_queries,
               "Number of rows in output indices, distances and candidates matrices must be equal"
               " with the number of rows in search matrix. Expected %d, got %d, %d, and %d",
               static_cast<int>(n_queries),
               static_cast<int>(indices.extent(0)),
               static_cast<int>(distances.extent(0)),
               static_cast<int>(candidates.extent(0)));

  RAFT_EXPECTS(indices.extent(1) == k,
               "Number of columns in output indices and distances matrices must be equal to k");

  RAFT_EXPECTS(queries.extent(1) == dataset.extent(1),
               "Number of columns must be equal for dataset and queries");

  RAFT_EXPECTS(candidates.extent(1) >= k,
               "Number of neighbor candidates must not be smaller than k (%d vs %d)",
               static_cast<int>(candidates.extent(1)),
               static_cast<int>(k));
}

/**
 * See raft::neighbors::refine for docs.
 */
template <typename IdxT, typename DataT, typename DistanceT, typename ExtentsT>
void refine_device(raft::resources const& handle,
                   raft::device_matrix_view<const DataT, ExtentsT, row_major> dataset,
                   raft::device_matrix_view<const DataT, ExtentsT, row_major> queries,
                   raft::device_matrix_view<const IdxT, ExtentsT, row_major> neighbor_candidates,
                   raft::device_matrix_view<IdxT, ExtentsT, row_major> indices,
                   raft::device_matrix_view<DistanceT, ExtentsT, row_major> distances,
                   distance::DistanceType metric = distance::DistanceType::L2Unexpanded)
{
  ExtentsT n_candidates = neighbor_candidates.extent(1);
  ExtentsT n_queries    = queries.extent(0);
  ExtentsT dim          = dataset.extent(1);
  auto k                = static_cast<uint32_t>(indices.extent(1));

  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "neighbors::refine(%zu, %u)", size_t(n_queries), uint32_t(n_candidates));

  check_input(dataset.extents(),
              queries.extents(),
              neighbor_candidates.extents(),
              indices.extents(),
              distances.extents(),
              metric);

  // The refinement search can be mapped to an IVF flat search:
  // - We consider that the candidate vectors form a cluster, separately for each query.
  // - In other words, the n_queries * n_candidates vectors form n_queries clusters, each with
  //   n_candidates elements.
  // - We consider that the coarse level search is already performed and assigned a single cluster
  //   to search for each query (the cluster formed from the corresponding candidates).
  // - We run IVF flat search with n_probes=1 to select the best k elements of the candidates.
  rmm::device_uvector<uint32_t> fake_coarse_idx(n_queries, resource::get_cuda_stream(handle));

  thrust::sequence(resource::get_thrust_policy(handle),
                   fake_coarse_idx.data(),
                   fake_coarse_idx.data() + n_queries);

  raft::neighbors::ivf_flat::index<DataT, IdxT> refinement_index(
    handle, metric, n_queries, false, true, dim);

  raft::neighbors::ivf_flat::detail::fill_refinement_index(handle,
                                                           &refinement_index,
                                                           dataset.data_handle(),
                                                           neighbor_candidates.data_handle(),
                                                           n_queries,
                                                           n_candidates);
  uint32_t grid_dim_x = 1;
  raft::neighbors::ivf_flat::detail::ivfflat_interleaved_scan<
    DataT,
    typename raft::spatial::knn::detail::utils::config<DataT>::value_t,
    IdxT>(refinement_index,
          queries.data_handle(),
          fake_coarse_idx.data(),
          static_cast<uint32_t>(n_queries),
          refinement_index.metric(),
          1,
          k,
          raft::distance::is_min_close(metric),
          indices.data_handle(),
          distances.data_handle(),
          grid_dim_x,
          resource::get_cuda_stream(handle));
}

template <typename DC, typename IdxT, typename DataT, typename DistanceT, typename ExtentsT>
[[gnu::optimize(3), gnu::optimize("tree-vectorize")]] void refine_host_impl(
  raft::host_matrix_view<const DataT, ExtentsT, row_major> dataset,
  raft::host_matrix_view<const DataT, ExtentsT, row_major> queries,
  raft::host_matrix_view<const IdxT, ExtentsT, row_major> neighbor_candidates,
  raft::host_matrix_view<IdxT, ExtentsT, row_major> indices,
  raft::host_matrix_view<DistanceT, ExtentsT, row_major> distances)
{
  size_t n_queries = queries.extent(0);
  size_t dim       = dataset.extent(1);
  size_t orig_k    = neighbor_candidates.extent(1);
  size_t refined_k = indices.extent(1);

  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "neighbors::refine_host(%zu, %zu -> %zu)", n_queries, orig_k, refined_k);

  auto suggested_n_threads = std::max(1, std::min(omp_get_num_procs(), omp_get_max_threads()));
  if (size_t(suggested_n_threads) > n_queries) { suggested_n_threads = n_queries; }

#pragma omp parallel num_threads(suggested_n_threads)
  {
    std::vector<std::tuple<DistanceT, IdxT>> refined_pairs(orig_k);
    for (size_t i = omp_get_thread_num(); i < n_queries; i += omp_get_num_threads()) {
      // Compute the refined distance using original dataset vectors
      const DataT* query = queries.data_handle() + dim * i;
      for (size_t j = 0; j < orig_k; j++) {
        IdxT id            = neighbor_candidates(i, j);
        const DataT* row   = dataset.data_handle() + dim * id;
        DistanceT distance = 0.0;
        for (size_t k = 0; k < dim; k++) {
          distance += DC::template eval<DistanceT>(query[k], row[k]);
        }
        refined_pairs[j] = std::make_tuple(distance, id);
      }
      // Sort the query neigbors by their refined distances
      std::sort(refined_pairs.begin(), refined_pairs.end());
      // Store forst refined_k neighbors
      for (size_t j = 0; j < refined_k; j++) {
        indices(i, j) = std::get<1>(refined_pairs[j]);
        if (distances.data_handle() != nullptr) {
          distances(i, j) = DC::template postprocess(std::get<0>(refined_pairs[j]));
        }
      }
    }
  }
}

struct distance_comp_l2 {
  template <typename DistanceT>
  static inline auto eval(const DistanceT& a, const DistanceT& b) -> DistanceT
  {
    auto d = a - b;
    return d * d;
  }
  template <typename DistanceT>
  static inline auto postprocess(const DistanceT& a) -> DistanceT
  {
    return a;
  }
};

struct distance_comp_inner {
  template <typename DistanceT>
  static inline auto eval(const DistanceT& a, const DistanceT& b) -> DistanceT
  {
    return -a * b;
  }
  template <typename DistanceT>
  static inline auto postprocess(const DistanceT& a) -> DistanceT
  {
    return -a;
  }
};

/**
 * Naive CPU implementation of refine operation
 *
 * All pointers are expected to be accessible on the host.
 */
template <typename IdxT, typename DataT, typename DistanceT, typename ExtentsT>
[[gnu::optimize(3), gnu::optimize("tree-vectorize")]] void refine_host(
  raft::host_matrix_view<const DataT, ExtentsT, row_major> dataset,
  raft::host_matrix_view<const DataT, ExtentsT, row_major> queries,
  raft::host_matrix_view<const IdxT, ExtentsT, row_major> neighbor_candidates,
  raft::host_matrix_view<IdxT, ExtentsT, row_major> indices,
  raft::host_matrix_view<DistanceT, ExtentsT, row_major> distances,
  distance::DistanceType metric = distance::DistanceType::L2Unexpanded)
{
  check_input(dataset.extents(),
              queries.extents(),
              neighbor_candidates.extents(),
              indices.extents(),
              distances.extents(),
              metric);

  switch (metric) {
    case raft::distance::DistanceType::L2Expanded:
      return refine_host_impl<distance_comp_l2>(
        dataset, queries, neighbor_candidates, indices, distances);
    case raft::distance::DistanceType::InnerProduct:
      return refine_host_impl<distance_comp_inner>(
        dataset, queries, neighbor_candidates, indices, distances);
    default: throw raft::logic_error("Unsupported metric");
  }
}

}  // namespace raft::neighbors::detail
