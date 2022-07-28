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

#pragma once

#include "detail/ivf_flat_build.cuh"
#include "detail/ivf_flat_search.cuh"
#include "ivf_flat_types.hpp"

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

namespace raft::spatial::knn::ivf_flat {

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::spatial::knn;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_flat::build(handle, index_params, dataset, N, D);
 *   // use default search parameters
 *   ivf_flat::search_params search_params;
 *   // search K nearest neighbours for each of the N queries
 *   ivf_flat::search(handle, search_params, index, queries, N, K, out_inds, out_dists);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param handle
 * @param params configure the index building
 * @param[in] dataset a device pointer to a row-major matrix [n_rows, dim]
 * @param n_rows the number of samples
 * @param dim the dimensionality of the data
 *
 * @return the constructed ivf-flat index
 */
template <typename T, typename IdxT = uint32_t>
inline auto build(
  const handle_t& handle, const index_params& params, const T* dataset, IdxT n_rows, uint32_t dim)
  -> index<T, IdxT>
{
  return raft::spatial::knn::ivf_flat::detail::build(handle, params, dataset, n_rows, dim);
}

/**
 * @brief Build a new index containing the data of the original plus new extra vectors.
 *
 * Implementation note:
 *    The new data is clustered according to existing kmeans clusters, then the cluster
 *    centers are adjusted to match the newly labeled data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::spatial::knn;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset, N, D);
 *   // fill the index with the data
 *   auto index = ivf_flat::extend(handle, index_empty, dataset, nullptr, N);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param handle
 * @param orig_index original index
 * @param[in] new_vectors a device pointer to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices a device pointer to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `nullptr`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param n_rows the number of samples
 *
 * @return the constructed extended ivf-flat index
 */
template <typename T, typename IdxT>
inline auto extend(const handle_t& handle,
                   const index<T, IdxT>& orig_index,
                   const T* new_vectors,
                   const IdxT* new_indices,
                   IdxT n_rows) -> index<T, IdxT>
{
  return raft::spatial::knn::ivf_flat::detail::extend(
    handle, orig_index, new_vectors, new_indices, n_rows);
}

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [ivf_flat::build](#ivf_flat::build) documentation for a usage example.
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices
 *
 * @param handle
 * @param params configure the search
 * @param index ivf-flat constructed index
 * @param[in] queries a device pointer to a row-major matrix [n_queries, index->dim()]
 * @param n_queries the batch size
 * @param k the number of neighbors to find for each query.
 * @param[out] neighbors a device pointer to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device pointer to the distances to the selected neighbors [n_queries, k]
 * @param mr an optional memory resource to use across the searches (you can provide a large enough
 *           memory pool here to avoid memory allocations within search).
 */
template <typename T, typename IdxT>
inline void search(const handle_t& handle,
                   const search_params& params,
                   const index<T, IdxT>& index,
                   const T* queries,
                   uint32_t n_queries,
                   uint32_t k,
                   IdxT* neighbors,
                   float* distances,
                   rmm::mr::device_memory_resource* mr = nullptr)
{
  return raft::spatial::knn::ivf_flat::detail::search(
    handle, params, index, queries, n_queries, k, neighbors, distances, mr);
}

}  // namespace raft::spatial::knn::ivf_flat
