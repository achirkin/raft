/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "bitonic.hpp"
#include "compute_distance.hpp"
#include "device_common.hpp"
#include "hashmap.hpp"
#include "search_plan.cuh"
#include "topk_by_radix.cuh"
#include "topk_for_cagra/topk_core.cuh"  // TODO replace with raft topk
#include "utils.hpp"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/sample_filter_types.hpp>
#include <raft/spatial/knn/detail/ann_utils.cuh>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>  // RAFT_CUDA_TRY_NOT_THROW is used TODO(tfeher): consider moving this to cuda_rt_essentials.hpp

#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cuda/atomic>
#include <cuda/std/atomic>

#include <atomic_queue/atomic_queue.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

namespace raft::neighbors::cagra::detail {
namespace single_cta_search {

// #define _CLK_BREAKDOWN

template <unsigned TOPK_BY_BITONIC_SORT, class INDEX_T>
__device__ void pickup_next_parents(std::uint32_t* const terminate_flag,
                                    INDEX_T* const next_parent_indices,
                                    INDEX_T* const internal_topk_indices,
                                    const std::size_t internal_topk_size,
                                    const std::size_t dataset_size,
                                    const std::uint32_t search_width)
{
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
  // if (threadIdx.x >= 32) return;

  for (std::uint32_t i = threadIdx.x; i < search_width; i += 32) {
    next_parent_indices[i] = utils::get_max_value<INDEX_T>();
  }
  std::uint32_t itopk_max = internal_topk_size;
  if (itopk_max % 32) { itopk_max += 32 - (itopk_max % 32); }
  std::uint32_t num_new_parents = 0;
  for (std::uint32_t j = threadIdx.x; j < itopk_max; j += 32) {
    std::uint32_t jj = j;
    if (TOPK_BY_BITONIC_SORT) { jj = device::swizzling(j); }
    INDEX_T index;
    int new_parent = 0;
    if (j < internal_topk_size) {
      index = internal_topk_indices[jj];
      if ((index & index_msb_1_mask) == 0) {  // check if most significant bit is set
        new_parent = 1;
      }
    }
    const std::uint32_t ballot_mask = __ballot_sync(0xffffffff, new_parent);
    if (new_parent) {
      const auto i = __popc(ballot_mask & ((1 << threadIdx.x) - 1)) + num_new_parents;
      if (i < search_width) {
        next_parent_indices[i] = jj;
        // set most significant bit as used node
        internal_topk_indices[jj] |= index_msb_1_mask;
      }
    }
    num_new_parents += __popc(ballot_mask);
    if (num_new_parents >= search_width) { break; }
  }
  if (threadIdx.x == 0 && (num_new_parents == 0)) { *terminate_flag = 1; }
}

template <unsigned MAX_CANDIDATES, class IdxT = void>
__device__ inline void topk_by_bitonic_sort_1st(float* candidate_distances,  // [num_candidates]
                                                IdxT* candidate_indices,     // [num_candidates]
                                                const std::uint32_t num_candidates,
                                                const std::uint32_t num_itopk,
                                                unsigned MULTI_WARPS = 0)
{
  const unsigned lane_id = threadIdx.x % 32;
  const unsigned warp_id = threadIdx.x / 32;
  if (MULTI_WARPS == 0) {
    if (warp_id > 0) { return; }
    constexpr unsigned N = (MAX_CANDIDATES + 31) / 32;
    float key[N];
    IdxT val[N];
    /* Candidates -> Reg */
    for (unsigned i = 0; i < N; i++) {
      unsigned j = lane_id + (32 * i);
      if (j < num_candidates) {
        key[i] = candidate_distances[j];
        val[i] = candidate_indices[j];
      } else {
        key[i] = utils::get_max_value<float>();
        val[i] = utils::get_max_value<IdxT>();
      }
    }
    /* Sort */
    bitonic::warp_sort<float, IdxT, N>(key, val);
    /* Reg -> Temp_itopk */
    for (unsigned i = 0; i < N; i++) {
      unsigned j = (N * lane_id) + i;
      if (j < num_candidates && j < num_itopk) {
        candidate_distances[device::swizzling(j)] = key[i];
        candidate_indices[device::swizzling(j)]   = val[i];
      }
    }
  } else {
    // Use two warps (64 threads)
    constexpr unsigned max_candidates_per_warp = (MAX_CANDIDATES + 1) / 2;
    constexpr unsigned N                       = (max_candidates_per_warp + 31) / 32;
    float key[N];
    IdxT val[N];
    if (warp_id < 2) {
      /* Candidates -> Reg */
      for (unsigned i = 0; i < N; i++) {
        unsigned jl = lane_id + (32 * i);
        unsigned j  = jl + (max_candidates_per_warp * warp_id);
        if (j < num_candidates) {
          key[i] = candidate_distances[j];
          val[i] = candidate_indices[j];
        } else {
          key[i] = utils::get_max_value<float>();
          val[i] = utils::get_max_value<IdxT>();
        }
      }
      /* Sort */
      bitonic::warp_sort<float, IdxT, N>(key, val);
      /* Reg -> Temp_candidates */
      for (unsigned i = 0; i < N; i++) {
        unsigned jl = (N * lane_id) + i;
        unsigned j  = jl + (max_candidates_per_warp * warp_id);
        if (j < num_candidates && jl < num_itopk) {
          candidate_distances[device::swizzling(j)] = key[i];
          candidate_indices[device::swizzling(j)]   = val[i];
        }
      }
    }
    __syncthreads();

    unsigned num_warps_used = (num_itopk + max_candidates_per_warp - 1) / max_candidates_per_warp;
    if (warp_id < num_warps_used) {
      /* Temp_candidates -> Reg */
      for (unsigned i = 0; i < N; i++) {
        unsigned jl = (N * lane_id) + i;
        unsigned kl = max_candidates_per_warp - 1 - jl;
        unsigned j  = jl + (max_candidates_per_warp * warp_id);
        unsigned k  = MAX_CANDIDATES - 1 - j;
        if (j >= num_candidates || k >= num_candidates || kl >= num_itopk) continue;
        float temp_key = candidate_distances[device::swizzling(k)];
        if (key[i] == temp_key) continue;
        if ((warp_id == 0) == (key[i] > temp_key)) {
          key[i] = temp_key;
          val[i] = candidate_indices[device::swizzling(k)];
        }
      }
    }
    if (num_warps_used > 1) { __syncthreads(); }
    if (warp_id < num_warps_used) {
      /* Merge */
      bitonic::warp_merge<float, IdxT, N>(key, val, 32);
      /* Reg -> Temp_itopk */
      for (unsigned i = 0; i < N; i++) {
        unsigned jl = (N * lane_id) + i;
        unsigned j  = jl + (max_candidates_per_warp * warp_id);
        if (j < num_candidates && j < num_itopk) {
          candidate_distances[device::swizzling(j)] = key[i];
          candidate_indices[device::swizzling(j)]   = val[i];
        }
      }
    }
    if (num_warps_used > 1) { __syncthreads(); }
  }
}

template <unsigned MAX_ITOPK, class IdxT = void>
__device__ inline void topk_by_bitonic_sort_2nd(float* itopk_distances,  // [num_itopk]
                                                IdxT* itopk_indices,     // [num_itopk]
                                                const std::uint32_t num_itopk,
                                                float* candidate_distances,  // [num_candidates]
                                                IdxT* candidate_indices,     // [num_candidates]
                                                const std::uint32_t num_candidates,
                                                std::uint32_t* work_buf,
                                                const bool first,
                                                unsigned MULTI_WARPS = 0)
{
  const unsigned lane_id = threadIdx.x % 32;
  const unsigned warp_id = threadIdx.x / 32;
  if (MULTI_WARPS == 0) {
    if (warp_id > 0) { return; }
    constexpr unsigned N = (MAX_ITOPK + 31) / 32;
    float key[N];
    IdxT val[N];
    if (first) {
      /* Load itopk results */
      for (unsigned i = 0; i < N; i++) {
        unsigned j = lane_id + (32 * i);
        if (j < num_itopk) {
          key[i] = itopk_distances[j];
          val[i] = itopk_indices[j];
        } else {
          key[i] = utils::get_max_value<float>();
          val[i] = utils::get_max_value<IdxT>();
        }
      }
      /* Warp Sort */
      bitonic::warp_sort<float, IdxT, N>(key, val);
    } else {
      /* Load itopk results */
      for (unsigned i = 0; i < N; i++) {
        unsigned j = (N * lane_id) + i;
        if (j < num_itopk) {
          key[i] = itopk_distances[device::swizzling(j)];
          val[i] = itopk_indices[device::swizzling(j)];
        } else {
          key[i] = utils::get_max_value<float>();
          val[i] = utils::get_max_value<IdxT>();
        }
      }
    }
    /* Merge candidates */
    for (unsigned i = 0; i < N; i++) {
      unsigned j = (N * lane_id) + i;  // [0:MAX_ITOPK-1]
      unsigned k = MAX_ITOPK - 1 - j;
      if (k >= num_itopk || k >= num_candidates) continue;
      float candidate_key = candidate_distances[device::swizzling(k)];
      if (key[i] > candidate_key) {
        key[i] = candidate_key;
        val[i] = candidate_indices[device::swizzling(k)];
      }
    }
    /* Warp Merge */
    bitonic::warp_merge<float, IdxT, N>(key, val, 32);
    /* Store new itopk results */
    for (unsigned i = 0; i < N; i++) {
      unsigned j = (N * lane_id) + i;
      if (j < num_itopk) {
        itopk_distances[device::swizzling(j)] = key[i];
        itopk_indices[device::swizzling(j)]   = val[i];
      }
    }
  } else {
    // Use two warps (64 threads) or more
    constexpr unsigned max_itopk_per_warp = (MAX_ITOPK + 1) / 2;
    constexpr unsigned N                  = (max_itopk_per_warp + 31) / 32;
    float key[N];
    IdxT val[N];
    if (first) {
      /* Load itop results (not sorted) */
      if (warp_id < 2) {
        for (unsigned i = 0; i < N; i++) {
          unsigned j = lane_id + (32 * i) + (max_itopk_per_warp * warp_id);
          if (j < num_itopk) {
            key[i] = itopk_distances[j];
            val[i] = itopk_indices[j];
          } else {
            key[i] = utils::get_max_value<float>();
            val[i] = utils::get_max_value<IdxT>();
          }
        }
        /* Warp Sort */
        bitonic::warp_sort<float, IdxT, N>(key, val);
        /* Store intermedidate results */
        for (unsigned i = 0; i < N; i++) {
          unsigned j = (N * threadIdx.x) + i;
          if (j >= num_itopk) continue;
          itopk_distances[device::swizzling(j)] = key[i];
          itopk_indices[device::swizzling(j)]   = val[i];
        }
      }
      __syncthreads();
      if (warp_id < 2) {
        /* Load intermedidate results */
        for (unsigned i = 0; i < N; i++) {
          unsigned j = (N * threadIdx.x) + i;
          unsigned k = MAX_ITOPK - 1 - j;
          if (k >= num_itopk) continue;
          float temp_key = itopk_distances[device::swizzling(k)];
          if (key[i] == temp_key) continue;
          if ((warp_id == 0) == (key[i] > temp_key)) {
            key[i] = temp_key;
            val[i] = itopk_indices[device::swizzling(k)];
          }
        }
        /* Warp Merge */
        bitonic::warp_merge<float, IdxT, N>(key, val, 32);
      }
      __syncthreads();
      /* Store itopk results (sorted) */
      if (warp_id < 2) {
        for (unsigned i = 0; i < N; i++) {
          unsigned j = (N * threadIdx.x) + i;
          if (j >= num_itopk) continue;
          itopk_distances[device::swizzling(j)] = key[i];
          itopk_indices[device::swizzling(j)]   = val[i];
        }
      }
    }
    const uint32_t num_itopk_div2 = num_itopk / 2;
    if (threadIdx.x < 3) {
      // work_buf is used to obtain turning points in 1st and 2nd half of itopk afer merge.
      work_buf[threadIdx.x] = num_itopk_div2;
    }
    __syncthreads();

    // Merge candidates (using whole threads)
    for (unsigned k = threadIdx.x; k < min(num_candidates, num_itopk); k += blockDim.x) {
      const unsigned j          = num_itopk - 1 - k;
      const float itopk_key     = itopk_distances[device::swizzling(j)];
      const float candidate_key = candidate_distances[device::swizzling(k)];
      if (itopk_key > candidate_key) {
        itopk_distances[device::swizzling(j)] = candidate_key;
        itopk_indices[device::swizzling(j)]   = candidate_indices[device::swizzling(k)];
        if (j < num_itopk_div2) {
          atomicMin(work_buf + 2, j);
        } else {
          atomicMin(work_buf + 1, j - num_itopk_div2);
        }
      }
    }
    __syncthreads();

    // Merge 1st and 2nd half of itopk (using whole threads)
    for (unsigned j = threadIdx.x; j < num_itopk_div2; j += blockDim.x) {
      const unsigned k = j + num_itopk_div2;
      float key_0      = itopk_distances[device::swizzling(j)];
      float key_1      = itopk_distances[device::swizzling(k)];
      if (key_0 > key_1) {
        itopk_distances[device::swizzling(j)] = key_1;
        itopk_distances[device::swizzling(k)] = key_0;
        IdxT val_0                            = itopk_indices[device::swizzling(j)];
        IdxT val_1                            = itopk_indices[device::swizzling(k)];
        itopk_indices[device::swizzling(j)]   = val_1;
        itopk_indices[device::swizzling(k)]   = val_0;
        atomicMin(work_buf + 0, j);
      }
    }
    if (threadIdx.x == blockDim.x - 1) {
      if (work_buf[2] < num_itopk_div2) { work_buf[1] = work_buf[2]; }
    }
    __syncthreads();
    // if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
    //     RAFT_LOG_DEBUG( "work_buf: %u, %u, %u\n", work_buf[0], work_buf[1], work_buf[2] );
    // }

    // Warp-0 merges 1st half of itopk, warp-1 does 2nd half.
    if (warp_id < 2) {
      // Load intermedidate itopk results
      const uint32_t turning_point = work_buf[warp_id];  // turning_point <= num_itopk_div2
      for (unsigned i = 0; i < N; i++) {
        unsigned k = num_itopk;
        unsigned j = (N * lane_id) + i;
        if (j < turning_point) {
          k = j + (num_itopk_div2 * warp_id);
        } else if (j >= (MAX_ITOPK / 2 - num_itopk_div2)) {
          j -= (MAX_ITOPK / 2 - num_itopk_div2);
          if ((turning_point <= j) && (j < num_itopk_div2)) { k = j + (num_itopk_div2 * warp_id); }
        }
        if (k < num_itopk) {
          key[i] = itopk_distances[device::swizzling(k)];
          val[i] = itopk_indices[device::swizzling(k)];
        } else {
          key[i] = utils::get_max_value<float>();
          val[i] = utils::get_max_value<IdxT>();
        }
      }
      /* Warp Merge */
      bitonic::warp_merge<float, IdxT, N>(key, val, 32);
      /* Store new itopk results */
      for (unsigned i = 0; i < N; i++) {
        const unsigned j = (N * lane_id) + i;
        if (j < num_itopk_div2) {
          unsigned k                            = j + (num_itopk_div2 * warp_id);
          itopk_distances[device::swizzling(k)] = key[i];
          itopk_indices[device::swizzling(k)]   = val[i];
        }
      }
    }
  }
}

template <unsigned MAX_ITOPK,
          unsigned MAX_CANDIDATES,
          class IdxT>
__device__ void topk_by_bitonic_sort(float* itopk_distances,  // [num_itopk]
                                     IdxT* itopk_indices,     // [num_itopk]
                                     const std::uint32_t num_itopk,
                                     float* candidate_distances,  // [num_candidates]
                                     IdxT* candidate_indices,     // [num_candidates]
                                     const std::uint32_t num_candidates,
                                     std::uint32_t* work_buf,
                                     const bool first,
                                     const unsigned MULTI_WARPS_1,
                                     const unsigned MULTI_WARPS_2)
{
  // The results in candidate_distances/indices are sorted by bitonic sort.
  topk_by_bitonic_sort_1st<MAX_CANDIDATES, IdxT>(
    candidate_distances, candidate_indices, num_candidates, num_itopk, MULTI_WARPS_1);

  // The results sorted above are merged with the internal intermediate top-k
  // results so far using bitonic merge.
  topk_by_bitonic_sort_2nd<MAX_ITOPK, IdxT>(itopk_distances,
                                            itopk_indices,
                                            num_itopk,
                                            candidate_distances,
                                            candidate_indices,
                                            num_candidates,
                                            work_buf,
                                            first,
                                            MULTI_WARPS_2);
}

template <class INDEX_T>
__device__ inline void hashmap_restore(INDEX_T* const hashmap_ptr,
                                       const size_t hashmap_bitlen,
                                       const INDEX_T* itopk_indices,
                                       const uint32_t itopk_size,
                                       const uint32_t first_tid = 0)
{
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
  if (threadIdx.x < first_tid) return;
  for (unsigned i = threadIdx.x - first_tid; i < itopk_size; i += blockDim.x - first_tid) {
    auto key = itopk_indices[i] & ~index_msb_1_mask;  // clear most significant bit
    hashmap::insert(hashmap_ptr, hashmap_bitlen, key);
  }
}

template <class T, unsigned BLOCK_SIZE>
__device__ inline void set_value_device(T* const ptr, const T fill, const std::uint32_t count)
{
  for (std::uint32_t i = threadIdx.x; i < count; i += BLOCK_SIZE) {
    ptr[i] = fill;
  }
}

// One query one thread block
template <uint32_t TEAM_SIZE,
          uint32_t DATASET_BLOCK_DIM,
          unsigned MAX_ITOPK,
          unsigned MAX_CANDIDATES,
          unsigned TOPK_BY_BITONIC_SORT,
          class DATASET_DESCRIPTOR_T,
          class SAMPLE_FILTER_T>
__device__ void search_core(
  typename DATASET_DESCRIPTOR_T::INDEX_T* const result_indices_ptr,       // [num_queries, top_k]
  typename DATASET_DESCRIPTOR_T::DISTANCE_T* const result_distances_ptr,  // [num_queries, top_k]
  const std::uint32_t top_k,
  DATASET_DESCRIPTOR_T dataset_desc,
  const typename DATASET_DESCRIPTOR_T::DATA_T* const queries_ptr,  // [num_queries, dataset_dim]
  const typename DATASET_DESCRIPTOR_T::INDEX_T* const knn_graph,   // [dataset_size, graph_degree]
  const std::uint32_t graph_degree,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const typename DATASET_DESCRIPTOR_T::INDEX_T* seed_ptr,  // [num_queries, num_seeds]
  const uint32_t num_seeds,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const
    visited_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
  const std::uint32_t internal_topk,
  const std::uint32_t search_width,
  const std::uint32_t min_iteration,
  const std::uint32_t max_iteration,
  std::uint32_t* const num_executed_iterations,  // [num_queries]
  const std::uint32_t hash_bitlen,
  const std::uint32_t small_hash_bitlen,
  const std::uint32_t small_hash_reset_interval,
  const std::uint32_t query_id,
  SAMPLE_FILTER_T sample_filter,
  raft::distance::DistanceType metric)
{
  using LOAD_T = device::LOAD_128BIT_T;

  using DATA_T     = typename DATASET_DESCRIPTOR_T::DATA_T;
  using INDEX_T    = typename DATASET_DESCRIPTOR_T::INDEX_T;
  using DISTANCE_T = typename DATASET_DESCRIPTOR_T::DISTANCE_T;
  using QUERY_T    = typename DATASET_DESCRIPTOR_T::QUERY_T;

#ifdef _CLK_BREAKDOWN
  std::uint64_t clk_init                 = 0;
  std::uint64_t clk_compute_1st_distance = 0;
  std::uint64_t clk_topk                 = 0;
  std::uint64_t clk_reset_hash           = 0;
  std::uint64_t clk_pickup_parents       = 0;
  std::uint64_t clk_restore_hash         = 0;
  std::uint64_t clk_compute_distance     = 0;
  std::uint64_t clk_start;
#define _CLK_START() clk_start = clock64()
#define _CLK_REC(V)  V += clock64() - clk_start;
#else
#define _CLK_START()
#define _CLK_REC(V)
#endif
  _CLK_START();

  extern __shared__ std::uint32_t smem[];

  // Layout of result_buffer
  // +----------------------+------------------------------+---------+
  // | internal_top_k       | neighbors of internal_top_k  | padding |
  // | <internal_topk_size> | <search_width * graph_degree> | upto 32 |
  // +----------------------+------------------------------+---------+
  // |<---             result_buffer_size              --->|
  std::uint32_t result_buffer_size    = internal_topk + (search_width * graph_degree);
  std::uint32_t result_buffer_size_32 = result_buffer_size;
  if (result_buffer_size % 32) { result_buffer_size_32 += 32 - (result_buffer_size % 32); }
  const auto small_hash_size = hashmap::get_size(small_hash_bitlen);

  const auto query_smem_buffer_length =
    raft::ceildiv<uint32_t>(dataset_desc.dim, DATASET_BLOCK_DIM) * DATASET_BLOCK_DIM;
  auto query_buffer          = reinterpret_cast<QUERY_T*>(smem);
  auto result_indices_buffer = reinterpret_cast<INDEX_T*>(query_buffer + query_smem_buffer_length);
  auto result_distances_buffer =
    reinterpret_cast<DISTANCE_T*>(result_indices_buffer + result_buffer_size_32);
  auto visited_hash_buffer =
    reinterpret_cast<INDEX_T*>(result_distances_buffer + result_buffer_size_32);
  auto parent_list_buffer = reinterpret_cast<INDEX_T*>(visited_hash_buffer + small_hash_size);
  auto distance_work_buffer_ptr =
    reinterpret_cast<std::uint8_t*>(parent_list_buffer + search_width);
  auto topk_ws        = reinterpret_cast<std::uint32_t*>(distance_work_buffer_ptr +
                                                  DATASET_DESCRIPTOR_T::smem_buffer_size_in_byte);
  auto terminate_flag = reinterpret_cast<std::uint32_t*>(topk_ws + 3);
  auto smem_work_ptr  = reinterpret_cast<std::uint32_t*>(terminate_flag + 1);

  // Set smem working buffer for the distance calculation
  dataset_desc.set_smem_ptr(distance_work_buffer_ptr);

  // A flag for filtering.
  auto filter_flag = terminate_flag;

  const DATA_T* const query_ptr = queries_ptr + query_id * dataset_desc.dim;
  dataset_desc.template copy_query<DATASET_BLOCK_DIM>(
    query_ptr, query_buffer, query_smem_buffer_length);

  if (threadIdx.x == 0) {
    terminate_flag[0] = 0;
    topk_ws[0]        = ~0u;
  }

  // Init hashmap
  INDEX_T* local_visited_hashmap_ptr;
  if (small_hash_bitlen) {
    local_visited_hashmap_ptr = visited_hash_buffer;
  } else {
    local_visited_hashmap_ptr = visited_hashmap_ptr + (hashmap::get_size(hash_bitlen) * blockIdx.y);
  }
  hashmap::init(local_visited_hashmap_ptr, hash_bitlen, 0);
  __syncthreads();
  _CLK_REC(clk_init);

  // compute distance to randomly selecting nodes
  _CLK_START();
  const INDEX_T* const local_seed_ptr = seed_ptr ? seed_ptr + (num_seeds * query_id) : nullptr;
  device::compute_distance_to_random_nodes<TEAM_SIZE, DATASET_BLOCK_DIM>(result_indices_buffer,
                                                                         result_distances_buffer,
                                                                         query_buffer,
                                                                         dataset_desc,
                                                                         result_buffer_size,
                                                                         num_distilation,
                                                                         rand_xor_mask,
                                                                         local_seed_ptr,
                                                                         num_seeds,
                                                                         local_visited_hashmap_ptr,
                                                                         hash_bitlen,
                                                                         metric);
  __syncthreads();
  _CLK_REC(clk_compute_1st_distance);

  std::uint32_t iter = 0;
  while (1) {
    // sort
    if constexpr (TOPK_BY_BITONIC_SORT) {
      // [Notice]
      // It is good to use multiple warps in topk_by_bitonic_sort() when
      // batch size is small (short-latency), but it might not be always good
      // when batch size is large (high-throughput).
      // topk_by_bitonic_sort() consists of two operations:
      // if MAX_CANDIDATES is greater than 128, the first operation uses two warps;
      // if MAX_ITOPK is greater than 256, the second operation used two warps.
      const unsigned multi_warps_1 = ((blockDim.x >= 64) && (MAX_CANDIDATES > 128)) ? 1 : 0;
      const unsigned multi_warps_2 = ((blockDim.x >= 64) && (MAX_ITOPK > 256)) ? 1 : 0;

      // reset small-hash table.
      if ((iter + 1) % small_hash_reset_interval == 0) {
        // Depending on the block size and the number of warps used in
        // topk_by_bitonic_sort(), determine which warps are used to reset
        // the small hash and whether they are performed in overlap with
        // topk_by_bitonic_sort().
        _CLK_START();
        unsigned hash_start_tid;
        if (blockDim.x == 32) {
          hash_start_tid = 0;
        } else if (blockDim.x == 64) {
          if (multi_warps_1 || multi_warps_2) {
            hash_start_tid = 0;
          } else {
            hash_start_tid = 32;
          }
        } else {
          if (multi_warps_1 || multi_warps_2) {
            hash_start_tid = 64;
          } else {
            hash_start_tid = 32;
          }
        }
        hashmap::init(local_visited_hashmap_ptr, hash_bitlen, hash_start_tid);
        _CLK_REC(clk_reset_hash);
      }

      // topk with bitonic sort
      _CLK_START();
      if (std::is_same<SAMPLE_FILTER_T,
                       raft::neighbors::filtering::none_cagra_sample_filter>::value ||
          *filter_flag == 0) {
        topk_by_bitonic_sort<MAX_ITOPK, MAX_CANDIDATES>(result_distances_buffer,
                                                        result_indices_buffer,
                                                        internal_topk,
                                                        result_distances_buffer + internal_topk,
                                                        result_indices_buffer + internal_topk,
                                                        search_width * graph_degree,
                                                        topk_ws,
                                                        (iter == 0),
                                                        multi_warps_1,
                                                        multi_warps_2);
        __syncthreads();
      } else {
        topk_by_bitonic_sort_1st<MAX_ITOPK + MAX_CANDIDATES>(
          result_distances_buffer,
          result_indices_buffer,
          internal_topk + search_width * graph_degree,
          internal_topk,
          false);
        if (threadIdx.x == 0) { *terminate_flag = 0; }
      }
      _CLK_REC(clk_topk);
    } else {
      _CLK_START();
      // topk with radix block sort
      topk_by_radix_sort<MAX_ITOPK, INDEX_T>{}(
        internal_topk,
        gridDim.x,
        result_buffer_size,
        reinterpret_cast<std::uint32_t*>(result_distances_buffer),
        result_indices_buffer,
        reinterpret_cast<std::uint32_t*>(result_distances_buffer),
        result_indices_buffer,
        nullptr,
        topk_ws,
        true,
        reinterpret_cast<std::uint32_t*>(smem_work_ptr));
      _CLK_REC(clk_topk);

      // reset small-hash table
      if ((iter + 1) % small_hash_reset_interval == 0) {
        _CLK_START();
        hashmap::init(local_visited_hashmap_ptr, hash_bitlen);
        _CLK_REC(clk_reset_hash);
      }
    }
    __syncthreads();

    if (iter + 1 == max_iteration) { break; }

    // pick up next parents
    if (threadIdx.x < 32) {
      _CLK_START();
      pickup_next_parents<TOPK_BY_BITONIC_SORT, INDEX_T>(terminate_flag,
                                                         parent_list_buffer,
                                                         result_indices_buffer,
                                                         internal_topk,
                                                         dataset_desc.size,
                                                         search_width);
      _CLK_REC(clk_pickup_parents);
    }

    // restore small-hash table by putting internal-topk indices in it
    if ((iter + 1) % small_hash_reset_interval == 0) {
      const unsigned first_tid = ((blockDim.x <= 32) ? 0 : 32);
      _CLK_START();
      hashmap_restore(
        local_visited_hashmap_ptr, hash_bitlen, result_indices_buffer, internal_topk, first_tid);
      _CLK_REC(clk_restore_hash);
    }
    __syncthreads();

    if (*terminate_flag && iter >= min_iteration) { break; }

    // compute the norms between child nodes and query node
    _CLK_START();
    constexpr unsigned max_n_frags = 8;
    device::compute_distance_to_child_nodes<TEAM_SIZE, DATASET_BLOCK_DIM, max_n_frags>(
      result_indices_buffer + internal_topk,
      result_distances_buffer + internal_topk,
      query_buffer,
      dataset_desc,
      knn_graph,
      graph_degree,
      local_visited_hashmap_ptr,
      hash_bitlen,
      parent_list_buffer,
      result_indices_buffer,
      search_width,
      metric);
    __syncthreads();
    _CLK_REC(clk_compute_distance);

    // Filtering
    if constexpr (!std::is_same<SAMPLE_FILTER_T,
                                raft::neighbors::filtering::none_cagra_sample_filter>::value) {
      if (threadIdx.x == 0) { *filter_flag = 0; }
      __syncthreads();

      constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
      const INDEX_T invalid_index        = utils::get_max_value<INDEX_T>();

      for (unsigned p = threadIdx.x; p < search_width; p += blockDim.x) {
        if (parent_list_buffer[p] != invalid_index) {
          const auto parent_id = result_indices_buffer[parent_list_buffer[p]] & ~index_msb_1_mask;
          if (!sample_filter(query_id, parent_id)) {
            // If the parent must not be in the resulting top-k list, remove from the parent list
            result_distances_buffer[parent_list_buffer[p]] = utils::get_max_value<DISTANCE_T>();
            result_indices_buffer[parent_list_buffer[p]]   = invalid_index;
            *filter_flag                                   = 1;
          }
        }
      }
      __syncthreads();
    }

    iter++;
  }

  // Post process for filtering
  if constexpr (!std::is_same<SAMPLE_FILTER_T,
                              raft::neighbors::filtering::none_cagra_sample_filter>::value) {
    constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
    const INDEX_T invalid_index        = utils::get_max_value<INDEX_T>();

    for (unsigned i = threadIdx.x; i < internal_topk + search_width * graph_degree;
         i += blockDim.x) {
      const auto node_id = result_indices_buffer[i] & ~index_msb_1_mask;
      if (node_id != (invalid_index & ~index_msb_1_mask) && !sample_filter(query_id, node_id)) {
        result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
        result_indices_buffer[i]   = invalid_index;
      }
    }

    __syncthreads();
    topk_by_bitonic_sort_1st<MAX_ITOPK + MAX_CANDIDATES>(
      result_distances_buffer,
      result_indices_buffer,
      internal_topk + search_width * graph_degree,
      top_k,
      false);
    __syncthreads();
  }

  for (std::uint32_t i = threadIdx.x; i < top_k; i += blockDim.x) {
    unsigned j  = i + (top_k * query_id);
    unsigned ii = i;
    if (TOPK_BY_BITONIC_SORT) { ii = device::swizzling(i); }
    if (result_distances_ptr != nullptr) { result_distances_ptr[j] = result_distances_buffer[ii]; }
    constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;

    result_indices_ptr[j] =
      result_indices_buffer[ii] & ~index_msb_1_mask;  // clear most significant bit
  }
  if (threadIdx.x == 0 && num_executed_iterations != nullptr) {
    num_executed_iterations[query_id] = iter + 1;
  }
#ifdef _CLK_BREAKDOWN
  if ((threadIdx.x == 0 || threadIdx.x == BLOCK_SIZE - 1) && ((query_id * 3) % gridDim.y < 3)) {
    RAFT_LOG_DEBUG(
      "query, %d, thread, %d"
      ", init, %d"
      ", 1st_distance, %lu"
      ", topk, %lu"
      ", reset_hash, %lu"
      ", pickup_parents, %lu"
      ", restore_hash, %lu"
      ", distance, %lu"
      "\n",
      query_id,
      threadIdx.x,
      clk_init,
      clk_compute_1st_distance,
      clk_topk,
      clk_reset_hash,
      clk_pickup_parents,
      clk_restore_hash,
      clk_compute_distance);
  }
#endif
}

template <uint32_t TEAM_SIZE,
          uint32_t DATASET_BLOCK_DIM,
          unsigned MAX_ITOPK,
          unsigned MAX_CANDIDATES,
          unsigned TOPK_BY_BITONIC_SORT,
          class DATASET_DESCRIPTOR_T,
          class SAMPLE_FILTER_T>
__launch_bounds__(1024, 1) RAFT_KERNEL search_kernel(
  typename DATASET_DESCRIPTOR_T::INDEX_T* const result_indices_ptr,       // [num_queries, top_k]
  typename DATASET_DESCRIPTOR_T::DISTANCE_T* const result_distances_ptr,  // [num_queries, top_k]
  const std::uint32_t top_k,
  DATASET_DESCRIPTOR_T dataset_desc,
  const typename DATASET_DESCRIPTOR_T::DATA_T* const queries_ptr,  // [num_queries, dataset_dim]
  const typename DATASET_DESCRIPTOR_T::INDEX_T* const knn_graph,   // [dataset_size, graph_degree]
  const std::uint32_t graph_degree,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const typename DATASET_DESCRIPTOR_T::INDEX_T* seed_ptr,  // [num_queries, num_seeds]
  const uint32_t num_seeds,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const
    visited_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
  const std::uint32_t internal_topk,
  const std::uint32_t search_width,
  const std::uint32_t min_iteration,
  const std::uint32_t max_iteration,
  std::uint32_t* const num_executed_iterations,  // [num_queries]
  const std::uint32_t hash_bitlen,
  const std::uint32_t small_hash_bitlen,
  const std::uint32_t small_hash_reset_interval,
  SAMPLE_FILTER_T sample_filter,
  raft::distance::DistanceType metric)
{
  const auto query_id = blockIdx.y;
  search_core<TEAM_SIZE,
              DATASET_BLOCK_DIM,
              MAX_ITOPK,
              MAX_CANDIDATES,
              TOPK_BY_BITONIC_SORT,
              DATASET_DESCRIPTOR_T,
              SAMPLE_FILTER_T>(result_indices_ptr,
                               result_distances_ptr,
                               top_k,
                               dataset_desc,
                               queries_ptr,
                               knn_graph,
                               graph_degree,
                               num_distilation,
                               rand_xor_mask,
                               seed_ptr,
                               num_seeds,
                               visited_hashmap_ptr,
                               internal_topk,
                               search_width,
                               min_iteration,
                               max_iteration,
                               num_executed_iterations,
                               hash_bitlen,
                               small_hash_bitlen,
                               small_hash_reset_interval,
                               query_id,
                               sample_filter,
                               metric);
}

// To make sure we avoid false sharing on both CPU and GPU, we enforce cache line size to the
// maximum of the two.
// This makes sync atomic significantly faster.
constexpr size_t kCacheLineBytes = 64;

constexpr uint32_t kMaxJobsNum              = 2048;
constexpr uint32_t kMaxWorkersNum           = 2048;
constexpr uint32_t kMaxWorkersPerThread     = 256;
constexpr uint32_t kSoftMaxWorkersPerThread = 16;

template <typename DATASET_DESCRIPTOR_T>
struct alignas(kCacheLineBytes) job_desc_t {
  using index_type    = typename DATASET_DESCRIPTOR_T::INDEX_T;
  using distance_type = typename DATASET_DESCRIPTOR_T::DISTANCE_T;
  using data_type     = typename DATASET_DESCRIPTOR_T::DATA_T;
  // The algorithm input parameters
  struct value_t {
    index_type* result_indices_ptr;       // [num_queries, top_k]
    distance_type* result_distances_ptr;  // [num_queries, top_k]
    const data_type* queries_ptr;         // [num_queries, dataset_dim]
    uint32_t top_k;
    uint32_t n_queries;
  };
  using blob_elem_type = uint4;
  constexpr static inline size_t kBlobSize =
    raft::div_rounding_up_safe(sizeof(value_t), sizeof(blob_elem_type));
  // Union facilitates loading the input by a warp in a single request
  union input_t {
    blob_elem_type blob[kBlobSize];  // NOLINT
    value_t value;
  } input;
  // Last thread triggers this flag.
  cuda::atomic<bool, cuda::thread_scope_system> completion_flag;
};

struct alignas(kCacheLineBytes) worker_handle_t {
  using handle_t = uint64_t;
  struct value_t {
    uint32_t desc_id;
    uint32_t query_id;
  };
  union data_t {
    handle_t handle;
    value_t value;
  };
  cuda::atomic<data_t, cuda::thread_scope_system> data;
};
static_assert(sizeof(worker_handle_t::value_t) == sizeof(worker_handle_t::handle_t));
static_assert(
  cuda::atomic<worker_handle_t::data_t, cuda::thread_scope_system>::is_always_lock_free);

constexpr worker_handle_t::handle_t kWaitForWork = std::numeric_limits<uint64_t>::max();
constexpr worker_handle_t::handle_t kNoMoreWork  = kWaitForWork - 1;

constexpr auto is_worker_busy(worker_handle_t::handle_t h) -> bool
{
  return (h != kWaitForWork) && (h != kNoMoreWork);
}

template <uint32_t TEAM_SIZE,
          uint32_t DATASET_BLOCK_DIM,
          unsigned MAX_ITOPK,
          unsigned MAX_CANDIDATES,
          unsigned TOPK_BY_BITONIC_SORT,
          class DATASET_DESCRIPTOR_T,
          class SAMPLE_FILTER_T>
__launch_bounds__(1024, 1) RAFT_KERNEL search_kernel_p(
  DATASET_DESCRIPTOR_T dataset_desc,
  worker_handle_t* worker_handles,
  job_desc_t<DATASET_DESCRIPTOR_T>* job_descriptors,
  uint32_t* completion_counters,
  const typename DATASET_DESCRIPTOR_T::INDEX_T* const knn_graph,  // [dataset_size, graph_degree]
  const std::uint32_t graph_degree,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const typename DATASET_DESCRIPTOR_T::INDEX_T* seed_ptr,  // [num_queries, num_seeds]
  const uint32_t num_seeds,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const
    visited_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
  const std::uint32_t internal_topk,
  const std::uint32_t search_width,
  const std::uint32_t min_iteration,
  const std::uint32_t max_iteration,
  std::uint32_t* const num_executed_iterations,  // [num_queries]
  const std::uint32_t hash_bitlen,
  const std::uint32_t small_hash_bitlen,
  const std::uint32_t small_hash_reset_interval,
  SAMPLE_FILTER_T sample_filter,
  raft::distance::DistanceType metric)
{
  using job_desc_type = job_desc_t<DATASET_DESCRIPTOR_T>;
  __shared__ typename job_desc_type::input_t job_descriptor;
  __shared__ worker_handle_t::data_t worker_data;

  auto& worker_handle = worker_handles[blockIdx.y].data;
  uint32_t job_ix;

  while (true) {
    // wait the writing phase
    if (threadIdx.x == 0) {
      worker_handle_t::data_t worker_data_local;
      do {
        worker_data_local = worker_handle.load(cuda::memory_order_relaxed);
      } while (worker_data_local.handle == kWaitForWork);
      job_ix = worker_data_local.value.desc_id;
      cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
      worker_data = worker_data_local;
    }
    if (threadIdx.x < WarpSize) {
      // Sync one warp and copy descriptor data
      static_assert(job_desc_type::kBlobSize <= WarpSize);
      job_ix = raft::shfl(job_ix, 0);
      if (threadIdx.x < job_desc_type::kBlobSize && job_ix < kMaxJobsNum) {
        job_descriptor.blob[threadIdx.x] = job_descriptors[job_ix].input.blob[threadIdx.x];
      }
    }
    __syncthreads();
    if (worker_data.handle == kNoMoreWork) { break; }
    if (threadIdx.x == 0) { worker_handle.store({kWaitForWork}, cuda::memory_order_relaxed); }

    // reading phase
    auto* result_indices_ptr   = job_descriptor.value.result_indices_ptr;
    auto* result_distances_ptr = job_descriptor.value.result_distances_ptr;
    auto* queries_ptr          = job_descriptor.value.queries_ptr;
    auto top_k                 = job_descriptor.value.top_k;
    auto n_queries             = job_descriptor.value.n_queries;
    auto query_id              = worker_data.value.query_id;

    // work phase
    search_core<TEAM_SIZE,
                DATASET_BLOCK_DIM,
                MAX_ITOPK,
                MAX_CANDIDATES,
                TOPK_BY_BITONIC_SORT,
                DATASET_DESCRIPTOR_T,
                SAMPLE_FILTER_T>(result_indices_ptr,
                                 result_distances_ptr,
                                 top_k,
                                 dataset_desc,
                                 queries_ptr,
                                 knn_graph,
                                 graph_degree,
                                 num_distilation,
                                 rand_xor_mask,
                                 seed_ptr,
                                 num_seeds,
                                 visited_hashmap_ptr,
                                 internal_topk,
                                 search_width,
                                 min_iteration,
                                 max_iteration,
                                 num_executed_iterations,
                                 hash_bitlen,
                                 small_hash_bitlen,
                                 small_hash_reset_interval,
                                 query_id,
                                 sample_filter,
                                 metric);

    // arrive to mark the end of the work phase
    __syncthreads();
    if (threadIdx.x == 0) {
      auto completed_count = atomicInc(completion_counters + job_ix, n_queries - 1) + 1;
      if (completed_count >= n_queries) {
        // we may need a memory fence here:
        //   - device - if the queries are accessed by the device
        //   - system - e.g. if we put them into managed/pinned memory.
        job_descriptors[job_ix].completion_flag.store(true, cuda::memory_order_relaxed);
      }
    }
  }
}

template <bool Persistent,
          uint32_t TEAM_SIZE,
          uint32_t DATASET_BLOCK_DIM,
          unsigned MAX_ITOPK,
          unsigned MAX_CANDIDATES,
          unsigned TOPK_BY_BITONIC_SORT,
          class DATASET_DESCRIPTOR_T,
          class SAMPLE_FILTER_T>
auto dispatch_kernel = []() {
  if constexpr (Persistent) {
    return search_kernel_p<TEAM_SIZE,
                           DATASET_BLOCK_DIM,
                           MAX_ITOPK,
                           MAX_CANDIDATES,
                           TOPK_BY_BITONIC_SORT,
                           DATASET_DESCRIPTOR_T,
                           SAMPLE_FILTER_T>;
  } else {
    return search_kernel<TEAM_SIZE,
                         DATASET_BLOCK_DIM,
                         MAX_ITOPK,
                         MAX_CANDIDATES,
                         TOPK_BY_BITONIC_SORT,
                         DATASET_DESCRIPTOR_T,
                         SAMPLE_FILTER_T>;
  }
}();

template <bool Persistent,
          uint32_t TEAM_SIZE,
          uint32_t DATASET_BLOCK_DIM,
          typename DATASET_DESCRIPTOR_T,
          typename SAMPLE_FILTER_T>
struct search_kernel_config {
  using kernel_t = decltype(dispatch_kernel<Persistent,
                                            TEAM_SIZE,
                                            DATASET_BLOCK_DIM,
                                            64,
                                            64,
                                            0,
                                            DATASET_DESCRIPTOR_T,
                                            SAMPLE_FILTER_T>);

  template <unsigned MAX_CANDIDATES, unsigned USE_BITONIC_SORT>
  static auto choose_search_kernel(unsigned itopk_size) -> kernel_t
  {
    if (itopk_size <= 64) {
      return dispatch_kernel<Persistent,
                             TEAM_SIZE,
                             DATASET_BLOCK_DIM,
                             64,
                             MAX_CANDIDATES,
                             USE_BITONIC_SORT,
                             DATASET_DESCRIPTOR_T,
                             SAMPLE_FILTER_T>;
    } else if (itopk_size <= 128) {
      return dispatch_kernel<Persistent,
                             TEAM_SIZE,
                             DATASET_BLOCK_DIM,
                             128,
                             MAX_CANDIDATES,
                             USE_BITONIC_SORT,
                             DATASET_DESCRIPTOR_T,
                             SAMPLE_FILTER_T>;
    } else if (itopk_size <= 256) {
      return dispatch_kernel<Persistent,
                             TEAM_SIZE,
                             DATASET_BLOCK_DIM,
                             256,
                             MAX_CANDIDATES,
                             USE_BITONIC_SORT,
                             DATASET_DESCRIPTOR_T,
                             SAMPLE_FILTER_T>;
    } else if (itopk_size <= 512) {
      return dispatch_kernel<Persistent,
                             TEAM_SIZE,
                             DATASET_BLOCK_DIM,
                             512,
                             MAX_CANDIDATES,
                             USE_BITONIC_SORT,
                             DATASET_DESCRIPTOR_T,
                             SAMPLE_FILTER_T>;
    }
    THROW("No kernel for parametels itopk_size %u, max_candidates %u", itopk_size, MAX_CANDIDATES);
  }

  static auto choose_itopk_and_mx_candidates(unsigned itopk_size,
                                             unsigned num_itopk_candidates,
                                             unsigned block_size) -> kernel_t
  {
    if (num_itopk_candidates <= 64) {
      // use bitonic sort based topk
      return choose_search_kernel<64, 1>(itopk_size);
    } else if (num_itopk_candidates <= 128) {
      return choose_search_kernel<128, 1>(itopk_size);
    } else if (num_itopk_candidates <= 256) {
      return choose_search_kernel<256, 1>(itopk_size);
    } else {
      // Radix-based topk is used
      constexpr unsigned max_candidates = 32;  // to avoid build failure
      if (itopk_size <= 256) {
        return dispatch_kernel<Persistent,
                               TEAM_SIZE,
                               DATASET_BLOCK_DIM,
                               256,
                               max_candidates,
                               0,
                               DATASET_DESCRIPTOR_T,
                               SAMPLE_FILTER_T>;
      } else if (itopk_size <= 512) {
        return dispatch_kernel<Persistent,
                               TEAM_SIZE,
                               DATASET_BLOCK_DIM,
                               512,
                               max_candidates,
                               0,
                               DATASET_DESCRIPTOR_T,
                               SAMPLE_FILTER_T>;
      }
    }
    THROW("No kernel for parametels itopk_size %u, num_itopk_candidates %u",
          itopk_size,
          num_itopk_candidates);
  }
};

/** Primitive fixed-size deque for single-threaded use. */
template <typename T>
struct local_deque_t {
  explicit local_deque_t(uint32_t size) : store_(size) {}

  [[nodiscard]] auto capacity() const -> uint32_t { return store_.size(); }
  [[nodiscard]] auto size() const -> uint32_t { return end_ - start_; }

  void push_back(T x) { store_[end_++ % capacity()] = x; }

  void push_front(T x)
  {
    if (start_ == 0) {
      start_ += capacity();
      end_ += capacity();
    }
    store_[--start_ % capacity()] = x;
  }

  // NB: non-blocking, unsafe functions
  auto pop_back() -> T { return store_[--end_ % capacity()]; }
  auto pop_front() -> T { return store_[start_++ % capacity()]; }

  auto try_push_back(T x) -> bool
  {
    if (size() >= capacity()) { return false; }
    push_back(x);
    return true;
  }

  auto try_push_front(T x) -> bool
  {
    if (size() >= capacity()) { return false; }
    push_front(x);
    return true;
  }

  auto try_pop_back(T& x) -> bool
  {
    if (start_ >= end_) { return false; }
    x = pop_back();
    return true;
  }

  auto try_pop_front(T& x) -> bool
  {
    if (start_ >= end_) { return false; }
    x = pop_front();
    return true;
  }

 private:
  std::vector<T> store_;
  uint32_t start_{0};
  uint32_t end_{0};
};

struct persistent_runner_base_t {
  using job_queue_type =
    atomic_queue::AtomicQueue<uint32_t, kMaxJobsNum, ~0u, false, true, false, false>;
  using worker_queue_type =
    atomic_queue::AtomicQueue<uint32_t, kMaxWorkersNum, ~0u, false, true, false, false>;
  rmm::mr::pinned_host_memory_resource worker_handles_mr;
  rmm::mr::pinned_host_memory_resource job_descriptor_mr;
  rmm::mr::cuda_memory_resource device_mr;
  cudaStream_t stream{};
  job_queue_type job_queue{};
  worker_queue_type worker_queue{};
  persistent_runner_base_t() : job_queue(), worker_queue()
  {
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  }
  virtual ~persistent_runner_base_t() noexcept { cudaStreamDestroy(stream); };
};

struct alignas(kCacheLineBytes) launcher_t {
  using job_queue_type           = persistent_runner_base_t::job_queue_type;
  using worker_queue_type        = persistent_runner_base_t::worker_queue_type;
  using pending_reads_queue_type = local_deque_t<uint32_t>;
  using completion_flag_type     = cuda::atomic<bool, cuda::thread_scope_system>;

  pending_reads_queue_type pending_reads;
  job_queue_type& job_ids;
  worker_queue_type& idle_worker_ids;
  worker_handle_t* worker_handles;
  uint32_t job_id;
  completion_flag_type* completion_flag;
  bool all_done = false;

  template <typename RecordWork>
  launcher_t(job_queue_type& job_ids,
             worker_queue_type& idle_worker_ids,
             worker_handle_t* worker_handles,
             uint32_t n_queries,
             RecordWork record_work)
    : pending_reads{std::min(n_queries, kMaxWorkersPerThread)},
      job_ids{job_ids},
      idle_worker_ids{idle_worker_ids},
      worker_handles{worker_handles},
      job_id{job_ids.pop()},
      completion_flag{record_work(job_id)}
  {
    // Submit all queries in the batch
    for (uint32_t i = 0; i < n_queries; i++) {
      uint32_t worker_id;
      while (!try_get_worker(worker_id)) {
        if (pending_reads.try_pop_front(worker_id)) {
          // TODO optimization: avoid the roundtrip through pending_worker_ids
          if (!try_return_worker(worker_id)) { pending_reads.push_front(worker_id); }
        } else {
          std::this_thread::yield();
        }
      }
      submit_query(worker_id, i);
      // Try to not hold too many workers in one thread
      if (i >= kSoftMaxWorkersPerThread && pending_reads.try_pop_front(worker_id)) {
        if (!try_return_worker(worker_id)) { pending_reads.push_front(worker_id); }
      }
    }
  }

  void submit_query(uint32_t worker_id, uint32_t query_id)
  {
    worker_handles[worker_id].data.store(worker_handle_t::data_t{.value = {job_id, query_id}},
                                         cuda::memory_order_relaxed);

    while (!pending_reads.try_push_back(worker_id)) {
      // The only reason pending_reads cannot push is that the job_queue is full.
      // It's local, so we must pop and wait for the returned worker to finish its work.
      auto pending_worker_id = pending_reads.pop_front();
      while (!try_return_worker(pending_worker_id)) {
        std::this_thread::yield();
      }
    }
  }

  /** Check if the worker has finished the work; if so, return it to the shared pool. */
  auto try_return_worker(uint32_t worker_id) -> bool
  {
    // Use the cached `all_done` - makes sense when called from the `wait()` routine.
    if (all_done ||
        !is_worker_busy(worker_handles[worker_id].data.load(cuda::memory_order_relaxed).handle)) {
      idle_worker_ids.push(worker_id);
      return true;
    } else {
      return false;
    }
  }

  /** Try get a free worker if any. */
  auto try_get_worker(uint32_t& worker_id) -> bool { return idle_worker_ids.try_pop(worker_id); }

  /** Check if all workers finished their work. */
  auto is_all_done()
  {
    // Cache the result of the check to avoid doing unnecessary atomic loads.
    if (all_done) { return true; }
    all_done = completion_flag->load(cuda::memory_order_relaxed);
    return all_done;
  }

  /** Wait for all work to finish and don't forget to return the workers to the shared pool. */
  void wait()
  {
    uint32_t worker_id;
    while (pending_reads.try_pop_front(worker_id)) {
      while (!try_return_worker(worker_id)) {
        if (!is_all_done()) { std::this_thread::yield(); }
      }
    }
    // terminal state, should be engaged only after the `pending_reads` is empty
    // and `queries_submitted == n_queries`
    while (!is_all_done()) {
      // Not sure if this improves the perf, but it does not seem to hurt it.
      // Let's hope this reduces cpu utilization
      std::this_thread::yield();
    }

    // Return the job descriptor
    job_ids.push(job_id);
  }
};

template <unsigned TEAM_SIZE,
          unsigned DATASET_BLOCK_DIM,
          typename DATASET_DESCRIPTOR_T,
          typename SAMPLE_FILTER_T>
struct alignas(kCacheLineBytes) persistent_runner_t : public persistent_runner_base_t {
  using index_type    = typename DATASET_DESCRIPTOR_T::INDEX_T;
  using distance_type = typename DATASET_DESCRIPTOR_T::DISTANCE_T;
  using data_type     = typename DATASET_DESCRIPTOR_T::DATA_T;
  using kernel_config_type =
    search_kernel_config<true, TEAM_SIZE, DATASET_BLOCK_DIM, DATASET_DESCRIPTOR_T, SAMPLE_FILTER_T>;
  using kernel_type   = typename kernel_config_type::kernel_t;
  using job_desc_type = job_desc_t<DATASET_DESCRIPTOR_T>;
  kernel_type kernel;
  uint32_t block_size;
  rmm::device_uvector<worker_handle_t> worker_handles;
  rmm::device_uvector<job_desc_type> job_descriptors;
  rmm::device_uvector<uint32_t> completion_counters;
  rmm::device_uvector<index_type> hashmap;
  std::atomic<uint64_t> heartbeat;

  persistent_runner_t(DATASET_DESCRIPTOR_T dataset_desc,
                      raft::device_matrix_view<const index_type, int64_t, row_major> graph,
                      uint32_t num_itopk_candidates,
                      uint32_t block_size,  //
                      uint32_t smem_size,
                      int64_t hash_bitlen,
                      size_t small_hash_bitlen,
                      size_t small_hash_reset_interval,
                      uint32_t num_random_samplings,
                      uint64_t rand_xor_mask,
                      uint32_t num_seeds,
                      size_t itopk_size,
                      size_t search_width,
                      size_t min_iterations,
                      size_t max_iterations,
                      SAMPLE_FILTER_T sample_filter,
                      raft::distance::DistanceType metric)
    : persistent_runner_base_t{},
      kernel{kernel_config_type::choose_itopk_and_mx_candidates(
        itopk_size, num_itopk_candidates, block_size)},
      block_size{block_size},
      worker_handles(0, stream, worker_handles_mr),
      job_descriptors(kMaxJobsNum, stream, job_descriptor_mr),
      completion_counters(kMaxJobsNum, stream, device_mr),
      hashmap(0, stream, device_mr)
  {
    // set kernel attributes same as in normal kernel
    RAFT_CUDA_TRY(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // set kernel launch parameters
    dim3 gs = calc_coop_grid_size(block_size, smem_size);
    dim3 bs(block_size, 1, 1);
    RAFT_LOG_DEBUG(
      "Launching persistent kernel with %u threads, %u block %u smem", bs.x, gs.y, smem_size);

    // initialize the job queue
    auto* completion_counters_ptr = completion_counters.data();
    auto* job_descriptors_ptr     = job_descriptors.data();
    for (uint32_t i = 0; i < kMaxJobsNum; i++) {
      auto& jd                = job_descriptors_ptr[i].input.value;
      jd.result_indices_ptr   = nullptr;
      jd.result_distances_ptr = nullptr;
      jd.queries_ptr          = nullptr;
      jd.top_k                = 0;
      jd.n_queries            = 0;
      job_descriptors_ptr[i].completion_flag.store(false);
      job_queue.push(i);
    }

    // initialize the worker queue
    worker_handles.resize(gs.y, stream);
    auto* worker_handles_ptr = worker_handles.data();
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    for (uint32_t i = 0; i < gs.y; i++) {
      worker_handles_ptr[i].data.store({kWaitForWork});
      worker_queue.push(i);
    }

    index_type* hashmap_ptr = nullptr;
    if (small_hash_bitlen == 0) {
      hashmap.resize(gs.y * hashmap::get_size(hash_bitlen), stream);
      hashmap_ptr = hashmap.data();
    }

    // launch the kernel
    auto* graph_ptr                   = graph.data_handle();
    uint32_t graph_degree             = graph.extent(1);
    uint32_t* num_executed_iterations = nullptr;  // optional arg [num_queries]
    const index_type* dev_seed_ptr    = nullptr;  // optional arg [num_queries, num_seeds]

    void* args[] =  // NOLINT
      {&dataset_desc,
       &worker_handles_ptr,
       &job_descriptors_ptr,
       &completion_counters_ptr,
       &graph_ptr,  // [dataset_size, graph_degree]
       &graph_degree,
       &num_random_samplings,
       &rand_xor_mask,
       &dev_seed_ptr,
       &num_seeds,
       &hashmap_ptr,  // visited_hashmap_ptr: [num_queries, 1 << hash_bitlen]
       &itopk_size,
       &search_width,
       &min_iterations,
       &max_iterations,
       &num_executed_iterations,
       &hash_bitlen,
       &small_hash_bitlen,
       &small_hash_reset_interval,
       &sample_filter,
       &metric};
    RAFT_CUDA_TRY(cudaLaunchCooperativeKernel<std::remove_pointer_t<kernel_type>>(
      kernel, gs, bs, args, smem_size, stream));
    RAFT_LOG_INFO(
      "Initialized the kernel in stream %zd; job_queue size = %u; worker_queue size = %u",
      int64_t((cudaStream_t)stream),
      job_queue.was_size(),
      worker_queue.was_size());
  }

  ~persistent_runner_t() noexcept override
  {
    auto whs = worker_handles.data();
    for (auto i = worker_handles.size(); i > 0; i--) {
      whs[worker_queue.pop()].data.store({kNoMoreWork}, cuda::memory_order_relaxed);
    }
    RAFT_CUDA_TRY_NO_THROW(cudaStreamSynchronize(stream));
    RAFT_LOG_INFO("Destroyed the persistent runner.");
  }

  void launch(index_type* result_indices_ptr,       // [num_queries, top_k]
              distance_type* result_distances_ptr,  // [num_queries, top_k]
              const data_type* queries_ptr,         // [num_queries, dataset_dim]
              uint32_t num_queries,
              uint32_t top_k)
  {
    // submit all queries
    launcher_t launcher{
      job_queue, worker_queue, worker_handles.data(), num_queries, [=](uint32_t job_ix) {
        auto& jd                = job_descriptors.data()[job_ix].input.value;
        auto cflag              = &job_descriptors.data()[job_ix].completion_flag;
        jd.result_indices_ptr   = result_indices_ptr;
        jd.result_distances_ptr = result_distances_ptr;
        jd.queries_ptr          = queries_ptr;
        jd.top_k                = top_k;
        jd.n_queries            = num_queries;
        cflag->store(false, cuda::memory_order_relaxed);
        cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
        return cflag;
      }};
    // update the keep-alive atomic in the meanwhile
    heartbeat.fetch_add(1, std::memory_order_relaxed);
    // wait for the results to arrive
    launcher.wait();
  }

  auto calc_coop_grid_size(uint32_t block_size, uint32_t smem_size) -> dim3
  {
    // We may need to run other kernels alongside this persistent kernel.
    // Leave a few SMs idle.
    // Note: even when we know there are no other kernels working at the same time, setting
    // kDeviceUsage to 1.0 surprisingly hurts performance.
    constexpr double kDeviceUsage = 0.9;

    // determine the grid size
    int ctas_per_sm = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor<kernel_type>(
      &ctas_per_sm, kernel, block_size, smem_size);
    int num_sm    = getMultiProcessorCount();
    auto n_blocks = static_cast<uint32_t>(kDeviceUsage * (ctas_per_sm * num_sm));
    if (n_blocks > kMaxWorkersNum) {
      RAFT_LOG_WARN("Limiting the grid size limit due to the size of the queue: %u -> %u",
                    n_blocks,
                    kMaxWorkersNum);
      n_blocks = kMaxWorkersNum;
    }

    return {1, n_blocks, 1};
  }
};

struct alignas(kCacheLineBytes) persistent_state {
  std::shared_ptr<persistent_runner_base_t> runner{nullptr};
  std::mutex lock;
};

inline persistent_state persistent{};

template <typename RunnerT, typename... Args>
auto create_runner(Args... args) -> std::shared_ptr<RunnerT>  // it's ok.. pass everything by values
{
  // NB: storing pointer-to-shared_ptr; otherwise, notify_one()/wait() do not seem to work.
  std::shared_ptr<RunnerT> runner_outer{nullptr};
  cuda::std::atomic_flag ready{};
  ready.clear(cuda::std::memory_order_relaxed);
  std::thread(
    [&runner_outer, &ready](Args... thread_args) {  // pass everything by values
      std::weak_ptr<RunnerT> runner_weak;
      {
        std::lock_guard<std::mutex> guard(persistent.lock);
        // Try to check the runner again:
        //   it may have been created by another thread since the last check
        runner_outer = std::dynamic_pointer_cast<RunnerT>(
          std::atomic_load_explicit(&persistent.runner, std::memory_order_relaxed));
        if (runner_outer) {
          runner_outer->heartbeat.fetch_add(1, std::memory_order_relaxed);
          ready.test_and_set(cuda::std::memory_order_release);
          ready.notify_one();
          return;
        }
        // Free the resources (if any) in advance
        std::atomic_store_explicit(&persistent.runner,
                                   std::shared_ptr<persistent_runner_base_t>{nullptr},
                                   std::memory_order_relaxed);
        runner_outer = std::make_shared<RunnerT>(thread_args...);
        runner_weak  = runner_outer;
        runner_outer->heartbeat.store(1, std::memory_order_relaxed);
        std::atomic_store_explicit(&persistent.runner,
                                   std::static_pointer_cast<persistent_runner_base_t>(runner_outer),
                                   std::memory_order_relaxed);
        ready.test_and_set(cuda::std::memory_order_release);
        ready.notify_one();
      }
      constexpr auto kInterval = std::chrono::milliseconds(500);
      size_t last_beat         = 0;
      while (true) {
        std::this_thread::sleep_for(kInterval);
        auto runner = runner_weak.lock();  // runner_weak is local - thread-safe
        if (!runner) {
          return;  // dead already
        }
        size_t this_beat = runner->heartbeat.load(std::memory_order_relaxed);
        if (this_beat == last_beat) {
          std::lock_guard<std::mutex> guard(persistent.lock);
          if (runner == std::atomic_load_explicit(
                          &persistent.runner,
                          std::memory_order_relaxed)) {  // compare pointers: this is thread-safe
            std::atomic_store_explicit(&persistent.runner,
                                       std::shared_ptr<persistent_runner_base_t>{nullptr},
                                       std::memory_order_relaxed);
          }
          return;
        }
        last_beat = this_beat;
      }
    },
    args...)
    .detach();
  ready.wait(false, cuda::std::memory_order_acquire);
  return runner_outer;
}

template <typename RunnerT, typename... Args>
auto get_runner_nocache(Args&&... args) -> std::shared_ptr<RunnerT>
{
  // We copy the shared pointer here, then using the copy is thread-safe.
  auto runner = std::dynamic_pointer_cast<RunnerT>(
    std::atomic_load_explicit(&persistent.runner, std::memory_order_relaxed));
  if (runner) { return runner; }
  return create_runner<RunnerT>(std::forward<Args>(args)...);
}

template <typename RunnerT, typename... Args>
auto get_runner(Args&&... args) -> std::shared_ptr<RunnerT>
{
  // Using a thread-local weak pointer allows us to avoid an extra atomic load of the persistent
  // runner shared pointer.
  static thread_local std::weak_ptr<RunnerT> weak;
  auto runner = weak.lock();
  if (runner) { return runner; }
  runner = get_runner_nocache<RunnerT, Args...>(std::forward<Args>(args)...);
  weak   = runner;
  return runner;
}

template <unsigned TEAM_SIZE,
          unsigned DATASET_BLOCK_DIM,
          typename DATASET_DESCRIPTOR_T,
          typename SAMPLE_FILTER_T>
void select_and_run(
  DATASET_DESCRIPTOR_T dataset_desc,
  raft::device_matrix_view<const typename DATASET_DESCRIPTOR_T::INDEX_T, int64_t, row_major> graph,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const topk_indices_ptr,       // [num_queries, topk]
  typename DATASET_DESCRIPTOR_T::DISTANCE_T* const topk_distances_ptr,  // [num_queries, topk]
  const typename DATASET_DESCRIPTOR_T::DATA_T* const queries_ptr,  // [num_queries, dataset_dim]
  const uint32_t num_queries,
  const typename DATASET_DESCRIPTOR_T::INDEX_T* dev_seed_ptr,  // [num_queries, num_seeds]
  uint32_t* const num_executed_iterations,                     // [num_queries,]
  uint32_t topk,
  uint32_t num_itopk_candidates,
  uint32_t block_size,  //
  uint32_t smem_size,
  int64_t hash_bitlen,
  typename DATASET_DESCRIPTOR_T::INDEX_T* hashmap_ptr,
  size_t small_hash_bitlen,
  size_t small_hash_reset_interval,
  uint32_t num_random_samplings,
  uint64_t rand_xor_mask,
  uint32_t num_seeds,
  size_t itopk_size,
  size_t search_width,
  size_t min_iterations,
  size_t max_iterations,
  SAMPLE_FILTER_T sample_filter,
  raft::distance::DistanceType metric,
  cudaStream_t stream)
{
  // hack: pass the 'is_persistent' flag in the highest bit of the `rand_xor_mask`
  //       to avoid changing the signature of `select_and_run` and updating all its
  //       instantiations...
  uint64_t pmask     = 0x8000000000000000LL;
  bool is_persistent = rand_xor_mask & pmask;
  rand_xor_mask &= ~pmask;
  if (is_persistent) {
    using runner_type =
      persistent_runner_t<TEAM_SIZE, DATASET_BLOCK_DIM, DATASET_DESCRIPTOR_T, SAMPLE_FILTER_T>;
    get_runner<runner_type>(dataset_desc,
                            graph,
                            num_itopk_candidates,
                            block_size,
                            smem_size,
                            hash_bitlen,
                            small_hash_bitlen,
                            small_hash_reset_interval,
                            num_random_samplings,
                            rand_xor_mask,
                            num_seeds,
                            itopk_size,
                            search_width,
                            min_iterations,
                            max_iterations,
                            sample_filter,
                            metric)
      ->launch(topk_indices_ptr, topk_distances_ptr, queries_ptr, num_queries, topk);
  } else {
    auto kernel =
      search_kernel_config<false,
                           TEAM_SIZE,
                           DATASET_BLOCK_DIM,
                           DATASET_DESCRIPTOR_T,
                           SAMPLE_FILTER_T>::choose_itopk_and_mx_candidates(itopk_size,
                                                                            num_itopk_candidates,
                                                                            block_size);
    RAFT_CUDA_TRY(cudaFuncSetAttribute(kernel,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       smem_size + DATASET_DESCRIPTOR_T::smem_buffer_size_in_byte));
    dim3 thread_dims(block_size, 1, 1);
    dim3 block_dims(1, num_queries, 1);
    RAFT_LOG_DEBUG(
      "Launching kernel with %u threads, %u block %u smem", block_size, num_queries, smem_size);
    kernel<<<block_dims, thread_dims, smem_size, stream>>>(topk_indices_ptr,
                                                           topk_distances_ptr,
                                                           topk,
                                                           dataset_desc,
                                                           queries_ptr,
                                                           graph.data_handle(),
                                                           graph.extent(1),
                                                           num_random_samplings,
                                                           rand_xor_mask,
                                                           dev_seed_ptr,
                                                           num_seeds,
                                                           hashmap_ptr,
                                                           itopk_size,
                                                           search_width,
                                                           min_iterations,
                                                           max_iterations,
                                                           num_executed_iterations,
                                                           hash_bitlen,
                                                           small_hash_bitlen,
                                                           small_hash_reset_interval,
                                                           sample_filter,
                                                           metric);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}
}  // namespace single_cta_search
}  // namespace raft::neighbors::cagra::detail
