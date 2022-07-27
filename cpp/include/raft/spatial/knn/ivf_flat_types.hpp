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

#include "common.hpp"

#include <raft/core/mdarray.hpp>
#include <raft/distance/distance_type.hpp>
#include <raft/integer_utils.h>

#include <optional>

namespace raft::spatial::knn::ivf_flat {

/** Size of the interleaved group (see `index::data` description). */
constexpr static uint32_t kIndexGroupSize = 32;

/**
 * @brief IVF-flat index.
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 */
template <typename T, typename IdxT>
struct index : knn::index {
  static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                "IdxT must be able to represent all values of uint32_t");

 public:
  /**
   * Vectorized load/store size in elements, determines the size of interleaved data chunks.
   *
   * TODO: in theory, we can lift this to the template parameter and keep it at hardware maximum
   * possible value by padding the `dim` of the data https://github.com/rapidsai/raft/issues/711
   */
  const uint32_t veclen;
  /** Distance metric used for clustering. */
  const raft::distance::DistanceType metric;
  /**
   * Inverted list data [size, dim].
   *
   * The data consists of the dataset rows, grouped by their labels (into clusters/lists).
   * Within each list (cluster), the data is grouped into blocks of `kGroupSize` interleaved
   * vectors. Note, the total index length is slightly larger than the source dataset length,
   * because each cluster is padded by `kGroupSize` elements.
   *
   * Interleaving pattern:
   * within groups of `kGroupSize` rows, the data is interleaved with the block size equal to
   * `veclen * sizeof(T)`. That is, a chunk of `veclen` consecutive components of one row is
   * followed by a chunk of the same size of the next row, and so on.
   *
   * __Example__: veclen = 2, dim = 6, kGroupSize = 32, list_size = 31
   *
   *     x[ 0, 0], x[ 0, 1], x[ 1, 0], x[ 1, 1], ... x[14, 0], x[14, 1], x[15, 0], x[15, 1],
   *     x[16, 0], x[16, 1], x[17, 0], x[17, 1], ... x[30, 0], x[30, 1],    -    ,    -    ,
   *     x[ 0, 2], x[ 0, 3], x[ 1, 2], x[ 1, 3], ... x[14, 2], x[14, 3], x[15, 2], x[15, 3],
   *     x[16, 2], x[16, 3], x[17, 2], x[17, 3], ... x[30, 2], x[30, 3],    -    ,    -    ,
   *     x[ 0, 4], x[ 0, 5], x[ 1, 4], x[ 1, 5], ... x[14, 4], x[14, 5], x[15, 4], x[15, 5],
   *     x[16, 4], x[16, 5], x[17, 4], x[17, 5], ... x[30, 4], x[30, 5],    -    ,    -    ,
   *
   */
  [[nodiscard]] inline auto data() const noexcept -> device_mdspan<const T, extent_2d, row_major>
  {
    return data_->view();
  }

  /** Inverted list indices: ids of items in the source data [size] */
  [[nodiscard]] inline auto indices() const noexcept
    -> device_mdspan<const IdxT, extent_1d, row_major>
  {
    return indices_->view();
  }
  /** Sizes of the lists (clusters) [n_lists] */
  [[nodiscard]] inline auto list_sizes() const noexcept
    -> device_mdspan<const uint32_t, extent_1d, row_major>
  {
    return list_sizes_->view();
  }
  /**
   * Offsets into the lists [n_lists + 1].
   * The last value contains the total length of the index.
   */
  [[nodiscard]] inline auto list_offsets() const noexcept
    -> device_mdspan<const IdxT, extent_1d, row_major>
  {
    return list_offsets_->view();
  }
  /** k-means cluster centers corresponding to the lists [n_lists, dim] */
  [[nodiscard]] inline auto centers() const noexcept
    -> device_mdspan<const float, extent_2d, row_major>
  {
    return centers_->view();
  }
  /**
   * (Optional) Precomputed norms of the `centers` w.r.t. the chosen distance metric [n_lists].
   *
   * NB: this may be empty if the index is empty or if the metric does not require the center norms
   * calculation.
   */
  [[nodiscard]] inline auto center_norms() const noexcept
    -> std::optional<device_mdspan<const float, extent_1d, row_major>>
  {
    if (center_norms_) {
      return std::make_optional<device_mdspan<const float, extent_1d, row_major>>(
        center_norms_->view());
    } else {
      return std::nullopt;
    }
  }

  /** Total length of the index. */
  [[nodiscard]] constexpr inline auto size() const noexcept -> IdxT
  {
    return static_cast<IdxT>(data_->extent(0));
  }
  /** Dimensionality of the data. */
  [[nodiscard]] constexpr inline auto dim() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(data_->extent(1));
  }
  /** Number of clusters/inverted lists. */
  [[nodiscard]] constexpr inline auto n_lists() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(centers_->extent(0));
  }

  // Don't allow copying the index for performance reasons (try avoiding copying data)
  index(const index&) = delete;
  index(index&& source) : knn::index(), veclen(source.veclen), metric(source.metric)
  {
    data_.swap(source.data_);
    indices_.swap(source.indices_);
    list_sizes_.swap(source.list_sizes_);
    list_offsets_.swap(source.list_offsets_);
    centers_.swap(source.centers_);
    center_norms_.swap(source.center_norms_);
    printf("moved to ivf_flat::index(%p): {", this);
    auto p = reinterpret_cast<uint8_t*>(this);
    for (size_t i = 0; i < sizeof(*this); i++) {
      printf("%02X", p[i]);
    }
    printf("}\n");
  }
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index& = delete;
  ~index()
  {
    printf("deleting ivf_flat::index(%p): {", this);
    auto p = reinterpret_cast<uint8_t*>(this);
    for (size_t i = 0; i < sizeof(*this); i++) {
      printf("%02X", p[i]);
    }
    printf("}\n");
    printf("Try to delete data_ (%p)\n", data_.get());
    data_.reset();
    printf("Has reset data_\n");
    printf("Try to delete indices_ (%p)\n", indices_.get());
    indices_.reset();
    printf("Has reset indices_\n");
    printf("Try to delete center_norms_ (%p)\n", center_norms_.get());
    center_norms_.reset();
    printf("Has reset center_norms_\n");
    printf("Try to delete centers_ (%p)\n", centers_.get());
    centers_.reset();
    printf("Has reset centers_\n");
    printf("Try to delete list_offsets_ (%p)\n", list_offsets_.get());
    list_offsets_.reset();
    printf("Has reset list_offsets_\n");
    printf("Try to delete list_sizes_ (%p)\n", list_sizes_.get());
    list_sizes_.reset();
    printf("Has reset list_sizes_\n");
  };

  /**
   * Construct the index. All data is moved to save the GPU memory
   * (hint: use std::move when necessary).
   */
  index(uint32_t veclen,
        raft::distance::DistanceType metric,
        const device_mdarray<T, extent_2d, row_major>& data,
        const device_mdarray<IdxT, extent_1d, row_major>& indices,
        const device_mdarray<uint32_t, extent_1d, row_major>& list_sizes,
        const device_mdarray<IdxT, extent_1d, row_major>& list_offsets,
        const device_mdarray<float, extent_2d, row_major>& centers,
        const std::optional<device_mdarray<float, extent_1d, row_major>>& center_norms)
    : knn::index(),
      veclen(veclen),
      metric(metric),
      data_(std::make_unique<device_mdarray<T, extent_2d, row_major>>(data)),
      indices_(std::make_unique<device_mdarray<IdxT, extent_1d, row_major>>(indices)),
      list_sizes_(std::make_unique<device_mdarray<uint32_t, extent_1d, row_major>>(list_sizes)),
      list_offsets_(std::make_unique<device_mdarray<IdxT, extent_1d, row_major>>(list_offsets)),
      centers_(std::make_unique<device_mdarray<float, extent_2d, row_major>>(centers)),
      center_norms_(center_norms.has_value()
                      ? std::make_unique<device_mdarray<float, extent_1d, row_major>>(*center_norms)
                      : nullptr)
  {
    // Throw an error if the index content is inconsistent.
    RAFT_EXPECTS(dim() % veclen == 0, "dimensionality is not a multiple of the veclen");
    RAFT_EXPECTS(data_->extent(0) == indices_->extent(0), "inconsistent index size");
    RAFT_EXPECTS(data_->extent(1) == centers_->extent(1), "inconsistent data dimensionality");
    RAFT_EXPECTS(                                                 //
      (centers_->extent(0) == list_sizes_->extent(0)) &&          //
        (centers_->extent(0) + 1 == list_offsets_->extent(0)) &&  //
        (!center_norms_ || centers_->extent(0) == center_norms_->extent(0)),
      "inconsistent number of lists (clusters)");
    RAFT_EXPECTS(reinterpret_cast<size_t>(data_->data()) % (veclen * sizeof(T)) == 0,
                 "The data storage pointer is not aligned to the vector length");
    printf("new      ivf_flat::index(%p): {", this);
    auto p = reinterpret_cast<uint8_t*>(this);
    for (size_t i = 0; i < sizeof(*this); i++) {
      printf("%02X", p[i]);
    }
    printf("}\n");
  }

 private:
  std::unique_ptr<device_mdarray<T, extent_2d, row_major>> data_;
  std::unique_ptr<device_mdarray<IdxT, extent_1d, row_major>> indices_;
  std::unique_ptr<device_mdarray<uint32_t, extent_1d, row_major>> list_sizes_;
  std::unique_ptr<device_mdarray<IdxT, extent_1d, row_major>> list_offsets_;
  std::unique_ptr<device_mdarray<float, extent_2d, row_major>> centers_;
  std::unique_ptr<device_mdarray<float, extent_1d, row_major>> center_norms_;
};

struct index_params : knn::index_params {
  /** The number of inverted lists (clusters) */
  uint32_t n_lists = 1024;
  /** The number of iterations searching for kmeans centers (index building). */
  uint32_t kmeans_n_iters = 20;
  /** The fraction of data to use during iterative kmeans building. */
  double kmeans_trainset_fraction = 0.5;
};

struct search_params : knn::search_params {
  /** The number of clusters to search. */
  uint32_t n_probes = 20;
};

// static_assert(std::is_standard_layout_v<index<float, uint32_t>>);
// static_assert(std::is_aggregate_v<index<float, uint32_t>>);
static_assert(std::is_aggregate_v<index_params>);
static_assert(std::is_aggregate_v<search_params>);

}  // namespace raft::spatial::knn::ivf_flat
