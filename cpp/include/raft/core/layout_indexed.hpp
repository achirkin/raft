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
#include <raft/core/memory_type.hpp>

#include <raft/thirdparty/mdspan/include/experimental/mdspan>

// #include "macros.hpp"
// #include "trait_backports.hpp"
// #include "extents.hpp"
// #include <stdexcept>
// #include "layout_stride.hpp"

#include <type_traits>

namespace raft {

namespace detail {

template <typename Extents>
struct extents_head {
};

template <typename Extents>
struct extents_tail {
};

template <template <typename, size_t...> typename ExtentsType,
          typename IndexType,
          size_t Head,
          size_t... Tail>
struct extents_head<ExtentsType<IndexType, Head, Tail...>> {
  using value = ExtentsType<IndexType, Head>;
};

template <template <typename, size_t...> typename ExtentsType,
          typename IndexType,
          size_t Head,
          size_t... Tail>
struct extents_tail<ExtentsType<IndexType, Head, Tail...>> {
  using value = ExtentsType<IndexType, Tail...>;
};

template <typename Extents>
struct source_extents_t {
};

template <template <typename, size_t...> typename ExtentsType,
          typename IndexType,
          size_t Head,
          size_t... Tail>
struct source_extents_t<ExtentsType<IndexType, Head, Tail...>> {
  using value = ExtentsType<IndexType, std::experimental::dynamic_extent, Tail...>;
};

}  // namespace detail

/**
 * @brief A mixin for selecting a subset of elements along the first dimension.
 */
template <typename LayoutPolicy>
struct layout_indexed {
  template <class Extents>
  class mapping
    : public LayoutPolicy::template mapping<typename detail::source_extents_t<Extents>::value> {
   public:
    using extents_type = Extents;
    using index_type   = typename extents_type::index_type;
    using size_type    = typename extents_type::size_type;
    using rank_type    = typename extents_type::rank_type;
    using layout_type  = layout_indexed<LayoutPolicy>;

    using source_extents_type  = typename detail::source_extents_t<Extents>::value;
    using source_mapping_type  = typename LayoutPolicy::template mapping<source_extents_type>;
    using indices_extents_type = typename detail::extents_head<Extents>::value;

   private:
    static_assert(std::experimental::detail::__is_extents_v<extents_type>,
                  "raft::layout_indexed::mapping must be instantiated with a "
                  "specialization of std::experimental::extents.");

    template <class>
    friend class mapping;

   public:
    MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping(mapping const&) noexcept = default;
    auto operator=(mapping const&) noexcept -> mapping& = default;

    MDSPAN_TEMPLATE_REQUIRES(
      class OtherExtents,
      /* requires */ (_MDSPAN_TRAIT(std::is_constructible, extents_type, OtherExtents)))
    MDSPAN_CONDITIONAL_EXPLICIT((extents_type::rank() > 0))
    constexpr mapping(typename LayoutPolicy::template mapping<OtherExtents> const& source_mapping,
                      const index_type* indices,
                      const indices_extents_type& indices_extents)
      : source_mapping_type(source_mapping), indices_(indices), indices_extents_(indices_extents)
    {
    }

    MDSPAN_TEMPLATE_REQUIRES(
      class Index0,
      class... Indices,
      /* requires */
      ((sizeof...(Indices) + 1 == extents_type::rank()) &&
       std::is_convertible_v<Index0, index_type> &&
       std::is_nothrow_constructible_v<index_type, Index0> &&
       _MDSPAN_FOLD_AND((_MDSPAN_TRAIT(std::is_convertible, Indices, index_type) &&
                         _MDSPAN_TRAIT(std::is_nothrow_constructible, index_type, Indices)))))
    constexpr auto operator()(Index0 idx0, Indices... idxs) const noexcept -> index_type
    {
      return source_mapping_type::operator()(indices_[idx0], idxs...);
    }

    inline static constexpr auto is_always_unique() noexcept { return false; }
    inline static constexpr auto is_always_exhaustive() noexcept { return false; }
    inline static constexpr auto is_always_strided() noexcept { return false; }
    inline constexpr auto is_unique() const noexcept { return false; }
    inline constexpr auto is_exhaustive() const noexcept { return false; }
    inline constexpr auto is_strided() const noexcept { return false; }

    template <class OtherExtents>
    inline friend constexpr auto operator==(mapping const& lhs,
                                            mapping<OtherExtents> const& rhs) noexcept -> bool
    {
      return lhs.indices_ == rhs.indices_ && lhs.indices_extents_ == rhs.indices_extents_ &&
             lhs.source_extents() == rhs.source_extents();
    }

    [[nodiscard]] constexpr auto extents() const noexcept -> extents_type
    {
      auto r      = source_extents_type::extents();
      r.extent(0) = indices_extents_.extent(0);
      return extents_type(r);
    }

    [[nodiscard]] constexpr auto source_extents() const noexcept -> const source_extents_type&
    {
      return source_extents_type::extents();
    }

   private:
    const index_type* indices_;
    _MDSPAN_NO_UNIQUE_ADDRESS indices_extents_type indices_extents_{};
  };
};

}  // namespace raft
