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

#include <raft/core/operators.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/limiting_resource_adaptor.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cstddef>
#include <optional>

namespace raft::resource {
class limiting_memory_resource : public resource {
 public:
  limiting_memory_resource(std::shared_ptr<rmm::mr::device_memory_resource> mr,
                           std::size_t allocation_limit,
                           std::optional<std::size_t> alignment)
    : upstream_(mr), mr_(make_adaptor(mr, allocation_limit, alignment))
  {
  }

  auto get_resource() -> void* override { return &mr_; }

  ~limiting_memory_resource() override = default;

 private:
  std::shared_ptr<rmm::mr::device_memory_resource> upstream_;
  rmm::mr::limiting_resource_adaptor<rmm::mr::device_memory_resource> mr_;

  static inline auto make_adaptor(std::shared_ptr<rmm::mr::device_memory_resource> upstream,
                                  std::size_t limit,
                                  std::optional<std::size_t> alignment)
    -> rmm::mr::limiting_resource_adaptor<rmm::mr::device_memory_resource>
  {
    auto p = upstream.get();
    if (alignment.has_value()) {
      return rmm::mr::limiting_resource_adaptor(p, limit, alignment.value());
    } else {
      return rmm::mr::limiting_resource_adaptor(p, limit);
    }
  }
};

/**
 * Factory that knows how to construct a specific raft::resource to populate
 * the resources instance.
 */
class workspace_resource_factory : public resource_factory {
 public:
  explicit workspace_resource_factory(
    std::shared_ptr<rmm::mr::device_memory_resource> mr = {nullptr},
    std::optional<std::size_t> allocation_limit         = std::nullopt,
    std::optional<std::size_t> alignment                = std::nullopt)
    : allocation_limit_(allocation_limit.value_or(default_allocation_limit())),
      alignment_(alignment),
      mr_(mr ? mr : default_memory_resource(allocation_limit_))
  {
  }

  auto get_resource_type() -> resource_type override { return resource_type::WORKSPACE_RESOURCE; }
  auto make_resource() -> resource* override
  {
    return new limiting_memory_resource(mr_, allocation_limit_, alignment_);
  }

  /** Construct a sensible default pool memory resource. */
  static inline auto default_pool_resource(std::size_t limit)
    -> std::shared_ptr<rmm::mr::device_memory_resource>
  {
    // Set the default granularity to 1 GiB
    constexpr std::size_t kOneGb = 1024lu * 1024lu * 1024lu;
    // The initial size of the pool. The choice of this value only affects the performance a little
    // bit. Heuristics:
    //   1) the pool shouldn't be too big from the beginning independently of the limit;
    //   2) otherwise, set it to half the max size to avoid too many resize calls.
    auto min_size = std::min<std::size_t>(kOneGb, limit / 2lu);
    // The pool is going to be place behind the limiting resource adaptor. This means the user won't
    // be able to allocate more than 'limit' bytes of memory anyway. At the same time, the pool
    // itself may consume a little bit more memory than the 'limit' due to memory fragmentation.
    // Therefore, we look for a compromise, such that:
    //   1) 'limit' is accurate - the user should be more likely to run into the limiting
    //      resource adaptor bad_alloc error than into the pool bad_alloc error.
    //   2) The pool doesn't grab too much memory on top of the 'limit'.
    auto max_size = std::min<std::size_t>(limit + kOneGb / 2lu, limit * 3lu / 2lu);
    auto upstream = rmm::mr::get_current_device_resource();
    return std::make_shared<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>>(
      upstream, min_size, max_size);
  }

  /** Get the global memory resource wrapped into an unmanaged shared_ptr (with no deleter). */
  static inline auto default_plain_resource() -> std::shared_ptr<rmm::mr::device_memory_resource>
  {
    return std::shared_ptr<rmm::mr::device_memory_resource>{rmm::mr::get_current_device_resource(),
                                                            void_op{}};
  }

 private:
  std::size_t allocation_limit_;
  std::optional<std::size_t> alignment_;
  std::shared_ptr<rmm::mr::device_memory_resource> mr_;

  static inline auto default_memory_resource(std::size_t limit)
    -> std::shared_ptr<rmm::mr::device_memory_resource>
  {
    if (rmm::mr::cuda_memory_resource{}.is_equal(*rmm::mr::get_current_device_resource())) {
      // Use the memory pool if only we're sure the global memory resource is set to its default,
      // which is the cuda_memory_resource.
      // The reason for this is that some raft algorithms rely on the workspace allocator to be
      // fast; e.g. some buffers are allocated and released in a loop in performance-critical paths
      // (batching), such as ANN-search routines. We don't want many allocations to happen there
      // unless the user insists on it.
      RAFT_LOG_DEBUG("The workspace uses the pool memory resource by default (limit: %zu)", limit);
      return default_pool_resource(limit);
    } else {
      // If the user sets the global (rmm) memory resource to anything but the trivial
      // cuda_memory_resource, we don't interfere that - they know better. In this case, the
      // limiting resource adaptor is set on top of the global (per-device) resource.
      RAFT_LOG_DEBUG("The workspace uses the global default memory resource (limit: %zu)", limit);
      return default_plain_resource();
    }
  }

  static inline auto default_allocation_limit() -> std::size_t
  {
    std::size_t free_size{};
    std::size_t total_size{};
    RAFT_CUDA_TRY(cudaMemGetInfo(&free_size, &total_size));
    // Note, the workspace does not claim all this memory from the start, so it's still usable by
    // the main resource as well.
    // This limit is merely an order for algorithm internals to plan the batching accordingly.
    return total_size / 2;
  }
};

/**
 * Load a temp workspace resource from a resources instance (and populate it on the res
 * if needed).
 *
 * @param res raft resources object for managing resources
 * @return device memory resource object
 */
inline auto get_workspace_resource(resources const& res)
  -> rmm::mr::limiting_resource_adaptor<rmm::mr::device_memory_resource>*
{
  if (!res.has_resource_factory(resource_type::WORKSPACE_RESOURCE)) {
    res.add_resource_factory(std::make_shared<workspace_resource_factory>());
  }
  return res.get_resource<rmm::mr::limiting_resource_adaptor<rmm::mr::device_memory_resource>>(
    resource_type::WORKSPACE_RESOURCE);
};

/** Get the total size of the workspace resource. */
inline auto get_workspace_total_bytes(resources const& res) -> size_t
{
  return get_workspace_resource(res)->get_allocation_limit();
};

/** Get the already allocated size of the workspace resource. */
inline auto get_workspace_used_bytes(resources const& res) -> size_t
{
  return get_workspace_resource(res)->get_allocated_bytes();
};

/** Get the available size of the workspace resource. */
inline auto get_workspace_free_bytes(resources const& res) -> size_t
{
  const auto* p = get_workspace_resource(res);
  return p->get_allocation_limit() - p->get_allocated_bytes();
};

/**
 * Set a temporary workspace resource on a resources instance.
 *
 * @param res raft resources object for managing resources
 * @param mr an optional RMM device_memory_resource
 * @param allocation_limit
 *   the total amount of memory in bytes available to the temporary workspace resources.
 * @param alignment optional alignment requirements passed to RMM allocations
 *
 */
inline void set_workspace_resource(resources const& res,
                                   std::shared_ptr<rmm::mr::device_memory_resource> mr = {nullptr},
                                   std::optional<std::size_t> allocation_limit = std::nullopt,
                                   std::optional<std::size_t> alignment        = std::nullopt)
{
  res.add_resource_factory(
    std::make_shared<workspace_resource_factory>(mr, allocation_limit, alignment));
};

/**
 * Set the temporary workspace resource to a pool on top of the global memory resource
 * (`rmm::mr::get_current_device_resource()`.
 *
 * @param res raft resources object for managing resources
 * @param allocation_limit
 *   the total amount of memory in bytes available to the temporary workspace resources;
 *   if not provided, a last used or default limit is used.
 *
 */
inline void set_workspace_to_pool_resource(
  resources const& res, std::optional<std::size_t> allocation_limit = std::nullopt)
{
  if (!allocation_limit.has_value()) { allocation_limit = get_workspace_total_bytes(res); }
  res.add_resource_factory(std::make_shared<workspace_resource_factory>(
    workspace_resource_factory::default_pool_resource(*allocation_limit),
    allocation_limit,
    std::nullopt));
};

/**
 * Set the temporary workspace resource the same as the global memory resource
 * (`rmm::mr::get_current_device_resource()`.
 *
 * Note, the workspace resource is always limited; the limit here defines how much of the global
 * memory resource can be consumed by the workspace allocations.
 *
 * @param res raft resources object for managing resources
 * @param allocation_limit
 *   the total amount of memory in bytes available to the temporary workspace resources.
 */
inline void set_workspace_to_global_resource(
  resources const& res, std::optional<std::size_t> allocation_limit = std::nullopt)
{
  res.add_resource_factory(std::make_shared<workspace_resource_factory>(
    workspace_resource_factory::default_plain_resource(), allocation_limit, std::nullopt));
};

}  // namespace raft::resource
