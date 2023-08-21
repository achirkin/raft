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

#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <memory>
#include <typeindex>

namespace raft::resource {

class user_resource : public resource {
 public:
  user_resource()                    = default;
  ~user_resource() noexcept override = default;
  auto get_resource() -> void* override { return this; }

  template <typename Store>
  auto load() -> Store*
  {
    std::lock_guard<std::mutex> _(lock_);
    auto key = std::type_index{typeid(Store)};
    auto pos = map_.find(key);
    if (pos != map_.end()) { return reinterpret_cast<Store*>(pos->second.get()); }
    auto store_ptr = new Store{};
    map_[key] =
      std::shared_ptr<void>(store_ptr, [](void* ptr) { delete reinterpret_cast<Store*>(ptr); });
    return store_ptr;
  }

 private:
  std::unordered_map<std::type_index, std::shared_ptr<void>> map_{};
  std::mutex lock_{};
};

/** Factory that knows how to construct a specific raft::resource to populate the res_t. */
class user_resource_factory : public resource_factory {
 public:
  auto get_resource_type() -> resource_type override { return resource_type::USER_DEFINED; }
  auto make_resource() -> resource* override { return new user_resource(); }
};

/**
 * Get the user-defined default-constructible resource if it exists, create it otherwise.
 * @param[in] res the raft resources object
 * @return a pointer to the user-defined resource.
 */
template <typename Store>
auto get_user_resource(resources const& res) -> Store*
{
  if (!res.has_resource_factory(resource_type::USER_DEFINED)) {
    res.add_resource_factory(std::make_shared<user_resource_factory>());
  }
  return res.get_resource<user_resource>(resource_type::USER_DEFINED)->load<Store>();
};

}  // namespace raft::resource
