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

#include <raft/matrix/detail/select_radix.cuh>
#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/matrix/select.cuh>

#include <rmm/mr/device/device_memory_resource.hpp>

namespace raft::matrix::select {

struct params {
  size_t batch_size;
  size_t len;
  int k;
  bool select_min;
  bool use_index_input = true;
};

inline auto operator<<(std::ostream& os, const params& ss) -> std::ostream&
{
  os << "params{batch_size: " << ss.batch_size;
  os << ", len: " << ss.len;
  os << ", k: " << ss.k;
  os << (ss.select_min ? ", asc" : ", dsc");
  os << (ss.use_index_input ? "}" : ", no-input-index}");
  return os;
}

enum class Algo {
  kPublicApi,
  kRadix8bits,
  kRadix11bits,
  kWarpAuto,
  kWarpImmediate,
  kWarpFiltered,
  kWarpDistributed,
  kWarpDistributedShm
};

inline auto operator<<(std::ostream& os, const Algo& algo) -> std::ostream&
{
  switch (algo) {
    case Algo::kPublicApi: return os << "kPublicApi";
    case Algo::kRadix8bits: return os << "kRadix8bits";
    case Algo::kRadix11bits: return os << "kRadix11bits";
    case Algo::kWarpAuto: return os << "kWarpAuto";
    case Algo::kWarpImmediate: return os << "kWarpImmediate";
    case Algo::kWarpFiltered: return os << "kWarpFiltered";
    case Algo::kWarpDistributed: return os << "kWarpDistributed";
    case Algo::kWarpDistributedShm: return os << "kWarpDistributedShm";
    default: return os << "unknown enum value";
  }
}

template <typename T, typename IdxT>
void select_k_impl(const Algo& algo,
                   const T* in,
                   const IdxT* in_idx,
                   size_t batch_size,
                   size_t len,
                   int k,
                   T* out,
                   IdxT* out_idx,
                   bool select_min,
                   rmm::cuda_stream_view stream,
                   rmm::mr::device_memory_resource* mr = nullptr)
{
  switch (algo) {
    case Algo::kPublicApi:
      return matrix::select_k(in, in_idx, batch_size, len, k, out, out_idx, select_min, stream, mr);
    case Algo::kRadix8bits:
      return detail::select::radix::select_k<T, IdxT, 8, 512>(
        in, in_idx, batch_size, len, k, out, out_idx, select_min, stream, mr);
    case Algo::kRadix11bits:
      return detail::select::radix::select_k<T, IdxT, 11, 512>(
        in, in_idx, batch_size, len, k, out, out_idx, select_min, stream, mr);
    case Algo::kWarpAuto:
      return detail::select::warpsort::select_k<T, IdxT>(
        in, in_idx, batch_size, len, k, out, out_idx, select_min, stream, mr);
    case Algo::kWarpImmediate:
      return detail::select::warpsort::
        select_k_impl<T, IdxT, detail::select::warpsort::warp_sort_immediate>(
          in, in_idx, batch_size, len, k, out, out_idx, select_min, stream, mr);
    case Algo::kWarpFiltered:
      return detail::select::warpsort::
        select_k_impl<T, IdxT, detail::select::warpsort::warp_sort_filtered>(
          in, in_idx, batch_size, len, k, out, out_idx, select_min, stream, mr);
    case Algo::kWarpDistributed:
      return detail::select::warpsort::
        select_k_impl<T, IdxT, detail::select::warpsort::warp_sort_distributed>(
          in, in_idx, batch_size, len, k, out, out_idx, select_min, stream, mr);
    case Algo::kWarpDistributedShm:
      return detail::select::warpsort::
        select_k_impl<T, IdxT, detail::select::warpsort::warp_sort_distributed_ext>(
          in, in_idx, batch_size, len, k, out, out_idx, select_min, stream, mr);
  }
}

}  // namespace raft::matrix::select
