/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include "../test_utils.h"
#include <gtest/gtest.h>
#include <limits>
#include <raft/cudart_utils.h>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/random/rng.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace raft {
namespace linalg {

template <typename InType, typename OutType, typename MapOp>
__global__ void naiveMapReduceKernel(OutType* out, const InType* in, size_t len, MapOp map)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) { raft::myAtomicAdd(out, (OutType)map(in[idx])); }
}

template <typename InType, typename OutType, typename MapOp>
void naiveMapReduce(OutType* out, const InType* in, size_t len, MapOp map, cudaStream_t stream)
{
  static const int TPB = 64;
  int nblks            = raft::ceildiv(len, (size_t)TPB);
  naiveMapReduceKernel<InType, OutType, MapOp><<<nblks, TPB, 0, stream>>>(out, in, len, map);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename T>
struct MapReduceInputs {
  T tolerance;
  size_t len;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const MapReduceInputs<T>& dims)
{
  return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename InType, typename OutType>
void mapReduceLaunch(
  OutType* out_ref, OutType* out, const InType* in, size_t len, cudaStream_t stream)
{
  auto op = [] __device__(InType in) { return in; };
  naiveMapReduce(out_ref, in, len, op, stream);
  mapThenSumReduce(out, len, op, 0, in);
}

template <typename InType, typename OutType>
class MapReduceTest : public ::testing::TestWithParam<MapReduceInputs<InType>> {
 public:
  MapReduceTest()
    : params(::testing::TestWithParam<MapReduceInputs<InType>>::GetParam()),
      stream(handle.get_stream()),
      in(params.len, stream),
      out_ref(params.len, stream),
      out(params.len, stream)

  {
  }

 protected:
  void SetUp() override
  {
    raft::random::Rng r(params.seed);
    auto len = params.len;
    r.uniform(in.data(), len, InType(-1.0), InType(1.0), stream);
    mapReduceLaunch(out_ref.data(), out.data(), in.data(), len, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  MapReduceInputs<InType> params;
  rmm::device_uvector<InType> in;
  rmm::device_uvector<OutType> out_ref, out;
};

const std::vector<MapReduceInputs<float>> inputsf = {{0.001f, 1024 * 1024, 1234ULL}};
typedef MapReduceTest<float, float> MapReduceTestFF;
TEST_P(MapReduceTestFF, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), params.len, CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_SUITE_P(MapReduceTests, MapReduceTestFF, ::testing::ValuesIn(inputsf));

typedef MapReduceTest<float, double> MapReduceTestFD;
TEST_P(MapReduceTestFD, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), params.len, CompareApprox<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_SUITE_P(MapReduceTests, MapReduceTestFD, ::testing::ValuesIn(inputsf));

const std::vector<MapReduceInputs<double>> inputsd = {{0.000001, 1024 * 1024, 1234ULL}};
typedef MapReduceTest<double, double> MapReduceTestDD;
TEST_P(MapReduceTestDD, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), params.len, CompareApprox<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_SUITE_P(MapReduceTests, MapReduceTestDD, ::testing::ValuesIn(inputsd));

template <typename T>
class MapGenericReduceTest : public ::testing::Test {
  using InType  = typename T::first_type;
  using OutType = typename T::second_type;

 protected:
  MapGenericReduceTest() : input(n, handle.get_stream()), output(handle.get_stream())
  {
    initInput(input.data(), input.size(), handle.get_stream());
  }

 public:
  void initInput(InType* input, int n, cudaStream_t stream)
  {
    raft::random::Rng r(137);
    r.uniform(input, n, InType(2), InType(3), handle.get_stream());
    InType val = 1;
    raft::update_device(input + 42, &val, 1, handle.get_stream());
    val = 5;
    raft::update_device(input + 337, &val, 1, handle.get_stream());
  }

  void testMin()
  {
    auto op               = [] __device__(InType in) { return in; };
    const OutType neutral = std::numeric_limits<InType>::max();
    mapThenReduce(
      output.data(), input.size(), neutral, op, cub::Min(), handle.get_stream(), input.data());
    EXPECT_TRUE(raft::devArrMatch(
      OutType(1), output.data(), 1, raft::Compare<OutType>(), handle.get_stream()));
  }
  void testMax()
  {
    auto op               = [] __device__(InType in) { return in; };
    const OutType neutral = std::numeric_limits<InType>::min();
    mapThenReduce(
      output.data(), input.size(), neutral, op, cub::Max(), handle.get_stream(), input.data());
    EXPECT_TRUE(raft::devArrMatch(
      OutType(5), output.data(), 1, raft::Compare<OutType>(), handle.get_stream()));
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  int n = 1237;
  rmm::device_uvector<InType> input;
  rmm::device_scalar<OutType> output;
};

using IoTypePair =
  ::testing::Types<std::pair<float, float>, std::pair<float, double>, std::pair<double, double>>;

TYPED_TEST_CASE(MapGenericReduceTest, IoTypePair);
TYPED_TEST(MapGenericReduceTest, min) { this->testMin(); }
TYPED_TEST(MapGenericReduceTest, max) { this->testMax(); }
}  // end namespace linalg
}  // end namespace raft
