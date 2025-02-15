/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <raft/cuda_utils.cuh>
#include <raft/pow2_utils.cuh>
#include <raft/vectorized.cuh>

namespace raft {
namespace linalg {

namespace {
template <size_t VecBytes>
struct AlignedAccess {
  template <typename T>
  static inline bool test(const T* matrix, size_t strideBytes)
  {
    return Pow2<VecBytes>::isAligned(matrix) && Pow2<VecBytes>::isAligned(strideBytes) &&
           Pow2<sizeof(T)>::isAligned(VecBytes);
  }
};
};  // namespace

template <typename Type, int veclen_, typename Lambda, typename IdxType>
__global__ void matrixVectorOpKernel(Type* out,
                                     const Type* matrix,
                                     const Type* vector,
                                     IdxType D,
                                     IdxType N,
                                     bool rowMajor,
                                     bool bcastAlongRows,
                                     Lambda op)
{
  typedef TxN_t<Type, veclen_> VecType;
  IdxType len = N * D;
  IdxType idx = threadIdx.x;
  idx += (IdxType)blockIdx.x * (IdxType)blockDim.x;
  idx *= VecType::Ratio;
  if (idx >= len) return;
  IdxType vIdx;
  VecType mat, vec;
  ///@todo: yikes! use fast-int-div here.
  ///@todo: shared mem for vector could help with perf
  if (rowMajor && bcastAlongRows) {
    vIdx = idx % D;
    vec.load(vector, vIdx);
  } else if (!rowMajor && !bcastAlongRows) {
    vIdx = idx % N;
    vec.load(vector, vIdx);
  } else if (rowMajor && !bcastAlongRows) {
    vIdx = idx / D;
    vec.fill(vector[vIdx]);
  } else {
    vIdx = idx / N;
    vec.fill(vector[vIdx]);
  }
  mat.load(matrix, idx);
#pragma unroll
  for (int i = 0; i < VecType::Ratio; ++i)
    mat.val.data[i] = op(mat.val.data[i], vec.val.data[i]);
  mat.store(out, idx);
}

template <typename Type, int veclen_, typename Lambda, typename IdxType, int TPB>
void matrixVectorOpImpl(Type* out,
                        const Type* matrix,
                        const Type* vec,
                        IdxType D,
                        IdxType N,
                        bool rowMajor,
                        bool bcastAlongRows,
                        Lambda op,
                        cudaStream_t stream)
{
  IdxType len   = N * D;
  IdxType nblks = raft::ceildiv(veclen_ ? len / veclen_ : veclen_, (IdxType)TPB);
  matrixVectorOpKernel<Type, veclen_, Lambda, IdxType>
    <<<nblks, TPB, 0, stream>>>(out, matrix, vec, D, N, rowMajor, bcastAlongRows, op);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief Operations for all the columns or rows with a given vector.
 * Caution : Threads process multiple elements to speed up processing. These
 * are loaded in a single read thanks to type promotion. Faster processing
 * would thus only be enabled when adresses are optimally aligned for it.
 * Note : the function will also check that the size of the window of accesses
 * is a multiple of the number of elements processed by a thread in order to
 * enable faster processing
 * @tparam Type the matrix/vector type
 * @tparam Lambda a device function which represents a binary operator
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads per block of the cuda kernel launched
 * @param out the output matrix (passing out = matrix makes it in-place)
 * @param matrix the input matrix
 * @param vec the vector
 * @param D number of columns of matrix
 * @param N number of rows of matrix
 * @param rowMajor whether input is row or col major
 * @param bcastAlongRows whether the broadcast of vector needs to happen along
 * the rows of the matrix or columns
 * @param op the mathematical operation
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename Lambda, typename IdxType = int, int TPB = 256>
void matrixVectorOp(Type* out,
                    const Type* matrix,
                    const Type* vec,
                    IdxType D,
                    IdxType N,
                    bool rowMajor,
                    bool bcastAlongRows,
                    Lambda op,
                    cudaStream_t stream)
{
  IdxType stride      = rowMajor ? D : N;
  size_t stride_bytes = stride * sizeof(Type);

  if (AlignedAccess<16>::test(matrix, stride_bytes)) {
    matrixVectorOpImpl<Type, 16 / sizeof(Type), Lambda, IdxType, TPB>(
      out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
  } else if (AlignedAccess<8>::test(matrix, stride_bytes)) {
    matrixVectorOpImpl<Type, 8 / sizeof(Type), Lambda, IdxType, TPB>(
      out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
  } else if (AlignedAccess<4>::test(matrix, stride_bytes)) {
    matrixVectorOpImpl<Type, 4 / sizeof(Type), Lambda, IdxType, TPB>(
      out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
  } else if (AlignedAccess<2>::test(matrix, stride_bytes)) {
    matrixVectorOpImpl<Type, 2 / sizeof(Type), Lambda, IdxType, TPB>(
      out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
  } else if (AlignedAccess<1>::test(matrix, stride_bytes)) {
    matrixVectorOpImpl<Type, 1 / sizeof(Type), Lambda, IdxType, TPB>(
      out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
  } else {
    matrixVectorOpImpl<Type, 1, Lambda, IdxType, TPB>(
      out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
  }
}

///@todo: come up with a cleaner interface to support these cases in future!

template <typename Type, int veclen_, typename Lambda, typename IdxType>
__global__ void matrixVectorOpKernel(Type* out,
                                     const Type* matrix,
                                     const Type* vector1,
                                     const Type* vector2,
                                     IdxType D,
                                     IdxType N,
                                     bool rowMajor,
                                     bool bcastAlongRows,
                                     Lambda op)
{
  typedef TxN_t<Type, veclen_> VecType;
  IdxType len = N * D;
  IdxType idx = (threadIdx.x + (blockIdx.x * blockDim.x)) * VecType::Ratio;
  if (idx >= len) return;
  IdxType vIdx;
  VecType mat, vec1, vec2;
  ///@todo: yikes! use fast-int-div here.
  ///@todo: shared mem for vector could help with perf
  if (rowMajor && bcastAlongRows) {
    vIdx = idx % D;
    vec1.load(vector1, vIdx);
    vec2.load(vector2, vIdx);
  } else if (!rowMajor && !bcastAlongRows) {
    vIdx = idx % N;
    vec1.load(vector1, vIdx);
    vec2.load(vector2, vIdx);
  } else if (rowMajor && !bcastAlongRows) {
    vIdx = idx / D;
    vec1.fill(vector1[vIdx]);
    vec2.fill(vector2[vIdx]);
  } else {
    vIdx = idx / N;
    vec1.fill(vector1[vIdx]);
    vec2.fill(vector2[vIdx]);
  }
  mat.load(matrix, idx);
#pragma unroll
  for (int i = 0; i < VecType::Ratio; ++i)
    mat.val.data[i] = op(mat.val.data[i], vec1.val.data[i], vec2.val.data[i]);
  mat.store(out, idx);
}

template <typename Type, int veclen_, typename Lambda, typename IdxType, int TPB>
void matrixVectorOpImpl(Type* out,
                        const Type* matrix,
                        const Type* vec1,
                        const Type* vec2,
                        IdxType D,
                        IdxType N,
                        bool rowMajor,
                        bool bcastAlongRows,
                        Lambda op,
                        cudaStream_t stream)
{
  IdxType nblks = raft::ceildiv(N * D, (IdxType)TPB);
  matrixVectorOpKernel<Type, veclen_, Lambda, IdxType>
    <<<nblks, TPB, 0, stream>>>(out, matrix, vec1, vec2, D, N, rowMajor, bcastAlongRows, op);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief Operations for all the columns or rows with the given vectors.
 * Caution : Threads process multiple elements to speed up processing. These
 * are loaded in a single read thanks to type promotion. Faster processing
 * would thus only be enabled when adresses are optimally aligned for it.
 * Note : the function will also check that the size of the window of accesses
 * is a multiple of the number of elements processed by a thread in order to
 * enable faster processing
 * @tparam Type the matrix/vector type
 * @tparam Lambda a device function which represents a binary operator
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads per block of the cuda kernel launched
 * @param out the output matrix (passing out = matrix makes it in-place)
 * @param matrix the input matrix
 * @param vec1 the first vector
 * @param vec2 the second vector
 * @param D number of columns of matrix
 * @param N number of rows of matrix
 * @param rowMajor whether input is row or col major
 * @param bcastAlongRows whether the broadcast of vector needs to happen along
 * the rows of the matrix or columns
 * @param op the mathematical operation
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename Lambda, typename IdxType = int, int TPB = 256>
void matrixVectorOp(Type* out,
                    const Type* matrix,
                    const Type* vec1,
                    const Type* vec2,
                    IdxType D,
                    IdxType N,
                    bool rowMajor,
                    bool bcastAlongRows,
                    Lambda op,
                    cudaStream_t stream)
{
  IdxType stride      = rowMajor ? D : N;
  size_t stride_bytes = stride * sizeof(Type);

  if (AlignedAccess<16>::test(matrix, stride_bytes)) {
    matrixVectorOpImpl<Type, 16 / sizeof(Type), Lambda, IdxType, TPB>(
      out, matrix, vec1, vec2, D, N, rowMajor, bcastAlongRows, op, stream);
  } else if (AlignedAccess<8>::test(matrix, stride_bytes)) {
    matrixVectorOpImpl<Type, 8 / sizeof(Type), Lambda, IdxType, TPB>(
      out, matrix, vec1, vec2, D, N, rowMajor, bcastAlongRows, op, stream);
  } else if (AlignedAccess<4>::test(matrix, stride_bytes)) {
    matrixVectorOpImpl<Type, 4 / sizeof(Type), Lambda, IdxType, TPB>(
      out, matrix, vec1, vec2, D, N, rowMajor, bcastAlongRows, op, stream);
  } else if (AlignedAccess<2>::test(matrix, stride_bytes)) {
    matrixVectorOpImpl<Type, 2 / sizeof(Type), Lambda, IdxType, TPB>(
      out, matrix, vec1, vec2, D, N, rowMajor, bcastAlongRows, op, stream);
  } else if (AlignedAccess<1>::test(matrix, stride_bytes)) {
    matrixVectorOpImpl<Type, 1 / sizeof(Type), Lambda, IdxType, TPB>(
      out, matrix, vec1, vec2, D, N, rowMajor, bcastAlongRows, op, stream);
  } else {
    matrixVectorOpImpl<Type, 1, Lambda, IdxType, TPB>(
      out, matrix, vec1, vec2, D, N, rowMajor, bcastAlongRows, op, stream);
  }
}

};  // end namespace linalg
};  // end namespace raft
