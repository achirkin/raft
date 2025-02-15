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

#pragma once

#include <cusolverDn.h>
#include <cusolverSp.h>
///@todo: enable this once logging is enabled
//#include <cuml/common/logger.hpp>
#include <raft/cudart_utils.h>
#include <type_traits>

#define _CUSOLVER_ERR_TO_STR(err) \
  case err: return #err;

namespace raft {

/**
 * @brief Exception thrown when a cuSOLVER error is encountered.
 */
struct cusolver_error : public raft::exception {
  explicit cusolver_error(char const* const message) : raft::exception(message) {}
  explicit cusolver_error(std::string const& message) : raft::exception(message) {}
};

namespace linalg {
namespace detail {

inline const char* cusolver_error_to_string(cusolverStatus_t err)
{
  switch (err) {
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_SUCCESS);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_NOT_INITIALIZED);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_ALLOC_FAILED);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_INVALID_VALUE);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_ARCH_MISMATCH);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_EXECUTION_FAILED);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_INTERNAL_ERROR);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_ZERO_PIVOT);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_NOT_SUPPORTED);
    default: return "CUSOLVER_STATUS_UNKNOWN";
  };
}

}  // namespace detail
}  // namespace linalg
}  // namespace raft

#undef _CUSOLVER_ERR_TO_STR

/**
 * @brief Error checking macro for cuSOLVER runtime API functions.
 *
 * Invokes a cuSOLVER runtime API function call, if the call does not return
 * CUSolver_STATUS_SUCCESS, throws an exception detailing the cuSOLVER error that occurred
 */
#define RAFT_CUSOLVER_TRY(call)                                              \
  do {                                                                       \
    cusolverStatus_t const status = (call);                                  \
    if (CUSOLVER_STATUS_SUCCESS != status) {                                 \
      std::string msg{};                                                     \
      SET_ERROR_MSG(msg,                                                     \
                    "cuSOLVER error encountered at: ",                       \
                    "call='%s', Reason=%d:%s",                               \
                    #call,                                                   \
                    status,                                                  \
                    raft::linalg::detail::cusolver_error_to_string(status)); \
      throw raft::cusolver_error(msg);                                       \
    }                                                                        \
  } while (0)

// FIXME: remove after consumer rename
#ifndef CUSOLVER_TRY
#define CUSOLVER_TRY(call) RAFT_CUSOLVER_TRY(call)
#endif

// /**
//  * @brief check for cuda runtime API errors but log error instead of raising
//  *        exception.
//  */
#define RAFT_CUSOLVER_TRY_NO_THROW(call)                               \
  do {                                                                 \
    cusolverStatus_t const status = call;                              \
    if (CUSOLVER_STATUS_SUCCESS != status) {                           \
      printf("CUSOLVER call='%s' at file=%s line=%d failed with %s\n", \
             #call,                                                    \
             __FILE__,                                                 \
             __LINE__,                                                 \
             raft::linalg::detail::cusolver_error_to_string(status));  \
    }                                                                  \
  } while (0)

// FIXME: remove after cuml rename
#ifndef CUSOLVER_CHECK
#define CUSOLVER_CHECK(call) CUSOLVER_TRY(call)
#endif

#ifndef CUSOLVER_CHECK_NO_THROW
#define CUSOLVER_CHECK_NO_THROW(call) CUSOLVER_TRY_NO_THROW(call)
#endif

namespace raft {
namespace linalg {

/**
 * @defgroup Getrf cusolver getrf operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDngetrf(cusolverDnHandle_t handle,
                                 int m,  // NOLINT
                                 int n,
                                 T* A,
                                 int lda,
                                 T* Workspace,
                                 int* devIpiv,
                                 int* devInfo,
                                 cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDngetrf(cusolverDnHandle_t handle,  // NOLINT
                                        int m,
                                        int n,
                                        float* A,
                                        int lda,
                                        float* Workspace,
                                        int* devIpiv,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

template <>
inline cusolverStatus_t cusolverDngetrf(cusolverDnHandle_t handle,  // NOLINT
                                        int m,
                                        int n,
                                        double* A,
                                        int lda,
                                        double* Workspace,
                                        int* devIpiv,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

template <typename T>
cusolverStatus_t cusolverDngetrf_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  T* A,
  int lda,
  int* Lwork);

template <>
inline cusolverStatus_t cusolverDngetrf_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  float* A,
  int lda,
  int* Lwork)
{
  return cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

template <>
inline cusolverStatus_t cusolverDngetrf_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  double* A,
  int lda,
  int* Lwork)
{
  return cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

/**
 * @defgroup Getrs cusolver getrs operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDngetrs(cusolverDnHandle_t handle,  // NOLINT
                                 cublasOperation_t trans,
                                 int n,
                                 int nrhs,
                                 const T* A,
                                 int lda,
                                 const int* devIpiv,
                                 T* B,
                                 int ldb,
                                 int* devInfo,
                                 cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDngetrs(cusolverDnHandle_t handle,  // NOLINT
                                        cublasOperation_t trans,
                                        int n,
                                        int nrhs,
                                        const float* A,
                                        int lda,
                                        const int* devIpiv,
                                        float* B,
                                        int ldb,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}

template <>
inline cusolverStatus_t cusolverDngetrs(cusolverDnHandle_t handle,  // NOLINT
                                        cublasOperation_t trans,
                                        int n,
                                        int nrhs,
                                        const double* A,
                                        int lda,
                                        const int* devIpiv,
                                        double* B,
                                        int ldb,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}
/** @} */

/**
 * @defgroup syevd cusolver syevd operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnsyevd_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int n,
  const T* A,
  int lda,
  const T* W,
  int* lwork);

template <>
inline cusolverStatus_t cusolverDnsyevd_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int n,
  const float* A,
  int lda,
  const float* W,
  int* lwork)
{
  return cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
}

template <>
inline cusolverStatus_t cusolverDnsyevd_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int n,
  const double* A,
  int lda,
  const double* W,
  int* lwork)
{
  return cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
}
/** @} */

/**
 * @defgroup syevj cusolver syevj operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnsyevj(cusolverDnHandle_t handle,  // NOLINT
                                 cusolverEigMode_t jobz,
                                 cublasFillMode_t uplo,
                                 int n,
                                 T* A,
                                 int lda,
                                 T* W,
                                 T* work,
                                 int lwork,
                                 int* info,
                                 syevjInfo_t params,
                                 cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnsyevj(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int n,
  float* A,
  int lda,
  float* W,
  float* work,
  int lwork,
  int* info,
  syevjInfo_t params,
  cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params);
}

template <>
inline cusolverStatus_t cusolverDnsyevj(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int n,
  double* A,
  int lda,
  double* W,
  double* work,
  int lwork,
  int* info,
  syevjInfo_t params,
  cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params);
}

template <typename T>
cusolverStatus_t cusolverDnsyevj_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int n,
  const T* A,
  int lda,
  const T* W,
  int* lwork,
  syevjInfo_t params);

template <>
inline cusolverStatus_t cusolverDnsyevj_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int n,
  const float* A,
  int lda,
  const float* W,
  int* lwork,
  syevjInfo_t params)
{
  return cusolverDnSsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params);
}

template <>
inline cusolverStatus_t cusolverDnsyevj_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int n,
  const double* A,
  int lda,
  const double* W,
  int* lwork,
  syevjInfo_t params)
{
  return cusolverDnDsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params);
}
/** @} */

/**
 * @defgroup syevd cusolver syevd operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnsyevd(cusolverDnHandle_t handle,  // NOLINT
                                 cusolverEigMode_t jobz,
                                 cublasFillMode_t uplo,
                                 int n,
                                 T* A,
                                 int lda,
                                 T* W,
                                 T* work,
                                 int lwork,
                                 int* devInfo,
                                 cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnsyevd(cusolverDnHandle_t handle,  // NOLINT
                                        cusolverEigMode_t jobz,
                                        cublasFillMode_t uplo,
                                        int n,
                                        float* A,
                                        int lda,
                                        float* W,
                                        float* work,
                                        int lwork,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo);
}

template <>
inline cusolverStatus_t cusolverDnsyevd(cusolverDnHandle_t handle,  // NOLINT
                                        cusolverEigMode_t jobz,
                                        cublasFillMode_t uplo,
                                        int n,
                                        double* A,
                                        int lda,
                                        double* W,
                                        double* work,
                                        int lwork,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo);
}
/** @} */

#if CUDART_VERSION >= 10010
/**
 * @defgroup syevdx cusolver syevdx operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnsyevdx_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cusolverEigRange_t range,
  cublasFillMode_t uplo,
  int n,
  const T* A,
  int lda,
  T vl,
  T vu,
  int il,
  int iu,
  int* h_meig,
  const T* W,
  int* lwork);

template <>
inline cusolverStatus_t cusolverDnsyevdx_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cusolverEigRange_t range,
  cublasFillMode_t uplo,
  int n,
  const float* A,
  int lda,
  float vl,
  float vu,
  int il,
  int iu,
  int* h_meig,
  const float* W,
  int* lwork)
{
  return cusolverDnSsyevdx_bufferSize(
    handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, h_meig, W, lwork);
}

template <>
inline cusolverStatus_t cusolverDnsyevdx_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cusolverEigRange_t range,
  cublasFillMode_t uplo,
  int n,
  const double* A,
  int lda,
  double vl,
  double vu,
  int il,
  int iu,
  int* h_meig,
  const double* W,
  int* lwork)
{
  return cusolverDnDsyevdx_bufferSize(
    handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, h_meig, W, lwork);
}

template <typename T>
cusolverStatus_t cusolverDnsyevdx(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cusolverEigRange_t range,
  cublasFillMode_t uplo,
  int n,
  T* A,
  int lda,
  T vl,
  T vu,
  int il,
  int iu,
  int* h_meig,
  T* W,
  T* work,
  int lwork,
  int* devInfo,
  cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnsyevdx(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cusolverEigRange_t range,
  cublasFillMode_t uplo,
  int n,
  float* A,
  int lda,
  float vl,
  float vu,
  int il,
  int iu,
  int* h_meig,
  float* W,
  float* work,
  int lwork,
  int* devInfo,
  cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSsyevdx(
    handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, h_meig, W, work, lwork, devInfo);
}

template <>
inline cusolverStatus_t cusolverDnsyevdx(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cusolverEigRange_t range,
  cublasFillMode_t uplo,
  int n,
  double* A,
  int lda,
  double vl,
  double vu,
  int il,
  int iu,
  int* h_meig,
  double* W,
  double* work,
  int lwork,
  int* devInfo,
  cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDsyevdx(
    handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, h_meig, W, work, lwork, devInfo);
}
/** @} */
#endif

/**
 * @defgroup svd cusolver svd operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDngesvd_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  int* lwork)
{
  if (std::is_same<std::decay_t<T>, float>::value) {
    return cusolverDnSgesvd_bufferSize(handle, m, n, lwork);
  } else {
    return cusolverDnDgesvd_bufferSize(handle, m, n, lwork);
  }
}
template <typename T>
cusolverStatus_t cusolverDngesvd(  // NOLINT
  cusolverDnHandle_t handle,
  signed char jobu,
  signed char jobvt,
  int m,
  int n,
  T* A,
  int lda,
  T* S,
  T* U,
  int ldu,
  T* VT,
  int ldvt,
  T* work,
  int lwork,
  T* rwork,
  int* devInfo,
  cudaStream_t stream);
template <>
inline cusolverStatus_t cusolverDngesvd(  // NOLINT
  cusolverDnHandle_t handle,
  signed char jobu,
  signed char jobvt,
  int m,
  int n,
  float* A,
  int lda,
  float* S,
  float* U,
  int ldu,
  float* VT,
  int ldvt,
  float* work,
  int lwork,
  float* rwork,
  int* devInfo,
  cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSgesvd(
    handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, devInfo);
}
template <>
inline cusolverStatus_t cusolverDngesvd(  // NOLINT
  cusolverDnHandle_t handle,
  signed char jobu,
  signed char jobvt,
  int m,
  int n,
  double* A,
  int lda,
  double* S,
  double* U,
  int ldu,
  double* VT,
  int ldvt,
  double* work,
  int lwork,
  double* rwork,
  int* devInfo,
  cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDgesvd(
    handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, devInfo);
}

template <typename T>
inline cusolverStatus_t CUSOLVERAPI cusolverDngesvdj_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  int econ,
  int m,
  int n,
  const T* A,
  int lda,
  const T* S,
  const T* U,
  int ldu,
  const T* V,
  int ldv,
  int* lwork,
  gesvdjInfo_t params);
template <>
inline cusolverStatus_t CUSOLVERAPI cusolverDngesvdj_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  int econ,
  int m,
  int n,
  const float* A,
  int lda,
  const float* S,
  const float* U,
  int ldu,
  const float* V,
  int ldv,
  int* lwork,
  gesvdjInfo_t params)
{
  return cusolverDnSgesvdj_bufferSize(
    handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params);
}
template <>
inline cusolverStatus_t CUSOLVERAPI cusolverDngesvdj_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  int econ,
  int m,
  int n,
  const double* A,
  int lda,
  const double* S,
  const double* U,
  int ldu,
  const double* V,
  int ldv,
  int* lwork,
  gesvdjInfo_t params)
{
  return cusolverDnDgesvdj_bufferSize(
    handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params);
}
template <typename T>
inline cusolverStatus_t CUSOLVERAPI cusolverDngesvdj(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  int econ,
  int m,
  int n,
  T* A,
  int lda,
  T* S,
  T* U,
  int ldu,
  T* V,
  int ldv,
  T* work,
  int lwork,
  int* info,
  gesvdjInfo_t params,
  cudaStream_t stream);
template <>
inline cusolverStatus_t CUSOLVERAPI cusolverDngesvdj(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  int econ,
  int m,
  int n,
  float* A,
  int lda,
  float* S,
  float* U,
  int ldu,
  float* V,
  int ldv,
  float* work,
  int lwork,
  int* info,
  gesvdjInfo_t params,
  cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSgesvdj(
    handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params);
}
template <>
inline cusolverStatus_t CUSOLVERAPI cusolverDngesvdj(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  int econ,
  int m,
  int n,
  double* A,
  int lda,
  double* S,
  double* U,
  int ldu,
  double* V,
  int ldv,
  double* work,
  int lwork,
  int* info,
  gesvdjInfo_t params,
  cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDgesvdj(
    handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params);
}
/** @} */

/**
 * @defgroup potrf cusolver potrf operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnpotrf_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cublasFillMode_t uplo,
  int n,
  T* A,
  int lda,
  int* Lwork);

template <>
inline cusolverStatus_t cusolverDnpotrf_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cublasFillMode_t uplo,
  int n,
  float* A,
  int lda,
  int* Lwork)
{
  return cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, Lwork);
}

template <>
inline cusolverStatus_t cusolverDnpotrf_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cublasFillMode_t uplo,
  int n,
  double* A,
  int lda,
  int* Lwork)
{
  return cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, Lwork);
}

template <typename T>
inline cusolverStatus_t cusolverDnpotrf(cusolverDnHandle_t handle,  // NOLINT
                                        cublasFillMode_t uplo,
                                        int n,
                                        T* A,
                                        int lda,
                                        T* Workspace,
                                        int Lwork,
                                        int* devInfo,
                                        cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnpotrf(cusolverDnHandle_t handle,  // NOLINT
                                        cublasFillMode_t uplo,
                                        int n,
                                        float* A,
                                        int lda,
                                        float* Workspace,
                                        int Lwork,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}

template <>
inline cusolverStatus_t cusolverDnpotrf(cusolverDnHandle_t handle,  // NOLINT
                                        cublasFillMode_t uplo,
                                        int n,
                                        double* A,
                                        int lda,
                                        double* Workspace,
                                        int Lwork,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}
/** @} */

/**
 * @defgroup potrs cusolver potrs operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnpotrs(cusolverDnHandle_t handle,  // NOLINT
                                 cublasFillMode_t uplo,
                                 int n,
                                 int nrhs,
                                 const T* A,
                                 int lda,
                                 T* B,
                                 int ldb,
                                 int* devInfo,
                                 cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnpotrs(cusolverDnHandle_t handle,  // NOLINT
                                        cublasFillMode_t uplo,
                                        int n,
                                        int nrhs,
                                        const float* A,
                                        int lda,
                                        float* B,
                                        int ldb,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
}

template <>
inline cusolverStatus_t cusolverDnpotrs(cusolverDnHandle_t handle,  // NOLINT
                                        cublasFillMode_t uplo,
                                        int n,
                                        int nrhs,
                                        const double* A,
                                        int lda,
                                        double* B,
                                        int ldb,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
}
/** @} */

/**
 * @defgroup geqrf cusolver geqrf operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDngeqrf(cusolverDnHandle_t handle,
                                 int m,  // NOLINT
                                 int n,
                                 T* A,
                                 int lda,
                                 T* TAU,
                                 T* Workspace,
                                 int Lwork,
                                 int* devInfo,
                                 cudaStream_t stream);
template <>
inline cusolverStatus_t cusolverDngeqrf(cusolverDnHandle_t handle,  // NOLINT
                                        int m,
                                        int n,
                                        float* A,
                                        int lda,
                                        float* TAU,
                                        float* Workspace,
                                        int Lwork,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}
template <>
inline cusolverStatus_t cusolverDngeqrf(cusolverDnHandle_t handle,  // NOLINT
                                        int m,
                                        int n,
                                        double* A,
                                        int lda,
                                        double* TAU,
                                        double* Workspace,
                                        int Lwork,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}

template <typename T>
cusolverStatus_t cusolverDngeqrf_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  T* A,
  int lda,
  int* Lwork);
template <>
inline cusolverStatus_t cusolverDngeqrf_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  float* A,
  int lda,
  int* Lwork)
{
  return cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, Lwork);
}
template <>
inline cusolverStatus_t cusolverDngeqrf_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  double* A,
  int lda,
  int* Lwork)
{
  return cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, Lwork);
}
/** @} */

/**
 * @defgroup orgqr cusolver orgqr operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnorgqr(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  int k,
  T* A,
  int lda,
  const T* tau,
  T* work,
  int lwork,
  int* devInfo,
  cudaStream_t stream);
template <>
inline cusolverStatus_t cusolverDnorgqr(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  int k,
  float* A,
  int lda,
  const float* tau,
  float* work,
  int lwork,
  int* devInfo,
  cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}
template <>
inline cusolverStatus_t cusolverDnorgqr(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  int k,
  double* A,
  int lda,
  const double* tau,
  double* work,
  int lwork,
  int* devInfo,
  cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}

template <typename T>
cusolverStatus_t cusolverDnorgqr_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  int k,
  const T* A,
  int lda,
  const T* TAU,
  int* lwork);
template <>
inline cusolverStatus_t cusolverDnorgqr_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  int k,
  const float* A,
  int lda,
  const float* TAU,
  int* lwork)
{
  return cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, TAU, lwork);
}
template <>
inline cusolverStatus_t cusolverDnorgqr_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  int k,
  const double* A,
  int lda,
  const double* TAU,
  int* lwork)
{
  return cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, TAU, lwork);
}
/** @} */

/**
 * @defgroup ormqr cusolver ormqr operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnormqr(cusolverDnHandle_t handle,  // NOLINT
                                 cublasSideMode_t side,
                                 cublasOperation_t trans,
                                 int m,
                                 int n,
                                 int k,
                                 const T* A,
                                 int lda,
                                 const T* tau,
                                 T* C,
                                 int ldc,
                                 T* work,
                                 int lwork,
                                 int* devInfo,
                                 cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnormqr(  // NOLINT
  cusolverDnHandle_t handle,
  cublasSideMode_t side,
  cublasOperation_t trans,
  int m,
  int n,
  int k,
  const float* A,
  int lda,
  const float* tau,
  float* C,
  int ldc,
  float* work,
  int lwork,
  int* devInfo,
  cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
}

template <>
inline cusolverStatus_t cusolverDnormqr(  // NOLINT
  cusolverDnHandle_t handle,
  cublasSideMode_t side,
  cublasOperation_t trans,
  int m,
  int n,
  int k,
  const double* A,
  int lda,
  const double* tau,
  double* C,
  int ldc,
  double* work,
  int lwork,
  int* devInfo,
  cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
}

template <typename T>
cusolverStatus_t cusolverDnormqr_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cublasSideMode_t side,
  cublasOperation_t trans,
  int m,
  int n,
  int k,
  const T* A,
  int lda,
  const T* tau,
  const T* C,
  int ldc,
  int* lwork);

template <>
inline cusolverStatus_t cusolverDnormqr_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cublasSideMode_t side,
  cublasOperation_t trans,
  int m,
  int n,
  int k,
  const float* A,
  int lda,
  const float* tau,
  const float* C,
  int ldc,
  int* lwork)
{
  return cusolverDnSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}

template <>
inline cusolverStatus_t cusolverDnormqr_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cublasSideMode_t side,
  cublasOperation_t trans,
  int m,
  int n,
  int k,
  const double* A,
  int lda,
  const double* tau,
  const double* C,
  int ldc,
  int* lwork)
{
  return cusolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}
/** @} */

/**
 * @defgroup csrqrBatched cusolver batched
 * @{
 */
template <typename T>
cusolverStatus_t cusolverSpcsrqrBufferInfoBatched(  // NOLINT
  cusolverSpHandle_t handle,
  int m,
  int n,
  int nnzA,
  const cusparseMatDescr_t descrA,
  const T* csrValA,
  const int* csrRowPtrA,
  const int* csrColIndA,
  int batchSize,
  csrqrInfo_t info,
  size_t* internalDataInBytes,
  size_t* workspaceInBytes);

template <>
inline cusolverStatus_t cusolverSpcsrqrBufferInfoBatched(  // NOLINT
  cusolverSpHandle_t handle,
  int m,
  int n,
  int nnzA,
  const cusparseMatDescr_t descrA,
  const float* csrValA,
  const int* csrRowPtrA,
  const int* csrColIndA,
  int batchSize,
  csrqrInfo_t info,
  size_t* internalDataInBytes,
  size_t* workspaceInBytes)
{
  return cusolverSpScsrqrBufferInfoBatched(handle,
                                           m,
                                           n,
                                           nnzA,
                                           descrA,
                                           csrValA,
                                           csrRowPtrA,
                                           csrColIndA,
                                           batchSize,
                                           info,
                                           internalDataInBytes,
                                           workspaceInBytes);
}

template <>
inline cusolverStatus_t cusolverSpcsrqrBufferInfoBatched(  // NOLINT
  cusolverSpHandle_t handle,
  int m,
  int n,
  int nnzA,
  const cusparseMatDescr_t descrA,
  const double* csrValA,
  const int* csrRowPtrA,
  const int* csrColIndA,
  int batchSize,
  csrqrInfo_t info,
  size_t* internalDataInBytes,
  size_t* workspaceInBytes)
{
  return cusolverSpDcsrqrBufferInfoBatched(handle,
                                           m,
                                           n,
                                           nnzA,
                                           descrA,
                                           csrValA,
                                           csrRowPtrA,
                                           csrColIndA,
                                           batchSize,
                                           info,
                                           internalDataInBytes,
                                           workspaceInBytes);
}

template <typename T>
cusolverStatus_t cusolverSpcsrqrsvBatched(  // NOLINT
  cusolverSpHandle_t handle,
  int m,
  int n,
  int nnzA,
  const cusparseMatDescr_t descrA,
  const T* csrValA,
  const int* csrRowPtrA,
  const int* csrColIndA,
  const T* b,
  T* x,
  int batchSize,
  csrqrInfo_t info,
  void* pBuffer,
  cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverSpcsrqrsvBatched(  // NOLINT
  cusolverSpHandle_t handle,
  int m,
  int n,
  int nnzA,
  const cusparseMatDescr_t descrA,
  const float* csrValA,
  const int* csrRowPtrA,
  const int* csrColIndA,
  const float* b,
  float* x,
  int batchSize,
  csrqrInfo_t info,
  void* pBuffer,
  cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverSpSetStream(handle, stream));
  return cusolverSpScsrqrsvBatched(
    handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer);
}

template <>
inline cusolverStatus_t cusolverSpcsrqrsvBatched(  // NOLINT
  cusolverSpHandle_t handle,
  int m,
  int n,
  int nnzA,
  const cusparseMatDescr_t descrA,
  const double* csrValA,
  const int* csrRowPtrA,
  const int* csrColIndA,
  const double* b,
  double* x,
  int batchSize,
  csrqrInfo_t info,
  void* pBuffer,
  cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverSpSetStream(handle, stream));
  return cusolverSpDcsrqrsvBatched(
    handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer);
}
/** @} */

#if CUDART_VERSION >= 11010
/**
 * @defgroup DnXsyevd cusolver DnXsyevd operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnxsyevd_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverDnParams_t params,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int64_t n,
  const T* A,
  int64_t lda,
  const T* W,
  size_t* workspaceInBytesOnDevice,
  size_t* workspaceInBytesOnHost,
  cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnxsyevd_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverDnParams_t params,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int64_t n,
  const float* A,
  int64_t lda,
  const float* W,
  size_t* workspaceInBytesOnDevice,
  size_t* workspaceInBytesOnHost,
  cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnXsyevd_bufferSize(handle,
                                     params,
                                     jobz,
                                     uplo,
                                     n,
                                     CUDA_R_32F,
                                     A,
                                     lda,
                                     CUDA_R_32F,
                                     W,
                                     CUDA_R_32F,
                                     workspaceInBytesOnDevice,
                                     workspaceInBytesOnHost);
}

template <>
inline cusolverStatus_t cusolverDnxsyevd_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverDnParams_t params,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int64_t n,
  const double* A,
  int64_t lda,
  const double* W,
  size_t* workspaceInBytesOnDevice,
  size_t* workspaceInBytesOnHost,
  cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnXsyevd_bufferSize(handle,
                                     params,
                                     jobz,
                                     uplo,
                                     n,
                                     CUDA_R_64F,
                                     A,
                                     lda,
                                     CUDA_R_64F,
                                     W,
                                     CUDA_R_64F,
                                     workspaceInBytesOnDevice,
                                     workspaceInBytesOnHost);
}

template <typename T>
cusolverStatus_t cusolverDnxsyevd(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverDnParams_t params,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int64_t n,
  T* A,
  int64_t lda,
  T* W,
  T* bufferOnDevice,
  size_t workspaceInBytesOnDevice,
  T* bufferOnHost,
  size_t workspaceInBytesOnHost,
  int* info,
  cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnxsyevd(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverDnParams_t params,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int64_t n,
  float* A,
  int64_t lda,
  float* W,
  float* bufferOnDevice,
  size_t workspaceInBytesOnDevice,
  float* bufferOnHost,
  size_t workspaceInBytesOnHost,
  int* info,
  cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnXsyevd(handle,
                          params,
                          jobz,
                          uplo,
                          n,
                          CUDA_R_32F,
                          A,
                          lda,
                          CUDA_R_32F,
                          W,
                          CUDA_R_32F,
                          bufferOnDevice,
                          workspaceInBytesOnDevice,
                          bufferOnHost,
                          workspaceInBytesOnHost,
                          info);
}

template <>
inline cusolverStatus_t cusolverDnxsyevd(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverDnParams_t params,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int64_t n,
  double* A,
  int64_t lda,
  double* W,
  double* bufferOnDevice,
  size_t workspaceInBytesOnDevice,
  double* bufferOnHost,
  size_t workspaceInBytesOnHost,
  int* info,
  cudaStream_t stream)
{
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnXsyevd(handle,
                          params,
                          jobz,
                          uplo,
                          n,
                          CUDA_R_64F,
                          A,
                          lda,
                          CUDA_R_64F,
                          W,
                          CUDA_R_64F,
                          bufferOnDevice,
                          workspaceInBytesOnDevice,
                          bufferOnHost,
                          workspaceInBytesOnHost,
                          info);
}
/** @} */
#endif

}  // namespace linalg
}  // namespace raft
