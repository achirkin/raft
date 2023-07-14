# =============================================================================
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
# Note: This file was copied from PyTorch and modified for use in the RAFT library.
# Refer to thirdparty/LICENSES/LICENSE.pytorch for license and additional
# copyright information.
# =============================================================================

INCLUDE(CheckCXXSourceRuns)

SET(AVX_CODE
    "
  #include <immintrin.h>

  int main()
  {
    __m256 a;
    a = _mm256_set1_ps(0);
    return 0;
  }
"
)

SET(AVX512_CODE
    "
  #include <immintrin.h>

  int main()
  {
    __m512i a = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0);
    __m512i b = a;
    __mmask64 equality_mask = _mm512_cmp_epi8_mask(a, b, _MM_CMPINT_EQ);
    return 0;
  }
"
)

SET(AVX2_CODE
    "
  #include <immintrin.h>

  int main()
  {
    __m256i a = {0};
    a = _mm256_abs_epi16(a);
    __m256i x;
    _mm256_extract_epi64(x, 0); // we rely on this in our AVX2 code
    return 0;
  }
"
)

SET(NEON_CODE "
#include <arm_neon.h>

int main()
{
  uint8x16_t a = vmovq_n_u8(1);
  uint8x16_t b = vmovq_n_u8(2);
  uint8x16_t c = vaddq_u8(a, b);
  return 0;
}
")

SET(SVE_CODE "
#include <arm_sve.h>

#ifndef SVE_BITS
#define SVE_BITS __ARM_FEATURE_SVE_BITS
#endif

int main()
{
    __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS))) svbool_t p = svptrue_b32();
    __attribute__((arm_sve_vector_bits(SVE_BITS))) svfloat32_t a = svdup_f32(1.0f);
    svfloat32_t b = svdup_f32(2.0f);
    svfloat32_t c = svadd_f32_z(p, a, b);
    return sizeof(a)*8 != __ARM_FEATURE_SVE_BITS;
}
")

MACRO(CHECK_SSE lang type flags)
  SET(__FLAG_I 1)
  SET(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
  FOREACH(__FLAG ${flags})
    IF(NOT ${lang}_${type}_FOUND)
      SET(CMAKE_REQUIRED_FLAGS ${__FLAG})
      CHECK_CXX_SOURCE_RUNS("${${type}_CODE}" ${lang}_HAS_${type}_${__FLAG_I})
      IF(${lang}_HAS_${type}_${__FLAG_I})
        SET(${lang}_${type}_FOUND
            TRUE
            CACHE BOOL "${lang} ${type} support"
        )
        separate_arguments(__SEPARATED_FLAG UNIX_COMMAND ${__FLAG})
        SET(${lang}_${type}_FLAGS
            ${__SEPARATED_FLAG}
            CACHE STRING "${lang} ${type} flags"
        )
      ENDIF()
      MATH(EXPR __FLAG_I "${__FLAG_I}+1")
    ENDIF()
  ENDFOREACH()
  SET(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

  IF(NOT ${lang}_${type}_FOUND)
    SET(${lang}_${type}_FOUND
        FALSE
        CACHE BOOL "${lang} ${type} support"
    )
    SET(${lang}_${type}_FLAGS
        ""
        CACHE STRING "${lang} ${type} flags"
    )
  ENDIF()

  MARK_AS_ADVANCED(${lang}_${type}_FOUND ${lang}_${type}_FLAGS)

ENDMACRO()

# CHECK_SSE(C "AVX" " ;-mavx;/arch:AVX") CHECK_SSE(C "AVX2" " ;-mavx2 -mfma;/arch:AVX2") CHECK_SSE(C
# "AVX512" " ;-mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma;/arch:AVX512")
#
CHECK_SSE(CXX "AVX" " ;-mavx;/arch:AVX")
CHECK_SSE(CXX "AVX2" " ;-mavx2 -mfma;/arch:AVX2")
CHECK_SSE(CXX "AVX512" " ;-mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma;/arch:AVX512")
CHECK_SSE(CXX "NEON" " ;-mfpu=neon")
CHECK_SSE(CXX "SVE" "\
 ;\
 -march=armv9-a+sve -msve-vector-bits=2048;-march=armv9-a+sve -msve-vector-bits=1024;-march=armv9-a+sve -msve-vector-bits=512;\
 -march=armv8.6-a+sve -msve-vector-bits=2048;-march=armv8.6-a+sve -msve-vector-bits=1024;-march=armv8.6-a+sve -msve-vector-bits=512")
