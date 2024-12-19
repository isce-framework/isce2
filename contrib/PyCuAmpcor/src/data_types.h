/**
 * @file debug.h
 * @brief Define flags to control the debugging
 *
 * CUAMPCOR_DEBUG is used to output debugging information and intermediate results,
 *    disabled when NDEBUG macro is defined.
 * CUDA_ERROR_CHECK is always enabled, to check CUDA routine errors
 *
 */

// code guard
#ifndef __CUAMPCOR_DATA_TYPES_H
#define __CUAMPCOR_DATA_TYPES_H

#include <float.h>

// disable this for single precision version
// #define CUAMPCOR_DOUBLE

#ifdef CUAMPCOR_DOUBLE
    #define real_type double
    #define complex_type double2
    #define real2_type double2
    #define real3_type double3
    #define make_real2 make_double2
    #define make_complex_type make_double2
    #define EPSILON DBL_EPSILON
    #define REAL_MAX DBL_MAX
#else
    #define real_type float
    #define complex_type float2
    #define real2_type float2
    #define real3_type float3
    #define make_real2 make_float2
    #define make_complex_type make_float2
    #define EPSILON FLT_EPSILON
    #define REAL_MAX FLT_MAX
#endif

#define image_complex_type float2
#define image_real_type float

#endif //__CUAMPCOR_DATA_TYPES_H
//end of file