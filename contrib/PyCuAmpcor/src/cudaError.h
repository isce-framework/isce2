/**
 * cudaError.h
 * Purpose: check various errors in cuda/cufft/cublas calls
 * Lijun Zhu
 * Last modified 09/07/2017
**/

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions for initialization and error checking

#ifndef _CUDAERROR_CUH
#define _CUDAERROR_CUH

#pragma once

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cufft.h>
#include "debug.h"
#include <cuda.h>


#ifdef __DRIVER_TYPES_H__
#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET
#endif
#endif

template<typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
#ifdef CUDA_ERROR_CHECK
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n",
                file, line, static_cast<unsigned int>(result), func);
        DEVICE_RESET
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
#endif
}

// This will output the proper error string when calling cudaGetLastError
inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString(err));
        DEVICE_RESET
        exit(EXIT_FAILURE);
    }
#endif
}

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(val)  check ( (val), #val, __FILE__, __LINE__ )
#define cufft_Error(val)     check ( (val), #val, __FILE__, __LINE__ )
#define cublas_Error(val)    check ( (val), #val, __FILE__, __LINE__ )
#define getLastCudaError(var)   __getLastCudaError (var, __FILE__, __LINE__)

#endif //__CUDAERROR_CUH
