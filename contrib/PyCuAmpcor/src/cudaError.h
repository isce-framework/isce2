/**
 * @file  cudaError.h
 * @brief Define error checking in cuda calls
 *
**/

// code guard
#ifndef _CUDAERROR_CUH
#define _CUDAERROR_CUH

#pragma once

#include "debug.h"

template<typename T >
void check(T result, char const *const func, const char *const file, int const line);

// This will output the proper error string when calling cudaGetLastError
void __getLastCudaError(const char *errorMessage, const char *file, const int line);

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#ifdef CUDA_ERROR_CHECK
#define checkCudaErrors(val)  check ( (val), #val, __FILE__, __LINE__ )
#define cufft_Error(val)     check ( (val), #val, __FILE__, __LINE__ )
#define getLastCudaError(var)   __getLastCudaError (var, __FILE__, __LINE__)
#else
#define checkCudaErrors(val) val
#define cufft_Error(val)  val
#define getLastCudaError(val)
#endif //CUDA_ERROR_CHECK

#endif //__CUDAERROR_CUH
