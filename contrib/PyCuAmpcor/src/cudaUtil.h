/**
 * @file cudaUtil.h
 * @brief Various cuda related parameters and utilities
 *
 * Some routines are adapted from Nvidia CUDA samples/common/inc/help_cuda.h
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 **/

#ifndef __CUDAUTIL_H
#define __CUDAUTIL_H

// for 2D FFT
#define NRANK 2

//typical choices of number of threads in a block
// for processing 1D and 2D arrays
#define NTHREADS 512  //
#define NTHREADS2D 16 //

#define WARPSIZE 32
#define MAXTHREADS 1024  //2048 for newer GPUs

#ifdef __FERMI__ //2.0: M2090
#define MAXBLOCKS 65535  //x
#define MAXBLOCKS2 65535 //y,z
#else //2.0 and above : K40, ...
#define MAXBLOCKS 4294967295 //x
#define MAXBLOCKS2 65535  //y,z
#endif

#define IDX2R(i,j,NJ) (((i)*(NJ))+(j))  //row-major order
#define IDX2C(i,j,NI) (((j)*(NI))+(i))  //col-major order

#define IDIVUP(i,j) ((i+j-1)/j)

#define IMUL(a, b) __mul24(a, b)

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

#ifndef MIN
#define MIN(a,b) (a > b ? b: a)
#endif

// compute the next integer in power of 2
inline int nextpower2(int value)
{
    int r=1;
    while (r<value) r<<=1;
    return r;
}

// General GPU Device CUDA Initialization
int gpuDeviceInit(int devID);

// This function lists all available GPUs
void gpuDeviceList();

#endif //__CUDAUTIL_H
//end of file
