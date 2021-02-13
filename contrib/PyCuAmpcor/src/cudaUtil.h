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

#include <cuda_runtime.h>
#include "cudaError.h"

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

// Float To Int conversion
inline int ftoi(float value)
{
    return (value >= 0 ? (int)(value + 0.5) : (int)(value - 0.5));
}

// compute the next integer in power of 2
inline int nextpower2(int value)
{
    int r=1;
    while (r<value) r<<=1;
    return r;
}


// General GPU Device CUDA Initialization
inline int gpuDeviceInit(int devID)
{
    int device_count;
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    if (device_count == 0)
    {
        fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    if (devID < 0 || devID > device_count-1)
    {
        fprintf(stderr, "gpuDeviceInit() Device %d is not a valid GPU device. \n", devID);
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaSetDevice(devID));
    printf("Using CUDA Device %d ...\n", devID);

    return devID;
}

// This function lists all available GPUs
inline void gpuDeviceList()
{
    int device_count = 0;
    int current_device = 0;
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    fprintf(stderr, "Detecting all CUDA devices ...\n");
    if (device_count == 0)
    {
        fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    while (current_device < device_count)
    {
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, current_device));
        if (deviceProp.computeMode == cudaComputeModeProhibited)
        {
            fprintf(stderr, "CUDA Device [%d]: \"%s\" is not available: device is running in <Compute Mode Prohibited> \n", current_device, deviceProp.name);
        }
        else if (deviceProp.major < 1)
        {
            fprintf(stderr, "CUDA Device [%d]: \"%s\" is not available: device does not support CUDA \n", current_device, deviceProp.name);
        }
        else {
            fprintf(stderr, "CUDA Device [%d]: \"%s\" is available.\n", current_device, deviceProp.name);
        }
        current_device++;
    }
}

#endif //__CUDAUTIL_H
//end of file
