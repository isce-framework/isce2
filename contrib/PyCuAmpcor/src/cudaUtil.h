/** 
 * cudaUtil.h
 * Purpose: various cuda related parameters and utilities
 * 
 * Some routines are adapted from Nvidia CUDA samples/common/inc/help_cuda.h
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 * 
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


inline int nextpower2(int value)
{
    int r=1;
    while (r<value) r<<=1;
    return r;
}

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
        { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
        { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
        { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
        { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}
// end of GPU Architecture definitions

#ifdef __CUDA_RUNTIME_H__
  
// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId()
{
    int current_device     = 0, sm_per_multiproc  = 0;
    int max_perf_device    = 0;
    int device_count       = 0, best_SM_arch      = 0;
    int devices_prohibited = 0;
    
    unsigned long long max_compute_perf = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount(&device_count);
    
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    if (device_count == 0)
    {
        fprintf(stderr, "gpuGetMaxGflopsDeviceId() CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    // Find the best major SM Architecture GPU device
    while (current_device < device_count)
    {
        cudaGetDeviceProperties(&deviceProp, current_device);

        // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
        if (deviceProp.computeMode != cudaComputeModeProhibited)
        {
            if (deviceProp.major > 0 && deviceProp.major < 9999)
            {
                best_SM_arch = MAX(best_SM_arch, deviceProp.major);
            }
        }
        else
        {
            devices_prohibited++;
        }

        current_device++;
    }

    if (devices_prohibited == device_count)
    {
    	fprintf(stderr, "gpuGetMaxGflopsDeviceId() CUDA error: all devices have compute mode prohibited.\n");
    	exit(EXIT_FAILURE);
    }

    // Find the best CUDA capable GPU device
    current_device = 0;

    while (current_device < device_count)
    {
        cudaGetDeviceProperties(&deviceProp, current_device);

        // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
        if (deviceProp.computeMode != cudaComputeModeProhibited)
        {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
            {
                sm_per_multiproc = 1;
            }
            else
            {
                sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
            }

            unsigned long long compute_perf  = (unsigned long long) deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;

            //fprintf(stderr, "Device %d has performamce %llu.\n", current_device, compute_perf);

            if (compute_perf  > max_compute_perf)
            {
                /* Let the GPU with max flops win! --LJ  
                // If we find GPU with SM major > 2, search only these
                if (best_SM_arch > 2)
                {
                    // If our device==best_SM_arch, choose this, or else pass
                    if (deviceProp.major == best_SM_arch)
                    {
                        max_compute_perf  = compute_perf;
                        max_perf_device   = current_device;
                    }
                }
                else
                {
                    max_compute_perf  = compute_perf;
                    max_perf_device   = current_device;
                } 
                */
                max_compute_perf  = compute_perf;
                max_perf_device   = current_device;
                
            }
        }

        ++current_device;
    }

    return max_perf_device;
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
        fprintf(stderr, "gpuDeviceInit() Finding the GPU with max GFlops instead ...\n");
        devID = gpuGetMaxGflopsDeviceId();
    }

    checkCudaErrors(cudaSetDevice(devID));
    printf("gpuDeviceInit() Using CUDA Device %d ...\n", devID);

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
    fprintf(stderr, "Device %d has the max Gflops\n", gpuGetMaxGflopsDeviceId());
}
#endif


#endif //__CUDAUTIL_H
