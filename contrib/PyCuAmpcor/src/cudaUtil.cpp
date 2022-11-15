#include "cudaUtil.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "cudaError.h"

int gpuDeviceInit(int devID)
{
    int device_count;
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    if (devID < 0 || devID > device_count - 1) {
        fprintf(stderr, "gpuDeviceInit() Device %d is not a valid GPU device. \n", devID);
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaSetDevice(devID));
    printf("Using CUDA Device %d ...\n", devID);

    return devID;
}

void gpuDeviceList()
{
    int device_count = 0;
    int current_device = 0;
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    fprintf(stderr, "Detecting all CUDA devices ...\n");
    if (device_count == 0) {
        fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    while (current_device < device_count) {
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, current_device));
        if (deviceProp.computeMode == cudaComputeModeProhibited) {
            fprintf(stderr, "CUDA Device [%d]: \"%s\" is not available: "
                    "device is running in <Compute Mode Prohibited> \n",
                    current_device, deviceProp.name);
        } else if (deviceProp.major < 1) {
            fprintf(stderr, "CUDA Device [%d]: \"%s\" is not available: "
                    "device does not support CUDA \n",
                    current_device, deviceProp.name);
        } else {
            fprintf(stderr, "CUDA Device [%d]: \"%s\" is available.\n",
                    current_device, deviceProp.name);
        }
        current_device++;
    }
}
