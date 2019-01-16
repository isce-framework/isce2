/*
    cuEstimateStats.cu

    9/23/2017, Minyan Zhong
*/

#include "cuArrays.h"
#include "float2.h"
#include <cfloat>
#include "cudaUtil.h"
#include "cudaError.h"
#include "cuAmpcorUtil.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

template <const int BLOCKSIZE>
__global__ void cudaKernel_estimateSnr(const float* corrSum, const int* corrValidCount, const float* maxval, float* snrValue, const int size)

{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= size) return;

    float mean = (corrSum[idx] - maxval[idx] * maxval[idx]) / (corrValidCount[idx] - 1);

    snrValue[idx] = maxval[idx] / mean;
}

void cuEstimateSnr(cuArrays<float> *corrSum, cuArrays<int> *corrValidCount, cuArrays<float> *maxval, cuArrays<float> *snrValue, cudaStream_t stream)
{

    int size = corrSum->getSize();

    //std::cout<<size<<std::endl;

    //corrSum->allocateHost();

    //corrSum->copyToHost(stream);

    //std::cout<<"corr sum"<<std::endl;

    //corrValidCount->allocateHost();

    //corrValidCount->copyToHost(stream);

    //std::cout<<"valid count"<<std::endl;

    //maxval->allocateHost();

    //maxval->copyToHost(stream);

    //std::cout<<"maxval"<<std::endl;


    //for (int i=0; i<size; i++){
        
    //    std::cout<<corrSum->hostData[i]<<std::endl;
    //    std::cout<<corrValidCount->hostData[i]<<std::endl;

    //    std::cout<<maxval->hostData[i]<<std::endl;

    //}

    cudaKernel_estimateSnr<NTHREADS><<< IDIVUP(size, NTHREADS), NTHREADS, 0, stream>>>
        (corrSum->devData, corrValidCount->devData, maxval->devData, snrValue->devData, size);

    getLastCudaError("cuda kernel estimate stats error\n");
}
