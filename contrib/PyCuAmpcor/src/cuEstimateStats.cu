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

    snrValue[idx] = maxval[idx] * maxval[idx] / mean;
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


template <const int BLOCKSIZE> // number of threads per block.
__global__ void cudaKernel_estimateVar(const float* corrBatchRaw, const int NX, const int NY, const int2* maxloc, const float* maxval, float3* covValue, const int size)
{

    // Find image id.
    int idxImage = threadIdx.x + blockDim.x*blockIdx.x;

    if (idxImage >= size) return;

    // Preparation.
    int px = maxloc[idxImage].x;
    int py = maxloc[idxImage].y;
    float peak = maxval[idxImage];

    // Check if maxval is on the margin.
    if (px-1 < 0 || py-1 <0 || px + 1 >=NX || py+1 >=NY)  {

        covValue[idxImage] = make_float3(99.0, 99.0, 99.0);

    }
    else {
        int offset = NX * NY * idxImage;
        int idx00 = offset + (px - 1) * NY + py - 1;
        int idx01 = offset + (px - 1) * NY + py    ;
        int idx02 = offset + (px - 1) * NY + py + 1;
        int idx10 = offset + (px    ) * NY + py - 1;
        int idx11 = offset + (px    ) * NY + py    ;
        int idx12 = offset + (px    ) * NY + py + 1;
        int idx20 = offset + (px + 1) * NY + py - 1;
        int idx21 = offset + (px + 1) * NY + py    ;
        int idx22 = offset + (px + 1) * NY + py + 1;

        float dxx = - ( corrBatchRaw[idx21] + corrBatchRaw[idx01] - 2*corrBatchRaw[idx11] ) * 0.5;
        float dyy = - ( corrBatchRaw[idx12] + corrBatchRaw[idx10] - 2*corrBatchRaw[idx11] ) * 0.5;
        float dxy = - ( corrBatchRaw[idx22] + corrBatchRaw[idx00] - corrBatchRaw[idx20] - corrBatchRaw[idx02] ) *0.25;

        float n2 = fmaxf(1 - peak, 0.0);

        int winSize = NX*NY;

        dxx = dxx * winSize;
        dyy = dyy * winSize;
        dxy = dxy * winSize;

        float n4 = n2*n2;
        n2 = n2 * 2;
        n4 = n4 * 0.5 * winSize;

        float u = dxy * dxy - dxx * dyy;
        float u2 = u*u;

        if (fabsf(u) < 1e-2) {

            covValue[idxImage] = make_float3(99.0, 99.0, 99.0);

        }
        else {
                float cov_xx = (- n2 * u * dyy + n4 * ( dyy*dyy + dxy*dxy) ) / u2;
                float cov_yy = (- n2 * u * dxx + n4 * ( dxx*dxx + dxy*dxy) ) / u2;
                float cov_xy = (  n2 * u * dxy - n4 * ( dxx + dyy ) * dxy ) / u2;
                covValue[idxImage] = make_float3(cov_xx, cov_yy, cov_xy);
        }
    }
}

void cuEstimateVariance(cuArrays<float> *corrBatchRaw, cuArrays<int2> *maxloc, cuArrays<float> *maxval, cuArrays<float3> *covValue, cudaStream_t stream)
{

    int size = corrBatchRaw->count;

    // One dimensional launching parameters to loop over every correlation surface.
    cudaKernel_estimateVar<NTHREADS><<< IDIVUP(size, NTHREADS), NTHREADS, 0, stream>>>
        (corrBatchRaw->devData, corrBatchRaw->height, corrBatchRaw->width, maxloc->devData, maxval->devData, covValue->devData, size);
    getLastCudaError("cudaKernel_estimateVar error\n");
}
