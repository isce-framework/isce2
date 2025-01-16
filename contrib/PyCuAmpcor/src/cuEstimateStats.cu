/**
 * @file  cuEstimateStats.cu
 * @brief Estimate the statistics of the correlation surface
 *
 * 9/23/2017, Minyan Zhong
 */

#include "cuArrays.h"
#include "float2.h"
#include "float.h"
#include "data_types.h"
#include "cudaUtil.h"
#include "cudaError.h"
#include "cuAmpcorUtil.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

// cuda kernel for cuEstimateSnr
__global__ void cudaKernel_estimateSnr(const real_type* corrSum, const real_type* maxval, real_type* snrValue, const int size, const int nImages)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= nImages) return;

    real_type peak =  maxval[idx];
    peak *= peak;
    real_type mean = (corrSum[idx] - peak) / (size - 1);
    snrValue[idx] = peak / mean;
#ifdef CUAMPCOR_DEBUG
    if(threadIdx.x==0)
        printf("debug snr %g %g %g\n", peak, mean, snrValue[idx]);
#endif
}

/**
 * Estimate the signal to noise ratio (SNR) of the correlation surface
 * @param[in] corrSum the sum of the correlation surface
 * @param[in] corrValidCount the number of valid pixels contributing to sum
 * @param[out] snrValue return snr value
 * @param[in] stream cuda stream
 */
void cuEstimateSnr(cuArrays<real_type> *corrSum, cuArrays<real_type> *maxval, cuArrays<real_type> *snrValue, const int size, cudaStream_t stream)
{
    int nImages = corrSum->getSize();
    cudaKernel_estimateSnr<<< IDIVUP(size, NTHREADS), NTHREADS, 0, stream>>>
        (corrSum->devData,  maxval->devData, snrValue->devData, size, nImages);
    getLastCudaError("cuda kernel estimate stats error\n");
}

__global__ void cudaKernel_estimateSnr(const float* corrSum, const int* corrValidCount, const float* maxval, float* snrValue, const int size)

{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= size) return;

    float mean = (corrSum[idx] - maxval[idx] * maxval[idx]) / (corrValidCount[idx] - 1);

    snrValue[idx] = maxval[idx] * maxval[idx] / mean;
}

/**
 * Estimate the signal to noise ratio (SNR) of the correlation surface
 * @param[in] corrSum the sum of the correlation surface
 * @param[in] corrValidCount the number of valid pixels contributing to sum
 * @param[out] snrValue return snr value
 * @param[in] stream cuda stream
 */
void cuEstimateSnr(cuArrays<float> *corrSum, cuArrays<int> *corrValidCount, cuArrays<float> *maxval, cuArrays<float> *snrValue, cudaStream_t stream)
{

    int size = corrSum->getSize();
    cudaKernel_estimateSnr<<< IDIVUP(size, NTHREADS), NTHREADS, 0, stream>>>
        (corrSum->devData, corrValidCount->devData, maxval->devData, snrValue->devData, size);
    getLastCudaError("cuda kernel estimate stats error\n");
}

// cuda kernel for cuEstimateVariance
__global__ void cudaKernel_estimateVar(const real_type* corrBatchRaw, const int NX, const int NY, const int2* maxloc,
        const real_type* maxval, const int templateSize, const int distance, real3_type* covValue, const int size)
{

    // Find image id.
    int idxImage = threadIdx.x + blockDim.x*blockIdx.x;

    if (idxImage >= size) return;

    // Preparation.
    int px = maxloc[idxImage].x;
    int py = maxloc[idxImage].y;
    real_type peak = maxval[idxImage];

    // Check if maxval is on the margin.
    if (px-distance < 0 || py-distance <0 || px + distance >=NX || py+distance >=NY)  {

        covValue[idxImage] = make_real3(99.0, 99.0, 0.0);

    }
    else {
        int offset = NX * NY * idxImage;
        int idx00 = offset + (px - distance) * NY + py - distance;
        int idx01 = offset + (px - distance) * NY + py    ;
        int idx02 = offset + (px - distance) * NY + py + distance;
        int idx10 = offset + (px    ) * NY + py - distance;
        int idx11 = offset + (px    ) * NY + py    ;
        int idx12 = offset + (px    ) * NY + py + distance;
        int idx20 = offset + (px + distance) * NY + py - distance;
        int idx21 = offset + (px + distance) * NY + py    ;
        int idx22 = offset + (px + distance) * NY + py + distance;

        // second-order derivatives
        real_type dxx = - ( corrBatchRaw[idx21] + corrBatchRaw[idx01] - 2.0*corrBatchRaw[idx11] );
        real_type dyy = - ( corrBatchRaw[idx12] + corrBatchRaw[idx10] - 2.0*corrBatchRaw[idx11] ) ;
        real_type dxy = ( corrBatchRaw[idx22] + corrBatchRaw[idx00] - corrBatchRaw[idx20] - corrBatchRaw[idx02] ) *0.25;

        real_type n2 = max(1.0 - peak, 0.0);

        dxx = dxx * templateSize;
        dyy = dyy * templateSize;
        dxy = dxy * templateSize;

        real_type n4 = n2*n2;
        n2 = n2 * 2;
        n4 = n4 * 0.5 * templateSize;

        real_type u = dxy * dxy - dxx * dyy;
        real_type u2 = u*u;

        // if the Gaussian curvature is too small
        if (fabsf(u) < 1e-2) {
            covValue[idxImage] = make_real3(99.0, 99.0, 0.0);
        }
        else {
                real_type cov_xx = (- n2 * u * dyy + n4 * ( dyy*dyy + dxy*dxy) ) / u2;
                real_type cov_yy = (- n2 * u * dxx + n4 * ( dxx*dxx + dxy*dxy) ) / u2;
                real_type cov_xy = (  n2 * u * dxy - n4 * ( dxx + dyy ) * dxy ) / u2;
                covValue[idxImage] = make_real3(cov_xx, cov_yy, cov_xy);
        }
    }
}

/**
 * Estimate the variance of the correlation surface
 * @param[in] templateSize size of reference chip
 * @param[in] distance distance between a pixel
 * @param[in] corrBatchRaw correlation surface
 * @param[in] maxloc maximum location
 * @param[in] maxval maximum value
 * @param[out] covValue variance value
 * @param[in] stream cuda stream
 */
void cuEstimateVariance(cuArrays<real_type> *corrBatchRaw, cuArrays<int2> *maxloc, cuArrays<real_type> *maxval, const int templateSize, const int distance, cuArrays<real3_type> *covValue, cudaStream_t stream)
{
    int size = corrBatchRaw->count;
    // One dimensional launching parameters to loop over every correlation surface.
    cudaKernel_estimateVar<<< IDIVUP(size, NTHREADS), NTHREADS, 0, stream>>>
        (corrBatchRaw->devData, corrBatchRaw->height, corrBatchRaw->width, maxloc->devData, maxval->devData, templateSize, distance, covValue->devData, size);
    getLastCudaError("cudaKernel_estimateVar error\n");
}
//end of file
