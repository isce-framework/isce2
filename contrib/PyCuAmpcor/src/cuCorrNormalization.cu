/*
 * @file cuCorrNormalization.cu
 * @brief Utilities to normalize the correlation surface
 *
 * The mean and variance of the normalization factor can be computed from the
 *   cumulative/prefix sum (or sum area table) s(u,v), and s2(u,v).
 * We follow the algorithm by Evghenii Gaburov, Tim Idzenga, Willem Vermin, in the nxcor package.
 * 1. Iterate over rows and for each row, the cumulative sum for elements in the row
 *    is computed as c_row(u,v) = \sum_(v'<v) f(u, v')
 *    and we keep track of the sum of area of width Ny, i.e.,
 *         c(u,v) = \sum_{u'<=u} [c_row(u', v+Ny) - c_row(u', v)],
 *         or c(u,v) = c(u-1, v) + [c_row(u, v+Ny) - c_row(u, v)]
 * 2. When row reaches the window height u=Nx-1,
 *    c(u,v) provides the sum of area for the first batch of windows sa(0,v).
 * 3. proceeding to row = u+1, we compute both c_row(u+1, v) and c_row(u-Nx, v)
 *    i.e., we add the sum from new row and remove the sum from the first row in c(u,v):
 *    c(u+1,v)= c(u,v) + [c_row(u+1,v+Ny)-c_row(u+1, v)] - [c_row(u-Nx, v+Ny)-c_row(u-Nx, v)].
 * 4. Iterate 3. over the rest rows, and c(u,v) provides the sum of areas for new row of windows.
 *
 */

#include "cuAmpcorUtil.h"
#include <stdio.h>

 // sum reduction within a block
 // the following implementation is compatible for sm_20 and above
 // newer architectures may support faster implementations, such as warp shuffle, cooperative groups
template <const int Nthreads>
__device__ real_type sumReduceBlock(real_type sum, volatile real_type *shmem)
{
    const int tid = threadIdx.x;
    shmem[tid] = sum;
    __syncthreads();

    if (Nthreads >=1024) { if (tid < 512) { shmem[tid] += shmem[tid + 512]; } __syncthreads(); }
    if (Nthreads >= 512) { if (tid < 256) { shmem[tid] += shmem[tid + 256]; } __syncthreads(); }
    if (Nthreads >= 256) { if (tid < 128) { shmem[tid] += shmem[tid + 128]; } __syncthreads(); }
    if (Nthreads >= 128) { if (tid <  64) { shmem[tid] += shmem[tid +  64]; } __syncthreads(); }
    if (tid < 32)
    {
        shmem[tid] += shmem[tid + 32];
        shmem[tid] += shmem[tid + 16];
        shmem[tid] += shmem[tid +  8];
        shmem[tid] += shmem[tid +  4];
        shmem[tid] += shmem[tid +  2];
        shmem[tid] += shmem[tid +  1];
    }

    __syncthreads();
    return shmem[0];
}

// cuda kernel to subtract mean value from the images
template<const int Nthreads>
__global__ void cuArraysMean_kernel(real_type *images, real_type *image_sum, int imageSize, real_type invSize, int nImages)
{
    __shared__ real_type shmem[Nthreads];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (bid >= nImages) return;

    const int       imageIdx = bid;
    const int imageOffset = imageIdx * imageSize;
    real_type   *imageD = images + imageOffset;

    real_type sum  = 0.0;
    // perform the reduction beyond one block
    // save the results for each thread in block
    for (int i = tid; i < imageSize; i += Nthreads)
            sum += imageD[i];
    // reduction within the block
    sum = sumReduceBlock<Nthreads>(sum, shmem);

    const real_type mean = sum * invSize;
    if(tid ==0) image_sum[bid] = mean;
}

/**
 * Compute mean values for images
 * @param[in] images Input images
 * @param[out] mean Output mean values
 * @param[in] stream cudaStream
 */
void cuArraysMeanValue(cuArrays<real_type> *images, cuArrays<real_type> *mean, cudaStream_t stream)
{
    const dim3 grid(images->count, 1, 1);
    const int imageSize = images->width*images->height;
    const real_type invSize = 1.0f/imageSize;

    cuArraysMean_kernel<NTHREADS> <<<grid,NTHREADS,0,stream>>>(images->devData, mean->devData, imageSize, invSize, images->count);
    getLastCudaError("cuArraysMeanValue kernel error\n");
}

// cuda kernel to compute and subtracts mean value from the images
template<const int Nthreads>
__global__ void cuArraysSubtractMean_kernel(real_type *images, int imageSize, real_type invSize, int nImages)
{
    __shared__ real_type shmem[Nthreads];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (bid >= nImages) return;

    const int       imageIdx = bid;
    const int imageOffset = imageIdx * imageSize;
    real_type   *imageD = images + imageOffset;

    // compute the sum
    real_type sum  = 0.0;
    for (int i = tid; i < imageSize; i += Nthreads)
            sum += imageD[i];
    sum = sumReduceBlock<Nthreads>(sum, shmem);

    // compute the mean
    const real_type mean = sum * invSize;
    // subtract the mean from each pixel
    for (int i = tid; i < imageSize; i += Nthreads)
            imageD[i] -= mean;
}

/**
 * Compute and subtract mean values from images
 * @param[inout] images Input/Output images
 * @param[out] mean Output mean values
 * @param[in] stream cudaStream
 */
void cuArraysSubtractMean(cuArrays<real_type> *images, cudaStream_t stream)
{
    const dim3 grid(images->count, 1, 1);
    const int imageSize = images->width*images->height;
    const real_type invSize = 1.0f/imageSize;

    cuArraysSubtractMean_kernel<NTHREADS> <<<grid,NTHREADS,0,stream>>>(images->devData, imageSize, invSize, images->count);
    getLastCudaError("cuArraysSubtractMean kernel error\n");
}


// cuda kernel to compute summation on extracted correlation surface (Minyan)
template<const int Nthreads>
__global__ void cuArraysSum_kernel(real_type *images, real_type *imagesSum, int imageSize, int nImages)
{
    __shared__ real_type shmem[Nthreads];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (bid >= nImages) return;

    const int imageIdx = bid;
    const int imageOffset = imageIdx * imageSize;
    real_type*    imageD = images + imageOffset;

    real_type sum  = 0.0;

    for (int i = tid; i < imageSize; i += Nthreads) {
            sum += imageD[i] * imageD[i];
    }

    sum = sumReduceBlock<Nthreads>(sum, shmem);

    if(tid ==0) {
        imagesSum[bid] = sum;
    }
}

/**
 * Compute the variance of images (for SNR)
 * @param[in] images Input images
 * @param[in] imagesValid validity flags for each pixel
 * @param[out] imagesSum variance
 * @param[out] imagesValidCount count of total valid pixels
 * @param[in] stream cudaStream
 */
void cuArraysSumSquare(cuArrays<real_type> *images, cuArrays<real_type> *imagesSum, cudaStream_t stream)
{
    const dim3 grid(images->count, 1, 1);
    const int imageSize = images->width*images->height;

    cuArraysSum_kernel<NTHREADS> <<<grid,NTHREADS,0,stream>>>(images->devData,
        imagesSum->devData, imageSize, images->count);
    getLastCudaError("cuArraysSumValue kernel error\n");
}

// cuda kernel to compute summation on extracted correlation surface (Minyan)
template<const int Nthreads>
__global__ void cuArraysSumCorr_kernel(float *images, int *imagesValid, float *imagesSum, int *imagesValidCount, int imageSize, int nImages)
{
    __shared__ float shmem[Nthreads];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (bid >= nImages) return;

    const int imageIdx = bid;
    const int imageOffset = imageIdx * imageSize;
    float*    imageD = images + imageOffset;
    int*      imageValidD = imagesValid + imageOffset;

    float sum  = 0.0f;
    int count = 0;

    for (int i = tid; i < imageSize; i += Nthreads) {
            sum += imageD[i] * imageD[i];
            count += imageValidD[i];
    }

    sum = sumReduceBlock<Nthreads>(sum, shmem);
    count = sumReduceBlock<Nthreads>(count, shmem);

    if(tid ==0) {
        imagesSum[bid] = sum;
        imagesValidCount[bid] = count;
    }
}

/**
 * Compute the variance of images (for SNR)
 * @param[in] images Input images
 * @param[in] imagesValid validity flags for each pixel
 * @param[out] imagesSum variance
 * @param[out] imagesValidCount count of total valid pixels
 * @param[in] stream cudaStream
 */
void cuArraysSumCorr(cuArrays<float> *images, cuArrays<int> *imagesValid, cuArrays<float> *imagesSum,
    cuArrays<int> *imagesValidCount, cudaStream_t stream)
{
    const dim3 grid(images->count, 1, 1);
    const int imageSize = images->width*images->height;

    cuArraysSumCorr_kernel<NTHREADS> <<<grid,NTHREADS,0,stream>>>(images->devData, imagesValid->devData,
        imagesSum->devData, imagesValidCount->devData, imageSize, images->count);
    getLastCudaError("cuArraysSumValueCorr kernel error\n");
}


// intra-block inclusive prefix sum
template<int Nthreads2>
__device__ void inclusive_prefix_sum(real_type sum, volatile real_type *shmem)
{
    const int tid = threadIdx.x;
    shmem[tid] = sum;
    __syncthreads();

#pragma unroll
    for (int i = 0; i < Nthreads2; i++)
    {
        const int offset = 1 << i;
        if (tid >= offset) sum += shmem[tid - offset];
        __syncthreads();
        shmem[tid] = sum;
        __syncthreads();
    }
}

// prefix sum of pixel value and pixel value^2
template<const int Nthreads2>
__device__ real2_type partialSums(const real_type v, volatile real_type* shmem, const int stride)
{
    const int tid = threadIdx.x;

    volatile real_type *shMem  = shmem + 1;
    volatile real_type *shMem2 = shMem + 1 + (1 << Nthreads2);

    inclusive_prefix_sum<Nthreads2>(v,   shMem);
    inclusive_prefix_sum<Nthreads2>(v*v, shMem2);
    const real_type Sum  = shMem [tid-1 + stride] - shMem [tid-1];
    const real_type Sum2 = shMem2[tid-1 + stride] - shMem2[tid-1];
    return make_real2(Sum, Sum2);
}

// cuda kernel for cuCorrNormalize
template<const int Nthreads2>
__global__ void cuCorrNormalize_kernel(
    int nImages,
    const real_type *templateIn, int templateNX, int templateNY, int templateSize,
    const real_type *imageIn, int imageNX, int imageNY, int imageSize,
    real_type *resultOut, int resultNX, int resultNY, int resultSize,
    real_type templateCoeff)
{
    const int Nthreads = 1<<Nthreads2;
    __shared__ real_type shmem[Nthreads*3];

    const int tid = threadIdx.x;
    const int imageIdx = blockIdx.z;
    if (imageIdx >= nImages) return;

    const int    imageOffset = imageIdx *    imageSize;
    const int templateOffset = imageIdx * templateSize;
    const int   resultOffset = imageIdx *   resultSize;

    const real_type *   imageD =    imageIn  +    imageOffset;
    const real_type *templateD = templateIn  + templateOffset;
    real_type *  resultD =   resultOut +   resultOffset;

    // template sum^2
    real_type templateSum2 = 0.0;
    for (int i = tid; i < templateSize; i += Nthreads)
        {
            const real_type t = templateD[i];
            templateSum2 += t*t;
        }
    templateSum2 = sumReduceBlock<Nthreads>(templateSum2, shmem);
    __syncthreads();

    // reset shared memory value
    shmem[tid] = shmem[tid + Nthreads] = shmem[tid + 2*Nthreads] = 0.0;
    __syncthreads();

    // perform the prefix sum and sum^2 for secondary window
    // see notes above
    real_type imageSum  = 0.0;
    real_type imageSum2 = 0.0;
    int iaddr = 0;
    const int windowSize = templateNX*imageNY;
    // iterative till reaching the templateNX row of the secondary window
    // or the first row of correlation surface may be computed
    while (iaddr < windowSize)
    {
        // cum sum for each row with a width=templateNY
        const real2_type res = partialSums<Nthreads2>(imageD[iaddr + tid], shmem, templateNY);
        // add to the total, which keeps track of the sum of area for each window
        imageSum  += res.x;
        imageSum2 += res.y;
        // move to next row
        iaddr     += imageNY;
    }
    // row reaches the end of first batch of windows
    // normalize the first row of the correlation surface
    if (tid < resultNY)
    {
        // normalizing factor
        const real_type norm2 = (imageSum2 - imageSum*imageSum*templateCoeff)*templateSum2;
        // normalize the correlation surface
        resultD[tid] *= rsqrt(norm2 + EPSILON);
    }
    // iterative over the rest rows
    while (iaddr < imageSize)
    {
        // the prefix sum of the row removed is recomputed, to be subtracted
        const real2_type res1 = partialSums<Nthreads2>(imageD[iaddr-windowSize + tid], shmem, templateNY);
        // the prefix sum of the new row, to be added
        const real2_type res2 = partialSums<Nthreads2>(imageD[iaddr            + tid], shmem, templateNY);
        imageSum  += res2.x - res1.x;
        imageSum2 += res2.y - res1.y;
        // move to next row
        iaddr     += imageNY;
        // normalize the correlation surface
        if (tid < resultNY)
        {
            const int ix = iaddr/imageNY; // get row index
            const int addr = (ix-templateNX)*resultNY; // get the correlation surface row index
            const real_type norm2 = (imageSum2 - imageSum*imageSum*templateCoeff)*templateSum2;
            resultD[addr + tid] *= rsqrt(norm2 + EPSILON);
        }
    }
}

/**
 * Normalize a correlation surface
 * @param[in] templates Reference windows with mean subtracted
 * @param[in] images Secondary windows
 * @param[inout] results un-normalized correlation surface as input and normalized as output
 * @param[in] stream cudaStream
 * @warning The current implementation uses one thread for one column, therefore,
 *   the secondary window width is limited to <=1024, the max threads in a block.
 */
void cuCorrNormalize(cuArrays<real_type> *templates, cuArrays<real_type> *images, cuArrays<real_type> *results, cudaStream_t stream)
{
    const int nImages = images->count;
    const int imageNY = images->width;
    const dim3 grid(1, 1, nImages);
    const real_type invTemplateSize = 1.0f/templates->size;

    if      (imageNY <=   64) {
        cuCorrNormalize_kernel< 6><<<grid,  64, 0, stream>>>(nImages,
            templates->devData, templates->height, templates->width, templates->size,
            images->devData, images->height, images->width, images->size,
            results->devData, results->height, results->width, results->size,
            invTemplateSize);
        getLastCudaError("cuCorrNormalize kernel error");
    }
    else if (imageNY <=  128) {
        cuCorrNormalize_kernel< 7><<<grid, 128, 0, stream>>>(nImages,
            templates->devData, templates->height, templates->width, templates->size,
            images->devData, images->height, images->width, images->size,
            results->devData, results->height, results->width, results->size,
            invTemplateSize);
        getLastCudaError("cuCorrNormalize kernel error");
    }
    else if (imageNY <=  256) {
        cuCorrNormalize_kernel< 8><<<grid, 256, 0, stream>>>(nImages,
            templates->devData, templates->height, templates->width, templates->size,
            images->devData, images->height, images->width, images->size,
            results->devData, results->height, results->width, results->size,
            invTemplateSize);
        getLastCudaError("cuCorrNormalize kernel error");
    }
    else if (imageNY <=  512) {
        cuCorrNormalize_kernel< 9><<<grid, 512, 0, stream>>>(nImages,
            templates->devData, templates->height, templates->width, templates->size,
            images->devData, images->height, images->width, images->size,
            results->devData, results->height, results->width, results->size,
            invTemplateSize);
        getLastCudaError("cuCorrNormalize kernel error");
    }
    else if (imageNY <= 1024) {
        cuCorrNormalize_kernel<10><<<grid,1024, 0, stream>>>(nImages,
            templates->devData, templates->height, templates->width, templates->size,
            images->devData, images->height, images->width, images->size,
            results->devData, results->height, results->width, results->size,
            invTemplateSize);
        getLastCudaError("cuCorrNormalize kernel error");
    }
    else
    {
        fprintf(stderr, "The (oversampled) window size along the across direction %d should be smaller than 1024.\n", imageNY);
        throw;
    }

}

template<int N> struct Log2;
template<> struct Log2<64> { static const int value = 6; };
template<> struct Log2<128> { static const int value = 7; };
template<> struct Log2<256> { static const int value = 8; };
template<> struct Log2<512> { static const int value = 9; };
template<> struct Log2<1024> { static const int value = 10; };

template<int Size>
void cuCorrNormalizeFixed(cuArrays<real_type> *correlation, cuArrays<real_type> *reference, cuArrays<real_type> *secondary, cudaStream_t stream)
{
    const int nImages = correlation->count;
    const dim3 grid(1, 1, nImages);
    const real_type invReferenceSize = 1.0f/reference->size;
    cuCorrNormalize_kernel<Log2<Size>::value><<<grid, Size, 0, stream>>>(nImages,
                reference->devData, reference->height, reference->width, reference->size,
                secondary->devData, secondary->height, secondary->width, secondary->size,
                correlation->devData, correlation->height, correlation->width, correlation->size,
                invReferenceSize);
    getLastCudaError("cuCorrNormalize kernel error");
}

template void cuCorrNormalizeFixed<64>(cuArrays<real_type> *correlation,
        cuArrays<real_type> *reference, cuArrays<real_type> *secondary,
        cudaStream_t stream);
template void cuCorrNormalizeFixed<128>(cuArrays<real_type> *correlation,
        cuArrays<real_type> *reference, cuArrays<real_type> *secondary,
        cudaStream_t stream);
template void cuCorrNormalizeFixed<256>(cuArrays<real_type> *correlation,
        cuArrays<real_type> *reference, cuArrays<real_type> *secondary,
        cudaStream_t stream);
template void cuCorrNormalizeFixed<512>(cuArrays<real_type> *correlation,
        cuArrays<real_type> *reference, cuArrays<real_type> *secondary,
        cudaStream_t stream);
template void cuCorrNormalizeFixed<1024>(cuArrays<real_type> *correlation,
        cuArrays<real_type> *reference, cuArrays<real_type> *secondary,
        cudaStream_t stream);

// end of file
