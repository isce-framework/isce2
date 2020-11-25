/*
 * @file  cuArraysPadding.cu
 * @brief Utilities for padding zeros to cuArrays
 */

#include "cuAmpcorUtil.h"
#include "float2.h"

// cuda kernel for cuArraysPadding
__global__ void cuArraysPadding_kernel(
    const float2 *image1, const int height1, const int width1,
    float2 *image2, const int height2, const int width2)
{
    int tx = threadIdx.x + blockDim.x*blockIdx.x;
    int ty = threadIdx.y + blockDim.y*blockIdx.y;
    if(tx < height1/2 && ty < width1/2)
    {
        int tx1 = height1 - 1 - tx;
        int ty1 = width1 -1 -ty;
        int tx2 = height2 -1 -tx;
        int ty2 = width2 -1 -ty;

        image2[IDX2R(tx, ty, width2)] = image1[IDX2R(tx, ty, width1)];
        image2[IDX2R(tx2, ty, width2)] = image1[IDX2R(tx1, ty, width1)];
        image2[IDX2R(tx, ty2, width2)] = image1[IDX2R(tx, ty1, width1)];
        image2[IDX2R(tx2, ty2, width2)] = image1[IDX2R(tx1, ty1, width1)];

    }
}

/**
 * Padding zeros in the middle, move quads to corners
 * @param[in] image1 input images
 * @param[out] image2 output images
 * @note This routine is for a single image, no longer used
 */
void cuArraysPadding(cuArrays<float2> *image1, cuArrays<float2> *image2, cudaStream_t stream)
{
    int ThreadsPerBlock = NTHREADS2D;
    int BlockPerGridx = IDIVUP (image1->height/2, ThreadsPerBlock);
    int BlockPerGridy = IDIVUP (image1->width/2, ThreadsPerBlock);
    dim3 dimBlock(ThreadsPerBlock, ThreadsPerBlock);
    dim3 dimGrid(BlockPerGridx, BlockPerGridy);
    // set output image to 0
    checkCudaErrors(cudaMemsetAsync(image2->devData, 0, image2->getByteSize(),stream));
    // copy the quads of input images to four corners of the output images
    cuArraysPadding_kernel<<<dimGrid, dimBlock, 0, stream>>>(
        image1->devData, image1->height, image1->width,
        image2->devData, image2->height, image2->width);
    getLastCudaError("cuArraysPadding_kernel");
}

inline __device__ float2 cmplxMul(float2 c, float a)
{
    return make_float2(c.x*a, c.y*a);
}

// cuda kernel for
__global__ void cuArraysPaddingMany_kernel(
    const float2 *image1, const int height1, const int width1, const int size1,
    float2 *image2, const int height2, const int width2, const int size2, const float factor )
{
    int tx = threadIdx.x + blockDim.x*blockIdx.x;
    int ty = threadIdx.y + blockDim.y*blockIdx.y;
    if(tx < height1/2 && ty < width1/2)
    {

        int tx1 = height1 - 1 - tx;
        int ty1 = width1 -1 -ty;
        int tx2 = height2 -1 -tx;
        int ty2 = width2 -1 -ty;

        int stride1 = blockIdx.z*size1;
        int stride2 = blockIdx.z*size2;

        image2[IDX2R(tx,  ty,  width2)+stride2] = image1[IDX2R(tx,  ty,  width1)+stride1]*factor;
        image2[IDX2R(tx2, ty,  width2)+stride2] = cmplxMul(image1[IDX2R(tx1, ty,  width1)+stride1], factor);
        image2[IDX2R(tx,  ty2, width2)+stride2] = cmplxMul(image1[IDX2R(tx,  ty1, width1)+stride1], factor);
        image2[IDX2R(tx2, ty2, width2)+stride2] = cmplxMul(image1[IDX2R(tx1, ty1, width1)+stride1], factor);
    }
}

/**
 * Padding zeros for FFT oversampling
 * @param[in] image1 input images
 * @param[out] image2 output images
 * @note To keep the band center at (0,0), move quads to corners and pad zeros in the middle
 */
void cuArraysPaddingMany(cuArrays<float2> *image1, cuArrays<float2> *image2, cudaStream_t stream)
{
    int ThreadsPerBlock = NTHREADS2D;
    int BlockPerGridx = IDIVUP (image1->height/2, ThreadsPerBlock);
    int BlockPerGridy = IDIVUP (image1->width/2, ThreadsPerBlock);
    dim3 dimBlock(ThreadsPerBlock, ThreadsPerBlock, 1);
    dim3 dimGrid(BlockPerGridx, BlockPerGridy, image1->count);

    checkCudaErrors(cudaMemsetAsync(image2->devData, 0, image2->getByteSize(),stream));
    float factor = 1.0f/image1->size;
    cuArraysPaddingMany_kernel<<<dimGrid, dimBlock, 0, stream>>>(
        image1->devData, image1->height, image1->width, image1->size,
        image2->devData, image2->height, image2->width, image2->size, factor);
    getLastCudaError("cuArraysPadding_kernel");
}
//end of file








