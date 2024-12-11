/*
 * @file  cuArraysPadding.cu
 * @brief Utilities for padding zeros to cuArrays for FFT (zero padding in the middle)
 */

#include "cuAmpcorUtil.h"
#include "float2.h"


// cuda kernel for zero padding for FFT oversampling,
// for both even and odd sequences
__global__ void cuArraysPaddingMany_kernel(
    const float2 *image1, const int height1, const int width1, const int size1,
    float2 *image2, const int height2, const int width2, const int size2, const float factor )
{

    int tx = threadIdx.x + blockDim.x*blockIdx.x;
    int ty = threadIdx.y + blockDim.y*blockIdx.y;
    int imageIdx =  blockIdx.z;

    //  make sure threads are in range
    if(tx < height1 && ty < width1)
    {
        // deciding which quadrature to copy to
        // for even N - positive f [0, ..., N/2-1] negative f [N/2, ..., N-1]
        // for odd N - positive f [0, ..., (N-1)/2] negative f [(N+1)/2, ..., N-1]
        int tx2 = (tx < IDIVUP(height1, 2)) ? tx : height2 - height1 + tx;
        int ty2 = (ty < IDIVUP(width1, 2)) ? ty : width2 - width1 + ty;

        // copy
        image2[IDX2R(tx2, ty2, width2)+imageIdx*size2]
            = image1[IDX2R(tx, ty, width1)+imageIdx*size1]*factor;
    }
}

/**
 * Padding zeros for FFT oversampling
 * @param[in] image1 input images
 * @param[out] image2 output images
 * @note To keep the band center at (0,0), move quads to corners and pad zeros in the middle
 */
void cuArraysFFTPaddingMany(cuArrays<float2> *image1, cuArrays<float2> *image2, cudaStream_t stream)
{
    int ThreadsPerBlock = NTHREADS2D;
    // up to IDIVUP(dim, 2) for odd-length sequences
    int BlockPerGridx = IDIVUP (image1->height, ThreadsPerBlock);
    int BlockPerGridy = IDIVUP (image1->width, ThreadsPerBlock);
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








