/*
 * @file  cuArraysPadding.cu
 * @brief Utilities for padding zeros to cuArrays for FFT (zero padding in the middle)
 */

#include "cuAmpcorUtil.h"
#include "float2.h"


// cuda kernel for zero padding for FFT oversampling,
// for both even and odd sequences
// @param[in] image1 input images
// @param[inout] image2 output images - memset to 0 in prior
// @note siding Nyquist frequency for even length with negative frequency
// for even N - positive f[0, ..., N/2-1],
//              zeros 0...0,
//              negative f[N/2], f[N/2+1, ..., N-1]
// for odd N - positive f[0, ..., (N-1)/2],
//             zeros 0...0,
//             negative f [(N+1)/2, ..., N-1]
__global__ void cuArraysPaddingMany_kernel(
    const complex_type *image1, const int height1, const int width1, const int size1,
    complex_type *image2, const int height2, const int width2, const int size2, const real_type factor )
{
    // thread indices are for input image1
    int x1 = threadIdx.x + blockDim.x*blockIdx.x;
    int y1 = threadIdx.y + blockDim.y*blockIdx.y;
    int imageIdx =  blockIdx.z;

    //  ensure threads are in the range of image1
    if (x1 >= height1 || y1 >= width1)
        return;

    // determine the quadrants
    // divup the length to be consistent with both even and odd lengths
    int x2 = (x1 < (height1+1)/2) ? x1 : height2 - height1 + x1;
    int y2 = (y1 < (width1+1)/2) ? y1 : width2 - width1 + y1;
    image2[IDX2R(x2, y2, width2)+imageIdx*size2]
            = image1[IDX2R(x1, y1, width1)+imageIdx*size1]*factor;
    return;
}

// cuda kernel for zero padding for FFT oversampling,
// for both even and odd sequences
// @param[in] image1 input images
// @param[inout] image2 output images - memset to 0 in prior
// @note for even length (height, width or both),
//       the Nyquist frequency (N/2) component is split between positive and negative frequencies.
__global__ void cuArraysPaddingMany_split_Nyquist_kernel(
    const complex_type *image1, const int height1, const int width1, const int size1,
    complex_type *image2, const int height2, const int width2, const int size2, const real_type factor )
{
    // thread indices are for input image1
    int x1 = threadIdx.x + blockDim.x*blockIdx.x;
    int y1 = threadIdx.y + blockDim.y*blockIdx.y;
    int imageIdx =  blockIdx.z;

    //  ensure threads are in the range of image1
    if (x1 >= height1 || y1 >= width1)
        return;

    int xcenter1 = (height1+1)/2; // divup
    int ycenter1 = (width1+1)/2;

    // for even N - positive f[0, ..., N/2-1], f[N/2]/2,
    //              zeros 0...0,
    //              negative f[N/2]/2, f[N/2+1, ..., N-1]
    //        f[N/2] is split to N/2 and M-N/2
    // for odd N - positive f[0, ..., (N-1)/2],
    //             zeros 0...0,
    //             negative f [(N+1)/2, ..., N-1]

    // split spectrum for even height1 at the center
    if (height1 % 2 ==0 && x1 == xcenter1)
    {
        // if also for the width
        if (width1 % 2 ==0 && y1 == ycenter1)
        {
            // split into 4
            complex_type input = image1[IDX2R(x1, y1, width1)+imageIdx*size1];
            input *= 0.25f*factor;
            image2[IDX2R(x1, y1, width2)+imageIdx*size2] = input;
            int x2 = x1 + height2 - height1;
            int y2 = y1 + width2 - width1;
            image2[IDX2R(x2, y1, width2)+imageIdx*size2] = input;
            image2[IDX2R(x1, y2, width2)+imageIdx*size2] = input;
            image2[IDX2R(x2, y2, width2)+imageIdx*size2] = input;
            // all done for this pixel, so return
            return;
        }
        // odd width or not at the center
        complex_type input = image1[IDX2R(x1, y1, width1)+imageIdx*size1];
        input *= 0.5f*factor;
        int x2 = x1 + height2 - height1;
        image2[IDX2R(x1, y1, width2)+imageIdx*size2] = input;
        image2[IDX2R(x2, y1, width2)+imageIdx*size2] = input;
        return;
    }
    // split spectrum for even width1
    if (width1 % 2 ==0 && y1 == ycenter1)
    {
        // even height even and at the center has been considered earlier
        // so just odd height or not at the center
        complex_type input = image1[IDX2R(x1, y1, width1)+imageIdx*size1];
        input *= 0.5f*factor;
        int y2 = y1 + width2 - width1;
        image2[IDX2R(x1, y1, width2)+imageIdx*size2] = input;
        image2[IDX2R(x1, y2, width2)+imageIdx*size2] = input;
        return;
    }

    // all other cases, simply copy
    // determine the quadrants
    int x2 = (x1 < xcenter1) ? x1 : height2 - height1 + x1;
    int y2 = (y1 < ycenter1) ? y1 : width2 - width1 + y1;
    image2[IDX2R(x2, y2, width2)+imageIdx*size2]
            = image1[IDX2R(x1, y1, width1)+imageIdx*size1]*factor;
    return;
}

/**
 * Padding zeros for FFT oversampling
 * @param[in] image1 input images
 * @param[out] image2 output images
 * @note To keep the band center at (0,0), move quads to corners and pad zeros in the middle
 */
void cuArraysFFTPaddingMany(cuArrays<complex_type> *image1, cuArrays<complex_type> *image2, cudaStream_t stream)
{
    int ThreadsPerBlock = NTHREADS2D;
    // up to IDIVUP(dim, 2) for odd-length sequences
    int BlockPerGridx = IDIVUP (image1->height, ThreadsPerBlock);
    int BlockPerGridy = IDIVUP (image1->width, ThreadsPerBlock);
    dim3 dimBlock(ThreadsPerBlock, ThreadsPerBlock, 1);
    dim3 dimGrid(BlockPerGridx, BlockPerGridy, image1->count);

    checkCudaErrors(cudaMemsetAsync(image2->devData, 0, image2->getByteSize(),stream));
    real_type factor = 1.0f/image1->size;
    cuArraysPaddingMany_kernel<<<dimGrid, dimBlock, 0, stream>>>(
        image1->devData, image1->height, image1->width, image1->size,
        image2->devData, image2->height, image2->width, image2->size, factor);
    getLastCudaError("cuArraysPadding_kernel");
}
//end of file








