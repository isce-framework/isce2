/*
 * @file  cuCorrTimetime.cu
 * @brief Correlation between two sets of images in time domain
 *
 * This code is adapted from the nxcor package.
 */

#include "cuAmpcorUtil.h"


// cuda kernel for cuCorrTimeDomain
template<const int nthreads, const int NPT>
__global__ void cuArraysCorrTime_kernel(
    const int nImages,
    const real_type *templateIn, const int templateNX, const int templateNY, const int templateSize,
    const real_type *imageIn, const int imageNX, const int imageNY, const int imageSize,
    real_type *resultOut, const int resultNX, const int resultNY, const int resultSize)
{
    __shared__ real_type shmem[nthreads*(1+NPT)];
    const int tid = threadIdx.x;
    const int bid =  blockIdx.x;
    const int  yc =  blockIdx.y*NPT;

    const int       imageIdx = bid;
    const int    imageOffset = imageIdx *    imageSize;
    const int templateOffset = imageIdx * templateSize;
    const int   resultOffset = imageIdx *   resultSize;

    const real_type *   imageD =    imageIn  +    imageOffset + tid;
    const real_type *templateD = templateIn  + templateOffset + tid;
     real_type *  resultD =   resultOut +   resultOffset;

    const int q  = min(nthreads/resultNY, 4);
    const int nt = nthreads/q;
    const int ty = threadIdx.x / nt;
    const int tx = threadIdx.x - nt * ty;

    const int templateNYq = templateNY/q;
    const int jbeg = templateNYq * ty;
    const int jend = ty+1 >= q ? templateNY : templateNYq + jbeg;

    real_type *shTemplate = shmem;
    real_type *shImage    = shmem + nthreads;
    real_type *shImage1   = shImage + tx;

    real_type corrCoeff[NPT];
    for (int k = 0; k < NPT; k++)
        corrCoeff[k] = 0.0;

    int iaddr = yc*imageNY;


    real_type img[NPT];
    for (int k = 0; k < NPT-1; k++, iaddr += imageNY)
        img[k] = imageD[iaddr];
    for (int taddr = 0; taddr < templateSize; taddr += templateNY, iaddr += imageNY)
    {
        shTemplate[tid] = templateD[taddr];
        img     [NPT-1] =    imageD[iaddr];
        for (int k = 0; k < NPT; k++)
            shImage[tid + nthreads*k] = img[k];
        for (int k = 0; k < NPT-1; k++)
            img[k] = img[k+1];
        __syncthreads();

        if (tx < resultNY && ty < q)
        {
#pragma unroll 8
            for (int j = jbeg; j < jend; j++)
                for (int k = 0; k < NPT; k++)
                    corrCoeff[k] += shTemplate[j]*shImage1[j + nthreads*k];
        }
        __syncthreads();
    }

    for (int k = 0; k < NPT; k++)
        shmem[tid + nthreads*k] = corrCoeff[k];
    __syncthreads();

    for (int j = tx + nt; j < nthreads; j += nt)
        for (int k = 0; k < NPT; k++)
            corrCoeff[k] += shmem[j + nthreads*k];
    __syncthreads();

    if (tid < resultNY)
    {
        int raddr = yc*resultNY + tid;
        for (int k = 0; k < NPT; k++, raddr += resultNY)
            if (raddr < resultSize)
                resultD[raddr] = corrCoeff[k];
    }
}

/**
 * Perform cross correlation in time domain
 * @param[in] templates Reference images
 * @param[in] images Secondary images
 * @param[out] results Output correlation surface
 * @param[in] stream cudaStream
 */
void cuCorrTimeDomain(cuArrays<real_type> *templates,
               cuArrays<real_type> *images,
               cuArrays<real_type> *results,
               cudaStream_t stream)
{
    /* compute correlation matrix */
    const int nImages = images->count;
    const int imageNY = images->width;

#if defined(CUAMPCOR_DOUBLE) && __CUDA_ARCH__ < 800
    // For GPUs older than A100, static shared memory limit is 48K
    const int NPT = 4;
#else
    const int NPT = 8;
#endif


    const dim3 grid(nImages, (results->width-1)/NPT+1, 1);
    if      (imageNY <=   64) {
        cuArraysCorrTime_kernel<  64,NPT><<<grid,  64, 0, stream>>>(nImages,
            templates->devData, templates->height, templates->width, templates->size,
            images->devData, images->height, images->width, images->size,
            results->devData, results->height, results->width, results->size);
        getLastCudaError("cuArraysCorrTime error");
    }
    else if (imageNY <=  128)  {
        cuArraysCorrTime_kernel< 128,NPT><<<grid, 128, 0, stream>>>(nImages,
            templates->devData, templates->height, templates->width, templates->size,
            images->devData, images->height, images->width, images->size,
            results->devData, results->height, results->width, results->size);
        getLastCudaError("cuArraysCorrTime error");
    }
    else if (imageNY <=  192) {
        cuArraysCorrTime_kernel< 192,NPT><<<grid, 192, 0, stream>>>(nImages,
            templates->devData, templates->height, templates->width, templates->size,
            images->devData, images->height, images->width, images->size,
            results->devData, results->height, results->width, results->size);
        getLastCudaError("cuArraysCorrTime error");
    }
    else if (imageNY <=  256) {
        cuArraysCorrTime_kernel< 256,NPT><<<grid, 256, 0, stream>>>(nImages,
            templates->devData, templates->height, templates->width, templates->size,
            images->devData, images->height, images->width, images->size,
            results->devData, results->height, results->width, results->size);
        getLastCudaError("cuArraysCorrTime error");
    }
    else if (imageNY <=  384) {
        cuArraysCorrTime_kernel< 384,NPT><<<grid, 384, 0, stream>>>(nImages,
            templates->devData, templates->height, templates->width, templates->size,
            images->devData, images->height, images->width, images->size,
            results->devData, results->height, results->width, results->size);
            getLastCudaError("cuArraysCorrTime error");
    }
    else if (imageNY <=  512) {
        cuArraysCorrTime_kernel< 512,NPT><<<grid, 512, 0, stream>>>(nImages,
            templates->devData, templates->height, templates->width, templates->size,
            images->devData, images->height, images->width, images->size,
            results->devData, results->height, results->width, results->size);
        getLastCudaError("cuArraysCorrTime error");
    }
    else if (imageNY <=  640) {
        cuArraysCorrTime_kernel< 640,NPT><<<grid, 640, 0, stream>>>(nImages,
            templates->devData, templates->height, templates->width, templates->size,
            images->devData, images->height, images->width, images->size,
            results->devData, results->height, results->width, results->size);
        getLastCudaError("cuArraysCorrTime error");
    }
    else if (imageNY <=  768) {
        cuArraysCorrTime_kernel< 768,NPT><<<grid, 768, 0, stream>>>(nImages,
            templates->devData, templates->height, templates->width, templates->size,
            images->devData, images->height, images->width, images->size,
            results->devData, results->height, results->width, results->size);
        getLastCudaError("cuArraysCorrTime error");
    }
    else if (imageNY <=  896) {
        cuArraysCorrTime_kernel< 896,NPT><<<grid, 896, 0, stream>>>(nImages,
            templates->devData, templates->height, templates->width, templates->size,
            images->devData, images->height, images->width, images->size,
            results->devData, results->height, results->width, results->size);
        getLastCudaError("cuArraysCorrTime error");
    }
    else if (imageNY <= 1024) {
        cuArraysCorrTime_kernel<1024,NPT><<<grid,1024, 0, stream>>>(nImages,
            templates->devData, templates->height, templates->width, templates->size,
            images->devData, images->height, images->width, images->size,
            results->devData, results->height, results->width, results->size);
        getLastCudaError("cuArraysCorrTime error");
    }
    else {
        fprintf(stderr, "The (oversampled) window size along the across direction %d should be smaller than 1024.\n", imageNY);
        throw;
    }
}
// end of file
