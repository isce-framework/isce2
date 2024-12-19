/*
 * @file  cuDeramp.cu
 * @brief Derampling a batch of 2D complex images with GPU
 *
 * A phase ramp is equivalent to a frequency shift in frequency domain,
 *   which needs to be removed (deramping) in order to move the band center
 *   to zero. This is necessary before oversampling a complex signal.
 * Method 1: each signal is decomposed into real and imaginary parts,
 *   and the average phase shift is obtained as atan(\sum imag / \sum real).
 *   The average is weighted by the amplitudes (coherence).
 * Method 0 or else: skip deramping
 *
 */
 
#include "cuArrays.h" 
#include "float2.h"
#include "data_types.h"
#include "cudaError.h"
#include "cudaUtil.h"
#include "cuAmpcorUtil.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>


// cuda does not have a good support on volatile vector struct, e.g. real2_type
// have to use regular real_type type for shared memory (volatile) data
// the following methods are defined to operate real2_type/complex objects through real_type
inline static __device__ void copyToShared(volatile real_type *s, const int i, const real2_type x, const int block) 
{ s[i] = x.x; s[i+block] = x.y; }

inline static __device__ void copyFromShared(real2_type &x, volatile real_type *s, const int i, const int block) 
{ x.x = s[i]; x.y = s[i+block]; }


inline static __device__ void addInShared(volatile real_type *s, const int i, const int j, const int block) 
{ s[i] += s[i+j]; s[i+block] += s[i+j+block];}


// kernel to do sum reduction for real2_type within a block
template <const int nthreads>
__device__ void complexSumReduceBlock(real2_type& sum, volatile real_type *shmem)
{
    const int tid = threadIdx.x;
    copyToShared(shmem, tid, sum, nthreads);
    __syncthreads();
    
    if (nthreads >=1024) { if (tid < 512) { addInShared(shmem, tid, 512, nthreads); } __syncthreads(); }
    if (nthreads >= 512) { if (tid < 256) { addInShared(shmem, tid, 256, nthreads); } __syncthreads(); }
    if (nthreads >= 256) { if (tid < 128) { addInShared(shmem, tid, 128, nthreads); } __syncthreads(); }
    if (nthreads >= 128) { if (tid <  64) { addInShared(shmem, tid,  64, nthreads); } __syncthreads(); }
    if (tid < 32)
    {	
        addInShared(shmem, tid, 32, nthreads);
        addInShared(shmem, tid, 16, nthreads);
        addInShared(shmem, tid,  8, nthreads);
        addInShared(shmem, tid,  4, nthreads);
        addInShared(shmem, tid,  2, nthreads);
        addInShared(shmem, tid,  1, nthreads); 
    }
    __syncthreads();
    copyFromShared(sum, shmem, 0, nthreads);
}

// cuda kernel for cuDerampMethod1
template<const int nthreads>
__global__ void cuDerampMethod1_kernel(real2_type *images, const int imageNX, int const imageNY, 
    const int imageSize, const int nImages, const real_type normCoef)
{
    __shared__ real_type shmem[2*nthreads];
    int pixelIdx, pixelIdxX, pixelIdxY;
    
    const int bid = blockIdx.x;    
    if(bid >= nImages) return;
    real2_type *image = images+ bid*imageSize;
    const int tid = threadIdx.x;  
    real2_type phaseDiffY  = make_real2(0.0, 0.0);
    for (int i = tid; i < imageSize; i += nthreads) {
        pixelIdxY = i % imageNY;
        if(pixelIdxY < imageNY -1) {
            pixelIdx = i;
            real2_type cprod = complexMulConj( image[pixelIdx], image[pixelIdx+1]);   
            phaseDiffY += cprod;
        } 
    }       
    complexSumReduceBlock<nthreads>(phaseDiffY, shmem);
    //phaseDiffY *= normCoef;
    real_type phaseY=atan2(phaseDiffY.y, phaseDiffY.x);

    real2_type phaseDiffX  = make_real2(0.0, 0.0);
    for (int i = tid; i < imageSize; i += nthreads)  {
        pixelIdxX = i / imageNY; 
        if(pixelIdxX < imageNX -1) {
            pixelIdx = i;
            real2_type cprod = complexMulConj(image[i], image[i+imageNY]);
            phaseDiffX += cprod;
        }
    }   
    
    complexSumReduceBlock<nthreads>(phaseDiffX, shmem);
   
    //phaseDiffX *= normCoef;
    real_type phaseX = atan2(phaseDiffX.y, phaseDiffX.x);  //+FLT_EPSILON
     
    for (int i = tid; i < imageSize; i += nthreads)
    { 
        pixelIdxX = i%imageNY;
        pixelIdxY = i/imageNY;
        real_type phase = pixelIdxX*phaseX + pixelIdxY*phaseY;
        real2_type phase_factor = make_real2(cosf(phase), sinf(phase));
        image[i] *= phase_factor;
    }     
}

/**
 * Deramp a complex signal with Method 1
 * @brief Each signal is decomposed into real and imaginary parts,
 *   and the average phase shift is obtained as atan(\sum imag / \sum real).
 * @param[inout] images input/output complex signals
 * @param[in] stream cuda stream
 */
void cuDerampMethod1(cuArrays<real2_type> *images, cudaStream_t stream)
{
    
    const dim3 grid(images->count);
    const int imageSize = images->width*images->height;
    const real_type invSize = 1.0f/imageSize;

    if(imageSize <=64) {
        cuDerampMethod1_kernel<64> <<<grid, 64, 0, stream>>>
        (images->devData, images->height, images->width, 
        imageSize, images->count, invSize); }
     else if(imageSize <=128) {
        cuDerampMethod1_kernel<128> <<<grid, 128, 0, stream>>>
        (images->devData, images->height, images->width, 
        imageSize, images->count, invSize); }   
     else if(imageSize <=256) {
        cuDerampMethod1_kernel<256> <<<grid, 256, 0, stream>>>
        (images->devData, images->height, images->width, 
        imageSize, images->count, invSize); }  
    else  {
        cuDerampMethod1_kernel<512> <<<grid, 512, 0, stream>>>
        (images->devData, images->height, images->width, 
        imageSize, images->count, invSize); }
    getLastCudaError("cuDerampMethod1 kernel error\n");

}
        
void cuDeramp(int method, cuArrays<real2_type> *images, cudaStream_t stream)
{
    switch(method) {
    case 1:
        cuDerampMethod1(images, stream);
        break;
    default:
        break;
    }
}

// end of file