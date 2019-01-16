/*
 * cuCorrFrequency.cu
 * define a class to save FFT plans and intermediate data for cross correlation in frequency domain
 */

#include "cuCorrFrequency.h"
#include "cuAmpcorUtil.h"

cuFreqCorrelator::cuFreqCorrelator(int imageNX, int imageNY, int nImages, cudaStream_t stream_)
{
    int imageSize = imageNX*imageNY; 
    int fImageSize = imageNX*(imageNY/2+1);
    int n[NRANK] ={imageNX, imageNY};
    
    cufft_Error(cufftPlanMany(&forwardPlan, NRANK, n, 
                              NULL, 1, imageSize,
                              NULL, 1, fImageSize, 
                              CUFFT_R2C, nImages));
    cufft_Error(cufftPlanMany(&backwardPlan, NRANK, n, 
                              NULL, 1, fImageSize,
                              NULL, 1, imageSize, 
                              CUFFT_C2R, nImages));
    stream = stream_;
    cufftSetStream(forwardPlan, stream);
    cufftSetStream(backwardPlan, stream);
    
    workFM = new cuArrays<float2>(imageNX, (imageNY/2+1), nImages);
    workFM->allocate();
    workFS = new cuArrays<float2>(imageNX, (imageNY/2+1), nImages);
    workFS->allocate();
    workT = new cuArrays<float> (imageNX, imageNY, nImages);
    workT->allocate();
}

cuFreqCorrelator::~cuFreqCorrelator()
{
    cufft_Error(cufftDestroy(forwardPlan));
    cufft_Error(cufftDestroy(backwardPlan));	
    workFM->deallocate();
    workFS->deallocate();
    workT->deallocate();
}	

void cuFreqCorrelator::execute(cuArrays<float> *templates, cuArrays<float> *images, cuArrays<float> *results)
{
    cuArraysCopyPadded(templates, workT, stream);
    cufft_Error(cufftExecR2C(forwardPlan, workT->devData, workFM->devData));
    cufft_Error(cufftExecR2C(forwardPlan, images->devData, workFS->devData));
    float coef = 1.0/(images->size);
    cuArraysElementMultiplyConjugate(workFM, workFS, coef, stream);
    cufft_Error(cufftExecC2R(backwardPlan, workFM->devData, workT->devData));
    cuArraysCopyExtract(workT, results, make_int2(0, 0), stream); 
    //workT->outputToFile("test",stream);
} 



__global__ void cudaKernel_elementMulC(float2 *ainout, float2 *bin, size_t size)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < size) {
        cuComplex prod; 
        prod = cuCmulf(ainout[idx], bin[idx]);
        ainout [idx] = prod;
    }
} 

void cuArraysElementMultiply(cuArrays<float2> *image1, cuArrays<float2> *image2, cudaStream_t stream)
{
    int size = image1->getSize();
    int threadsperblock = NTHREADS;
    int blockspergrid = IDIVUP (size, threadsperblock);
    cudaKernel_elementMulC<<<blockspergrid, threadsperblock, 0, stream>>>(image1->devData, image2->devData, size );
    getLastCudaError("cuArraysElementMultiply error\n");
} 

inline __device__ float2 cuMulConj(float2 a, float2 b)
{
    return make_float2(a.x*b.x + a.y*b.y, -a.y*b.x + a.x*b.y);
}


__global__ void cudaKernel_elementMulConjugate(float2 *ainout, float2 *bin, size_t size, float coef)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < size) {
        cuComplex prod; 
        prod = cuMulConj(ainout[idx], bin[idx]);
        ainout [idx] = prod*coef;
    }
} 

void cuArraysElementMultiplyConjugate(cuArrays<float2> *image1, cuArrays<float2> *image2, float coef, cudaStream_t stream)
{
    int size = image1->getSize();
    int threadsperblock = NTHREADS;
    int blockspergrid = IDIVUP (size, threadsperblock);
    cudaKernel_elementMulConjugate<<<blockspergrid, threadsperblock, 0, stream>>>(image1->devData, image2->devData, size, coef );
    getLastCudaError("cuArraysElementMultiply error\n");
} 
