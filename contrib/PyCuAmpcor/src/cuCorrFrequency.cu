/*
 * @file  cuCorrFrequency.cu
 * @brief A class performs cross correlation in frequency domain
 */

#include "cuCorrFrequency.h"
#include "cuAmpcorUtil.h"

/*
 * cuFreqCorrelator Constructor
 * @param imageNX height of each image
 * @param imageNY width of each image
 * @param nImages number of images in the batch
 * @param stream CUDA stream
 */
cuFreqCorrelator::cuFreqCorrelator(int imageNX, int imageNY, int nImages, cudaStream_t stream_)
{

    int imageSize = imageNX*imageNY;
    int fImageSize = imageNX*(imageNY/2+1);
    int n[NRANK] ={imageNX, imageNY};
    
    // set up fft plans
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

    // set up work arrays
    workFM = new cuArrays<complex_type>(imageNX, (imageNY/2+1), nImages);
    workFM->allocate();
    workFS = new cuArrays<complex_type>(imageNX, (imageNY/2+1), nImages);
    workFS->allocate();
    workT = new cuArrays<real_type> (imageNX, imageNY, nImages);
    workT->allocate();
}

/// destructor
cuFreqCorrelator::~cuFreqCorrelator()
{
    cufft_Error(cufftDestroy(forwardPlan));
    cufft_Error(cufftDestroy(backwardPlan));	
    workFM->deallocate();
    workFS->deallocate();
    workT->deallocate();
}	


/**
 * Execute the cross correlation
 * @param[in] templates the reference windows
 * @param[in] images the search windows
 * @param[out] results the correlation surfaces
 */

void cuFreqCorrelator::execute(cuArrays<real_type> *templates, cuArrays<real_type> *images, cuArrays<real_type> *results)
{
    // pad the reference windows to the the size of search windows
    cuArraysCopyPadded(templates, workT, stream);
    // forward fft to frequency domain
#ifdef CUAMPCOR_DOUBLE
    cufft_Error(cufftExecD2Z(forwardPlan, workT->devData, workFM->devData));
    cufft_Error(cufftExecD2Z(forwardPlan, images->devData, workFS->devData));
#else
    cufft_Error(cufftExecR2C(forwardPlan, workT->devData, workFM->devData));
    cufft_Error(cufftExecR2C(forwardPlan, images->devData, workFS->devData));
#endif

    // cufft doesn't normalize, so manually get the image size for normalization
    real_type coef = 1.0/(images->size);
    // multiply reference with secondary windows in frequency domain
    cuArraysElementMultiplyConjugate(workFM, workFS, coef, stream);
    // backward fft to get correlation surface in time domain
#ifdef CUAMPCOR_DOUBLE
    cufft_Error(cufftExecZ2D(backwardPlan, workFM->devData, workT->devData));
#else
    cufft_Error(cufftExecC2R(backwardPlan, workFM->devData, workT->devData));
#endif
    // extract to get proper size of correlation surface
    cuArraysCopyExtract(workT, results, make_int2(0, 0), stream);
    // all done
}

// a = a^* * b
inline __device__ complex_type cuMulConj(complex_type a, complex_type b, real_type f)
{
    return make_complex_type(f*(a.x*b.x + a.y*b.y), f*(-a.y*b.x + a.x*b.y));
}

// cuda kernel for cuArraysElementMultiplyConjugate
__global__ void cudaKernel_elementMulConjugate(complex_type *ainout, complex_type *bin, int size, real_type coef)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < size) {
        ainout [idx] = cuMulConj(ainout[idx], bin[idx], coef);
    }
} 

/**
 * Perform multiplication of coef*Conjugate[image1]*image2 for each element
 * @param[inout] image1, the first image
 * @param[in] image2, the secondary image
 * @param[in] coef, usually the normalization factor
 */
void cuArraysElementMultiplyConjugate(cuArrays<complex_type> *image1, cuArrays<complex_type> *image2, real_type coef, cudaStream_t stream)
{
    int size = image1->getSize();
    int threadsperblock = NTHREADS;
    int blockspergrid = IDIVUP (size, threadsperblock);
    cudaKernel_elementMulConjugate<<<blockspergrid, threadsperblock, 0, stream>>>(image1->devData, image2->devData, size, coef );
    getLastCudaError("cuArraysElementMultiply error\n");
} 
//end of file
