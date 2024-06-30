/* 
 * @file cuOverSampler.cu
 * @brief Implementations of cuOverSamplerR2R (C2C) class
 */

// my declarations
#include "cuOverSampler.h"

// dependencies
#include "cuArrays.h"
#include "cuArrays.h"
#include "cudaUtil.h"
#include "cudaError.h"
#include "cuAmpcorUtil.h"

/**
 * Constructor for cuOversamplerC2C
 * @param input image size inNX x inNY
 * @param output image size outNX x outNY
 * @param nImages batches
 * @param stream_ cuda stream
 */
cuOverSamplerC2C::cuOverSamplerC2C(int inNX, int inNY, int outNX, int outNY, int nImages, cudaStream_t stream_)
{
    
    int inNXp2 = inNX;
    int inNYp2 = inNY;
    int outNXp2 = outNX;
    int outNYp2 = outNY;
    
    /* if expanded to 2^n
    int inNXp2 = nextpower2(inNX);
    int inNYp2 = nextpower2(inNY);
    int outNXp2 = inNXp2*outNX/inNX;
    int outNYp2 = inNYp2*outNY/inNY; 
    */

    // set up work arrays
    workIn = new cuArrays<float2>(inNXp2, inNYp2, nImages);
    workIn->allocate();
    workOut = new cuArrays<float2>(outNXp2, outNYp2, nImages);
    workOut->allocate();

    // set up fft plans
    int imageSize = inNXp2*inNYp2;
    int n[NRANK] ={inNXp2, inNYp2};
    int fImageSize = inNXp2*inNYp2;
    int nOverSample[NRANK] = {outNXp2, outNYp2};
    int fImageOverSampleSize = outNXp2*outNYp2;
    cufft_Error(cufftPlanMany(&forwardPlan, NRANK, n, NULL, 1, imageSize, NULL, 1, fImageSize, CUFFT_C2C, nImages));
    cufft_Error(cufftPlanMany(&backwardPlan, NRANK, nOverSample, NULL, 1, fImageOverSampleSize, NULL, 1, fImageOverSampleSize, CUFFT_C2C, nImages));
    // set cuda stream
    setStream(stream_);
}

/**
 * Set up cuda stream
 */
void cuOverSamplerC2C::setStream(cudaStream_t stream_)
{
    this->stream = stream_;
    cufftSetStream(forwardPlan, stream);
    cufftSetStream(backwardPlan, stream);
}

/**
 * Execute fft oversampling
 * @param[in] imagesIn input batch of images
 * @param[out] imagesOut output batch of images
 * @param[in] method phase deramping method
 */
void cuOverSamplerC2C::execute(cuArrays<float2> *imagesIn, cuArrays<float2> *imagesOut, int method)
{   
    cuDeramp(method, imagesIn, stream);         
    cufft_Error(cufftExecC2C(forwardPlan, imagesIn->devData, workIn->devData, CUFFT_INVERSE ));
    cuArraysPaddingMany(workIn, workOut, stream);
    cufft_Error(cufftExecC2C(backwardPlan, workOut->devData, imagesOut->devData, CUFFT_FORWARD));
}

/// destructor
cuOverSamplerC2C::~cuOverSamplerC2C() 
{
    // destroy fft handles
    cufft_Error(cufftDestroy(forwardPlan));
    cufft_Error(cufftDestroy(backwardPlan));
    // deallocate work arrays
    delete(workIn);
    delete(workOut);	
}

// end of cuOverSamplerC2C

/**
 * Constructor for cuOversamplerR2R
 * @param input image size inNX x inNY
 * @param output image size outNX x outNY
 * @param nImages the number of images
 * @param stream_ cuda stream
 */
cuOverSamplerR2R::cuOverSamplerR2R(int inNX, int inNY, int outNX, int outNY, int nImages, cudaStream_t stream)
{
    
    int inNXp2 = inNX;
    int inNYp2 = inNY;
    int outNXp2 = outNX;
    int outNYp2 = outNY;

    /* if expanded to 2^n
    int inNXp2 = nextpower2(inNX);
    int inNYp2 = nextpower2(inNY);
    int outNXp2 = inNXp2*outNX/inNX;
    int outNYp2 = inNYp2*outNY/inNY;
    */

    int imageSize = inNXp2 *inNYp2;
    int n[NRANK] ={inNXp2, inNYp2};
    int fImageSize = inNXp2*inNYp2;
    int nUpSample[NRANK] = {outNXp2, outNYp2};
    int fImageUpSampleSize = outNXp2*outNYp2;
    workSizeIn = new cuArrays<float2>(inNXp2, inNYp2, nImages);
    workSizeIn->allocate();
    workSizeOut = new cuArrays<float2>(outNXp2, outNYp2, nImages);
    workSizeOut->allocate();
    cufft_Error(cufftPlanMany(&forwardPlan, NRANK, n, NULL, 1, imageSize, NULL, 1, fImageSize, CUFFT_C2C, nImages));
    cufft_Error(cufftPlanMany(&backwardPlan, NRANK, nUpSample, NULL, 1, fImageUpSampleSize, NULL, 1, outNX*outNY, CUFFT_C2C, nImages));
    setStream(stream);
}

void cuOverSamplerR2R::setStream(cudaStream_t stream_)
{
    stream = stream_;
    cufftSetStream(forwardPlan, stream);
    cufftSetStream(backwardPlan, stream);
}

/**
 * Execute fft oversampling
 * @param[in] imagesIn input batch of images
 * @param[out] imagesOut output batch of images
 */
void cuOverSamplerR2R::execute(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut)
{
    cuArraysCopyPadded(imagesIn, workSizeIn, stream);
    cufft_Error(cufftExecC2C(forwardPlan, workSizeIn->devData, workSizeIn->devData, CUFFT_INVERSE));
    cuArraysPaddingMany(workSizeIn, workSizeOut, stream);
    cufft_Error(cufftExecC2C(backwardPlan, workSizeOut->devData, workSizeOut->devData,CUFFT_FORWARD ));
    cuArraysCopyExtract(workSizeOut, imagesOut, make_int2(0,0), stream);	
}

/// destructor
cuOverSamplerR2R::~cuOverSamplerR2R() 
{
    cufft_Error(cufftDestroy(forwardPlan));
    cufft_Error(cufftDestroy(backwardPlan));	
    workSizeIn->deallocate();
    workSizeOut->deallocate();
}

// end of file





