/* 
 * cuOverSampler.cu
 * define cuOverSampler class, to save cufft plans and perform oversampling calculations
 */
#include "cuArrays.h"
#include "cuOverSampler.h"
#include "cuArrays.h"
#include "cudaUtil.h"
#include "cudaError.h"
#include "cuAmpcorUtil.h"

// Oversampler for complex data 
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
    
    workIn = new cuArrays<float2>(inNXp2, inNYp2, nImages);
    workIn->allocate();
    workOut = new cuArrays<float2>(outNXp2, outNYp2, nImages);
    workOut->allocate();
    int imageSize = inNXp2*inNYp2;
    int n[NRANK] ={inNXp2, inNYp2};
    int fImageSize = inNXp2*inNYp2;
    int nOverSample[NRANK] = {outNXp2, outNYp2};
    int fImageOverSampleSize = outNXp2*outNYp2;
    cufft_Error(cufftPlanMany(&forwardPlan, NRANK, n, NULL, 1, imageSize, NULL, 1, fImageSize, CUFFT_C2C, nImages));
    cufft_Error(cufftPlanMany(&backwardPlan, NRANK, nOverSample, NULL, 1, fImageOverSampleSize, NULL, 1, fImageOverSampleSize, CUFFT_C2C, nImages));
    setStream(stream_);
}

void cuOverSamplerC2C::setStream(cudaStream_t stream_)
{
    this->stream = stream_;
    cufftSetStream(forwardPlan, stream);
    cufftSetStream(backwardPlan, stream);
}

//tested
void cuOverSamplerC2C::execute(cuArrays<float2> *imagesIn, cuArrays<float2> *imagesOut)
{
    //cuArraysCopyPadded(imagesIn, workIn, stream);  
    cufft_Error(cufftExecC2C(forwardPlan, imagesIn->devData, workIn->devData, CUFFT_INVERSE));
    cuArraysPaddingMany(workIn, workOut, stream);
    cufft_Error(cufftExecC2C(backwardPlan, workOut->devData, imagesOut->devData, CUFFT_FORWARD));
    //cuArraysCopyExtract(workOut, imagesOut, make_int2(0,0), stream);
}

void cuOverSamplerC2C::execute(cuArrays<float2> *imagesIn, cuArrays<float2> *imagesOut, int method)
{   
    cuDeramp(method, imagesIn, stream);         
    cufft_Error(cufftExecC2C(forwardPlan, imagesIn->devData, workIn->devData, CUFFT_INVERSE ));
    cuArraysPaddingMany(workIn, workOut, stream);
    cufft_Error(cufftExecC2C(backwardPlan, workOut->devData, imagesOut->devData, CUFFT_FORWARD));
}

cuOverSamplerC2C::~cuOverSamplerC2C() 
{
    cufft_Error(cufftDestroy(forwardPlan));
    cufft_Error(cufftDestroy(backwardPlan));
    delete(workIn);
    delete(workOut);	
}


// oversampler for real data
cuOverSamplerR2R::cuOverSamplerR2R(int inNX, int inNY, int outNX, int outNY, int nImages, cudaStream_t stream)
{
    
/*    
    int inNXp2 = nextpower2(inNX);
    int inNYp2 = nextpower2(inNY);
    int outNXp2 = inNXp2*outNX/inNX;
    int outNYp2 = inNYp2*outNY/inNY;    
*/
    
    int inNXp2 = inNX;
    int inNYp2 = inNY;
    int outNXp2 = outNX;
    int outNYp2 = outNY;

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

//tested
void cuOverSamplerR2R::execute(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut)
{
    cuArraysCopyPadded(imagesIn, workSizeIn, stream);
    cufft_Error(cufftExecC2C(forwardPlan, workSizeIn->devData, workSizeIn->devData, CUFFT_INVERSE));
    cuArraysPaddingMany(workSizeIn, workSizeOut, stream);
    cufft_Error(cufftExecC2C(backwardPlan, workSizeOut->devData, workSizeOut->devData,CUFFT_FORWARD ));
    cuArraysCopyExtract(workSizeOut, imagesOut, make_int2(0,0), stream);	
}

cuOverSamplerR2R::~cuOverSamplerR2R() 
{
    cufft_Error(cufftDestroy(forwardPlan));
    cufft_Error(cufftDestroy(backwardPlan));	
    workSizeIn->deallocate();
    workSizeOut->deallocate();
}






