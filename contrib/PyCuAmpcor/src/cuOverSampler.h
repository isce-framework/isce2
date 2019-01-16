/* 
 * cuOverSampler.h 
 * oversampling with FFT padding method
 * define cuOverSampler class, to save cufft plans and perform oversampling calculations
 * one float image use cuOverSamplerR2R
 * one complex image use cuOverSamplerC2C
 * many complex images use cuOverSamplerManyC2C
 */

#ifndef __CUOVERSAMPLER_H
#define __CUOVERSAMPLER_H
 
#include "cuArrays.h"
#include "cudaUtil.h"


class cuOverSamplerC2C
{
private:
	 cufftHandle forwardPlan;
	 cufftHandle backwardPlan;
	 cudaStream_t stream;
     cuArrays<float2> *workIn;
     cuArrays<float2> *workOut;
public:
	 cuOverSamplerC2C(int inNX, int inNY, int outNX, int outNY, int nImages, cudaStream_t stream_);
	 void setStream(cudaStream_t stream_);
	 void execute(cuArrays<float2> *imagesIn, cuArrays<float2> *imagesOut);
     void execute(cuArrays<float2> *imagesIn, cuArrays<float2> *imagesOut, int deramp_method);
	 ~cuOverSamplerC2C(); 
};


class cuOverSamplerR2R
{
private:
	 cufftHandle forwardPlan;
	 cufftHandle backwardPlan;
	 cuArrays<float2> *workSizeIn;
	 cuArrays<float2> *workSizeOut;	 
	 cudaStream_t stream;
	 
public:
	cuOverSamplerR2R(int inNX, int inNY, int outNX, int outNY, int nImages, cudaStream_t stream_);
	void setStream(cudaStream_t stream_);
    void execute(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut);	
	~cuOverSamplerR2R(); 
};


#endif



