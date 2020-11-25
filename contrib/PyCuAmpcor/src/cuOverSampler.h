/*
 * @file cuOverSampler.h
 * @brief Oversampling with FFT padding method
 *
 * Define cuOverSampler class, to save cufft plans and perform oversampling calculations
 * For float images use cuOverSamplerR2R
 * For complex images use cuOverSamplerC2C
 * @todo use template class to unify these two classes
 */

#ifndef __CUOVERSAMPLER_H
#define __CUOVERSAMPLER_H

#include "cuArrays.h"
#include "cudaUtil.h"

// FFT Oversampler for complex images
class cuOverSamplerC2C
{
private:
     cufftHandle forwardPlan;   // forward fft handle
     cufftHandle backwardPlan;  // backward fft handle
     cudaStream_t stream;       // cuda stream
     cuArrays<float2> *workIn;  // work array to hold forward fft data
     cuArrays<float2> *workOut; // work array to hold padded data
public:
     // disable the default constructor
     cuOverSamplerC2C() = delete;
     // constructor
     cuOverSamplerC2C(int inNX, int inNY, int outNX, int outNY, int nImages, cudaStream_t stream_);
     // set cuda stream
     void setStream(cudaStream_t stream_);
     // execute oversampling
     void execute(cuArrays<float2> *imagesIn, cuArrays<float2> *imagesOut, int deramp_method=0);
     // destructor
     ~cuOverSamplerC2C();
};

// FFT Oversampler for complex images
class cuOverSamplerR2R
{
private:
     cufftHandle forwardPlan;
     cufftHandle backwardPlan;
     cudaStream_t stream;
     cuArrays<float2> *workSizeIn;
     cuArrays<float2> *workSizeOut;

public:
    cuOverSamplerR2R() = delete;
    cuOverSamplerR2R(int inNX, int inNY, int outNX, int outNY, int nImages, cudaStream_t stream_);
    void setStream(cudaStream_t stream_);
    void execute(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut);
    ~cuOverSamplerR2R();
};


#endif //__CUOVERSAMPLER_H
// end of file



