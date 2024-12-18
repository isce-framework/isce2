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
#include "data_types.h"
#include <cufft.h>

// FFT Oversampler for complex images
class cuOverSamplerC2C
{
private:
     cufftHandle forwardPlan;   // forward fft handle
     cufftHandle backwardPlan;  // backward fft handle
     cudaStream_t stream;       // cuda stream
     cuArrays<complex_type> *workIn;  // work array to hold forward fft data
     cuArrays<complex_type> *workOut; // work array to hold padded data
public:
     // disable the default constructor
     cuOverSamplerC2C() = delete;
     // constructor
     cuOverSamplerC2C(int inNX, int inNY, int outNX, int outNY, int nImages, cudaStream_t stream_);
     // set cuda stream
     void setStream(cudaStream_t stream_);
     // execute oversampling
     void execute(cuArrays<complex_type> *imagesIn, cuArrays<complex_type> *imagesOut, int deramp_method=0);
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
     cuArrays<complex_type> *workSizeIn;
     cuArrays<complex_type> *workSizeOut;

public:
    cuOverSamplerR2R() = delete;
    cuOverSamplerR2R(int inNX, int inNY, int outNX, int outNY, int nImages, cudaStream_t stream_);
    void setStream(cudaStream_t stream_);
    void execute(cuArrays<real_type> *imagesIn, cuArrays<real_type> *imagesOut);
    ~cuOverSamplerR2R();
};


#endif //__CUOVERSAMPLER_H
// end of file



