/*
 * @file  cuCorrFrequency.h
 * @brief A class performs cross correlation in frequency domain
 */

// code guard
#ifndef __CUCORRFREQUENCY_H
#define __CUCORRFREQUENCY_H

// dependencies
#include "cuArrays.h"
#include "data_types.h"
#include <cufft.h>

class cuFreqCorrelator
{
private:
    // handles for forward/backward fft
    cufftHandle forwardPlan;
    cufftHandle backwardPlan;
    // work data
    cuArrays<complex_type> *workFM;
    cuArrays<complex_type> *workFS;
    cuArrays<real_type> *workT;
    // cuda stream
    cudaStream_t stream;

public:
    // constructor
    cuFreqCorrelator(int imageNX, int imageNY, int nImages, cudaStream_t stream_);
    // destructor
    ~cuFreqCorrelator();
    // executor
    void execute(cuArrays<real_type> *templates, cuArrays<real_type> *images, cuArrays<real_type> *results);
};

#endif //__CUCORRFREQUENCY_H
// end of file
