/*
 * cuCorrFrequency.h
 * define a class to save FFT plans and intermediate data for cross correlation in frequency domain
 */
 
#ifndef __CUCORRFREQUENCY_H
#define __CUCORRFREQUENCY_H
 
#include "cudaUtil.h"
#include "cuArrays.h"

class cuFreqCorrelator
{
private:
	cufftHandle forwardPlan;
	cufftHandle backwardPlan;
	 
	cuArrays<float2> *workFM;
	cuArrays<float2> *workFS;
	cuArrays<float> *workT;
	cudaStream_t stream;

public:
	cuFreqCorrelator(int imageNX, int imageNY, int nImages, cudaStream_t stream_);
	 ~cuFreqCorrelator();
	void execute(cuArrays<float> *templates, cuArrays<float> *images, cuArrays<float> *results);
};

#endif
