/*
 * cuSincOverSampler.h
 * oversampling with sinc interpolation method
 */

#ifndef __CUSINCOVERSAMPLER_H
#define __CUSINCOVERSAMPLER_H

#include "cuArrays.h"
#include "cudaUtil.h"

#define PI 3.141592654f

class cuSincOverSamplerR2R
{
 private:
    static const int i_sincwindow = 2;
    // the oversampling is only performed within \pm i_sincwindow*i_covs around the peak
    static const int i_weight = 1;       // weight for cos() pedestal

    const float r_pedestal = 0.0f;       // height of pedestal
    const float r_beta = 0.75f;          // a low-band pass
    const float r_relfiltlen = 6.0f;     // relative filter length

    static const int i_decfactor = 4096; // decimals between original grid to set up the sinc kernel

    int i_covs;         // oversampling factor
    int i_intplength;   // actual filter length
    int i_filtercoef;   // length of the sinc kernel i_intplength*i_decfactor+1

    float * r_filter;   // sinc kernel with size i_filtercoef

    cudaStream_t stream;

 public:
    // constructor
    cuSincOverSamplerR2R(const int i_covs_, cudaStream_t stream_);
    // local methods
    void setStream(cudaStream_t stream_);
    void cuSetupSincKernel();
    // call interface
    void execute(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut, cuArrays<int2> *center, int oversamplingFactor);

    ~cuSincOverSamplerR2R();
};

#endif // _CUSINCOVERSAMPLER_H

// end of file



