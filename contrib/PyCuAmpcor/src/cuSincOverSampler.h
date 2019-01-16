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
    static const int i_decfactor = 4096; // division between orignal pixels
    static const int i_weight = 1;       // weight for cos() pedestal
    const float r_pedestal = 0.0f; // height of pedestal  
    const float r_beta = 0.75f;     // factor r_relfiltlen/i_intplength 
    
    int i_covs;
    int i_intplength;
    float r_relfiltlen;  
    int i_filtercoef;
    float r_wgthgt;
    float r_soff;
    float r_soff_inverse;
    float r_decfactor_inverse;
    
    cudaStream_t stream;
    float * r_filter; 
    
 public:
    cuSincOverSamplerR2R(const int i_intplength_, const int i_covs_, cudaStream_t stream_);
    void setStream(cudaStream_t stream_);
    void cuSetupSincKernel();
    void execute(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut);	
    ~cuSincOverSamplerR2R(); 
};

#endif // _CUSINCOVERSAMPLER_H

// end of file



