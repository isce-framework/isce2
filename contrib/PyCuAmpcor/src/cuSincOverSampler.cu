/* 
 * cuSincOverSampler.cu
 */
#include "cuArrays.h"
#include "cuSincOverSampler.h"
#include "cuArrays.h"
#include "cudaUtil.h"
#include "cudaError.h"
#include "cuAmpcorUtil.h"

cuSincOverSamplerR2R::cuSincOverSamplerR2R(const int i_intplength_, const int i_covs_, cudaStream_t stream_)
 : i_intplength(i_intplength_), i_covs(i_covs_)
{
    setStream(stream_);
    //i_intplength = int(r_relfiltlen/r_beta);
    r_relfiltlen = r_beta * i_intplength;
    i_filtercoef = i_intplength*i_decfactor;
    r_wgthgt = (1.0f - r_pedestal)/2.0f;
    r_soff = (i_filtercoef)/2.0f;
    r_soff_inverse = 1.0f/r_soff;
    r_decfactor_inverse = 1.0f/i_decfactor;
    checkCudaErrors(cudaMalloc((void **)&r_filter, (i_filtercoef+1)*sizeof(float)));
    cuSetupSincKernel();
}

void cuSincOverSamplerR2R::setStream(cudaStream_t stream_)
{
    stream = stream_;
}

cuSincOverSamplerR2R::~cuSincOverSamplerR2R() 
{
    checkCudaErrors(cudaFree(r_filter));
}


__global__ void cuSetupSincKernel_kernel(float *r_filter_, const int i_filtercoef_, 
    const float r_soff_, const float r_wgthgt_, const int i_weight_,
    const float r_soff_inverse_, const float r_beta_, const float r_decfactor_inverse_,
    const float r_relfiltlen_inverse_)
{
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i > i_filtercoef_) return;
    float r_wa = i - r_soff_;
    float r_wgt = (1.0f - r_wgthgt_) + r_wgthgt_*cos(PI*r_wa*r_soff_inverse_);
    float r_s = r_wa*r_beta_*r_decfactor_inverse_*PI;
    float r_fct; 
    if(r_s != 0.0f) {
        r_fct = sin(r_s)/r_s;
    }
    else {
        r_fct = 1.0f;
    }
    if(i_weight_ == 1) {
        r_filter_[i] = r_fct*r_wgt;
    }
    else {
        r_filter_[i] = r_fct;
    }
    //printf("kernel %d %f\n", i, r_filter_[i]);
}


void cuSincOverSamplerR2R::cuSetupSincKernel()
{
    const int nthreads = 128;
    const int nblocks = IDIVUP(i_filtercoef, nthreads);
    float r_relfiltlen_inverse = 1.0f/r_relfiltlen; 
    cuSetupSincKernel_kernel<<<nblocks, nthreads, 0, stream>>> (
        r_filter, i_filtercoef, r_soff, r_wgthgt, i_weight, 
        r_soff_inverse, r_beta, r_decfactor_inverse, r_relfiltlen_inverse);
    getLastCudaError("cuSetupSincKernel_kernel");
}

__global__ void cuSincInterpolation_kernel(const int nImages, 
    const float * imagesIn, const int inNX, const int inNY,
    float * imagesOut, const int outNX, const int outNY, 
    const float * r_filter_, const int i_covs_, const int i_decfactor_, const int i_intplength_, 
    const int i_startX, const int i_startY, const int i_int_size)
{
    int idxImage = blockIdx.z;
    int idxX = threadIdx.x + blockDim.x*blockIdx.x; 
    int idxY = threadIdx.y + blockDim.y*blockIdx.y;
    if(idxImage >=nImages || idxX >= i_int_size || idxY >= i_int_size) return;
    int outx = idxX + i_startX;
    int outy = idxY + i_startY;
    int idxOut = idxImage*outNX*outNY + outx*outNY + outy;
    
    float r_xout = (float)outx/i_covs_;
    int i_xout = int(r_xout);
    float r_xfrac = r_xout - i_xout;
    int i_xfrac = int(r_xfrac*i_decfactor_);
    
    float r_yout = (float)outy/i_covs_;
    int i_yout = int(r_yout);
    float r_yfrac = r_yout - i_yout;
    int i_yfrac = int(r_yfrac*i_decfactor_);
    
    float intpData = 0.0f;
    float r_sincwgt = 0.0f;
    float r_sinc_coef;
    
    for(int i=0; i < inNX; i++) {
        int i_xindex = i_xout - i + i_intplength_/2;
        if(i_xindex < 0) i_xindex+= i_intplength_;
        if(i_xindex >= i_intplength_) i_xindex-=i_intplength_;  
        float r_xsinc_coef = r_filter_[i_xindex*i_decfactor_+i_xfrac];
        
        for(int j=0; j< inNY; j++) {
            int i_yindex = i_yout - j + i_intplength_/2;
            if(i_yindex < 0) i_yindex+= i_intplength_;
            if(i_yindex >= i_intplength_) i_yindex-=i_intplength_;  
            float r_ysinc_coef = r_filter_[i_yindex*i_decfactor_+i_yfrac];
            r_sinc_coef = r_xsinc_coef*r_ysinc_coef;
            r_sincwgt += r_sinc_coef;
            intpData += imagesIn[idxImage*inNX*inNY+i*inNY+j]*r_sinc_coef;
            /*
              if(outx == 0 && outy == 1) {
                printf("intp kernel %d %d %d %d %d %d %d %f\n", i, j, i_xindex, i_yindex, i_xindex*i_decfactor_+i_xfrac,
                   i_yindex*i_decfactor_+i_yfrac, idxImage*inNX*inNY+i*inNY+j, r_sinc_coef);
              }*/
        }
    }
    imagesOut[idxOut] = intpData/r_sincwgt;
    //printf("test int kernel %d %d %f %f %f\n", outx, outy, intpData, r_sincwgt, imagesOut[idxOut]);
}


void cuSincOverSamplerR2R::execute(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut)
{
    const int nImages = imagesIn->count;
    const int inNX = imagesIn->height;
    const int inNY = imagesIn->width;
    const int outNX = imagesOut->height; 
    const int outNY = imagesOut->width;
    
    const int i_int_range = i_sincwindow * i_covs; 
    const int i_int_startX = outNX/2 - i_int_range;
    const int i_int_startY = outNY/2 - i_int_range;
    const int i_int_size = 2*i_int_range + 1;
      
    imagesOut->setZero(stream);
    
    static const int nthreads = 16;
    dim3 threadsperblock(nthreads, nthreads, 1);
    dim3 blockspergrid (IDIVUP(i_int_size, nthreads), IDIVUP(i_int_size, nthreads), nImages);
    cuSincInterpolation_kernel<<<blockspergrid, threadsperblock, 0, stream>>>(nImages, 
        imagesIn->devData, inNX, inNY,
        imagesOut->devData, outNX, outNY,
        r_filter, i_covs, i_decfactor, i_intplength, i_int_startX, i_int_startY, i_int_size);
    getLastCudaError("cuSincInterpolation_kernel");
}

// end of file



