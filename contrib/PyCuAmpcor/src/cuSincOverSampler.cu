/**
 * @file cuSincOverSampler.cu
 * @brief Implementation for cuSinOversampler class
 *
 */

// my declaration
#include "cuSincOverSampler.h"

// dependencies
#include "cuArrays.h"
#include "cudaUtil.h"
#include "cudaError.h"
#include "cuAmpcorUtil.h"

/**
 * cuSincOverSamplerR2R constructor
 * @param i_covs oversampling factor
 * @param stream cuda stream
 */
cuSincOverSamplerR2R::cuSincOverSamplerR2R(const int i_covs_, cudaStream_t stream_)
 : i_covs(i_covs_)
{
    stream = stream_;
    i_intplength = int(r_relfiltlen/r_beta+0.5f);
    i_filtercoef = i_intplength*i_decfactor;
    checkCudaErrors(cudaMalloc((void **)&r_filter, (i_filtercoef+1)*sizeof(real_type)));
    cuSetupSincKernel();
}

/// destructor
cuSincOverSamplerR2R::~cuSincOverSamplerR2R()
{
    checkCudaErrors(cudaFree(r_filter));
}

// cuda kernel for cuSetupSincKernel
__global__ void cuSetupSincKernel_kernel(real_type *r_filter_, const int i_filtercoef_,
    const real_type r_soff_, const real_type r_wgthgt_, const int i_weight_,
    const real_type r_soff_inverse_, const real_type r_beta_, const real_type r_decfactor_inverse_)
{
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i > i_filtercoef_) return;
    real_type r_wa = i - r_soff_;
    real_type r_wgt = (1.0f - r_wgthgt_) + r_wgthgt_*cos(PI*r_wa*r_soff_inverse_);
    real_type r_s = r_wa*r_beta_*r_decfactor_inverse_*PI;
    real_type r_fct;
    if(r_s != 0.0) {
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
}

/**
 * Set up the sinc interpolation kernel (coefficient)
 */
void cuSincOverSamplerR2R::cuSetupSincKernel()
{
    const int nthreads = 128;
    const int nblocks = IDIVUP(i_filtercoef+1, nthreads);

    // compute some commonly used constants at first
    real_type r_wgthgt =  (1.0f - r_pedestal)/2.0f;
    real_type r_soff = (i_filtercoef-1.0f)/2.0f;
    real_type r_soff_inverse = 1.0f/r_soff;
    real_type r_decfactor_inverse = 1.0f/i_decfactor;

    cuSetupSincKernel_kernel<<<nblocks, nthreads, 0, stream>>> (
        r_filter, i_filtercoef, r_soff, r_wgthgt, i_weight,
        r_soff_inverse, r_beta, r_decfactor_inverse);
    getLastCudaError("cuSetupSincKernel_kernel");
}


// cuda kernel for cuSincOverSamplerR2R::execute
// define a device function to overload the cases with constant or varying center shifts
__device__ void cuSincInterpolation_kernel_common(const int nImages,
    const real_type * imagesIn, const int inNX, const int inNY,
    real_type * imagesOut, const int outNX, const int outNY,
    int2 centerShift, int factor,
    const real_type * r_filter_, const int i_covs_, const int i_decfactor_, const int i_intplength_,
    const int i_startX, const int i_startY, const int i_int_size)
{
    // get image index
    int idxImage = blockIdx.z;
    // get the xy threads for output image pixel indices
    int idxX = threadIdx.x + blockDim.x*blockIdx.x;
    int idxY = threadIdx.y + blockDim.y*blockIdx.y;
    // cuda: to make sure extra allocated threads doing nothing
    if(idxImage >=nImages || idxX >= i_int_size || idxY >= i_int_size) return;
    // determine the output pixel indices
    int outx = idxX + i_startX + centerShift.x*factor;
    if (outx >= outNX) outx-=outNX;
    int outy = idxY + i_startY +  centerShift.y*factor;
    if (outy >= outNY) outy-=outNY;
    // flattened to 1d
    int idxOut = idxImage*outNX*outNY + outx*outNY + outy;

    // index in input grids
    real_type r_xout = (real_type)outx/i_covs_;
     // integer part
    int i_xout = int(r_xout);
    // factional part
    real_type r_xfrac = r_xout - i_xout;
    // fractional part in terms of the interpolation kernel grids
    int i_xfrac = int(r_xfrac*i_decfactor_);

    // same procedure for y
    real_type r_yout = (real_type)outy/i_covs_;
    int i_yout = int(r_yout);
    real_type r_yfrac = r_yout - i_yout;
    int i_yfrac = int(r_yfrac*i_decfactor_);

    // temp variables
    real_type intpData = 0.0; // interpolated value
    real_type r_sincwgt = 0.0; // total filter weight
    real_type r_sinc_coef; // filter weight

    // iterate over lines of input image
    // i=0 -> -i_intplength/2
    for(int i=0; i < i_intplength_; i++) {
        // find the corresponding pixel in input(unsampled) image

        int inx = i_xout - i + i_intplength_/2;

        if(inx < 0) inx+= inNX;
        if(inx >= inNX) inx-= inNY;

        real_type r_xsinc_coef = r_filter_[i*i_decfactor_+i_xfrac];

        for(int j=0; j< i_intplength_; j++) {
            // find the corresponding pixel in input(unsampled) image
            int iny = i_yout - j + i_intplength_/2;
            if(iny < 0) iny += inNY;
            if(iny >= inNY) iny -= inNY;

            real_type r_ysinc_coef = r_filter_[j*i_decfactor_+i_yfrac];
            // multiply the factors from xy
            r_sinc_coef = r_xsinc_coef*r_ysinc_coef;
            // add to total sinc weight
            r_sincwgt += r_sinc_coef;
            // multiply by the original signal and add to results
            intpData += imagesIn[idxImage*inNX*inNY+inx*inNY+iny]*r_sinc_coef;

        }
    }
    imagesOut[idxOut] = intpData/r_sincwgt;
}

//constant shift
__global__ void cuSincInterpolation_kernel(const int nImages,
    const real_type * imagesIn, const int inNX, const int inNY,
    real_type * imagesOut, const int outNX, const int outNY,
    int2 centerShift, int factor,
    const real_type * r_filter_, const int i_covs_, const int i_decfactor_, const int i_intplength_,
    const int i_startX, const int i_startY, const int i_int_size)
 {
    // pass through
    cuSincInterpolation_kernel_common(nImages, imagesIn, inNX, inNY,
        imagesOut, outNX, outNY, centerShift, factor,
        r_filter_, i_covs_, i_decfactor_, i_intplength_,
        i_startX, i_startY, i_int_size);
 }

__global__ void cuSincInterpolation_kernel(const int nImages,
    const real_type * imagesIn, const int inNX, const int inNY,
    real_type * imagesOut, const int outNX, const int outNY,
    int2 *centerShift, int factor,
    const real_type * r_filter_, const int i_covs_, const int i_decfactor_, const int i_intplength_,
    const int i_startX, const int i_startY, const int i_int_size)
 {
    // get image index
    int idxImage = blockIdx.z;
    cuSincInterpolation_kernel_common(nImages, imagesIn, inNX, inNY,
        imagesOut, outNX, outNY, centerShift[idxImage], factor,
        r_filter_, i_covs_, i_decfactor_, i_intplength_,
        i_startX, i_startY, i_int_size);
 }

/**
 * Execute sinc interpolation
 * @param[in] imagesIn input images
 * @param[out] imagesOut output images
 * @param[in] centerShift the shift of interpolation center
 * @param[in] rawOversamplingFactor the multiplier of the centerShift
 * @note rawOversamplingFactor is for the centerShift, not the signal oversampling factor
 */

void cuSincOverSamplerR2R::execute(cuArrays<real_type> *imagesIn, cuArrays<real_type> *imagesOut,
    cuArrays<int2> *centerShift, int rawOversamplingFactor)
{
    const int nImages = imagesIn->count;
    const int inNX = imagesIn->height;
    const int inNY = imagesIn->width;
    const int outNX = imagesOut->height;
    const int outNY = imagesOut->width;

    // only compute the overampled signals within a window
    const int i_int_range = i_sincwindow * i_covs;
    // set the start pixel, will be shifted by centerShift*oversamplingFactor (from raw image)
    const int i_int_startX = outNX/2 - i_int_range;
    const int i_int_startY = outNY/2 - i_int_range;
    const int i_int_size = 2*i_int_range + 1;
    // preset all pixels in out image to 0
    imagesOut->setZero(stream);

    static const int nthreads = 16;
    dim3 threadsperblock(nthreads, nthreads, 1);
    dim3 blockspergrid (IDIVUP(i_int_size, nthreads), IDIVUP(i_int_size, nthreads), nImages);
    cuSincInterpolation_kernel<<<blockspergrid, threadsperblock, 0, stream>>>(nImages,
        imagesIn->devData, inNX, inNY,
        imagesOut->devData, outNX, outNY,
        centerShift->devData, rawOversamplingFactor,
        r_filter, i_covs, i_decfactor, i_intplength, i_int_startX, i_int_startY, i_int_size);
    getLastCudaError("cuSincInterpolation_kernel");
}

void cuSincOverSamplerR2R::execute(cuArrays<real_type> *imagesIn, cuArrays<real_type> *imagesOut,
    int2 centerShift, int rawOversamplingFactor)
{
    const int nImages = imagesIn->count;
    const int inNX = imagesIn->height;
    const int inNY = imagesIn->width;
    const int outNX = imagesOut->height;
    const int outNY = imagesOut->width;

    // only compute the overampled signals within a window
    const int i_int_range = i_sincwindow * i_covs;
    // set the start pixel, will be shifted by centerShift*oversamplingFactor (from raw image)
    const int i_int_startX = outNX/2 - i_int_range;
    const int i_int_startY = outNY/2 - i_int_range;
    const int i_int_size = 2*i_int_range + 1;
    // preset all pixels in out image to 0
    imagesOut->setZero(stream);

    static const int nthreads = 16;
    dim3 threadsperblock(nthreads, nthreads, 1);
    dim3 blockspergrid (IDIVUP(i_int_size, nthreads), IDIVUP(i_int_size, nthreads), nImages);
    cuSincInterpolation_kernel<<<blockspergrid, threadsperblock, 0, stream>>>(nImages,
        imagesIn->devData, inNX, inNY,
        imagesOut->devData, outNX, outNY,
        centerShift, rawOversamplingFactor,
        r_filter, i_covs, i_decfactor, i_intplength, i_int_startX, i_int_startY, i_int_size);
    getLastCudaError("cuSincInterpolation_kernel");
}

// end of file