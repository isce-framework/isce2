/*
 * @file  cuAmpcorChunk.h
 * @brief Ampcor processor for a batch of windows
 *
 *
 */

#ifndef __CUAMPCORPROCESSOR_H
#define __CUAMPCORPROCESSOR_H

#include "GDALImage.h"
#include "data_types.h"
#include "cuArrays.h"
#include "cuAmpcorParameter.h"
#include "cuOverSampler.h"
#include "cuSincOverSampler.h"
#include "cuCorrFrequency.h"
#include "cuCorrNormalizer.h"


/**
 * cuAmpcor batched processor (virtual class)
 */
class cuAmpcorProcessor{
// shared variables
protected:
    int idxChunkDown;     ///< index of the chunk in total batches, down
    int idxChunkAcross;   ///< index of the chunk in total batches, across
    int idxChunk;         ///<
    int nWindowsDown;     ///< number of windows in one chunk, down
    int nWindowsAcross;   ///< number of windows in one chunk, across

    int devId;            ///< GPU device ID to use
    cudaStream_t stream;  ///< CUDA stream to use

    GDALImage *referenceImage;  ///< reference image object
    GDALImage *secondaryImage;  ///< secondary image object
    cuAmpcorParameter *param;   ///< reference to the (global) parameters
    cuArrays<real2_type> *offsetImage; ///< output offsets image
    cuArrays<real_type> *snrImage;     ///< snr image
    cuArrays<real3_type> *covImage;    ///< cov image
    cuArrays<real_type> *peakValueImage;     ///< peak value image


public:
    // default constructor and destructor
    cuAmpcorProcessor(cuAmpcorParameter *param_,
        GDALImage *reference_, GDALImage *secondary_,
        cuArrays<real2_type> *offsetImage_, cuArrays<real_type> *snrImage_,
        cuArrays<real3_type> *covImage_, cuArrays<real_type> *peakValueImage_,
        cudaStream_t stream_);
    virtual ~cuAmpcorProcessor() = default;

    // Factory method (virtual constructor)
    static std::unique_ptr<cuAmpcorProcessor> create(int workflow,
        cuAmpcorParameter *param_,
        GDALImage *reference_, GDALImage *secondary_,
        cuArrays<real2_type> *offsetImage_, cuArrays<real_type> *snrImage_,
        cuArrays<real3_type> *covImage_, cuArrays<real_type> *peakValueImage_,
        cudaStream_t stream_);

    // workflow specific methods
    virtual void run(int, int) = 0;

protected:
    // shared methods
    void setIndex(int idxDown_, int idxAcross_);
    void getRelativeOffset(int *rStartPixel, const int *oStartPixel, int diff);

};

#endif //__CUAMPCORPROCESSOR_H
