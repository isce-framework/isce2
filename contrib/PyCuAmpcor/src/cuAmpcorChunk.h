/*
 * @file  cuAmpcorChunk.h
 * @brief Ampcor processor for a batch of windows
 *
 *
 */

#ifndef __CUAMPCORCHUNK_H
#define __CUAMPCORCHUNK_H

#include "GDALImage.h"
#include "data_types.h"
#include "cuArrays.h"
#include "cuAmpcorParameter.h"
#include "cuOverSampler.h"
#include "cuSincOverSampler.h"
#include "cuCorrFrequency.h"
#include "cuCorrNormalizer.h"


/**
 * cuAmpcor processor for a chunk (a batch of windows)
 */
class cuAmpcorChunk{
private:
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

    // local variables and workers
    // gpu buffer to load images from file
    // image_complex_type uses original image type,
    //    convert to complex_type when copied to c_referenceBatchRaw
    cuArrays<image_complex_type> * c_referenceChunkRaw, * c_secondaryChunkRaw;
    cuArrays<image_real_type> * r_referenceChunkRaw, * r_secondaryChunkRaw;

    // windows raw (not oversampled) data, complex and real
    cuArrays<complex_type> * c_referenceBatchRaw, * c_secondaryBatchRaw;
    cuArrays<real_type> * r_referenceBatchRaw, * r_secondaryBatchRaw;

    // windows oversampled data
    cuArrays<complex_type> * c_referenceBatchOverSampled, * c_secondaryBatchOverSampled;
    cuArrays<real_type> * r_referenceBatchOverSampled, * r_secondaryBatchOverSampled;
    cuArrays<real_type> * r_corrBatch, * r_corrBatchZoomIn, * r_corrBatchZoomInOverSampled;

    // offset data
    cuArrays<int> *ChunkOffsetDown, *ChunkOffsetAcross;

    // oversampling processors for complex images
    cuOverSamplerC2C *referenceBatchOverSampler, *secondaryBatchOverSampler;

    // oversampling processor for correlation surface
    cuOverSamplerR2R *corrOverSampler;
    cuSincOverSamplerR2R *corrSincOverSampler;

    // cross-correlation processor with frequency domain algorithm
    cuFreqCorrelator *cuCorrFreqDomain, *cuCorrFreqDomain_OverSampled;

    // correlation surface normalizer
    std::unique_ptr<cuNormalizeProcessor> corrNormalizerRaw;
    std::unique_ptr<cuNormalizeProcessor> corrNormalizerOverSampled;

    // save offset results in different stages
    cuArrays<int2> *offsetInit;
    cuArrays<int2> *offsetZoomIn;
    cuArrays<complex_type> *offsetFinal;
    cuArrays<int2> *maxLocShift; // record the maxloc from the extract center
    cuArrays<real_type> *corrMaxValue;
    cuArrays<int2> *i_maxloc;
    cuArrays<real_type> *r_maxval;

    // SNR estimation
    cuArrays<real_type> *r_corrBatchRawZoomIn;
    cuArrays<real_type> *r_corrBatchSum;
    cuArrays<real_type> *r_snrValue;

    // Variance estimation
    cuArrays<real3_type> *r_covValue;

public:
    // constructor
    cuAmpcorChunk(cuAmpcorParameter *param_,
        GDALImage *reference_, GDALImage *secondary_,
        cuArrays<real2_type> *offsetImage_, cuArrays<real_type> *snrImage_,
        cuArrays<real3_type> *covImage_, cuArrays<real_type> *peakValueImage_,
        cudaStream_t stream_);
    // destructor
    ~cuAmpcorChunk();

    // local methods
    void setIndex(int idxDown_, int idxAcross_);
    void loadReferenceChunk();
    void loadSecondaryChunk();
    void getRelativeOffset(int *rStartPixel, const int *oStartPixel, int diff);
    // run the given chunk
    void run(int, int);
};



#endif
