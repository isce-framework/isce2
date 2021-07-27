/*
 * @file  cuAmpcorChunk.h
 * @brief Ampcor processor for a batch of windows
 *
 *
 */

#ifndef __CUAMPCORCHUNK_H
#define __CUAMPCORCHUNK_H

#include "GDALImage.h"
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
    cuArrays<float2> *offsetImage; ///< output offsets image
    cuArrays<float> *snrImage;     ///< snr image
    cuArrays<float3> *covImage;    ///< cov image

    // local variables and workers
    // gpu buffer to load images from file
    cuArrays<float2> * c_referenceChunkRaw, * c_secondaryChunkRaw;
    cuArrays<float> * r_referenceChunkRaw, * r_secondaryChunkRaw;

    // windows raw (not oversampled) data, complex and real
    cuArrays<float2> * c_referenceBatchRaw, * c_secondaryBatchRaw, * c_secondaryBatchZoomIn;
    cuArrays<float> * r_referenceBatchRaw, * r_secondaryBatchRaw;

    // windows oversampled data
    cuArrays<float2> * c_referenceBatchOverSampled, * c_secondaryBatchOverSampled;
    cuArrays<float> * r_referenceBatchOverSampled, * r_secondaryBatchOverSampled;
    cuArrays<float> * r_corrBatchRaw, * r_corrBatchZoomIn, * r_corrBatchZoomInOverSampled, * r_corrBatchZoomInAdjust;

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
    cuArrays<float2> *offsetFinal;
    cuArrays<int2> *maxLocShift; // record the maxloc from the extract center
    cuArrays<float> *corrMaxValue;
    cuArrays<int2> *i_maxloc;
    cuArrays<float> *r_maxval;

    // SNR estimation
    cuArrays<float> *r_corrBatchRawZoomIn;
    cuArrays<float> *r_corrBatchSum;
    cuArrays<int> *i_corrBatchZoomInValid, *i_corrBatchValidCount;
    cuArrays<float> *r_snrValue;

    // Variance estimation
    cuArrays<float3> *r_covValue;

public:
    // constructor
    cuAmpcorChunk(cuAmpcorParameter *param_,
        GDALImage *reference_, GDALImage *secondary_,
        cuArrays<float2> *offsetImage_, cuArrays<float> *snrImage_,
        cuArrays<float3> *covImage_, cudaStream_t stream_);
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
