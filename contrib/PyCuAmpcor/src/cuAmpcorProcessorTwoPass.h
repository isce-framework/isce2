/*
 * @file  cuAmpcorProcessorTwoPass.h
 * @brief Ampcor processor for a batch of windows with TwoPass workflow
 *
 *
 */

#ifndef __CUAMPCORPROCESSORTwoPass_H
#define __CUAMPCORPROCESSORTwoPass_H

#include "cuAmpcorProcessor.h"


/**
 * cuAmpcor processor for a chunk (a batch of windows)
 */
class cuAmpcorProcessorTwoPass : public cuAmpcorProcessor{
private:

    // local variables and workers
    // gpu buffer to load images from file
    cuArrays<image_complex_type> * c_referenceChunkRaw, * c_secondaryChunkRaw;
    cuArrays<image_real_type> * r_referenceChunkRaw, * r_secondaryChunkRaw;

    // windows raw (not oversampled) data, complex and real
    cuArrays<complex_type> * c_referenceBatchRaw, * c_secondaryBatchRaw, * c_secondaryBatchZoomIn;
    cuArrays<real_type> * r_referenceBatchRaw, * r_secondaryBatchRaw;

    // windows oversampled data
    cuArrays<complex_type> * c_referenceBatchOverSampled, * c_secondaryBatchOverSampled;
    cuArrays<real_type> * r_referenceBatchOverSampled, * r_secondaryBatchOverSampled;
    cuArrays<real_type> * r_corrBatchRaw, * r_corrBatchZoomIn, * r_corrBatchZoomInOverSampled, * r_corrBatchZoomInAdjust;

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
    cuArrays<int> *i_corrBatchZoomInValid, *i_corrBatchValidCount;
    cuArrays<real_type> *r_snrValue;

    // Variance estimation
    cuArrays<real3_type> *r_covValue;

public:
    // constructor
    cuAmpcorProcessorTwoPass(cuAmpcorParameter *param_,
        SlcImage *reference_, SlcImage *secondary_,
        cuArrays<complex_type> *offsetImage_, cuArrays<real_type> *snrImage_,
        cuArrays<real3_type> *covImage_, cuArrays<real_type> *peakValueImage_,
        cudaStream_t stream_);
    // destructor
    ~cuAmpcorProcessorTwoPass() override;

    // local methods
    void loadReferenceChunk();
    void loadSecondaryChunk();
    // run the given chunk
    void run(int, int) override;
};



#endif //__CUAMPCORPROCESSORTwoPass_H
