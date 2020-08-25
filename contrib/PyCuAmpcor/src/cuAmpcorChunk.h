/*
 * cuAmpcorChunk.h
 * Purpose: a group of chips processed at the same time
*/

#ifndef __CUAMPCORCHUNK_H
#define __CUAMPCORCHUNK_H

#include "GDALImage.h"
#include "cuArrays.h"
#include "cuAmpcorParameter.h"
#include "cuOverSampler.h"
#include "cuSincOverSampler.h"
#include "cuCorrFrequency.h"

class cuAmpcorChunk{
private:
    int idxChunkDown;
	int idxChunkAcross;
    int idxChunk;
    int nWindowsDown;
    int nWindowsAcross;

	int devId;
	cudaStream_t stream;

	GDALImage *referenceImage;
	GDALImage *secondaryImage;
	cuAmpcorParameter *param;
	cuArrays<float2> *offsetImage;
	cuArrays<float> *snrImage;
	cuArrays<float3> *covImage;

	// added for test
    cuArrays<int> *intImage1;
    cuArrays<float> *floatImage1;

    // gpu buffer
	cuArrays<float2> * c_referenceChunkRaw, * c_secondaryChunkRaw;
	cuArrays<float> * r_referenceChunkRaw, * r_secondaryChunkRaw;

	// gpu windows raw data
    cuArrays<float2> * c_referenceBatchRaw, * c_secondaryBatchRaw, * c_secondaryBatchZoomIn;
    cuArrays<float> * r_referenceBatchRaw, * r_secondaryBatchRaw;

    // gpu windows oversampled data
    cuArrays<float2> * c_referenceBatchOverSampled, * c_secondaryBatchOverSampled;
    cuArrays<float> * r_referenceBatchOverSampled, * r_secondaryBatchOverSampled;
    cuArrays<float> * r_corrBatchRaw, * r_corrBatchZoomIn, * r_corrBatchZoomInOverSampled, * r_corrBatchZoomInAdjust;

    cuArrays<int> *ChunkOffsetDown, *ChunkOffsetAcross;

	cuOverSamplerC2C *referenceBatchOverSampler, *secondaryBatchOverSampler;

    cuOverSamplerR2R *corrOverSampler;
    cuSincOverSamplerR2R *corrSincOverSampler;

	//for frequency domain
	cuFreqCorrelator *cuCorrFreqDomain, *cuCorrFreqDomain_OverSampled;

	cuArrays<int2> *offsetInit;
	cuArrays<int2> *offsetZoomIn;
	cuArrays<float2> *offsetFinal;
    cuArrays<float> *corrMaxValue;


    //SNR estimation

    cuArrays<float> *r_corrBatchRawZoomIn;
    cuArrays<float> *r_corrBatchSum;
    cuArrays<int> *i_corrBatchZoomInValid, *i_corrBatchValidCount;

    cuArrays<float> *r_snrValue;

    cuArrays<int2> *i_maxloc;
    cuArrays<float> *r_maxval;

    // Varince estimation.
    cuArrays<float3> *r_covValue;

public:
	cuAmpcorChunk()	{}
	//cuAmpcorChunk(cuAmpcorParameter *param_, SlcImage *reference_, SlcImage *secondary_);

	void setIndex(int idxDown_, int idxAcross_);

	cuAmpcorChunk(cuAmpcorParameter *param_, GDALImage *reference_, GDALImage *secondary_, cuArrays<float2> *offsetImage_,
	            cuArrays<float> *snrImage_, cuArrays<float3> *covImage_, cuArrays<int> *intImage1_, cuArrays<float> *floatImage1_, cudaStream_t stream_);


    void loadReferenceChunk();
    void loadSecondaryChunk();
    void getRelativeOffset(int *rStartPixel, const int *oStartPixel, int diff);

    ~cuAmpcorChunk();

	void run(int, int);
};



#endif
