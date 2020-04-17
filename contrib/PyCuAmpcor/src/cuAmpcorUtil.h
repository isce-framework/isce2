/*
 * cuAmpcorUtil.h
 * header file to include the various routines for ampcor
 * serves as an index
 */


#ifndef __CUAMPCORUTIL_H
#define __CUMAPCORUTIL_H

#include "cuArrays.h"
#include "cuAmpcorParameter.h"
#include "cudaError.h"
#include "debug.h"
#include "cudaUtil.h"
#include "float2.h"


//in cuArraysCopy.cu: various utitlies for copy images file in gpu memory
void cuArraysCopyToBatch(cuArrays<float2> *image1, cuArrays<float2> *image2, int strideH, int strideW, cudaStream_t stream);
void cuArraysCopyToBatchWithOffset(cuArrays<float2> *image1, const int lda1, cuArrays<float2> *image2,
	const int *offsetH, const int* offsetW, cudaStream_t stream);
void cuArraysCopyToBatchAbsWithOffset(cuArrays<float2> *image1, const int lda1, cuArrays<float2> *image2,
	const int *offsetH, const int* offsetW, cudaStream_t stream);
void cuArraysCopyToBatchWithOffsetR2C(cuArrays<float> *image1, const int lda1, cuArrays<float2> *image2,
	const int *offsetH, const int* offsetW, cudaStream_t stream);
void cuArraysCopyC2R(cuArrays<float2> *image1, cuArrays<float> *image2, int strideH, int strideW, cudaStream_t stream);

// same routine name overloaded for different data type
void cuArraysCopyExtract(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut, cuArrays<int2> *offset, cudaStream_t stream);
void cuArraysCopyExtract(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut, int2 offset, cudaStream_t stream);
void cuArraysCopyExtract(cuArrays<float2> *imagesIn, cuArrays<float> *imagesOut, int2 offset, cudaStream_t stream);
void cuArraysCopyExtract(cuArrays<float2> *imagesIn, cuArrays<float2> *imagesOut, int2 offset, cudaStream_t stream);
void cuArraysCopyExtract(cuArrays<float2> *imagesIn, cuArrays<float2> *imagesOut, cuArrays<int2> *offsets, cudaStream_t stream);
void cuArraysCopyExtract(cuArrays<float3> *imagesIn, cuArrays<float3> *imagesOut, int2 offset, cudaStream_t stream);

void cuArraysCopyInsert(cuArrays<float2> *imageIn, cuArrays<float2> *imageOut, int offsetX, int offersetY, cudaStream_t stream);
void cuArraysCopyInsert(cuArrays<float3> *imageIn, cuArrays<float3> *imageOut, int offsetX, int offersetY, cudaStream_t stream);
void cuArraysCopyInsert(cuArrays<float> *imageIn, cuArrays<float> *imageOut, int offsetX, int offsetY, cudaStream_t stream);
void cuArraysCopyInsert(cuArrays<int> *imageIn, cuArrays<int> *imageOut, int offsetX, int offersetY, cudaStream_t stream);

void cuArraysCopyInversePadded(cuArrays<float> *imageIn, cuArrays<float> *imageOut,cudaStream_t stream);

void cuArraysCopyPadded(cuArrays<float> *imageIn, cuArrays<float> *imageOut,cudaStream_t stream);
void cuArraysCopyPadded(cuArrays<float> *imageIn, cuArrays<float2> *imageOut,cudaStream_t stream);
void cuArraysCopyPadded(cuArrays<float2> *imageIn, cuArrays<float2> *imageOut,cudaStream_t stream);
void cuArraysSetConstant(cuArrays<float> *imageIn, float value, cudaStream_t stream);

//in cuDeramp.cu: deramping phase
void cuDeramp(int method, cuArrays<float2> *images, cudaStream_t stream);
void cuDerampMethod1(cuArrays<float2> *images, cudaStream_t stream);
void cuDerampMethod2(cuArrays<float2> *images, cudaStream_t stream);
void cpuDerampMethod3(cuArrays<float2> *images, cudaStream_t stream);

//in cuArraysPadding.cu: various utilities for oversampling padding
void cuArraysPadding(cuArrays<float2> *image1, cuArrays<float2> *image2, cudaStream_t stream);
void cuArraysPaddingMany(cuArrays<float2> *image1, cuArrays<float2> *image2, cudaStream_t stream);
void cuArraysR2C(cuArrays<float> *image1, cuArrays<float2> *image2, cudaStream_t stream);
void cuArraysC2R(cuArrays<float2> *image1, cuArrays<float> *image2, cudaStream_t stream);
void cuArraysAbs(cuArrays<float2> *image1, cuArrays<float> *image2, cudaStream_t stream);

//in cuCorrNormalization.cu: utities to normalize the cross correlation function
void cuArraysSubtractMean(cuArrays<float> *images, cudaStream_t stream);
void cuCorrNormalize(cuArrays<float> *templates, cuArrays<float> *images, cuArrays<float> *results, cudaStream_t stream);

//in cuOffset.cu: utitilies for determining the max locaiton of cross correlations or the offset
void cuArraysMaxloc2D(cuArrays<float> *images, cuArrays<int2> *maxloc, cuArrays<float> *maxval, cudaStream_t stream);
void cuArraysMaxloc2D(cuArrays<float> *images, cuArrays<int2> *maxloc, cudaStream_t stream);
void cuSubPixelOffset(cuArrays<int2> *offsetInit, cuArrays<int2> *offsetZoomIn, cuArrays<float2> *offsetFinal,
                      int OverSampleRatioZoomin, int OverSampleRatioRaw,
                      int xHalfRangeInit,  int yHalfRangeInit, int xHalfRangeZoomIn, int yHalfRangeZoomIn,
                      cudaStream_t stream);

void cuDetermineInterpZone(cuArrays<int2> *maxloc, cuArrays<int2> *zoomInOffset, cuArrays<float> *corrOrig, cuArrays<float> *corrZoomIn, cudaStream_t stream);
void cuDetermineSlaveExtractOffset(cuArrays<int2> *maxLoc, int xOldRange, int yOldRange, int xNewRange, int yNewRange, cudaStream_t stream);

//in cuCorrTimeDomain.cu: cross correlation in time domain
void cuCorrTimeDomain(cuArrays<float> *templates, cuArrays<float> *images, cuArrays<float> *results, cudaStream_t stream);

//in cuCorrFrequency.cu: cross correlation in freq domain, also include fft correlatior class
void cuArraysElementMultiply(cuArrays<float2> *image1, cuArrays<float2> *image2, cudaStream_t stream);
void cuArraysElementMultiplyConjugate(cuArrays<float2> *image1, cuArrays<float2> *image2, float coef, cudaStream_t stream);


// For SNR estimation on Correlation surface (Minyan Zhong)
// implemented in cuArraysCopy.cu
void cuArraysCopyExtractCorr(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut, cuArrays<int> *imagesValid, cuArrays<int2> *maxloc, cudaStream_t stream);
// implemented in cuCorrNormalization.cu
void cuArraysSumCorr(cuArrays<float> *images, cuArrays<int> *imagesValid, cuArrays<float> *imagesSum, cuArrays<int> *imagesValidCount, cudaStream_t stream);

// implemented in cuEstimateStats.cu
void cuEstimateSnr(cuArrays<float> *corrSum, cuArrays<int> *corrValidCount, cuArrays<float> *maxval, cuArrays<float> *snrValue, cudaStream_t stream);

// implemented in cuEstimateStats.cu
void cuEstimateVariance(cuArrays<float> *corrBatchRaw, cuArrays<int2> *maxloc, cuArrays<float> *maxval, cuArrays<float3> *covValue, cudaStream_t stream);

#endif
