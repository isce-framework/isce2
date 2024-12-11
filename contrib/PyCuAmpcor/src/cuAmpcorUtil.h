/*
 * @file cuAmpcorUtil.h
 * @brief Header file to include various routines for cuAmpcor
 *
 *
 */

// code guard
#ifndef __CUAMPCORUTIL_H
#define __CUAMPCORUTIL_H

#include "cuArrays.h"
#include "cuAmpcorParameter.h"
#include "cudaError.h"
#include "debug.h"
#include "cudaUtil.h"
#include "float2.h"


//in cuArraysCopy.cu: various utilities for copy images file in gpu memory
void cuArraysCopyToBatch(cuArrays<float2> *image1, cuArrays<float2> *image2, int strideH, int strideW, cudaStream_t stream);
void cuArraysCopyToBatchWithOffset(cuArrays<float2> *image1, const int inNX, const int inNY,
    cuArrays<float2> *image2, const int *offsetH, const int* offsetW, cudaStream_t stream);
void cuArraysCopyToBatchAbsWithOffset(cuArrays<float2> *image1, const int inNX, const int inNY,
    cuArrays<float2> *image2, const int *offsetH, const int* offsetW, cudaStream_t stream);
void cuArraysCopyToBatchWithOffsetR2C(cuArrays<float> *image1, const int inNX, const int inNY,
    cuArrays<float2> *image2, const int *offsetH, const int* offsetW, cudaStream_t stream);
void cuArraysCopyC2R(cuArrays<float2> *image1, cuArrays<float> *image2, int strideH, int strideW, cudaStream_t stream);

// same routine name overloaded for different data type
// extract data from a large image
template<typename T>
void cuArraysCopyExtract(cuArrays<T> *imagesIn, cuArrays<T> *imagesOut, cuArrays<int2> *offset, cudaStream_t);
template<typename T_in, typename T_out>
void cuArraysCopyExtract(cuArrays<T_in> *imagesIn, cuArrays<T_out> *imagesOut, int2 offset, cudaStream_t);

template<typename T>
void cuArraysCopyInsert(cuArrays<T> *in, cuArrays<T> *out, int offsetX, int offsetY, cudaStream_t);

template<typename T_in, typename T_out>
void cuArraysCopyPadded(cuArrays<T_in> *imageIn, cuArrays<T_out> *imageOut,cudaStream_t stream);
void cuArraysSetConstant(cuArrays<float> *imageIn, float value, cudaStream_t stream);

void cuArraysR2C(cuArrays<float> *image1, cuArrays<float2> *image2, cudaStream_t stream);
void cuArraysC2R(cuArrays<float2> *image1, cuArrays<float> *image2, cudaStream_t stream);
void cuArraysAbs(cuArrays<float2> *image1, cuArrays<float> *image2, cudaStream_t stream);

// cuDeramp.cu: deramping phase
void cuDeramp(int method, cuArrays<float2> *images, cudaStream_t stream);
void cuDerampMethod1(cuArrays<float2> *images, cudaStream_t stream);

// cuArraysPadding.cu: various utilities for oversampling padding
void cuArraysFFTPaddingMany(cuArrays<float2> *image1, cuArrays<float2> *image2, cudaStream_t stream);

//in cuCorrNormalization.cu: utilities to normalize the cross correlation function
void cuArraysSubtractMean(cuArrays<float> *images, cudaStream_t stream);
void cuCorrNormalize(cuArrays<float> *templates, cuArrays<float> *images, cuArrays<float> *results, cudaStream_t stream);
void cuCorrNormalize64(cuArrays<float> *correlation, cuArrays<float> *reference, cuArrays<float> *secondary, cudaStream_t stream);
void cuCorrNormalize128(cuArrays<float> *correlation, cuArrays<float> *reference, cuArrays<float> *secondary, cudaStream_t stream);
void cuCorrNormalize256(cuArrays<float> *correlation, cuArrays<float> *reference, cuArrays<float> *secondary, cudaStream_t stream);
void cuCorrNormalize512(cuArrays<float> *correlation, cuArrays<float> *reference, cuArrays<float> *secondary, cudaStream_t stream);
void cuCorrNormalize1024(cuArrays<float> *correlation, cuArrays<float> *reference, cuArrays<float> *secondary, cudaStream_t stream);

// in cuCorrNormalizationSAT.cu: to normalize the cross correlation function with sum area table
void cuCorrNormalizeSAT(cuArrays<float> *correlation, cuArrays<float> *reference, cuArrays<float> *secondary,
    cuArrays<float> * referenceSum2, cuArrays<float> *secondarySAT, cuArrays<float> *secondarySAT2, cudaStream_t stream);

template<int Size>
void cuCorrNormalizeFixed(cuArrays<float> *correlation, cuArrays<float> *reference, cuArrays<float> *secondary, cudaStream_t stream);

// in cuCorrNormalizationSAT.cu: to normalize the cross correlation function with sum area table
void cuCorrNormalizeSAT(cuArrays<float> *correlation, cuArrays<float> *reference, cuArrays<float> *secondary,
    cuArrays<float> * referenceSum2, cuArrays<float> *secondarySAT, cuArrays<float> *secondarySAT2, cudaStream_t stream);

//in cuOffset.cu: utitilies for determining the max locaiton of cross correlations or the offset
void cuArraysMaxloc2D(cuArrays<float> *images, cuArrays<int2> *maxloc, cuArrays<float> *maxval, cudaStream_t stream);
void cuArraysMaxloc2D(cuArrays<float> *images, cuArrays<int2> *maxloc, cudaStream_t stream);
void cuSubPixelOffset(cuArrays<int2> *offsetInit, cuArrays<int2> *offsetRoomIn, cuArrays<float2> *offsetFinal, const int2 initOrigin, const int initFactor, const int2 zoomInOrigin, const int zoomInFactor, cudaStream_t stream);

void cuDetermineSecondaryExtractOffset(cuArrays<int2> *maxLoc, cuArrays<int2> *maxLocShift,
        int xOldRange, int yOldRange, int xNewRange, int yNewRange, cudaStream_t stream);

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
void cuEstimateVariance(cuArrays<float> *corrBatchRaw, cuArrays<int2> *maxloc, cuArrays<float> *maxval, int templateSize, cuArrays<float3> *covValue, cudaStream_t stream);

#endif

// end of file
