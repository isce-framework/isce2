/*
 * @file cuAmpcorUtil.h
 * @brief Header file to include various routines for cuAmpcor
 *
 *
 */

// code guard
#ifndef __CUAMPCORUTIL_H
#define __CUAMPCORUTIL_H

#include "data_types.h"
#include "cuArrays.h"
#include "cuAmpcorParameter.h"
#include "cudaError.h"
#include "debug.h"
#include "cudaUtil.h"
#include "float2.h"


//in cuArraysCopy.cu: various utilities for copy images file in gpu memory
void cuArraysCopyToBatch(cuArrays<image_complex_type> *image1, cuArrays<complex_type> *image2, int strideH, int strideW, cudaStream_t stream);
void cuArraysCopyToBatchWithOffset(cuArrays<image_complex_type> *image1, const int inNX, const int inNY,
    cuArrays<complex_type> *image2, const int *offsetH, const int* offsetW, cudaStream_t stream);
void cuArraysCopyToBatchAbsWithOffset(cuArrays<image_complex_type> *image1, const int inNX, const int inNY,
    cuArrays<complex_type> *image2, const int *offsetH, const int* offsetW, cudaStream_t stream);
void cuArraysCopyToBatchWithOffsetR2C(cuArrays<image_real_type> *image1, const int inNX, const int inNY,
    cuArrays<complex_type> *image2, const int *offsetH, const int* offsetW, cudaStream_t stream);
void cuArraysCopyC2R(cuArrays<complex_type> *image1, cuArrays<real_type> *image2, int strideH, int strideW, cudaStream_t stream);

// same routine name overloaded for different data type
// extract data from a large image
template<typename T>
void cuArraysCopyExtract(cuArrays<T> *imagesIn, cuArrays<T> *imagesOut, cuArrays<int2> *offset, cudaStream_t);
template<typename T_in, typename T_out>
void cuArraysCopyExtract(cuArrays<T_in> *imagesIn, cuArrays<T_out> *imagesOut, int2 offset, cudaStream_t);
void cuArraysCopyExtractAbs(cuArrays<complex_type> *imagesIn, cuArrays<real_type> *imagesOut, int2 offset, cudaStream_t stream);

template<typename T>
void cuArraysCopyInsert(cuArrays<T> *in, cuArrays<T> *out, int offsetX, int offsetY, cudaStream_t);

template<typename T_in, typename T_out>
void cuArraysCopyPadded(cuArrays<T_in> *imageIn, cuArrays<T_out> *imageOut,cudaStream_t stream);
void cuArraysSetConstant(cuArrays<real_type> *imageIn, real_type value, cudaStream_t stream);

void cuArraysR2C(cuArrays<real_type> *image1, cuArrays<complex_type> *image2, cudaStream_t stream);
void cuArraysC2R(cuArrays<complex_type> *image1, cuArrays<real_type> *image2, cudaStream_t stream);
void cuArraysAbs(cuArrays<complex_type> *image1, cuArrays<real_type> *image2, cudaStream_t stream);

// cuDeramp.cu: deramping phase
void cuDeramp(int method, cuArrays<complex_type> *images, cudaStream_t stream);
void cuDerampMethod1(cuArrays<complex_type> *images, cudaStream_t stream);

// cuArraysPadding.cu: various utilities for oversampling padding
void cuArraysFFTPaddingMany(cuArrays<complex_type> *image1, cuArrays<complex_type> *image2, cudaStream_t stream);

//in cuCorrNormalization.cu: utilities to normalize the cross correlation function
void cuArraysSubtractMean(cuArrays<real_type> *images, cudaStream_t stream);
void cuCorrNormalize(cuArrays<real_type> *templates, cuArrays<real_type> *images, cuArrays<real_type> *results, cudaStream_t stream);
void cuCorrNormalize64(cuArrays<real_type> *correlation, cuArrays<real_type> *reference, cuArrays<real_type> *secondary, cudaStream_t stream);
void cuCorrNormalize128(cuArrays<real_type> *correlation, cuArrays<real_type> *reference, cuArrays<real_type> *secondary, cudaStream_t stream);
void cuCorrNormalize256(cuArrays<real_type> *correlation, cuArrays<real_type> *reference, cuArrays<real_type> *secondary, cudaStream_t stream);
void cuCorrNormalize512(cuArrays<real_type> *correlation, cuArrays<real_type> *reference, cuArrays<real_type> *secondary, cudaStream_t stream);
void cuCorrNormalize1024(cuArrays<real_type> *correlation, cuArrays<real_type> *reference, cuArrays<real_type> *secondary, cudaStream_t stream);

// in cuCorrNormalizationSAT.cu: to normalize the cross correlation function with sum area table
void cuCorrNormalizeSAT(cuArrays<real_type> *correlation, cuArrays<real_type> *reference, cuArrays<real_type> *secondary,
    cuArrays<real_type> * referenceSum2, cuArrays<real_type> *secondarySAT, cuArrays<real_type> *secondarySAT2, cudaStream_t stream);

template<int Size>
void cuCorrNormalizeFixed(cuArrays<real_type> *correlation, cuArrays<real_type> *reference, cuArrays<real_type> *secondary, cudaStream_t stream);

// in cuCorrNormalizationSAT.cu: to normalize the cross correlation function with sum area table
void cuCorrNormalizeSAT(cuArrays<real_type> *correlation, cuArrays<real_type> *reference, cuArrays<real_type> *secondary,
    cuArrays<real_type> * referenceSum2, cuArrays<real_type> *secondarySAT, cuArrays<real_type> *secondarySAT2, cudaStream_t stream);

//in cuOffset.cu: utitilies for determining the max locaiton of cross correlations or the offset
void cuArraysMaxloc2D(cuArrays<real_type> *images, cuArrays<int2> *maxloc, cuArrays<real_type> *maxval, cudaStream_t stream);
void cuArraysMaxloc2D(cuArrays<real_type> *images, const int2 start, const int2 range, cuArrays<int2> *maxloc, cuArrays<real_type> *maxval, cudaStream_t stream);
void cuSubPixelOffset(cuArrays<int2> *offsetInit, cuArrays<int2> *offsetRoomIn, cuArrays<complex_type> *offsetFinal, const int2 initOrigin, const int initFactor, const int2 zoomInOrigin, const int zoomInFactor, cudaStream_t stream);

void cuDetermineSecondaryExtractOffset(cuArrays<int2> *maxLoc, cuArrays<int2> *maxLocShift,
        int xOldRange, int yOldRange, int xNewRange, int yNewRange, cudaStream_t stream);

//in cuCorrTimeDomain.cu: cross correlation in time domain
void cuCorrTimeDomain(cuArrays<real_type> *templates, cuArrays<real_type> *images, cuArrays<real_type> *results, cudaStream_t stream);

//in cuCorrFrequency.cu: cross correlation in freq domain, also include fft correlatior class
void cuArraysElementMultiply(cuArrays<complex_type> *image1, cuArrays<complex_type> *image2, cudaStream_t stream);
void cuArraysElementMultiplyConjugate(cuArrays<complex_type> *image1, cuArrays<complex_type> *image2, real_type coef, cudaStream_t stream);


// For SNR estimation on Correlation surface (Minyan Zhong)
// implemented in cuArraysCopy.cu
void cuArraysCopyExtractCorr(cuArrays<real_type> *imagesIn, cuArrays<real_type> *imagesOut, cuArrays<int> *imagesValid, cuArrays<int2> *maxloc, cudaStream_t stream);
// implemented in cuCorrNormalization.cu
void cuArraysSumCorr(cuArrays<real_type> *images, cuArrays<int> *imagesValid, cuArrays<real_type> *imagesSum, cuArrays<int> *imagesValidCount, cudaStream_t stream);

// implemented in cuEstimateStats.cu
void cuEstimateSnr(cuArrays<real_type> *corrSum, cuArrays<int> *corrValidCount, cuArrays<real_type> *maxval, cuArrays<real_type> *snrValue, cudaStream_t stream);

// implemented in cuEstimateStats.cu
void cuEstimateVariance(cuArrays<real_type> *corrBatchRaw, cuArrays<int2> *maxloc, cuArrays<real_type> *maxval, int templateSize, cuArrays<real3_type> *covValue, cudaStream_t stream);

#endif

// end of file
