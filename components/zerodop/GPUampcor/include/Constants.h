//
// Author: Joshua Cohen
// Copyright 2016
//

#ifndef AMPCOR_CONSTANTS_H
#define AMPCOR_CONSTANTS_H

const int DTYPE_REAL = 0;
const int DTYPE_COMPLEX = 1;
const int DTYPE_MAG = 2;

const int INPUT_STYLE_NEW = 0;
const int INPUT_STYLE_OLD = 1;
const int INPUT_STYLE_RDF = 2;

const int OSAMP_SINC = 1;
const int OSAMP_FOURIER = 2;

const int MAXDECFACTOR = 4096; // maximum lags in interpolation kernels
const int MAXINTKERLGH = 256; // maximum interpolation kernel length
const int MAXINTLGH = MAXINTKERLGH * MAXDECFACTOR; // maximum interpolation kernel array size

const int FFTW_NMAX = 32768;

#endif
