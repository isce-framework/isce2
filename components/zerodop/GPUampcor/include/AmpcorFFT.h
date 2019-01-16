//
// Author: Joshua Cohen
// Copyright 2016
//

#ifndef AMPCOR_FFT_H
#define AMPCOR_FFT_H

#include <complex>
#include <fftw3.h>
#include <vector>
#include "Constants.h"

struct AmpcorFFT {

    std::vector<fftwf_plan> plani, planf;
    std::vector<std::complex<float> > inArr;
    std::vector<int> planFlagCreate, planFlagDestroy;
    bool firstTime = true;

    AmpcorFFT() : plani(16), planf(16), inArr(FFTW_NMAX,std::complex<float>(0.,0.)),
                  planFlagCreate(16,0), planFlagDestroy(16,0) {};
    void fft1d(int,std::complex<float>*,int);
};

#endif
