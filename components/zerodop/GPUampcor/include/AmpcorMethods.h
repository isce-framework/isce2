//
// Author: Joshua Cohen
// Copyright 2016
//

#ifndef AMPCOR_METHODS_H
#define AMPCOR_METHODS_H

#include <complex>
#include <ctime>
#include <vector>
#include "AmpcorFFT.h"
#include "Constants.h"

struct AmpcorMethods {

    std::vector<double> filter;
    double beta = .75;
    double relfiltlen = 6.;
    double pedestal = 0.;

    int filtercoef;
    int decfactor = 4096;
    int hasWeight = 1;

    std::clock_t innerStart, outerStart;

    // Storing in here as Ampcor only has one line call (looped), but the fourn2d method needs more direct access
    AmpcorFFT aFFT;

    AmpcorMethods() : filter(MAXINTLGH) {};
    void fill_sinc(int&,float&,std::vector<float>&);
    void startOuterClock();
    void startInnerClock();
    double getOuterClock();
    double getInnerClock();
    void correlate(std::vector<std::vector<float> >&,std::vector<std::vector<float> >&,int,int,int,int,int,int,float&,
                    std::vector<float>&,std::vector<std::vector<float> >&,int&,int&,std::vector<int>&,int,bool);
    void derampc(std::vector<std::complex<float> >&,int,int);
    void fourn2d(std::vector<std::complex<float> >&,std::vector<int>&,int);
};

#endif
