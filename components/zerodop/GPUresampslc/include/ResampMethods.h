//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef RESAMP_METHODS_H
#define RESAMP_METHODS_H

#include <complex>
#include <vector>
using std::complex;
using std::vector;

struct ResampMethods {
    vector<float> fintp;
    float f_delay;

    ResampMethods();
    void prepareMethods(int);
    complex<float> interpolate_cx(vector<vector<complex<float> > >&,int,int,double,double,int,int,int);
};

#endif
