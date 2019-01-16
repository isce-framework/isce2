//
// Author: Joshua Cohen
// Copyright 2016
//

#ifndef TOPOMETHODS_H
#define TOPOMETHODS_H

#include <vector>

struct TopoMethods {
    std::vector<float> fintp;
    float f_delay;

    TopoMethods();
    TopoMethods(const TopoMethods&);
    void prepareMethods(int);
    float interpolate(std::vector<std::vector<float> >&,int,int,double,double,int,int,int);
    float intp_sinc(std::vector<std::vector<float> >&,int,int,double,double,int,int);
    float intp_bilinear(std::vector<std::vector<float> >&,int,int,double,double,int,int);
    float intp_bicubic(std::vector<std::vector<float> >&,int,int,double,double,int,int);
    float intp_nearest(std::vector<std::vector<float> >&,int,int,double,double,int,int);
    float intp_akima(std::vector<std::vector<float> >&,int,int,double,double,int,int);
    float intp_biquintic(std::vector<std::vector<float> >&,int,int,double,double,int,int);
};

#endif

