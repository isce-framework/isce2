//
// Author: Joshua Cohen
// Copyright 2016
//

#ifndef UNIFORMINTERP_H
#define UNIFORMINTERP_H

#include <complex>
#include <vector>

struct UniformInterp {
    template <class U, class V> U bilinear(double,double,std::vector<std::vector<V> >&);
    template <class U, class V> U bicubic(double,double,std::vector<std::vector<V> >&);
    void sinc_coef(double,double,int,double,int,int&,int&,std::vector<double>&);
    std::complex<double> sinc_eval(std::vector<std::complex<double> >&,int,std::vector<float>&,int,int,int,double);
    template <class U, class V, class W> U sinc_eval_2d(std::vector<std::vector<V> >&,std::vector<W>&,int,int,int,int,double,double,int,int);
    float interp2DSpline(int,int,int,std::vector<std::vector<float> >&,double,double);
    void initSpline(std::vector<double>&,int,std::vector<double>&,std::vector<double>&);
    double spline(double,std::vector<double>&,int,std::vector<double>&);
    int ifrac(double);
    double quadInterpolate(std::vector<double>&,std::vector<double>&,double);
};

#endif
