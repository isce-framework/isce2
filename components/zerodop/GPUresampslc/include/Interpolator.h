//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

#include <cfloat>
#include <cstdlib>
#include <vector>

struct Interpolator {
    Interpolator() = default;

    template<class U>
    static U bilinear(double,double,std::vector<std::vector<U>>&);

    template<class U>
    static U bicubic(double,double,std::vector<std::vector<U>>&);

    static void sinc_coef(double,double,int,double,int,int&,int&,std::vector<double>&);

    template<class U, class V>
    static U sinc_eval(std::vector<U>&,std::vector<V>&,int,int,int,double,int);

    template<class U, class V>
    static U sinc_eval_2d(std::vector<std::vector<U>>&,std::vector<V>&,int,int,int,int,double,double,int,int);

    static float interp_2d_spline(int,int,int,std::vector<std::vector<float>>&,double,double);
    static double quadInterpolate(std::vector<double>&,std::vector<double>&,double);
    static double akima(int,int,std::vector<std::vector<float>>&,double,double);
};

void initSpline(std::vector<double>&,int,std::vector<double>&,std::vector<double>&);
double spline(double,std::vector<double>&,int,std::vector<double>&);

#endif
