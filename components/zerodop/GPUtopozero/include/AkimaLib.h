//
// Author: Joshua Cohen
// Copyright 2016
//

#ifndef AKIMALIB_H
#define AKIMALIB_H

#include <vector>

struct AkimaLib {
    bool aki_almostEqual(double,double);
    void printAkiNaN(int,int,std::vector<std::vector<float> >&,int,int,double,double,double);
    void getParDer(int,int,std::vector<std::vector<float> >&,int,int,std::vector<std::vector<double> >&,std::vector<std::vector<double> >&,std::vector<std::vector<double> >&);
    void polyfitAkima(int,int,std::vector<std::vector<float> >&,int,int,std::vector<double>&);
    double polyvalAkima(int,int,double,double,std::vector<double>&);
    double akima_intp(int,int,std::vector<std::vector<float> >&,double,double);
};

#endif
