//
// Author: Joshua Cohen
// Copyright 2016
//

#ifndef POLY2D_H
#define POLY2D_H

#include <vector>

struct Poly2d {
    std::vector<std::vector<double> > coeffs;
    double meanRange;
    double meanAzimuth;
    double normRange;
    double normAzimuth;
    int rangeOrder;
    int azimuthOrder;

    Poly2d(int,int);
    void setCoeff2d(int,int,double);
    double getCoeff2d(int,int);
    double evalPoly2d(double,double);
    void getBasis2d(double,double,std::vector<int>&,std::vector<double>&,int);
    void printPoly2d();
    void modifyNorm(Poly2d&,double,double);
};

#endif
