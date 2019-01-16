//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef POLY1D_H
#define POLY1D_H

struct Poly1d {
    double *coeffs;
    double mean, norm;
    int order;

    Poly1d();
    Poly1d(int);
    Poly1d(const Poly1d&);
    ~Poly1d();
    void setPoly(int,double,double);
    double eval(double);
    void setCoeff(int,double);
    double getCoeff(int);
    void printPoly();
};

#endif
