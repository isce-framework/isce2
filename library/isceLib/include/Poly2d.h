//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCELIB_POLY2D_H
#define ISCELIB_POLY2D_H

namespace isceLib {
    struct Poly2d {
        int rangeOrder;
        int azimuthOrder;
        double rangeMean;
        double azimuthMean;
        double rangeNorm;
        double azimuthNorm;
        double *coeffs;

        Poly2d();
        Poly2d(int,int,double,double,double,double);
        Poly2d(int,int,double,double,double,double,double*);
        Poly2d(const Poly2d&);
        ~Poly2d();
        int isNull();
        void resetCoeffs();
        void setCoeff(int,int,double);
        double getCoeff(int,int);
        double eval(double,double);
    };
}

#endif
