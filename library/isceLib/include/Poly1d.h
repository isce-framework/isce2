//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCELIB_POLY1D_H
#define ISCELIB_POLY1D_H

namespace isceLib {
    struct Poly1d {
        int order;
        double mean;
        double norm;
        double *coeffs;

        Poly1d();
        Poly1d(int,double,double);
        Poly1d(int,double,double,double*);
        Poly1d(const Poly1d&);
        ~Poly1d();
        int isNull();
        void resetCoeffs();
        void setCoeff(int,double);
        double getCoeff(int);
        double eval(double);
    };
}

#endif
