//
// Author: Joshua Cohen
// Copyright 2016
//

#ifndef ELLIPSOID_H
#define ELLIPSOID_H

struct Ellipsoid {
    double a;
    double e2;

    Ellipsoid();
    Ellipsoid(double,double);
    Ellipsoid(const Ellipsoid&);
    void latlon(double[3],double[3],int);
    double reast(double);
    double rnorth(double);
    double rdir(double,double);
    void getangs(double[3],double[3],double[3],double&,double&);
    void getTVN_TCvec(double[3],double[3],double[3],double[3]);
    void tcnbasis(double[3],double[3],double[3],double[3],double[3]);
};

#endif
