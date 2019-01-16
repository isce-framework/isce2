//
// Author: Joshua Cohen
// Copyright 2016
//

#ifndef ELLIPSOID_H
#define ELLIPSOID_H

#include <vector>

struct Ellipsoid {
    double a;
    double e2;

    Ellipsoid();
    Ellipsoid(double,double);
    Ellipsoid(const Ellipsoid&);
    void latlon(std::vector<double>&,std::vector<double>&,int);
    double reast(double);
    double rnorth(double);
    double rdir(double,double);
    void getangs(std::vector<double>&,std::vector<double>&,std::vector<double>&,double&,double&);
    void getTVN_TCvec(std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&);
    void tcnbasis(std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&);
};

#endif
