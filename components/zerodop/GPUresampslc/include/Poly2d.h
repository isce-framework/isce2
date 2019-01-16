//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef POLY2D_H
#define POLY2D_H

#include <stdexcept>
#include <string>
#include <vector>
#include "Constants.h"

struct Poly2d {
    int rangeOrder;
    int azimuthOrder;
    double rangeMean;
    double azimuthMean;
    double rangeNorm;
    double azimuthNorm;
    std::vector<double> coeffs;

    Poly2d(int ro, int ao, double rm, double am, double rn, double an) : rangeOrder(ro), azimuthOrder(ao), rangeMean(rm), azimuthMean(am), 
                                                                            rangeNorm(rn), azimuthNorm(an), coeffs((ro+1)*(ao+1)) {}
    Poly2d() : Poly2d(-1,-1,0.,0.,1.,1.) {}
    Poly2d(const Poly2d &p) : rangeOrder(p.rangeOrder), azimuthOrder(p.azimuthOrder), rangeMean(p.rangeMean), azimuthMean(p.azimuthMean),
                                rangeNorm(p.rangeNorm), azimuthNorm(p.azimuthNorm), coeffs(p.coeffs) {}
    inline Poly2d& operator=(const Poly2d&);

    inline void setCoeff(int,int,double);
    inline double getCoeff(int,int);
    double eval(double,double);
    void printPoly();
};

inline Poly2d& Poly2d::operator=(const Poly2d &rhs) {
    rangeOrder = rhs.rangeOrder;
    azimuthOrder = rhs.azimuthOrder;
    rangeMean = rhs.rangeMean;
    azimuthMean = rhs.azimuthMean;
    rangeNorm = rhs.rangeNorm;
    azimuthNorm = rhs.azimuthNorm;
    return *this;
}

inline void Poly2d::setCoeff(int row, int col, double val) {
    if ((row < 0) || (row > azimuthOrder)) {
        std::string errstr = "Poly2d::setCoeff - Trying to set coefficient for row "+std::to_string(row+1)+" out of "+std::to_string(azimuthOrder+1);
        throw std::out_of_range(errstr);
    }
    if ((col < 0) || (col > rangeOrder)) {
        std::string errstr = "Poly2d::setCoeff - Trying to set coefficient for col "+std::to_string(col+1)+" out of "+std::to_string(rangeOrder+1);
        throw std::out_of_range(errstr);
    }
    coeffs[IDX1D(row,col,rangeOrder+1)] = val;
}

inline double Poly2d::getCoeff(int row, int col) {
    if ((row < 0) || (row > azimuthOrder)) {
        std::string errstr = "Poly2d::getCoeff - Trying to get coefficient for row "+std::to_string(row+1)+" out of "+std::to_string(azimuthOrder+1);
        throw std::out_of_range(errstr);
    }
    if ((col < 0) || (col > rangeOrder)) {
        std::string errstr = "Poly2d::getCoeff - Trying to get coefficient for col "+std::to_string(col+1)+" out of "+std::to_string(rangeOrder+1);
        throw std::out_of_range(errstr);
    }
    return coeffs[IDX1D(row,col,rangeOrder+1)];
}

#endif
