//
// Author: Joshua Cohen
// Copyright 2017
//

#include <stdio.h>
#include "Poly2d.h"
using isceLib::Poly2d;

Poly2d::Poly2d() {
    // Empty constructor

    azimuthOrder = -1;
    rangeOrder = -1;
    azimuthMean = 0.;
    rangeMean = 0.;
    azimuthNorm = 1.;
    rangeNorm = 1.;
    coeffs = NULL;
}

Poly2d::Poly2d(int aord, int rord, double amn, double rmn, double anrm, double rnrm) {
    // Non-empty constructor

    azimuthOrder = aord;
    rangeOrder = rord;
    azimuthMean = amn;
    rangeMean = rmn;
    azimuthNorm = anrm;
    rangeNorm = rnrm;
    coeffs = new double[(aord+1)*(rord+1)];
}

Poly2d::Poly2d(int aord, int rord, double amn, double rmn, double anrm, double rnrm, double *cfs) {
    // Non-empty constructor (with coefficients)
    
    azimuthOrder = aord;
    rangeOrder = rord;
    azimuthMean = amn;
    rangeMean = rmn;
    azimuthNorm = anrm;
    rangeNorm = rnrm;
    coeffs = new double[(aord+1)*(rord+1)];
    for (int i=0; i<((aord+1)*(rord+1)); i++) coeffs[i] = cfs[i];
}

Poly2d::Poly2d(const Poly2d &poly) {
    // Copy constructor
    
    azimuthOrder = poly.azimuthOrder;
    rangeOrder = poly.rangeOrder;
    azimuthMean = poly.azimuthMean;
    rangeMean = poly.rangeMean;
    azimuthNorm = poly.azimuthNorm;
    rangeNorm = poly.rangeNorm;
    coeffs = new double[(azimuthOrder+1)*(rangeOrder+1)];
    for (int i=0; i<((azimuthOrder+1)*(rangeOrder+1)); i++) coeffs[i] = poly.coeffs[i];
}

Poly2d::~Poly2d() {
    // Destructor
    
    if (coeffs) delete[] coeffs;
}

int Poly2d::isNull() {
    // Safe check method for Python to determine if coeff memory has been malloc'd
    
    if (coeffs) return 0;
    return 1;
}

void Poly2d::resetCoeffs() {
    // Scrub the coefficients and reset the memory based on the stored order
    
    if (coeffs) delete[] coeffs;
    coeffs = new double[(azimuthOrder+1)*(rangeOrder+1)];
    for (int i=0; i<((azimuthOrder+1)*(rangeOrder+1)); i++) coeffs[i] = 0.;
}

void Poly2d::setCoeff(int row, int col, double val) {
    // Set a given coefficient in the polynomial (0-indexed, 2D->1D indexing)

    if ((row > azimuthOrder) || (col > rangeOrder)) printf("Error: Trying to set coefficient (%d,%d) out of [%d,%d] in Poly1d.\n", row, col, azimuthOrder, rangeOrder);
    else coeffs[(row*(rangeOrder+1))+col] = val;
}


double Poly2d::getCoeff(int row, int col) {
    // Get a given coefficient in the polynomial (0-indexed, 2D->1D indexing)

    if ((row > azimuthOrder) || (col > rangeOrder)) printf("Error: Trying to get coefficient (%d,%d) out of [%d,%d] in Poly1d.\n", row, col, azimuthOrder, rangeOrder);
    else return coeffs[(row*(rangeOrder+1))+col];
}

double Poly2d::eval(double azi, double rng) {
    double val, scalex, scaley, xval, yval;
    val = 0.;
    scaley = 1.;
    xval = (rng - rangeMean) / rangeNorm;
    yval = (azi - azimuthMean) / azimuthNorm;
    for (int i=0; i<=azimuthOrder; i++,scaley*=yval) {
        scalex = 1.;
        for (int j=0; j<=rangeOrder; j++,scalex*=xval) {
            val += scalex * scaley * coeffs[(i*(rangeOrder+1))+j];
        }
    }
    return val;
}

