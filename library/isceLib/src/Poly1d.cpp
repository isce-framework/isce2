//
// Author: Joshua Cohen
// Copyright 2017
//

#include <stdio.h>
#include "Poly1d.h"
using isceLib::Poly1d;

Poly1d::Poly1d() {
    // Empty constructor
    
    order = -1;
    mean = 0.;
    norm = 1.;
    coeffs = NULL;
}

Poly1d::Poly1d(int ord, double mn, double nrm) {
    // Non-empty constructor

    order = ord;
    mean = mn;
    norm = nrm;
    coeffs = new double[ord+1];
}

Poly1d::Poly1d(int ord, double mn, double nrm, double *cfs) {
    // Non-empty constructor (with coefficients)
    
    order = ord;
    mean = mn;
    norm = nrm;
    coeffs = new double[ord+1];
    for (int i=0; i<=ord; i++) coeffs[i] = cfs[i]; // Copy by value not reference
}

Poly1d::Poly1d(const Poly1d &poly) {
    // Copy constructor

    order = poly.order;
    mean = poly.mean;
    norm = poly.norm;
    coeffs = new double[poly.order+1];
    for (int i=0; i<=poly.order; i++) coeffs[i] = poly.coeffs[i];
}

Poly1d::~Poly1d() {
    // Destructor

    if (coeffs) delete[] coeffs;
}

int Poly1d::isNull() {
    // Safe check method for Python to determine if coeff memory has been malloc'd
    
    if (coeffs) return 0;
    return 1;
}

void Poly1d::resetCoeffs() {
    // Scrub the coefficients and reset the memory based on the stored order

    if (coeffs) delete[] coeffs;
    coeffs = new double[order+1];
    for (int i=0; i<=order; i++) coeffs[i] = 0.;
}

void Poly1d::setCoeff(int idx, double val) {
    // Set a given coefficient in the polynomial (0-indexed)

    if (idx > order) printf("Error: Trying to set coefficient %d out of %d in Poly1d.\n", idx, order);
    else coeffs[idx] = val;
}

double Poly1d::getCoeff(int idx) {
    // Get a given coefficient in the polynomial (0-indexed)
    
    if (idx > order) printf("Error: Trying to get coefficient %d out of %d in Poly1d.\n", idx, order);
    else return coeffs[idx];
}

double Poly1d::eval(double xin) {
    // Evaluate the polynomial at a given position

    double val, scalex, xmod;
    
    val = 0.;
    scalex = 1.;
    xmod = (xin - mean) / norm;
    for (int i=0; i<=order; i++,scalex*=xmod) val += scalex * coeffs[i];
    return val;
}
