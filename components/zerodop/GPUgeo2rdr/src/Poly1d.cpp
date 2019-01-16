//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cstddef>
#include <cstdio>
#include "Poly1d.h"

Poly1d::Poly1d() {
    coeffs = NULL;
    order = 0;
    mean = 0.;
    norm = 1.;
}

Poly1d::Poly1d(int ord) {
    order = ord;
    mean = 0.;
    norm = 1.;
    coeffs = new double[order+1];
}

Poly1d::Poly1d(const Poly1d &poly) {
    order = poly.order;
    mean = poly.mean;
    norm = poly.norm;
    coeffs = new double[order+1];
    for (int i=0; i<=order; i++) coeffs[i] = poly.coeffs[i];
}

Poly1d::~Poly1d() {
    if (coeffs) delete[] coeffs;
}

void Poly1d::setPoly(int ord, double mn, double nrm) {
    if (coeffs) delete[] coeffs;
    coeffs = new double[ord+1];
    order = ord;
    mean = mn;
    norm = nrm;
}

double Poly1d::eval(double xin) {
    double value, xval, scalex;
    value = 0.;
    scalex = 1.;

    xval = (xin - mean) / norm;
    for (int i=0; i<=order; i++,scalex*=xval) value += scalex * coeffs[i];

    return value;
}

void Poly1d::setCoeff(int idx, double val) {
    coeffs[idx] = val;
}

double Poly1d::getCoeff(int idx) {
    return coeffs[idx];
}

void Poly1d::printPoly() {
    printf("%d %f %f\n", order, mean, norm);
    for (int i=0; i<=order; i++) printf("%g ",coeffs[i]);
    printf("\n");
}
