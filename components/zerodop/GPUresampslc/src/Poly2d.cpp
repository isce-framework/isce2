//
// Author: Joshua Cohen
// Copyright 2017
//

#include <iostream>
#include "Poly2d.h"
#include "Constants.h"
using std::cout;
using std::endl;

double Poly2d::eval(double azi, double rng) {
    
    double xval = (rng - rangeMean) / rangeNorm;
    double yval = (azi - azimuthMean) / azimuthNorm;
    
    double scalex;
    double scaley = 1.;
    double val = 0.;
    for (int i=0; i<=azimuthOrder; i++,scaley*=yval) {
        scalex = 1.;
        for (int j=0; j<=rangeOrder; j++,scalex*=xval) {
            val += scalex * scaley * coeffs[IDX1D(i,j,rangeOrder+1)];
        }
    }
    return val;
}

void Poly2d::printPoly() {
    cout << "Polynomial Order: " << azimuthOrder << " - by - " << rangeOrder << endl;
    for (int i=0; i<=azimuthOrder; i++) {
        for (int j=0; j<=rangeOrder; j++) {
            cout << getCoeff(i,j) << " ";
        }
        cout << endl;
    }
}

