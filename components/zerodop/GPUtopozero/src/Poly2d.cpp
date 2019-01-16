//
// Author: Joshua Cohen
// Copyright 2016
//
// Note that this is essentially the same thing as the Poly1d, just with
// a set of 2D-type accessor methods

#include <cstdio>
#include <cstdlib>
#include <vector>
#include "Poly2d.h"
using std::vector;

// Direct constructor
Poly2d::Poly2d(int azOrder, int rgOrder) :
    coeffs((azOrder+1),vector<double>(rgOrder+1)) {
    meanRange = 0.0;
    meanAzimuth = 0.0;
    normRange = 1.0;
    normAzimuth = 1.0;
    rangeOrder = rgOrder;
    azimuthOrder = azOrder;
}

void Poly2d::setCoeff2d(int i, int j, double val) {
    coeffs[i][j] = val;
}

double Poly2d::getCoeff2d(int i, int j) {
    double ret;

    ret = coeffs[i][j];
    return ret;
}

double Poly2d::evalPoly2d(double azi, double rng) {
    double scalex,scaley,xval,yval;
    double ret = 0.0;

    xval = (rng - meanRange) / normRange;
    yval = (azi - meanAzimuth) / normAzimuth;
    scaley = 1.0;
    for (int i=0; i<=azimuthOrder; i++,scaley*=yval) {
        scalex = 1.0;
        for (int j=0; j<=rangeOrder; j++,scalex*=xval) {
            ret = ret + (scalex * scaley * coeffs[i][j]);
        }
    }
    return ret;
}

void Poly2d::getBasis2d(double azi, double rng, vector<int> &indices, vector<double> &values, int len) {
    double xval,yval,scalex,scaley;
    int k,ind,ind1;

    xval = (rng - meanRange) / normRange;
    yval = (azi - meanAzimuth) / normAzimuth;
    k = 0;
    ind = indices[0];
    scaley = 1.0;
    for (int i=0; i<=azimuthOrder; i++,scaley*=yval) {
        scalex = scaley;
        for (int j=0; j<=rangeOrder; j++,scalex*=xval) {
            ind1 = (i * (rangeOrder + 1)) + j;
            if (ind1 == ind) {
                values[k] = scalex;
                ind = indices[++k];
            }
        }
    }
}

void Poly2d::printPoly2d() {
    printf("Polynomial Order: %d - by - %d\n", azimuthOrder, rangeOrder);
    for (int i=0; i<=azimuthOrder; i++) {
        for (int j=0; j<=rangeOrder; j++) {
            printf("%g\t", coeffs[i][j]);
        }
        printf("\n");
    }
}

void Poly2d::modifyNorm(Poly2d &targ, double azinorm, double rngnorm) {
    double azfact,rgfact,azratio,rgratio,val;

    if (azimuthOrder > targ.azimuthOrder) {
        printf("Error in Poly2d::modifyNorm - Azimuth orders of source and target are not compatible.\n");
        exit(1);
    }
    if (rangeOrder > targ.rangeOrder) {
        printf("Error in Poly2d::modifyNorm - Range orders of source and target are not compatible.\n");
        exit(1);
    }
    
    azratio = normAzimuth / azinorm;
    rgratio = normRange / rngnorm;
    azfact = 1.0 / azratio;
    
    for (int i=0; i<=azimuthOrder; i++) {
        azfact = azfact * azratio;
        rgfact = 1.0 / rgratio;
        for (int j=0; j<=rangeOrder; j++) {
            rgfact = rgfact * rgratio;
            val = coeffs[i][j];
            targ.setCoeff2d(i,j,(val*rgfact*azfact));
        }
    }
    targ.normAzimuth = azinorm;
    targ.normRange = rngnorm;

    for (int i=(azimuthOrder+1); i<=targ.azimuthOrder; i++) {
        for (int j=(rangeOrder+1); j<=targ.rangeOrder; j++) {
            targ.setCoeff2d(i,j,0.0);
        }
    }
}

