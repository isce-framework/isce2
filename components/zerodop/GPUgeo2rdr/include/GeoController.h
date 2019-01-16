//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef GEOCONTROLLER_H
#define GEOCONTROLLER_H

#include "Geo2rdr.h"

struct GeoController {
    Geo2rdr geo;

    void runGeo2rdr();
    void createOrbit();
    void createPoly();
    
    void setEllipsoidMajorSemiAxis(double);
    void setEllipsoidEccentricitySquared(double);
    void setRangePixelSpacing(double);
    void setRangeFirstSample(double);
    void setPRF(double);
    void setRadarWavelength(double);
    void setSensingStart(double);
    void setLatAccessor(uint64_t);
    void setLonAccessor(uint64_t);
    void setHgtAccessor(uint64_t);
    void setAzAccessor(uint64_t);
    void setRgAccessor(uint64_t);
    void setAzOffAccessor(uint64_t);
    void setRgOffAccessor(uint64_t);
    void setLength(int);
    void setWidth(int);
    void setDemLength(int);
    void setDemWidth(int);
    void setNumberRangeLooks(int);
    void setNumberAzimuthLooks(int);
    void setBistaticFlag(int);
    void setOrbitMethod(int);
    void setOrbitNvecs(int);
    void setOrbitBasis(int);
    void setOrbitVector(int,double,double,double,double,double,double,double);
    void setPolyOrder(int);
    void setPolyMean(double);
    void setPolyNorm(double);
    void setPolyCoeff(int,double);
};

#endif
