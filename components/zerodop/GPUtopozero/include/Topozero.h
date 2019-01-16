//
// Author: Joshua Cohen
// Copyright 2016
//

#ifndef TOPOZERO_H
#define TOPOZERO_H

#include "Topo.h"

struct Topozero {
    Topo topo;

    void runTopo();
    void createOrbit();

    void setFirstLat(double);
    void setFirstLon(double);
    void setDeltaLat(double);
    void setDeltaLon(double);
    void setMajor(double);
    void setEccentricitySquared(double);
    void setRspace(double);
    void setR0(double);
    void setPegHdg(double);
    void setPrf(double);
    void setT0(double);
    void setWvl(double);
    void setThresh(double);
    void setDemAccessor(uint64_t);
    void setDopAccessor(uint64_t);
    void setSlrngAccessor(uint64_t);
    void setLatAccessor(uint64_t);
    void setLonAccessor(uint64_t);
    void setLosAccessor(uint64_t);
    void setHeightAccessor(uint64_t);
    void setIncAccessor(uint64_t);
    void setMaskAccessor(uint64_t);
    void setNumIter(int);
    void setIdemWidth(int);
    void setIdemLength(int);
    void setIlrl(int);
    void setExtraIter(int);
    void setLength(int);
    void setWidth(int);
    void setNrngLooks(int);
    void setNazLooks(int);
    void setDemMethod(int);
    void setOrbitMethod(int);
    void setOrbitNvecs(int);
    void setOrbitBasis(int);
    void setOrbitVector(int,double,double,double,double,double,double,double);
};

#endif
