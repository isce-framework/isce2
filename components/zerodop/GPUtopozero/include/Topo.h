//
// Author: Joshua Cohen
// Copyright 2016
//

#ifndef TOPO_H
#define TOPO_H

#include <cstdint>
#include "Orbit.h"

struct Topo {
    double firstlat, firstlon, deltalat, deltalon;
    double major, eccentricitySquared, rspace, r0;
    double peghdg, prf, t0, wvl, thresh;

    uint64_t demAccessor, dopAccessor, slrngAccessor;
    uint64_t latAccessor, lonAccessor, losAccessor;
    uint64_t heightAccessor, incAccessor, maskAccessor;

    int numiter, idemwidth, idemlength, ilrl, extraiter;
    int length, width, Nrnglooks, Nazlooks, dem_method;
    int orbit_method, orbit_nvecs, orbit_basis;

    Orbit orb;

    void createOrbit();
    //void writeToFile(void**,double**,bool,bool,int,int,bool);
    void topo();
};

#endif
