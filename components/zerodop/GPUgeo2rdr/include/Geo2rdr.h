//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef GEO2RDR_H
#define GEO2RDR_H

#include <cstdint>
#include "Orbit.h"
#include "Poly1d.h"

struct Geo2rdr {
    double major, eccentricitySquared, drho, rngstart, wvl, tstart, prf;

    uint64_t latAccessor, lonAccessor, hgtAccessor, azAccessor, rgAccessor, azOffAccessor, rgOffAccessor;
    
    int imgLength, imgWidth, demLength, demWidth, nRngLooks, nAzLooks, orbit_nvecs, orbit_basis, orbitMethod;
    int poly_order, poly_mean, poly_norm;

    bool bistatic, usr_enable_gpu;

    Poly1d dop;
    Orbit orb;

    Geo2rdr();
    void geo2rdr();
    void createOrbit();
    void createPoly();
};

#endif
