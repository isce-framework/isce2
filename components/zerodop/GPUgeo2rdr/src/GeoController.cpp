//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cstdint>
#include <cstdio>
#include "Geo2rdr.h"
#include "GeoController.h"
#include "Orbit.h"
#include "Poly1d.h"

void GeoController::runGeo2rdr() {
    geo.geo2rdr();
}

void GeoController::createOrbit() {
    geo.createOrbit();
}

void GeoController::createPoly() {
    geo.createPoly();
}

void GeoController::setEllipsoidMajorSemiAxis(double v) { geo.major = v; }
void GeoController::setEllipsoidEccentricitySquared(double v) { geo.eccentricitySquared = v; }
void GeoController::setRangePixelSpacing(double v) { geo.drho = v; }
void GeoController::setRangeFirstSample(double v) { geo.rngstart = v; }
void GeoController::setPRF(double v) { geo.prf = v; }
void GeoController::setRadarWavelength(double v) { geo.wvl = v; }
void GeoController::setSensingStart(double v) { geo.tstart = v; }
void GeoController::setLatAccessor(uint64_t v) { geo.latAccessor = v; }
void GeoController::setLonAccessor(uint64_t v) { geo.lonAccessor = v; }
void GeoController::setHgtAccessor(uint64_t v) { geo.hgtAccessor = v; }
void GeoController::setAzAccessor(uint64_t v) { geo.azAccessor = v; }
void GeoController::setRgAccessor(uint64_t v) { geo.rgAccessor = v; }
void GeoController::setAzOffAccessor(uint64_t v) { geo.azOffAccessor = v; }
void GeoController::setRgOffAccessor(uint64_t v) { geo.rgOffAccessor = v; }
void GeoController::setLength(int v) { geo.imgLength = v; }
void GeoController::setWidth(int v) { geo.imgWidth = v; }
void GeoController::setDemLength(int v) { geo.demLength = v; }
void GeoController::setDemWidth(int v) { geo.demWidth = v; }
void GeoController::setNumberRangeLooks(int v) { geo.nRngLooks = v; }
void GeoController::setNumberAzimuthLooks(int v) { geo.nAzLooks = v; }
void GeoController::setBistaticFlag(int v) { geo.bistatic = bool(v); }
void GeoController::setOrbitMethod(int v) { geo.orbitMethod = v; }
void GeoController::setOrbitNvecs(int v) { geo.orbit_nvecs = v; }
void GeoController::setOrbitBasis(int v) { geo.orbit_basis = v; }
void GeoController::setOrbitVector(int idx, double t, double px, double py, double pz, double vx, double vy, double vz) {
    double pos[3] = {px, py, pz};
    double vel[3] = {vx, vy, vz};
    geo.orb.setStateVector(idx,t,pos,vel);
}
void GeoController::setPolyOrder(int ord) { geo.poly_order = ord; }
void GeoController::setPolyMean(double mean) { geo.poly_mean = mean; }
void GeoController::setPolyNorm(double norm) { geo.poly_norm = norm; }
void GeoController::setPolyCoeff(int idx, double c) { 
    geo.dop.setCoeff(idx,c); 
}

