//
// Author: Joshua Cohen
// Copyright 2016
//
//  This is a wrapper class for Topo so that we can hide most of the ugly stuff
//  from Cython and the users.

#include <vector>
#include "Topo.h"
#include "Topozero.h"
using std::vector;

void Topozero::runTopo() {
    topo.topo();
}

void Topozero::createOrbit() {
    topo.createOrbit();
}

void Topozero::setFirstLat(double v) {topo.firstlat = v;};
void Topozero::setFirstLon(double v) {topo.firstlon = v;};
void Topozero::setDeltaLat(double v) {topo.deltalat = v;};
void Topozero::setDeltaLon(double v) {topo.deltalon = v;};
void Topozero::setMajor(double v) {topo.major = v;};
void Topozero::setEccentricitySquared(double v) {topo.eccentricitySquared = v;};
void Topozero::setRspace(double v) {topo.rspace = v;};
void Topozero::setR0(double v) {topo.r0 = v;};
void Topozero::setPegHdg(double v) {topo.peghdg = v;};
void Topozero::setPrf(double v) {topo.prf = v;};
void Topozero::setT0(double v) {topo.t0 = v;};
void Topozero::setWvl(double v) {topo.wvl = v;};
void Topozero::setThresh(double v) {topo.thresh = v;};
void Topozero::setDemAccessor(uint64_t v) {topo.demAccessor = v;};
void Topozero::setDopAccessor(uint64_t v) {topo.dopAccessor = v;};
void Topozero::setSlrngAccessor(uint64_t v) {topo.slrngAccessor = v;};
void Topozero::setLatAccessor(uint64_t v) {topo.latAccessor = v;};
void Topozero::setLonAccessor(uint64_t v) {topo.lonAccessor = v;};
void Topozero::setLosAccessor(uint64_t v) {topo.losAccessor = v;};
void Topozero::setHeightAccessor(uint64_t v) {topo.heightAccessor = v;};
void Topozero::setIncAccessor(uint64_t v) {topo.incAccessor = v;};
void Topozero::setMaskAccessor(uint64_t v) {topo.maskAccessor = v;};
void Topozero::setNumIter(int v) {topo.numiter = v;};
void Topozero::setIdemWidth(int v) {topo.idemwidth = v;};
void Topozero::setIdemLength(int v) {topo.idemlength = v;};
void Topozero::setIlrl(int v) {topo.ilrl = v;};
void Topozero::setExtraIter(int v) {topo.extraiter = v;};
void Topozero::setLength(int v) {topo.length = v;};
void Topozero::setWidth(int v) {topo.width = v;};
void Topozero::setNrngLooks(int v) {topo.Nrnglooks = v;};
void Topozero::setNazLooks(int v) {topo.Nazlooks = v;};
void Topozero::setDemMethod(int v) {topo.dem_method = v;};
void Topozero::setOrbitMethod(int v) {topo.orbit_method = v;};
void Topozero::setOrbitNvecs(int v) {topo.orbit_nvecs = v;};
void Topozero::setOrbitBasis(int v) {topo.orbit_basis = v;};
// Passed in as single values as interfacing with Cython using arrays/vectors is costly and not worth it
void Topozero::setOrbitVector(int idx, double t, double px, double py, double pz, double vx, double vy, double vz) {
    double pos[] = {px,py,pz};
    double vel[] = {vx,vy,vz};
    vector<double> position(pos,(pos+3)), velocity(vel,(vel+3));
    topo.orb.setStateVector(idx,t,position,velocity);
}
