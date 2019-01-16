//
// Author: Joshua Cohen
// Copyright 2016
//

#ifndef ORBIT_H
#define ORBIT_H

struct Orbit {
    int nVectors;
    int basis;
    double *position;
    double *velocity;
    double *UTCtime;

    Orbit();
    Orbit(const Orbit&);
    ~Orbit();
    void setOrbit(int,int);
    void setOrbit(const char*,int);
    void getPositionVelocity(double,double[3],double[3]);
    void setStateVector(int,double,double[3],double[3]);
    void getStateVector(int,double&,double[3],double[3]);
    int interpolateOrbit(double,double[3],double[3],int);
    int interpolateSCHOrbit(double,double[3],double[3]);
    int interpolateWGS84Orbit(double,double[3],double[3]);
    int interpolateLegendreOrbit(double,double[3],double[3]);
    int computeAcceleration(double,double[3]);
    void orbitHermite(double[4][3],double[4][3],double[3],double,double[3],double[3]);
    void dumpToHDR(const char*);
    void printOrbit();
};

#endif
