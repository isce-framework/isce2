//
// Author: Joshua Cohen
// Copyright 2016
//

#ifndef ORBIT_H
#define ORBIT_H

#include <vector>

struct Orbit {
    int nVectors;
    int basis;
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> UTCtime;

    Orbit();
    void setOrbit(int,int);
    void setOrbit(const char*,int);
    void getPositionVelocity(double,std::vector<double>&,std::vector<double>&);
    void setStateVector(int,double,std::vector<double>&,std::vector<double>&);
    void getStateVector(int,double&,std::vector<double>&,std::vector<double>&);
    int interpolateOrbit(double,std::vector<double>&,std::vector<double>&,int);
    int interpolateSCHOrbit(double,std::vector<double>&,std::vector<double>&);
    int interpolateWGS84Orbit(double,std::vector<double>&,std::vector<double>&);
    int interpolateLegendreOrbit(double,std::vector<double>&,std::vector<double>&);
    int computeAcceleration(double,std::vector<double>&);
    void orbitHermite(std::vector<std::vector<double> >&,std::vector<std::vector<double> >&,std::vector<double>&,double,std::vector<double>&,std::vector<double>&);
    void dumpToHDR(const char*);
    void printOrbit();
};

#endif
