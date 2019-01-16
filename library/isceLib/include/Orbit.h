//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCELIB_ORBIT_H
#define ISCELIB_ORBIT_H

namespace isceLib {
    struct Orbit {
        int basis;
        int nVectors;
        double *position;
        double *velocity;
        double *UTCtime;

        Orbit();
        Orbit(int,int);
        Orbit(const Orbit&);
        ~Orbit();
        int isNull();
        void resetStateVectors();
        void getPositionVelocity(double,double[3],double[3]);
        void getStateVector(int,double&,double[3],double[3]);
        void setStateVector(int,double,double[3],double[3]);
        int interpolate(double,double[3],double[3],int);
        int interpolateWGS84Orbit(double,double[3],double[3]);
        int interpolateLegendreOrbit(double,double[3],double[3]);
        int interpolateSCHOrbit(double,double[3],double[3]);
        int computeAcceleration(double,double[3]);
        void printOrbit();
        void loadFromHDR(const char*,int);
        void dumpToHDR(const char*);
    };

    void orbitHermite(double[4][3],double[4][3],double[4],double,double[3],double[3]);
}

#endif
