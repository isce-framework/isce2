//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCELIB_ELLIPSOID_H
#define ISCELIB_ELLIPSOID_H

namespace isceLib {
    struct Ellipsoid {
        double a, e2;

        Ellipsoid();
        Ellipsoid(double,double);
        Ellipsoid(const Ellipsoid&);
        void setMajorSemiAxis(double);
        void setEccentricitySquared(double);
        double rEast(double);
        double rNorth(double);
        double rDir(double,double);
        void latLon(double[3],double[3],int);
        void getAngs(double[3],double[3],double[3],double&,double&);
        void getTCN_TCvec(double[3],double[3],double[3],double[3]);
    };
}

#endif
