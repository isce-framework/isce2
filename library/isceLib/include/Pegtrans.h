//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCELIB_PEGTRANS_H
#define ISCELIB_PEGTRANS_H

#include "Ellipsoid.h"
#include "Peg.h"

namespace isceLib {
    struct Pegtrans {
        double mat[3][3], matinv[3][3];
        double ov[3];
        double radcur;

        Pegtrans();
        Pegtrans(const Pegtrans&);
        void radarToXYZ(Ellipsoid&,Peg&);
        void convertSCHtoXYZ(double[3],double[3],int);
        void convertSCHdotToXYZdot(double[3],double[3],double[3],double[3],int);
        void SCHbasis(double[3],double[3][3],double[3][3]);
    };
}

#endif
