//
// Author: Joshua Cohen
// Copyright 2017
//

#include <math.h>
#include "LinAlg.h"
#include "Position.h"
using isceLib::LinAlg;
using isceLib::Position;

Position::Position() {
    // Empty constructor
    
    return;
}

Position::Position(const Position &p) {
    // Copy constructor
    
    for (int i=0; i<3; i++) {
        j[i] = p.j[i];
        jdot[i] = p.jdot[i];
        jddt[i] = p.jddt[i];
    }
}

void Position::lookVec(double look, double az, double v[3]) {
    // Computes the look vector given the look angle, azimuth angle, and position vector
    
    double n[3], temp[3], c[3], t[3], w[3];
    LinAlg alg;

    alg.unitVec(j, n);
    for (int i=0; i<3; i++) n[i] = -n[i];
    alg.cross(n, jdot, temp);
    alg.unitVec(temp, c);
    alg.cross(c, n, temp);
    alg.unitVec(temp, t);
    alg.linComb(cos(az), t, sin(az), c, temp);
    alg.linComb(cos(look), n, sin(look), temp, w);
    alg.unitVec(w, v);
}
