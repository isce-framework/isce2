//
// Author: Joshua Cohen
// Copyright 2016
//
// Note that this is as-of-yet unused

#include <vector>
#include "LinAlg.h"
#include "Position.h"
using std::vector;

// Default constructor
Position::Position() :
    j(3),
    jdot(3),
    jddt(3) {
}

void Position::lookvec(double look, double az, vector<double> &v) {
    vector<double> c(3), n(3), t(3), w(3), temp(3);

    LinAlg linalg;
    linalg.unitvec(j,n);
    for (int i=0; i<3; i++) n[i] = -n[i];
    linalg.cross(n,jdot,temp);
    linalg.unitvec(temp,c);
    linalg.cross(c,n,temp);
    linalg.unitvec(temp,t);
    linalg.lincomb(cos(az),t,sin(az),c,temp);
    linalg.lincomb(cos(look),n,sin(look),temp,w);
    linalg.unitvec(w,v);
}
