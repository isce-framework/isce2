//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef CONSTANTS_H
#define CONSTANTS_H

// General
static const double SPEED_OF_LIGHT = 299792458.;
static const float BAD_VALUE = -999999.;

// Orbit interpolation
static const int HERMITE_METHOD = 0;
static const int SCH_METHOD = 1;
static const int LEGENDRE_METHOD = 2;

static const int WGS84_ORBIT = 1;
static const int SCH_ORBIT = 2;

// Ellipsoid latlon
static const int LLH_2_XYZ = 1;
static const int XYZ_2_LLH = 2;
static const int XYZ_2_LLH_OLD = 3;

#endif
