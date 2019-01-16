//
// Author: Joshua Cohen
// Copyright 2016
//

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cfloat>

// AkimaLib
const int AKI_NSYS = 16;
const int AKI_EPS = DBL_EPSILON;

// Ellipsoid + PegTrans
const int LLH_2_XYZ = 1;
const int XYZ_2_LLH = 2;
const int XYZ_2_LLH_OLD = 3;

// Orbit
const int WGS84_ORBIT = 1;
const int SCH_ORBIT = 2;

// Orbit + topozeroState
const int HERMITE_METHOD = 0;
const int SCH_METHOD = 1;
const int LEGENDRE_METHOD = 2;

// PegTrans
const int SCH_2_XYZ = 0;
const int XYZ_2_SCH = 1;
const int LLH_2_UTM = 1;
const int UTM_2_LLH = 2;

// TopoMethods
const int SINC_LEN = 8;
const int SINC_SUB = 8192;
const int SINC_METHOD = 0;
const int BILINEAR_METHOD = 1;
const int BICUBIC_METHOD = 2;
const int NEAREST_METHOD = 3;
const int AKIMA_METHOD = 4;
const int BIQUINTIC_METHOD = 5;
const float BADVALUE = -1000.0;

// topozeroState
const double MIN_H = -500.0;
const double MAX_H = -1000.0;
const double MARGIN = 0.15;

#endif
