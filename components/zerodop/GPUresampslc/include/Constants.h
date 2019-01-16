//
// Author: Joshua Cohen
// Co0pyright 2017
//

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cmath>

// Macro wrapper to provide 2D indexing to a 1D array
#define IDX1D(i,j,w) (((i)*(w))+(j))
// Since fmod(a,b) in C++ != MODULO(a,b) in Fortran for all a,b, define a C++ equivalent
#define modulo_f(a,b) std::fmod(std::fmod(a,b)+(b),(b))

// Data interpolation
static const int SINC_METHOD = 1;
static const int BILINEAR_METHOD = 2;
static const int BICUBIC_METHOD = 3;
static const int NEAREST_METHOD = 4;
static const int AKIMA_METHOD = 5;
static const int BIQUINTIC_METHOD = 6;

// Sinc-specific interpolation
static const int SINC_HALF = 4;
static const int SINC_LEN = 8;
static const int SINC_ONE = 9;
static const int SINC_SUB = 8192;

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
