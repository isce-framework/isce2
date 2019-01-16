//#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//# Author: Piyush Agram
//# Copyright 2013, by the California Institute of Technology. ALL RIGHTS RESERVED.
//# United States Government Sponsorship acknowledged.
//# Any commercial use must be negotiated with the Office of Technology Transfer at
//# the California Institute of Technology.
//# This software may be subject to U.S. export control laws.
//# By accepting this software, the user agrees to comply with all applicable U.S.
//# export laws and regulations. User has the responsibility to obtain export licenses,
//# or other export authority as may be required before exporting such information to
//# foreign countries or providing access to foreign persons.
//#
//#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#ifndef poly2d_h
#define poly2d_h

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#ifdef __cplusplus
extern "C"
{
#endif
typedef struct cPoly2d
{
    int rangeOrder;      //Python range order
    int azimuthOrder;    //Python azimuth order
    double meanRange;    //Python mean range
    double meanAzimuth;  //Python mean azimuth
    double normRange;    //Python norm range
    double normAzimuth;  //Python norm azimuth
    double *coeffs;      //Python coeffs in row major order
} cPoly2d;


//To evaluate polynomial
double evalPoly2d(cPoly2d* poly, double azi, double rng);

//To modify the reference point for polynomial
void modifyMean2d(cPoly2d* src, cPoly2d* targ, double azioff, double rngoff);

//To modify the scaling factors for polynomial
void modifyNorm2d(cPoly2d* src, cPoly2d* targ, double azinorm, double rngnorm);

//Modify one polynomial to that of another order
void scalePoly2d(cPoly2d* src, cPoly2d* targ, double minaz, double maxaz, double minrg, double maxrg);

//Get / Set
void setCoeff2d(cPoly2d* src, int i, int j, double value);
double getCoeff2d(cPoly2d* src, int i, int j);

//Basis for polynomial fitting
void getBasis2d(cPoly2d* src, double azi, double rng, int* indices, double* values, int len);

//Create/Destroy
cPoly2d* createPoly2d(int azOrder, int rgOrder);
void initPoly2d(cPoly2d* poly, int azOrder, int rgOrder);
void deletePoly2d(cPoly2d *src);
void cleanPoly2d(cPoly2d *src);

//Print for debugging
void printPoly2d(cPoly2d* poly);
#ifdef __cplusplus
}
#endif
#endif //poly2d_h
