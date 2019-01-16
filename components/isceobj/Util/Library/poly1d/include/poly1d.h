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



#ifndef poly1d_h
#define poly1d_h

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#ifdef __cplusplus
extern "C"
{
#endif
typedef struct cPoly1d
{
    int order;      //Python range order
    double mean;    //Python mean
    double norm;  //Python norm 
    double *coeffs;      //Python coeffs in row major order
} cPoly1d;


//To evaluate polynomial
double evalPoly1d(cPoly1d* poly, double x);

//To modify the reference point for polynomial
void modifyMean1d(cPoly1d* src, cPoly1d* targ, double off);

//To modify the scaling factors for polynomial
void modifyNorm1d(cPoly1d* src, cPoly1d* targ, double norm);

//Modify one polynomial to that of another order
void scalePoly1d(cPoly1d* src, cPoly1d* targ, double minx, double maxx);

//Get / Set 
void setCoeff1d(cPoly1d* src, int i, double value);
double getCoeff1d(cPoly1d* src, int i);


//Basis for polynomial fitting
void getBasis1d(cPoly1d *src, double xin, int* indices, double* values, int len);

//Create/Destroy
cPoly1d* createPoly1d(int order);
void initPoly1d(cPoly1d* poly, int order);
void deletePoly1d(cPoly1d *src);
void cleanPoly1d(cPoly1d *src);

//Print for debugging
void printPoly1d(cPoly1d* poly);
#ifdef __cplusplus
}
#endif
#endif  
