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


#ifndef linalg3_h
#define linalg3_h

#include <stdlib.h>
#include <stdio.h>

void cross_C(double r_u[3], double r_v[3], double r_w[3]);
double dot_C(double r_v[3], double r_w[3]);
double lincomb_C(double k1, double u[3], double k2, double v[3], double w[3]);
double norm_C(double a[3]);
void unitvec_C(double v[3], double u[3]);

//Defined for Fortran
void matmat_F(double a[3][3], double b[3][3], double c[3][3]);
void matvec_F(double a[3][3], double b[3], double c[3]);
void tranmat_F(double a[3][3], double b[3][3]);

//Defined for C
void matmat_C(double a[3][3], double b[3][3], double c[3][3]);
void matvec_C(double a[3][3], double b[3], double c[3]);
void tranmat_C(double a[3][3], double b[3][3]);
#endif //linalg3_h
