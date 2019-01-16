/*#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*# Author: Piyush Agram*/
/*# Copyright 2014, by the California Institute of Technology. ALL RIGHTS RESERVED.*/
/*# United States Government Sponsorship acknowledged.*/
/*# Any commercial use must be negotiated with the Office of Technology Transfer at*/
/*# the California Institute of Technology.*/
/*# This software may be subject to U.S. export control laws.*/
/*# By accepting this software, the user agrees to comply with all applicable U.S.*/
/*# export laws and regulations. User has the responsibility to obtain export licenses,*/
/*# or other export authority as may be required before exporting such information to*/
/*# foreign countries or providing access to foreign persons.*/
/*#*/
/*#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#include "linalg3.h"

void matmat_C(double a[3][3], double b[3][3], double c[3][3])
{
    int i;

    for(i=0; i<3; i++)
    {
        c[i][0] = a[i][0]*b[0][0] + a[i][1]*b[1][0] + a[i][2]*b[2][0];
        c[i][1] = a[i][0]*b[0][1] + a[i][1]*b[1][1] + a[i][2]*b[2][1];
        c[i][2] = a[i][0]*b[0][2] + a[i][1]*b[1][2] + a[i][2]*b[2][2];
    }
}

void matvec_C(double a[3][3], double b[3], double c[3])
{
    c[0] = a[0][0]*b[0] + a[0][1]*b[1] + a[0][2]*b[2];
    c[1] = a[1][0]*b[0] + a[1][1]*b[1] + a[1][2]*b[2];
    c[2] = a[2][0]*b[0] + a[2][1]*b[1] + a[2][2]*b[2];
}


void tranmat_C(double a[3][3], double b[3][3])
{
    b[0][0]=a[0][0]; b[0][1]=a[1][0]; b[0][2]=a[2][0];
    b[1][0]=a[0][1]; b[1][1]=a[1][1]; b[1][2]=a[2][1];
    b[2][0]=a[0][2]; b[2][1]=a[1][2]; b[2][2]=a[2][2];
}

