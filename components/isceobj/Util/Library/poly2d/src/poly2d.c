/*#!/usr/bin/env python*/
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



#include "poly2d.h"


//Create a polynomial object
cPoly2d* createPoly2d(int azOrder, int rgOrder)
{
    cPoly2d* newObj = (cPoly2d*) malloc(sizeof(cPoly2d));
    if(newObj == NULL)
    {
        printf("Not enough memory for polynomial object");
    }

    initPoly2d(newObj, azOrder, rgOrder);
    return newObj;
}

//Initialize polynomial
void initPoly2d(cPoly2d* poly, int azOrder, int rgOrder)
{
    poly->coeffs = (double*) malloc(sizeof(double)*(azOrder+1)*(rgOrder+1));
    if (poly->coeffs == NULL)
    {
        printf( "Not enough memory for polynomial object of order %d -by-  %d \n", azOrder, rgOrder);
    }
    poly->azimuthOrder = azOrder;
    poly->rangeOrder = rgOrder;
    //Currently only these.
    poly->meanRange = 0.0;
    poly->meanAzimuth = 0.0;
    poly->normRange = 1.0;
    poly->normAzimuth = 1.0;
}

//Delete polynomial object
void deletePoly2d(cPoly2d* obj)
{
    cleanPoly2d(obj);
    free((char*) obj);
}

//Clean up polynomial memory
void cleanPoly2d(cPoly2d* obj)
{
    free((char*) obj->coeffs);
}

//Set polynomial coefficient
void setCoeff2d(cPoly2d *src, int i, int j, double value)
{
    int index = i*(src->rangeOrder+1) + j;
/*    if (i >= src->azimuthOrder)
    {
        cout << "Index exceeds azimuth order bounds \n"; 
        exit(1);
    }
    if (j >= src->rangeOrder)
    {
        cout << "Index exceeds range order bounds \n";
        exit(1);
    } */
    src->coeffs[index] = value; 
}

//To get the coefficient
double getCoeff2d(cPoly2d *src, int i, int j)
{
    double value;
    int index = i*(src->rangeOrder+1) + j;
    value = src->coeffs[index];
    return value;
}


//Evaluate the polynomial
double evalPoly2d(cPoly2d* poly, double azi, double rng)
{
    int i,j;
    double value = 0.0;
    double scalex,scaley;
    double xval, yval;
    xval = (rng - poly->meanRange)/(poly->normRange);
    yval = (azi - poly->meanAzimuth)/(poly->normAzimuth);
    scaley = 1.0;
    for(i = 0; i <= poly->azimuthOrder; i++,scaley*=yval)
    {
        scalex = 1.0;
        for(j = 0; j <= poly->rangeOrder; j++,scalex*=xval)
        {
            value += scalex*scaley*getCoeff2d(poly,i,j);
//            printf("evalPoly2d %f %d %d %f %f\n",getCoeff2d(poly,i,j),i,j,azi,rng);
        }
    }
    return value;
}


//Setup Basis
void getBasis2d(cPoly2d* src, double azi, double rng, int* indices, double* values, int len)
{
    int i,j,k,ind, ind1;
    double xval, yval;
    double scalex,scaley;

    xval = (rng - src->meanRange)/(src->normRange);
    yval = (azi - src->meanAzimuth)/(src->normAzimuth);

    k = 0;
    ind = indices[0];

    scaley = 1.0;
    for(i=0; i<= src->azimuthOrder; i++, scaley*=yval)
    {
        scalex = scaley;
        for(j=0; j<= src->rangeOrder; j++, scalex *= xval)
        {
            ind1 = i*(src->rangeOrder+1)+j;
            if(ind1 == ind)
            {
                values[k] = scalex;
                ind = indices[++k];
            }
        }
    }
}



//Print polynomial for debugging
void printPoly2d(cPoly2d* poly)
{
    int i,j;
    printf("Polynomial Order: %d - by - %d \n", poly->azimuthOrder, poly->rangeOrder);

    for(i=0; i<= (poly->azimuthOrder); i++)
    {
        for(j=0; j<= (poly->rangeOrder); j++)
        {
            printf("%g\t", getCoeff2d(poly, i, j));
        }
        printf("\n");
    }
}

//Modify the reference for the polynomial
//To be added
void modifyMean(cPoly2d* src, cPoly2d *targ, double azioff, double rngoff)
{
    0;
}

//Modify the scaling factors for the polynomial
//To be added
void modifyNorm(cPoly2d* src, cPoly2d *targ, double azinorm, double rngnorm)
{
    double azfact, rgfact, val;
    double azratio, rgratio;
    int i,j;
    if(src->azimuthOrder > targ->azimuthOrder)
    {
        printf("Azimuth orders of source and target are not compatible.");
    }
    if(src->rangeOrder > src->azimuthOrder)
    {
        printf("Range orders of source and target are not compatible.");
    }

    azratio = src->normAzimuth / azinorm;
    rgratio = src->normRange / rngnorm;

    azfact = 1.0/azratio;
    for(i=0; i<src->azimuthOrder; i++)
    {
        azfact *= azratio;
        rgfact = 1.0/rgratio;
        for(j=0; j<src->rangeOrder; j++)
        {
            rgfact *= rgratio;
            val = getCoeff2d(src, i, j);
            setCoeff2d(targ, i, j, val*rgfact*azfact);
        }
    }
    targ->normAzimuth = azinorm;
    targ->normRange = rngnorm;

    for(i=src->azimuthOrder+1; i<=targ->azimuthOrder; i++)
    {
        for(j=src->rangeOrder+1; j<=targ->rangeOrder; j++)
        {
            setCoeff2d(targ, i, j, 0.0);
        }
    }
}


//Scale polynomial from one order to another with support
//To be added
void scalePoly2d(cPoly2d *src, cPoly2d* targ, double minaz, double maxaz, double minrg, double maxrg)
{
    0;
}



