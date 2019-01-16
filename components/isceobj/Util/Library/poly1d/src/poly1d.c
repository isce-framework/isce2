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



#include "poly1d.h"


//Create a polynomial object
cPoly1d* createPoly1d(int order)
{
    cPoly1d* newObj = (cPoly1d*) malloc(sizeof(cPoly1d));
    if(newObj == NULL)
    {
        printf("Not enough memory for polynomial object");
    }

    initPoly1d(newObj, order);
    return newObj;
}

//Initialize polynomial
void initPoly1d(cPoly1d* poly, int order)
{
    poly->coeffs = (double*) malloc(sizeof(double)*(order+1));
    if (poly->coeffs == NULL)
    {
        printf( "Not enough memory for polynomial object of order %d \n", order);
    }
    poly->order = order;
    //Currently only these.
    poly->mean = 0.0;
    poly->norm = 1.0;
}

//Delete polynomial object
void deletePoly1d(cPoly1d* obj)
{
    cleanPoly1d(obj);
    free((char*) obj);
}

//Clean up polynomial memory
void cleanPoly1d(cPoly1d* obj)
{
    free((char*) obj->coeffs);
}

//Set polynomial coefficient
void setCoeff1d(cPoly1d *src, int i, double value)
{
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
    src->coeffs[i] = value; 
}

//To get the coefficient
double getCoeff1d(cPoly1d *src, int i)
{
    double value;
    value = src->coeffs[i];
    return value;
}


//Evaluate the polynomial
double evalPoly1d(cPoly1d* poly, double xin)
{
    int i;
    double value = 0.0;
    double scalex;
    double xval;

    xval = (xin - poly->mean)/(poly->norm);

    scalex = 1.0;
    for(i = 0; i <= poly->order; i++,scalex*=xval)
    {

        value += scalex*getCoeff1d(poly,i);
    }

    return value;
}

//Setup Basis
void getBasis1d(cPoly1d* src, double xin, int* indices, double* values, int len)
{
   int i, j, ind;
   double xval;
   double val=1.0;

   xval = (xin - src->mean)/(src->norm);

   j = 0;
   ind = indices[0];
   for(i=0;i <= src->order;i++, val*=xval)
   {
       if (ind == i)
       {
           values[j] = val;
           ind = indices[++j];
       }
   }
}
//Print polynomial for debugging
void printPoly1d(cPoly1d* poly)
{
    int i;
    printf("Polynomial Order: %d \n", poly->order);

    for(i=0; i<= (poly->order); i++)
    {
            printf("%f\t", getCoeff1d(poly, i));
    }
        printf("\n");
}

//Modify the reference for the polynomial
//To be added
void modifyMean1d(cPoly1d* src, cPoly1d *targ, double off)
{
    return;
}

//Modify the scaling factors for the polynomial
//To be added
void modifyNorm1d(cPoly1d* src, cPoly1d *targ, double norm)
{
    double fact, val;
    double ratio;
    int i;
    if(src->order > targ->order)
    {
        printf("Orders of source and target are not compatible.");
    }

    ratio = src->norm / norm;

    fact = 1.0;
    for(i=0; i<src->order; i++, fact*=ratio)
    {
        val = getCoeff1d(src, i);
        setCoeff1d(targ, i, val*fact);
    }
    targ->norm = norm;

    for(i=src->order+1; i<=targ->order; i++)
    {
        setCoeff1d(targ, i, 0.0);
    }
}


//Scale polynomial from one order to another
//To be added
void scalePoly1d(cPoly1d *src, cPoly1d* targ, double minx, double maxx)
{
    return;
}



