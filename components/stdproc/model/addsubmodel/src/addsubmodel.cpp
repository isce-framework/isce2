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
#include <iostream>
#include <cmath>
#include <complex>
#include "addsubmodel.h"
#include "DataAccessor.h"

using namespace std;

void addsubmodel::setDims(int wid, int len)
{
    width = wid;
    length = len;
}

void addsubmodel::setScaleFactor(float scale)
{
    scaleFactor = scale;
}

void addsubmodel::setFlip(int flag)
{
    flip = flag;
}

void addsubmodel::cpxCpxprocess(uint64_t input, uint64_t model, uint64_t out)
{

    DataAccessor* modelAcc = (DataAccessor*) model;
    DataAccessor* inAcc = (DataAccessor*) input;
    DataAccessor* outAcc = (DataAccessor*) out;

    int i,j,k;
    int wid;

    if (scaleFactor != 1.0)
    {
        cout << "scaleFactor is not used when both input and model are complex floats. \n";
    }

    complex <float> *data = new complex<float>[width];
    complex <float> *modarr = new complex<float>[width];

    for(i=0;i<length;i++)
    {
        k=i;
        modelAcc->getLine((char*)modarr,k);

        k=i;
        inAcc->getLine((char*)data,k);
          
        wid = width;
        
        if(flip !=0 )
        {
#pragma omp parallel for private(j) \
            shared(modarr,wid,data)
            for(j=0;j<wid;j++)
            {
                data[j] = data[j]*conj(modarr[j]);
            }
        }
        else
        {
#pragma omp parallel for private(j) \
            shared(modarr,data,wid)
            for(j=0; j<wid; j++)
            {
                data[j] = data[j]*modarr[j];
            }
        }

        k = i;
        outAcc->setLine((char*)data,k);
    }

    delete [] data;
    delete [] modarr;
    modelAcc = NULL;
    inAcc = NULL;
    outAcc = NULL;
}

void addsubmodel::unwUnwprocess(uint64_t input, uint64_t model, uint64_t out)
{

    DataAccessor* modelAcc = (DataAccessor*) model;
    DataAccessor* inAcc = (DataAccessor*) input;
    DataAccessor* outAcc = (DataAccessor*) out;

    int i,j,k;
    int wid;
    float mult;

    if (flip != 0)
    {
        mult = -scaleFactor;
    }
    else
    {
        mult = scaleFactor;
    }

    float *data = new float[width];
    float *modarr = new float[width];

    for(i=0;i<length;i++)
    {
        k=i;
        modelAcc->getLine((char*)modarr,k);

        k=i;
        inAcc->getLine((char*)data,k);
          
        wid = width;
        
#pragma omp parallel for private(j) \
            shared(modarr,data,wid,mult)
        for(j=0; j<wid; j++)
        {
            data[j] = data[j] + mult*modarr[j];
        }

        k = i;
        outAcc->setLine((char*)data,k);
    }

    delete [] data;
    delete [] modarr;
    modelAcc = NULL;
    inAcc = NULL;
    outAcc = NULL;
}

void addsubmodel::cpxUnwprocess(uint64_t input, uint64_t model, uint64_t out)
{
    DataAccessor* modelAcc = (DataAccessor*) model;
    DataAccessor* inAcc = (DataAccessor*) input;
    DataAccessor* outAcc = (DataAccessor*) out;

    //Complex J
    complex <float> cJ;
    cJ = (0.0,1.0);

    int i,j,k;
    int wid;
    float mult;

    if (flip != 0)
    {
        mult = -scaleFactor;
    }
    else
    {
        mult = scaleFactor;
    }

    complex <float> *data = new complex<float>[width];
    float *modarr = new float[width];

    for(i=0;i<length;i++)
    {
        k=i;
        modelAcc->getLine((char*)modarr,k);

        k=i;
        inAcc->getLine((char*)data,k);
          
        wid = width;
        
#pragma omp parallel for private(j) \
            shared(modarr,data,wid,mult,cJ)
        for(j=0; j<wid; j++)
        {
            data[j] = data[j] * exp(cJ*mult*modarr[j]);
        }

        k = i;
        outAcc->setLine((char*)data,k);
    }

    delete [] data;
    delete [] modarr;
    modelAcc = NULL;
    inAcc = NULL;
    outAcc = NULL;
}


void addsubmodel::print()
{
    cout << "**************************\n";
    cout << "Length:      " << length <<"\n";
    cout << "Width:       " << width << "\n";
    cout << "Scale :      " << scaleFactor << "\n";
    cout << "Flip :       " << flip << "\n";
}

