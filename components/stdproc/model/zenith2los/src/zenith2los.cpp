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
#include "zenith2los.h"
#include "DataAccessor.h"

using namespace std;

void zenith2los::setGeoDims(int wid, int len)
{
    geoWidth = wid;
    geoLength = len;
}

void zenith2los::setDims(int wid, int len)
{
    width = wid;
    length = len;
}

void zenith2los::setWavelength(float wvl)
{
    wavelength = wvl;
}

void zenith2los::setScaleFactor(float scale)
{
    scaleFactor = scale;
}

void zenith2los::setLatitudeInfo(float firstLat, float stepLat)
{
    startLatitude = firstLat;
    deltaLatitude = stepLat;
}

void zenith2los::setLongitudeInfo(float firstLon, float stepLon)
{
    startLongitude = firstLon;
    deltaLongitude = stepLon;
}


void zenith2los::process(uint64_t modelin, uint64_t latin, uint64_t lonin,
        uint64_t losin, uint64_t outin)
{

    float PI = atan(1.0)*4.0;
    DataAccessor* modelAcc = (DataAccessor*) modelin;

    DataAccessor* latAcc = NULL;
    if(latin != 0)
    {
        latAcc = (DataAccessor*) latin;
    }

    DataAccessor* lonAcc = NULL;
    if(lonin !=0)
        lonAcc = (DataAccessor*) lonin;

    DataAccessor* losAcc = (DataAccessor*) losin;
    DataAccessor* outAcc  = (DataAccessor*) outin;

    //OpenMP variables
    float clk, intp;
    int wid,geowid,geolen;
    float startLat,startLon;
    float deltaLat,deltaLon;

    float D2R = atan(1.0)/45.0;

    float *data=NULL;
    float *lat = NULL;
    float *lon = NULL;
    if ((latin==0) || (lonin==0))
        data = new float[geoWidth];
    else
    {
        data = new float[geoWidth*geoLength];
        lat = new float[width];
        lon = new float[width];
    }
    
    float *los = new float[2*width];
    float *proj = new float[width];

    int i,j,k;

    //Pixel indexing
    int iLat,iLon;
    float frLat, frLon;
    float maxIndy, maxIndx;

    float zeroFloat = 0.0f;
    int zeroInt = 0;

    float MULT = scaleFactor*4.0*PI/wavelength;

    if ((latin==0) || (lonin==0))
    {
        for(i=0;i<length;i++)
        {
            k=i;
            losAcc->getLine((char*)los,k);

            k=i;
            modelAcc->getLine((char*)data,k);
          
            wid = width;
#pragma omp parallel for private(j,clk)\
            shared(los,data,proj,D2R,wid)
            for(j=0; j<wid; j++)
            {
                clk = cos(D2R*los[2*j]); 
                proj[j] = MULT*data[j]/clk;
            }

            k = i;
            outAcc->setLine((char*)proj,k);
        }
    }
    else
    {
        //Read in the whole model as this involves interpolation
        for(i=0; i< geoLength; i++)
        {
            k = i;
            modelAcc -> getLine((char*) (data+i*geoWidth), k);
        }

        for(i=0; i<length;i++)
        {
            k=i;
            losAcc->getLine((char*)los,k);
            k=i;
            latAcc->getLine((char*)lat,k);
            k=i;
            lonAcc->getLine((char*)lon,k);

            wid = width;
            geowid = geoWidth;
            geolen = geoLength;
            startLat = startLatitude;
            startLon = startLongitude;
            deltaLat = deltaLatitude;
            deltaLon = deltaLongitude;
            maxIndy = geolen-1.0;
            maxIndx = geowid-1.0;
#pragma omp parallel for private(j,clk,k) \
            private(iLat,iLon,frLat,frLon,intp) \
            shared(los,data,proj,D2R,lat,lon)\
            shared(startLat,deltaLat,maxIndy)\
            shared(startLon,deltaLon,maxIndx)\
            shared(geolen,geowid,wid)\
            shared(zeroInt,zeroFloat)
            for(j=0; j<wid;j++)
            {
                frLat = (lat[j] - startLat)/deltaLat;
                frLat = max(zeroFloat, min(frLat,maxIndy));
                iLat = (int) floor(frLat);
                iLat = max(zeroInt, min(iLat,geolen-2));
                frLat -= iLat;

                frLon = (lon[j] - startLon)/deltaLon;
                frLon = max(zeroFloat, min(frLon,maxIndx));
                iLon = (int) floor(frLon);
                iLon = max(zeroInt, min(iLon,geowid-2));
                frLon -= iLon;

                clk = cos(D2R*los[2*j]);

                intp = (1-frLon) * (1-frLat) * data[iLat*geowid+iLon] + 
                            frLon * (1-frLat) * data[(iLat+1)*geowid+iLon] +
                             (1-frLon) * frLat * data[iLat*geowid+iLon+1] +
                             frLon * frLat * data[(iLat+1)*geowid+iLon+1];

                proj[j] = MULT*intp/clk;
            }


            k=i;
            outAcc->setLine((char*) proj, k);
        }

        delete [] lat;
        delete [] lon;
    }

    delete [] data;
    delete [] proj;
    delete [] los;
    modelAcc = NULL;
    outAcc = NULL;
    losAcc = NULL;
    lonAcc = NULL;
    latAcc = NULL;
}

