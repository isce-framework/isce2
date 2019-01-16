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
#include "enu2los.h"
#include "DataAccessor.h"

using namespace std;

void enu2los::setGeoDims(int wid, int len)
{
    geoWidth = wid;
    geoLength = len;
}

void enu2los::setDims(int wid, int len)
{
    width = wid;
    length = len;
}

void enu2los::setWavelength(float wvl)
{
    wavelength = wvl;
}

void enu2los::setScaleFactor(float scale)
{
    scaleFactor = scale;
}

void enu2los::setLatitudeInfo(float firstLat, float stepLat)
{
    startLatitude = firstLat;
    deltaLatitude = stepLat;
}

void enu2los::setLongitudeInfo(float firstLon, float stepLon)
{
    startLongitude = firstLon;
    deltaLongitude = stepLon;
}


void enu2los::process(uint64_t modelin, uint64_t latin, uint64_t lonin,
        uint64_t losin, uint64_t outin)
{

    float PI = atan(1.0)*4.0;
    float enu[3];  //To store the angles 
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
    float slk, clk;
    float saz, caz;
    int wid,geowid,geolen;
    float startLat,startLon;
    float deltaLat,deltaLon;

    float D2R = atan(1.0)/45.0;

    float *data=NULL;
    float *lat = NULL;
    float *lon = NULL;

//    print();
    if ((latin==0) || (lonin==0))
        data = new float[3*geoWidth];
    else
    {
        data = new float[3*geoWidth*geoLength];
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
        cout << "Input Data already appears to be in radar coordinates \n";

        for(i=0;i<length;i++)
        {
            k=i;
            losAcc->getLine((char*)los,k);

            k=i;
            modelAcc->getLine((char*)data,k);
          
            wid = width;

#pragma omp parallel for private(j,saz,caz,slk,clk)\
            shared(los,data,proj,D2R,wid)
            for(j=0; j<wid; j++)
            {
                caz = cos(D2R*los[2*j+1]); saz = sin(D2R*los[2*j+1]);
                clk = cos(D2R*los[2*j]); slk = sin(D2R*los[2*j]);

                proj[j] = MULT*(slk*saz*data[3*j] + slk*caz*data[3*j+1] + clk*data[3*j+2]);
            }

            k = i;
            outAcc->setLine((char*)proj,k);
        }
    }
    else
    {
        cout << "Input data in geocoded coordinates \n";
        
        //Read in the whole model as this involves interpolation
        for(i=0; i< geoLength; i++)
        {
            k = i;
            modelAcc -> getLine((char*) (data+3*i*geoWidth), k);
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

#pragma omp parallel for private(j,saz,caz,slk,clk,k) \
            private(iLat,iLon,frLat,frLon,enu) \
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

                caz = cos(D2R*los[2*j+1]); saz = sin(D2R*los[2*j+1]);
                clk = cos(D2R*los[2*j]); slk = sin(D2R*los[2*j]);


                //Bilinear interpolation
                for(k=0;k<3;k++)
                {
                    enu[k] = (1-frLon) * (1-frLat) * data[3*(iLat*geowid+iLon)+k] + 
                            frLon * (1-frLat) * data[3*(iLat*geowid+iLon+1)+k] +
                             (1-frLon) * frLat * data[3*((iLat+1)*geowid+iLon)+k] +
                             frLon * frLat * data[3*((iLat+1)*geowid+iLon+1)+k];

                }

                proj[j] = MULT*(slk*saz*enu[0] + slk*caz*enu[1] + clk*enu[2]);
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

void enu2los::print()
{
    cout << "**************************\n";
    cout << "Length:      " << length <<"\n";
    cout << "Width:       " << width << "\n";
    cout << "GeoLength:   " << geoLength << "\n";
    cout << "GeoWidth:    " << geoWidth << "\n";
    cout << "Scale :      " << scaleFactor << "\n";
    cout << "Wavelength:  " << wavelength << "\n";
    cout << "startLat:    " << startLatitude << "\n";
    cout << "deltaLat:    " << deltaLatitude << "\n";
    cout << "startLon:    " << startLongitude << "\n";
    cout << "deltaLon:    " << deltaLongitude << "\n";
}

