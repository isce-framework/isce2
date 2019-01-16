/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Piyush Agram
# Copyright 2014, by the California Institute of Technology. ALL RIGHTS RESERVED.
# United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer at
# the California Institute of Technology.
# This software may be subject to U.S. export control laws.
# By accepting this software, the user agrees to comply with all applicable U.S.
# export laws and regulations. User has the responsibility to obtain export licenses,
# or other export authority as may be required before exporting such information to
# foreign countries or providing access to foreign persons.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


#include "watermask.h"
#include <fstream>
#include <iostream>

Polygon::Polygon()
{
    npoints = 0;
    xy = NULL;
}

//Allocate memory
void Polygon::allocate(int n)
{
    npoints = n;
    xy = new double*[n];
    for(int i=0; i<n; i++)
        xy[i] = new double[2];
}

Polygon::~Polygon()
{
    if (npoints > 0)
    {
        //Deallocate memory for polygon
        for(int i=0; i<npoints; i++)
            delete [] xy[i];


        delete [] xy;
    }
}

void Polygon::print()
{
    std::cout << "Polygon vertices : " << npoints << std::endl;
    for(int i=0; i<npoints; i++)
        std::cout << xy[i][0] << "    " << xy[i][1] << std::endl;
}

//Setup polygon data
void Polygon::setPoint(int i, double x, double y)
{
   xy[i][0] = x;
   xy[i][1] = y;
}

//Check if test point is within the given polygon
int Polygon::isInside(double testx, double testy)
{
    int i, j, c = 0;
    double xi, yi;
    double xj, yj;
    
    for (i = 0, j = npoints-1; i < npoints; j = i++)
    {
        xi = xy[i][0]; yi = xy[i][1];
        xj = xy[j][0]; yj = xy[j][1];

        if ( ((yi >testy) != (yj >testy)) &&
         (testx < (xj-xi) * (testy-yi) / (yj-yi) + xi) )
            c = 1-c;
    }
//    std::cout << testx << " " << testy << " " << c << std::endl; 
    return c;
}

//Allocate memory
WaterBody::WaterBody(int n)
{
    nshapes = n;
    shapes = new Polygon[n];
}

//Clear memory
WaterBody::~WaterBody()
{
    delete [] shapes;
}

//Check if given point is in water
int WaterBody::isWater(double x, double y)
{
    for(int i=0; i<nshapes;i++)
        if(shapes[i].isInside(x,y))
            return 1;

    return 0;
}

//Allocate memory
void WaterBody::allocate(int ind, int n)
{
    shapes[ind].allocate(n);
}

//Set data point for a polygon
void WaterBody::setShapeData(int ind, int i, double x, double y)
{
    shapes[ind].setPoint(i,x,y);
}

void WaterBody::printShape(int i)
{
    shapes[i].print();
}

//Set size of regular grid
void WaterBody::setDimensions(int ww, int ll)
{
    width = ww;
    height = ll;
}

void WaterBody::setTopLeft(double xx, double yy)
{
    x0 = xx;
    y0 = yy;
}

void WaterBody::setSpacing(double ddx, double ddy)
{
    dx = ddx;
    dy = ddy;
}

void WaterBody::fillGrid(char* filename)
{

    int i,j;
    double xx, yy;

    short * line;
    line = new short [width];

    std::ofstream maskfile(filename, std::ios::out | std::ios::binary);

//    std::cout << "Top Left: " << x0 << "  " << y0 << std::endl;
//    std::cout << "Spacing : " << dx << "  " << dy << std::endl;
//    std::cout << "Dims    : " << width << "  " << height << std::endl;
//    std::cout << "Int size: " << sizeof(short) << std::endl;

    for(i=0; i< height;i++)
    {
        yy = y0 + i*dy;
        if((i+1)%200 == 0)
            std::cout << "Line :" << i+1 << std::endl;

        for(j=0; j< width; j++)
        {
            xx = x0 + j*dx;
            line[j] = 1 - isWater(xx,yy);
//            std::cout << " " << xx << " " << yy << "  " << line[j] << std::endl;
        }
        maskfile.write(reinterpret_cast<const char*>(line), width*sizeof(short));
    }
    delete [] line;
    maskfile.close();
}


void WaterBody::makemask(char* lonfile, char* latfile, char* outfile)
{

    int i,j;
    double xx, yy;

    short * line;
    line = new short [width];

    float *lat;
    float *lon;
    lat = new float[width];
    lon = new float[width];

    std::ofstream maskfile(outfile, std::ios::out | std::ios::binary);
    std::ifstream lonf(lonfile, std::ios::in | std::ios::binary);
    std::ifstream latf(lonfile, std::ios::in | std::ios::binary);

//    std::cout << "Dims    : " << width << "  " << height << std::endl;

    for(i=0; i< height;i++)
    {
        lonf.read((char*)(&lon[0]), sizeof(float)*width);
        latf.read((char*)(&lat[0]), sizeof(float)*width);

        if((i+1)%200 == 0)
            std::cout << "Line :" << i+1 << std::endl;

        for(j=0; j< width; j++)
        {
            xx = lon[j];
            yy = lat[j];
            line[j] = 1 - isWater(xx,yy);
//            std::cout << " " << xx << " " << yy << "  " << line[j] << std::endl;
        }
        maskfile.write(reinterpret_cast<const char*>(line), width*sizeof(short));
    }
    delete [] lat;
    delete [] lon;
    delete [] line;
    maskfile.close();
    lonf.close();
    latf.close();
}


