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



#ifndef orbit_h
#define orbit_h

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "geometry.h"


static const int WGS84_ORBIT = 1;
static const int SCH_ORBIT = 2;

typedef struct cOrbit
{
    int nVectors;       //Number of state vectors
    char yyyymmdd[256];     //Date string
    double *position;   //Double array for position
    double *velocity;   //Double array for velocity
    double *UTCtime;    //Double array for UTCtimes
    int basis ;         //Integer for basis
} cOrbit;


typedef struct cStateVector
{
    double time;            //UTC time in seconds
    double position[3];     //Position in meters
    double velocity[3];     //Velocity in meters / sec
} cStateVector;


//Create and Delete
cOrbit* createOrbit(int nvec, int basis);
void initOrbit(cOrbit* orb, int nvec, int basis);
void cleanOrbit(cOrbit *orb);
void deleteOrbit(cOrbit *orb);

//Get position and Velocity
void getPostionVelocity(cOrbit* orb, double tintp, double* pos, double* vel);
void getStateVector(cOrbit* orb, int index, double *t, double* pos, double *vel);
void setStateVector(cOrbit* orb, int index, double t, double* pos, double* vel);

//Interpolation for differnt types of orbits
int interpolateWGS84Orbit(cOrbit* orb, double tintp, double *pos, double* vel);
int interpolateLegendreOrbit(cOrbit* orb, double tintp, double *pos, double *vel);
int interpolateSCHOrbit(cOrbit* orb, double tintp, double *pos, double* vel);
int computeAcceleration(cOrbit* orb, double tintp, double *acc);

//Print for debugging
void printOrbit(cOrbit* orb);

cOrbit* loadFromHDR(const char* filename, int basis);
void dumpToHDR(cOrbit* orb, const char* filename);

#endif  
