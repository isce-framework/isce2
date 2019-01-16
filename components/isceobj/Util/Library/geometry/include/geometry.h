//#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//#
//#
//# Author: Piyush Agram
//# Copyright 2014, by the California Institute of Technology. ALL RIGHTS RESERVED.
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


#ifndef geometry_h
#define geometry_h

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct cPeg
{
    double lat;
    double lon;
    double hdg;
} cPeg;

typedef struct cEllipsoid
{
    double a;
    double e2;
} cEllipsoid;

typedef struct cPegtrans
{
    double mat[3][3];
    double matinv[3][3];
    double ov[3];
    double radcur;
} cPegtrans;


typedef struct cPosition
{
    double j[3];
    double jdot[3];
    double jddt[3];
} cPosition;

//SCH to XYZ conversions
static const int SCH_2_XYZ = 0;
static const int XYZ_2_SCH = 1;

//Lat Lon to UTM conversions
static const int LLH_2_UTM = 1;
static const int UTM_2_LLH = 2;

//Lat Lon to XYZ conversions
static const int LLH_2_XYZ = 1;
static const int XYZ_2_LLH = 2;
static const int XYZ_2_LLH_OLD = 3;

//Function declarations
void convert_sch_to_xyz_C(cEllipsoid* ptm, double r_schv[3], double r_xyzv[3], int i_type);
void convert_schdot_to_xyzdot_C(cEllipsoid* ptm, double r_sch[3], double r_xyz[3], double r_schdot[3], double r_xyzdot[3], int i_type);
double reast_C(cEllipsoid* elp, double r_lat);
double rnorth_C(cEllipsoid* elp, double r_lat);
double rdir_C(cEllipsoid* elp,double r_hdg, double r_lat);
void enubasis_C(double r_lat, double r_lon, double r_enumat[3][3]);
void latlon_C(cEllipsoid* elp, double r_v[3], double r_llh[3], int i_type);
void lookvec_C(cPosition* pos, double r_look, double r_az, double r_v[3]);
void radar_to_xyz_C(cEllipsoid* elp, cPeg* peg, cPegtrans* ptm);
void schbasis_C(cPegtrans* ptm, double r_sch[3], double r_xyzschmat[3][3], double r_schxyzmat[3][3]);
void getangs_C(double pos[3], double vel[3], double vec[3], cEllipsoid* elp, double *r_az, double *r_lk);
void getTCN_TCvec_C(double pos[3], double vel[3], double vec[3], cEllipsoid* elp, double TCVec[3]);
double cosineC_C(double a, double b, double c);
#endif
