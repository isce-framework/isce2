//
// Author: Joshua Cohen
// Copyright 2017
//

#include <math.h>
#include <stdio.h>
#include "isceLibConstants.h"
#include "Ellipsoid.h"
#include "LinAlg.h"
#include "Peg.h"
#include "Pegtrans.h"
using isceLib::Ellipsoid;
using isceLib::LinAlg;
using isceLib::Peg;
using isceLib::Pegtrans;
using isceLib::SCH_2_XYZ;
using isceLib::XYZ_2_SCH;
using isceLib::XYZ_2_LLH;
using isceLib::LLH_2_XYZ;

Pegtrans::Pegtrans() {
    // Empty constructor
    
    return;
}

Pegtrans::Pegtrans(const Pegtrans &p) {
    // Copy constructor
    
    for (int i=0; i<3; i++) {
        ov[i] = p.ov[i];
        for (int j=0; j<3; j++) {
            mat[i][j] = p.mat[i][j];
            matinv[i][j] = p.matinv[i][j];
        }
    }
    radcur = p.radcur;
}

void Pegtrans::radarToXYZ(Ellipsoid &elp, Peg &peg) {
    /* 
     * Computes the transformation matrix and translation vector needed to convert
     * between radar (s,c,h) coordinates and WGS-84 (x,y,z) coordinates
    */
    
    double llh[3], p[3], up[3];

    mat[0][0] = cos(peg.lat) * cos(peg.lon);
    mat[0][1] = -(sin(peg.hdg) * sin(peg.lon)) - (sin(peg.lat) * cos(peg.lon) * cos(peg.hdg));
    mat[0][2] = (sin(peg.lon) * cos(peg.hdg)) - (sin(peg.lat) * cos(peg.lon) * sin(peg.hdg));
    mat[1][0] = cos(peg.lat) * sin(peg.lon);
    mat[1][1] = (cos(peg.lon) * sin(peg.hdg)) - (sin(peg.lat) * sin(peg.lon) * cos(peg.hdg));
    mat[1][2] = -(cos(peg.lon) * cos(peg.hdg)) - (sin(peg.lat) * sin(peg.lon) * sin(peg.hdg));
    mat[2][0] = sin(peg.lat);
    mat[2][1] = cos(peg.lat) * cos(peg.hdg);
    mat[2][2] = cos(peg.lat) * sin(peg.hdg);
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            matinv[i][j] = mat[j][i];
        }
    }
    radcur = elp.rDir(peg.hdg, peg.lat);
    llh[0] = peg.lat;
    llh[1] = peg.lon;
    llh[2] = 0.;
    elp.latLon(p, llh, LLH_2_XYZ);
    up[0] = cos(peg.lat) * cos(peg.lon);
    up[1] = cos(peg.lat) * sin(peg.lon);
    up[2] = sin(peg.lat);
    for (int i=0; i<3; i++) ov[i] = p[i] - (radcur * up[i]);
}

void Pegtrans::convertSCHtoXYZ(double schv[3], double xyzv[3], int ctype) {
    /*
     * Applies the affine matrix provided to convert from the radar sch coordinates
     * to WGS-84 xyz coordinates or vice-versa
    */
    
    double schvt[3], llh[3];
    Ellipsoid sph;
    LinAlg alg;
    
    sph.a = radcur;
    sph.e2 = 0.;
    if (ctype == SCH_2_XYZ) {
        llh[0] = schv[1] / radcur;
        llh[1] = schv[0] / radcur;
        llh[2] = schv[2];
        sph.latLon(schvt, llh, LLH_2_XYZ);
        alg.matVec(mat, schvt, xyzv);
        alg.linComb(1., xyzv, 1., ov, xyzv);
    } else if (ctype == XYZ_2_SCH) {
        alg.linComb(1., xyzv, -1., ov, schvt);
        alg.matVec(matinv, schvt, schv);
        sph.latLon(schv, llh, XYZ_2_LLH);
        schv[0] = radcur * llh[1];
        schv[1] = radcur * llh[0];
        schv[2] = llh[2];
    } else {
        printf("Error: Unrecognized conversion type in Pegtrans::convertSCHtoXYZ (received %d).\n", ctype);
    }
}

void Pegtrans::convertSCHdotToXYZdot(double sch[3], double xyz[3], double schdot[3], double xyzdot[3], int ctype) {
    /*
     * Applies the affine matrix provided to convert from the radar sch velocity
     * to WGS-84 xyz velocity or vice-versa
    */
    
    double schxyzmat[3][3], xyzschmat[3][3];
    LinAlg alg;

    SCHbasis(sch, xyzschmat, schxyzmat);
    if (ctype == SCH_2_XYZ) alg.matVec(schxyzmat, schdot, xyzdot);
    else if (ctype == XYZ_2_SCH) alg.matVec(xyzschmat, xyzdot, schdot);
    else printf("Error: Unrecognized conversion type in Pegtrans::convertSCHdotToXYZdot (received %d).\n", ctype);
}

void Pegtrans::SCHbasis(double sch[3], double xyzschmat[3][3], double schxyzmat[3][3]) {
    // Computes the transformation matrix from xyz to a local sch frame
    
    double matschxyzp[3][3];
    LinAlg alg;
    
    matschxyzp[0][0] = -sin(sch[0] / radcur);
    matschxyzp[0][1] = -(sin(sch[1] / radcur) * cos(sch[0] / radcur));
    matschxyzp[0][2] = cos(sch[0] / radcur) * cos(sch[1] / radcur);
    matschxyzp[1][0] = cos(sch[0] / radcur);
    matschxyzp[1][1] = -(sin(sch[1] / radcur) * sin(sch[0] / radcur));
    matschxyzp[1][2] = sin(sch[0] / radcur) * cos(sch[1] / radcur);
    matschxyzp[2][0] = 0.;
    matschxyzp[2][1] = cos(sch[1] / radcur);
    matschxyzp[2][2] = sin(sch[1] / radcur);
    alg.matMat(mat, matschxyzp, schxyzmat);
    alg.tranMat(schxyzmat, xyzschmat);
}
