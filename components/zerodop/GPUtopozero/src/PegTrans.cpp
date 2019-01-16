//
// Author: Joshua Cohen
// Copyright 2016
//

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "Constants.h"
#include "LinAlg.h"
#include "PegTrans.h"
using std::vector;

// Default constructor
PegTrans::PegTrans() :
    mat(3,vector<double>(3)),
    matinv(3,vector<double>(3)), 
    ov(3) {
    radcur = 0.0;
}

// Copy constructor
PegTrans::PegTrans(const PegTrans& pt) {
    mat = pt.mat;
    matinv = pt.matinv;
    ov = pt.ov;
    radcur = pt.radcur;
}

void PegTrans::convert_sch_to_xyz(vector<double> &schv, vector<double> &xyzv, int type) {
    vector<double> schvt(3), llh(3);
 
    Ellipsoid sph(radcur,0.0);
    LinAlg linalg;

    if (type == SCH_2_XYZ) {
        llh[0] = schv[1] / radcur;
        llh[1] = schv[0] / radcur;
        llh[2] = schv[2];

        sph.latlon(schvt,llh,LLH_2_XYZ);
        linalg.matvec(mat,schvt,xyzv);
        linalg.lincomb(1.0,xyzv,1.0,ov,xyzv);
    } else if (type == XYZ_2_SCH) {
        linalg.lincomb(1.0,xyzv,-1.0,ov,schvt);
        linalg.matvec(matinv,schvt,schv);
        sph.latlon(schv,llh,XYZ_2_LLH);
        schv[0] = radcur * llh[1];
        schv[1] = radcur * llh[0];
        schv[2] = llh[2];
    } else {
        printf("Error in PegTrans::convert_sch_to_xyz - Unknown method passed as type.\n");
        exit(1);
    }
}

void PegTrans::convert_schdot_to_xyzdot(vector<double> &sch, vector<double> &xyz, vector<double> &schdot, vector<double> &xyzdot, int type) {
    vector<vector<double> > schxyzmat(3,vector<double>(3)), xyzschmat(3,vector<double>(3));
    
    LinAlg linalg;
    schbasis(sch,xyzschmat,schxyzmat);

    if (type == SCH_2_XYZ) linalg.matvec(schxyzmat,schdot,xyzdot);
    else if (type == XYZ_2_SCH) linalg.matvec(xyzschmat,xyzdot,schdot);
    else {
        printf("Error in PegTrans::convert_schdot_to_xyzdot - Unknown method passed as type.\n");
        exit(1);
    }
}

void PegTrans::schbasis(vector<double> &sch, vector<vector<double> > &xyzschmat, vector<vector<double> > &schxyzmat) {
    vector<vector<double> > matschxyzp(3,vector<double>(3));
    double coss,cosc,sins,sinc;

    coss = cos(sch[0] / radcur);
    sins = sin(sch[0] / radcur);
    cosc = cos(sch[1] / radcur);
    sinc = sin(sch[1] / radcur);
    matschxyzp[0][0] = -sins;
    matschxyzp[0][1] = -sinc * coss;
    matschxyzp[0][2] = coss * cosc;
    matschxyzp[1][0] = coss;
    matschxyzp[1][1] = -sinc * sins;
    matschxyzp[1][2] = sins * cosc;
    matschxyzp[2][0] = 0.0;
    matschxyzp[2][1] = cosc;
    matschxyzp[2][2] = sinc;

    LinAlg linalg;
    linalg.matmat(mat,matschxyzp,schxyzmat);
    linalg.tranmat(schxyzmat,xyzschmat);
}

void PegTrans::radar_to_xyz(Ellipsoid &elp, Peg &peg) {
    vector<double> llh(3), p(3), up(3);
    double plat = peg.lat;
    double plon = peg.lon;
    double phdg = peg.hdg;

    mat[0][0] = cos(plat) * cos(plon);
    mat[0][1] = (-sin(phdg) * sin(plon)) - (sin(plat) * cos(plon) * cos(phdg));
    mat[0][2] = (sin(plon) * cos(phdg)) - (sin(plat) * cos(plon) * sin(phdg));
    mat[1][0] = cos(plat) * sin(plon);
    mat[1][1] = (cos(plon) * sin(phdg)) - (sin(plat) * sin(plon) * cos(phdg));
    mat[1][2] = (-cos(plon) * cos(phdg)) - (sin(plat) * sin(plon) * sin(phdg));
    mat[2][0] = sin(plat);
    mat[2][1] = cos(plat) * cos(phdg);
    mat[2][2] = cos(plat) * sin(phdg);

    for (int i=0; i<3; i++) for (int j=0; j<3; j++) matinv[i][j] = mat[j][i];

    radcur = elp.rdir(phdg,plat);

    llh[0] = plat;
    llh[1] = plon;
    llh[2] = 0.0;
    elp.latlon(p,llh,LLH_2_XYZ);

    up[0] = cos(plat) * cos(plon);
    up[1] = cos(plat) * sin(plon);
    up[2] = sin(plat);

    for (int i=0; i<3; i++) ov[i] = p[i] - (radcur * up[i]);
}

