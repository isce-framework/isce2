//
// Author: Joshua Cohen
// Copyright 2016
//
// This code is based on older Fortran code, therefore the algorithms (especially array-accessors)
// are based on one-indexed arrays. Since some of the Fortran algorithms were adapted from
// languages that have zero-indexed arrays, it is easier and safer to simply modify the actual
// array-access calls as necessary (i.e. subtract 1 at the last possible moment)

#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdio>
#include "AkimaLib.h"
#include "Constants.h"
using std::max;
using std::min;
using std::vector;

bool AkimaLib::aki_almostEqual(double x, double y) {
    bool ret = (abs(x - y) <= AKI_EPS) ? true : false; // Compressed version is a little cleaner
    return ret;
}

void AkimaLib::printAkiNaN(int nx, int ny, vector<vector<float> > &ZZ, int ix, int iy, double slpx, double slpy, double slpxy) {
    int ii,jj;

    if (isnan(slpx) || isnan(slpy) || isnan(slpxy)) {
        printf("Slopes: %g %g %g\n", slpx, slpy, slpxy);
        printf("Location: %d %d\n", ix, iy);
        printf("Data:\n");
        for (int i=(iy-2); i<=(iy+2); i++) {
            ii = min(max(i,3),(ny-2));
            for (int j=(ix-2); j<=(ix+2); j++) {
                jj = min(max(j,3),(nx-2));
                printf("%g ",ZZ[jj-1][ii-1]);
            }
            printf("\n");
        }
    }
}

void AkimaLib::getParDer(int nx, int ny, vector<vector<float> > &ZZ, int ix, int iy, vector<vector<double> > &slpx, vector<vector<double> > &slpy, vector<vector<double> > &slpxy) {
    double m1,m2,m3,m4,wx2,wx3,wy2,wy3,d22,e22,d23,e23,d42,e32,d43,e33;
    int xx,yy;

    wx2 = wx3 = wy2 = wy3 = 0.0; // Avoid 'unused' warnings
    for (int ii=1; ii<=2; ii++) {
        yy = min(max((iy+ii),3),(ny-2)) - 1;
        for (int jj=1; jj<=2; jj++) {
            xx = min(max((ix+jj),3),(nx-2)) - 1;
            m1 = ZZ[(xx-1)][yy] - ZZ[(xx-2)][yy];
            m2 = ZZ[xx][yy] - ZZ[(xx-1)][yy];
            m3 = ZZ[(xx+1)][yy] - ZZ[xx][yy];
            m4 = ZZ[(xx+2)][yy] - ZZ[(xx+1)][yy];

            if (aki_almostEqual(m1,m2) && aki_almostEqual(m3,m4)) slpx[jj-1][ii-1] = 0.5 * (m2 + m3);
            else {
                wx2 = abs(m4 - m3);
                wx3 = abs(m2 - m1);
                slpx[jj-1][ii-1] = ((wx2 * m2) + (wx3 * m3)) / (wx2 + wx3);
            }

            m1 = ZZ[xx][(yy-1)] - ZZ[xx][(yy-2)];
            m2 = ZZ[xx][yy] - ZZ[xx][(yy-1)];
            m3 = ZZ[xx][(yy+1)] - ZZ[xx][yy];
            m4 = ZZ[xx][(yy+2)] - ZZ[xx][(yy+1)];

            if (aki_almostEqual(m1,m2) && aki_almostEqual(m3,m4)) slpx[jj-1][ii-1] = 0.5 * (m2 + m3);
            else {
                wy2 = abs(m4 - m3);
                wy3 = abs(m2 - m1);
                slpx[jj-1][ii-1] = ((wy2 * m2) + (wy3 * m3)) / (wy2 + wy3);
            }

            d22 = ZZ[(xx-1)][yy] - ZZ[(xx-1)][(yy-1)];
            d23 = ZZ[(xx-1)][(yy+1)] - ZZ[(xx-1)][yy];
            d42 = ZZ[(xx+1)][yy] - ZZ[(xx+1)][(yy-1)];
            d43 = ZZ[(xx+1)][(yy+1)] - ZZ[(xx+1)][yy];
            e22 = m2 - d22;
            e23 = m3 - d23;
            e32 = d42 - m2;
            e33 = d43 - m3;

            double dummyzero = 0.0;
            if (aki_almostEqual(wx2,dummyzero) && aki_almostEqual(wx3,dummyzero)) wx2 = wx3 = 1.0;
            if (aki_almostEqual(wy2,dummyzero) && aki_almostEqual(wy3,dummyzero)) wy2 = wy3 = 1.0;
            slpxy[jj-1][ii-1] = ((wx2 * ((wy2 * e22) + (wy3 * e23))) + (wx3 * ((wy2 * e32) + (wy3 * e33)))) /
                                    ((wx2 + wx3) * (wy2 + wy3));
        }
    }
}

void AkimaLib::polyfitAkima(int nx, int ny, vector<vector<float> > &ZZ, int ix, int iy, vector<double> &poly) {
    vector<vector<double> > sx(2,vector<double>(2)), sy(2,vector<double>(2)), sxy(2,vector<double>(2));
    vector<double> d(9);

    getParDer(nx,ny,ZZ,ix,iy,sx,sy,sxy);

    // Welp this'll be bad if they're all already zero-indexed...
    // See isceobj/Util/src/Akima_reg.F for original expanded version (this is somewhat compressed)
    d[0] = (ZZ[ix-1][iy-1] - ZZ[ix][iy-1]) + (ZZ[ix][iy] - ZZ[ix-1][iy]);
    d[1] = (sx[0][0] + sx[1][0]) - (sx[1][1] + sx[0][1]);
    d[2] = (sy[0][0] - sy[1][0]) - (sy[1][1] - sy[0][1]);
    d[3] = (sxy[0][0] + sxy[1][0]) + (sxy[1][1] + sxy[0][1]);
    d[4] = ((2 * sx[0][0]) + sx[1][0]) - (sx[1][1] + (2 * sx[0][1]));
    d[5] = (2 * (sy[0][0] - sy[1][0])) - (sy[1][1] - sy[0][1]);
    d[6] = (2 * (sxy[0][0] + sxy[1][0])) + (sxy[1][1] + sxy[0][1]);
    d[7] = ((2 * sxy[0][0]) + sxy[1][0]) + (sxy[1][1] + (2 * sxy[0][1]));
    d[8] = (2 * ((2 * sxy[0][0]) + sxy[1][0])) + (sxy[1][1] + (2 * sxy[0][1]));

    poly[0] = (2 * ((2 * d[0]) + d[1])) + ((2 * d[2]) + d[3]);
    poly[1] = -((3 * ((2 * d[0]) + d[1])) + ((2 * d[5]) + d[6]));
    poly[2] = (2 * (sy[0][0] - sy[1][0])) + (sxy[0][0] + sxy[1][0]);
    poly[3] = (2 * (ZZ[ix-1][iy-1] - ZZ[ix][iy-1])) + (sx[0][0] + sx[1][0]);
    poly[4] = -((2 * ((3 * d[0]) + d[4])) + ((3 * d[2]) + d[7]));
    poly[5] = (3 * ((3 * d[0]) + d[4])) + ((3 * d[5]) + d[8]);
    poly[6] = -((3 * (sy[0][0] - sy[1][0])) + ((2 * sxy[0][0]) + sxy[1][0]));
    poly[7] = -((3 * (ZZ[ix-1][iy-1] - ZZ[ix][iy-1])) + ((2 * sx[0][0]) + sx[1][0]));
    poly[8] = (2 * (sx[0][0] - sx[0][1])) + (sxy[0][0] + sxy[0][1]);
    poly[9] = -((3 * (sx[0][0] - sx[0][1])) + ((2 * sxy[0][0]) + sxy[0][1]));
    poly[10] = sxy[0][0];
    poly[11] = sx[0][0];
    poly[12] = (2 * (ZZ[ix-1][iy-1] - ZZ[ix-1][iy])) + (sy[0][0] + sy[0][1]);
    poly[13] = -((3 * (ZZ[ix-1][iy-1] - ZZ[ix-1][iy])) + ((2 * sy[0][0]) + sy[0][1]));
    poly[14] = sy[0][0];
    poly[15] = ZZ[ix-1][iy-1];
}

double AkimaLib::polyvalAkima(int ix, int iy, double xx, double yy, vector<double> &V) {
    double p1, p2, p3, p4, ret;

    p1 = (((((V[0] * (yy - iy)) + V[1]) * (yy - iy)) + V[2]) * (yy - iy)) + V[3];
    p2 = (((((V[4] * (yy - iy)) + V[5]) * (yy - iy)) + V[6]) * (yy - iy)) + V[7];
    p3 = (((((V[8] * (yy - iy)) + V[9]) * (yy - iy)) + V[10]) * (yy - iy)) + V[11];
    p4 = (((((V[12] * (yy - iy)) + V[13]) * (yy - iy)) + V[14]) * (yy - iy)) + V[15];
    ret = (((((p1 * (xx - ix)) + p2) * (xx - ix)) + p3) * (xx - ix)) + p4;
    return ret;
}

double AkimaLib::akima_intp(int nx, int ny, vector<vector<float> > &z, double x, double y) {
    vector<double> poly(AKI_NSYS);
    double ret;
    int xx,yy;

    xx = int(x);
    yy = int(y);
    polyfitAkima(nx,ny,z,xx,yy,poly);
    ret = polyvalAkima(xx,yy,x,y,poly);
    return ret;
}
