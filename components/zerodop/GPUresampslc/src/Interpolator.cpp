//
// Author: Joshua Cohen
// Copyright 2017
//

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>
#include "Interpolator.h"
#include "Constants.h"
using std::complex;
using std::invalid_argument;
using std::max;
using std::min;
using std::string;
using std::to_string;
using std::vector;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

template<class U>
U Interpolator::bilinear(double x, double y, vector<vector<U>> &z) {

    int x1 = floor(x);
    int x2 = ceil(x);
    int y1 = ceil(y);
    int y2 = floor(y);
    auto q11 = z[y1-1][x1-1];
    auto q12 = z[y2-1][x1-1];
    auto q21 = z[y1-1][x2-1];
    auto q22 = z[y2-1][x2-1];

    if ((y1 == y2) && (x1 == x2)) return q11;
    else if (y1 == y2) return (static_cast<U>((x2 - x) / (x2 - x1)) * q11) + (static_cast<U>((x - x1) / (x2 - x1)) * q21);
    else if (x1 == x2) return (static_cast<U>((y2 - y) / (y2 - y1)) * q11) + (static_cast<U>((y - y1) / (y2 - y1)) * q12);
    else {
        return  ((q11 * static_cast<U>((x2 - x) * (y2 - y))) / static_cast<U>((x2 - x1) * (y2 - y1))) +
                ((q21 * static_cast<U>((x - x1) * (y2 - y))) / static_cast<U>((x2 - x1) * (y2 - y1))) +
                ((q12 * static_cast<U>((x2 - x) * (y - y1))) / static_cast<U>((x2 - x1) * (y2 - y1))) +
                ((q22 * static_cast<U>((x - x1) * (y - y1))) / static_cast<U>((x2 - x1) * (y2 - y1)));
    }
}

template complex<double> Interpolator::bilinear(double,double,vector<vector<complex<double>>>&);
template complex<float> Interpolator::bilinear(double,double,vector<vector<complex<float>>>&);
template double Interpolator::bilinear(double,double,vector<vector<double>>&);
template float Interpolator::bilinear(double,double,vector<vector<float>>&);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

template<class U>
U Interpolator::bicubic(double x, double y, vector<vector<U>> &z) {

    vector<vector<float>> wt = {{1.0, 0.0,-3.0, 2.0, 0.0, 0.0, 0.0, 0.0,-3.0, 0.0, 9.0,-6.0, 2.0, 0.0,-6.0, 4.0},
                                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0,-9.0, 6.0,-2.0, 0.0, 6.0,-4.0},
                                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0,-6.0, 0.0, 0.0,-6.0, 4.0},
                                {0.0, 0.0, 3.0,-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0- 9.0, 6.0, 0.0, 0.0, 6.0,-4.0},
                                {0.0, 0.0, 0.0, 0.0, 1.0, 0.0,-3.0, 2.0,-2.0, 0.0, 6.0,-4.0, 1.0, 0.0,-3.0, 2.0},
                                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-1.0, 0.0, 3.0,-2.0, 1.0, 0.0,-3.0, 2.0},
                                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-3.0, 2.0, 0.0, 0.0, 3.0,-2.0},
                                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0,-2.0, 0.0, 0.0,-6.0, 4.0, 0.0, 0.0, 3.0,-2.0},
                                {0.0, 1.0,-2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,-3.0, 6.0,-3.0, 0.0, 2.0,-4.0, 2.0},
                                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0,-6.0, 3.0, 0.0,-2.0, 4.0,-2.0},
                                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-3.0, 3.0, 0.0, 0.0, 2.0,-2.0},
                                {0.0, 0.0,-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0,-3.0, 0.0, 0.0,-2.0, 2.0},
                                {0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-2.0, 1.0, 0.0,-2.0, 4.0,-2.0, 0.0, 1.0,-2.0, 1.0},
                                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-1.0, 2.0,-1.0, 0.0, 1.0,-2.0, 1.0},
                                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-1.0, 0.0, 0.0,-1.0, 1.0},
                                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-1.0, 1.0, 0.0, 0.0, 2.0,-2.0, 0.0, 0.0,-1.0, 1.0}};

    int x1 = floor(x) - 1;
    int x2 = ceil(x) - 1;
    int y1 = floor(y) - 1;
    int y2 = ceil(y) - 1;

    vector<U> zz = {z[y1][x1], z[y1][x2], z[y2][x2], z[y2][x1]};
    vector<U> dzdx = {(z[y1][x1+1] - z[y1][x1-1]) / static_cast<U>(2.0), (z[y1][x2+1] - z[y1][x2-1]) / static_cast<U>(2.0),
                      (z[y2][x2+1] - z[y2][x2-1]) / static_cast<U>(2.0), (z[y2][x1+1] - z[y2][x1-1]) / static_cast<U>(2.0)};
    vector<U> dzdy = {(z[y1+1][x1] - z[y1-1][x1]) / static_cast<U>(2.0), (z[y1+1][x2+1] - z[y1-1][x2]) / static_cast<U>(2.0),
                      (z[y2+1][x2+1] - z[y2-1][x2]) / static_cast<U>(2.0), (z[y2+1][x1+1] - z[y2-1][x1]) / static_cast<U>(2.0)};
    vector<U> dzdxy = {static_cast<U>(.25)*(z[y1+1][x1+1] - z[y1-1][x1+1] - z[y1+1][x1-1] + z[y1-1][x1-1]),
                       static_cast<U>(.25)*(z[y1+1][x2+1] - z[y1-1][x2+1] - z[y1+1][x2-1] + z[y1-1][x2-1]),
                       static_cast<U>(.25)*(z[y2+1][x2+1] - z[y2-1][x2+1] - z[y2+1][x2-1] + z[y2-1][x2-1]),
                       static_cast<U>(.25)*(z[y2+1][x1+1] - z[y2-1][x1+1] - z[y2+1][x1-1] + z[y2-1][x1-1])};

    vector<U> q(16);
    for (int i=0; i<4; i++) {
        q[i] = zz[i];
        q[i+4] = dzdx[i];
        q[i+8] = dzdy[i];
        q[i+12] = dzdxy[i];
    }

    vector<U> c(16,0.);
    for (int i=0; i<16; i++) {
        for (int j=0; j<16; j++) {
            c[i] += static_cast<U>(wt[i][j]) * q[j];
        }
    }

    U t = x - x1;
    U u = y - y1;
    U ret = 0.;
    for (int i=3; i>=0; i--) ret = (t * ret) + c[IDX1D(i,0,4)] + (((((c[IDX1D(i,3,4)] * u) + c[IDX1D(i,2,4)]) * u) + c[IDX1D(i,1,4)]) * u);
    return ret;
}

template complex<double> Interpolator::bicubic(double,double,vector<vector<complex<double>>>&);
template complex<float> Interpolator::bicubic(double,double,vector<vector<complex<float>>>&);
template double Interpolator::bicubic(double,double,vector<vector<double>>&);
template float Interpolator::bicubic(double,double,vector<vector<float>>&);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

void Interpolator::sinc_coef(double beta, double relfiltlen, int decfactor, double pedestal, int weight,
    int &intplength, int &filtercoef, vector<double> &filter) {

    intplength = rint(relfiltlen / beta);
    filtercoef = intplength * decfactor;
    double wgthgt = (1. - pedestal) / 2.;
    double soff = filtercoef / 2.;

    double wgt, s, fct;
    for (int i=0; i<filtercoef; i++) {
        wgt = (1. - wgthgt) + wgthgt * cos(M_PI * (i - soff) / soff);
        s = (i - soff) * beta / (1. * decfactor);
        if (s != 0.) fct = sin(M_PI * s) / (M_PI * s);
        else fct = 1.;
        if (weight == 1) filter[i] = fct * wgt;
        else filter[i] = fct;
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

template<class U, class V>
U Interpolator::sinc_eval(vector<U> &arr, vector<V> &intarr, int idec, int ilen, int intp, double frp, int nsamp) {

    U ret = 0.;
    if ((intp >= (ilen-1)) && (intp < nsamp)) {
        int ifrc = min(max(0, int(frp*idec)), idec-1);
        for (int i=0; i<ilen; i++) ret += arr[intp-i] * static_cast<U>(intarr[IDX1D(ifrc,i,ilen)]);
    }
    return ret;
}

template complex<double> Interpolator::sinc_eval(vector<complex<double>>&,vector<double>&,int,int,int,double,int);
template complex<double> Interpolator::sinc_eval(vector<complex<double>>&,vector<float>&,int,int,int,double,int);
template complex<float> Interpolator::sinc_eval(vector<complex<float>>&,vector<double>&,int,int,int,double,int);
template complex<float> Interpolator::sinc_eval(vector<complex<float>>&,vector<float>&,int,int,int,double,int);
template double Interpolator::sinc_eval(vector<double>&,vector<double>&,int,int,int,double,int);
template double Interpolator::sinc_eval(vector<double>&,vector<float>&,int,int,int,double,int);
template float Interpolator::sinc_eval(vector<float>&,vector<double>&,int,int,int,double,int);
template float Interpolator::sinc_eval(vector<float>&,vector<float>&,int,int,int,double,int);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

template<class U, class V>
U Interpolator::sinc_eval_2d(vector<vector<U>> &arrin, vector<V> &intarr, int idec, int ilen, int intpx, int intpy, double frpx, double frpy, int xlen, int ylen) {

    U ret(0.);
    if ((intpx >= (ilen-1)) && (intpx < xlen) && (intpy >= (ilen-1)) && (intpy < ylen)) {
        int ifracx = min(max(0, int(frpx*idec)), idec-1);
        int ifracy = min(max(0, int(frpy*idec)), idec-1);
        for (int i=0; i<ilen; i++) {
            for (int j=0; j<ilen; j++) {
                ret += arrin[intpx-i][intpy-j] * static_cast<U>(intarr[IDX1D(ifracx,i,ilen)]) * static_cast<U>(intarr[IDX1D(ifracy,j,ilen)]);
            }
        }
    }
    return ret;
}

template complex<double> Interpolator::sinc_eval_2d(vector<vector<complex<double>>>&,vector<double>&,int,int,int,int,double,double,int,int);
template complex<double> Interpolator::sinc_eval_2d(vector<vector<complex<double>>>&,vector<float>&,int,int,int,int,double,double,int,int);
template complex<float> Interpolator::sinc_eval_2d(vector<vector<complex<float>>>&,vector<double>&,int,int,int,int,double,double,int,int);
template complex<float> Interpolator::sinc_eval_2d(vector<vector<complex<float>>>&,vector<float>&,int,int,int,int,double,double,int,int);
template double Interpolator::sinc_eval_2d(vector<vector<double>>&,vector<double>&,int,int,int,int,double,double,int,int);
template double Interpolator::sinc_eval_2d(vector<vector<double>>&,vector<float>&,int,int,int,int,double,double,int,int);
template float Interpolator::sinc_eval_2d(vector<vector<float>>&,vector<double>&,int,int,int,int,double,double,int,int);
template float Interpolator::sinc_eval_2d(vector<vector<float>>&,vector<float>&,int,int,int,int,double,double,int,int);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

float Interpolator::interp_2d_spline(int order, int nx, int ny, vector<vector<float>> &z, double x, double y) {

    // Error checking
    if ((order < 3) || (order > 20)) {
        string errstr = "Interpolator::interp_2d_spline - Spline order must be between 3 and 20 (received "+to_string(order)+")";
        throw invalid_argument(errstr);
    }

    int i0, j0;
    if ((order % 2) != 0) {
        i0 = y - .5;
        j0 = x - .5;
    } else {
        i0 = y;
        j0 = x;
    }
    i0 = i0 - (order / 2) + 1;
    j0 = j0 - (order / 2) + 1;

    vector<double> A(order), R(order), Q(order), HC(order);
    int indi, indj;
    for (int i=1; i<=order; i++) {
        indi = min(max((i0+i), 1), ny);
        for (int j=1; j<=order; j++) {
            indj = min(max((j0+j), 1), nx);
            A[j-1] = z[indi-1][indj-1];
        }
        initSpline(A,order,R,Q);
        HC[i-1] = spline((x-j0),A,order,R);
    }

    initSpline(HC,order,R,Q);
    return static_cast<float>(spline((y-i0),HC,order,R));
}

void initSpline(vector<double> &Y, int n, vector<double> &R, vector<double> &Q) {

    Q[0] = 0.;
    R[0] = 0.;
    for (int i=2; i<n; i++) {
        Q[i-1] = -.5 / ((Q[i-2] / 2.) + 2.);
        R[i-1] = ((3 * (Y[i] - (2 * Y[i-1]) + Y[i-2])) - (R[i-2] / 2)) / ((Q[i-2] / 2.) + 2.);
    }
    R[n-1] = 0.;
    for (int i=(n-1); i>=2; i--) R[i-1] = (Q[i-1] * R[i]) + R[i-1];
}

double spline(double x, vector<double> &Y, int n, vector<double> &R) {

    if (x < 1.) return Y[0] + ((x - 1.) * (Y[1] - Y[0] - (R[1] / 6.)));
    else if (x > n) return Y[n-1] + ((x - n) * (Y[n-1] - Y[n-2] + (R[n-2] / 6.)));
    else {
        int j = floor(x);
        auto xx = x - j;
        auto t0 = Y[j] - Y[j-1] - (R[j-1] / 3.) - (R[j] / 6.);
        auto t1 = xx * ((R[j-1] / 2.) + (xx * ((R[j] - R[j-1]) / 6)));
        return Y[j-1] + (xx * (t0 + t1));
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

double Interpolator::quadInterpolate(vector<double> &x, vector<double> &y, double xintp) {

    auto xin = xintp - x[0];
    vector<double> x1(3), y1(3);
    for (int i=0; i<3; i++) {
        x1[i] = x[i] - x[0];
        y1[i] = y[i] - y[0];
    }
    double a = ((-y1[1] * x1[2]) + (y1[2] * x1[1])) / ((-x1[2] * x1[1] * x1[1]) + (x1[1] * x1[2] * x1[2]));
    double b = (y1[1] - (a * x1[1] * x1[1])) / x1[1];
    return y[0] + (a * xin * xin) + (b * xin);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

double Interpolator::akima(int nx, int ny, vector<vector<float>> &z, double x, double y) {

    vector<vector<double>> sx(2,vector<double>(2)), sy(2,vector<double>(2)), sxy(2,vector<double>(2)), e(2,vector<double>(2));
    vector<double> m(4);
    double wx2,wx3,wy2,wy3;
    int ix = x;
    int iy = y;

    wx2 = wx3 = wy2 = wy3 = 0.;
    for (int ii=0; ii<2; ii++) {
        int xx = min(max((ix+ii+1),3),(nx-2)) - 1;
        for (int jj=0; jj<2; jj++) {
            int yy = min(max((iy+jj+1),3),(ny-2)) - 1;

            m[0] = z[xx-1][yy] - z[xx-2][yy];
            m[1] = z[xx][yy] - z[xx-1][yy];
            m[2] = z[xx+1][yy] - z[xx][yy];
            m[3] = z[xx+2][yy] - z[xx+1][yy];

            if ((abs(m[0] - m[1]) <= DBL_EPSILON) && (abs(m[2] - m[3]) <= DBL_EPSILON)) sx[ii][jj] = 0.5 * (m[1] + m[2]);
            else {
                wx2 = abs(m[3] - m[2]);
                wx3 = abs(m[1] - m[0]);
                sx[ii][jj] = ((wx2 * m[1]) + (wx3 * m[2])) / (wx2 + wx3);
            }

            m[0] = z[xx][yy-1] - z[xx][yy-2];
            m[1] = z[xx][yy] - z[xx][yy-1];
            m[2] = z[xx][yy+1] - z[xx][yy];
            m[3] = z[xx][yy+2] - z[xx][yy+1];

            if ((abs(m[0] - m[1]) <= DBL_EPSILON) && (abs(m[2] - m[3]) <= DBL_EPSILON)) sy[ii][jj] = 0.5 * (m[1] + m[2]);
            else {
                wy2 = abs(m[3] - m[2]);
                wy3 = abs(m[1] - m[0]);
                sy[ii][jj] = ((wy2 * m[1]) + (wy3 * m[2])) / (wy2 + wy3);
            }

            e[0][0] = m[1] - z[xx-1][yy] - z[xx-1][yy-1];
            e[0][1] = m[2] - z[xx-1][yy+1] - z[xx-1][yy];
            e[1][0] = z[xx+1][yy] - z[xx+1][yy-1] - m[1];
            e[1][1] = z[xx+1][yy+1] - z[xx+1][yy] - m[2];

            if ((abs(wx2) <= DBL_EPSILON) && (abs(wx3) <= DBL_EPSILON)) wx2 = wx3 = 1.;
            if ((abs(wy2) <= DBL_EPSILON) && (abs(wy3) <= DBL_EPSILON)) wy2 = wy3 = 1.;
            sxy[ii][jj] = ((wx2 * ((wy2 * e[0][0]) + (wy3 * e[0][1]))) + (wx3 * ((wy2 * e[1][0]) + (wy3 * e[1][1])))) / ((wx2 + wx3) * (wy2 + wy3));
        }
    }

    vector<double> d(9);
    d[0] = (z[ix-1][iy-1] - z[ix][iy-1]) + (z[ix][iy] - z[ix-1][iy]);
    d[1] = (sx[0][0] + sx[1][0]) - (sx[1][1] + sx[0][1]);
    d[2] = (sy[0][0] - sy[1][0]) - (sy[1][1] - sy[0][1]);
    d[3] = (sxy[0][0] + sxy[1][0]) + (sxy[1][1] + sxy[0][1]);
    d[4] = ((2 * sx[0][0]) + sx[1][0]) - (sx[1][1] + (2 * sx[0][1]));
    d[5] = (2 * (sy[0][0] - sy[1][0])) - (sy[1][1] - sy[0][1]);
    d[6] = (2 * (sxy[0][0] + sxy[1][0])) + (sxy[1][1] + sxy[0][1]);
    d[7] = ((2 * sxy[0][0]) + sxy[1][0]) + (sxy[1][1] + (2 * sxy[0][1]));
    d[8] = (2 * ((2 * sxy[0][0]) + sxy[1][0])) + (sxy[1][1] + (2 * sxy[0][1]));

    vector<double> poly(16);
    poly[0] = (2 * ((2 * d[0]) + d[1])) + ((2 * d[2]) + d[3]);
    poly[1] = -((3 * ((2 * d[0]) + d[1])) + ((2 * d[5]) + d[6]));
    poly[2] = (2 * (sy[0][0] - sy[1][0])) + (sxy[0][0] + sxy[1][0]);
    poly[3] = (2 * (z[ix-1][iy-1] - z[ix][iy-1])) + (sx[0][0] + sx[1][0]);
    poly[4] = -((2 * ((3 * d[0]) + d[4])) + ((3 * d[2]) + d[7]));
    poly[5] = (3 * ((3 * d[0]) + d[4])) + ((3 * d[5]) + d[8]);
    poly[6] = -((3 * (sy[0][0] - sy[1][0])) + ((2 * sxy[0][0]) + sxy[1][0]));
    poly[7] = -((3 * (z[ix-1][iy-1] - z[ix][iy-1])) + ((2 * sx[0][0]) + sx[1][0]));
    poly[8] = (2 * (sx[0][0] - sx[0][1])) + (sxy[0][0] + sxy[0][1]);
    poly[9] = -((3 * (sx[0][0] - sx[0][1])) + ((2 * sxy[0][0]) + sxy[0][1]));
    poly[10] = sxy[0][0];
    poly[11] = sx[0][0];
    poly[12] = (2 * (z[ix-1][iy-1] - z[ix-1][iy])) + (sy[0][0] + sy[0][1]);
    poly[13] = -((3 * (z[ix-1][iy-1] - z[ix-1][iy])) + ((2 * sy[0][0]) + sy[0][1]));
    poly[14] = sy[0][0];
    poly[15] = z[ix-1][iy-1];

    //return polyvalAkima(int(x),int(y),x,y,poly);
    m[0] = (((((poly[0] * (y - iy)) + poly[1]) * (y - iy)) + poly[2]) * (y - iy)) + poly[3];
    m[1] = (((((poly[4] * (y - iy)) + poly[5]) * (y - iy)) + poly[6]) * (y - iy)) + poly[7];
    m[2] = (((((poly[8] * (y - iy)) + poly[9]) * (y - iy)) + poly[10]) * (y - iy)) + poly[11];
    m[3] = (((((poly[12] * (y - iy)) + poly[13]) * (y - iy)) + poly[14]) * (y - iy)) + poly[15];
    return (((((m[0] * (x - ix)) + m[1]) * (x - ix)) + m[2]) * (x - ix)) + m[3];
}

