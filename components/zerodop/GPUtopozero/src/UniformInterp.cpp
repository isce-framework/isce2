//
// Author: Joshua Cohen
// Copyright 2016
//
// Lots of template functions here because they're MUCH cleaner than writing each one
// out individually

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "UniformInterp.h"
using std::complex;
using std::max;
using std::min;
using std::vector;

// Options:
// U-double, V-float
// U-complex<double>, V-complex<double>
// U-float, V-float
template<class U, class V>
U UniformInterp::bilinear(double x, double y, vector<vector<V> > &z) {
    double x1, x2, y1, y2;
    U q11, q12, q21, q22, ret;

    x1 = floor(x);
    x2 = ceil(x);
    y1 = ceil(y);
    y2 = floor(y);
    q11 = z[int(y1)-1][int(x1)-1];
    q12 = z[int(y2)-1][int(x1)-1];
    q21 = z[int(y1)-1][int(x2)-1];
    q22 = z[int(y2)-1][int(x2)-1];

    if ((y1 == y2) && (x1 == x2)) {
        ret = q11;
    } else if (y1 == y2) {
        ret = (((x2 - x)/(x2 - x1))*q11) + (((x - x1)/(x2 - x1))*q21);
    } else if (x1 == x2) {
        ret = (((y2 - y)/(y2 - y1))*q11) + (((y - y1)/(y2 - y1))*q12);
    } else {
        ret = (q11*(x2 - x)*(y2 - y))/((x2 - x1)*(y2 - y1)) + 
              (q21*(x - x1)*(y2 - y))/((x2 - x1)*(y2 - y1)) + 
              (q12*(x2 - x)*(y - y1))/((x2 - x1)*(y2 - y1)) +
              (q22*(x - x1)*(y - y1))/((x2 - x1)*(y2 - y1));
    }
    return ret;
}

// Note that template functions need explicit type-based forward declarations to compile unambiguously
template double UniformInterp::bilinear(double,double,vector<vector<float> >&);
template complex<double> UniformInterp::bilinear(double,double,vector<vector<complex<double> > >&);
template float UniformInterp::bilinear(double,double,vector<vector<float> >&);

// Options
// U-double, V-float
// U-complex<double>, V-complex<double>
template<class U, class V>
U UniformInterp::bicubic(double x, double y, vector<vector<V> > &z) {
    int x1, x2, y1, y2;
    vector<U> dzdx(4), dzdy(4), dzdxy(4), zz(4), q(16), cl(16);
    vector<vector<U> > c(4,vector<U>(4));
    U qq,ret;
    double t,u;
    // Unfortunately there's no other way for this function to work than to have this...
    double wt_arr[16][16] = {{1.0,0.0,-3.0,2.0,0.0,0.0,0.0,0.0,-3.0,0.0,9.0,-6.0,2.0,0.0,-6.0,4.0},
                                  {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,3.0,0.0,-9.0,6.0,-2.0,0.0,6.0,-4.0},
                                  {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,9.0,-6.0,0.0,0.0,-6.0,4.0},
                                  {0.0,0.0,3.0,-2.0,0.0,0.0,0.0,0.0,0.0,0.0-9.0,6.0,0.0,0.0,6.0,-4.0},
                                  {0.0,0.0,0.0,0.0,1.0,0.0,-3.0,2.0,-2.0,0.0,6.0,-4.0,1.0,0.0,-3.0,2.0},
                                  {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,0.0,3.0,-2.0,1.0,0.0,-3.0,2.0},
                                  {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-3.0,2.0,0.0,0.0,3.0,-2.0},
                                  {0.0,0.0,0.0,0.0,0.0,0.0,3.0,-2.0,0.0,0.0,-6.0,4.0,0.0,0.0,3.0,-2.0},
                                  {0.0,1.0,-2.0,1.0,0.0,0.0,0.0,0.0,0.0,-3.0,6.0,-3.0,0.0,2.0,-4.0,2.0},
                                  {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,3.0,-6.0,3.0,0.0,-2.0,4.0,-2.0},
                                  {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-3.0,3.0,0.0,0.0,2.0,-2.0},
                                  {0.0,0.0,-1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,3.0,-3.0,0.0,0.0,-2.0,2.0},
                                  {0.0,0.0,0.0,0.0,0.0,1.0,-2.0,1.0,0.0,-2.0,4.0,-2.0,0.0,1.0,-2.0,1.0},
                                  {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,2.0,-1.0,0.0,1.0,-2.0,1.0},
                                  {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,-1.0,0.0,0.0,-1.0,1.0},
                                  {0.0,0.0,0.0,0.0,0.0,0.0,-1.0,1.0,0.0,0.0,2.0,-2.0,0.0,0.0,-1.0,1.0}};
    vector<vector<double> > wt(16);
    for (int i=0; i<16; i++) {
        vector<double> temp(wt_arr[i],wt_arr[i]+16);
        wt.push_back(temp);
    }

    x1 = floor(x) - 1;
    x2 = ceil(x) - 1;
    y1 = floor(y) - 1;
    y2 = ceil(y) - 1;
    zz[0] = z[y1][x1];
    zz[3] = z[y2][x1];
    zz[1] = z[y1][x2];
    zz[2] = z[y2][x2];
    dzdx[0] = (z[y1][x1+1] - z[y1][x1-1]) / 2.0;
    dzdx[1] = (z[y1][x2+1] - z[y1][x2-1]) / 2.0;
    dzdx[2] = (z[y2][x2+1] - z[y2][x2-1]) / 2.0;
    dzdx[3] = (z[y2][x1+1] - z[y2][x1-1]) / 2.0;
    dzdy[0] = (z[y1+1][x1] - z[y1-1][x1]) / 2.0;
    dzdy[1] = (z[y1+1][x2+1] - z[y1-1][x2]) / 2.0;
    dzdy[2] = (z[y2+1][x2+1] - z[y2-1][x2]) / 2.0;
    dzdy[3] = (z[y2+1][x1+1] - z[y2-1][x1]) / 2.0;
    dzdxy[0] = 0.25 * (z[y1+1][x1+1] - z[y1-1][x1+1] - z[y1+1][x1-1] + z[y1-1][x1-1]);
    dzdxy[3] = 0.25 * (z[y2+1][x1+1] - z[y2-1][x1+1] - z[y2+1][x1-1] + z[y2-1][x1-1]);
    dzdxy[1] = 0.25 * (z[y1+1][x2+1] - z[y1-1][x2+1] - z[y1+1][x2-1] + z[y1-1][x2-1]);
    dzdxy[2] = 0.25 * (z[y2+1][x2+1] - z[y2-1][x2+1] - z[y2+1][x2-1] + z[y2-1][x2-1]);

    for (int i=0; i<4; i++) {
        q[i] = zz[i];
        q[i+4] = dzdx[i];
        q[i+8] = dzdy[i];
        q[i+12] = dzdxy[i];
    }
    for (int i=0; i<16; i++) {
        qq = 0.0;
        for (int j=0; j<16; j++) {
            qq = qq + (wt[i][j] * q[j]);
        }
        cl[i] = qq;
    }
    for (int i=0; i<4; i++) {
        for (int j=0; j<4; j++) {
            c[i][j] = cl[(4*i)+j];
        }
    }
    t = (x - x1);
    u = (y - y1);
    ret = 0.0;
    for (int i=3; i>=0; i--) {
        ret = (t * ret) + (((((c[i][3] * u) + c[i][2])*u) + c[i][1]) * u) + c[i][0];
    }
    return ret;
}

template double UniformInterp::bicubic(double,double,vector<vector<float> >&);
template complex<double> UniformInterp::bicubic(double,double,vector<vector<complex<double> > >&);

void UniformInterp::sinc_coef(double beta, double relfiltlen, int decfactor, double pedestal, int weight, int &intplength, int &filtercoef, vector<double> &filter) {
    double wgt,s,fct,wgthgt,soff;

    intplength = round(relfiltlen / beta);
    filtercoef = intplength * decfactor;
    wgthgt = (1.0 - pedestal / 2.0);
    soff = (filtercoef - 1.0) / 2.0;

    for (int i=0; i<filtercoef; i++) {
        wgt = (1.0 - wgthgt) + (wgthgt * cos((M_PI * (i - soff)) / soff));
        s = (floor(i - soff) * beta) / (1.0 * decfactor);
        if (s != 0.0) fct = sin(M_PI * s) / (M_PI * s);
        else fct = 1.0;
        if (weight == 1) filter[i] = fct * wgt;
        else filter[i] = fct;
    }
}

// complex<double> == complex*8 (which was the return type in the original Fortran
complex<double> UniformInterp::sinc_eval(vector<complex<double> > &arrin, int nsamp, vector<float> &intarr, int idec, int ilen, int intp, double frp) {
    complex<double> ret(0.0,0.0);
    int ifrc;
    if ((intp >= (ilen-1)) && (intp < nsamp)) {
        ifrc = min(max(0,int(frp*idec)),idec-1);
        for (int i=0; i<ilen; i++) {
            ret = ret + (arrin[intp-i] * double(intarr[i + (ifrc*ilen)]));
        }
    }
    return ret;
}

// Options:
// U-float, V-float, W-float
// U-float, V-double, W-double
// U-complex<double>, V-complex<double>, W-float
template<class U,class V,class W>
U UniformInterp::sinc_eval_2d(vector<vector<V> > &arrin, vector<W> &intarr, int idec, int ilen, int intpx, int intpy, double frpx, double frpy, int xlen, int ylen) {
    U ret(0.0); // Will initialize ret to 0 regardless of type (complex<double> initializes to (0.0,0.0))
    int ifracx,ifracy;

    if (((intpx >= (ilen-1)) && (intpx < xlen)) && ((intpy >= (ilen-1)) && (intpy < ylen))) {
        ifracx = min(max(0,int(frpx*idec)),(idec-1));
        ifracy = min(max(0,int(frpy*idec)),(idec-1));
        for (int i=0; i<ilen; i++) {
            for (int j=0; j<ilen; j++) {
                ret = ret + (arrin[(intpx-i)][(intpy-j)] * double(intarr[(i+(ifracx*ilen))]) * double(intarr[(j+(ifracy*ilen))]));
            }
        }
    }
    return ret;
}

template float UniformInterp::sinc_eval_2d(vector<vector<float> >&,vector<float>&,int,int,int,int,double,double,int,int);
template float UniformInterp::sinc_eval_2d(vector<vector<double> >&,vector<double>&,int,int,int,int,double,double,int,int);
template complex<double> UniformInterp::sinc_eval_2d(vector<vector<complex<double> > >&,vector<float>&,int,int,int,int,double,double,int,int);

// Spline-related functions originally were in spline.f, but they make more sense to be here instead of in a separate object
float UniformInterp::interp2DSpline(int order, int nx, int ny, vector<vector<float> > &Z, double x, double y) {
    vector<double> A(order), R(order), Q(order), HC(order);
    double temp;
    float ret;
    int i0,j0,indi,indj;
    bool lodd;
    
    if ((order < 3) || (order > 20)) {
        printf("Error in UniformInterp::interp2DSpline - Spline order must be between 3 and 20\n");
        printf("(given order was %d)\n",order);
        exit(1);
    }
    lodd = (((order / 2) * 2) != order);
    if (lodd) {
        i0 = y - 0.5;
        j0 = x - 0.5;
    } else {
        i0 = y;
        j0 = x;
    }
    i0 = i0 - (order / 2) + 1;
    j0 = j0 - (order / 2) + 1;

    for (int i=1; i<=order; i++) {
        indi = min(max((i0+i),1),ny); 
        for (int j=1; j<=order; j++) {
            indj = min(max((j0+j),1),nx);
            A[(j-1)] = Z[(indi-1)][(indj-1)];
        }
        initSpline(A,order,R,Q);
        HC[i-1] = spline((x-j0),A,order,R);
    }
    initSpline(HC,order,R,Q);
    temp = spline(y-i0,HC,order,R);
    ret = float(temp);
    return ret;
}

void UniformInterp::initSpline(vector<double> &Y, int n, vector<double> &R, vector<double> &Q) {
    double p;

    Q[0] = 0.0;
    R[0] = 0.0;
    for (int i=2; i<n; i++) {
        p = (Q[i-2] / 2.) + 2.;
        Q[i-1] = -0.5 / p;
        R[i-1] = ((3 * (Y[i] - (2 * Y[i-1]) + Y[i-2])) - (R[i-2] / 2)) / p;
    }
    R[n-1] = 0.0;
    for (int i=(n-1); i>=2; i--) R[i-1] = (Q[i-1] * R[i]) + R[i-1];
}

double UniformInterp::spline(double x, vector<double> &Y, int n, vector<double> &R) {
    double xx,t0,t1,ret;
    int j;

    if (x < 1.) ret = Y[0] + ((x-1.) * (Y[1] - Y[0] - (R[1] / 6.)));
    else if (x > n) ret = Y[n-1] + ((x-n) * (Y[n-1] - Y[n-2] + (R[n-2] / 6.)));
    else {
        j = ifrac(x);
        xx = x - j;
        t0 = Y[j] - Y[j-1] - (R[j-1] / 3.) - (R[j] / 6.);
        t1 = xx * ((R[j-1] / 2.) + (xx * ((R[j] - R[j-1]) / 6)));
        ret = Y[j-1] + (xx * (t0 + t1));
    }
    return ret;
}

int UniformInterp::ifrac(double r) {
    int ret;
    ret = r;
    if (r >= 0.) return ret;
    if (r == ret) return ret;
    ret = ret - 1;
    return ret;
}

double UniformInterp::quadInterpolate(vector<double> &x, vector<double> &y, double xintp) {
    vector<double> x1(3), y1(3);
    double a,b,xin,ret;

    xin = xintp - x[0];
    for (int i=0; i<3; i++) {
        x1[i] = x[i] - x[0];
        y1[i] = y[i] - y[0];
    }
    a = ((-y1[1] * x1[2]) + (y1[2] * x1[1])) / ((-x1[2] * x1[1] * x1[1]) + (x1[1] * x1[2] * x1[2]));
    b = (y1[1] - (a * x1[1] * x1[1])) / x1[1];
    ret = y[0] + (a * xin * xin) + (b * xin);
    return ret;
}

