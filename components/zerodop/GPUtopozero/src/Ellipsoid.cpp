//
// Author: Joshua Cohen
// Copyright 2016
//

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "Constants.h"
#include "Ellipsoid.h"
#include "LinAlg.h"
using std::vector;

// Default constructor
Ellipsoid::Ellipsoid() {
    a = 0.0;
    e2 = 0.0;
}

// Direct constructor
Ellipsoid::Ellipsoid(double i1, double i2) {
    a = i1;
    e2 = i2;
}

// Copy constructor
Ellipsoid::Ellipsoid(const Ellipsoid &elp) {
    a = elp.a;
    e2 = elp.e2;
}

void Ellipsoid::latlon(vector<double> &v, vector<double> &llh, int type) {
    if (type == LLH_2_XYZ) {
        double re;
        
        re = a / sqrt(1.0 - (e2 * pow(sin(llh[0]),2)));
        v[0] = (re + llh[2]) * cos(llh[0]) * cos(llh[1]);
        v[1] = (re + llh[2]) * cos(llh[0]) * sin(llh[1]);
        v[2] = ((re * (1.0 - e2)) + llh[2]) * sin(llh[0]);
    } else if (type == XYZ_2_LLH) { // More accurate version derived from newer Python code
        double d,k,p,q,r,rv,s,t,u,w;

        p = (pow(v[0],2) + pow(v[1],2)) / pow(a,2);
        q = ((1.0 - e2) * pow(v[2],2)) / pow(a,2);
        r = (p + q - pow(e2,2)) / 6.0;
        s = (pow(e2,2) * p * q) / (4.0 * pow(r,3));
        t = pow((1.0 + s + sqrt(s * (2.0 + s))),(1.0/3.0));
        u = r * (1.0 + t + (1.0 / t));
        rv = sqrt(pow(u,2) + (pow(e2,2) * q));
        w = (e2 * (u + rv - q)) / (2.0 * rv);
        k = sqrt(u + rv + pow(w,2)) - w;
        d = (k * sqrt(pow(v[0],2) + pow(v[1],2))) / (k + e2);
        llh[0] = atan2(v[2], d);
        llh[1] = atan2(v[1], v[0]);
        llh[2] = ((k + e2 - 1.0) * sqrt(pow(d,2) + pow(v[2],2))) / k;
    } else if (type == XYZ_2_LLH_OLD) { // Less accurate version derived from original Fortran code
        double b,p,q,q3,re,theta;
        
        q = sqrt(1.0 / (1.0 - e2));
        q3 = (1.0 / (1.0 - e2)) - 1.0;
        b = a * sqrt(1.0 - e2);
        llh[1] = atan2(v[1], v[0]);
        p = sqrt(pow(v[0],2) + pow(v[1],2));
        theta = atan((v[2] / p) * q);
        llh[0] = atan((v[2] + (q3 * b * pow(sin(theta),3))) / (p - (e2 * a * pow(cos(theta),3))));
        re = a / sqrt(1.0 - (e2 * pow(sin(llh[0]),2)));
        llh[2] = (p / cos(llh[0])) - re;
    } else {
        printf("Error in Ellipsoid::latlon - Unknown method passed as type.\n");
        exit(1);
    }
}

double Ellipsoid::reast(double lat) {
    double ret;

    ret = a / sqrt(1.0 - (e2 * pow(sin(lat),2)));
    return ret;
}

double Ellipsoid::rnorth(double lat) {
    double ret;

    ret = (a * (1.0 - e2)) / pow((1.0 - (e2 * pow(sin(lat),2))),1.5);
    return ret;
}

double Ellipsoid::rdir(double hdg, double lat) {
    double re,rn,ret;

    re = reast(lat);
    rn = rnorth(lat);
    ret = (re * rn) / ((re * pow(cos(hdg),2)) + (rn * pow(sin(hdg),2)));
    return ret;
}

void Ellipsoid::getangs(vector<double> &pos, vector<double> &vel, vector<double> &vec, double &az, double &lk) {
    vector<double> c(3), n(3), t(3), llh(3), temp(3);
    double tvt,tvc,dd,vecnorm;
    
    LinAlg linalg;

    latlon(pos,llh,XYZ_2_LLH);
    n[0] = -cos(llh[0]) * cos(llh[1]);
    n[1] = -cos(llh[0]) * sin(llh[1]);
    n[2] = -sin(llh[0]);
    dd = linalg.dot(n,vec);
    vecnorm = linalg.norm(vec);
    lk = acos(dd / vecnorm);
    linalg.cross(n,vel,temp);
    linalg.unitvec(temp,c);
    linalg.cross(c,n,temp);
    linalg.unitvec(temp,t);
    tvt = linalg.dot(t,vec);
    tvc = linalg.dot(c,vec);
    az = atan2(tvc,tvt);
}

void Ellipsoid::getTVN_TCvec(vector<double> &pos, vector<double> &vel, vector<double> &vec, vector<double> &TCvec) {
    vector<double> c(3), n(3), t(3), llh(3), temp(3);
    double tvt,tvc;

    LinAlg linalg;
    
    latlon(pos,llh,XYZ_2_LLH);
    n[0] = -cos(llh[0]) * cos(llh[1]);
    n[1] = -cos(llh[0]) * sin(llh[1]);
    n[2] = -sin(llh[0]);
    linalg.cross(n,vel,temp);
    linalg.unitvec(temp,c);
    linalg.cross(c,n,temp);
    linalg.unitvec(temp,t);
    tvt = linalg.dot(t,vec);
    tvc = linalg.dot(c,vec);
    for (int i=0; i<3; i++) TCvec[i] = (tvt * t[i]) + (tvc * c[i]);
}

void Ellipsoid::tcnbasis(vector<double> &pos, vector<double> &vel, vector<double> &t, vector<double> &c, vector<double> &n) {
    vector<double> llh(3), temp(3);

    LinAlg linalg;

    latlon(pos,llh,XYZ_2_LLH);
    n[0] = -cos(llh[0]) * cos(llh[1]);
    n[1] = -cos(llh[0]) * sin(llh[1]);
    n[2] = -sin(llh[0]);
    linalg.cross(n,vel,temp);
    linalg.unitvec(temp,c);
    linalg.cross(c,n,temp);
    linalg.unitvec(temp,t);
}

