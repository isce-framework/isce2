//
// Author: Joshua Cohen
// Copyright 2017
//

#include <math.h>
#include <stdio.h>
#include "isceLibConstants.h"
#include "Ellipsoid.h"
#include "LinAlg.h"
using isceLib::Ellipsoid;
using isceLib::LinAlg;
using isceLib::LLH_2_XYZ;
using isceLib::XYZ_2_LLH;
using isceLib::XYZ_2_LLH_OLD;

Ellipsoid::Ellipsoid() {
    // Empty constructor

    return;
}

Ellipsoid::Ellipsoid(double maj, double ecc) {
    // Value constructor
    
    a = maj;
    e2 = ecc;
}

Ellipsoid::Ellipsoid(const Ellipsoid &e) {
    // Copy constructor
    
    a = e.a;
    e2 = e.e2;
}

void Ellipsoid::setMajorSemiAxis(double maj) {
    // Setter for object (used primarily by Python)
    
    a = maj;
}

void Ellipsoid::setEccentricitySquared(double ecc) {
    // Setter for object (used primarily by Python)
    
    e2 = ecc;
}

double Ellipsoid::rEast(double lat) {
    // One of several curvature functions used in ellipsoidal/spherical earth calculations
    
    return a / sqrt(1. - (e2 * pow(sin(lat), 2.)));
}

double Ellipsoid::rNorth(double lat) {
    // One of several curvature functions used in ellipsoidal/spherical earth calculations

    return (a * (1. - e2)) / pow((1. - (e2 * pow(lat, 2.))), 1.5);
}

double Ellipsoid::rDir(double hdg, double lat) {
    // One of several curvature functions used in ellipsoidal/spherical earth calculations
    
    double re, rn;

    re = rEast(lat);
    rn = rNorth(lat);
    return (re * rn) / ((re * pow(cos(hdg), 2.)) + (rn * pow(sin(hdg), 2.)));
}

void Ellipsoid::latLon(double v[3], double llh[3], int ctype) {
    /* 
     * Given a conversion type ('ctype'), either converts a vector to lat, lon, and height
     * above the reference ellipsoid, or given a lat, lon, and height produces a geocentric
     * vector.
    */

    if (ctype == LLH_2_XYZ) {
        double re;

        re = a / sqrt(1. - (e2 * pow(sin(llh[0]), 2.)));
        v[0] = (re + llh[2]) * cos(llh[0]) * cos(llh[1]);
        v[1] = (re + llh[2]) * cos(llh[0]) * sin(llh[1]);
        v[2] = ((re * (1. - e2)) + llh[2]) * sin(llh[0]);
    } else if (ctype == XYZ_2_LLH) { // Originally translated from python code in isceobj.Ellipsoid.xyz_to_llh
        double p, q, r, s, t, u, rv, w, k, d;

        p = (pow(v[0], 2.) + pow(v[1], 2.)) / pow(a, 2.);
        q = ((1. - e2) * pow(v[2], 2.)) / pow(a, 2.);
        r = (p + q - pow(e2, 2.)) / 6.;
        s = (pow(e2, 2.) * p * q) / (4. * pow(r, 3.));
        t = pow((1. + s + sqrt(s * (2. + s))), (1. / 3.));
        u = r * (1. + t + (1. / t));
        rv = sqrt(pow(u, 2.) + (pow(e2, 2.) * q));
        w = (e2 * (u + rv - q)) / (2. * rv);
        k = sqrt(u + rv + pow(w, 2.)) - w;
        d = (k * sqrt(pow(v[0], 2.) + pow(v[1], 2.))) / (k + e2);
        llh[0] = atan2(v[2], d);
        llh[1] = atan2(v[1], v[0]);
        llh[2] = ((k + e2 - 1.) * sqrt(pow(d, 2.) + pow(v[2], 2.))) / k;
    } else if (ctype == XYZ_2_LLH_OLD) {
        double b, p, tant, theta;

        b = a * sqrt(1. - e2);
        p = sqrt(pow(v[0], 2.) + pow(v[1], 2.));
        tant = (v[2] / p) * sqrt(1. / (1. - e2));
        theta = atan(tant);
        tant = (v[2] + (((1. / (1. - e2)) - 1.) * b * pow(sin(theta), 3.))) / (p - (e2 * a * pow(cos(theta), 3.)));
        llh[0] = atan(tant);
        llh[1] = atan2(v[1], v[0]);
        llh[2] = (p / cos(llh[0])) - (a / sqrt(1. - (e2 * pow(sin(llh[0]), 2.))));
    } else {
        printf("Error: Unrecognized conversion type in Ellipsoid::latLon (received %d).\n", ctype);
    }
}

void Ellipsoid::getAngs(double pos[3], double vel[3], double vec[3], double &az, double &lk) {
    // Computes the look vector given the look angle, azimuth angle, and position vector
    
    double llh[3], n[3], temp[3], c[3], t[3];
    LinAlg alg;

    latLon(pos, llh, XYZ_2_LLH);
    n[0] = -cos(llh[0]) * cos(llh[1]);
    n[1] = -cos(llh[0]) * sin(llh[1]);
    n[2] = -sin(llh[0]);
    lk = acos(alg.dot(n, vec) / alg.norm(vec));
    alg.cross(n, vel, temp);
    alg.unitVec(temp, c);
    alg.cross(c, n, temp);
    alg.unitVec(temp, t);
    az = atan2(alg.dot(c, vec), alg.dot(t, vec));
}

void Ellipsoid::getTCN_TCvec(double pos[3], double vel[3], double vec[3], double TCVec[3]) {
    // Computes the projection of an xyz vector on the TC plane in xyz
    
    double llh[3], n[3], temp[3], c[3], t[3];
    LinAlg alg;

    latLon(pos, llh, XYZ_2_LLH);
    n[0] = -cos(llh[0]) * cos(llh[1]);
    n[1] = -cos(llh[0]) * sin(llh[1]);
    n[2] = -sin(llh[0]);
    alg.cross(n, vel, temp);
    alg.unitVec(temp, c);
    alg.cross(c, n, temp);
    alg.unitVec(temp, t);
    for (int i=0; i<3; i++) TCVec[i] = (alg.dot(t, vec) * t[i]) + (alg.dot(c, vec) * c[i]);
}
