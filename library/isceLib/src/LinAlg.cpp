//
// Author: Joshua Cohen
// Copyright 2017
//

#include <math.h>
#include "LinAlg.h"
using isceLib::LinAlg;

LinAlg::LinAlg() {
    return;
}

void LinAlg::cross(double u[3], double v[3], double w[3]) {
    w[0] = (u[1] * v[2]) - (u[2] * v[1]);
    w[1] = (u[2] * v[0]) - (u[0] * v[2]);
    w[2] = (u[0] * v[1]) - (u[1] * v[0]);
}

double LinAlg::dot(double v[3], double w[3]) {
    return (v[0] * w[0]) + (v[1] * w[1]) + (v[2] * w[2]);
}

void LinAlg::linComb(double k1, double u[3], double k2, double v[3], double w[3]) {
    for (int i=0; i<3; i++) w[i] = (k1 * u[i]) + (k2 * v[i]);
}

void LinAlg::matMat(double a[3][3], double b[3][3], double c[3][3]) {
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            c[i][j] = (a[i][0] * b[0][j]) + (a[i][1] * b[1][j]) + (a[i][2] * b[2][j]);
        }
    }
}

void LinAlg::matVec(double t[3][3], double v[3], double w[3]) {
    for (int i=0; i<3; i++) w[i] = (t[i][0] * v[0]) + (t[i][1] * v[1]) + (t[i][2] * v[2]);
}

double LinAlg::norm(double v[3]) {
    return sqrt(pow(v[0], 2.) + pow(v[1], 2.) + pow(v[2], 2.));
}

void LinAlg::tranMat(double a[3][3], double b[3][3]) {
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            b[i][j] = a[j][i];
        }
    }
}

void LinAlg::unitVec(double u[3], double v[3]) {
    double n;

    n = norm(u);
    if (n != 0.) {
        for (int i=0; i<3; i++) v[i] = u[i] / n;
    }
}

