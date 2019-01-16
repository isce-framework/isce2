//
// Author: Joshua Cohen
// Copyright 2016
//

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "LinAlg.h"

using std::abs;

void LinAlg::matmat(double a[3][3], double b[3][3], double c[3][3]) {
    for (int i=0; i<3; i++ ) {
        c[i][0] = (a[i][0] * b[0][0]) + (a[i][1] * b[1][0]) + (a[i][2] * b[2][0]);
        c[i][1] = (a[i][0] * b[0][1]) + (a[i][1] * b[1][1]) + (a[i][2] * b[2][1]);
        c[i][2] = (a[i][0] * b[0][2]) + (a[i][1] * b[1][2]) + (a[i][2] * b[2][2]);
    }
}

void LinAlg::matvec(double a[3][3], double b[3], double c[3]) {
    c[0] = (a[0][0] * b[0]) + (a[0][1] * b[1]) + (a[0][2] *b[2]);
    c[1] = (a[1][0] * b[0]) + (a[1][1] * b[1]) + (a[1][2] *b[2]);
    c[2] = (a[2][0] * b[0]) + (a[2][1] * b[1]) + (a[2][2] *b[2]);
}

void LinAlg::tranmat(double a[3][3], double b[3][3]) {
    b[0][0]=a[0][0]; b[0][1]=a[1][0]; b[0][2]=a[2][0];
    b[1][0]=a[0][1]; b[1][1]=a[1][1]; b[1][2]=a[2][1];
    b[2][0]=a[0][2]; b[2][1]=a[1][2]; b[2][2]=a[2][2];
}

void LinAlg::cross(double u[3], double v[3], double w[3]) {
    w[0] = (u[1] * v[2]) - (u[2] * v[1]);
    w[1] = (u[2] * v[0]) - (u[0] * v[2]);
    w[2] = (u[0] * v[1]) - (u[1] * v[0]);
}

double LinAlg::dot(double v[3], double w[3]) {
    return (v[0] * w[0]) + (v[1] * w[1]) + (v[2] * w[2]);
}

void LinAlg::lincomb(double k1, double u[3], double k2, double v[3], double w[3]) {
    w[0] = (k1 * u[0]) + (k2 * v[0]);
    w[1] = (k1 * u[1]) + (k2 * v[1]);
    w[2] = (k1 * u[2]) + (k2 * v[2]);
}

double LinAlg::norm(double v[3]) {
    return sqrt(pow(v[0],2) + pow(v[1],2) + pow(v[2],2));
}

void LinAlg::unitvec(double v[3], double u[3]) {
    double n;

    n = norm(v);
    if (n != 0) {
        u[0] = v[0] / n;
        u[1] = v[1] / n;
        u[2] = v[2] / n;
    } else {
        printf("Error in LinAlg::unitvec - vector normalization divide by zero.\n");
        exit(1);
    }
}

double LinAlg::cosineC(double a, double b, double c) {
    double val,ret;

    val = (pow(a,2) + pow(b,2) - pow(c,2)) / (2 * a * b);
    ret = acos(val);
    return ret;
}

void LinAlg::enubasis(double lat, double lon, double enumat[3][3]) {
    enumat[0][0] = -sin(lon);
    enumat[0][1] = -sin(lat) * cos(lon);
    enumat[0][2] = cos(lat) * cos(lon);
    enumat[1][0] = cos(lon);
    enumat[1][1] = -sin(lat) * sin(lon);
    enumat[1][2] = cos(lat) * sin(lon);
    enumat[2][0] = 0.0;
    enumat[2][1] = cos(lat);
    enumat[2][2] = sin(lat);
}

// These two functions aren't linear algebra, but they work structurally in here
void LinAlg::insertionSort(double *arr, int len) {
    double temp;
    int j;
    for (int i=0; i<len; i++) {
        j = i;
        while ((j > 0) && (arr[j] < arr[(j-1)])) {
            temp = arr[j];       // could use <algorithm>'s std::swap, but not worth pulling in
            arr[j] = arr[(j-1)]; // whole library for one function...
            arr[(j-1)] = temp;
            j--;
        }
    }
}

// Adapted standard recursive binary search algorithm to allow for values not in
// the array (using a simple linear nearest-neighbor algorithm). Unfortunately
// to take all cases needs to run one more iteration than the standard binary
// search algo (due to needing to account for non-present elements)
int LinAlg::binarySearch(double *arr, int lft, int rght, double val) {
    if (rght >= lft) {
        int mid = (lft + rght) / 2;
        if (arr[mid] == val) return mid;
        else if (arr[mid] > val) {
            if (mid == lft) {
                if (mid > 0) {  // Check for nearest neighbor
                    if (abs(arr[(mid-1)] - val) < abs(arr[mid] - val)) return (mid-1);
                    else return mid;
                } else return 0;
            } else return binarySearch(arr,lft,(mid-1),val);
        } else {
            if (mid == rght) return rght;
            else return binarySearch(arr,(mid+1),rght,val);
        }
    } else return -1; // only hit if you pass in an initial width (rght) < 0
}

