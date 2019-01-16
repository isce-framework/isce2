//
// Author: Joshua Cohen
// Copyright 2016
//

#ifndef LINALG_H
#define LINALG_H

struct LinAlg {
    void matmat(double[3][3],double[3][3],double[3][3]);
    void matvec(double[3][3],double[3],double[3]);
    void tranmat(double[3][3],double[3][3]);
    void cross(double[3],double[3],double[3]);
    double dot(double[3],double[3]);
    void lincomb(double,double[3],double,double[3],double[3]);
    double norm(double[3]);
    void unitvec(double[3],double[3]);
    double cosineC(double,double,double);
    void enubasis(double,double,double[3][3]);
    void insertionSort(double*,int);
    int binarySearch(double*,int,int,double);
};

#endif
