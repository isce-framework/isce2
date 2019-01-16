//
// Author: Joshua Cohen
// Copyright 2016
//

#ifndef LINALG_H
#define LINALG_H

#include <vector>

struct LinAlg {
    void matmat(std::vector<std::vector<double> >&,std::vector<std::vector<double> >&,std::vector<std::vector<double> >&);
    void matvec(std::vector<std::vector<double> >&,std::vector<double>&,std::vector<double>&);
    void tranmat(std::vector<std::vector<double> >&,std::vector<std::vector<double> >&);
    void cross(std::vector<double>&,std::vector<double>&,std::vector<double>&);
    double dot(std::vector<double>&,std::vector<double>&);
    void lincomb(double,std::vector<double>&,double,std::vector<double>&,std::vector<double>&);
    double norm(std::vector<double>&);
    void unitvec(std::vector<double>&,std::vector<double>&);
    double cosineC(double,double,double);
    void enubasis(double,double,std::vector<std::vector<double> >&);
    void insertionSort(std::vector<double>&,int);
    int binarySearch(std::vector<double>&,int,int,double);
};

#endif
