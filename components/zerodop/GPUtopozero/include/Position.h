//
// Author: Joshua Cohen
// Copyright 2016
//

#ifndef POSITION_H
#define POSITION_H

#include <vector>

struct Position {
    std::vector<double> j;
    std::vector<double> jdot;
    std::vector<double> jddot;

    Position();
    void lookvec(double,double,std::vector<double>&);
};

#endif
