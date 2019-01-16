//
// Author: Joshua Cohen
// Copyright 2016
//

#ifndef PEGTRANS_H
#define PEGTRANS_H

#include <vector>
#include "Ellipsoid.h"
#include "Peg.h"

struct PegTrans {
    std::vector<std::vector<double> > mat;
    std::vector<std::vector<double> > matinv;
    std::vector<double> ov;
    double radcur;

    PegTrans();
    PegTrans(const PegTrans&);
    void convert_sch_to_xyz(std::vector<double>&,std::vector<double>&,int);
    void convert_schdot_to_xyzdot(std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&,int);
    void schbasis(std::vector<double>&,std::vector<std::vector<double> >&,std::vector<std::vector<double> >&);
    void radar_to_xyz(Ellipsoid&,Peg&);
};

#endif
