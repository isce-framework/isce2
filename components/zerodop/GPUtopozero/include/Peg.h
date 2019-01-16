//
// Author: Joshua Cohen
// Copyright 2016
//

#ifndef PEG_H
#define PEG_H

struct Peg {
    double lat;
    double lon;
    double hdg;

    Peg();
    Peg(double,double,double);
};

#endif
