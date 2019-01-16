//
// Author: Joshua Cohen
// Copyright 2016
//

#include "Peg.h"

// Default constructor
Peg::Peg() {
    lat = 0.0;
    lon = 0.0;
    hdg = 0.0;
}

// Direct constructor
Peg::Peg(double a, double b, double c) {
    lat = a;
    lon = b;
    hdg = c;
}
