//
// Author: Joshua Cohen
// Copyright 2017
//

#include "Peg.h"
using isceLib::Peg;

Peg::Peg() {
    // Empty constructor

    return;
}

Peg::Peg(const Peg &p) {
    // Copy constructor
    
    lat = p.lat;
    lon = p.lon;
    hdg = p.hdg;
}
