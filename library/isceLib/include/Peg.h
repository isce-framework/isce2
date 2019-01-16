//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCELIB_PEG_H
#define ISCELIB_PEG_H

namespace isceLib {
    struct Peg {
        double lat, lon, hdg;

        Peg();
        Peg(const Peg&);
    };
}

#endif
