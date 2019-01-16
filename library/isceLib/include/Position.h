//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCELIB_POSITION_H
#define ISCELIB_POSITION_H

namespace isceLib {
    struct Position {
        double j[3], jdot[3], jddt[3];

        Position();
        Position(const Position&);
        void lookVec(double,double,double[3]);
    };
}

#endif
