//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCELIB_LINALG_H
#define ISCELIB_LINALG_H

namespace isceLib {
    struct LinAlg {
        LinAlg();
        void cross(double[3],double[3],double[3]);
        double dot(double[3],double[3]);
        void linComb(double,double[3],double,double[3],double[3]);
        void matMat(double[3][3],double[3][3],double[3][3]);
        void matVec(double[3][3],double[3],double[3]);
        double norm(double[3]);
        void tranMat(double[3][3],double[3][3]);
        void unitVec(double[3],double[3]);
    };
}

#endif

