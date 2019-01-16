//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCELIB_CONSTANTS_H
#define ISCELIB_CONSTANTS_H

namespace isceLib {
    static const int SCH_2_XYZ = 0;
    static const int XYZ_2_SCH = 1;

    static const int LLH_2_XYZ = 1;
    static const int XYZ_2_LLH = 2;
    static const int XYZ_2_LLH_OLD = 3;

    static const int WGS84_ORBIT = 1;
    static const int SCH_ORBIT = 2;

    static const int HERMITE_METHOD = 0;
    static const int SCH_METHOD = 1;
    static const int LEGENDRE_METHOD = 2;
}

#endif
