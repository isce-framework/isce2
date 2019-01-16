//#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//#
//# Author: Piyush Agram
//# Copyright 2013, by the California Institute of Technology. ALL RIGHTS RESERVED.
//# United States Government Sponsorship acknowledged.
//# Any commercial use must be negotiated with the Office of Technology Transfer at
//# the California Institute of Technology.
//# This software may be subject to U.S. export control laws.
//# By accepting this software, the user agrees to comply with all applicable U.S.
//# export laws and regulations. User has the responsibility to obtain export licenses,
//# or other export authority as may be required before exporting such information to
//# foreign countries or providing access to foreign persons.
//#
//#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifndef addsubmodel_h
#define addsubmodel_h

#ifndef MESSAGE
#define MESSAGE cout<< "file " << __FILE__ << " line " <<__LINE__ << endl;
#endif

#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__ << " Exiting" << endl; exit(1);
#endif

#include "DataAccessor.h"
#include <stdint.h>

using namespace std;

class addsubmodel
{
    public:
        addsubmodel(){};
        ~addsubmodel(){};
        void setDims(int width, int length);
        void setScaleFactor(float scale);
        void setFlip(int flag);
        void cpxUnwprocess(uint64_t input, uint64_t model, uint64_t out);
        void cpxCpxprocess(uint64_t input, uint64_t model, uint64_t out);
        void unwUnwprocess(uint64_t input, uint64_t model, uint64_t out);
        void print();

    protected:
        int width;
        int length;
        float scaleFactor;
        int flip;
};

#endif
