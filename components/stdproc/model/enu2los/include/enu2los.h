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

#ifndef enu2los_h
#define enu2los_h

#ifndef MESSAGE
#define MESSAGE cout<< "file " << __FILE__ << " line " <<__LINE__ << endl;
#endif

#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__ << " Exiting" << endl; exit(1);
#endif

#include "DataAccessor.h"
#include <stdint.h>

using namespace std;

class enu2los
{
    public:
        enu2los(){};
        ~enu2los(){};
        void setGeoDims(int width, int length);
        void setDims(int width, int length);
        void setWavelength(float wvl);
        void setScaleFactor(float scale);
        void setLatitudeInfo(float startLat, float delLat);
        void setLongitudeInfo(float startLon, float delLon);
        void process(uint64_t model, uint64_t lat, uint64_t lon, uint64_t los, uint64_t out);
        void print();

    protected:
        int width;
        int length;
        int geoWidth;
        int geoLength;
        float wavelength;
        float scaleFactor;
        float startLatitude;
        float deltaLatitude;
        float startLongitude;
        float deltaLongitude;
};

#endif
