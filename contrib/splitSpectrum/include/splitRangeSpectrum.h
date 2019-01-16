//
// Author: Heresh Fattahi
// Copyright 2017
//

#ifndef SPLITRANGESPECTRUM_H
#define SPLITRANGESPECTRUM_H

#include <iostream>

typedef std::string str;

namespace splitSpectrum {
    struct splitRangeSpectrum {
        str inputDS;
        str lbDS;
        str hbDS;
        int memsize, blocksize;
        float rangeSamplingRate;
        double lowBandWidth, highBandWidth;
        double lowCenterFrequency, highCenterFrequency; 
        void setInputDataset(str);
        void setLowbandDataset(str, str);
        void setMemorySize(int);
        void setBlockSize(int);
        void setBandwidth(double, double, double);
        void setSubBandCenterFrequencies(double, double);
        int split_spectrum_process();

    };
}

#endif
