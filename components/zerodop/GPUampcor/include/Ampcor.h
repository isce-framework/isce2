//
// Author: Joshua Cohen
// Copyright 2016
//

#ifndef AMPCOR_H
#define AMPCOR_H

#include <vector>
#include "AmpcorMethods.h"
#include "Constants.h"

struct Ampcor {

    uint64_t imgAccessor1, imgAccessor2, offImgAccessor, offQualImgAccessor;
    
    std::vector<float> locationAcrossOffsetArr, locationDownOffsetArr, snrArr;
    std::vector<float> cov1Arr, cov2Arr, cov3Arr;
    float snrThresh, covThresh, xScaleFactor, yScaleFactor;

    std::vector<int> locationAcrossArr, locationDownArr;
    int imgDatatypes[2], imgWidths[2], imgBands[2], isMag[2];
    int firstRow, lastRow, rowSkip, firstCol, lastCol, colSkip, refChipWidth, refChipHeight;
    int schMarginX, schMarginY, nLookAcross, nLookDown, osampFact;
    int zoomWinSize, acrossGrossOff, downGrossOff, numRowTable;

    bool corr_debug, corr_display, usr_enable_gpu;

    Ampcor();
    ~Ampcor();

    void ampcor();
    int getLocationAcrossAt(int);
    int getLocationDownAt(int);
    float getLocationAcrossOffsetAt(int);
    float getLocationDownOffsetAt(int);
    float getSnrAt(int);
    float getCov1At(int);
    float getCov2At(int);
    float getCov3At(int);
    void dumpToFiles();
};

#endif
