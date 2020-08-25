//
// Author: Joshua Cohen
// Copyright 2016
//

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <string>
#include <vector>
#include "DataAccessor.h"
#include "Constants.h"
#include "Ampcor.h"
#include "AmpcorMethods.h"
#ifdef GPU_ACC_ENABLED
#include "GPUamp.h"
#endif

using std::complex;
using std::fill;
using std::min;
using std::max;
using std::string;
using std::vector;


Ampcor::Ampcor() {
    imgDatatypes[0] = DTYPE_COMPLEX;
    imgDatatypes[1] = DTYPE_COMPLEX;
    imgWidths[0] = -1;
    imgWidths[1] = -1;
    imgBands[0] = -1;
    imgBands[1] = -1;
    isMag[0] = -1;
    isMag[1] = -1;
    usr_enable_gpu = true;
}

Ampcor::~Ampcor() {
    locationAcrossArr.clear();
    locationDownArr.clear();
    locationAcrossOffsetArr.clear();
    locationDownOffsetArr.clear();
    snrArr.clear();
    cov1Arr.clear();
    cov2Arr.clear();
    cov3Arr.clear();
}

int Ampcor::getLocationAcrossAt(int idx) { return locationAcrossArr[idx]; }
int Ampcor::getLocationDownAt(int idx) { return locationDownArr[idx]; }
float Ampcor::getLocationAcrossOffsetAt(int idx) { return locationAcrossOffsetArr[idx]; }
float Ampcor::getLocationDownOffsetAt(int idx) { return locationDownOffsetArr[idx]; }
float Ampcor::getSnrAt(int idx) { return snrArr[idx]; }
float Ampcor::getCov1At(int idx) { return cov1Arr[idx]; }
float Ampcor::getCov2At(int idx) { return cov2Arr[idx]; }
float Ampcor::getCov3At(int idx) { return cov3Arr[idx]; }

void Ampcor::dumpToFiles() {
    // Broad function to write internally-stored arrays to files
    printf("Writing offsets and quality metrics to files...\n");
    DataAccessor *offAccObj = (DataAccessor*)offImgAccessor;
    DataAccessor *offQualAccObj = (DataAccessor*)offQualImgAccessor;
    int nCols = int(ceil((lastRow - firstRow) / (1. * rowSkip)));
    int nLines = int(ceil((lastCol - firstCol) / (1. * colSkip)));
    vector<float> offsetLine(2*nCols);      // Interleaved locationOffsetArr line
    vector<float> offsetQualLine(4*nCols);  // Interleaved snrArr/covArr line
    for (int i=0; i<nLines; i++) {
        for (int j=0; j<nCols; j++) {
            offsetLine[2*j] = locationDownOffsetArr[i*nCols+j]; // Channel 1: Azimuth offsets
            offsetLine[2*j+1] = locationAcrossOffsetArr[i*nCols+j]; // Channel 2: Range offsets
            offsetQualLine[4*j] = snrArr[i*nCols+j]; // Channel 1: SNR
            offsetQualLine[4*j+1] = cov1Arr[i*nCols+j]; // Channel 2: Cov1
            offsetQualLine[4*j+2] = cov2Arr[i*nCols+j]; // Channel 3: Cov2
            offsetQualLine[4*j+3] = cov3Arr[i*nCols+j]; // Channel 4: Cov3
        }
        offAccObj->setLineSequential((char*)&offsetLine[0]);
        offQualAccObj->setLineSequential((char*)&offsetQualLine[0]);
    }
    printf("Written offsets to 'offsets.bil' and quality metrics (SNR+covs) to 'offsets_qual.bil'.\n");
}

void Ampcor::ampcor() {

    vector<vector<complex<float> > > refImg, schImg;
    vector<complex<float> > corrWin, osampCorrWin, interpCorr, osampInterpCorr;
    vector<complex<float> > padRefChip, padSchWin, osampPadRefChip, osampPadSchWin;

    vector<vector<float> > refChip, schWin, corrSurface;
    vector<vector<float> > osampRefChip, osampSchWin, osampCorrSurface;
    vector<float> covs(3), osampCovs(3), osampCorrOffset(2), sincInterp(MAXINTLGH);
    float downOffset, acrossOffset, corrPeak, snr, snrNormFactor, osampCorrPeak, osampAcrossOffset, osampDownOffset;
    float locationAcrossOffset, locationDownOffset, resampFactor, sincWeight, maxCorr, sincDelay;

    vector<int> isEdge(2), numPoints(2), corrPeaks(2);
    int schWinHeight, schWinWidth, padRefChipHeight, padRefChipWidth, padSchWinHeight, padSchWinWidth;
    int idx, idx2, idx3, peakMargin, mainArrIdx, peakRow, peakCol, corr_flag, xScaled, yScaled; 
    int counter, fft_direction, osampPeakRow, osampPeakCol;
    int osampRefChipWidth, osampRefChipHeight, osampSchWinWidth, osampSchWinHeight, osampCorrWidth, osampCorrHeight;
    int peakWinWidth, peakWinHeight, resampLength, sincInterpLength, maxImgWidth;

    vector<string> dtypeMap(2);

    DataAccessor *imgAccObj1 = (DataAccessor*)imgAccessor1;
    DataAccessor *imgAccObj2 = (DataAccessor*)imgAccessor2;

    AmpcorMethods aMethods;

    // Set defaults
    dtypeMap[0] = "real";
    dtypeMap[1] = "complex";
    corr_flag = 0;
    #ifndef GPU_ACC_ENABLED
    usr_enable_gpu = false;
    #endif

    aMethods.startOuterClock(); // start timer

    // Sinc interpolation kernel
    aMethods.fill_sinc(sincInterpLength, sincDelay, sincInterp);

    for (int i=3; i<15; i++) {
        int k = pow(2,i);
        aMethods.aFFT.fft1d(k, &osampPadRefChip[0], 0);
    }

    schMarginX = max(schMarginX,1);
    schMarginY = max(schMarginY,1);
    schWinWidth = refChipWidth + (2 * schMarginX);
    schWinHeight = refChipHeight + (2 * schMarginY);
    peakMargin = min(schMarginY, 4);
    peakMargin = min(peakMargin, schMarginX);
    peakWinWidth = refChipWidth + (2 * peakMargin);
    peakWinHeight = refChipHeight + (2 * peakMargin);
    padRefChipWidth = pow(2, ceil(log(refChipWidth)/log(2)));
    padRefChipHeight = pow(2, ceil(log(refChipHeight)/log(2)));
    padSchWinWidth = pow(2, ceil(log(peakWinWidth)/log(2)));
    padSchWinHeight = pow(2, ceil(log(peakWinHeight)/log(2)));
   
    
    // Outer "allocations"
    maxImgWidth = max(imgWidths[0], imgWidths[1]);
    int numPtsAcross = int(ceil((lastRow - firstRow) / (1. * rowSkip)));
    int numPtsDown = int(ceil((lastCol - firstCol) / (1. * colSkip)));
    locationAcrossArr.resize(numPtsAcross * numPtsDown);
    locationAcrossOffsetArr.resize(numPtsAcross * numPtsDown);
    locationDownArr.resize(numPtsAcross * numPtsDown);
    locationDownOffsetArr.resize(numPtsAcross * numPtsDown);
    snrArr.resize(numPtsAcross * numPtsDown);
    cov1Arr.resize(numPtsAcross * numPtsDown);
    cov2Arr.resize(numPtsAcross * numPtsDown);
    cov3Arr.resize(numPtsAcross * numPtsDown);
    
    // Begin ruggedize ... a bunch of input checking
    if ((imgDatatypes[0] != DTYPE_COMPLEX) && (imgDatatypes[0] != DTYPE_REAL)) {
        printf("WARNING - Do not understand data type for reference image\n");
        printf("Expecting flag to be real ('%s' [%d]) or complex ('%s' [%d])\n", dtypeMap[DTYPE_REAL].c_str(), DTYPE_REAL, 
                                                                                 dtypeMap[DTYPE_COMPLEX].c_str(), DTYPE_COMPLEX);
        printf("Data type flag set to %d\n", imgDatatypes[0]);
        printf("Resetting type flag to be complex [%d]\n", DTYPE_COMPLEX);
        imgDatatypes[0] = DTYPE_COMPLEX;
    }
    if ((imgDatatypes[1] != DTYPE_COMPLEX) && (imgDatatypes[1] != DTYPE_REAL)) {
        printf("WARNING - Do not understand data type for search image\n");
        printf("Expecting flag to be real ('%s' [%d]) or complex ('%s' [%d])\n", dtypeMap[DTYPE_REAL].c_str(), DTYPE_REAL, 
                                                                                 dtypeMap[DTYPE_COMPLEX].c_str(), DTYPE_COMPLEX);
        printf("Data type flag set to %d\n", imgDatatypes[0]);
        printf("Resetting type flag to be complex [%d]\n", DTYPE_COMPLEX);
        imgDatatypes[0] = DTYPE_COMPLEX;
    }
    if (imgWidths[0] > maxImgWidth) {
        printf("ERROR - Requesting processing of too wide a file\n");
        printf("             Image 1 width is %d pixels\n", imgWidths[0]);
        printf("Maximum allowed file width is %d pixels\n", maxImgWidth);
        exit(0);
    }
    if (imgWidths[1] > maxImgWidth) {
        printf("ERROR - Requesting processing of too wide a file\n");
        printf("             Image 2 width is %d pixels\n", imgWidths[1]);
        printf("Maximum allowed file width is %d pixels\n", maxImgWidth);
        exit(0);
    }
   
    // Read in refChipHeight lines of data into the refImg buffer for each chip
    // Read in schWinHeight=(refChipHeight+(2*schMarginY)) lines of data into the schImg buffer for each chip
    // Read in refChipWidth imgWidths of data into the refImg buffer for each chip
    // Read in schWinWidth=(refChipWidth+(2*schMarginX)) imgWidths of data into the schImg buffer for each chip
    if (schMarginX < 5) {
        printf("CAUTION - Requesting very small search window width\n");
        printf("Reference Window Width is            %10d sample pixels\n", refChipWidth);
        printf("Search Window Width Margin is        %10d sample pixels\n", schMarginX);
        printf("The rule of thumb is that the search window margin is at least 5\n");
        printf("pixels and is less than the reference window size divided by 5.\n");
        int check_temp = max(5, int(round(refChipWidth/6.)));
        printf("Suggested Search Window Width Margin is %d sample pixels\n\n", check_temp);
    }
    int check_bound = int(round((1.*refChipWidth)/schMarginX));
    if (check_bound < 5) {
        printf("CAUTION - Requesting very large search window width\n");
        printf("Reference Window Width is             %10d sample pixels\n", refChipWidth);
        printf("Search Window Width Margin is         %10d sample pixels\n", schMarginX);
        printf("The rule of thumb is that the search window margin is at least 5\n");
        printf("pixels and is less than the reference window size divided by 5.\n");
        int check_temp = max(5, int(round(refChipWidth/6.)));
        printf("Suggested Search Window Width Margin is %d sample pixels\n\n\n", check_temp);
    }
    if (schMarginY < 5) {
        printf("CAUTION - Requesting very small search window height\n");
        printf("Reference Window Height is             %10d sample pixels\n", refChipHeight);
        printf("Search Window Height Margin is       %10d sample pixels\n", schMarginY);
        printf("The rule of thumb is that the search window margin is at least 5\n");
        printf("pixels and is less than the reference window size divided by 5.\n");
        int check_temp = max(5, int(round(refChipHeight/6.)));
        printf("Suggested Search Window Height Margin is %d sample pixels\n\n", check_temp);
    }
    check_bound = int(round((1.*refChipHeight)/schMarginY));
    if (check_bound < 5) {
        printf("CAUTION - Requesting very large search window height\n");
        printf("Reference Window Height is             %10d sample pixels\n", refChipHeight);
        printf("Search Window Height Margin is           %10d sample pixels\n", schMarginY);
        printf("The rule of thumb is that the search window margin is at least 5\n");
        printf("pixels and is less than the reference window size divided by 5.\n");
        int check_temp = max(5, int(round(refChipHeight/6.)));
        printf("Suggested Search Window Height Margin is %d sample pixels\n\n\n", check_temp);
    }

    if (zoomWinSize < 8) {
        printf("WARNING - Covariance Surface Window Size Very Small\n");
        printf("It is the number of pixels in the Correlation Surface to oversample.\n");
        printf("Minimum Recommended Value for the Covariance Surface Window Size is 8.\n");
        printf("Requested covariance surface window size of %d pixels\n\n", zoomWinSize);
    }

    printf("Requested resolving shifts to 1/%d of a pixel\n\n", (osampFact*2));
  
    firstCol = max(firstCol, 1);
    lastCol = min(lastCol, imgWidths[0]);

    if ((rowSkip < refChipHeight) || (colSkip < refChipWidth)) {
        printf("INFORMATION - you choose skips which are small for your window sizes\n");
        printf("Normally the skip size is bigger than the box size\n");
        printf("Across your skip is %10d but your window is %10d\n", colSkip, refChipWidth);
        printf("Down   your skip is %10d but your window is %10d\n", rowSkip, refChipHeight);
        printf("This means that the image chips are larger than the separation between chips\n\n");
    }

    covThresh = min(covThresh, float(999.999999));
    nLookAcross = max(1, nLookAcross);
    nLookDown = max(1, nLookDown);

    if ((nLookAcross > 1) || (nLookDown > 1)) {
        printf("INFORMATION - You are looking down the data before cross correlation.\n");
        printf("Averaging the samples across the file by a factor of %d\n", nLookAcross);
        printf("Averaging the lines   down   the file by a factor of %d\n\n", nLookDown);
    }

    // end ruggedize

    if (usr_enable_gpu) { // gpu ampcor
        #ifdef GPU_ACC_ENABLED
        vector<int*> outputArrs_int(2,0);
        vector<float*> outputArrs_flt(6,0);
        int inputs_int[10];
        float inputs_flt[2];

        int padWinWidth = pow(2, ceil(log(2*schWinWidth)/log(2)));
        int padWinHeight = pow(2, ceil(log(2*schWinHeight)/log(2)));

        inputs_int[0] = refChipWidth;
        inputs_int[1] = refChipHeight;
        inputs_int[2] = schWinWidth;
        inputs_int[3] = schWinHeight;
        inputs_int[4] = acrossGrossOff;
        inputs_int[5] = downGrossOff;
        inputs_int[6] = padWinWidth;
        inputs_int[7] = padWinHeight;
        inputs_int[8] = zoomWinSize;

        inputs_flt[0] = snrThresh;
        inputs_flt[1] = covThresh;

        int nBlocksToRead = min((numPtsAcross * numPtsDown), nBlocksPossible(inputs_int)); // Make sure we don't tell it to run 2000 pairs if the image is only 1500
        inputs_int[9] = nBlocksToRead;
        int nGpuIter = (numPtsAcross * numPtsDown) / nBlocksToRead; // ex 1600 total blocks, 500 blocks/run, 3 full runs (1 partial run calculated later)
        int globalBlock = 0; // block n out of nGpuIter * nBlocksToRead

        vector<complex<float>*> refChips(nBlocksToRead,0), schWins(nBlocksToRead,0); // containers for pointers to chip/win arrs
        for (int i=0; i<nBlocksToRead; i++) { // Malloc the arrays
            refChips[i] = new complex<float>[refChipWidth*refChipHeight];
            schWins[i] = new complex<float>[schWinWidth*schWinHeight];
        }
        outputArrs_int[0] = new int[nBlocksToRead];
        outputArrs_int[1] = new int[nBlocksToRead];
        outputArrs_flt[0] = new float[nBlocksToRead];
        outputArrs_flt[1] = new float[nBlocksToRead];
        outputArrs_flt[2] = new float[nBlocksToRead];
        outputArrs_flt[3] = new float[nBlocksToRead];
        outputArrs_flt[4] = new float[nBlocksToRead];
        outputArrs_flt[5] = new float[nBlocksToRead];

        int globalX[nBlocksToRead], globalY[nBlocksToRead];

        printf("GPU-accelerated Ampcor enabled.\nRunning Ampcor in %d batch(es) of %d reference/search sub-image pairs", nGpuIter, nBlocksToRead);
        if ((nGpuIter*nBlocksToRead) < (numPtsAcross*numPtsDown)) {
            printf(", with one final partial batch of %d blocks.\n", ((numPtsAcross*numPtsDown)-(nGpuIter*nBlocksToRead)));
        } else {
            printf(".\n");
        }
        // GPU full iterations
        complex<float> *cx_read_line = new complex<float>[maxImgWidth];
        float *rl_read_line = new float[maxImgWidth];
        for (int i=0; i<nGpuIter; i++) {

            // Step 1: Copy new data blocks into container.

            int nBlocksLeft = nBlocksToRead;
            while (nBlocksLeft > 0) {
                int firstImgLine = firstRow + schMarginY + (int(globalBlock / numPtsDown) * rowSkip); // First line in image corresponding to first block
                int firstImgLineScaled = round(firstImgLine * yScaleFactor);
                int blocksInRow = min((numPtsDown - (globalBlock % numPtsDown)), nBlocksLeft); // Number of blocks to read in the line
                int colOffset = firstCol + schMarginX + ((globalBlock % numPtsDown) * colSkip); // Location along the line for first block
                
                // Read in reference blocks
                if (imgDatatypes[0] == DTYPE_COMPLEX) {
                    for (int yy=0; yy<refChipHeight; yy++) { // Iterate over number of image lines to read
                        imgAccObj1->getLineBand((char*)cx_read_line, (firstImgLine + yy - 2), imgBands[0]); // Read in a line
                        for (int block=0; block<blocksInRow; block++) { // Iterate over number of blocks to read
                            int blockArrIdx = (globalBlock + block) % nBlocksToRead; // Index of block in GPU reference array
                            int startCol = colOffset + (block * colSkip) - 1; // Get leftmost pixel of each block ### CHECK -1 FACTOR LATER, NOT SURE IF RIGHT
                            for (int xx=0; xx<refChipWidth; xx++) { // Iterate over block width
                                if (isMag[0] == 0) refChips[blockArrIdx][(xx*refChipHeight)+yy] = cx_read_line[startCol+xx];
                                else refChips[blockArrIdx][(xx*refChipHeight)+yy] = complex<float>(abs(cx_read_line[startCol+xx]),0.);
                            }
                            globalX[blockArrIdx] = startCol + 1;
                            globalY[blockArrIdx] = firstImgLine;
                        }
                    }
                } else if (imgDatatypes[0] == DTYPE_REAL) {
                    for (int yy=0; yy<refChipHeight; yy++) {
                        imgAccObj1->getLineBand((char*)rl_read_line, (firstImgLine + yy - 2), imgBands[0]);
                        for (int block=0; block<blocksInRow; block++) {
                            int blockArrIdx = (globalBlock + block) % nBlocksToRead;
                            int startCol = colOffset + (block * colSkip) - 1;
                            for (int xx=0; xx<refChipWidth; xx++) {
                                refChips[blockArrIdx][(xx*refChipHeight)+yy] = complex<float>(rl_read_line[startCol+xx],0.);
                            }
                            globalX[blockArrIdx] = startCol + 1;
                            globalY[blockArrIdx] = firstImgLine;
                        }
                    }
                }

                // Read in search blocks
                if (imgDatatypes[1] == DTYPE_COMPLEX) {
                    for (int yy=0; yy<schWinHeight; yy++) {
                        imgAccObj2->getLineBand((char*)cx_read_line, (firstImgLineScaled + yy - schMarginY + downGrossOff - 2), imgBands[1]);
                        for (int block=0; block<blocksInRow; block++) {
                            int blockArrIdx = (globalBlock + block) % nBlocksToRead;
                            int startCol = colOffset + (block * colSkip) - 1;
                            for (int xx=0; xx<schWinWidth; xx++) {
                                if (isMag[1] == 0) schWins[blockArrIdx][(xx*schWinHeight)+yy] = cx_read_line[startCol+xx-schMarginX+acrossGrossOff];
                                else schWins[blockArrIdx][(xx*schWinHeight)+yy] = complex<float>(abs(cx_read_line[startCol+xx-schMarginX+acrossGrossOff]),0.);
                            }
                        }
                    }
                } else if (imgDatatypes[1] == DTYPE_REAL) {
                    for (int yy=0; yy<schWinHeight; yy++) {
                        imgAccObj2->getLineBand((char*)rl_read_line, (firstImgLineScaled + yy - schMarginY + downGrossOff - 2), imgBands[1]);
                        for (int block=0; block<blocksInRow; block++) {
                            int blockArrIdx = (globalBlock + block) % nBlocksToRead;
                            int startCol = colOffset + (block * colSkip) - 1;
                            for (int xx=0; xx<schWinWidth; xx++) {
                                schWins[blockArrIdx][(xx*schWinHeight)+yy] = complex<float>(rl_read_line[startCol+xx-schMarginX+acrossGrossOff],0.);
                            }
                        }
                    }
                }
                nBlocksLeft = nBlocksLeft - blocksInRow; // Update how many blocks left in the batch to read
                globalBlock = globalBlock + blocksInRow; // Update block position globally in the image
            }

            // Step 2: Call CUDA version of Ampcor
            
            runGPUAmpcor(inputs_flt, inputs_int, (void **)(&(refChips[0])), (void **)(&(schWins[0])), globalX, globalY, &(outputArrs_int[0]), &(outputArrs_flt[0]));
           
            for (int j=0; j<nBlocksToRead; j++) {
                locationAcrossArr[(i*nBlocksToRead)+j] = outputArrs_int[0][j];
                locationDownArr[(i*nBlocksToRead)+j] = outputArrs_int[1][j];
                locationAcrossOffsetArr[(i*nBlocksToRead)+j] = outputArrs_flt[0][j];
                locationDownOffsetArr[(i*nBlocksToRead)+j] = outputArrs_flt[1][j];
                snrArr[(i*nBlocksToRead)+j] = outputArrs_flt[2][j];
                cov1Arr[(i*nBlocksToRead)+j] = outputArrs_flt[3][j];
                cov2Arr[(i*nBlocksToRead)+j] = outputArrs_flt[4][j];
                cov3Arr[(i*nBlocksToRead)+j] = outputArrs_flt[5][j];
            }
            
            //free(outputArrs_int[0]);
            //free(outputArrs_int[1]);
            //free(outputArrs_flt[0]);
            //free(outputArrs_flt[1]);
            //free(outputArrs_flt[2]);
            //free(outputArrs_flt[3]);
            //free(outputArrs_flt[4]);
            //free(outputArrs_flt[5]);
        }
        
        int lastBlocksToRead = (numPtsAcross * numPtsDown) - (nGpuIter * nBlocksToRead); // 0 if no final partial batch needed, will not trigger this last part if so
        if (lastBlocksToRead > 0) {
            int nBlocksLeft = lastBlocksToRead;
            
            while (nBlocksLeft > 0) {
                int firstImgLine = firstRow + schMarginY + (int(globalBlock / numPtsDown) * rowSkip);
                int firstImgLineScaled = round(firstImgLine * yScaleFactor);
                int blocksInRow = min((numPtsDown - (globalBlock % numPtsDown)), nBlocksLeft);
                int colOffset = firstCol + schMarginX + ((globalBlock % numPtsDown) * colSkip);
                if (imgDatatypes[0] == DTYPE_COMPLEX) {
                    for (int yy=0; yy<refChipHeight; yy++) { // Iterate over number of image lines to read
                        imgAccObj1->getLineBand((char*)cx_read_line, (firstImgLine + yy - 2), imgBands[0]); // Read in a line
                        for (int block=0; block<blocksInRow; block++) { // Iterate over number of blocks to read
                            int blockArrIdx = (globalBlock + block) % nBlocksToRead;
                            int startCol = colOffset + (block * colSkip) - 1; // Get leftmost pixel of each block ### CHECK -1 FACTOR LATER, NOT SURE IF RIGHT
                            for (int xx=0; xx<refChipWidth; xx++) { // Iterate over block width
                                if (isMag[0] == 0) refChips[blockArrIdx][(xx*refChipHeight)+yy] = cx_read_line[startCol+xx];
                                else refChips[blockArrIdx][(xx*refChipHeight)+yy] = complex<float>(abs(cx_read_line[startCol+xx]),0.);
                            }
                            globalX[blockArrIdx] = startCol + 1;
                            globalY[blockArrIdx] = firstImgLine;
                        }
                    }
                } else if (imgDatatypes[0] == DTYPE_REAL) {
                    for (int yy=0; yy<refChipHeight; yy++) {
                        imgAccObj1->getLineBand((char*)rl_read_line, (firstImgLine + yy - 2), imgBands[0]);
                        for (int block=0; block<blocksInRow; block++) {
                            int blockArrIdx = (globalBlock + block) % nBlocksToRead;
                            int startCol = colOffset + (block * colSkip) - 1;
                            for (int xx=0; xx<refChipWidth; xx++) {
                                refChips[blockArrIdx][(xx*refChipHeight)+yy] = complex<float>(rl_read_line[startCol+xx],0.);
                            }
                            globalX[blockArrIdx] = startCol + 1;
                            globalY[blockArrIdx] = firstImgLine;
                        }
                    }
                }
                if (imgDatatypes[1] == DTYPE_COMPLEX) {
                    for (int yy=0; yy<schWinHeight; yy++) {
                        imgAccObj2->getLineBand((char*)cx_read_line, (firstImgLineScaled + yy - schMarginY + downGrossOff - 2), imgBands[1]);
                        for (int block=0; block<blocksInRow; block++) {
                            int blockArrIdx = (globalBlock + block) % nBlocksToRead;
                            int startCol = colOffset + (block * colSkip) - 1;
                            for (int xx=0; xx<schWinWidth; xx++) {
                                if (isMag[1] == 0) schWins[blockArrIdx][(xx*schWinHeight)+yy] = cx_read_line[startCol+xx-schMarginX+acrossGrossOff];
                                else schWins[blockArrIdx][(xx*schWinHeight)+yy] = complex<float>(abs(cx_read_line[startCol+xx-schMarginX+acrossGrossOff]),0.);
                            }
                        }
                    }
                } else if (imgDatatypes[1] == DTYPE_REAL) {
                    for (int yy=0; yy<schWinHeight; yy++) {
                        imgAccObj2->getLineBand((char*)rl_read_line, (firstImgLineScaled + yy - schMarginY + downGrossOff - 2), imgBands[1]);
                        for (int block=0; block<blocksInRow; block++) {
                            int blockArrIdx = (globalBlock + block) % nBlocksToRead;
                            int startCol = colOffset + (block * colSkip) - 1;
                            for (int xx=0; xx<schWinWidth; xx++) {
                                schWins[blockArrIdx][(xx*schWinHeight)+yy] = complex<float>(rl_read_line[startCol+xx-schMarginX+acrossGrossOff],0.);
                            }
                        }
                    }
                }
                nBlocksLeft = nBlocksLeft - blocksInRow; // Update how many blocks left in the batch to read
                globalBlock = globalBlock + blocksInRow; // Update block position globally in the image
            }

            inputs_int[9] = lastBlocksToRead;

            runGPUAmpcor(inputs_flt, inputs_int, (void **)(&(refChips[0])), (void **)(&(schWins[0])), globalX, globalY, &(outputArrs_int[0]), &(outputArrs_flt[0]));
            
            for (int j=0; j<lastBlocksToRead; j++) {
                locationAcrossArr[(nGpuIter*nBlocksToRead)+j] = outputArrs_int[0][j];
                locationDownArr[(nGpuIter*nBlocksToRead)+j] = outputArrs_int[1][j];
                locationAcrossOffsetArr[(nGpuIter*nBlocksToRead)+j] = outputArrs_flt[0][j];
                locationDownOffsetArr[(nGpuIter*nBlocksToRead)+j] = outputArrs_flt[1][j];
                snrArr[(nGpuIter*nBlocksToRead)+j] = outputArrs_flt[2][j];
                cov1Arr[(nGpuIter*nBlocksToRead)+j] = outputArrs_flt[3][j];
                cov2Arr[(nGpuIter*nBlocksToRead)+j] = outputArrs_flt[4][j];
                cov3Arr[(nGpuIter*nBlocksToRead)+j] = outputArrs_flt[5][j];
            }
            //free(outputArrs_int[0]);
            //free(outputArrs_int[1]);
            //free(outputArrs_flt[0]);
            //free(outputArrs_flt[1]);
            //free(outputArrs_flt[2]);
            //free(outputArrs_flt[3]);
            //free(outputArrs_flt[4]);
            //free(outputArrs_flt[5]);
        }

        // clean up at the end
        for (int i=0; i<nBlocksToRead; i++) {
            delete[] refChips[i];
            delete[] schWins[i];
        }
        delete[] cx_read_line;
        delete[] rl_read_line;
        for (int i=0; i<2; i++) delete[] outputArrs_int[i];
        for (int i=0; i<6; i++) delete[] outputArrs_flt[i];
        numRowTable = numPtsDown * numPtsAcross;
    
        dumpToFiles();
        #endif 
    } else { // non-gpu ampcor

        padRefChip.resize(padRefChipWidth*padRefChipHeight);
        osampPadRefChip.resize(4*padRefChipWidth*padRefChipHeight);
        refImg.resize(maxImgWidth);
        refChip.resize(refChipWidth);
        osampRefChip.resize(2*refChipWidth);

        padSchWin.resize(padSchWinWidth*padSchWinHeight);
        osampPadSchWin.resize(4*padSchWinWidth*padSchWinHeight);
        schImg.resize(maxImgWidth);
        schWin.resize(schWinWidth);
        osampSchWin.resize(2*peakWinWidth);

        osampCorrWin.resize(osampFact*zoomWinSize*osampFact*zoomWinSize);
        corrWin.resize(zoomWinSize*zoomWinSize);
        interpCorr.resize(osampFact*zoomWinSize*zoomWinSize);
        osampInterpCorr.resize(osampFact*zoomWinSize*osampFact*zoomWinSize);
        corrSurface.resize(schWinWidth);
        osampCorrSurface.resize(2*peakWinWidth);

        // Inner "allocations"
        for (int i=0; i<maxImgWidth; i++) {
            refImg[i].resize(refChipHeight);
            schImg[i].resize(schWinHeight);
        }
        for (int i=0; i<refChipWidth; i++) {
            refChip[i].resize(refChipHeight);
        }
        for (int i=0; i<schWinWidth; i++) {
            schWin[i].resize(schWinHeight);
            corrSurface[i].resize(schWinHeight);
        }
        for (int i=0; i<(2*refChipWidth); i++) {
            osampRefChip[i].resize(2*refChipHeight);
        }
        for (int i=0; i<(2*peakWinWidth); i++) {
            osampSchWin[i].resize(2*peakWinHeight);
            osampCorrSurface[i].resize(2*peakWinHeight);
        }

        // loop over data begins. initialize number of rows in output table
        mainArrIdx = 0;

        complex<float> *cx_read_line = new complex<float>[maxImgWidth];
        float *rl_read_line = new float[maxImgWidth];

        for (int y=(firstRow+schMarginY); y<=(lastRow+schMarginY); y+=rowSkip) {
            // ----------------------------------
            // NOTE:
            //      refChipHeight is the Reference image window size in line pixels
            //      imgWidths[0] is pixel width of image 1
            //      imgWidths[1] is pixel width of image 2
            //      refImg[0][yy]: image lines are read into each refImg[r][c] 'column'
            // ----------------------------------

            printf("At line = %d\n", (y-schMarginY+1));
            yScaled = round(yScaleFactor * y);

            if (imgDatatypes[0] == DTYPE_COMPLEX) {
                // search lines from current image line y down to the refChipHeight lines below
                for (int yy=0; yy<refChipHeight; yy++) {
                    // using getLineBand(char *destination_array, int line_to_read_from, int band_to_read_from)
                    imgAccObj1->getLineBand((char*)cx_read_line, (y + yy - 2), imgBands[0]);
                    for (int i=0; i<imgWidths[0]; i++) {
                        if (isMag[0] == 0) refImg[i][yy] = cx_read_line[i];
                        else refImg[i][yy] = complex<float>(abs(cx_read_line[i]),0.);
                    }
                }
            } else if (imgDatatypes[0] == DTYPE_REAL) {
               for (int yy=0; yy<refChipHeight; yy++) {
                    imgAccObj1->getLineBand((char*)rl_read_line, (y + yy - 2), imgBands[0]);
                    for (int xx=0; xx<imgWidths[0]; xx++) refImg[xx][yy] = complex<float>(rl_read_line[xx],0.);
               }
            }
            
            if (imgDatatypes[1] == DTYPE_COMPLEX) {
                for (int yy=0; yy<schWinHeight; yy++) {
                    imgAccObj2->getLineBand((char*)cx_read_line, (yScaled + yy - schMarginY + downGrossOff - 2), imgBands[1]);
                    for (int i=0; i<imgWidths[1]; i++) {
                        if (isMag[1] == 0) schImg[i][yy] = cx_read_line[i];
                        else schImg[i][yy] = complex<float>(abs(cx_read_line[i]),0.);
                    }
                }
            } else if (imgDatatypes[1] == DTYPE_REAL) {
                for (int yy=0; yy<schWinHeight; yy++) {
                    imgAccObj2->getLineBand((char*)rl_read_line, (yScaled + yy - schMarginY + downGrossOff - 2), imgBands[1]);
                    for (int xx=0; xx<imgWidths[1]; xx++) schImg[xx][yy] = complex<float>(rl_read_line[xx],0.);
                }
            }
            
            aMethods.startInnerClock();

            for (int x=(firstCol+schMarginX); x<=(lastCol+schMarginX); x+=colSkip) {
                xScaled = round(xScaleFactor * x);
                
                // get the reference image and search images

                for (int yy=0; yy<refChipHeight; yy++) {
                    for (int xx=0; xx<refChipWidth; xx++) refChip[xx][yy] = abs(refImg[x+xx-1][yy]);
                }
                for (int yy=0; yy<schWinHeight; yy++) {
                    for (int xx=0; xx<schWinWidth; xx++) schWin[xx][yy] = abs(schImg[xScaled+xx-1-schMarginX+acrossGrossOff][yy]);
                }

                // correlate the subimages

                aMethods.correlate(refChip, schWin, refChipWidth, refChipHeight, schWinWidth, schWinHeight, nLookAcross, nLookDown,
                                    corrPeak, covs, corrSurface, peakRow, peakCol, isEdge, corr_flag, corr_debug);
          
                acrossOffset = (1. * peakRow * nLookAcross) - schMarginX + acrossGrossOff;
                downOffset = (1. * peakCol * nLookDown) - schMarginY + downGrossOff;

                // decide with points are good notches and print out the notch values

                if ((corr_flag == 0) && (isEdge[0] == 0) && (isEdge[1] == 0)) { // found a potentially good data point

                    // compute the "snr"

                    if (corr_display) {
                        printf("\nCorrelation Surface at %d %d\n", (x+((refChipWidth-1)/2)), (y+((refChipHeight-1)/2)));
                        for (int l=max(peakCol-3,1); l<=min(peakCol+5,schWinHeight-refChipHeight); l++) {
                            for (int k=max(peakRow-3,1); k<=min(peakRow+5,schWinWidth-refChipWidth); k++) {
                                printf("%f ", (pow(corrSurface[k-1][l-1],2) / pow(corrPeak,2)));
                            }
                            printf("\n");
                        }
                    }

                    snrNormFactor = 0.;
                    counter = 0;
                    for (int l=max(peakCol-9,1); l<=min(peakCol+11,schWinHeight-refChipHeight); l++) {
                        for (int k=max(peakRow-9,1); k<=min(peakRow+11,schWinWidth-refChipWidth); k++) {
                            counter += 1;
                            snrNormFactor = snrNormFactor + pow(corrSurface[k-1][l-1],2);
                        }
                    }
                    
                    snrNormFactor = (snrNormFactor - pow(corrPeak,2)) / (counter - 1);
                    snr = pow(corrPeak,2) / max(snrNormFactor,float(1.e-10));

                    if ((snr > snrThresh) && (covs[0] < covThresh) && (covs[1] < covThresh)) {

                        // oversample the region around the peak 2 to 1 to estimate the fractional offset
                        // write the reference image and search image around the peak into arrays

                        // Fill padRefChip and padSchWin with zeros in the fastest way
                        fill(padRefChip.begin(), padRefChip.end(), complex<float>(0.,0.));
                        fill(padSchWin.begin(), padSchWin.end(), complex<float>(0.,0.));

                        for (int yy=0; yy<refChipHeight; yy++) {
                            for (int xx=0; xx<refChipWidth; xx++) {
                                idx = (yy * padRefChipWidth) + xx;
                                if (((x + xx) >= 1) && ((x + xx) <= imgWidths[0])) padRefChip[idx] = refImg[x+xx-1][yy];
                            }
                        }
                        
                        // now the search image
                       
                        for (int yy=0; yy<peakWinHeight; yy++) {
                            for (int xx=0; xx<peakWinWidth; xx++) {
                                idx = (yy * padSchWinWidth) + xx;
                                idx2 = xScaled + xx + (peakRow * nLookAcross) - peakMargin + acrossGrossOff - schMarginX;
                                idx3 = yy + (nLookDown * peakCol) - peakMargin;
                                if ((idx2 > 0) && (idx2 < imgWidths[1]) && (idx3 >= 0) && (idx3 < schWinHeight)) {
                                    padSchWin[idx] = schImg[idx2-1][idx3];
                                }
                            }
                        }

                        // Deramp data prior to FFT

                        aMethods.derampc(padRefChip, padRefChipWidth, padRefChipHeight);
                        aMethods.derampc(padSchWin, padSchWinWidth, padSchWinHeight);
                       
                        // forward fft the data

                        numPoints[0] = padSchWinWidth;
                        numPoints[1] = padSchWinHeight;
                        fft_direction = 1;

                        aMethods.fourn2d(padSchWin, numPoints, fft_direction);
                        
                        numPoints[0] = padRefChipWidth;
                        numPoints[1] = padRefChipHeight;

                        aMethods.fourn2d(padRefChip, numPoints, fft_direction);
                      
                        // spread the spectral data out for inverse transforms

                        numPoints[0] = padRefChipWidth * 2;
                        numPoints[1] = padRefChipHeight * 2;
                        fft_direction = -1;

                        fill(osampPadRefChip.begin(), osampPadRefChip.end(), complex<float>(0.,0.));

                        for (int k=0; k<(padRefChipHeight/2); k++) {
                            for (int l=0; l<(padRefChipWidth/2); l++) {
                                idx = (k * numPoints[0]) + l;
                                idx2 = (k * padRefChipWidth) + l;
                                osampPadRefChip[idx] = padRefChip[idx2];
                                idx = ((numPoints[1] - (padRefChipHeight / 2) + k) * numPoints[0]) + l;
                                idx2 = ((k + (padRefChipHeight / 2)) * padRefChipWidth) + l;
                                osampPadRefChip[idx] = padRefChip[idx2];
                                idx = (k * numPoints[0]) + numPoints[0] - (padRefChipWidth / 2) + l;
                                idx2 = (k * padRefChipWidth) + (padRefChipWidth / 2) + l;
                                osampPadRefChip[idx] = padRefChip[idx2];
                                idx = ((numPoints[1] - (padRefChipHeight / 2) + k) * numPoints[0]) + numPoints[0] - (padRefChipWidth / 2) + l;
                                idx2 = ((k + (padRefChipHeight / 2)) * padRefChipWidth) + l + (padRefChipWidth / 2);
                                osampPadRefChip[idx] = padRefChip[idx2];
                            }
                        }
                        
                        aMethods.fourn2d(osampPadRefChip, numPoints, fft_direction);
                        
                        numPoints[0] = padSchWinWidth * 2;
                        numPoints[1] = padSchWinHeight * 2;
                        fft_direction = -1;

                        fill(osampPadSchWin.begin(), osampPadSchWin.end(), complex<float>(0.,0.));

                        for (int k=0; k<(padSchWinHeight/2); k++) {
                            for (int l=0; l<(padSchWinWidth/2); l++) {
                                idx = (k * numPoints[0]) + l;
                                idx2 = (k * padSchWinWidth) + l;
                                osampPadSchWin[idx] = padSchWin[idx2];
                                idx = ((numPoints[1] - (padSchWinHeight/2) + k) * numPoints[0]) + l;
                                idx2 = ((k + (padSchWinHeight / 2)) * padSchWinWidth) + l;
                                osampPadSchWin[idx] = padSchWin[idx2];
                                idx = (k * numPoints[0]) + numPoints[0] - (padSchWinWidth / 2) + l;
                                idx2 = (k * padSchWinWidth) + (padSchWinWidth / 2) + l;
                                osampPadSchWin[idx] = padSchWin[idx2];
                                idx = ((numPoints[1] - (padSchWinHeight / 2) + k) * numPoints[0]) + numPoints[0] - (padSchWinWidth / 2) + l;
                                idx2 = ((k + (padSchWinHeight / 2)) * padSchWinWidth) + l + (padSchWinWidth / 2);
                                osampPadSchWin[idx] = padSchWin[idx2];
                            }
                        }

                        // inverse transform

                        aMethods.fourn2d(osampPadSchWin, numPoints, fft_direction);
                        
                        // detect images and put into correlation arrays

                        for (int yy=0; yy<(refChipHeight*2); yy++) {
                            for (int xx=0; xx<(refChipWidth*2); xx++) {
                                idx = xx + (yy * padRefChipWidth * 2);
                                osampRefChip[xx][yy] = abs(osampPadRefChip[idx] / float(padRefChipWidth * padRefChipHeight));
                            }
                        }

                        for (int yy=0; yy<(peakWinHeight*2); yy++) {
                            for (int xx=0; xx<(peakWinWidth*2); xx++) {
                                idx = xx + (yy * padSchWinWidth * 2);
                                osampSchWin[xx][yy] = abs(osampPadSchWin[idx] / float(padSchWinWidth * padSchWinHeight));
                            }
                        }

                        // correlate the oversampled chips

                        osampRefChipWidth = refChipWidth * 2;
                        osampRefChipHeight = refChipHeight * 2;
                        osampSchWinWidth = peakWinWidth * 2;
                        osampSchWinHeight = peakWinHeight * 2;
                        osampCorrWidth = osampSchWinWidth - osampRefChipWidth + 1;
                        osampCorrHeight = osampSchWinHeight - osampRefChipHeight + 1;
                      
                        aMethods.correlate(osampRefChip, osampSchWin, osampRefChipWidth, osampRefChipHeight, osampSchWinWidth, osampSchWinHeight, 1,
                                            1, osampCorrPeak, osampCovs, osampCorrSurface, osampPeakRow, osampPeakCol, isEdge, corr_flag, corr_debug);
                        
                        osampAcrossOffset = (osampPeakRow / 2.) - ((osampCorrWidth - 1) / 4.) + acrossOffset;
                        osampDownOffset = (osampPeakCol / 2.) - ((osampCorrHeight - 1) / 4.) + downOffset;

                        // display the correlation surface

                        if (corr_display) {
                            printf("\nCorrelation Surface of oversampled image at %d, %d\n", (x+((refChipWidth-1)/2)), (y+((refChipHeight-1)/2)));
                            for (int l=max(osampPeakCol-3,1); l<=min(osampPeakCol+5,osampCorrHeight); l++) {
                                for (int k=max(osampPeakRow-3,1); k<=min(osampPeakRow+5,osampCorrWidth); k++) {
                                    printf("%f ", (pow(osampCorrSurface[k-1][l-1],2) / pow(osampCorrPeak,2)));
                                }
                                printf("\n");
                            }
                        }

                        // oversample the oversampled correlation surface
                        fill(corrWin.begin(), corrWin.end(), complex<float>(0.,0.));

                        for (int yy=(-zoomWinSize/2); yy<(zoomWinSize/2); yy++) {
                            for (int xx=(-zoomWinSize/2); xx<(zoomWinSize/2); xx++) {

                                idx = ((yy + (zoomWinSize / 2)) * zoomWinSize) + xx + (zoomWinSize / 2);

                                if (((xx + osampPeakRow) >= 0) && ((xx + osampPeakRow) < ((4 * peakMargin) + 2)) &&
                                    ((yy + osampPeakCol) >= 0) && ((yy + osampPeakCol) < ((4 * peakMargin) + 2))) {
                                    corrWin[idx] = complex<float>(osampCorrSurface[xx+osampPeakRow][yy+osampPeakCol]/osampCorrPeak,0.);
                                }
                            }
                        }

                        //if (bcount == 0) for (int i=0; i<30; i++) printf("%f\n", corrWin[i].real());

                        // Use SINC interpolation to oversample the correlation surface. Note will cheat and
                        // do a series of 1-d interpolations. Assume correlation function is periodic and 
                        // do a circular convolution.
                        fill(interpCorr.begin(), interpCorr.end(), complex<float>(0.,0.));

                        for (int yy=(-zoomWinSize/2); yy<(zoomWinSize/2); yy++) {
                            for (int xx=(-2*osampFact); xx<=(2*osampFact); xx++) {
                                        
                                idx = ((yy + (zoomWinSize / 2)) * osampFact * zoomWinSize) + xx + (zoomWinSize * (osampFact / 2));
                                resampFactor = (float(xx + (zoomWinSize * (osampFact / 2)) + osampFact) / osampFact) + sincDelay;
                                resampLength = int((resampFactor - int(resampFactor)) * 4096);
                                sincWeight = 0.;
                                        
                                for (int k=0; k<sincInterpLength; k++) {
                                    idx2 = ((yy + (zoomWinSize / 2)) * zoomWinSize) + int(resampFactor) - k;
                                    if ((int(resampFactor) - k) < 1) idx2 = idx2 + zoomWinSize;
                                    else if ((int(resampFactor) - k) > zoomWinSize) idx2 = idx2 - zoomWinSize;

                                    interpCorr[idx] = interpCorr[idx] + (corrWin[idx2-1] * sincInterp[k+(resampLength*sincInterpLength)]);
                                    sincWeight = sincWeight + sincInterp[k+(resampLength*sincInterpLength)];
                                }
                                interpCorr[idx] = interpCorr[idx] / sincWeight;
                            }
                        }

                        //if (bcount == 0) for (int i=32; i<52; i++) printf("%f\n", interpCorr[i].real());

                        // along track resample
                        fill(osampInterpCorr.begin(), osampInterpCorr.end(), complex<float>(0.,0.));

                        for (int yy=(-2*osampFact); yy<=(2*osampFact); yy++) {
                            for (int xx=(-2*osampFact); xx<=(2*osampFact); xx++) {
                                        
                                idx = ((yy + (zoomWinSize * (osampFact / 2))) * zoomWinSize * osampFact) + xx + (zoomWinSize * (osampFact / 2));
                                resampFactor = ((1. * (yy + (zoomWinSize * (osampFact / 2)) + osampFact)) / osampFact) + sincDelay;
                                resampLength = int((resampFactor - int(resampFactor)) * 4096);
                                sincWeight = 0.;
                                        
                                for (int k=0; k<sincInterpLength; k++) {
                                    idx2 = int(resampFactor) - k;
                                    if ((int(resampFactor) - k) < 1) idx2 = idx2 + zoomWinSize;
                                    else if ((int(resampFactor) - k) > zoomWinSize) idx2 = idx2 - zoomWinSize;
                                    
                                    idx3 = ((idx2 - 1) * zoomWinSize * osampFact) + xx + (zoomWinSize * (osampFact / 2));
                                    osampInterpCorr[idx] = osampInterpCorr[idx] + (interpCorr[idx3] * sincInterp[k+(resampLength*sincInterpLength)]);
                                    sincWeight = sincWeight + sincInterp[k+(resampLength*sincInterpLength)];
                                }
                                osampCorrWin[idx] = osampInterpCorr[idx] / sincWeight;
                            }
                        }
                        
                        // detect the peak
                        maxCorr = 0.;
                            
                        for (int yy=0; yy<(zoomWinSize*osampFact); yy++) {
                            for (int xx=0; xx<(zoomWinSize*osampFact); xx++) {
                                    
                                idx = (yy * zoomWinSize * osampFact) + xx;
                                    
                                if ((abs(xx + 1 - (zoomWinSize * (osampFact / 2))) <= osampFact) && (abs(yy + 1 - (zoomWinSize * (osampFact / 2))) <= osampFact)) {
                                    if (abs(osampCorrWin[idx]) >= maxCorr) {
                                        maxCorr = abs(osampCorrWin[idx]);
                                        corrPeaks[0] = xx - ((zoomWinSize / 2) * osampFact) + 1;
                                        corrPeaks[1] = yy - ((zoomWinSize / 2) * osampFact) + 1;
                                    }
                                }
                                    
                            }
                        }
                        
                        osampCorrOffset[0] = (corrPeaks[0] - 1.) / float(osampFact);
                        osampCorrOffset[1] = (corrPeaks[1] - 1.) / float(osampFact);
                        locationAcrossOffset = (osampCorrOffset[0] / 2) + osampAcrossOffset + xScaled - x;
                        locationDownOffset = (osampCorrOffset[1] / 2) + osampDownOffset + yScaled - y;
                        snr = min(snr, float(9999.99999));
                            
                        locationAcrossArr[mainArrIdx] = x + ((refChipWidth - 1) / 2);
                        locationDownArr[mainArrIdx] = y + ((refChipHeight - 1) / 2);
                        locationAcrossOffsetArr[mainArrIdx] = locationAcrossOffset;
                        locationDownOffsetArr[mainArrIdx] = locationDownOffset;
                        snrArr[mainArrIdx] = snr;
                        cov1Arr[mainArrIdx] = covs[0];
                        cov2Arr[mainArrIdx] = covs[1];
                        cov3Arr[mainArrIdx] = covs[2];
                           
                        mainArrIdx++;
                        
                    } else {

                        printf("Bad match at level 1\n");

                    } // thresholds
                } // not edge point or no data point
            } // Loop over width

            printf("XXX time for inner loop %.2f\n", aMethods.getInnerClock());

        } // Loop over length

        printf("Elapsed time. %.2f\n", aMethods.getOuterClock());

        numRowTable = mainArrIdx;

        delete[] rl_read_line;
        delete[] cx_read_line;

        dumpToFiles();
    } // Non-gpu ampcor
}

