/**
 * @file cuAmpcorParameter.cu
 * Input parameters for ampcor
 */

#include "cuAmpcorParameter.h"
#include <stdio.h>

#ifndef IDIVUP
#define IDIVUP(i,j) ((i+j-1)/j)
#endif

///
/// Constructor for cuAmpcorParameter class
/// also sets the default/initial values of various parameters
///

cuAmpcorParameter::cuAmpcorParameter()
{
    // default settings
    // will be changed if they are set by python scripts
    algorithm = 0; //0 freq; 1 time
    deviceID = 0;
    nStreams = 1;
    derampMethod = 1;

    windowSizeWidthRaw = 64;
    windowSizeHeightRaw = 64;
    halfSearchRangeDownRaw = 20;
    halfSearchRangeAcrossRaw = 20;

    skipSampleAcrossRaw = 64;
    skipSampleDownRaw = 64;
    rawDataOversamplingFactor = 2;
    zoomWindowSize = 16;
    oversamplingFactor = 16;
    oversamplingMethod = 0;

    referenceImageName = "reference.slc";
    referenceImageWidth = 1000;
    referenceImageHeight = 1000;
    secondaryImageName = "secondary.slc";
    secondaryImageWidth = 1000;
    secondaryImageHeight = 1000;
    offsetImageName = "DenseOffset.off";
    grossOffsetImageName = "GrossOffset.off";
    snrImageName = "snr.snr";
    covImageName = "cov.cov";
    numberWindowDown =  1;
    numberWindowAcross = 1;
    numberWindowDownInChunk = 1;
    numberWindowAcrossInChunk = 1 ;

    referenceStartPixelDown0 = 0;
    referenceStartPixelAcross0 = 0;

    corrStatWindowSize = 21; // 10*2+1 as in RIOPAC

    useMmap = 1; // use mmap
    mmapSizeInGB = 1;

    mergeGrossOffset = 0; // default to separate gross offset

}

/**
 * To determine other process parameters after reading essential parameters from python
 */

void cuAmpcorParameter::setupParameters()
{
    // Size to extract the raw correlation surface for snr/cov
    corrRawZoomInHeight = std::min(corrStatWindowSize, 2*halfSearchRangeDownRaw+1);
    corrRawZoomInWidth = std::min(corrStatWindowSize, 2*halfSearchRangeAcrossRaw+1);

    // Size to extract the resampled correlation surface for oversampling
    // users should use 16 for zoomWindowSize, no need to multiply by 2
    // zoomWindowSize *= rawDataOversamplingFactor; //8 * 2
    // to check the search range
    int corrSurfaceActualSize =
        std::min(halfSearchRangeAcrossRaw, halfSearchRangeDownRaw)*
        2*rawDataOversamplingFactor;
    zoomWindowSize = std::min(zoomWindowSize, corrSurfaceActualSize);

    halfZoomWindowSizeRaw = zoomWindowSize/(2*rawDataOversamplingFactor); // 8*2/(2*2) = 4

    windowSizeWidth = windowSizeWidthRaw*rawDataOversamplingFactor;  //
    windowSizeHeight = windowSizeHeightRaw*rawDataOversamplingFactor;

    searchWindowSizeWidthRaw =  windowSizeWidthRaw + 2*halfSearchRangeDownRaw;
    searchWindowSizeHeightRaw = windowSizeHeightRaw + 2*halfSearchRangeAcrossRaw;

    searchWindowSizeWidthRawZoomIn = windowSizeWidthRaw + 2*halfZoomWindowSizeRaw;
    searchWindowSizeHeightRawZoomIn = windowSizeHeightRaw + 2*halfZoomWindowSizeRaw;

    searchWindowSizeWidth = searchWindowSizeWidthRawZoomIn*rawDataOversamplingFactor;
    searchWindowSizeHeight = searchWindowSizeHeightRawZoomIn*rawDataOversamplingFactor;

    numberWindows = numberWindowDown*numberWindowAcross;
    if(numberWindows <=0) {
        fprintf(stderr, "Incorrect number of windows! (%d, %d)\n", numberWindowDown, numberWindowAcross);
        exit(EXIT_FAILURE);
    }

    numberChunkDown = IDIVUP(numberWindowDown, numberWindowDownInChunk);
    numberChunkAcross = IDIVUP(numberWindowAcross, numberWindowAcrossInChunk);
    numberChunks = numberChunkDown*numberChunkAcross;
    allocateArrays();
}


void cuAmpcorParameter::allocateArrays()
{
    int arraySize = numberWindows*sizeof(int);
    grossOffsetDown = (int *)malloc(arraySize);
    grossOffsetAcross = (int *)malloc(arraySize);
    referenceStartPixelDown = (int *)malloc(arraySize);
    referenceStartPixelAcross =  (int *)malloc(arraySize);
    secondaryStartPixelDown = (int *)malloc(arraySize);
    secondaryStartPixelAcross =  (int *)malloc(arraySize);

    int arraySizeChunk = numberChunks*sizeof(int);
    referenceChunkStartPixelDown = (int *)malloc(arraySizeChunk);
    referenceChunkStartPixelAcross = (int *)malloc(arraySizeChunk);
    secondaryChunkStartPixelDown = (int *)malloc(arraySizeChunk);
    secondaryChunkStartPixelAcross = (int *)malloc(arraySizeChunk);
    referenceChunkHeight = (int *)malloc(arraySizeChunk);
    referenceChunkWidth = (int *)malloc(arraySizeChunk);
    secondaryChunkHeight = (int *)malloc(arraySizeChunk);
    secondaryChunkWidth = (int *)malloc(arraySizeChunk);
}

void cuAmpcorParameter::deallocateArrays()
{
    free(grossOffsetDown);
    free(grossOffsetAcross);
    free(referenceStartPixelDown);
    free(referenceStartPixelAcross);
    free(secondaryStartPixelDown);
    free(secondaryStartPixelAcross);
    free(referenceChunkStartPixelDown);
    free(referenceChunkStartPixelAcross);
    free(secondaryChunkStartPixelDown);
    free(secondaryChunkStartPixelAcross);
    free(referenceChunkHeight);
    free(referenceChunkWidth);
    free(secondaryChunkHeight);
    free(secondaryChunkWidth);
}


/// Set starting pixels for reference and secondary windows from arrays
/// set also gross offsets between reference and secondary windows
///
void cuAmpcorParameter::setStartPixels(int *mStartD, int *mStartA, int *gOffsetD, int *gOffsetA)
{
    for(int i=0; i<numberWindows; i++)
    {
        referenceStartPixelDown[i] = mStartD[i];
        grossOffsetDown[i] = gOffsetD[i];
        secondaryStartPixelDown[i] = referenceStartPixelDown[i] + grossOffsetDown[i] - halfSearchRangeDownRaw;
        referenceStartPixelAcross[i] = mStartA[i];
        grossOffsetAcross[i] = gOffsetA[i];
        secondaryStartPixelAcross[i] = referenceStartPixelAcross[i] + grossOffsetAcross[i] - halfSearchRangeAcrossRaw;
    }
    setChunkStartPixels();
}

/// set starting pixels for each window with a varying gross offset
void cuAmpcorParameter::setStartPixels(int mStartD, int mStartA, int *gOffsetD, int *gOffsetA)
{
    for(int row=0; row<numberWindowDown; row++)
    {
        for(int col = 0; col < numberWindowAcross; col++)
        {
            int i = row*numberWindowAcross + col;
            referenceStartPixelDown[i] = mStartD + row*skipSampleDownRaw;
            grossOffsetDown[i] = gOffsetD[i];
            secondaryStartPixelDown[i] = referenceStartPixelDown[i] + grossOffsetDown[i] - halfSearchRangeDownRaw;
            referenceStartPixelAcross[i] = mStartA + col*skipSampleAcrossRaw;
            grossOffsetAcross[i] = gOffsetA[i];
            secondaryStartPixelAcross[i] = referenceStartPixelAcross[i] + grossOffsetAcross[i] - halfSearchRangeAcrossRaw;
        }
    }
    setChunkStartPixels();
}

/// set starting pixels for each window with a constant gross offset
void cuAmpcorParameter::setStartPixels(int mStartD, int mStartA, int gOffsetD, int gOffsetA)
{
    for(int row=0; row<numberWindowDown; row++)
    {
        for(int col = 0; col < numberWindowAcross; col++)
        {
            int i = row*numberWindowAcross + col;
            referenceStartPixelDown[i] = mStartD + row*skipSampleDownRaw;
            grossOffsetDown[i] = gOffsetD;
            secondaryStartPixelDown[i] = referenceStartPixelDown[i] + grossOffsetDown[i] - halfSearchRangeDownRaw;
            referenceStartPixelAcross[i] = mStartA + col*skipSampleAcrossRaw;
            grossOffsetAcross[i] = gOffsetA;
            secondaryStartPixelAcross[i] = referenceStartPixelAcross[i] + grossOffsetAcross[i] - halfSearchRangeAcrossRaw;
        }
    }
    setChunkStartPixels();
}

/// set starting pixels for each chunk
void cuAmpcorParameter::setChunkStartPixels()
{

    maxReferenceChunkHeight = 0;
    maxReferenceChunkWidth = 0;
    maxSecondaryChunkHeight = 0;
    maxSecondaryChunkWidth = 0;

    for(int ichunk=0; ichunk <numberChunkDown; ichunk++)
    {
        for (int jchunk =0; jchunk<numberChunkAcross; jchunk++)
        {

            int idxChunk = ichunk*numberChunkAcross+jchunk;
            int mChunkSD = referenceImageHeight;
            int mChunkSA = referenceImageWidth;
            int mChunkED = 0;
            int mChunkEA = 0;
            int sChunkSD = secondaryImageHeight;
            int sChunkSA = secondaryImageWidth;
            int sChunkED = 0;
            int sChunkEA = 0;

            int numberWindowDownInChunkRun = numberWindowDownInChunk;
            int numberWindowAcrossInChunkRun = numberWindowAcrossInChunk;
            // modify the number of windows in last chunk
            if(ichunk == numberChunkDown -1)
                numberWindowDownInChunkRun = numberWindowDown - numberWindowDownInChunk*(numberChunkDown -1);
            if(jchunk == numberChunkAcross -1)
                numberWindowAcrossInChunkRun = numberWindowAcross - numberWindowAcrossInChunk*(numberChunkAcross -1);

            for(int i=0; i<numberWindowDownInChunkRun; i++)
            {
                for(int j=0; j<numberWindowAcrossInChunkRun; j++)
                {
                    int idxWindow = (ichunk*numberWindowDownInChunk+i)*numberWindowAcross + (jchunk*numberWindowAcrossInChunk+j);
                    int vpixel = referenceStartPixelDown[idxWindow];
                    if(mChunkSD > vpixel) mChunkSD = vpixel;
                    if(mChunkED < vpixel) mChunkED = vpixel;
                    vpixel = referenceStartPixelAcross[idxWindow];
                    if(mChunkSA > vpixel) mChunkSA = vpixel;
                    if(mChunkEA < vpixel) mChunkEA = vpixel;
                    vpixel = secondaryStartPixelDown[idxWindow];
                    if(sChunkSD > vpixel) sChunkSD = vpixel;
                    if(sChunkED < vpixel) sChunkED = vpixel;
                    vpixel = secondaryStartPixelAcross[idxWindow];
                    if(sChunkSA > vpixel) sChunkSA = vpixel;
                    if(sChunkEA < vpixel) sChunkEA = vpixel;
                }
            }
            referenceChunkStartPixelDown[idxChunk]   = mChunkSD;
            referenceChunkStartPixelAcross[idxChunk] = mChunkSA;
            secondaryChunkStartPixelDown[idxChunk]    = sChunkSD;
            secondaryChunkStartPixelAcross[idxChunk]  = sChunkSA;
            referenceChunkHeight[idxChunk] = mChunkED - mChunkSD + windowSizeHeightRaw;
            referenceChunkWidth[idxChunk]  = mChunkEA - mChunkSA + windowSizeWidthRaw;
            secondaryChunkHeight[idxChunk]  = sChunkED - sChunkSD + searchWindowSizeHeightRaw;
            secondaryChunkWidth[idxChunk]   = sChunkEA - sChunkSA + searchWindowSizeWidthRaw;
            if(maxReferenceChunkHeight < referenceChunkHeight[idxChunk]) maxReferenceChunkHeight = referenceChunkHeight[idxChunk];
            if(maxReferenceChunkWidth  < referenceChunkWidth[idxChunk] ) maxReferenceChunkWidth  = referenceChunkWidth[idxChunk];
            if(maxSecondaryChunkHeight  < secondaryChunkHeight[idxChunk]) maxSecondaryChunkHeight = secondaryChunkHeight[idxChunk];
            if(maxSecondaryChunkWidth   < secondaryChunkWidth[idxChunk] ) maxSecondaryChunkWidth  = secondaryChunkWidth[idxChunk];
        }
    }
}

/// check whether reference and secondary windows are within the image range
void cuAmpcorParameter::checkPixelInImageRange()
{
    int endPixel;
    for(int row=0; row<numberWindowDown; row++)
    {
        for(int col = 0; col < numberWindowAcross; col++)
        {
            int i = row*numberWindowAcross + col;
            if(referenceStartPixelDown[i] <0)
            {
                fprintf(stderr, "Reference Window start pixel out ot range in Down, window (%d,%d), pixel %d\n", row, col, referenceStartPixelDown[i]);
                exit(EXIT_FAILURE); //or raise range error
            }
            if(referenceStartPixelAcross[i] <0)
            {
                fprintf(stderr, "Reference Window start pixel out ot range in Across, window (%d,%d), pixel %d\n", row, col, referenceStartPixelAcross[i]);
                exit(EXIT_FAILURE);
            }
            endPixel = referenceStartPixelDown[i] + windowSizeHeightRaw;
            if(endPixel >= referenceImageHeight)
            {
                fprintf(stderr, "Reference Window end pixel out ot range in Down, window (%d,%d), pixel %d\n", row, col, endPixel);
                exit(EXIT_FAILURE);
            }
            endPixel = referenceStartPixelAcross[i] + windowSizeWidthRaw;
            if(endPixel >= referenceImageWidth)
            {
                fprintf(stderr, "Reference Window end pixel out ot range in Across, window (%d,%d), pixel %d\n", row, col, endPixel);
                exit(EXIT_FAILURE);
            }
            //secondary
            if(secondaryStartPixelDown[i] <0)
            {
                fprintf(stderr, "Secondary Window start pixel out ot range in Down, window (%d,%d), pixel %d\n", row, col, secondaryStartPixelDown[i]);
                exit(EXIT_FAILURE);
            }
            if(secondaryStartPixelAcross[i] <0)
            {
                fprintf(stderr, "Secondary Window start pixel out ot range in Across, window (%d,%d), pixel %d\n", row, col, secondaryStartPixelAcross[i]);
                exit(EXIT_FAILURE);
            }
            endPixel = secondaryStartPixelDown[i] + searchWindowSizeHeightRaw;
            if(endPixel >= secondaryImageHeight)
            {
                fprintf(stderr, "Secondary Window end pixel out ot range in Down, window (%d,%d), pixel %d\n", row, col, endPixel);
                exit(EXIT_FAILURE);
            }
            endPixel = secondaryStartPixelAcross[i] + searchWindowSizeWidthRaw;
            if(endPixel >= secondaryImageWidth)
            {
                fprintf(stderr, "Secondary Window end pixel out ot range in Across, window (%d,%d), pixel %d\n", row, col, endPixel);
                exit(EXIT_FAILURE);
            }

        }
    }
}


cuAmpcorParameter::~cuAmpcorParameter()
{
    deallocateArrays();
}
// end of file
