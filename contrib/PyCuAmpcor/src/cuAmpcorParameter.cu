/**
 * cuAmpcorParameter.cu
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
    zoomWindowSize = 8;
    oversamplingFactor = 16;
    oversamplingMethod = 0;

    masterImageName = "master.slc";
    masterImageWidth = 1000;
    masterImageHeight = 1000;
    slaveImageName = "slave.slc";
    slaveImageWidth = 1000;
    slaveImageHeight = 1000;
    offsetImageName = "DenseOffset.off";
    grossOffsetImageName = "GrossOffset.off";
    snrImageName = "snr.snr";
    covImageName = "cov.cov";
    numberWindowDown =  1;
    numberWindowAcross = 1;
    numberWindowDownInChunk = 1;
    numberWindowAcrossInChunk = 1 ;

    masterStartPixelDown0 = 0;
    masterStartPixelAcross0 = 0;

    corrRawZoomInHeight = 17; // 8*2+1
    corrRawZoomInWidth = 17;

    useMmap = 1; // use mmap
    mmapSizeInGB = 1;

}

/**
 * To determine other process parameters after reading essential parameters from python
 */

void cuAmpcorParameter::setupParameters()
{
    zoomWindowSize *= rawDataOversamplingFactor; //8 * 2
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

    // modified 02/12/2018 to include one more chunk
    // e.g. numberWindowDownInChunk=102, numberWindowDown=10, results in numberChunkDown=11
    // the last chunk will include 2 windows, numberWindowDownInChunkRun = 2.

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
    masterStartPixelDown = (int *)malloc(arraySize);
    masterStartPixelAcross =  (int *)malloc(arraySize);
    slaveStartPixelDown = (int *)malloc(arraySize);
    slaveStartPixelAcross =  (int *)malloc(arraySize);

    int arraySizeChunk = numberChunks*sizeof(int);
    masterChunkStartPixelDown = (int *)malloc(arraySizeChunk);
    masterChunkStartPixelAcross = (int *)malloc(arraySizeChunk);
    slaveChunkStartPixelDown = (int *)malloc(arraySizeChunk);
    slaveChunkStartPixelAcross = (int *)malloc(arraySizeChunk);
    masterChunkHeight = (int *)malloc(arraySizeChunk);
    masterChunkWidth = (int *)malloc(arraySizeChunk);
    slaveChunkHeight = (int *)malloc(arraySizeChunk);
    slaveChunkWidth = (int *)malloc(arraySizeChunk);
}

void cuAmpcorParameter::deallocateArrays()
{
    free(grossOffsetDown);
    free(grossOffsetAcross);
    free(masterStartPixelDown);
    free(masterStartPixelAcross);
    free(slaveStartPixelDown);
    free(slaveStartPixelAcross);
    free(masterChunkStartPixelDown);
    free(masterChunkStartPixelAcross);
    free(slaveChunkStartPixelDown);
    free(slaveChunkStartPixelAcross);
    free(masterChunkHeight);
    free(masterChunkWidth);
    free(slaveChunkHeight);
    free(slaveChunkWidth);
}


/// Set starting pixels for master and slave windows from arrays
/// set also gross offsets between master and slave windows
///
void cuAmpcorParameter::setStartPixels(int *mStartD, int *mStartA, int *gOffsetD, int *gOffsetA)
{
    for(int i=0; i<numberWindows; i++)
    {
		masterStartPixelDown[i] = mStartD[i];
		grossOffsetDown[i] = gOffsetD[i];
		slaveStartPixelDown[i] = masterStartPixelDown[i] + grossOffsetDown[i] - halfSearchRangeDownRaw;
		masterStartPixelAcross[i] = mStartA[i];
		grossOffsetAcross[i] = gOffsetA[i];
		slaveStartPixelAcross[i] = masterStartPixelAcross[i] + grossOffsetAcross[i] - halfSearchRangeAcrossRaw;
    }
    setChunkStartPixels();
}

void cuAmpcorParameter::setStartPixels(int mStartD, int mStartA, int *gOffsetD, int *gOffsetA)
{
    for(int row=0; row<numberWindowDown; row++)
    {
		for(int col = 0; col < numberWindowAcross; col++)
		{
			int i = row*numberWindowAcross + col;
			masterStartPixelDown[i] = mStartD + row*skipSampleDownRaw;
			grossOffsetDown[i] = gOffsetD[i];
			slaveStartPixelDown[i] = masterStartPixelDown[i] + grossOffsetDown[i] - halfSearchRangeDownRaw;
			masterStartPixelAcross[i] = mStartA + col*skipSampleAcrossRaw;
			grossOffsetAcross[i] = gOffsetA[i];
			slaveStartPixelAcross[i] = masterStartPixelAcross[i] + grossOffsetAcross[i] - halfSearchRangeAcrossRaw;
		}
    }
    setChunkStartPixels();
}

void cuAmpcorParameter::setStartPixels(int mStartD, int mStartA, int gOffsetD, int gOffsetA)
{
    //fprintf(stderr, "set start pixels %d %d %d %d\n", mStartD, mStartA, gOffsetD, gOffsetA);
    for(int row=0; row<numberWindowDown; row++)
    {
		for(int col = 0; col < numberWindowAcross; col++)
		{
			int i = row*numberWindowAcross + col;
			masterStartPixelDown[i] = mStartD + row*skipSampleDownRaw;
			grossOffsetDown[i] = gOffsetD;
			slaveStartPixelDown[i] = masterStartPixelDown[i] + grossOffsetDown[i] - halfSearchRangeDownRaw;
			masterStartPixelAcross[i] = mStartA + col*skipSampleAcrossRaw;
			grossOffsetAcross[i] = gOffsetA;
			slaveStartPixelAcross[i] = masterStartPixelAcross[i] + grossOffsetAcross[i] - halfSearchRangeAcrossRaw;
		}
    }
    setChunkStartPixels();
}

void cuAmpcorParameter::setChunkStartPixels()
{

    maxMasterChunkHeight = 0;
    maxMasterChunkWidth = 0;
    maxSlaveChunkHeight = 0;
    maxSlaveChunkWidth = 0;

    for(int ichunk=0; ichunk <numberChunkDown; ichunk++)
    {
        for (int jchunk =0; jchunk<numberChunkAcross; jchunk++)
        {

            int idxChunk = ichunk*numberChunkAcross+jchunk;
            int mChunkSD = masterImageHeight;
            int mChunkSA = masterImageWidth;
            int mChunkED = 0;
            int mChunkEA = 0;
            int sChunkSD = slaveImageHeight;
            int sChunkSA = slaveImageWidth;
            int sChunkED = 0;
            int sChunkEA = 0;

            // modified 02/12/2018
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
                    int vpixel = masterStartPixelDown[idxWindow];
                    if(mChunkSD > vpixel) mChunkSD = vpixel;
                    if(mChunkED < vpixel) mChunkED = vpixel;
                    vpixel = masterStartPixelAcross[idxWindow];
                    if(mChunkSA > vpixel) mChunkSA = vpixel;
                    if(mChunkEA < vpixel) mChunkEA = vpixel;
                    vpixel = slaveStartPixelDown[idxWindow];
                    if(sChunkSD > vpixel) sChunkSD = vpixel;
                    if(sChunkED < vpixel) sChunkED = vpixel;
                    vpixel = slaveStartPixelAcross[idxWindow];
                    if(sChunkSA > vpixel) sChunkSA = vpixel;
                    if(sChunkEA < vpixel) sChunkEA = vpixel;
                }
            }
            masterChunkStartPixelDown[idxChunk]   = mChunkSD;
            masterChunkStartPixelAcross[idxChunk] = mChunkSA;
            slaveChunkStartPixelDown[idxChunk]    = sChunkSD;
            slaveChunkStartPixelAcross[idxChunk]  = sChunkSA;
            masterChunkHeight[idxChunk] = mChunkED - mChunkSD + windowSizeHeightRaw;
            masterChunkWidth[idxChunk]  = mChunkEA - mChunkSA + windowSizeWidthRaw;
            slaveChunkHeight[idxChunk]  = sChunkED - sChunkSD + searchWindowSizeHeightRaw;
            slaveChunkWidth[idxChunk]   = sChunkEA - sChunkSA + searchWindowSizeWidthRaw;
            if(maxMasterChunkHeight < masterChunkHeight[idxChunk]) maxMasterChunkHeight = masterChunkHeight[idxChunk];
            if(maxMasterChunkWidth  < masterChunkWidth[idxChunk] ) maxMasterChunkWidth  = masterChunkWidth[idxChunk];
            if(maxSlaveChunkHeight  < slaveChunkHeight[idxChunk]) maxSlaveChunkHeight = slaveChunkHeight[idxChunk];
            if(maxSlaveChunkWidth   < slaveChunkWidth[idxChunk] ) maxSlaveChunkWidth  = slaveChunkWidth[idxChunk];
        }
    }
}

/// check whether master and slave windows are within the image range
void cuAmpcorParameter::checkPixelInImageRange()
{
	int endPixel;
	for(int row=0; row<numberWindowDown; row++)
    {
		for(int col = 0; col < numberWindowAcross; col++)
		{
			int i = row*numberWindowAcross + col;
			if(masterStartPixelDown[i] <0)
			{
				fprintf(stderr, "Master Window start pixel out ot range in Down, window (%d,%d), pixel %d\n", row, col, masterStartPixelDown[i]);
				exit(EXIT_FAILURE); //or raise range error
			}
			if(masterStartPixelAcross[i] <0)
			{
				fprintf(stderr, "Master Window start pixel out ot range in Across, window (%d,%d), pixel %d\n", row, col, masterStartPixelAcross[i]);
				exit(EXIT_FAILURE);
			}
			endPixel = masterStartPixelDown[i] + windowSizeHeightRaw;
			if(endPixel >= masterImageHeight)
			{
				fprintf(stderr, "Master Window end pixel out ot range in Down, window (%d,%d), pixel %d\n", row, col, endPixel);
				exit(EXIT_FAILURE);
			}
			endPixel = masterStartPixelAcross[i] + windowSizeWidthRaw;
			if(endPixel >= masterImageWidth)
			{
				fprintf(stderr, "Master Window end pixel out ot range in Across, window (%d,%d), pixel %d\n", row, col, endPixel);
				exit(EXIT_FAILURE);
			}
			//slave
			if(slaveStartPixelDown[i] <0)
			{
				fprintf(stderr, "Slave Window start pixel out ot range in Down, window (%d,%d), pixel %d\n", row, col, slaveStartPixelDown[i]);
				exit(EXIT_FAILURE);
			}
			if(slaveStartPixelAcross[i] <0)
			{
				fprintf(stderr, "Slave Window start pixel out ot range in Across, window (%d,%d), pixel %d\n", row, col, slaveStartPixelAcross[i]);
				exit(EXIT_FAILURE);
			}
			endPixel = slaveStartPixelDown[i] + searchWindowSizeHeightRaw;
			if(endPixel >= slaveImageHeight)
			{
				fprintf(stderr, "Slave Window end pixel out ot range in Down, window (%d,%d), pixel %d\n", row, col, endPixel);
				exit(EXIT_FAILURE);
			}
			endPixel = slaveStartPixelAcross[i] + searchWindowSizeWidthRaw;
			if(endPixel >= slaveImageWidth)
			{
				fprintf(stderr, "Slave Window end pixel out ot range in Across, window (%d,%d), pixel %d\n", row, col, endPixel);
				exit(EXIT_FAILURE);
			}

		}
    }
}


cuAmpcorParameter::~cuAmpcorParameter()
{
	deallocateArrays();
}
