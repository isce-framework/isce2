/**
 * cuAmpcorController.h
 * Header file for the controller class (interfaces to Python/Cython)
 *
 */

#ifndef CU_AMPCOR_CONTROLLER_H
#define CU_AMPCOR_CONTROLLER_H

#include "cuAmpcorParameter.h"
#include <cstring>

class cuAmpcorController {
public:    
    cuAmpcorParameter *param;   
    cuAmpcorController(); 
    ~cuAmpcorController(); 
    void runAmpcor();
    void outputGrossOffsets();
    /*
    void setAlgorithm(int);
    int getAlgorithm();
    void setDeviceID(int);
    int getDeviceID();
    void setNStreams(int);
    int getNStreams();
    void setWindowSizeHeight(int);
    int getWindowSizeHeight();
    void setWindowSizeWidth(int);
    int getWindowSizeWidth();
    void setSearchWindowSizeHeight(int);
    int getSearchWindowSizeHeight();
    void setSearchWindowSizeWidth(int);
    int setSearchWindowSizeWidth();
    void setRawOversamplingFactor(int);
    int getRawOversamplingFactor();
    void setZoomWindowSize(int);
    int getZoomWindowSize();
    void setOversamplingFactor(int);
    int getOversamplingFactor();
    void setAcrossLooks(int);
    int getAcrossLoos();
    void setDownLooks(int);
    int getDownLooks();
    void setSkipSampleAcrossRaw(int);
    int getSkipSampleAcrossRaw();
    void setSkipSampleDownRaw(int);
    int getSkipSampleDownRaw();
    void setNumberWindowMethod(int);
    int getNumberWindowMethod();
    void setNumberWindowDown(int);
    int getNumberWindowDown();
    void setNumberWindowAcross(int);
    int getNumberWindowAcross();
    void setNumberWindowDownInChunk(int);
    int getNumberWindowDownInChunk();
    void setNumberWindowAcrossInChunk(int);
    int getNumberWindowAcrossInChunk();
    void setRangeSpacing1(float);
    float getRangeSpacing1();
    void setRangeSpacing2(float);
    float getRangeSpacing2();
    void setImageDatatype1(int);
    int getImageDatatype1();
    void setImageDatatype2(int);
    int getImageDatatype2();
    void setThresholdSNR(float);
    float getThresholdSNR();
    void setThresholdCov(float);
    float getThresholdCov();
    void setBand1(int);
    int getBand1();
    void setBand2(int);
    int getBand2();
    void setReferenceImageName(std::string);
    std::string getReferenceImageName();
    void setSecondaryImageName(std::string);
    std::string getSecondaryImageName();
    void setReferenceImageWidth(int);
    int getReferenceImageWidth();
    void setReferenceImageHeight(int);
    int getReferenceImageHeight();
    void setSecondaryImageWidth(int);
    int getSecondaryImageWidth();
    void setSecondaryImageHeight(int);
    int getSecondaryImageHeight();
    void setStartPixelMethod(int);
    int getStartPixelMethod();
    void setReferenceStartPixelAcross(int);
    int getReferenceStartPixelAcross();
    void setReferenceStartPixelDown(int);
    int setReferenceStartPixelDown();
    void setSecondaryStartPixelAcross(int);
    int getSecondaryStartPixelAcross();
    void setSecondaryStartPixelDown(int);
    int getSecondaryStartPixelDown();
    void setGrossOffsetMethod(int);
    int getGrossOffsetMethod();
    void setAcrossGrossOffset(int);
    int getAcrossGrossOffset();
    void setDownGrossOffset(int);
    int getDownGrossOffset();
    void setGrossOffsets(int *, int);
    int* getGrossOffsets();
    void setOffsetImageName(std::string);
    std::string getOffsetImageName();
    void setSNRImageName(std::string);
    std::string getSNRImageName(); 
    //void setMargin(int);
    //int getMargin();
    //void setNumberThreads(int);
    void setDerampMethod(int);
    int getDerampMethod();*/
};
#endif
