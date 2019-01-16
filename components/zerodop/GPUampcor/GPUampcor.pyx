#
# Author: Joshua Cohen
# Copyright 2016
#

from libc.stdint cimport uint64_t
from libcpp cimport bool


cdef extern from "Ampcor.h":
    cdef cppclass Ampcor:
        uint64_t imgAccessor1, imgAccessor2, offImgAccessor, offQualImgAccessor
        float snrThresh, covThresh, xScaleFactor, yScaleFactor
        int imgDatatypes[2]
        int imgWidths[2]
        int imgBands[2]
        int isMag[2]
        int firstRow, lastRow, rowSkip, firstCol, lastCol, colSkip, refChipWidth, refChipHeight
        int schMarginX, schMarginY, nLookAcross, nLookDown, osampFact, zoomWinSize
        int acrossGrossOff, downGrossOff, numRowTable
        bool corr_debug, corr_display, usr_enable_gpu
        
        Ampcor() except +
        void ampcor()
        int getLocationAcrossAt(int)
        int getLocationDownAt(int)
        float getLocationAcrossOffsetAt(int)
        float getLocationDownOffsetAt(int)
        float getSnrAt(int)
        float getCov1At(int)
        float getCov2At(int)
        float getCov3At(int)


cdef class PyAmpcor:
    cdef Ampcor c_ampcor
    
    def __cinit__(self):
        return
    
    @property
    def imageBand1(self):
        return self.c_ampcor.imgBands[0]
    @imageBand1.setter
    def imageBand1(self, int a):
        self.c_ampcor.imgBands[0] = a
    @property
    def imageBand2(self):
        return self.c_ampcor.imgBands[1]
    @imageBand2.setter
    def imageBand2(self, int a):
        self.c_ampcor.imgBands[1] = a
    @property
    def imageAccessor1(self):
        return self.c_ampcor.imgAccessor1
    @imageAccessor1.setter
    def imageAccessor1(self, uint64_t a):
        self.c_ampcor.imgAccessor1 = a
    @property
    def imageAccessor2(self):
        return self.c_ampcor.imgAccessor2
    @imageAccessor2.setter
    def imageAccessor2(self, uint64_t a):
        self.c_ampcor.imgAccessor2 = a
    @property
    def offsetImageAccessor(self):
        return self.c_ampcor.offImgAccessor
    @offsetImageAccessor.setter
    def offsetImageAccessor(self, uint64_t a):
        self.c_ampcor.offImgAccessor = a
    @property
    def offsetQualImageAccessor(self):
        return self.c_ampcor.offQualImgAccessor
    @offsetQualImageAccessor.setter
    def offsetQualImageAccessor(self, uint64_t a):
        self.c_ampcor.offQualImgAccessor = a
    @property
    def thresholdSNR(self):
        return self.c_ampcor.snrThresh
    @thresholdSNR.setter
    def thresholdSNR(self, float a):
        self.c_ampcor.snrThresh = a
    @property
    def thresholdCov(self):
        return self.c_ampcor.covThresh
    @thresholdCov.setter
    def thresholdCov(self, float a):
        self.c_ampcor.covThresh = a
    @property
    def scaleFactorX(self):
        return self.c_ampcor.xScaleFactor
    @scaleFactorX.setter
    def scaleFactorX(self, float a):
        self.c_ampcor.xScaleFactor = a
    @property
    def scaleFactorY(self):
        return self.c_ampcor.yScaleFactor
    @scaleFactorY.setter
    def scaleFactorY(self, float a):
        self.c_ampcor.yScaleFactor = a
    @property
    def datatype1(self):
        dt = self.c_ampcor.imgDatatypes[0]
        mg = self.c_ampcor.isMag[0]
        if (dt + mg == 0):
            return 'real'
        elif (dt + mg == 1):
            return 'complex'
        else: # dt + mg == 2
            return 'mag'
    @datatype1.setter
    def datatype1(self, str a):
        if (a[0].lower() == 'r'):
            self.c_ampcor.isMag[0] = 0
            self.c_ampcor.imgDatatypes[0] = 0
        elif (a[0].lower() == 'c'):
            self.c_ampcor.isMag[0] = 0
            self.c_ampcor.imgDatatypes[0] = 1
        elif (a[0].lower() == 'm'):
            self.c_ampcor.isMag[0] = 1
            self.c_ampcor.imgDatatypes[0] = 1
        else:
            print("Error: Unrecognized datatype. Expected 'complex', 'real', or 'mag'.")
    @property
    def datatype2(self):
        dt = self.c_ampcor.imgDatatypes[1]
        mg = self.c_ampcor.isMag[1]
        if (dt + mg == 0):
            return 'real'
        elif (dt + mg == 1):
            return 'complex'
        else: # dt + mg == 2
            return 'mag'
    @datatype2.setter
    def datatype2(self, str a):
        if (a[0].lower() == 'r'):
            self.c_ampcor.isMag[1] = 0
            self.c_ampcor.imgDatatypes[1] = 0
        elif (a[0].lower() == 'c'):
            self.c_ampcor.isMag[1] = 0
            self.c_ampcor.imgDatatypes[1] = 1
        elif (a[0].lower() == 'm'):
            self.c_ampcor.isMag[1] = 1
            self.c_ampcor.imgDatatypes[1] = 1
        else:
            print("Error: Unrecognized datatype. Expected 'complex', 'real', or 'mag'.")
    @property
    def lineLength1(self):
        return self.c_ampcor.imgWidths[0]
    @lineLength1.setter
    def lineLength1(self, int a):
        self.c_ampcor.imgWidths[0] = a
    @property
    def lineLength2(self):
        return self.c_ampcor.imgWidths[1]
    @lineLength2.setter
    def lineLength2(self, int a):
        self.c_ampcor.imgWidths[1] = a
    @property
    def firstSampleDown(self):
        return self.c_ampcor.firstRow
    @firstSampleDown.setter
    def firstSampleDown(self, int a):
        self.c_ampcor.firstRow = a
    @property
    def lastSampleDown(self):
        return self.c_ampcor.lastRow
    @lastSampleDown.setter
    def lastSampleDown(self, int a):
        self.c_ampcor.lastRow = a
    @property
    def skipSampleDown(self):
        return self.c_ampcor.rowSkip
    @skipSampleDown.setter
    def skipSampleDown(self, int a):
        self.c_ampcor.rowSkip = a
    @property
    def firstSampleAcross(self):
        return self.c_ampcor.firstCol
    @firstSampleAcross.setter
    def firstSampleAcross(self, int a):
        self.c_ampcor.firstCol = a
    @property
    def lastSampleAcross(self):
        return self.c_ampcor.lastCol
    @lastSampleAcross.setter
    def lastSampleAcross(self, int a):
        self.c_ampcor.lastCol = a
    @property
    def skipSampleAcross(self):
        return self.c_ampcor.colSkip
    @skipSampleAcross.setter
    def skipSampleAcross(self, int a):
        self.c_ampcor.colSkip = a
    @property
    def windowSizeWidth(self):
        return self.c_ampcor.refChipWidth
    @windowSizeWidth.setter
    def windowSizeWidth(self, int a):
        self.c_ampcor.refChipWidth = a
    @property
    def windowSizeHeight(self):
        return self.c_ampcor.refChipHeight
    @windowSizeHeight.setter
    def windowSizeHeight(self, int a):
        self.c_ampcor.refChipHeight = a
    @property
    def searchWindowSizeWidth(self):
        return self.c_ampcor.schMarginX
    @searchWindowSizeWidth.setter
    def searchWindowSizeWidth(self, int a):
        self.c_ampcor.schMarginX = a
    @property
    def searchWindowSizeHeight(self):
        return self.c_ampcor.schMarginY
    @searchWindowSizeHeight.setter
    def searchWindowSizeHeight(self, int a):
        self.c_ampcor.schMarginY = a
    @property
    def acrossLooks(self):
        return self.c_ampcor.nLookAcross
    @acrossLooks.setter
    def acrossLooks(self, int a):
        self.c_ampcor.nLookAcross = a
    @property
    def downLooks(self):
        return self.c_ampcor.nLookDown
    @downLooks.setter
    def downLooks(self, int a):
        self.c_ampcor.nLookDown = a
    @property
    def oversamplingFactor(self):
        return self.c_ampcor.osampFact
    @oversamplingFactor.setter
    def oversamplingFactor(self, int a):
        self.c_ampcor.osampFact = a
    @property
    def zoomWindowSize(self):
        return self.c_ampcor.zoomWinSize
    @zoomWindowSize.setter
    def zoomWindowSize(self, int a):
        self.c_ampcor.zoomWinSize = a
    @property
    def acrossGrossOffset(self):
        return self.c_ampcor.acrossGrossOff
    @acrossGrossOffset.setter
    def acrossGrossOffset(self, int a):
        self.c_ampcor.acrossGrossOff = a
    @property
    def downGrossOffset(self):
        return self.c_ampcor.downGrossOff
    @downGrossOffset.setter
    def downGrossOffset(self, int a):
        self.c_ampcor.downGrossOff = a
    @property
    def debugFlag(self):
        return self.c_ampcor.corr_debug
    @debugFlag.setter
    def debugFlag(self, bool a):
        self.c_ampcor.corr_debug = a
    @property
    def displayFlag(self):
        return self.c_ampcor.corr_display
    @displayFlag.setter
    def displayFlag(self, bool a):
        self.c_ampcor.corr_display = a
    @property
    def usr_enable_gpu(self):
        return self.c_ampcor.usr_enable_gpu
    @usr_enable_gpu.setter
    def usr_enable_gpu(self, bool a):
        self.c_ampcor.usr_enable_gpu = a
    @property
    def numElem(self):
        return self.c_ampcor.numRowTable

    def runAmpcor(self):
        self.c_ampcor.ampcor()
    def getLocationAcrossAt(self, int idx):
        if (idx >= self.numElem):
            print("Error: Invalid element number ("+str(self.numElem)+" elements available).")
        else:
            return self.c_ampcor.getLocationAcrossAt(idx)
    def getLocationDownAt(self, int idx):
        if (idx >= self.numElem):
            print("Error: Invalid element number ("+str(self.numElem)+" elements available).")
        else:
            return self.c_ampcor.getLocationDownAt(idx)
    def getLocationAcrossOffsetAt(self, int idx):
        if (idx >= self.numElem):
            print("Error: Invalid element number ("+str(self.numElem)+" elements available).")
        else:
            return self.c_ampcor.getLocationAcrossOffsetAt(idx)
    def getLocationDownOffsetAt(self, int idx):
        if (idx >= self.numElem):
            print("Error: Invalid element number ("+str(self.numElem)+" elements available).")
        else:
            return self.c_ampcor.getLocationDownOffsetAt(idx)
    def getSNRAt(self, int idx):
        if (idx >= self.numElem):
            print("Error: Invalid element number ("+str(self.numElem)+" elements available).")
        else:
            return self.c_ampcor.getSnrAt(idx)
    def getCov1At(self, int idx):
        if (idx >= self.numElem):
            print("Error: Invalid element number ("+str(self.numElem)+" elements available).")
        else:
            return self.c_ampcor.getCov1At(idx)
    def getCov2At(self, int idx):
        if (idx >= self.numElem):
            print("Error: Invalid element number ("+str(self.numElem)+" elements available).")
        else:
            return self.c_ampcor.getCov2At(idx)
    def getCov3At(self, int idx):
        if (idx >= self.numElem):
            print("Error: Invalid element number ("+str(self.numElem)+" elements available).")
        else:
            return self.c_ampcor.getCov3At(idx)
    
