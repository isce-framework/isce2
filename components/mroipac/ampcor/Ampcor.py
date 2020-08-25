#!/usr/bin/env python3 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# copyright: 2012 to the present, california institute of technology.
# all rights reserved. united states government sponsorship acknowledged.
# any commercial use must be negotiated with the office of technology transfer
# at the california institute of technology.
# 
# this software may be subject to u.s. export control laws. by accepting this
# software, the user agrees to comply with all applicable u.s. export laws and
# regulations. user has the responsibility to obtain export licenses,  or other
# export authority as may be required before exporting such information to
# foreign countries or providing access to foreign persons.
# 
# installation and use of this software is restricted by a license agreement
# between the licensee and the california institute of technology. it is the
# user's responsibility to abide by the terms of the license agreement.
#
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





from __future__ import print_function
import sys
import os
import math
from isceobj.Location.Offset import OffsetField,Offset
from iscesys.Component.Component import Component,Port
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from mroipac.ampcor import ampcor
from isceobj.Util.mathModule import is_power2
#from isceobj.Util.decorators import use_api

WINDOW_SIZE_WIDTH = Component.Parameter('windowSizeWidth',
        public_name='WINDOW_SIZE_WIDTH',
        default = 64,
        type = int,
        mandatory = False,
        doc = 'Width of the reference data window to be used for correlation')

WINDOW_SIZE_HEIGHT = Component.Parameter('windowSizeHeight',
        public_name='WINDOW_SIZE_HEIGHT',
        default = 64,
        type = int,
        mandatory = False,
        doc = 'Height of the reference data window to be used for correlation')

SEARCH_WINDOW_SIZE_WIDTH = Component.Parameter('searchWindowSizeWidth',
        public_name='SEARCH_WINDOW_SIZE_WIDTH',
        default = 100,
        type = int,
        mandatory = False,
        doc = 'Width of the search data window to be used for correlation')

SEARCH_WINDOW_SIZE_HEIGHT = Component.Parameter('searchWindowSizeHeight',
        public_name='SEARCH_WINDOW_SIZE_HEIGHT',
        default = 100,
        type = int,
        mandatory = False,
        doc = 'Height of the search data window to be used for correlation')

ZOOM_WINDOW_SIZE = Component.Parameter('zoomWindowSize',
        public_name = 'ZOOM_WINDOW_SIZE',
        default = 8,
        type = int,
        mandatory = False,
        doc = 'Zoom window around the local maximum for first pass')

OVERSAMPLING_FACTOR = Component.Parameter('oversamplingFactor',
        public_name = 'OVERSAMPLING_FACTOR',
        default = 16,
        type = int,
        mandatory = False,
        doc = 'Oversampling factor for the FFTs to get sub-pixel shift.')

ACROSS_GROSS_OFFSET = Component.Parameter('acrossGrossOffset',
        public_name = 'ACROSS_GROSS_OFFSET',
        default = None,
        type = int,
        mandatory = False,
        doc = 'Gross offset in the range direction.')

DOWN_GROSS_OFFSET = Component.Parameter('downGrossOffset',
        public_name = 'DOWN_GROSS_OFFSET',
        default = None,
        type = int,
        mandatory = False,
        doc = 'Gross offset in the azimuth direction.')

ACROSS_LOOKS = Component.Parameter('acrossLooks',
        public_name = 'ACROSS_LOOKS',
        default = 1,
        type = int,
        mandatory = False,
        doc = 'Number of looks to take in range before correlation')

DOWN_LOOKS = Component.Parameter('downLooks',
        public_name = 'DOWN_LOOKS',
        default = 1,
        type = int,
        mandatory = False,
        doc = 'Number of looks to take in azimuth before correlation')

NUMBER_WINDOWS_ACROSS = Component.Parameter('numberLocationAcross',
        public_name = 'NUMBER_WINDOWS_ACROSS',
        default = 40,
        type = int,
        mandatory = False,
        doc = 'Number of windows in range direction')

NUMBER_WINDOWS_DOWN = Component.Parameter('numberLocationDown',
        public_name = 'NUMBER_WINDOWS_DOWN',
        default = 40,
        type = int,
        mandatory = False,
        doc = 'Number of windows in azimuth direction')

SKIP_SAMPLE_ACROSS = Component.Parameter('skipSampleAcross',
        public_name = 'SKIP_SAMPLE_ACROSS',
        default = None,
        type = int,
        mandatory = False,
        doc = 'Number of samples to skip between windows in range direction.')

SKIP_SAMPLE_DOWN = Component.Parameter('skipSampleDown',
        public_name = 'SKIP_SAMPLE_DOWN',
        default = None,
        type = int,
        mandatory=False,
        doc = 'Number of lines to skip between windows in azimuth direction.')

DOWN_SPACING_PRF1 = Component.Parameter('prf1',
        public_name = 'DOWN_SPACING_PRF1',
        default = 1.0,
        type = float,
        mandatory = False,
        doc = 'PRF or a similar scale factor for azimuth spacing of reference image.')

DOWN_SPACING_PRF2 = Component.Parameter('prf2',
        public_name = 'DOWN_SPACING_PRF2',
        default = 1.0,
        type = float, 
        mandatory = False,
        doc = 'PRF or a similar scale factor for azimuth spacing of search image.')

ACROSS_SPACING1 = Component.Parameter('rangeSpacing1',
        public_name = 'ACROSS_SPACING1',
        default = 1.0,
        type = float,
        mandatory = False,
        doc = 'Range pixel spacing or similar scale factor for reference image.')

ACROSS_SPACING2 = Component.Parameter('rangeSpacing2',
        public_name = 'ACROSS_SPACING2',
        default = 1.0,
        type = float,
        mandatory = False,
        doc = 'Range pixel spacing or similar scale for search image.')

FIRST_SAMPLE_ACROSS = Component.Parameter('firstSampleAcross',
        public_name = 'FIRST_SAMPLE_ACROSS',
        default = None,
        type = int,
        mandatory=False,
        doc = 'Position of first window in range.')

LAST_SAMPLE_ACROSS = Component.Parameter('lastSampleAcross',
        public_name='LAST_SAMPLE_ACROSS',
        default=None,
        type=int,
        mandatory=False,
        doc = 'Position of last window in range.')

FIRST_SAMPLE_DOWN = Component.Parameter('firstSampleDown',
        public_name = 'FIRST_SAMPLE_DOWN',
        default = None,
        type = int,
        mandatory=False,
        doc = 'Position of first window in azimuth.')

LAST_SAMPLE_DOWN = Component.Parameter('lastSampleDown',
        public_name = 'LAST_SAMPLE_DOWN',
        default = None,
        type = int,
        mandatory=False,
        doc = 'Position of last window in azimuth.')


IMAGE_DATATYPE1 = Component.Parameter('imageDataType1',
        public_name = 'IMAGE_DATATYPE1',
        default='',
        type = str,
        mandatory = False,
        doc = 'Image data type for reference image (complex / real / mag)')

IMAGE_DATATYPE2 = Component.Parameter('imageDataType2',
        default='',
        type = str,
        mandatory=False,
        doc = 'Image data type for search image (complex / real/ mag)')


SNR_THRESHOLD = Component.Parameter('thresholdSNR',
        public_name = 'SNR_THRESHOLD',
        default = 0.001,
        type = float,
        mandatory=False,
        doc = 'SNR threshold for valid matches.')

COV_THRESHOLD = Component.Parameter('thresholdCov',
        public_name = 'COV_THRESHOLD',
        default = 1000.0,
        type = float,
        mandatory=False,
        doc = 'Covariance threshold for valid matches.')

BAND1 = Component.Parameter('band1',
        public_name='BAND1',
        default=0,
        type = int,
        mandatory = False,
        doc = 'Band number of image1')

BAND2 = Component.Parameter('band2',
        public_name='BAND2',
        default=0,
        type=int,
        mandatory=False,
        doc = 'Band number of image2')

MARGIN = Component.Parameter('margin',
        public_name='MARGIN',
        default=50,
        type=int,
        mandatory=False,
        doc ='Margin around the image to avoid.')


DEBUG_FLAG = Component.Parameter('debugFlag',
        public_name = 'DEBUG_FLAG',
        default = False,
        type = bool,
        doc = 'Dump debug files.')

DISPLAY_FLAG = Component.Parameter('displayFlag',
        public_name = 'DISPLAY_FLAG',
        default = False,
        type = bool,
        doc  = 'Display debugging information.')

class Ampcor(Component):

    family = 'ampcor'
    logging_name = 'isce.mroipac.ampcor'

    parameter_list = (WINDOW_SIZE_WIDTH,
                      WINDOW_SIZE_HEIGHT,
                      SEARCH_WINDOW_SIZE_WIDTH,
                      SEARCH_WINDOW_SIZE_HEIGHT,
                      ZOOM_WINDOW_SIZE,
                      OVERSAMPLING_FACTOR,
                      ACROSS_GROSS_OFFSET,
                      DOWN_GROSS_OFFSET,
                      ACROSS_LOOKS,
                      DOWN_LOOKS,
                      NUMBER_WINDOWS_ACROSS,
                      NUMBER_WINDOWS_DOWN,
                      SKIP_SAMPLE_ACROSS,
                      SKIP_SAMPLE_DOWN,
                      DOWN_SPACING_PRF1,
                      DOWN_SPACING_PRF2,
                      ACROSS_SPACING1,
                      ACROSS_SPACING2,
                      FIRST_SAMPLE_ACROSS,
                      LAST_SAMPLE_ACROSS,
                      FIRST_SAMPLE_DOWN,
                      LAST_SAMPLE_DOWN,
                      IMAGE_DATATYPE1,
                      IMAGE_DATATYPE2,
                      SNR_THRESHOLD,
                      COV_THRESHOLD,
                      BAND1,
                      BAND2,
                      MARGIN,
                      DEBUG_FLAG,
                      DISPLAY_FLAG)

#    @use_api
    def ampcor(self,slcImage1 = None,slcImage2 = None, band1=None, band2=None):
        if not (slcImage1 == None):
            self.slcImage1 = slcImage1
        if (self.slcImage1 == None):
            print("Error. reference slc image not set.")
            raise Exception
        if not (slcImage2 == None):
            self.slcImage2 = slcImage2
        if (self.slcImage2 == None):
            print("Error. secondary slc image not set.")
            raise Exception

        if band1 is not None:
            self.band1 = int(band1)

        if self.band1 >= self.slcImage1.bands:
            raise ValueError('Requesting band %d from image1 with %d bands'%(self.band1+1, self.slcImage1.bands))

        if band2 is not None:
            self.band2 = int(band2)

        if self.band2 >= self.slcImage2.bands:
            raise ValueError('requesting band %d from image2 with %d bands'%(self.band2+1, self.slcImage2.bands))
       
        slcAccessor1 = self.slcImage1.getImagePointer()
        slcAccessor2 = self.slcImage2.getImagePointer()
        self.lineLength1 = self.slcImage1.getWidth()
        self.fileLength1 = self.slcImage1.getLength()
        self.lineLength2 = self.slcImage2.getWidth()
        self.fileLength2 = self.slcImage2.getLength()

        if (self.numberLocationAcross is not None) and (self.skipSampleAcross is not None):
            raise ValueError('Cannot set both numberLocationAcross and skipSampleAcross. Set any one of the two inputs.')

        if (self.numberLocationDown is not None) and (self.skipSampleDown is not None):
            raise ValueError('Cannot set both numberLocationDown and skipSampleDown. Set any of the two inputs.')

        self.checkTypes()
        self.checkWindows()
        self.checkSkip()

        self.allocateArrays()
        self.setState()
        
#        self.checkInitialization()
#        self.checkImageLimits()
       
        b1 = int(self.band1)
        b2 = int(self.band2)
        ampcor.ampcor_Py(slcAccessor1,slcAccessor2, b1, b2)
        
        self.getState()
        self.deallocateArrays()

        return

    def checkTypes(self):
        '''Check if the image datatypes are set.'''

        if self.imageDataType1 == '':
            if self.slcImage1.getDatatype().upper().startswith('C'):
                self.imageDataType1 = 'complex'
            else:
                self.imageDataType1 = 'real'
        else:
            if self.imageDataType1 not in ('complex','real','mag'):
                raise ValueError('ImageDataType1 should be either complex/real/mag.')

        if self.imageDataType2 == '':
            if self.slcImage2.getDatatype().upper().startswith('C'):
                self.imageDataType2 = 'complex'
            else:
                self.imageDataType2 = 'real'
        else:
            if self.imageDataType2 not in ('complex','real','mag'):
                raise ValueError('ImageDataType2 should be either complex/real/mag.')
        

    def checkWindows(self):
        '''Ensure that the window sizes are valid for the code to work.'''
        if (self.windowSizeWidth%2 == 1):
            raise ValueError('Window size width needs to be multiple of 2.')

        if (self.windowSizeHeight%2 == 1):
            raise ValueError('Window size height needs to be multiple of 2.')

        if not is_power2(self.zoomWindowSize):
            raise ValueError('Zoom window size needs to be a power of 2.')

        if not is_power2(self.oversamplingFactor):
            raise ValueError('Oversampling factor needs to be a power of 2.')

        #if self.searchWindowSizeWidth >=  2*self.windowSizeWidth :
        #    raise ValueError('Search Window Size Width should be < 2 * Window Size Width')

        #if self.searchWindowSizeHeight >= 2*self.windowSizeHeight :
        #    raise ValueError('Search Window Size Height should be < 2 * Window Size Height')

        #if self.zoomWindowSize >= min(self.searchWindowSizeWidth, self.searchWindowSizeHeight):
        #    raise ValueError('Zoom window size should be <= Search window size')


    def checkSkip(self):
        '''
        Check if the first, last and skip values are initialized.
        '''

        xMargin = 2*self.searchWindowSizeWidth + self.windowSizeWidth
        yMargin = 2*self.searchWindowSizeHeight + self.windowSizeHeight
        if self.scaleFactorY is None:
            if (self.prf1 is None) or (self.prf2 is None):
                self.scaleFactorY = 1.
            else:
                self.scaleFactorY = self.prf2 / self.prf1

        if (self.scaleFactorY < 0.9) or (self.scaleFactorY > 1.1):
            raise ValueError('Ampcor is designed to work on images with maximum of 10%% scale difference in azimuth. Attempting to use images with scale difference of %2.2f'%(self.scaleFactorY))

        if self.scaleFactorX is None:
            if (self.rangeSpacing1 is None) or (self.rangeSpacing2 is None):
                self.scaleFactorX = 1.
            else:
                self.scaleFactorX = self.rangeSpacing1/self.rangeSpacing2

        if (self.scaleFactorX < 0.9) or (self.scaleFactorX > 1.1):
            raise ValueError('Ampcor is designed to work on images with maximum of 10%% scale difference in range. Attempting to use images with scale difference of %2.2f'%(self.scaleFactorX))

        print('Scale Factor in Range: ', self.scaleFactorX)
        print('Scale Factor in Azimuth: ', self.scaleFactorY)

        offAcmax = int(self.acrossGrossOffset + (self.scaleFactorX-1)*self.lineLength1)

        offDnmax = int(self.downGrossOffset + (self.scaleFactorY-1)*self.fileLength1)

        if self.firstSampleDown is None:
            self.firstSampleDown = max(self.margin, -self.downGrossOffset) + yMargin + 1

        if self.lastSampleDown is None:
            self.lastSampleDown = int( min(self.fileLength1, self.fileLength2-offDnmax) - yMargin - 1 - self.margin)

        if (self.skipSampleDown is None) and (self.numberLocationDown is not None):
            self.skipSampleDown = int((self.lastSampleDown - self.firstSampleDown) / (self.numberLocationDown - 1.))
            print('Skip Sample Down: %d'%(self.skipSampleDown))
        else:
            raise ValueError('Both skipSampleDown and numberLocationDown undefined. Need atleast one input.')

        if self.firstSampleAcross is None:
            self.firstSampleAcross = max(self.margin, -self.acrossGrossOffset) + xMargin + 1

        if self.lastSampleAcross is None:
            self.lastSampleAcross = int(min(self.lineLength1, self.lineLength2 - offAcmax) - xMargin - 1 -self.margin)

        if (self.skipSampleAcross is None) and (self.numberLocationAcross is not None):
            self.skipSampleAcross = int((self.lastSampleAcross - self.firstSampleAcross) / (self.numberLocationAcross - 1.))
            print('Skip Sample Across: %d'%(self.skipSampleAcross))
        else:
            raise ValueError('Both skipSampleDown and numberLocationDown undefined. Need atleast one input.')

        return

    def checkImageLimits(self):
        '''Ensure that the search region in the images is valid.'''
       
        xMargin = 2*self.searchWindowSizeWidth + self.windowSizeWidth
        yMargin = 2*self.searchWindowSizeHeight + self.windowSizeHeight
        ######Checks related to the reference image only 
        #if( self.firstSampleAcross < xMargin):
        #    raise ValueError('First sample is not far enough from the left edge in reference image.')

        #if( self.firstSampleDown < yMargin):
        #    raise ValueError('First sample is not far enough from the top edge of the reference image.')

        #if( self.lastSampleAcross > (self.lineLength1 - xMargin) ):
        #    raise ValueError('Last sample is not far enough from the right edge of the reference image.')

        #if( self.lastSampleDown > (self.fileLength1 - yMargin) ):
        #    raise ValueError('Last sample line %d is not far enough from the bottom edge %d of the reference image.'%(self.lastSampleDown,(self.fileLength1 - yMargin)))

        #if( (self.lastSampleAcross - self.firstSampleAcross) < (2*xMargin)):
            #raise ValueError('Too small a reference image in the width direction.')

        #if( (self.lastSampleDown - self.firstSampleDown) < (2*yMargin)):
            #raise ValueError('Too small a reference image in the height direction.')

        if ( self.lastSampleAcross <= self.firstSampleAcross):
            raise ValueError('Last Sample Across requested is to the left of first sample across')

        if (self.lastSampleDown <= self.firstSampleDown):
            raise ValueError('Last Sample Down requested is above first sample down')

    def setState(self):
        ampcor.setImageDataType1_Py(str(self.imageDataType1))
        ampcor.setImageDataType2_Py(str(self.imageDataType2))
        ampcor.setLineLength1_Py(int(self.lineLength1))
        ampcor.setLineLength2_Py(int(self.lineLength2))
        ampcor.setImageLength1_Py(int(self.fileLength1))
        ampcor.setImageLength2_Py(int(self.fileLength2))
        ampcor.setFirstSampleAcross_Py(int(self.firstSampleAcross))
        ampcor.setLastSampleAcross_Py(int(self.lastSampleAcross))
        ampcor.setSkipSampleAcross_Py(int(self.skipSampleAcross))
        ampcor.setFirstSampleDown_Py(int(self.firstSampleDown))
        ampcor.setLastSampleDown_Py(int(self.lastSampleDown))
        ampcor.setSkipSampleDown_Py(int(self.skipSampleDown))
        ampcor.setAcrossGrossOffset_Py(int(self.acrossGrossOffset))
        ampcor.setDownGrossOffset_Py(int(self.downGrossOffset))
        ampcor.setDebugFlag_Py(self.debugFlag)
        ampcor.setDisplayFlag_Py(self.displayFlag)

        ampcor.setWindowSizeWidth_Py(self.windowSizeWidth)
        ampcor.setWindowSizeHeight_Py(self.windowSizeHeight)
        ampcor.setSearchWindowSizeWidth_Py(self.searchWindowSizeWidth)
        ampcor.setSearchWindowSizeHeight_Py(self.searchWindowSizeHeight)
        ampcor.setZoomWindowSize_Py(self.zoomWindowSize)
        ampcor.setOversamplingFactor_Py(self.oversamplingFactor)
        ampcor.setThresholdSNR_Py(self.thresholdSNR)
        ampcor.setThresholdCov_Py(self.thresholdCov)
        ampcor.setScaleFactorX_Py(self.scaleFactorX)
        ampcor.setScaleFactorY_Py(self.scaleFactorY)
        ampcor.setAcrossLooks_Py(self.acrossLooks)
        ampcor.setDownLooks_Py(self.downLooks)

        #reference values
        #self.winsizeFilt = 8
        #self.oversamplingFactorFilt = 64
        ampcor.setWinsizeFilt_Py(self.winsizeFilt)
        ampcor.setOversamplingFactorFilt_Py(self.oversamplingFactorFilt)

        return

    def setImageDataType1(self, var):
        self.imageDataType1 = str(var)
        return

    def setImageDataType2(self, var):
        self.imageDataType2 = str(var)
        return

    def setLineLength1(self,var):
        self.lineLength1 = int(var)
        return

    def setLineLength2(self, var):
        self.LineLength2 = int(var)
        return

    def setFileLength1(self,var):
        self.fileLength1 = int(var)
        return

    def setFileLength2(self, var):
        self.fileLength2 = int(var)

    def setFirstSampleAcross(self,var):
        self.firstSampleAcross = int(var)
        return

    def setLastSampleAcross(self,var):
        self.lastSampleAcross = int(var)
        return

    def setSkipSampleAcross(self, var):
        self.skipSampleAcross = int(var)
        return

    def setNumberLocationAcross(self,var):
        self.numberLocationAcross = int(var)
        return

    def setFirstSampleDown(self,var):
        self.firstSampleDown = int(var)
        return

    def setLastSampleDown(self,var):
        self.lastSampleDown = int(var)
        return

    def setSkipSampleDown(self,var):
        self.skipSampleDown = int(var)
        return

    def setNumberLocationDown(self,var):
        self.numberLocationDown = int(var)
        return

    def setAcrossGrossOffset(self,var):
        self.acrossGrossOffset = int(var)
        return

    def setDownGrossOffset(self,var):
        self.downGrossOffset = int(var)
        return

    def setFirstPRF(self,var):
        self.prf1 = float(var)
        return

    def setSecondPRF(self,var):
        self.prf2 = float(var)
        return

    def setFirstRangeSpacing(self,var):
        self.rangeSpacing1 = float(var)
        return
    
    def setSecondRangeSpacing(self,var):
        self.rangeSpacing2 = float(var)

    def setDebugFlag(self,var):
        self.debugFlag = bool(var)
        return

    def setDisplayFlag(self, var):
        self.displayFlag = bool(var)
        return
    
    def setReferenceSlcImage(self,im):
        self.slcImage1 = im
        return
    
    def setSecondarySlcImage(self,im):
        self.slcImage2 = im
        return

    def setWindowSizeWidth(self, var):
        temp = int(var)
        if (temp%2):
            raise ValueError('Window width must be a multiple of 2.')
        self.windowSizeWidth = temp
        return

    def setWindowSizeHeight(self, var):
        temp = int(var)
        if (temp%2):
            raise ValueError('Window height must be a multiple of 2.')
        self.windowSizeHeight = temp
        return

    def setZoomWindowSize(self, var):
        temp = int(var)
        if not is_power2(temp):
            raise ValueError('Zoom window size needs to be a power of 2.')
        self.zoomWindowSize = temp

    def setOversamplingFactor(self, var):
        temp = int(var)
        if not is_power2(temp):
            raise ValueError('Oversampling factor needs to be a power of 2.')
        self.oversamplingFactor = temp

    def setWinsizeFilt(self, var):
        temp = int(var)
        self.winsizeFilt = temp

    def setOversamplingFactorFilt(self, var):
        temp = int(var)
        self.oversamplingFactorFilt = temp

    def setSearchWindowSizeWidth(self, var):
        self.searchWindowSizeWidth = int(var)
        return

    def setSearchWindowSizeHeight(self, var):
        self.searchWindowSizeHeight = int(var)
        return

    def setAcrossLooks(self, var):
        self.acrossLooks = int(var)
        return

    def setDownLooks(self, var):
        self.downLooks = int(var)
        return


    def getResultArrays(self):
        retList = []
        retList.append(self.locationAcross)
        retList.append(self.locationAcrossOffset)
        retList.append(self.locationDown)
        retList.append(self.locationDownOffset)
        retList.append(self.snrRet)
        return retList


    def getOffsetField(self):
        """Return and OffsetField object instead of an array of results"""
        offsets = OffsetField()
        for i in range(self.numRows):
            across = self.locationAcross[i]
            down = self.locationDown[i]
            acrossOffset = self.locationAcrossOffset[i]
            downOffset = self.locationDownOffset[i]
            snr = self.snrRet[i]
            sigx = self.cov1Ret[i]
            sigy = self.cov2Ret[i]
            sigxy = self.cov3Ret[i]
            offset = Offset()
            offset.setCoordinate(across,down)
            offset.setOffset(acrossOffset,downOffset)
            offset.setSignalToNoise(snr)
            offset.setCovariance(sigx,sigy,sigxy)
            offsets.addOffset(offset)
        
        return offsets


    def getState(self):
        self.numRows = ampcor.getNumRows_Py()
        self.locationAcross = ampcor.getLocationAcross_Py(self.numRows)
        self.locationAcrossOffset = ampcor.getLocationAcrossOffset_Py(self.numRows)
        self.locationDown = ampcor.getLocationDown_Py(self.numRows)
        self.locationDownOffset = ampcor.getLocationDownOffset_Py(self.numRows)
        self.snrRet = ampcor.getSNR_Py(self.numRows)
        self.cov1Ret = ampcor.getCov1_Py(self.numRows)
        self.cov2Ret = ampcor.getCov2_Py(self.numRows)
        self.cov3Ret = ampcor.getCov3_Py(self.numRows)

        return


    def getLocationAcross(self):
        return self.locationAcross

    def getLocationAcrossOffset(self):
        return self.locationAcrossOffset

    def getLocationDown(self):
        return self.locationDown

    def getLocationDownOffset(self):
        return self.locationDownOffset

    def getSNR(self):
        return self.snrRet

    def getCov1(self):
        return self.cov1Ret
    
    def getCov2(self):
        return self.cov2Ret

    def getCov3(self):
        return self.cov3Ret


    def allocateArrays(self):
        import numpy as np
        self.numberLocationAcross = len(np.arange(self.firstSampleAcross, self.lastSampleAcross, self.skipSampleAcross)) + 1
        self.numberLocationDown = len(np.arange(self.firstSampleDown, self.lastSampleDown, self.skipSampleDown)) + 1
        numEl = self.numberLocationAcross * self.numberLocationDown

        if (self.dim1_locationAcross == None):
            self.dim1_locationAcross = numEl

        if (not self.dim1_locationAcross):
            print("Error. Trying to allocate zero size array")

            raise Exception

        ampcor.allocate_locationAcross_Py(self.dim1_locationAcross)

        if (self.dim1_locationAcrossOffset == None):
            self.dim1_locationAcrossOffset = numEl

        if (not self.dim1_locationAcrossOffset):
            print("Error. Trying to allocate zero size array")

            raise Exception

        ampcor.allocate_locationAcrossOffset_Py(self.dim1_locationAcrossOffset)

        if (self.dim1_locationDown == None):
            self.dim1_locationDown = numEl

        if (not self.dim1_locationDown):
            print("Error. Trying to allocate zero size array")

            raise Exception

        ampcor.allocate_locationDown_Py(self.dim1_locationDown)

        if (self.dim1_locationDownOffset == None):
            self.dim1_locationDownOffset = numEl

        if (not self.dim1_locationDownOffset):
            print("Error. Trying to allocate zero size array")

            raise Exception

        ampcor.allocate_locationDownOffset_Py(self.dim1_locationDownOffset)

        if (self.dim1_snrRet == None):
            self.dim1_snrRet = numEl

        if (not self.dim1_snrRet):
            print("Error. Trying to allocate zero size array")

            raise Exception

        ampcor.allocate_snrRet_Py(self.dim1_snrRet)
        ampcor.allocate_cov1Ret_Py(self.dim1_snrRet)
        ampcor.allocate_cov2Ret_Py(self.dim1_snrRet)
        ampcor.allocate_cov3Ret_Py(self.dim1_snrRet)

        return


    def deallocateArrays(self):
        ampcor.deallocate_locationAcross_Py()
        ampcor.deallocate_locationAcrossOffset_Py()
        ampcor.deallocate_locationDown_Py()
        ampcor.deallocate_locationDownOffset_Py()
        ampcor.deallocate_snrRet_Py()
        ampcor.deallocate_cov1Ret_Py()
        ampcor.deallocate_cov2Ret_Py()
        ampcor.deallocate_cov3Ret_Py()

        return

    def __init__(self, name=''):
        super(Ampcor, self).__init__(family=self.__class__.family, name=name)
        self.locationAcross = []
        self.dim1_locationAcross = None
        self.locationAcrossOffset = []
        self.dim1_locationAcrossOffset = None
        self.locationDown = []
        self.dim1_locationDown = None
        self.locationDownOffset = []
        self.dim1_locationDownOffset = None
        self.snrRet = []
        self.dim1_snrRet = None
        self.lineLength1 = None
        self.lineLength2 = None
        self.fileLength1 = None
        self.fileLength2 = None
        self.scaleFactorX = None
        self.scaleFactorY = None
        self.numRows = None
        self.winsizeFilt = 1
        self.oversamplingFactorFilt = 64
        self.dictionaryOfVariables = { \
                                      'IMAGETYPE1' : ['imageDataType1', 'str', 'optional'], \
                                      'IMAGETYPE2' : ['imageDataType2', 'str', 'optional'], \
                                      'FIRST_SAMPLE_ACROSS' : ['firstSampleAcross', 'int','mandatory'], \
                                      'LAST_SAMPLE_ACROSS' : ['lastSampleAcross', 'int','mandatory'], \
                                      'NUMBER_LOCATION_ACROSS' : ['numberLocationAcross', 'int','mandatory'], \
                                      'FIRST_SAMPLE_DOWN' : ['firstSampleDown', 'int','mandatory'], \
                                      'LAST_SAMPLE_DOWN' : ['lastSampleDown', 'int','mandatory'], \
                                      'NUMBER_LOCATION_DOWN' : ['numberLocationDown', 'int','mandatory'], \
                                      'ACROSS_GROSS_OFFSET' : ['acrossGrossOffset', 'int','optional'], \
                                      'DOWN_GROSS_OFFSET' : ['downGrossOffset', 'int','optional'], \
                                      'PRF1' : ['prf1', 'float','optional'], \
                                      'PRF2' : ['prf2', 'float','optional'], \
                                      'RANGE_SPACING1' : ['rangeSpacing1', 'float', 'optional'], \
                                      'RANGE_SPACING2' : ['rangeSpacing2', 'float', 'optional'], \
                                      'DEBUG_FLAG' : ['debugFlag', 'str','optional'] \
                                      }
        self.dictionaryOfOutputVariables = { \
                                            'LOCATION_ACROSS' : 'locationAcross', \
                                            'LOCATION_ACROSS_OFFSET' : 'locationAcrossOffset', \
                                            'LOCATION_DOWN' : 'locationDown', \
                                            'LOCATION_DOWN_OFFSET' : 'locationDownOffset', \
                                            'SNR' : 'snrRet' \
                                            }
        #self.descriptionOfVariables = {}
        #self.mandatoryVariables = []
        #self.optionalVariables = []
        #self.initOptionalAndMandatoryLists()
        return


#end class
if __name__ == "__main__":
    sys.exit(main())
