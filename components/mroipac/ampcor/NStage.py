#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2012 California Institute of Technology. ALL RIGHTS RESERVED.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# United States Government Sponsorship acknowledged. This software is subject to
# U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
# (No [Export] License Required except when exporting to an embargoed country,
# end user, or in support of a prohibited end use). By downloading this software,
# the user agrees to comply with all applicable U.S. export laws and regulations.
# The user has the responsibility to obtain export licenses, or other export
# authority as may be required before exporting this software to any 'EAR99'
# embargoed foreign country or citizen of those countries.
#
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





from __future__ import print_function
import sys
import os
import math
import isceobj
from isceobj.Location.Offset import OffsetField,Offset
from iscesys.Component.Component import Component,Port
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from iscesys.StdOEL.StdOELPy import create_writer
from .Ampcor import Ampcor
from isceobj.Util.mathModule import is_power2
import logging

logger = logging.getLogger('mroipac.ampcor.nstage')

NUMBER_STAGES = Component.Parameter('nStages',
        public_name='NUMBER_STAGES',
        default=4,
        type=int,
        mandatory=False,
        doc = 'Number of stages for multi-scale offset estimator.')


SCALE = Component.Parameter('scale',
        public_name='SCALE',
        default=2,
        type=int,
        mandatory=False,
        doc = 'Scale factor in between each stage of ampcor.')

COARSE_NUMBER_WINDOWS_ACROSS = Component.Parameter('coarseNumWinAcross',
        public_name='COARSE_NUMBER_WINDOWS_ACROSS',
        default=10,
        type = int,
        mandatory = False,
        doc = 'Number of windows in range for coarse scales.')

COARSE_NUMBER_WINDOWS_DOWN = Component.Parameter('coarseNumWinDown',
        public_name='COARSE_NUMBER_WINDOWS_DOWN',
        default=10,
        type=int,
        mandatory=False,
        doc = 'Number of windows in azimuth for coarse scales.')

COARSE_OVERSAMPLING_FACTOR = Component.Parameter('coarseOversamplingFactor',
        public_name='COARSE_OVERSAMPLING_FACTOR',
        default=4,
        type=int,
        mandatory=False,
        doc = 'Oversampling factor for coarse scales.')

COARSE_SNR_THRESHOLD = Component.Parameter('coarseSNRThreshold',
        public_name='COARSE_SNR_THRESHOLD',
        default = 2.0,
        type = float,
        mandatory=False,
        doc = 'SNR threshold for culling at coarser scales.')

COARSE_DISTANCE_THRESHOLD = Component.Parameter('coarseDistance',
        public_name='COARSE_DISTANCE_THRESHOLD',
        default = 10.0,
        type=float,
        mandatory=False,
        doc = 'SNR threshold for culling at coarser scales.')

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

COARSE_ZOOM_WINDOW_SIZE = Component.Parameter('coarseZoomWindowSize',
        public_name='COARSE_ZOOM_WINDOW_SIZE',
        default = 32,
        type=int,
        mandatory=False,
        doc = 'Zoom window around local maxima at coarse scales.')

ZOOM_WINDOW_SIZE = Component.Parameter('zoomWindowSize',
        public_name = 'ZOOM_WINDOW_SIZE',
        default = 16,
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

IMAGE_DATATYPE1 = Component.Parameter('imageDataType1',
        public_name = 'IMAGE_DATATYPE1',
        default='',
        type = str,
        mandatory = False,
        doc = 'Image data type for reference image (complex / real)')

IMAGE_DATATYPE2 = Component.Parameter('imageDataType2',
        default='',
        type = str,
        mandatory=False,
        doc = 'Image data type for search image (complex / real)')


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


class NStage(Component):

    family = 'nstage'
    logging_name = 'isce.mroipac.nstage'

    parameter_list = (NUMBER_STAGES,
                      SCALE,
                      COARSE_NUMBER_WINDOWS_ACROSS,
                      COARSE_NUMBER_WINDOWS_DOWN,
                      COARSE_OVERSAMPLING_FACTOR,
                      COARSE_SNR_THRESHOLD,
                      COARSE_DISTANCE_THRESHOLD,
                      COARSE_ZOOM_WINDOW_SIZE,
                      WINDOW_SIZE_WIDTH,
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
                      DOWN_SPACING_PRF1,
                      DOWN_SPACING_PRF2,
                      ACROSS_SPACING1,
                      ACROSS_SPACING2,
                      IMAGE_DATATYPE1,
                      IMAGE_DATATYPE2,
                      SNR_THRESHOLD,
                      COV_THRESHOLD,
                      BAND1,
                      BAND2)

    def nstage(self,slcImage1 = None,slcImage2 = None):
        if not (slcImage1 == None):
            self.slcImage1 = slcImage1
        if (self.slcImage1 == None):
            logger.error("Error. reference slc image not set.")
            raise Exception
        if not (slcImage2 == None):
            self.slcImage2 = slcImage2
        if (self.slcImage2 == None):
            logger.error("Error. secondary slc image not set.")
            raise Exception

        self.fileLength1 = self.slcImage1.getLength()
        self.lineLength1 = self.slcImage1.getWidth()
        self.fileLength2 = self.slcImage2.getLength()
        self.lineLength2 = self.slcImage2.getWidth()

        ####Run checks
        self.checkTypes()
        self.checkWindows()

        ####Actual processing
        mSlc = self.slcImage1
        sSlc = self.slcImage2
        coarseAcross = self.acrossGrossOffset
        coarseDown = self.downGrossOffset
        nStageName = self.name
        logger.info('NSTAGE NAME = %s'%(self.name))
        for iterNum in range(self.nStages-1, -1, -1):
            ####Rewind the images
            try:
                mSlc.rewind()
                sSlc.rewind()
            except:
                logger.error('Issues when rewinding images.')
                raise Exception

            objOff = None
            objAmpcor = None

            logger.debug('Starting Iteration Stage: %d'%(iterNum))
            logger.debug('Gross Across: %s'%(coarseAcross))
            logger.debug('Gross Down  : %s'%(coarseDown))

            objAmpcor = Ampcor(name='%s_%d'%(nStageName,iterNum))
            objAmpcor.configure()
            objAmpcor.setImageDataType1(self.imageDataType1)
            objAmpcor.setImageDataType2(self.imageDataType2)
            objAmpcor.setFirstPRF(self.prf1)
            objAmpcor.setSecondPRF(self.prf2)
            objAmpcor.setFirstRangeSpacing(self.rangeSpacing1)
            objAmpcor.setSecondRangeSpacing(self.rangeSpacing2)

            ######Scale all the reference and search windows
            scaleFactor = self.scale**iterNum
            logger.debug('Scale Factor: %d'%(int(scaleFactor)))
            objAmpcor.windowSizeWidth = scaleFactor*self.windowSizeWidth
            objAmpcor.windowSizeHeight = scaleFactor*self.windowSizeHeight
            objAmpcor.searchWindowSizeWidth = scaleFactor*self.searchWindowSizeWidth
            objAmpcor.searchWindowSizeHeight = scaleFactor*self.searchWindowSizeHeight

            xMargin = 2*objAmpcor.searchWindowSizeWidth + objAmpcor.windowSizeWidth
            yMargin = 2*objAmpcor.searchWindowSizeHeight + objAmpcor.windowSizeHeight

            #####Set image limits for search
            offAc = max(objAmpcor.margin,-coarseAcross)+xMargin
            offDn = max(objAmpcor.margin,-coarseDown)+yMargin

            offAcmax = int(coarseAcross + ((self.rangeSpacing1/self.rangeSpacing2)-1)*self.lineLength1)
            logger.debug("Gross Max Across: %s" % (offAcmax))
            lastAc = int(min(self.lineLength1, self.lineLength2-offAcmax) - xMargin)

            offDnmax = int(coarseDown + ((self.prf2/self.prf1)-1)*self.lineLength1)
            logger.debug("Gross Max Down: %s" % (offDnmax))

            lastDn = int(min(self.fileLength1, self.fileLength2-offDnmax)  - yMargin)

            objAmpcor.setFirstSampleAcross(offAc)
            objAmpcor.setLastSampleAcross(lastAc)
            objAmpcor.setFirstSampleDown(offDn)
            objAmpcor.setLastSampleDown(lastDn)

            objAmpcor.setAcrossGrossOffset(coarseAcross)
            objAmpcor.setDownGrossOffset(coarseDown)

            if (offAc > lastAc) or (offDn > lastDn):
                logger.info('Search window scale is too large.')
                logger.info('Skipping scale: %d'%(iterNum+1))
                continue

            logger.debug('First Sample Across = %d'%(offAc))
            logger.debug('Last Sampe Across = %d'%(lastAc))
            logger.debug('First Sample Down = %d'%(offDn))
            logger.debug('Last Sample Down = %d'%(lastDn))
            logger.debug('Looks = %d'%(scaleFactor))
            logger.debug('Correlation window sizes: %d %d'%(objAmpcor.windowSizeWidth, objAmpcor.windowSizeHeight))
            logger.debug('Search window sizes: %d %d'%(objAmpcor.searchWindowSizeWidth, objAmpcor.searchWindowSizeHeight))

            objAmpcor.band1 = self.band1
            objAmpcor.band2 = self.band2
            if (iterNum == 0):
                objAmpcor.setNumberLocationAcross(self.numberLocationAcross)
                objAmpcor.setNumberLocationDown(self.numberLocationDown)
                objAmpcor.setAcrossLooks(self.acrossLooks)
                objAmpcor.setDownLooks(self.downLooks)
                objAmpcor.setZoomWindowSize(self.zoomWindowSize)
                objAmpcor.setOversamplingFactor(self.oversamplingFactor)
            else:
                objAmpcor.setNumberLocationAcross(self.coarseNumWinAcross)
                objAmpcor.setNumberLocationDown(self.coarseNumWinDown)
                objAmpcor.setAcrossLooks(scaleFactor*self.acrossLooks)
                objAmpcor.setDownLooks(scaleFactor*self.downLooks)
                objAmpcor.setZoomWindowSize(self.coarseZoomWindowSize)
                objAmpcor.setOversamplingFactor(self.coarseOversamplingFactor)

            objAmpcor.ampcor(mSlc, sSlc)

            offField = objAmpcor.getOffsetField()

            if (iterNum != 0):
                objOff = isceobj.createOffoutliers()
                objOff.wireInputPort(name='offsets', object=offField)
                objOff.setSNRThreshold(self.coarseSNRThreshold)
                objOff.setDistance(self.coarseDistance)
                self._stdWriter.setFileTag("nstage_offoutliers"+str(iterNum), "log")
                self._stdWriter.setFileTag("nstage_offoutliers"+str(iterNum), "err")
                self._stdWriter.setFileTag("nstage_offoutliers"+str(iterNum), "out")
                objOff.setStdWriter(self._stdWriter)
                objOff.offoutliers()

                fracLeft = len(objOff.indexArray)/(1.0*len(offField._offsets))

                print('FracLEft = ', fracLeft)
                if (fracLeft < 0.1):
                    logger.error('NStage - Iteration: %d, Fraction Windows left: %d. Increase number of windows or improve gross offset estimate manually.'%(iterNum, int(100*fracLeft)))
                    raise Exception('NStage matching failed at iteration : %d'%(iterNum))
                elif (fracLeft < 0.2):
                    logger.error('NStage - Iteration: %d, Fraction Windows left: %d. Increase number of windows or improve gross offset estimate manually.'%(iterNum, int(100*fracLeft)))


                coarseAcross = int(objOff.averageOffsetAcross)
                coarseDown = int(objOff.averageOffsetDown)

        mSlc.finalizeImage()
        sSlc.finalizeImage()

        self.getState(offField)
        objOff = None
        objAmpcor = None
        return

    def getState(self, off):
        '''
        Set up the output variables.
        '''
        upack = off.unpackOffsetswithCovariance()
        for val in upack:
            self.locationAcross.append(val[0])
            self.locationAcrossOffset.append(val[1])
            self.locationDown.append(val[2])
            self.locationDownOffset.append(val[3])
            self.snrRet.append(val[4])
            self.cov1Ret.append(val[5])
            self.cov2Ret.append(val[6])
            self.cov3Ret.append(val[7])

        self.numRows = len(upack)
        return

    def checkTypes(self):
        '''Check if the image datatypes are set.'''

        if self.imageDataType1 == '':
            if self.slcImage1.getDatatype().upper().startswith('C'):
                self.imageDataType1 = 'complex'
            else:
                raise ValueError('Undefined value for imageDataType1. Should be complex/real/rmg1/rmg2')
        else:
            if self.imageDataType1 not in ('complex','real'):
                raise ValueError('ImageDataType1 should be either complex/real/rmg1/rmg2.')

        if self.imageDataType2 == '':
            if self.slcImage2.getDatatype().upper().startswith('C'):
                self.imageDataType2 = 'complex'
            else:
                raise ValueError('Undefined value for imageDataType2. Should be complex/real/rmg1/rmg2')
        else:
            if self.imageDataType2 not in ('complex','real'):
                raise ValueError('ImageDataType1 should be either complex/real.')


    def checkWindows(self):
        '''Ensure that the window sizes are valid for the code to work.'''

        if (self.windowSizeWidth%2 == 1):
            raise ValueError('Window size width needs to be an even number.')

        if (self.windowSizeHeight%2 == 1):
            raise ValueError('Window size height needs to be an even number.')

        if not is_power2(self.zoomWindowSize):
            raise ValueError('Zoom window size needs to be a power of 2.')

        if not is_power2(self.oversamplingFactor):
            raise ValueError('Oversampling factor needs to be a power of 2.')

        if not is_power2(self.coarseOversamplingFactor):
            raise ValueError('Coarse oversampling factor needs to be a power of 2.')

        if self.searchWindowSizeWidth >=  2*self.windowSizeWidth :
            raise ValueError('Search Window Size Width should be < 2 * Window Size Width')

        if self.searchWindowSizeHeight >= 2*self.windowSizeHeight :
            raise ValueError('Search Window Size Height should be < 2 * Window Size Height')

        if self.zoomWindowSize >= min(self.searchWindowSizeWidth, self.searchWindowSizeHeight):
            raise ValueError('Zoom window size should be <= Search window size')

        if self._stdWriter is None:
            self._stdWriter = create_writer("log", "", True, filename="nstage.log")

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

    def setNumberLocationAcross(self,var):
        self.numberLocationAcross = int(var)
        return

    def setCoarseNumWinAcross(self,var):
        self.coarseNumWinAcross = int(var)
        return

    def setCoarseNumWinDown(self,var):
        self.coarseNumWinDown = int(var)
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


    def setReferenceSlcImage(self,im):
        self.slcImage1 = im
        return

    def setSecondarySlcImage(self,im):
        self.slcImage2 = im
        return

    def setWindowSizeWidth(self, var):
        temp = int(var)
        if (temp%2 == 1):
            raise ValueError('Window width needs to be an even number.')
        self.windowSizeWidth = temp
        return

    def setWindowSizeHeight(self, var):
        temp = int(var)
        if (temp%2 == 1):
            raise ValueError('Window height needs to be an even number.')
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

    def setCoarseOversamplingFactor(self, var):
        temp = int(var)
        if not is_power2(temp):
            raise ValueError('Coarse oversampling factor needs to be a power of 2.')
        self.coarseOversamplingFactor = temp

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

    def stdWriter(self, var):
        self._stdWriter = var
        return


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


    def __init__(self, name=''):
        super(NStage, self).__init__(family=self.__class__.family, name=name)
        self.locationAcross = []
        self.locationAcrossOffset = []
        self.locationDown = []
        self.locationDownOffset = []
        self.snrRet = []
        self.cov1Ret = []
        self.cov2Ret = []
        self.cov3Ret = []
        self.lineLength1 = None
        self.lineLength2 = None
        self.fileLength1 = None
        self.fileLength2 = None
        self.scaleFactorX = None
        self.scaleFactorY = None
        self.numRows = None
        self._stdWriter = None
        self.dictionaryOfVariables = { \
                                      'IMAGETYPE1' : ['imageDataType1', 'str', 'optional'], \
                                      'IMAGETYPE2' : ['imageDataType2', 'str', 'optional'], \
                                      'NUMBER_LOCATION_ACROSS' : ['numberLocationAcross', 'int','mandatory'], \
                                      'NUMBER_LOCATION_DOWN' : ['numberLocationDown', 'int','mandatory'], \
                                      'COARSE_NUMBER_LOCATION_ACROSS' : ['coarseNumWinAcross','int','mandatory'], \
                                      'COARSE_NUMBER_LOCATION_DOWN' : ['coarseNumWinDown', 'int', 'mandatory'], \
                                      'ACROSS_GROSS_OFFSET' : ['acrossGrossOffset', 'int','optional'], \
                                      'DOWN_GROSS_OFFSET' : ['downGrossOffset', 'int','optional'], \
                                      'PRF1' : ['prf1', 'float','optional'], \
                                      'PRF2' : ['prf2', 'float','optional'], \
                                      'RANGE_SPACING1' : ['rangeSpacing1', 'float', 'optional'], \
                                      'RANGE_SPACING2' : ['rangeSpacing2', 'float', 'optional'], \
                                      }
        self.dictionaryOfOutputVariables = { \
                                            'LOCATION_ACROSS' : 'locationAcross', \
                                            'LOCATION_ACROSS_OFFSET' : 'locationAcrossOffset', \
                                            'LOCATION_DOWN' : 'locationDown', \
                                            'LOCATION_DOWN_OFFSET' : 'locationDownOffset', \
                                            'SNR' : 'snrRet' \
                                            }
#        self.descriptionOfVariables = {}
#        self.mandatoryVariables = []
#        self.optionalVariables = []
#        self.initOptionalAndMandatoryLists()
        return None


#end class
if __name__ == "__main__":
    sys.exit(main())
