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

from isceobj.Location.Offset import OffsetField,Offset
from iscesys.Component.Component import Component,Port
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from isceobj.Util import estimateoffsets
from isceobj.Util.mathModule import is_power2

import logging
logger = logging.getLogger('isce.Util.estimateoffsets')

SensorSearchWindowSize = {'ALOS':20, 'COSMO_SKYMED':20, 'COSMO_SKYMED_SLC':40,
                          'ENVISAT':20, 'ERS':40, 'JERS':20, 'RADARSAT1':20,
                          'RADARSAT2':20, 'TERRASARX':20, 'TANDEMX':20,
                          'UAVSAR_SLC':20, 'SAOCOM':20, 'GENERIC':20}
DefaultSearchWindowSize = 20


WINDOW_SIZE = Component.Parameter('windowSize',
        public_name='WINDOW_SIZE',
        default=32,
        type=int,
        mandatory = False,
        doc = 'Dimensions of the reference data window for correlation.')

SEARCH_WINDOW_SIZE = Component.Parameter('searchWindowSize',
        public_name='SEARCH_WINDOW_SIZE',
        default = None,
        type = int,
        mandatory = False,
        doc = 'Dimensions of the search data window for correlation.')

ZOOM_WINDOW_SIZE = Component.Parameter('zoomWindowSize',
        public_name='ZOOM_WINDOW_SIZE',
        default = 8,
        type=int,
        mandatory=False,
        doc = 'Dimensions of the zoom window around first pass correlation peak.')

ACROSS_GROSS_OFFSET = Component.Parameter('acrossGrossOffset',
        public_name='ACROSS_GROSS_OFFSET',
        default=None,
        type=int,
        mandatory=False,
        doc = 'Offset in the range direction')

DOWN_GROSS_OFFSET = Component.Parameter('downGrossOffset',
        public_name='DOWN_GROSS_OFFSET',
        default=None,
        type=int,
        mandatory=False,
        doc = 'Offset in the azimuth direction')

BAND1 = Component.Parameter('band1',
        public_name='BAND1',
        default=None,
        type=int,
        mandatory=False,
        doc = 'Band number for reference image')

BAND2 = Component.Parameter('band2',
        public_name='BAND2',
        default=None,
        type=int,
        mandatory=False,
        doc = 'Band number for search image')

NUMBER_WINDOWS_DOWN = Component.Parameter('numberLocationDown',
        public_name='NUMBER_WINDOWS_DOWN',
        default = None,
        type=int,
        mandatory=False,
        doc = 'Number of windows in azimuth direction')

NUMBER_WINDOWS_ACROSS = Component.Parameter('numberLocationAcross',
        public_name='NUMBER_WINDOWS_ACROSS',
        default=None,
        type=int,
        mandatory=False,
        doc = 'Number of windows in range direction')

OVERSAMPLING_FACTOR = Component.Parameter('oversamplingFactor',
        public_name='OVERSAMPLING_FACTOR',
        default = 16,
        type=int,
        mandatory=False,
        doc = 'Oversampling factor for the correlation surface')


FIRST_SAMPLE_ACROSS = Component.Parameter('firstSampleAcross',
        public_name='FIRST_SAMPLE_ACROSS',
        default=None,
        type=int,
        mandatory=False,
        doc = 'First pixel in range')

LAST_SAMPLE_ACROSS = Component.Parameter('lastSampleAcross',
        public_name='LAST_SAMPLE_ACROSS',
        default=None,
        type=int,
        mandatory=False,
        doc = 'Last pixel in range')

FIRST_SAMPLE_DOWN = Component.Parameter('firstSampleDown',
        public_name='FIRST_SAMPLE_DOWN',
        default=None,
        type=int,
        mandatory=False,
        doc = 'First pixel in azimuth')

LAST_SAMPLE_DOWN = Component.Parameter('lastSampleDown',
        public_name='LAST_SAMPLE_DOWN',
        default=None,
        type=int,
        mandatory=False,
        doc = 'Last pixel in azimuth')

DOWN_SPACING_PRF1 = Component.Parameter('prf1',
        public_name='DOWN_SPACING_PRF1',
        default=1.0,
        type=float,
        mandatory=False,
        doc = 'PRF or similar scalefactor for reference image')

DOWN_SPACING_PRF2 = Component.Parameter('prf2',
        public_name='DOWN_SPACING_PRF2',
        default=1.0,
        type=float,
        mandatory=False,
        doc = 'PRF or similar scalefactor for search image')

ISCOMPLEX_IMAGE1 = Component.Parameter('isComplex1',
        public_name='ISCOMPLEX_IMAGE1',
        default=None,
        type=bool,
        mandatory=False,
        doc='Is the reference image complex')

ISCOMPLEX_IMAGE2 = Component.Parameter('isComplex2',
        public_name='ISCOMPLEX_IMAGE2',
        default=None,
        type=bool,
        mandatory=False,
        doc='Is the search image complex.')

MARGIN = Component.Parameter('margin',
        public_name='MARGIN',
        default=50,
        type=int,
        mandatory=False,
        doc='Margin around the image to avoid')

DEBUG_FLAG = Component.Parameter('debugFlag',
        public_name='DEBUG_FLAG',
        default='n',
        type=str,
        mandatory=False,
        doc = 'Print debug information.')

class EstimateOffsets(Component):

    family = 'estimateoffsets'
    logging_name = 'isce.isceobj.estimateoffsets'

    parameter_list = (WINDOW_SIZE,
                      SEARCH_WINDOW_SIZE,
                      ZOOM_WINDOW_SIZE,
                      OVERSAMPLING_FACTOR,
                      ACROSS_GROSS_OFFSET,
                      DOWN_GROSS_OFFSET,
                      NUMBER_WINDOWS_ACROSS,
                      NUMBER_WINDOWS_DOWN,
                      DOWN_SPACING_PRF1,
                      DOWN_SPACING_PRF2,
                      FIRST_SAMPLE_ACROSS,
                      LAST_SAMPLE_ACROSS,
                      FIRST_SAMPLE_DOWN,
                      LAST_SAMPLE_DOWN,
                      BAND1,
                      BAND2,
                      ISCOMPLEX_IMAGE1,
                      ISCOMPLEX_IMAGE2,
                      DEBUG_FLAG)


    def estimateoffsets(self,image1 = None,image2 = None, band1=None, band2=None):
        if image1 is not None:
            self.image1 = image1
        if (self.image1 == None):
            raise ValueError("Error. reference image not set.")

        if image2 is not None:
            self.image2 = image2

        if (self.image2 == None):
            raise ValueError("Error. secondary image not set.")

        if band1 is not None:
            self.band1 = int(band1)

        if self.band1 >= self.image1.bands:
            raise ValueError('Requesting band %d from image with %d bands'%(self.band1+1, self.image1.bands))

        if band2 is not None:
            self.band2 = int(band2)

        if self.band2 >= self.image2.bands:
            raise ValueError('Requesting band %d from image with %d bands'%(self.band2+1, self.image2.bands))

        print('Bands: %d %d'%(self.band1,self.band2))
        bAccessor1 = self.image1.getImagePointer()
        bAccessor2 = self.image2.getImagePointer()
        self.lineLength1 = self.image1.getWidth()
        self.fileLength1 = self.image1.getLength()
        self.lineLength2 = self.image2.getWidth()
        self.fileLength2 = self.image2.getLength()


        if not self.numberLocationAcross:
            raise ValueError('Number of windows across has not been set')

        if not self.numberLocationDown:
            raise ValueError('Number of windows down has not been set')

        self.locationAcross = []
        self.locationAcrossOffset = []
        self.locationDown = []
        self.locationDownOffset = []
        self.snrRet = []


        self.checkTypes()
        self.checkWindows()
        self.checkImageLimits()

        self.allocateArrays()
        self.setState()

#        self.checkInitialization()

        estimateoffsets.estimateoffsets_Py(bAccessor1,bAccessor2)

        self.getState()
        self.deallocateArrays()

        return

    def checkTypes(self):
        '''Check if the image datatypes are set.'''

        if not self.isComplex1:
            self.isComplex1 = self.image1.getDataType().upper().startswith('C')
        else:
            if not isinstance(self.isComplex1, bool):
                raise ValueError('isComplex1 must be boolean')

        if not self.isComplex2:
            self.isComplex2 = self.image2.getDataType().upper().startswith('C')
        else:
            if not isinstance(self.isComplex2, bool):
                raise ValueError('isComplex2 must be boolean')

        return


    def checkWindows(self):
        '''
        Ensure that the window sizes are valid for the code to work.
        '''

        if not is_power2(self.windowSize):
            raise ValueError('Window size needs to be power of 2.')

        if not is_power2(self.zoomWindowSize):
            raise ValueError('Zoom window size needs to be a power of 2.')

        if not is_power2(self.oversamplingFactor):
            raise ValueError('Oversampling factor needs to be a power of 2.')

        if self.searchWindowSize >=  (2*self.windowSize):
            raise ValueError('Search Window Size should be < = 2 * Window Size')

        if self.zoomWindowSize >= self.searchWindowSize:
            raise ValueError('Zoom window size should be <= Search window size')

        return

    def checkImageLimits(self):
        '''
        Check if the first and last samples are set correctly.
        '''
        margin = 2*self.searchWindowSize + self.windowSize
        scaleFactor = self.prf2 / self.prf1

        if (scaleFactor < 0.9) or (scaleFactor > 1.1):
            raise ValueError('Module designed for scale factors in range 0.9 - 1.1. Requested scale factor = %f'%(scaleFactor))

        offDnmax = int(self.downGrossOffset + (scaleFactor-1)*self.fileLength1)

        if self.firstSampleDown is None:
            self.firstSampleDown = max(self.margin, -self.downGrossOffset)+ margin+1


        if self.lastSampleDown is None:
            self.lastSampleDown = int( min(self.fileLength1, self.fileLength2-offDnmax) - margin-1-self.margin)


        if self.firstSampleAcross is None:
            self.firstSampleAcross = max(self.margin, -self.acrossGrossOffset) + margin + 1

        if self.lastSampleAcross is None:
            self.lastSampleAcross = int(min(self.fileLength1, self.fileLength2-self.acrossGrossOffset) - margin - 1 - self.margin)


        if self.firstSampleAcross < margin:
            raise ValueError('First sample is not far enough from the left edge of reference image.')

        if self.firstSampleDown < margin:
            raise ValueError('First sample is not far enought from the top edge of reference image.')

        if self.lastSampleAcross  > (self.lineLength1 - margin):
            raise ValueError('Last sample is not far enough from the right edge of reference image.')

        if self.lastSampleDown > (self.fileLength1 - margin):
            raise ValueError('Last Sample is not far enought from the bottom edge of the reference image.')


        if (self.lastSampleAcross - self.firstSampleAcross) < 2*margin:
            raise ValueError('Too small a reference image in the width direction')

        if (self.lastSampleDown - self.firstSampleDown) < 2*margin:
            raise ValueError('Too small a reference image in the height direction')

        return

    def setState(self):
        estimateoffsets.setLineLength1_Py(int(self.lineLength1))
        estimateoffsets.setFileLength1_Py(int(self.fileLength1))
        estimateoffsets.setLineLength2_Py(int(self.lineLength2))
        estimateoffsets.setFileLength2_Py(int(self.fileLength2))
        estimateoffsets.setFirstSampleAcross_Py(int(self.firstSampleAcross+2*self.windowSize))
        estimateoffsets.setLastSampleAcross_Py(int(self.lastSampleAcross-2*self.windowSize))
        estimateoffsets.setNumberLocationAcross_Py(int(self.numberLocationAcross))
        estimateoffsets.setFirstSampleDown_Py(int(self.firstSampleDown+2*self.windowSize))
        estimateoffsets.setLastSampleDown_Py(int(self.lastSampleDown-2*self.windowSize))
        estimateoffsets.setNumberLocationDown_Py(int(self.numberLocationDown))
        estimateoffsets.setAcrossGrossOffset_Py(int(self.acrossGrossOffset))
        estimateoffsets.setDownGrossOffset_Py(int(self.downGrossOffset))
        estimateoffsets.setFirstPRF_Py(float(self.prf1))
        estimateoffsets.setSecondPRF_Py(float(self.prf2))
        estimateoffsets.setDebugFlag_Py(self.debugFlag)

        estimateoffsets.setWindowSize_Py(self.windowSize)
        estimateoffsets.setSearchWindowSize_Py(self.searchWindowSize)
        estimateoffsets.setZoomWindowSize_Py(self.zoomWindowSize)
        estimateoffsets.setOversamplingFactor_Py(self.oversamplingFactor)
        estimateoffsets.setIsComplex1_Py(int(self.isComplex1))
        estimateoffsets.setIsComplex2_Py(int(self.isComplex2))
        estimateoffsets.setBand1_Py(int(self.band1))
        estimateoffsets.setBand2_Py(int(self.band2))

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

    def setNumberLocationAcross(self,var):
        self.numberLocationAcross = int(var)
        return

    def setFirstSampleDown(self,var):
        self.firstSampleDown = int(var)
        return

    def setLastSampleDown(self,var):
        self.lastSampleDown = int(var)
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

    def setDebugFlag(self,var):
        self.debugFlag = str(var)
        return

    def setReferenceImage(self,im):
        self.image1 = im
        return

    def setSecondaryImage(self,im):
        self.image2 = im
        return

    def setWindowSize(self, var):
        temp = int(var)
        if not is_power2(temp):
            raise ValueError('Window size needs to be a power of 2.')
        self.windowSize = temp

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

    def setSearchWindowSize(self, searchWindowSize=None, sensorName=None):
        """
        Set the searchWindowSize for estimating offsets
        """
        #Input value takes precedence
        if searchWindowSize:
            self.searchWindowSize = int(searchWindowSize)

        #Use default for sensor if sensorName is given and in the
        #SensorSearchWindowSize dictionary defined in this module
        elif sensorName:
            if sensorName.upper() in SensorSearchWindowSize.keys():
                self.searchWindowSize = SensorSearchWindowSize[sensorName.upper()]
                return
            else:
                #Log that a sensorName was given but not found in the
                #dictionary of known sensors
                logger.warning((
                    "sensorName %s not in SensorSearchWindowSize dictionary. "+
                    "The DefaultSearchWindowSize = %d will be used") %
                    (sensorName, DefaultSearchWindowSize))

        #Use the default defined in this module if all else fails
        self.searchWindowSize = DefaultSearchWindowSize

        return

    def getResultArrays(self):
        retList = []
        retList.append(self.locationAcross)
        retList.append(self.locationAcrossOffset)
        retList.append(self.locationDown)
        retList.append(self.locationDownOffset)
        retList.append(self.snrRet)
        return retList

    def roundSnr(self,snr):
        pw = 10
        ret = 0
        while pw > -7:
            if  snr//10**pw:
                break
            pw -= 1
        if pw < 0:
            ret = round(snr,6)
        else:
            ret = round(snr*10**(6 - (pw + 1)))/10**(6 - (pw + 1))
       
        return ret
     
    def getOffsetField(self):
        """Return and OffsetField object instead of an array of results"""
        offsets = OffsetField()
        for i in range(len(self.locationAcross)):
            across = self.locationAcross[i]
            down = self.locationDown[i]
            acrossOffset = self.locationAcrossOffset[i]
            downOffset = self.locationDownOffset[i]
            snr = self.snrRet[i]
            offset = Offset()
            offset.setCoordinate(across,down)
            offset.setOffset(acrossOffset,downOffset)
            offset.setSignalToNoise(snr)
            offsets.addOffset(offset)

        return offsets


    def getState(self):
        self.locationAcross = estimateoffsets.getLocationAcross_Py(self.dim1_locationAcross)
        self.locationAcrossOffset = estimateoffsets.getLocationAcrossOffset_Py(self.dim1_locationAcrossOffset)
        self.locationDown = estimateoffsets.getLocationDown_Py(self.dim1_locationDown)
        self.locationDownOffset = estimateoffsets.getLocationDownOffset_Py(self.dim1_locationDownOffset)
        self.snrRet = estimateoffsets.getSNR_Py(self.dim1_snrRet)
        for i in range(len(self.snrRet)):
            self.snrRet[i] = self.roundSnr(self.snrRet[i])
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






    def allocateArrays(self):
        '''Allocate arrays in fortran module.'''
        numEl = self.numberLocationAcross * self.numberLocationDown

        if (self.dim1_locationAcross == None):
            self.dim1_locationAcross = numEl

        if (not self.dim1_locationAcross):
            print("Error. Trying to allocate zero size array")

            raise Exception

        estimateoffsets.allocate_locationAcross_Py(self.dim1_locationAcross)

        if (self.dim1_locationAcrossOffset == None):
            self.dim1_locationAcrossOffset = numEl

        if (not self.dim1_locationAcrossOffset):
            print("Error. Trying to allocate zero size array")

            raise Exception

        estimateoffsets.allocate_locationAcrossOffset_Py(self.dim1_locationAcrossOffset)

        if (self.dim1_locationDown == None):
            self.dim1_locationDown = numEl

        if (not self.dim1_locationDown):
            print("Error. Trying to allocate zero size array")

            raise Exception

        estimateoffsets.allocate_locationDown_Py(self.dim1_locationDown)

        if (self.dim1_locationDownOffset == None):
            self.dim1_locationDownOffset = numEl

        if (not self.dim1_locationDownOffset):
            print("Error. Trying to allocate zero size array")

            raise Exception

        estimateoffsets.allocate_locationDownOffset_Py(self.dim1_locationDownOffset)

        if (self.dim1_snrRet == None):
            self.dim1_snrRet = numEl

        if (not self.dim1_snrRet):
            print("Error. Trying to allocate zero size array")

            raise Exception

        estimateoffsets.allocate_snrRet_Py(self.dim1_snrRet)

        return


    def deallocateArrays(self):
        estimateoffsets.deallocate_locationAcross_Py()
        estimateoffsets.deallocate_locationAcrossOffset_Py()
        estimateoffsets.deallocate_locationDown_Py()
        estimateoffsets.deallocate_locationDownOffset_Py()
        estimateoffsets.deallocate_snrRet_Py()
        return

    def __init__(self, name=''):
        super(EstimateOffsets,self).__init__(family=self.__class__.family, name=name)
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
        
        self.dictionaryOfOutputVariables = { \
                                            'LOCATION_ACROSS' : 'locationAcross', \
                                            'LOCATION_ACROSS_OFFSET' : 'locationAcrossOffset', \
                                            'LOCATION_DOWN' : 'locationDown', \
                                            'LOCATION_DOWN_OFFSET' : 'locationDownOffset', \
                                            'SNR' : 'snrRet' \
                                            }
        self.descriptionOfVariables = {}
        self.mandatoryVariables = []
        self.optionalVariables = []
        self.initOptionalAndMandatoryLists()
        return

#end class




if __name__ == "__main__":
    import sys
    sys.exit(main())
