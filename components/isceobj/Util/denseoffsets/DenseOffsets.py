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
import isce
import isceobj
from isceobj.Location.Offset import OffsetField,Offset
from iscesys.Component.Component import Component,Port
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from isceobj.Util import denseoffsets
from isceobj.Util.mathModule import is_power2

import logging
logger = logging.getLogger('isce.Util.denseoffsets')


WINDOW_SIZE_WIDTH = Component.Parameter('windowSizeWidth',
        public_name='WINDOW_SIZE_WIDTH',
        default=32,
        type=int,
        mandatory = False,
        doc = 'Window width of the reference data window for correlation.')

WINDOW_SIZE_HEIGHT = Component.Parameter('windowSizeHeight',
        public_name='WINDOW_SIZE_HEIGHT',
        default=32,
        type=int,
        mandatory=False,
        doc = 'Window height of the reference data window for correlation.') 

SEARCH_WINDOW_SIZE_WIDTH = Component.Parameter('searchWindowSizeWidth',
        public_name='SEARCH_WINDOW_SIZE_WIDTH',
        default = 20,
        type = int,
        mandatory = False,
        doc = 'Width of the search data window for correlation.')

SEARCH_WINDOW_SIZE_HEIGHT = Component.Parameter('searchWindowSizeHeight',
        public_name='SEARCH_WINDOW_SIZE_HEIGHT',
        default=20,
        type=int,
        mandatory=False,
        doc = 'Height of the search data window for correlation.')

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

SKIP_SAMPLE_DOWN = Component.Parameter('skipSampleDown',
        public_name='SKIP_SAMPLE_DOWN',
        default = None,
        type=int,
        mandatory=False,
        doc = 'Number of samples to skip in azimuth direction')

SKIP_SAMPLE_ACROSS = Component.Parameter('skipSampleAcross',
        public_name='SKIP_SAMPLE_ACROSS',
        default=None,
        type=int,
        mandatory=False,
        doc = 'Number of samples to skip in range direction')

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

ACROSS_SPACING1 = Component.Parameter('rangeSpacing1',
        public_name = 'ACROSS_SPACING1',
        default=1.0,
        type=float,
        mandatory=False,
        doc = 'Range spacing or similar scale factor for reference image')

ACROSS_SPACING2 = Component.Parameter('rangeSpacing2',
        public_name='ACROSS_SPACING2',
        default=1.0,
        type=float,
        mandatory=False,
        doc = 'Range spacing or similar scale factor for search image')

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
        default=0,
        type=int,
        mandatory=False,
        doc='Margin around the image to avoid')

DEBUG_FLAG = Component.Parameter('debugFlag',
        public_name='DEBUG_FLAG',
        default='n',
        type=str,
        mandatory=False,
        doc = 'Print debug information.')

OFFSET_IMAGE_NAME = Component.Parameter('offsetImageName',
        public_name='OFFSET_IMAGE_NAME',
        default='pixel_offsets.bil',
        type=str,
        mandatory=False,
        doc = 'BIL pixel offset file')

SNR_IMAGE_NAME = Component.Parameter('snrImageName',
        public_name='SNR_IMAGE_NAME',
        default = 'pixel_offsets_snr.rdr',
        type=str,
        mandatory=False,
        doc = 'SNR of the pixel offset estimates')

NORMALIZE_FLAG = Component.Parameter('normalize',
        public_name='NORMALIZE_FLAG',
        default=True,
        type=bool,
        mandatory=False,
        doc = "False = Acchen's code and True = Ampcor hybrid") 

class DenseOffsets(Component):

    family = 'denseoffsets'
    logging_name = 'isce.isceobj.denseoffsets'

    parameter_list = (WINDOW_SIZE_WIDTH,
                      WINDOW_SIZE_HEIGHT,
                      SEARCH_WINDOW_SIZE_WIDTH,
                      SEARCH_WINDOW_SIZE_HEIGHT,
                      ZOOM_WINDOW_SIZE,
                      OVERSAMPLING_FACTOR,
                      ACROSS_GROSS_OFFSET,
                      DOWN_GROSS_OFFSET,
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
                      BAND1,
                      BAND2,
                      ISCOMPLEX_IMAGE1,
                      ISCOMPLEX_IMAGE2,
                      DEBUG_FLAG,
                      OFFSET_IMAGE_NAME,
                      SNR_IMAGE_NAME,
                      NORMALIZE_FLAG)
                      

    def denseoffsets(self,image1 = None,image2 = None, band1=None, band2=None):
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
      
 
        if not self.skipSampleAcross:
            raise ValueError('Skip Sample Across across has not been set')

        if not self.skipSampleDown:
            raise ValueError('Skip Sample Down has not been set')


        ######Sanity checks
        self.checkTypes()
        self.checkWindows()
        self.checkImageLimits()

        #####Create output images
        self.outLines = (self.lastSampleDown - self.firstSampleDown) // (self.skipSampleDown)
        self.outSamples = (self.lastSampleAcross - self.firstSampleAcross) // (self.skipSampleAcross)

        offImage = isceobj.createImage()
        offImage.dataType = 'FLOAT'
        offImage.scheme = 'BIL'
        offImage.bands = 2
        offImage.setAccessMode('write')
        offImage.setWidth(self.outSamples)
        offImage.setLength(self.outLines)
        offImage.setFilename(self.offsetImageName)
        offImage.createImage()
        offImageAcc = offImage.getImagePointer()

        snrImage = isceobj.createImage()
        snrImage.dataType = 'FLOAT'
        snrImage.scheme='BIL'
        snrImage.bands = 1
        snrImage.setAccessMode('write')
        snrImage.setWidth(self.outSamples)
        snrImage.setLength(self.outLines)
        snrImage.setFilename(self.snrImageName)
        snrImage.createImage()
        snrImageAcc = snrImage.getImagePointer()

        self.setState()

        denseoffsets.denseoffsets_Py(bAccessor1,bAccessor2,offImageAcc,snrImageAcc)

        offImage.finalizeImage()
        snrImage.finalizeImage()
        offImage.renderHdr()
        snrImage.renderHdr()

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

#        if not is_power2(self.windowSizeWidth):
#            raise ValueError('Window size width needs to be power of 2.')

        if (self.windowSizeWidth%2 ==1):
            raise ValueError('Window size width needs to be even.')

#        if not is_power2(self.windowSizeHeight):
#            raise ValueError('Window size height needs to be power of 2.')

        if (self.windowSizeHeight%2 ==1):
            raise ValueError('Window size height needs to be even.')

        if not is_power2(self.zoomWindowSize):
            raise ValueError('Zoom window size needs to be a power of 2.')

        if not is_power2(self.oversamplingFactor):
            raise ValueError('Oversampling factor needs to be a power of 2.')

        if self.searchWindowSizeWidth >=  (2*self.windowSizeWidth):
            raise ValueError('Search Window Size Width should be < = 2 * Window Size Width')

        if self.searchWindowSizeHeight >= (2*self.searchWindowSizeHeight):
            raise ValueError('Search Window Size Height should be <= 2 * Window Size Height')

        if self.zoomWindowSize >= self.searchWindowSizeWidth:
            raise ValueError('Zoom window size should be <= Search window size width')

        if self.zoomWindowSize >= self.searchWindowSizeHeight:
            raise ValueError('Zoom window size should be <= Search window size height')

        return

    def checkImageLimits(self):
        '''
        Check if the first and last samples are set correctly.
        '''
        scaleFactorY = self.prf2 / self.prf1

        if (scaleFactorY < 0.9) or (scaleFactorY > 1.1):
            raise ValueError('Module designed for scale factors in range 0.9 - 1.1. Requested scale factor = %f'%(scaleFactorY))

        self.scaleFactorY = scaleFactorY


        scaleFactorX = self.rangeSpacing1 / self.rangeSpacing2
        if (scaleFactorX < 0.9) or (scaleFactorX > 1.1):
            raise ValueError('Module designed for scale factors in range 0.9 - 1.1. Requested scale factor = %d'%(scaleFactorX))

        self.scaleFactorX = scaleFactorX

        if self.firstSampleDown is None:
            self.firstSampleDown = 0 


        if self.lastSampleDown is None:
            self.lastSampleDown = self.fileLength1-1
       

        if self.firstSampleAcross is None:
            self.firstSampleAcross = 1
    
        if self.lastSampleAcross is None:
            self.lastSampleAcross = self.lineLength1-1 


        if self.firstSampleAcross < 0:
            raise ValueError('First sample of reference image is not positive.')

        if self.firstSampleDown < 0:
            raise ValueError('First line of reference image is not positive.')

        if self.lastSampleAcross  >= self.lineLength1:
            raise ValueError('Last sample of reference image is greater than line length.')

        if self.lastSampleDown >= self.fileLength1:
            raise ValueError('Last Line of reference image is greater than line length.')

        return

    def setState(self):
        denseoffsets.setLineLength1_Py(int(self.lineLength1))
        denseoffsets.setFileLength1_Py(int(self.fileLength1))
        denseoffsets.setLineLength2_Py(int(self.lineLength2))
        denseoffsets.setFileLength2_Py(int(self.fileLength2))
        denseoffsets.setFirstSampleAcross_Py(int(self.firstSampleAcross))
        denseoffsets.setLastSampleAcross_Py(int(self.lastSampleAcross))
        denseoffsets.setSkipSampleAcross_Py(int(self.skipSampleAcross))
        denseoffsets.setFirstSampleDown_Py(int(self.firstSampleDown))
        denseoffsets.setLastSampleDown_Py(int(self.lastSampleDown))
        denseoffsets.setSkipSampleDown_Py(int(self.skipSampleDown))
        denseoffsets.setAcrossGrossOffset_Py(int(self.acrossGrossOffset))
        denseoffsets.setDownGrossOffset_Py(int(self.downGrossOffset))
        denseoffsets.setScaleFactorX_Py(float(self.scaleFactorX))
        denseoffsets.setScaleFactorY_Py(float(self.scaleFactorY))
        denseoffsets.setDebugFlag_Py(self.debugFlag)

        denseoffsets.setWindowSizeWidth_Py(int(self.windowSizeWidth))
        denseoffsets.setWindowSizeHeight_Py(int(self.windowSizeHeight))
        denseoffsets.setSearchWindowSizeHeight_Py(int(self.searchWindowSizeHeight))
        denseoffsets.setSearchWindowSizeWidth_Py(int(self.searchWindowSizeWidth))
        denseoffsets.setZoomWindowSize_Py(self.zoomWindowSize)
        denseoffsets.setOversamplingFactor_Py(self.oversamplingFactor)
        denseoffsets.setIsComplex1_Py(int(self.isComplex1))
        denseoffsets.setIsComplex2_Py(int(self.isComplex2))
        denseoffsets.setBand1_Py(int(self.band1))
        denseoffsets.setBand2_Py(int(self.band2))
        denseoffsets.setNormalizeFlag_Py(int(self.normalize))

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

    def setSkipSampleAcross(self,var):
        self.skipSampleAcross = int(var)
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

    def setWindowSizeWidth(self, var):
        temp = int(var)
        if (temp%2 == 1):
            raise ValueError('Window size width needs to be even.')
        self.windowSizeWidth = temp

    def setWindowSizeHeight(self, var):
        temp = int(var)
        if (temp%2 == 1):
            raise ValueError('Window size height needs to be even.')
        self.windowSizeHeight = temp

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

    def setSearchWindowSizeWidth(self, var):
        temp = int(var)
        if (temp%2 == 1):
            raise ValueError('Search Window Size width needs to be even.')
        self.searchWindowSizeWidth = temp

    def setSearchWindowSizeHeight(self, var):
        temp = int(var)
        if (temp%2 == 1):
            raise ValueError('Search Window Size height needs to be even.')
        self.searchWindowSizeHeight = temp

    def setNormalizeFlag(self, var):
        self.normalize = var

    def __init__(self, name=''):
        super(DenseOffsets,self).__init__(family=self.__class__.family, name=name)
        self.lineLength1 = None
        self.lineLength2 = None
        self.fileLength1 = None
        self.fileLength2 = None
        self.scaleFactorX = None
        self.scaleFactorY = None
        self.outLines = None
        self.outSamples = None
        self.dictionaryOfVariables = {
                                      'LENGTH1' : ['lineLength1', 'int','mandatory'],
                                      'LENGTH2' : ['lineLength2', 'int', 'mandatory'],
                                      'F_LENGTH1' : ['fileLength1', 'int','mandatory'],
                                      'F_LENGTH2' : ['fileLength2', 'int', 'mandatory'],
                                      'FIRST_SAMPLE_ACROSS' : ['firstSampleAcross', 'int','mandatory'],
                                      'LAST_SAMPLE_ACROSS' : ['lastSampleAcross', 'int','mandatory'],
                                      'NUMBER_LOCATION_ACROSS' : ['numberLocationAcross', 'int','mandatory'],
                                      'FIRST_SAMPLE_DOWN' : ['firstSampleDown', 'int','mandatory'],
                                      'LAST_SAMPLE_DOWN' : ['lastSampleDown', 'int','mandatory'],
                                      'NUMBER_LOCATION_DOWN' : ['numberLocationDown', 'int','mandatory'],
                                      'ACROSS_GROSS_OFFSET' : ['acrossGrossOffset', 'int','optional'],
                                      'DOWN_GROSS_OFFSET' : ['downGrossOffset', 'int','optional'],
                                      'PRF1' : ['prf1', 'float','optional'],
                                      'PRF2' : ['prf2', 'float','optional'],
                                      'DEBUG_FLAG' : ['debugFlag', 'str','optional'],
                                      'SEARCH_WINDOW_SIZE' : ['searchWindowSize', 'int', 'optional'],
                                      'WINDOW_SIZE' : ['windowSize', 'int', 'optional']
                                      }
        self.dictionaryOfOutputVariables = {
                                            'LENGTH' : 'outLines', \
                                            'WIDTH'  : 'outSamples', \
                                            'OFFSETIMAGE' : 'offsetImageName', \
                                            'SNRIMAGE' : 'snrImageName'
                                           }
        self.descriptionOfVariables = {}
        self.mandatoryVariables = []
        self.optionalVariables = []
        self.initOptionalAndMandatoryLists()
        return

#end class




if __name__ == "__main__":
    from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
    from isceobj import Constants as CN

    def load_pickle(step='formslc'):
        import cPickle

        insarObj = cPickle.load(open('PICKLE/{0}'.format(step), 'rb'))
        return insarObj

    def runDenseOffset(insar):
        from isceobj.Catalog import recordInputs

        referenceFrame = insar.getReferenceFrame()
        secondaryFrame = insar.getSecondaryFrame()
        referenceOrbit = insar.getReferenceOrbit()
        secondaryOrbit = insar.getSecondaryOrbit()
        prf1 = referenceFrame.getInstrument().getPulseRepetitionFrequency()
        prf2 = secondaryFrame.getInstrument().getPulseRepetitionFrequency()
        nearRange1 = insar.formSLC1.startingRange
        nearRange2 = insar.formSLC2.startingRange
        fs1 = referenceFrame.getInstrument().getRangeSamplingRate()

        ###There seems to be no other way of determining image length - Piyush
        patchSize = insar.getPatchSize() 
        numPatches = insar.getNumberPatches()
        valid_az_samples =  insar.getNumberValidPulses()
        firstAc =  insar.getFirstSampleAcrossPrf()
        firstDown =  insar.getFirstSampleDownPrf()
        

        objSlc =  insar.getReferenceSlcImage()
        widthSlc = objSlc.getWidth()

        coarseRange = (nearRange1 - nearRange2) / (CN.SPEED_OF_LIGHT / (2 * fs1))
        coarseAcross = int(coarseRange + 0.5)
        if(coarseRange <= 0):
            coarseAcross = int(coarseRange - 0.5)


        time1, schPosition1, schVelocity1, offset1 = referenceOrbit._unpackOrbit()
        time2, schPosition2, schVelocity2, offset2 = secondaryOrbit._unpackOrbit()
        s1 = schPosition1[0][0]
        s1_2 = schPosition1[1][0]
        s2 = schPosition2[0][0]
        s2_2 = schPosition2[1][0]
        
        coarseAz = int(
            (s1 - s2)/(s2_2 - s2) + prf2*(1/prf1 - 1/prf2)*
            (patchSize - valid_az_samples)/2
            )
        coarseDown = int(coarseAz + 0.5)
        if(coarseAz <= 0):
            coarseDown = int(coarseAz - 0.5)
            pass
        

        coarseAcross = 0 + coarseAcross
        coarseDown = 0 + coarseDown

        mSlcImage = insar.getReferenceSlcImage()
        mSlc = isceobj.createSlcImage()
        IU.copyAttributes(mSlcImage, mSlc)
    #    scheme = 'BIL'
    #    mSlc.setInterleavedScheme(scheme)    #Faster access with bands
        accessMode = 'read'
        mSlc.setAccessMode(accessMode)
        mSlc.createImage()
        
        sSlcImage = insar.getSecondarySlcImage()
        sSlc = isceobj.createSlcImage()
        IU.copyAttributes(sSlcImage, sSlc)
    #    scheme = 'BIL'
    #    sSlc.setInterleavedScheme(scheme)   #Faster access with bands
        accessMode = 'read'
        sSlc.setAccessMode(accessMode)
        sSlc.createImage()

        objOffset = isceobj.Util.createDenseOffsets(name='dense')
      

        mWidth = mSlc.getWidth()
        sWidth = sSlc.getWidth()
        mLength = mSlc.getLength()
        sLength = sSlc.getLength()

        print('Gross Azimuth Offset: %d'%(coarseDown))
        print('Gross Range Offset: %d'%(coarseAcross))

        objOffset.setFirstSampleAcross(0)
        objOffset.setLastSampleAcross(mWidth-1)
        objOffset.setFirstSampleDown(0)
        objOffset.setLastSampleDown(mLength-1)
        objOffset.setSkipSampleAcross(20)
        objOffset.setSkipSampleDown(20)
        objOffset.setAcrossGrossOffset(int(coarseAcross))
        objOffset.setDownGrossOffset(int(coarseDown))

        ###Always set these values
        objOffset.setFirstPRF(prf1)
        objOffset.setSecondPRF(prf2)
        
        outImages = objOffset.denseoffsets(image1=mSlc,image2=sSlc,band1=0,band2=0)

        mSlc.finalizeImage()
        sSlc.finalizeImage()

        return


    ####The main program
    iObj = load_pickle()
    print('Done loading pickle')
    runDenseOffset(iObj)
