#! /usr/bin/env python 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2014 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Brent Minchew
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
import numpy as np
import multiprocessing as mp
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from isceobj.Util.decorators import use_api


def getThreadCount():
    '''
    Return number of threads available.
    '''

    cpus = os.cpu_count()

    try:
        ompnum = int(os.environ['OMP_NUM_THREADS'])
    except KeyError:
        ompnum = None

    if ompnum is None:
        return cpus
    else:
        return ompnum




def intround(n):
    if (n <= 0):
        return int(n-0.5)
    else:
        return int(n+0.5)

logger = logging.getLogger('mroipac.ampcor.denseampcor')

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

SKIP_SAMPLE_ACROSS = Component.Parameter('skipSampleAcross',
        public_name = 'SKIP_SAMPLE_ACROSS',
        default = None,
        type = int,
        mandatory = False,
        doc = 'Number of samples to skip in range direction')

SKIP_SAMPLE_DOWN = Component.Parameter('skipSampleDown',
        public_name = 'SKIP_SAMPLE_DOWN',
        default = None,
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
        doc = 'Image data type for reference image (complex / real/ mag)')

IMAGE_DATATYPE2 = Component.Parameter('imageDataType2',
        public_name = 'IMAGE_DATATYPE2',
        default='',
        type = str,
        mandatory=False,
        doc = 'Image data type for search image (complex / real/ mag)')

IMAGE_SCALING_FACTOR = Component.Parameter('scaling_factor',
        public_name = 'IMAGE_SCALING_FACTOR',
        default = 1.0,
        type = float,
        mandatory=False,
        doc = 'Image data scaling factor (unit magnitude conversion from pixels)')

SNR_THRESHOLD = Component.Parameter('thresholdSNR',
        public_name = 'SNR_THRESHOLD',
        default = 0.0,
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

OFFSET_IMAGE_NAME = Component.Parameter('offsetImageName',
        public_name='OFFSET_IMAGE_NAME',
        default='dense_ampcor.bil',
        type=str,
        mandatory=False,
        doc = 'File name for two channel output')

SNR_IMAGE_NAME = Component.Parameter('snrImageName',
        public_name = 'SNR_IMAGE_NAME',
        default = 'dense_ampcor_snr.bil',
        type=str,
        mandatory=False,
        doc = 'File name for output SNR')

COV_IMAGE_NAME = Component.Parameter('covImageName',
        public_name = 'COV_IMAGE_NAME',
        default = 'dense_ampcor_cov.bil',
        type=str,
        mandatory=False,
        doc = 'File name for output covariance')

MARGIN = Component.Parameter('margin',
        public_name = 'MARGIN',
        default = 50,
        type = int,
        mandatory=False,
        doc = 'Margin around the edge of the image to avoid')

NUMBER_THREADS = Component.Parameter('numberThreads',
        public_name = 'NUMBER_THREADS',
        default=getThreadCount(),
        type=int,
        mandatory=False,
        doc = 'Number of parallel ampcor threads to launch')


class DenseAmpcor(Component):

    family = 'denseampcor'
    logging_name = 'isce.mroipac.denseampcor'

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
                      SKIP_SAMPLE_ACROSS,
                      SKIP_SAMPLE_DOWN,
                      DOWN_SPACING_PRF1,
                      DOWN_SPACING_PRF2,
                      ACROSS_SPACING1,
                      ACROSS_SPACING2,
                      IMAGE_DATATYPE1,
                      IMAGE_DATATYPE2,
                      IMAGE_SCALING_FACTOR,
                      SNR_THRESHOLD,
                      COV_THRESHOLD,
                      BAND1,
                      BAND2,
                      OFFSET_IMAGE_NAME,
                      SNR_IMAGE_NAME,
                      COV_IMAGE_NAME,
                      MARGIN,
                      NUMBER_THREADS)

    @use_api
    def denseampcor(self,slcImage1 = None,slcImage2 = None):
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
        coarseAcross = self.acrossGrossOffset
        coarseDown = self.downGrossOffset

        xMargin = 2*self.searchWindowSizeWidth + self.windowSizeWidth
        yMargin = 2*self.searchWindowSizeHeight + self.windowSizeHeight

        #####Set image limits for search
        offAc = max(self.margin,-coarseAcross)+xMargin
        if offAc % self.skipSampleAcross != 0:
            leftlim = offAc
            offAc = self.skipSampleAcross*(1 + int(offAc/self.skipSampleAcross)) - self.pixLocOffAc
            while offAc < leftlim:
                offAc += self.skipSampleAcross

        offDn = max(self.margin,-coarseDown)+yMargin
        if offDn % self.skipSampleDown != 0:
            toplim = offDn
            offDn = self.skipSampleDown*(1 + int(offDn/self.skipSampleDown)) - self.pixLocOffDn
            while offDn < toplim:
                offDn += self.skipSampleDown

        offAcmax = int(coarseAcross + ((self.rangeSpacing1/self.rangeSpacing2)-1)*self.lineLength1)
        lastAc = int(min(self.lineLength1, self.lineLength2-offAcmax) - xMargin -1 - self.margin) 

        offDnmax = int(coarseDown + ((self.prf2/self.prf1)-1)*self.fileLength1)
        lastDn = int(min(self.fileLength1, self.fileLength2-offDnmax)  - yMargin -1 - self.margin) 


        self.gridLocAcross = range(offAc + self.pixLocOffAc, lastAc - self.pixLocOffAc, self.skipSampleAcross)
        self.gridLocDown = range(offDn + self.pixLocOffDn, lastDn - self.pixLocOffDn, self.skipSampleDown)

        startAc, endAc = offAc, self.gridLocAcross[-1] - self.pixLocOffAc
        self.numLocationAcross = int((endAc-startAc)/self.skipSampleAcross + 1)
        self.numLocationDown = len(self.gridLocDown)

        self.offsetCols, self.offsetLines = self.numLocationAcross, self.numLocationDown

        print('Pixels: ', self.lineLength1, self.lineLength2)
        print('Lines: ', self.fileLength1, self.fileLength2)
        print('Wins : ', self.windowSizeWidth, self.windowSizeHeight)
        print('Srch: ', self.searchWindowSizeWidth, self.searchWindowSizeHeight)


        #####Create shared memory objects
        numlen = self.numLocationAcross * self.numLocationDown
        self.locationDown = np.frombuffer(mp.Array('i', numlen).get_obj(), dtype='i')
        self.locationDownOffset = np.frombuffer(mp.Array('f', numlen).get_obj(), dtype='f')
        self.locationAcross = np.frombuffer(mp.Array('i', numlen).get_obj(), dtype='i')
        self.locationAcrossOffset = np.frombuffer(mp.Array('f', numlen).get_obj(), dtype='f')
        self.snr = np.frombuffer(mp.Array('f', numlen).get_obj(), dtype='f')
        self.cov1 = np.frombuffer(mp.Array('f', numlen).get_obj(), dtype='f')
        self.cov2 = np.frombuffer(mp.Array('f', numlen).get_obj(), dtype='f')
        self.cov3 = np.frombuffer(mp.Array('f', numlen).get_obj(), dtype='f')

        self.locationDownOffset[:] = -10000.0
        self.locationAcrossOffset[:] = -10000.0
        self.snr[:] = 0.0
        self.cov1[:] = 999.0
        self.cov2[:] = 999.0
        self.cov3[:] = 999.0

        ###run ampcor on parallel processes
        threads = []
        nominal_load = self.numLocationDown // self.numberThreads
        flat_indices = np.arange(numlen).reshape((self.numLocationDown,self.numLocationAcross))
        ofmt = 'Thread %d: %7d%7d%7d%7d%7d%7d'
        for thrd in range(self.numberThreads):

            # Determine location down grid indices for thread
            if thrd == self.numberThreads - 1:
                proc_num_grid = self.numLocationDown - thrd * nominal_load
            else:
                proc_num_grid = nominal_load
            istart = thrd * nominal_load
            iend = istart + proc_num_grid

            # Compute corresponding global line/down indices
            proc_loc_down = self.gridLocDown[istart:iend]
            startDown, endDown = proc_loc_down[0], proc_loc_down[-1]
            numDown = int((endDown - startDown)//self.skipSampleDown + 1)

            # Get flattened grid indices
            firstind = flat_indices[istart:iend,:].ravel()[0]
            lastind = flat_indices[istart:iend,:].ravel()[-1]

            print(ofmt % (thrd, firstind, lastind, startAc, endAc, startDown, endDown))

            # Launch job
            args = (startAc,endAc,startDown,endDown,self.numLocationAcross,
                numDown,firstind,lastind)
            threads.append(mp.Process(target=self._run_ampcor, args=args))
            threads[-1].start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        self.firstSampAc, self.firstSampDown = self.locationAcross[0], self.locationDown[0]
        self.lastSampAc, self.lastSampDown = self.locationAcross[-1], self.locationDown[-1]

        #### Scale images (default is 1.0 to keep as pixel)
        self.locationDownOffset *= self.scaling_factor
        self.locationAcrossOffset *= self.scaling_factor

        self.write_slantrange_images()


    def _run_ampcor(self, firstAc, lastAc, firstDn, lastDn,
                        numAc, numDn, firstind, lastind):
        '''
        Individual calls to ampcor.
        '''

        os.environ['VRT_SHARED_SOURCE'] = "0"

        objAmpcor = Ampcor()

        objAmpcor.setWindowSizeWidth(self.windowSizeWidth)
        objAmpcor.setWindowSizeHeight(self.windowSizeHeight)
        objAmpcor.setSearchWindowSizeWidth(self.searchWindowSizeWidth)
        objAmpcor.setSearchWindowSizeHeight(self.searchWindowSizeHeight)
        objAmpcor.setImageDataType1(self.imageDataType1)
        objAmpcor.setImageDataType2(self.imageDataType2)

        objAmpcor.setFirstSampleAcross(firstAc)
        objAmpcor.setLastSampleAcross(lastAc)
        objAmpcor.setNumberLocationAcross(numAc)

        objAmpcor.setFirstSampleDown(firstDn)
        objAmpcor.setLastSampleDown(lastDn)
        objAmpcor.setNumberLocationDown(numDn)

        objAmpcor.setAcrossGrossOffset(self.acrossGrossOffset)
        objAmpcor.setDownGrossOffset(self.downGrossOffset)
        objAmpcor.setFirstPRF(self.prf1)
        objAmpcor.setSecondPRF(self.prf2)
        objAmpcor.setFirstRangeSpacing(self.rangeSpacing1)
        objAmpcor.setSecondRangeSpacing(self.rangeSpacing2)
        objAmpcor.thresholdSNR = 1.0e-6
        objAmpcor.thresholdCov = self.thresholdCov
        objAmpcor.oversamplingFactor = self.oversamplingFactor 
        
        mSlc = isceobj.createImage()
        IU.copyAttributes(self.slcImage1, mSlc)
        mSlc.setAccessMode('read')
        mSlc.createImage()

        sSlc = isceobj.createImage()
        IU.copyAttributes(self.slcImage2, sSlc)
        sSlc.setAccessMode('read')
        sSlc.createImage()

        objAmpcor.ampcor(mSlc, sSlc)
        mSlc.finalizeImage()
        sSlc.finalizeImage()

        j = 0 
        length = len(objAmpcor.locationDown)
        for i in range(lastind-firstind):
            acInd = firstAc + self.pixLocOffAc + (i % numAc)*self.skipSampleAcross
            downInd = firstDn + self.pixLocOffDn + (i//numAc)*self.skipSampleDown
        
            if j < length and objAmpcor.locationDown[j] == downInd and objAmpcor.locationAcross[j] == acInd: 
                self.locationDown[firstind+i] = objAmpcor.locationDown[j]
                self.locationDownOffset[firstind+i] = objAmpcor.locationDownOffset[j]
                self.locationAcross[firstind+i] = objAmpcor.locationAcross[j]
                self.locationAcrossOffset[firstind+i] = objAmpcor.locationAcrossOffset[j]
                self.snr[firstind+i] = objAmpcor.snrRet[j]
                self.cov1[firstind+i] = objAmpcor.cov1Ret[j]
                self.cov2[firstind+i] = objAmpcor.cov2Ret[j]
                self.cov3[firstind+i] = objAmpcor.cov3Ret[j]
                j += 1
            else:
                self.locationDown[firstind+i] = downInd
                self.locationDownOffset[firstind+i] = -10000.
                self.locationAcross[firstind+i] = acInd
                self.locationAcrossOffset[firstind+i] = -10000.
                self.snr[firstind+i] = 0.
                self.cov1[firstind+i] = 999.
                self.cov2[firstind+i] = 999.
                self.cov3[firstind+i] = 999.

        return


    def write_slantrange_images(self):
        '''Write output images'''

        ####Snsure everything is 2D image first

        if self.locationDownOffset.ndim == 1:
            self.locationDownOffset = self.locationDownOffset.reshape(-1,self.offsetCols)

        if self.locationAcrossOffset.ndim == 1:
            self.locationAcrossOffset = self.locationAcrossOffset.reshape(-1,self.offsetCols)

        if self.snr.ndim == 1:
            self.snr = self.snr.reshape(-1,self.offsetCols)

        if self.locationDown.ndim == 1:
            self.locationDown = self.locationDown.reshape(-1,self.offsetCols)

        if self.locationAcross.ndim == 1:
            self.locationAcross = self.locationAcross.reshape(-1,self.offsetCols)

        if self.cov1.ndim == 1:
            self.cov1 = self.cov1.reshape(-1,self.offsetCols)

        if self.cov2.ndim == 1:
            self.cov2 = self.cov2.reshape(-1,self.offsetCols)

        if self.cov3.ndim == 1:
            self.cov3 = self.cov3.reshape(-1,self.offsetCols)

        outdata = np.empty((2*self.offsetLines, self.offsetCols), dtype=np.float32)
        outdata[::2,:] = self.locationDownOffset
        outdata[1::2,:] = self.locationAcrossOffset
        outdata.tofile(self.offsetImageName)
        del outdata
        outImg = isceobj.createImage()
        outImg.setDataType('FLOAT')
        outImg.setFilename(self.offsetImageName)
        outImg.setBands(2)
        outImg.scheme = 'BIL'
        outImg.setWidth(self.offsetCols)
        outImg.setLength(self.offsetLines)
        outImg.setAccessMode('read')
        outImg.renderHdr()

        ####Create SNR image
        self.snr.astype(np.float32).tofile(self.snrImageName)
        snrImg = isceobj.createImage()
        snrImg.setFilename(self.snrImageName)
        snrImg.setDataType('FLOAT')
        snrImg.setBands(1)
        snrImg.setWidth(self.offsetCols)
        snrImg.setLength(self.offsetLines)
        snrImg.setAccessMode('read')
        snrImg.renderHdr()

        ####Create covariance image
        covdata = np.empty((3*self.offsetLines, self.offsetCols), dtype=np.float32)
        covdata[::3,:] = self.cov1
        covdata[1::3,:] = self.cov2
        covdata[2::3,:] = self.cov3
        covdata.tofile(self.covImageName)
        del covdata
        covImg = isceobj.createImage()
        covImg.setDataType('FLOAT')
        covImg.setFilename(self.covImageName)
        covImg.setBands(3)
        covImg.scheme = 'BIL'
        covImg.setWidth(self.offsetCols)
        covImg.setLength(self.offsetLines)
        covImg.setAccessMode('read')
        covImg.renderHdr()

    def checkTypes(self):
        '''Check if the image datatypes are set.'''

        if self.imageDataType1 == '':
            if self.slcImage1.getDataType().upper().startswith('C'):
                self.imageDataType1 = 'complex'
            else:
                raise ValueError('Undefined value for imageDataType1. Should be complex/real/mag')
        else:
            if self.imageDataType1 not in ('complex','real','mag'):
                raise ValueError('ImageDataType1 should be either complex/real/rmg1/rmg2.')

        if self.imageDataType2 == '':
            if self.slcImage2.getDataType().upper().startswith('C'):
                self.imageDataType2 = 'complex'
            else:
                raise ValueError('Undefined value for imageDataType2. Should be complex/real/mag')
        else:
            if self.imageDataType2 not in ('complex','real','mag'):
                raise ValueError('ImageDataType1 should be either complex/real/mag.')
        

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

        if self.searchWindowSizeWidth >=  2*self.windowSizeWidth :
            raise ValueError('Search Window Size Width should be < 2 * Window Size Width')

        if self.searchWindowSizeHeight >= 2*self.windowSizeHeight :
            raise ValueError('Search Window Size Height should be < 2 * Window Size Height')

        if self.zoomWindowSize > min(self.searchWindowSizeWidth*2+1, self.searchWindowSizeHeight*2+1):
            raise ValueError('Zoom window size should be <= Search window size * 2 + 1')

        if self._stdWriter is None:
            self._stdWriter = create_writer("log", "", True, filename="denseampcor.log")

        self.pixLocOffAc = self.windowSizeWidth//2 + self.searchWindowSizeWidth - 1
        self.pixLocOffDn = self.windowSizeHeight//2 + self.searchWindowSizeHeight - 1

    def setImageDataType1(self, var):
        self.imageDataType1 = str(var)
        return

    def setImageDataType2(self, var):
        self.imageDataType2 = str(var)
        return

    def setImageScalingFactor(self, var):
        self.scaling_factor = float(var)
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

    def setSkipSampleAcross(self,var):
        self.skipSampleAcross = int(var)
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

    def __init__(self, name=''):
        super(DenseAmpcor, self).__init__(family=self.__class__.family, name=name)
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
        self.firstSampAc = None
        self.lastSampAc = None
        self.firstSampDown = None
        self.lastSampDown = None
        self.numLocationAcross = None
        self.numLocationDown = None
        self.offsetCols = None
        self.offsetLines = None
        self.gridLocAcross = None
        self.gridLocDown = None
        self.pixLocOffAc = None
        self.pixLocOffDn = None
        self._stdWriter = None
        self.offsetLines = None
        self.offsetCols = None
        self.dictionaryOfVariables = { \
                                      'IMAGETYPE1' : ['imageDataType1', 'str', 'optional'], \
                                      'IMAGETYPE2' : ['imageDataType2', 'str', 'optional'], \
                                      'IMAGE_SCALING_FACTOR' : ['scaling_factor', 'float', 'optional'], \
                                      'SKIP_SAMPLE_ACROSS' : ['skipSampleAcross', 'int','mandatory'], \
                                      'SKIP_SAMPLE_DOWN' : ['skipSampleDown', 'int','mandatory'], \
                                      'COARSE_NUMBER_LOCATION_ACROSS' : ['coarseNumWinAcross','int','mandatory'], \
                                      'COARSE_NUMBER_LOCATION_DOWN' : ['coarseNumWinDown', 'int', 'mandatory'], \
                                      'ACROSS_GROSS_OFFSET' : ['acrossGrossOffset', 'int','optional'], \
                                      'DOWN_GROSS_OFFSET' : ['downGrossOffset', 'int','optional'], \
                                      'PRF1' : ['prf1', 'float','optional'], \
                                      'PRF2' : ['prf2', 'float','optional'], \
                                      'RANGE_SPACING1' : ['rangeSpacing1', 'float', 'optional'], \
                                      'RANGE_SPACING2' : ['rangeSpacing2', 'float', 'optional'], \
                                      }
        self.dictionaryOfOutputVariables = {
                                            'FIRST_SAMPLE_ACROSS' : 'firstSampAc',
                                            'FIRST_SAMPLE_DOWN' : 'firstSampDn',
                                            'NUMBER_LINES': 'offsetLines',
                                            'NUMBER_PIXELS' : 'offsetCols'}
        return None


#end class
if __name__ == "__main__":
    sys.exit(main())
