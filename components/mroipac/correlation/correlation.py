#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Walter Szeliga
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import os
import ctypes
from iscesys.Component.Component import Component, Port
import operator



WINDOW_SIZE = Component.Parameter('windowSize',
        public_name='BOX_WIDTH',
        default=5,
        type = int,
        mandatory=False,
        doc = 'Width of the correlation estimation box')

SMOOTHING_WINDOW_WIDTH = Component.Parameter('smoothingWindowWidth',
        public_name='SMOOTHING_WINDOW_WIDTH',
        default=5,
        type = int,
        mandatory = False,
        doc = 'Width of the smoothing window box')

SMOOTHING_WINDOW_HEIGHT = Component.Parameter('smoothingWindowHeight',
        public_name='SMOOTHING_WINDOW_HEIGHT',
        default=5,
        type=int,
        mandatory=False,
        doc = 'Height of smoothing window')

GRADIENT_THRESHOLD = Component.Parameter('gradientThreshold',
        public_name = 'GRADIENT_THRESHOLD',
        default=0.0,
        type = float,
        mandatory=False,
        doc ='Gradient threshold for effective correlation calculation')

STD_DEV_THRESHOLD = Component.Parameter('stdDevThreshold',
        public_name = 'STD_DEV_THRESHOLD',
        default = 1.0,
        type = float,
        mandatory=False,
        doc = 'Phase std dev threshold for effective correlation calculation')

MAGNITUDE_THRESHOLD = Component.Parameter('magnitudeThreshold',
        public_name = 'MAGNITUDE_THRESHOLD',
        default = 5.0e-5,
        type=float,
        mandatory=False,
        doc = 'Magnitude threshold for effective correlation calculation')

GRADIENT_FILENAME = Component.Parameter('gradientFilename',
        public_name = 'GRADIENT_FILENAME',
        default = None,
        type = str,
        mandatory = False,
        doc = 'Name of gradient file if effective correlation is computed')

STDDEV_FILENAME = Component.Parameter('stddevFilename',
        public_name = 'STDDEV_FILENAME',
        default = None,
        type = str,
        mandatory = False,
        doc = 'Name of phase std dev file if effective correlation is computed')

COREGISTERED_SLC_FLAG = Component.Parameter('coregisteredSlcFlag',
        public_name = 'COREGISTERED SLC FLAG',
        default = False,
        type = bool,
        mandatory = False,
        doc = 'Flag to indicate inputs are coregistered SLCs and not int, amp')

class Correlation(Component):
    family = 'correlation'    
    logging_name = 'isce.mroipac.correlation'

    parameter_list = (WINDOW_SIZE,
                      SMOOTHING_WINDOW_WIDTH,
                      SMOOTHING_WINDOW_HEIGHT,
                      GRADIENT_THRESHOLD,
                      STD_DEV_THRESHOLD,
                      MAGNITUDE_THRESHOLD,
                      GRADIENT_FILENAME,
                      STDDEV_FILENAME,
                      COREGISTERED_SLC_FLAG)

    def __init__(self, name=''):
        super(Correlation, self).__init__(family=self.__class__.family, name=name)
        # Interferogram file
        self.interferogram = None
        # Amplitude file
        self.amplitude = None
        # Correlation file
        self.correlation = None
        #Slc1 file
        self.slc1 = None
        #Slc2 file
        self.slc2 = None

#        self.logger = logging.getLogger('isce.mroipac.correlation')
#        self.createPorts()

        return None

    def createPorts(self):
        interferogramPort = Port(name='interferogram', method=self.addInterferogram)
        amplitudePort = Port(name='amplitude', method=self.addAmplitude)
        correlationPort = Port(name='correlation', method=self.addCorrelation)
        slc1Port = Port(name='slc1', method=self.addSlc1)
        slc2Port = Port(name='slc2', method=self.addSlc2)

        self._inputPorts.add(interferogramPort)
        self._inputPorts.add(amplitudePort)
        self._inputPorts.add(slc1Port)
        self._inputPorts.add(slc2Port)
        self._outputPorts.add(correlationPort)
        return None
    
    def addInterferogram(self):
        ifg = self._inputPorts.getPort(name='interferogram').getObject()
        self.interferogram = ifg

    def addSlc1(self):
        ifg = self._inputPorts.getPort(name='slc1').getObject()
        self.slc1 = ifg

    def addSlc2(self):
        ifg = self._inputPorts.getPort(name='slc2').getObject()
        self.slc2 = ifg

    def addAmplitude(self):
        amp = self._inputPorts.getPort(name='amplitude').getObject()
        self.amplitude = amp

    def addCorrelation(self):
        cor = self._outputPorts.getPort(name='correlation').getObject()
        self.correlation = cor


    def calculateCorrelation(self):
        """
        Calculate the interferometric correlation using the maximum likelihood estimator.
        """
        from mroipac.correlation import correlationlib
        self.activateInputPorts()
        self.activateOutputPorts()

        if self.coregisteredSlcFlag:
            intAcc = self.slc1.getImagePointer()
            if intAcc is None:
                self.slc1.createImage()
                intAcc = self.slc1.getImagePointer()

            ampAcc = self.slc2.getImagePointer()
            if ampAcc is None:
                self.slc2.createImage()
                ampAcc = self.slc2.getImagePointer()
        else:
            intAcc = self.interferogram.getImagePointer()
            if intAcc is None:
                self.interferogram.createImage()
                intAcc = self.interferogram.getImagePointer()

            ampAcc = self.amplitude.getImagePointer()
            if ampAcc is None:
                self.amplitude.createImage()
                ampAcc = self.amplitude.getImagePointer()

        corAcc = self.correlation.getImagePointer()
        if corAcc is None:
            self.correlation.createImage()
            corAcc = self.correlation.getImagePointer()

        bx = int(self.windowSize)
        flag = int(self.coregisteredSlcFlag)
        
        self.logger.info("Calculating Correlation")
        correlationlib.correlation_Py(flag, intAcc, ampAcc, corAcc, bx)
        self.correlation.imageType = 'cor'
        self.correlation.renderHdr()

        return None

    '''
    def calculateCorrelation(self):
        """
        Calculate the interferometric correlation using the maximum likelihood estimator.
        """
        self.activateInputPorts()
        self.activateOutputPorts()

        if self.coregisteredSlcFlag:
            intFile_C = ctypes.c_char_p(bytes(self.slc1.getFilename(), 'utf-8'))
            ampFile_C = ctypes.c_char_p(bytes(self.slc2.getFilename(),'utf-8'))
            width_C = ctypes.c_int(self.slc1.getWidth())
        else:
            intFile_C = ctypes.c_char_p(bytes(self.interferogram.getFilename(), 'utf-8'))
            ampFile_C = ctypes.c_char_p(bytes(self.amplitude.getFilename(),'utf-8'))
            width_C = ctypes.c_int(self.interferogram.getWidth())

        corFile_C = ctypes.c_char_p(bytes(self.correlation.getFilename(),'utf-8'))
        bx_C = ctypes.c_int(int(self.windowSize))
        xmin_C = ctypes.c_int(0)
        xmax_C = ctypes.c_int(-1)
        ymin_C = ctypes.c_int(0)
        ymax_C = ctypes.c_int(-1)
        flag = ctypes.c_int(int(self.coregisteredSlcFlag))
        
        self.logger.info("Calculating Correlation")
        self.correlationlib.cchz_wave(flag,intFile_C,ampFile_C, corFile_C, width_C, bx_C, xmin_C, xmax_C, ymin_C, ymax_C) 
        self.correlation.imageType = 'cor'
        self.correlation.renderHdr()
        return None
    '''

    def calculateEffectiveCorrelation(self):
        """
        Calculate the effective correlation using the phase gradient.

        @param windowSize (\a int) The window size for calculating the phase gradient
        @param smoothingWindow (\a tuple) The range and azimuth smoothing window for the phase gradient
        @param gradientThreshold (\a float) The gradient threshold for phase gradient masking
        @param standardDeviationThreshold (\a float) The standard deviation threshold for phase gradient masking
        @param magnitudeThreshold (\a float) The magnitude threshold for phase gradient masking
        """
        self.activateInputPorts()
        self.activateOutputPorts()

        intFile = self.interferogram.getFilename()
        gradFile = self.gradientFilename
        stdFile = self.stddevFilename

        if gradFile is None:
            gradFile = os.path.splitext(intFile)[0] + '.grd'

        if stdFile is None:
            stdFile = os.path.splitext(intFile)[0] + '.std'

        intFile_C = ctypes.c_char_p(intFile)
        gradFile_C = ctypes.c_char_p(gradFile.name)
        stdFile_C = ctypes.c_char_p(stdFile.name)
        maskFile_C = ctypes.c_char_p(self.correlation.getFilename())
        width_C = ctypes.c_int(self.interferogram.getWidth())
        windowSize_C = ctype.c_int(int(self.windowSize))
        rangeSmoothing_C = ctype.c_int(int(self.smoothingWindowWidth))
        azimuthSmoothing_C = ctype.c_int(int(self.smoothingWindowHeight))
        gradThreshold_C = ctype.c_double(float(self.gradientThreshold))
        stdThreshold_C = ctype.c_double(float(self.stdDevThreshold))
        magThreshold_C = ctype.c_double(float(self.magnitudeThreshold))
        xmin_C = ctypes.c_int(0)
        xmax_C = ctypes.c_int(-1)
        ymin_C = ctypes.c_int(0)
        ymax_C = ctypes.c_int(-1)
        self.logger.info("Calculating Phase Gradient")
        self.correlationlib.phase_slope(intFile_C,gradFile_C,width_C,windowSize_C,gradThreshold_C,
                                  xmin_C,xmax_C,ymin_C,ymax_C)
        self.logger.info("Creating Phase Gradient Mask")
        self.correlationlib.phase_mask(intFile_C,gradFile_C,stdFile_C,stdThreshold_C,width_C,
                                 rangeSmoothing_C,azimuthSmoothing_C,xmin_C,xmax_C,ymin_C,ymax_C)
        self.correlationlib.magnitude_threshold(intFile_C,stdFile_C,maskFile_C,magThreshold_C,width_C)
