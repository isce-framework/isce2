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
import logging
import isceobj
from iscesys.Component.Component import Component, Port

class Filter(Component):

    logging_name = 'isce.mroipac.filter'
    def __init__(self):
        super(Filter, self).__init__()
        self.image = None
        self.filteredImage = None
        return None
        
    def createPorts(self):
        ifgPort = Port(name='interferogram', method=self.addInterferogram)
        filtPort = Port(name='filtered interferogram', method=self.addFilteredInterferogram)
        self._inputPorts.add(ifgPort)
        self._outputPorts.add(filtPort)
        return None

    def addInterferogram(self):
        ifg = self._inputPorts.getPort(name='interferogram').getObject()
        self.image = ifg

    def addFilteredInterferogram(self):
        filt = self._outputPorts.getPort(name='filtered interferogram').getObject()
        self.filteredImage = filt

    def goldsteinWerner(self, alpha=0.5):
        """
        Apply a power-spectral smoother to the phase of the
        interferogram.  This requires four steps, first, separate the
        magnitude and phase of the interferogram and save both bands.
        second, apply the power-spectral smoother to the original
        interferogram.  third, take the phase regions that were zero
        in the original image and apply them to the smoothed phase
        [possibly optional] fourth, combine the smoothed phase with
        the original magnitude, since the power-spectral filter
        distorts the magnitude.  

        @param alpha the smoothing parameter
        """
        self.activateInputPorts()
        self.activateOutputPorts()

        input = self.image.getFilename()
        output = self.filteredImage.getFilename()
        width = self.image.getWidth()
        length = self.image.getLength()

        self.logger.debug("width: %s" % (width))
        self.logger.debug("length: %s" % (length))
        self.logger.debug("input: %s" % (input))
        self.logger.debug("output: %s" % (output))
        self.logger.debug("filter strength: %s"%(alpha))

        # Filter the interferometric phase
        self.logger.info("Filtering interferogram")
        self._psfilt(input,output,width,length,alpha)
        self._rescale_magnitude(input,output,width,length)
        self.filteredImage.renderHdr()

    def _psfilt(self,input,output,width,length,alpha):
        """
        Actually apply the filter.

        @param input the input interferogram filename
        @param output the output interferogram filename
        @param width the number of samples in the range direction
        @param length the number of samples in the azimuth direction
        @param alpha the amount of smoothing
        """
        lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__),'libfilter.so'))

        input_c = ctypes.c_char_p(bytes(input,'utf-8')) # The input interferogram
        output_c = ctypes.c_char_p(bytes(output,'utf-8')) # The output smoothed interferogram
        alpha_c = ctypes.c_double(alpha)
        step_c = ctypes.c_int(16) # Stepsize in range and azimuth for the filter
        width_c = ctypes.c_int(width)
        length_c = ctypes.c_int(length)
        ymax_c = ctypes.c_int(length-1) # default to length
        ymin_c = ctypes.c_int(0)
        xmax_c = ctypes.c_int(width-1) # default to width
        xmin_c = ctypes.c_int(0)

        lib.psfilt(input_c,output_c,width_c,length_c,alpha_c,step_c,xmin_c,xmax_c,ymin_c,ymax_c)

    def _rescale_magnitude(self,input,output,width,length):
        """
        Rescale the magnitude output of the power-spectral filter using the
        original image magnitude, in place.

        @param input the original complex image
        @param output the smoothed complex image
        @param width the number of samples in the range direction
        @param length the number of samples in the azimuth direction
        """
        lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__),'libfilter.so'))

        input_c = ctypes.c_char_p(bytes(input,'utf-8')) # The input interferogram
        output_c = ctypes.c_char_p(bytes(output,'utf-8')) # The output smoothed interferogram
        width_c = ctypes.c_int(width)
        length_c = ctypes.c_int(length)
   
        lib.rescale_magnitude(input_c,output_c,width_c,length_c)
