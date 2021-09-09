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
import isceobj

STARTX = Component.Parameter('startX',
        public_name = 'STARTX',
        default = -1,
        type = int,
        mandatory = False,
        doc = 'Starting point in range for unwrapping. If negative, starts from middle of image.')

STARTY = Component.Parameter('startY',
        public_name = 'STARTY',
        default = -1,
        type = int,
        mandatory = False,
        doc = 'Starting point in azimuth for unwrapping. If negative, starts from middle of image.')

CORR_THRESHOLD = Component.Parameter('corrThreshold',
        public_name = 'CORR_THRESHOLD',
        default = 0.1,
        type = float,
        mandatory = False,
        doc = 'Coherence threshold for unwrapping.')

FLAG_FILE = Component.Parameter('flagFile',
        public_name = 'FLAG_FILE',
        default = None,
        type = str,
        mandatory = False,
        doc = 'Name of the flag file created')

MAX_BRANCH_LENGTH = Component.Parameter('maxBranchLength',
        public_name = 'MAX_BRANCH_LENGTH',
        default = 64,
        type = int,
        mandatory = False,
        doc = 'Maximum length of a branch')

COR_BANDS = Component.Parameter('corrFilebands',
        public_name = 'COR_BANDS',
        default = None,
        type = int,
        mandatory = False,
        doc = 'Number of bands in correlation file')

class Grass(Component):
    """This is a python interface to the grass unwrapper that comes with ROI_PAC."""

    family = 'grass'
    logging_name = 'isce.mroipac.grass'

    parameter_list = (STARTX,
                      STARTY,
                      CORR_THRESHOLD,
                      FLAG_FILE,
                      MAX_BRANCH_LENGTH,
                      COR_BANDS)

    def __init__(self, name=''):
        super(Grass, self).__init__(family=self.__class__.family, name=name)
        self.grasslib = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + '/libgrass.so')
        self.interferogram = None
        self.correlation = None
        self.unwrapped = None
        return None

    def createPorts(self):
        self.inputPorts['interferogram'] = self.addInterferogram
        self.inputPorts['correlation'] = self.addCorrelation
        self.inputPorts['unwrapped interferogram'] = self.addUnwrapped
        return None

    def addInterferogram(self):
        ifg = self.inputPorts['interferogram'] 
        self.interferogram = ifg

    def addCorrelation(self):
        cor = self.inputPorts['correlation']
        self.correlation = cor
        self.corrFilebands = cor.bands

    def addUnwrapped(self):
        unw = self.inputPorts['unwrapped interferogram']
        self.unwrapped = unw

    def unwrap(self,x=None,y=None,threshold=None):
        """
        Create a flag file from the correlation port, and unwrap the interferogram wired to the interferogram port.

        @param x (\a int) The pixel coordinate in the x direction (range) at which to begin unwrapping
        @param y (\a int) The pixel coordinate in the y direction (azimuth) at which to begin unwrapping
        @param threshold (\a float) The correlation threshold for mask generation, default 0.1
        """

        for item in self.inputPorts:
            item()
        ####Create a temporary file for storing flags
        flagFile = self.flagFile
        if flagFile is None:
            flagFile = os.path.splitext(self.interferogram.getFilename())[0] + '.msk'
        

        if threshold is None:
            threshold = self.corrThreshold

        if x is None:
            x = self.startX

        if y is None:
            y = self.startY

        self.makeFlagFile(flagFile,threshold=threshold)
        self.grass(flagFile,x=x,y=y)


    def makeFlagFile(self,flagFilename,threshold=None):
        """
        Create the flag file for masking out areas of low correlation.

        @param flagFilename (\a string) The file name for the output flag file
        @param threshold (\a float) The correlation threshold for mask generation, default 0.1
        """
        import shutil
#        self.activateInputPorts()

        if threshold is None:
            threshold = self.corrThreshold

        if flagFilename is None:
            flagFilename = os.path.splitext(self.interferogram.getFilename())[0] + '.msk'

        #####Old files need to be cleaned out
        #####Otherwise will use old result
        if os.path.exists(flagFilename):
            self.logger.warning('Old Mask File found. Will be deleted.')
            os.remove(flagFilename)
    
        intFile_C = ctypes.c_char_p(self.interferogram.getFilename().encode('utf-8'))
        flagFile_C = ctypes.c_char_p(flagFilename.encode('utf-8'))
        maskFile_C = ctypes.c_char_p(self.correlation.getFilename().encode('utf-8'))
        width_C = ctypes.c_int(self.interferogram.getWidth())         
        corThreshold_C = ctypes.c_double(threshold)
        bands_C = ctypes.c_int(self.corrFilebands)
        xmin_C = ctypes.c_int(0)
        xmax_C = ctypes.c_int(-1)
        ymin_C = ctypes.c_int(0)
        ymax_C = ctypes.c_int(-1)
        start_C = ctypes.c_int(1)
        mbl_C = ctypes.c_int(self.maxBranchLength)

        self.logger.info("Calculating Residues")
        self.grasslib.residues(intFile_C, flagFile_C, width_C, xmin_C, xmax_C, ymin_C, ymax_C)
        self.grasslib.trees(flagFile_C,width_C,mbl_C,start_C,xmin_C,xmax_C,ymin_C,ymax_C)
        self.grasslib.corr_flag(maskFile_C,flagFile_C,width_C,corThreshold_C,start_C,xmin_C,xmax_C,ymin_C,ymax_C, bands_C)
        ###Create ISCE XML for mask file
        ####Currently image API does not support UINT8
        mskImage = isceobj.createImage()
        mskImage.dataType = 'BYTE'
        mskImage.width  = self.interferogram.getWidth()
        mskImage.bands = 1
        mskImage.scheme = 'BSQ'
        mskImage.filename = flagFilename
        mskImage.accessMode = 'READ'
        mskImage.imageType = 'bsq'
        mskImage.renderHdr()

    def grass(self,flagFilename,x=None,y=None):
        """
        The grass unwrapping algorithm.

        @param flagFilename (\a string) The file name for the mask file.
        @param x (\a int) The pixel coordinate in the x direction (range) at which to begin unwrapping
        @param y (\a int) The pixel coordinate in the y direction (azimuth) at which to begin unwrapping
        @note If either the x or y coordinates are set to a value less than zero, the center of the image 
        in that coordinate will be chosen as the staring point for unwrapping.
        """
#        self.activateInputPorts()
#        self.activateOutputPorts()

        if x is None:
            x = self.startX

        if y is None:
            y = self.startY

        intFile_C = ctypes.c_char_p(self.interferogram.getFilename().encode('utf-8'))
        flagFile_C = ctypes.c_char_p(flagFilename.encode('utf-8'))
        unwFile_C = ctypes.c_char_p(self.unwrapped.getFilename().encode('utf-8'))
        width_C = ctypes.c_int(self.interferogram.getWidth())         
        xmin_C = ctypes.c_int(0)
        xmax_C = ctypes.c_int(-1)
        ymin_C = ctypes.c_int(0)
        ymax_C = ctypes.c_int(-1)
        start_C = ctypes.c_int(1)
        xinit_C = ctypes.c_int(x)
        yinit_C = ctypes.c_int(y)

        self.grasslib.grass(intFile_C,flagFile_C,unwFile_C,width_C,start_C,xmin_C,xmax_C,ymin_C,ymax_C,xinit_C,yinit_C)

        pass
