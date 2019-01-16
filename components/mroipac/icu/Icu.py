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
import isce
import isceobj
from iscesys.Component.Component import Component,Port
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from mroipac.icu import icu

DEFAULT_BUFFER_SIZE = 3700

AZIMUTH_BUFFER_SIZE = Component.Parameter('azimuthBufferSize',
        public_name = 'AZIMUTH_BUFFER_SIZE',
        default=None,
        type=int,
        mandatory=False,
        doc = 'Azimuth Buffer size for a single patch')

AZIMUTH_OVERLAP = Component.Parameter('overlap',
        public_name = 'AZIMUTH_OVERLAP',
        default = 200,
        type=int,
        mandatory = False,
        doc = 'Azimuth overlap between patches')


FILTERING_FLAG = Component.Parameter('filteringFlag',
        public_name = 'FILTERING_FLAG',
        default= False,
        type = bool,
        mandatory = False,
        doc = 'For extra filtering before unwrapping')

UNWRAPPING_FLAG = Component.Parameter('unwrappingFlag',
        public_name = 'UNWRAPPING_FLAG',
        default = True,
        type = bool,
        mandatory = False,
        doc = 'For unwrapping')

FILTER_TYPE = Component.Parameter('filterType',
        public_name = 'FILTER_TYPE',
        default = 'PS',
        type = str,
        mandatory = False,
        doc = 'Type of filter to use')

LOW_PASS_WINDOW_WIDTH = Component.Parameter('LPRangeWinSize',
        public_name = 'LOW_PASS_WINDOW_WIDTH',
        default = 5,
        type = int,
        mandatory = False,
        doc = 'Range window size for low pass filter')

LOW_PASS_WINDOW_HEIGHT = Component.Parameter('LPAzimuthWinSize',
        public_name = 'LOW_PASS_WINDOW_HEIGHT',
        default = 5,
        type = int,
        mandatory = False,
        doc = 'Azimuth window size for low pass filter')

PS_FILTER_EXPONENT = Component.Parameter('filterExponent',
        public_name = 'PS_FILTER_EXPONENT',
        default = 0.5,
        type = float,
        mandatory = False,
        doc = 'Filter exponent of power spectral filter')

USE_AMPLITUDE_FLAG = Component.Parameter('useAmplitudeFlag',
        public_name = 'USE_AMPLITUDE_FLAG',
        default = True,
        type = bool,
        mandatory = False,
        doc =  'Use amplitude information for filtering')

CORRELATION_TYPE = Component.Parameter('correlationType',
        public_name = 'CORRELATION_TYPE',
        default = 'PHASESIGMA',
        type = str,
        mandatory = False,
        doc = 'Type of correlation to use')

CORRELATION_WINDOW = Component.Parameter('correlationBoxSize',
        public_name = 'CORRELATION_BOX_SIZE',
        default = 5,
        type = int,
        mandatory = False,
        doc = 'Box size for correlation estimation')

PHASE_SIGMA_WINDOW = Component.Parameter('phaseSigmaBoxSize',
        public_name = 'PHASE_SIGMA_WINDOW',
        default = 5,
        type = int,
        mandatory = False,
        doc = 'Box size for phase sigma estimation')

PHASE_VAR_THRESHOLD = Component.Parameter('phaseVarThreshold',
        public_name = 'PHASE_VAR_THRESHOLD',
        default = 8.0,
        type = float,
        mandatory = False,
        doc = 'Phase variance threshold')


INIT_CORR_THRESHOLD = Component.Parameter('initCorrThreshold',
        public_name = 'INIT_CORR_THRESHOLD',
        default = 0.1,
        type = float,
        mandatory = False,
        doc = 'Initial coherence threshold')

MAX_CORR_THRESHOLD = Component.Parameter('corrThreshold',
        public_name = 'MAX_CORR_THRESHOLD',
        default = 0.9,
        type = float,
        mandatory = False,
        doc = 'Final coherence threshold')

CORR_THRESHOLD_INC = Component.Parameter('corrThresholdInc',
        public_name = 'CORR_THRESHOLD_INC',
        default = 0.1,
        type = float,
        mandatory = False,
        doc = 'Coherence increment')

USE_PHASE_GRADIENT = Component.Parameter('usePhaseGradient',
        public_name = 'USE_PHASE_GRADIENT',
        default = False,
        type = bool,
        mandatory = False,
        doc = 'Use phase gradient neutrons')

USE_INTENSITY = Component.Parameter('useIntensity',
        public_name = 'USE_INTENSITY',
        default = False,
        type = bool,
        mandatory = False,
        doc = 'Use intensity neutrons')

BOOTSTRAP_POINTS = Component.Parameter('bootstrapPoints',
        public_name = 'BOOTSTRAP_POINTS',
        default = 16,
        type = int,
        mandatory = False,
        doc = 'Number of points in range for bootstrapping')

BOOTSTRAP_LINES = Component.Parameter('bootstrapLines',
        public_name = 'BOOTSTRAP_LINES',
        default = 16,
        type = int,
        mandatory = False,
        doc = 'Number of points in azimuth for bootstrapping')

NUMBER_TREESETS = Component.Parameter('numTreeSets',
        public_name = 'NUMBER_TREESETS',
        default = 7,
        type = int,
        mandatory = False,
        doc = 'Number of tree sets for unwrapping')

TREE_TYPE = Component.Parameter('treeType',
        public_name = 'TREE_TYPE',
        default = 'GZW',
        type = str,
        mandatory = False,
        doc = 'Algorithm to use for unwrapping')

PHASE_GRADIENT_NEUTRON_THRESHOLD = Component.Parameter('phaseGradientNeutronThreshold',
        public_name = 'PHASE_GRADIENT_NEUTRON_THRESHOLD',
        default = 3.0,
        type = float,
        mandatory = False,
        doc = 'Phase gradient threshold for neutrons')


INTENSITY_NEUTRON_THRESHOLD = Component.Parameter('intensityNeutronThreshold',
        public_name = 'INTENSITY_NEUTRON_THRESHOLD',
        default = 8.0,
        type = float,
        mandatory = False,
        doc = 'Intensity variance threshold')

CORRELATION_NEUTRON_THRESHOLD = Component.Parameter('corrNeutronThreshold',
        public_name = 'CORRELATION_NEUTRON_THRESHOLD',
        default = 0.8,
        type = float,
        mandatory = False,
        doc = 'Correlation neutron threshold')

SINGLE_PATCH = Component.Parameter('singlePatch',
        public_name = 'SINGLE_PATCH',
        default = False,
        type=bool,
        mandatory = False,
        doc = 'Should unwrap whole data in one patch')


corrTypes = ['NONE', 'NOSLOPE', 'SLOPE', 'PHASESIGMA', 'ALL']
filtTypes = ['LP','PS']
unwTreeTypes = ['GZW', 'CC']

class Icu(Component):

    family = 'icu'
    logging_name = 'mroipac.icu'

    parameter_list = (AZIMUTH_BUFFER_SIZE,
                      AZIMUTH_OVERLAP,
                      FILTERING_FLAG,
                      UNWRAPPING_FLAG,
                      USE_AMPLITUDE_FLAG,
                      USE_PHASE_GRADIENT,
                      USE_INTENSITY,
                      FILTER_TYPE,
                      TREE_TYPE,
                      CORRELATION_TYPE,
                      LOW_PASS_WINDOW_WIDTH,
                      LOW_PASS_WINDOW_HEIGHT,
                      CORRELATION_WINDOW,
                      PHASE_SIGMA_WINDOW,
                      PS_FILTER_EXPONENT,
                      INIT_CORR_THRESHOLD,
                      CORR_THRESHOLD_INC,
                      MAX_CORR_THRESHOLD,
                      PHASE_VAR_THRESHOLD,
                      PHASE_GRADIENT_NEUTRON_THRESHOLD,
                      INTENSITY_NEUTRON_THRESHOLD,
                      CORRELATION_NEUTRON_THRESHOLD,
                      BOOTSTRAP_POINTS,
                      BOOTSTRAP_LINES,
                      SINGLE_PATCH,
                      NUMBER_TREESETS)


    def icu(self, intImage=None, ampImage=None, filtImage=None, unwImage=None,
            corrImage=None, gccImage=None, phsigImage=None, conncompImage=None):

        if (intImage == None):
            print("Error. interferogram image not set.")
            raise Exception

        if self.useAmplitudeFlag and (ampImage == None):
            print("Error. Use Amplitude flag but amplitude image not provided.")
            raise Exception

        if all(x is None for x in [filtImage, unwImage, corrImage, gccImage, phsigImage]):
            print("Error. No output files have been set.")
            raise Exception

        if self.treeType not in unwTreeTypes:
            raise ValueError(self.treeType + ' must be in ' + str(unwTreeTypes))

        if self.filterType not in filtTypes:
            raise ValueError( self.filterType + ' must be in ' + str(filtTypes))

        if self.correlationType not in corrTypes:
            raise ValueError( self.correlationType + ' must be in ' + str(corrTypes))

        self.intAcc = intImage.getImagePointer()

        if ampImage:
            self.ampAcc = ampImage.getImagePointer()

        if filtImage:
            self.filtAcc = filtImage.getImagePointer()

        if unwImage:
            self.unwAcc = unwImage.getImagePointer()

        if corrImage:
            self.corrAcc = corrImage.getImagePointer()

        if gccImage:
            self.gccAcc = gccImage.getImagePointer()

        if phsigImage:
            self.phsigAcc = phsigImage.getImagePointer()

        if conncompImage:
            self.conncompAcc = conncompImage.getImagePointer()


        self.width = intImage.getWidth()
        self.length = intImage.getLength()
        self.startSample = 0
        self.endSample =  self.width
        self.startingLine = 0

        if self.singlePatch:
            self.azimuthBufferSize = self.length + 200

        if self.azimuthBufferSize is None:
            self.azimuthBufferSize = DEFAULT_BUFFER_SIZE

        self.createImages(intImage)
        self.setState()

        icu.icu_Py(self.intAcc, self.ampAcc, self.filtAcc,
                self.corrAcc, self.gccAcc, self.phsigAcc,
                self.unwAcc, self.conncompAcc)

        self.finalizeImages()
        return


    def finalizeImages(self):
        '''
        Close any images that were created here and
        not provided by user.
        '''

        for img in self._createdHere:
            img.finalizeImage()
            img.renderHdr()

        self._createdHere = []

        return

    from isceobj.Util.decorators import use_api
    @use_api
    def createImages(self, intImage):
        '''
        Create any outputs that need to be generated always here.
        '''
        if (self.conncompAcc == 0) and self.unwrappingFlag:
            img = isceobj.createImage()
            img.filename = os.path.splitext(intImage.getFilename())[0] + '.conncomp'
            self.conncompFilename = img.filename
            img.dataType = 'BYTE'
            img.scheme = 'BIL'
            img.imageType = 'bil'
            img.width = intImage.getWidth()
            img.setAccessMode('WRITE')
            img.createImage()
            self._createdHere.append(img)

            self.conncompAcc = img.getImagePointer()


    def setState(self):

        icu.setWidth_Py(self.width)
        icu.setStartSample_Py(self.startSample)
        icu.setEndSample_Py(self.endSample)
        icu.setStartingLine_Py(self.startingLine)
        icu.setLength_Py(self.length)

        icu.setAzimuthBufferSize_Py(self.azimuthBufferSize)
        icu.setOverlap_Py(self.overlap)

        icu.setFilteringFlag_Py(int(self.filteringFlag))
        icu.setFilterType_Py(filtTypes.index(self.filterType.upper()))

        icu.setUnwrappingFlag_Py(int(self.unwrappingFlag))
        icu.setLPRangeWinSize_Py(self.LPRangeWinSize)
        icu.setLPAzimuthWinSize_Py(self.LPAzimuthWinSize)
        icu.setFilterExponent_Py(self.filterExponent)
        icu.setUseAmplitudeFlag_Py(int(self.useAmplitudeFlag))
        icu.setCorrelationType_Py(corrTypes.index(self.correlationType.upper()))

        icu.setCorrelationBoxSize_Py(self.correlationBoxSize)
        icu.setPhaseSigmaBoxSize_Py(self.phaseSigmaBoxSize)
        icu.setPhaseVarThreshold_Py(self.phaseVarThreshold)
        icu.setInitCorrThreshold_Py(self.initCorrThreshold)
        icu.setCorrThreshold_Py(self.corrThreshold)
        icu.setCorrThresholdInc_Py(self.corrThresholdInc)

        icu.setNeuTypes_Py(int(self.usePhaseGradient), int(self.useIntensity))

        icu.setNeuThreshold_Py(self.phaseGradientNeutronThreshold, self.intensityNeutronThreshold, self.corrNeutronThreshold)

        icu.setBootstrapSize_Py(self.bootstrapPoints, self.bootstrapLines)
        icu.setNumTreeSets_Py(self.numTreeSets)
        icu.setTreeType_Py(unwTreeTypes.index(self.treeType.upper()))

        return


    def __init__(self, name=''):
        super(Icu, self).__init__(family=self.__class__.family, name=name)
        self.width = None
        self.startSample = None
        self.endSample = None
        self.startingLine = None
        self.length = None

        self.intAcc = 0
        self.ampAcc = 0
        self.filtAcc = 0
        self.corrAcc = 0
        self.gccAcc = 0
        self.phsigAcc = 0
        self.unwAcc = 0
        self.conncompAcc = 0
        self.conncompFilename = ''

        self._createdHere = []

        return


#end class
if __name__ == "__main__":

    import isceobj
    from iscesys.ImageUtil.ImageUtil import ImageUtil as IU

    wid = 2659
    objInt = isceobj.createIntImage()
    objInt.initImage('test.int', 'read', wid)
    objInt.createImage()

    objAmp = isceobj.createAmpImage()
    objAmp.initImage('test.amp','read',wid)
    objAmp.createImage()

    objFilt = isceobj.createIntImage()
    objFilt.setFilename('test.filt')
    objFilt.setWidth(wid)
    objFilt.setAccessMode('write')
    objFilt.createImage()

    objUnw = isceobj.createAmpImage()
    objUnw.bands = 2
    objUnw.scheme = 'BIL'
    objUnw.setFilename('test.unw')
    objUnw.setWidth(wid)
    objUnw.setAccessMode('write')
    objUnw.createImage()

    icuObj = Icu()
    icuObj.filteringFlag = False
    icuObj.icu(intImage=objInt, ampImage=objAmp,
            filtImage=objFilt, unwImage=objUnw)


    objInt.finalizeImage()
    objAmp.finalizeImage()
    objFilt.finalizeImage()
    objUnw.finalizeImage()
