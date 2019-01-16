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
# Author: Piyush Agram
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



from __future__ import print_function
from iscesys.Component.Component import Component,Port
from iscesys.Compatibility import Compatibility
from stdproc.stdproc.offsetpoly import offsetpoly

class Offsetpoly(Component):
    
    def offsetpoly(self):
        self.numberOffsets = len(self.offset)
        self.allocateArrays()
        self.setState()
        offsetpoly.offsetpoly_Py()
        self.getState()
        self.deallocateArrays()

        return

    def setState(self):
        offsetpoly.setLocationAcross_Py(self.locationAcross,
                                     self.numberOffsets)
        offsetpoly.setOffset_Py(self.offset,
                                           self.numberOffsets)
        offsetpoly.setLocationDown_Py(self.locationDown, self.numberOffsets)
        offsetpoly.setSNR_Py(self.snr, self.numberOffsets)
        return

    def setNumberFitCoefficients(self, var):
        self.numberFitCoefficients = int(var)
        return


    def setLocationAcross(self, var):
        self.locationAcross = var
        return

    def setOffset(self, var):
        self.offset = var
        return

    def setLocationDown(self, var):
        self.locationDown = var
        return

    def setSNR(self, var):
        self.snr = var
        return

    def getState(self):
        self.offsetPoly = offsetpoly.getOffsetPoly_Py(
            self.numberFitCoefficients
            )
        return

    def allocateArrays(self):
        offsetpoly.allocateFieldArrays_Py(self.numberOffsets)
        offsetpoly.allocatePolyArray_Py(self.numberFitCoefficients)
        return

    def deallocateArrays(self):
        offsetpoly.deallocateFieldArrays_Py()
        offsetpoly.deallocatePolyArray_Py()
        return

    logging_name = 'isce.stdproc.offsetpoly'
    def __init__(self):
        super(Offsetpoly, self).__init__()
        self.numberFitCoefficients = 6
        self.numberOffsets = None 
        self.locationAcross = []
        self.offset=[]
        self.locationDown = []
        self.snr = []
        self.offsetPoly = []
        self.downOffsetPoly = []
        self.dictionaryOfVariables = { 
            'NUMBER_FIT_COEFFICIENTS' : ['self.numberFitCoefficients', 'int','optional'],
            'NUMBER_OFFSETS' : ['self.numberOffsets', 'int', 'mandatory'],
            }
        self.dictionaryOfOutputVariables = { 
            'OFFSET_POLYNOMIAL' : 'self.offsetPoly',
            }
        self.descriptionOfVariables = {}
        self.mandatoryVariables = []
        self.optionalVariables = []
        return
