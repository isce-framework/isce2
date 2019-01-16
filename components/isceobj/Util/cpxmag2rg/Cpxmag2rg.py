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
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



from __future__ import print_function
import sys
import os
import math
from iscesys.Component.Component import Component
from isceobj.Image.RgImageBase import RgImage
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from isceobj.Util import cpxmag2rg

class Cpxmag2rg(Component):

    def cpxmag2rg(self, image1=None, image2=None, imageOut=None):
        
        if image1 is not None:
            self.image1 = image1
        if self.image1 is None:
            raise ValueError("Error. First slc image not set.")

        if image2 is not None:
            self.image2 = image2
        if self.image2 is None:
            raise ValueError("Error. Second slc image not set.")

        if imageOut is not None:
            self.imageOut = imageOut

        outImCreatedHere = False
        if self.imageOut is None:
            # allow to create the image output just by giving the name. we 
            # have all the info to do it
            if not self.imageOutName:
                raise ValueError(
                    "Error. Output image not set. Provide at least the name using the method setOutputImageName()."
                    )
            self.imageOut = self.createOutputImage()
            outImCreatedHere = True
        
        self.image1Accessor = self.image1.getImagePointer()
        self.image2Accessor = self.image2.getImagePointer()
        self.imageOutAccessor = self.imageOut.getImagePointer()
        self.fileLength = self.image1.getLength()
        self.lineLength = max(self.image1.getWidth(), self.image2.getWidth())
        self.setState()
        cpxmag2rg.cpxmag2rg_Py(self.image1Accessor,
                               self.image2Accessor,
                               self.imageOutAccessor)

        if outImCreatedHere:
            self.finalizeOutputImage(self.imageOut)

        self.imageOut.trueDataType = 'BANDED'
        self.imageOut.numBands = 2
        self.imageOut.bandScheme = 'BIP'
        self.imageOut.bandDataType = ['REAL4','REAL4']
        self.imageOut.bandDescription = ['','']
        self.imageOut.width = self.lineLength
        self.imageOut.length = self.fileLength
        self.imageOut.renderHdr()

        return None


    def createOutputImage(self):
        obj = RgImage()
        accessMode = "write"
        width = max(self.image1.getWidth(),self.image2.getWidth())
        #obj.initLineAccessor(self.imageOutName,accessMode,byteOrder,dataType,tileHeight,width)
        obj.setFilename(filename)
        obj.setAccessMode(accessMode)
        obj.setWidth(width)
        obj.createImage()
        return obj

    def finalizeOutputImage(self,obj):
        obj.finalizeImage()
        return

    def setState(self):
        cpxmag2rg.setStdWriter_Py(int(self.stdWriter))
        cpxmag2rg.setLineLength_Py(int(self.lineLength))
        cpxmag2rg.setFileLength_Py(int(self.fileLength))
        cpxmag2rg.setAcOffset_Py(int(self.acOffset))
        cpxmag2rg.setDnOffset_Py(int(self.dnOffset))

        return


    def setOutputImageName(self,var):
        self.imageOutName = var


    def setFirstImage(self,var):
        self.image1 = var
        return

    def setSecondImage(self,var):
        self.image2 = var
        return

    def setOutputImage(self,var):
        self.imageOut = var
        return


    def setAcOffset(self,var):
        self.acOffset = int(var)
        return

    def setDnOffset(self,var):
        self.dnOffset = int(var)
        return






    def __init__(self):
        Component.__init__(self)
        self.image1 = None
        self.image2 = None
        self.imageOut = None
        self.image1Accessor = None
        self.image2Accessor = None
        self.imageOutAccessor = None
        self.imageOutName =  ''
        self.lineLength = None
        self.fileLength = None
        self.acOffset = 0
        self.dnOffset = 0
        self.dictionaryOfVariables = { \
                                      'OUTPUT_IMAGE_NAME' : ['self.imageOutName', 'str','optional'], \
                                      'AC_OFFSET' : ['self.acOffset', 'int','optional'], \
                                      'DN_OFFSET' : ['self.dnOffset', 'int','optional'] \
                                      }
        self.descriptionOfVariables = {}
        self.mandatoryVariables = []
        self.optionalVariables = []
        typePos = 2
        for key , val in self.dictionaryOfVariables.items():
            if val[typePos] == 'mandatory':
                self.mandatoryVariables.append(key)
            elif val[typePos] == 'optional':
                self.optionalVariables.append(key)
            else:
                print('Error. Variable can only be optional or mandatory')
                raise Exception
        return
    pass
