#!/usr/bin/env python3 

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





import sys
import os
import math
import isce
import isceobj
from iscesys.Component.Component import Component
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from mroipac.looks import looks

class Looks(Component):

    def looks(self):
       
        inImage = self.inputImage.clone()
        inImage.setAccessMode('READ') 
        inImage.setCaster('read', inImage.dataType)
        inImage.createImage()
        outWidth = inImage.getWidth() // self.acrossLooks
        outLength = inImage.getLength() // self.downLooks
        
        outImage = self.inputImage.clone()
        #if the image is not a geo the part below will fail
        try:
            outImage.coord1.coordDelta = self.inputImage.coord1.coordDelta * self.acrossLooks
            outImage.coord2.coordDelta = self.inputImage.coord2.coordDelta * self.downLooks
            outImage.coord1.coordStart = self.inputImage.coord1.coordStart + \
                        0.5*(self.acrossLooks - 1)*self.inputImage.coord1.coordDelta
            outImage.coord2.coordStart = self.inputImage.coord2.coordStart + \
                        0.5*(self.downLooks - 1)*self.inputImage.coord2.coordDelta

        except:
            pass
        
        outImage.setWidth(outWidth)
        #need to do this since if length != 0 when calling createImage it
        #performs a sanity check on the filesize on disk and the size obtained from the meta
        #and exits if not consistent
        outImage.setLength(0)
        outImage.setFilename(self.outputFilename)
        outImage.setAccessMode('WRITE')

        outImage.setCaster('write', inImage.dataType)
        outImage.createImage()
        outImage.createFile(outLength)

        inPtr = inImage.getImagePointer()
        outPtr = outImage.getImagePointer()

        looks.looks_Py(inPtr, outPtr, self.downLooks, self.acrossLooks, inImage.dataType.upper())

        inImage.finalizeImage()
        outImage.finalizeImage()
        outImage.renderHdr()

        return outImage

    def setInputImage(self,var):
        self.inputImage = var
        return

    def setAcrossLooks(self, var):
        self.acrossLooks = int(var)
        return

    def setDownLooks(self, var):
        self.downLooks = int(var)

    def setOutputFilename(self, var):
        self.outputFilename = str(var)

    def __init__(self):
        Component.__init__(self)
        self.acrossLooks = None
        self.downLooks= None
        self.inputImage = None
        self.outputFilename = None

#end class




if __name__ == "__main__":
    sys.exit(main())
