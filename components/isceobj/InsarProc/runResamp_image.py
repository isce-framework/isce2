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
# Author: Brett George
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import logging
import isceobj
import stdproc

from iscesys.ImageUtil.ImageUtil import ImageUtil as IU


logger = logging.getLogger('isce.insar.runResamp_image')

def runResamp_image(self):
    imageSlc =  self._insar.getReferenceSlcImage()
    widthSlc = max(self._insar.getReferenceSlcImage().getWidth(), self._insar.getSecondarySlcImage().getWidth())
    offsetField = self._insar.getRefinedOffsetField()                
    
    instrument = self._insar.getReferenceFrame().getInstrument()
    
    dopplerCoeff = self._insar.getDopplerCentroid().getDopplerCoefficients(inHz=False)
    
    pixelSpacing = self._insar.getSlantRangePixelSpacing()
    looks = self._insar.getNumberLooks() 
    lines = self._insar.getNumberResampLines() 
    numFitCoeff = self._insar.getNumberFitCoefficients() 
    
    offsetFilename = self._insar.getOffsetImageName()
    offsetAz = 'azimuth' + offsetFilename.capitalize()
    offsetRn = 'range' + offsetFilename.capitalize()
    widthOffset = int(widthSlc / looks)
    imageAz = isceobj.createOffsetImage()
    imageAz.setFilename(offsetAz)
    imageAz.setWidth(widthOffset)
    imageRn = isceobj.createOffsetImage()
    imageRn.setFilename(offsetRn)
    imageRn.setWidth(widthOffset)
   
    self._insar.setOffsetAzimuthImage(imageAz)
    self._insar.setOffsetRangeImage(imageRn)

    objAz = isceobj.createOffsetImage()
    objRn = isceobj.createOffsetImage()
    IU.copyAttributes(imageAz, objAz)
    IU.copyAttributes(imageRn, objRn)
    objAz.setAccessMode('write')
    objAz.createImage()
    objRn.setAccessMode('write')
    objRn.createImage()
    
    
    objResamp_image = stdproc.createResamp_image()
    objResamp_image.wireInputPort(name='offsets', object=offsetField)
    objResamp_image.wireInputPort(name='instrument', object=instrument)
    objResamp_image.setSlantRangePixelSpacing(pixelSpacing)
    objResamp_image.setDopplerCentroidCoefficients(dopplerCoeff)
    objResamp_image.setNumberLooks(looks)
    objResamp_image.setNumberLines(lines)
    objResamp_image.setNumberRangeBin(widthSlc)
    objResamp_image.setNumberFitCoefficients(numFitCoeff)
    #set the tag used in the outfile. each message is precided by this tag
    #is the writer is not of "file" type the call has no effect
    self._stdWriter.setFileTag("resamp_image", "log")
    self._stdWriter.setFileTag("resamp_image", "err")
    self._stdWriter.setFileTag("resamp_image", "out")
    objResamp_image.setStdWriter(self._stdWriter)

    objResamp_image.resamp_image(objRn, objAz)

    # Record the inputs and outputs
    from isceobj.Catalog import recordInputsAndOutputs
    recordInputsAndOutputs(self._insar.procDoc, objResamp_image, "runResamp_image", \
                  logger, "runResamp_image")

    objRn.finalizeImage()
    objAz.finalizeImage()
