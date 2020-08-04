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
import stdproc
import isceobj

from iscesys.ImageUtil.ImageUtil import ImageUtil as IU

logger = logging.getLogger('isce.self._insar.runResamp_only')

def runResamp_only(self):
    imageInt = self._insar.getResampIntImage()
    imageAmp = self._insar.getResampAmpImage()
    
    objInt = isceobj.createIntImage()
    objIntOut = isceobj.createIntImage()
    IU.copyAttributes(imageInt, objInt)
    IU.copyAttributes(imageInt, objIntOut)
    outIntFilename = self._insar.getResampOnlyImageName()
    objInt.setAccessMode('read')
    objIntOut.setFilename(outIntFilename)
    
    self._insar.setResampOnlyImage(objIntOut)
    
    objIntOut.setAccessMode('write')
    objInt.createImage()
    objIntOut.createImage()

    objAmp = isceobj.createAmpImage()
    objAmpOut = isceobj.createAmpImage()
    IU.copyAttributes(imageAmp, objAmp)
    IU.copyAttributes(imageAmp, objAmpOut)
    outAmpFilename = self.insar.resampOnlyAmpName
    objAmp.setAccessMode('read')
    objAmpOut.setFilename(outAmpFilename)
    
    self._insar.setResampOnlyAmp(objAmpOut)
    
    objAmpOut.setAccessMode('write')
    objAmp.createImage()
    objAmpOut.createImage()
    
    numRangeBin = objInt.getWidth() 
    lines = objInt.getLength() 
    instrument = self._insar.getReferenceFrame().getInstrument()
    
    offsetField = self._insar.getRefinedOffsetField()                
    
    
    dopplerCoeff = self._insar.getDopplerCentroid().getDopplerCoefficients(inHz=False)
    numFitCoeff = self._insar.getNumberFitCoefficients() 
    
    pixelSpacing = self._insar.getSlantRangePixelSpacing()

    objResamp = stdproc.createResamp_only()

    objResamp.setNumberLines(lines) 
    objResamp.setNumberFitCoefficients(numFitCoeff)
    objResamp.setSlantRangePixelSpacing(pixelSpacing)
    objResamp.setNumberRangeBin(numRangeBin)
    objResamp.setDopplerCentroidCoefficients(dopplerCoeff)
    
    objResamp.wireInputPort(name='offsets', object=offsetField)
    objResamp.wireInputPort(name='instrument', object=instrument)
    #set the tag used in the outfile. each message is precided by this tag
    #is the writer is not of "file" type the call has no effect
    self._stdWriter.setFileTag("resamp_only", "log")
    self._stdWriter.setFileTag("resamp_only", "err")
    self._stdWriter.setFileTag("resamp_only", "out")
    objResamp.setStdWriter(self._stdWriter)

    objResamp.resamp_only(objInt, objIntOut, objAmp, objAmpOut)

    # Record the inputs and outputs
    from isceobj.Catalog import recordInputsAndOutputs
    recordInputsAndOutputs(self._insar.procDoc, objResamp, "runResamp_only", \
                  logger, "runResamp_only")
    objInt.finalizeImage()
    objIntOut.finalizeImage()
    objAmp.finalizeImage()
    objAmpOut.finalizeImage()
