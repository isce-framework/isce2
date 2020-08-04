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
from isceobj import Constants as CN
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU

logger = logging.getLogger('isce.insar.runResamp')

def runResamp(self):
    logger.info("Resampling interferogram")

    imageSlc1 =  self.insar.referenceSlcImage
    imageSlc2 =  self.insar.secondarySlcImage
    

    resampName = self.insar.resampImageName
    resampAmp = resampName + '.amp'
    resampInt = resampName + '.int'

    azLooks = self.insar.numberAzimuthLooks
    rLooks = self.insar.numberRangeLooks               

    objSlc1 = isceobj.createSlcImage()
    IU.copyAttributes(imageSlc1, objSlc1)
    objSlc1.setAccessMode('read')
    objSlc1.createImage()

    objSlc2 = isceobj.createSlcImage()
    IU.copyAttributes(imageSlc2,  objSlc2)
    objSlc2.setAccessMode('read')
    objSlc2.createImage()

    #slcWidth = max(imageSlc1.getWidth(), imageSlc2.getWidth())
    slcWidth = imageSlc1.getWidth()
    intWidth = int(slcWidth / rLooks)
    dataType = 'CFLOAT'

    objInt = isceobj.createIntImage()
    objInt.setFilename(resampInt)
    objInt.setWidth(intWidth)
    imageInt = isceobj.createIntImage()
    IU.copyAttributes(objInt, imageInt)

    objInt.setAccessMode('write')

    objInt.createImage()
    objAmp = isceobj.createAmpImage()
    objAmp.setFilename(resampAmp)
    objAmp.setWidth(intWidth)
    imageAmp = isceobj.createAmpImage()
    IU.copyAttributes(objAmp, imageAmp)
    
    objAmp.setAccessMode('write')
    objAmp.createImage()

    self.insar.resampIntImage = imageInt
    self.insar.resampAmpImage = imageAmp

    
    instrument = self.insar.referenceFrame.getInstrument()
    
    offsetField = self.insar.refinedOffsetField                
    
    lines = self.insar.numberResampLines
   
    ####Modified to deal with secondary PRF correctly
    dopplerCoeff = self.insar.dopplerCentroid.getDopplerCoefficients(inHz=True)
    for num in range(len(dopplerCoeff)):
        dopplerCoeff[num] /= self.insar.secondaryFrame.getInstrument().getPulseRepetitionFrequency()

    numFitCoeff = self.insar.numberFitCoefficients
    
#    pixelSpacing = self.insar.slantRangePixelSpacing
    fS = self._insar.getSecondaryFrame().getInstrument().getRangeSamplingRate()
    pixelSpacing = CN.SPEED_OF_LIGHT/(2.*fS) 

    objResamp = stdproc.createResamp()
    objResamp.setNumberLines(lines) 
    objResamp.setNumberFitCoefficients(numFitCoeff)
    objResamp.setNumberAzimuthLooks(azLooks)
    objResamp.setNumberRangeLooks(rLooks)
    objResamp.setSlantRangePixelSpacing(pixelSpacing)
    objResamp.setDopplerCentroidCoefficients(dopplerCoeff)

    objResamp.wireInputPort(name='offsets', object=offsetField)
    objResamp.wireInputPort(name='instrument', object=instrument)
    #set the tag used in the outfile. each message is precided by this tag
    #is the writer is not of "file" type the call has no effect
    objResamp.stdWriter = self._writer_set_file_tags("resamp", "log", "err", 
                                                     "out")
    objResamp.resamp(objSlc1, objSlc2, objInt, objAmp) 
    # Record the inputs and outputs
    from isceobj.Catalog import recordInputsAndOutputs
    recordInputsAndOutputs(self._insar.procDoc, objResamp, "runResamp",
                  logger, "runResamp")
    
    objInt.finalizeImage()
    objAmp.finalizeImage()
    objSlc1.finalizeImage()
    objSlc2.finalizeImage()

    return None
