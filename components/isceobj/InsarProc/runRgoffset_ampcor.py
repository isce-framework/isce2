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
# Author: Pietro Milillo
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# Comment: Adapted from InsarProc/runRgoffsetprf.py
import logging
import isceobj
import mroipac

from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from mroipac.ampcor.Ampcor import Ampcor
from isceobj import Constants as CN

logger = logging.getLogger('isce.insar.runRgoffset')

def runRgoffset(self):
    numLocationAcross =  self._insar.getNumberLocationAcrossPrf()
    numLocationDown =  self._insar.getNumberLocationDownPrf()
    firstAc = self._insar.getFirstSampleAcrossPrf()
    firstDown = self._insar.getFirstSampleDownPrf()
  
    #Fake amplitude image as a complex image
    imageAmp = self._insar.getResampAmpImage()
    objAmp = isceobj.createImage()
    objAmp.setAccessMode('read')
    objAmp.dataType = 'CFLOAT'
    objAmp.bands = 1
    objAmp.setFilename(imageAmp.filename)
    objAmp.setWidth(imageAmp.width)
    objAmp.createImage()
    widthAmp = objAmp.getWidth()
    intLength = objAmp.getLength()

    imageSim = self._insar.getSimAmpImage()
    objSim = isceobj.createImage()
    objSim.setFilename(imageSim.filename)
    objSim.setWidth(imageSim.width)
    objSim.dataType='FLOAT'
    objSim.setAccessMode('read')
    objSim.createImage()
  
    simWidth = imageSim.getWidth()
    simLength = imageSim.getLength()
    fs1 = self._insar.getReferenceFrame().getInstrument().getRangeSamplingRate()  ##check
    delRg1 = CN.SPEED_OF_LIGHT / (2*fs1)                                          ## if it's correct
  
    objAmpcor = Ampcor(name='insarapp_intsim_ampcor')
    objAmpcor.configure()
    objAmpcor.setImageDataType1('real')
    objAmpcor.setImageDataType2('mag')

    ####Adjust first and last values using window sizes
    xMargin = 2*objAmpcor.searchWindowSizeWidth + objAmpcor.windowSizeWidth
    yMargin = 2*objAmpcor.searchWindowSizeHeight + objAmpcor.windowSizeHeight

    if not objAmpcor.acrossGrossOffset:
        coarseAcross = 0
    else:
        coarseAcross = objAmpcor.acrossGrossOffset

    if not objAmpcor.downGrossOffset:
        coarseDown = 0
    else:
        coarseDown = objAmpcor.downGrossOffset

    offAc = max(firstAc, -coarseAcross) + xMargin + 1
    offDn = max(firstDown, -coarseDown) + yMargin + 1
    lastAc = int(min(widthAmp, simWidth-offAc) - xMargin -1)
    lastDn = int(min(intLength, simLength-offDn) - yMargin -1)

    if not objAmpcor.firstSampleAcross:
        objAmpcor.setFirstSampleAcross(offAc)

    if not objAmpcor.lastSampleAcross:
        objAmpcor.setLastSampleAcross(lastAc)

    if not objAmpcor.numberLocationAcross:
        objAmpcor.setNumberLocationAcross(numLocationAcross)

    if not objAmpcor.firstSampleDown:
        objAmpcor.setFirstSampleDown(offDn)

    if not objAmpcor.lastSampleDown:
        objAmpcor.setLastSampleDown(lastDn)

    if not objAmpcor.numberLocationDown:
        objAmpcor.setNumberLocationDown(numLocationDown)

    #set the tag used in the outfile. each message is precided by this tag
    #is the writer is not of "file" type the call has no effect
    self._stdWriter.setFileTag("rgoffset", "log")
    self._stdWriter.setFileTag("rgoffset", "err")
    self._stdWriter.setFileTag("rgoffset", "out")
    objAmpcor.setStdWriter(self._stdWriter)
    prf = self._insar.getReferenceFrame().getInstrument().getPulseRepetitionFrequency()
    
    
    objAmpcor.setFirstPRF(prf)
    objAmpcor.setSecondPRF(prf)
    
    if not objAmpcor.acrossGrossOffset:
        objAmpcor.setAcrossGrossOffset(coarseAcross)

    if not objAmpcor.downGrossOffset:
        objAmpcor.setDownGrossOffset(coarseDown)

    objAmpcor.setFirstRangeSpacing(delRg1)
    objAmpcor.setSecondRangeSpacing(delRg1)
    
    objAmpcor.ampcor(objSim,objAmp)
    
    # Record the inputs and outputs
    from isceobj.Catalog import recordInputsAndOutputs
    recordInputsAndOutputs(self._insar.procDoc, objAmpcor, "runRgoffset_ampcor", \
                  logger, "runRgoffset_ampcor")

    self._insar.setOffsetField(objAmpcor.getOffsetField())
    self._insar.setRefinedOffsetField(objAmpcor.getOffsetField())
   
    objAmp.finalizeImage()
    objSim.finalizeImage()
