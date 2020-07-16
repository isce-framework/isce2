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

from iscesys.ImageUtil.ImageUtil import ImageUtil as IU

logger = logging.getLogger('isce.insar.runRgoffset')

def runRgoffset(self):
    firstAc =  self._insar.getFirstSampleAcross()
    firstDown =  self._insar.getFirstSampleDown()
    numLocationAcross =  self._insar.getNumberLocationAcross()
    numLocationDown =  self._insar.getNumberLocationDown()
   
    imageAmp = self._insar.getResampAmpImage()
    objAmp = isceobj.createIntImage()
    IU.copyAttributes(imageAmp, objAmp)
    objAmp.setAccessMode('read')
    objAmp.createImage()
    widthAmp = objAmp.getWidth()
    intLength = objAmp.getLength()
    lastAc = widthAmp - firstAc
    lastDown = intLength - firstDown

    imageSim = self._insar.getSimAmpImage()
    objSim = isceobj.createImage()
    IU.copyAttributes(imageSim, objSim)
    objSim.setAccessMode('read')
    objSim.createImage()
    
    objOffset = isceobj.createEstimateOffsets(name='insarapp_intsim_estoffset')
    objOffset.configure()
    if objOffset.acrossGrossOffset is not None:
        coarseAcross = objOffset.acrossGrossOffset
    else:
        coarseAcross = 0

    if objOffset.downGrossOffset is not None:
        coarseDown = objOffset.downGrossOffset
    else:
        coarseDown = 0

    if objOffset.searchWindowSize is None:
        objOffset.setSearchWindowSize(self.offsetSearchWindowSize, self.sensorName)
    
    margin = 2*objOffset.searchWindowSize + objOffset.windowSize

    simWidth = objSim.getWidth()
    simLength = objSim.getLength()

    firAc = max(firstAc, -coarseAcross) + margin + 1
    firDn = max(firstDown, -coarseDown) + margin + 1
    lastAc = int(min(widthAmp, simWidth-coarseAcross) - margin - 1)
    lastDn = int(min(intLength, simLength-coarseDown) - margin - 1)


    if not objOffset.firstSampleAcross:
        objOffset.setFirstSampleAcross(firAc)

    if not objOffset.lastSampleAcross:
        objOffset.setLastSampleAcross(lastAc)

    if not objOffset.numberLocationAcross:
        objOffset.setNumberLocationAcross(numLocationAcross)

    if not objOffset.firstSampleDown:
        objOffset.setFirstSampleDown(firDn)

    if not objOffset.lastSampleDown:
        objOffset.setLastSampleDown(lastDn)

    if not objOffset.numberLocationDown:
        objOffset.setNumberLocationDown(numLocationDown)

    #set the tag used in the outfile. each message is precided by this tag
    #is the writer is not of "file" type the call has no effect
    self._stdWriter.setFileTag("rgoffset", "log")
    self._stdWriter.setFileTag("rgoffset", "err")
    self._stdWriter.setFileTag("rgoffset", "out")
    objOffset.setStdWriter(self._stdWriter)
    prf = self._insar.getReferenceFrame().getInstrument().getPulseRepetitionFrequency()
   
    objOffset.setFirstPRF(prf)
    objOffset.setSecondPRF(prf)

    if not objOffset.acrossGrossOffset:
        objOffset.setAcrossGrossOffset(0)

    if not objOffset.downGrossOffset:
        objOffset.setDownGrossOffset(0)

    objOffset.estimateoffsets(image1=objSim, image2=objAmp, band1=0, band2=0)

    # Record the inputs and outputs
    from isceobj.Catalog import recordInputsAndOutputs
    recordInputsAndOutputs(self._insar.procDoc, objOffset, "runRgoffset", \
                  logger, "runRgoffset")

    self._insar.setOffsetField(objOffset.getOffsetField())
    self._insar.setRefinedOffsetField(objOffset.getOffsetField())
   
    objAmp.finalizeImage()
    objSim.finalizeImage()
