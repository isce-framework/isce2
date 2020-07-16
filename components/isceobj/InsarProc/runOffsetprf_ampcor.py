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
# Author: Gaiangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import logging
import isceobj
import mroipac

from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from isceobj import Constants as CN
from mroipac.ampcor.Ampcor import Ampcor

logger = logging.getLogger('isce.insar.runOffsetprf')

def runOffsetprf(self):
    from isceobj.Catalog import recordInputs

    logger.info("Calculate offset between slcs using ampcor")
    referenceFrame = self._insar.getReferenceFrame()
    secondaryFrame = self._insar.getSecondaryFrame()
    referenceOrbit = self._insar.getReferenceOrbit()
    secondaryOrbit = self._insar.getSecondaryOrbit()
    prf1 = referenceFrame.getInstrument().getPulseRepetitionFrequency()
    prf2 = secondaryFrame.getInstrument().getPulseRepetitionFrequency()
    nearRange1 = self.insar.formSLC1.startingRange
    nearRange2 = self.insar.formSLC2.startingRange
    fs1 = referenceFrame.getInstrument().getRangeSamplingRate()
    fs2 = secondaryFrame.getInstrument().getRangeSamplingRate()

    ###There seems to be no other way of determining image length - Piyush
    patchSize = self._insar.getPatchSize() 
    numPatches = self._insar.getNumberPatches()
    valid_az_samples =  self._insar.getNumberValidPulses()
    firstAc =  self._insar.getFirstSampleAcrossPrf()
    firstDown =  self._insar.getFirstSampleDownPrf()
    numLocationAcross =  self._insar.getNumberLocationAcrossPrf()
    numLocationDown =  self._insar.getNumberLocationDownPrf()

    delRg1 = CN.SPEED_OF_LIGHT / (2*fs1)
    delRg2 = CN.SPEED_OF_LIGHT / (2*fs2)

    coarseRange = (nearRange1 - nearRange2) / delRg2
    coarseAcross = int(coarseRange + 0.5)
    if(coarseRange <= 0):
        coarseAcross = int(coarseRange - 0.5)
        pass


    if self.grossRg is not None:
        coarseAcross = self.grossRg
        pass

    s1 = self.insar.formSLC1.mocompPosition[1][0]
    s1_2 = self.insar.formSLC1.mocompPosition[1][1]
    s2 = self.insar.formSLC2.mocompPosition[1][0]
    s2_2 = self.insar.formSLC2.mocompPosition[1][1]

    coarseAz = int(
        (s1 - s2)/(s2_2 - s2) + prf2*(1/prf1 - 1/prf2)*(patchSize - valid_az_samples)/2)

    coarseDown = int(coarseAz + 0.5)
    if(coarseAz <= 0):
        coarseDown = int(coarseAz - 0.5)
        pass
    
    if self.grossAz is not None:
        coarseDown = self.grossAz
        pass

    coarseAcross = 0 + coarseAcross
    coarseDown = 0 + coarseDown

    mSlcImage = self._insar.getReferenceSlcImage()
    mSlc = isceobj.createSlcImage()
    IU.copyAttributes(mSlcImage, mSlc)
    accessMode = 'read'
    mSlc.setAccessMode(accessMode)
    mSlc.createImage()
    referenceWidth = mSlc.getWidth()
    referenceLength = mSlc.getLength()
    
    sSlcImage = self._insar.getSecondarySlcImage()
    sSlc = isceobj.createSlcImage()
    IU.copyAttributes(sSlcImage, sSlc)
    accessMode = 'read'
    sSlc.setAccessMode(accessMode)
    sSlc.createImage()
    secondaryWidth = sSlc.getWidth()
    secondaryLength = sSlc.getLength()

    objAmpcor = Ampcor(name='insarapp_slcs_ampcor')
    objAmpcor.configure()
    objAmpcor.setImageDataType1('complex')
    objAmpcor.setImageDataType2('complex')

    if objAmpcor.acrossGrossOffset:
        coarseAcross = objAmpcor.acrossGrossOffset

    if objAmpcor.downGrossOffset:
        coarseDown = objAmpcor.downGrossOffset

    logger.debug("Gross Across: %s" % (coarseAcross))
    logger.debug("Gross Down: %s" % (coarseDown))

    xMargin = 2*objAmpcor.searchWindowSizeWidth + objAmpcor.windowSizeWidth
    yMargin = 2*objAmpcor.searchWindowSizeHeight + objAmpcor.windowSizeHeight

    #####Compute image positions
    offAc = max(firstAc,-coarseAcross)+xMargin
    offDn = max(firstDown,-coarseDown)+yMargin

    offAcmax = int(coarseAcross + ((fs2/fs1)-1)*referenceWidth)
    logger.debug("Gross Max Across: %s" % (offAcmax))
    lastAc = int(min(referenceWidth, secondaryWidth- offAcmax) - xMargin)

    offDnmax = int(coarseDown + ((prf2/prf1)-1)*referenceLength)
    logger.debug("Gross Max Down: %s" % (offDnmax))
    lastDown = int( min(referenceLength, secondaryLength-offDnmax) - yMargin)


    if not objAmpcor.firstSampleAcross:
        objAmpcor.setFirstSampleAcross(offAc)

    if not objAmpcor.lastSampleAcross:
        objAmpcor.setLastSampleAcross(lastAc)

    if not objAmpcor.numberLocationAcross:
        objAmpcor.setNumberLocationAcross(numLocationAcross)

    if not objAmpcor.firstSampleDown:
        objAmpcor.setFirstSampleDown(offDn)

    if not objAmpcor.lastSampleDown:
        objAmpcor.setLastSampleDown(lastDown)

    if not objAmpcor.numberLocationDown:
        objAmpcor.setNumberLocationDown(numLocationDown)

    #####Override gross offsets if not provided
    if not objAmpcor.acrossGrossOffset:
        objAmpcor.setAcrossGrossOffset(coarseAcross)

    if not objAmpcor.downGrossOffset:
        objAmpcor.setDownGrossOffset(coarseDown)


    #####User inputs are overriden here
    objAmpcor.setFirstPRF(prf1)
    objAmpcor.setSecondPRF(prf2)
    objAmpcor.setFirstRangeSpacing(delRg1)
    objAmpcor.setSecondRangeSpacing(delRg2)
    
    
    # Record the inputs
    recordInputs(self._insar.procDoc,
                 objAmpcor,
                 "runOffsetprf",
                 logger,
                 "runOffsetprf")
    
    objAmpcor.ampcor(mSlc,sSlc)

    # Record the outputs
    from isceobj.Catalog import recordOutputs
    recordOutputs(self._insar.procDoc,
                  objAmpcor,
                  "runOffsetprf",
                  logger,
                  "runOffsetprf")

    mSlc.finalizeImage()
    sSlc.finalizeImage()
    
   
    # save the input offset field for the record
    self._insar.setOffsetField(objAmpcor.getOffsetField())
    self._insar.setRefinedOffsetField(objAmpcor.getOffsetField())
