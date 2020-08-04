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


from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from isceobj import Constants as CN

logger = logging.getLogger('isce.insar.runOffsetprf')

def runOffsetprf(self):
    from isceobj.Catalog import recordInputs

    logger.info("Calculate offset between slcs")

    referenceFrame = self._insar.getReferenceFrame()
    secondaryFrame = self._insar.getSecondaryFrame()
    referenceOrbit = self._insar.getReferenceOrbit()
    secondaryOrbit = self._insar.getSecondaryOrbit()
    prf1 = referenceFrame.getInstrument().getPulseRepetitionFrequency()
    prf2 = secondaryFrame.getInstrument().getPulseRepetitionFrequency()
    nearRange1 = self.insar.formSLC1.startingRange
    nearRange2 = self.insar.formSLC2.startingRange
    fs1 = referenceFrame.getInstrument().getRangeSamplingRate()

    ###There seems to be no other way of determining image length - Piyush
    patchSize = self._insar.getPatchSize()
    numPatches = self._insar.getNumberPatches()
    valid_az_samples =  self._insar.getNumberValidPulses()
    firstAc =  self._insar.getFirstSampleAcrossPrf()
    firstDown =  self._insar.getFirstSampleDownPrf()
    numLocationAcross =  self._insar.getNumberLocationAcrossPrf()
    numLocationDown =  self._insar.getNumberLocationDownPrf()
    objSlc =  self._insar.getReferenceSlcImage()
#    widthSlc = max(self._insar.getReferenceSlcImage().getWidth(),
#                   self._insar.getSecondarySlcImage().getWidth())
    widthSlc = self._insar.getReferenceSlcImage().getWidth()

    coarseRange = (nearRange1 - nearRange2) / (CN.SPEED_OF_LIGHT / (2 * fs1))
    coarseAcross = int(coarseRange + 0.5)
    if(coarseRange <= 0):
        coarseAcross = int(coarseRange - 0.5)
        pass

    print("gross Rg: ",self.grossRg)

    if self.grossRg is not None:
        coarseAcross = self.grossRg
        pass

    time1, schPosition1, schVelocity1, offset1 = referenceOrbit._unpackOrbit()
    time2, schPosition2, schVelocity2, offset2 = secondaryOrbit._unpackOrbit()
    s1 = schPosition1[0][0]
    s1_2 = schPosition1[1][0]
    s2 = schPosition2[0][0]
    s2_2 = schPosition2[1][0]

    coarseAz = int(
        (s1 - s2)/(s2_2 - s2) + prf2*(1/prf1 - 1/prf2)*
        (patchSize - valid_az_samples)/2
        )
    coarseDown = int(coarseAz + 0.5)
    if(coarseAz <= 0):
        coarseDown = int(coarseAz - 0.5)
        pass

    print("gross Az: ", self.grossAz)

    if self.grossAz is not None:
        coarseDown = self.grossAz
        pass

    coarseAcross = 0 + coarseAcross
    coarseDown = 0 + coarseDown

    mSlcImage = self._insar.getReferenceSlcImage()
    mSlc = isceobj.createSlcImage()
    IU.copyAttributes(mSlcImage, mSlc)
#    scheme = 'BIL'
#    mSlc.setInterleavedScheme(scheme)    #Faster access with bands
    accessMode = 'read'
    mSlc.setAccessMode(accessMode)
    mSlc.createImage()

    sSlcImage = self._insar.getSecondarySlcImage()
    sSlc = isceobj.createSlcImage()
    IU.copyAttributes(sSlcImage, sSlc)
#    scheme = 'BIL'
#    sSlc.setInterleavedScheme(scheme)   #Faster access with bands
    accessMode = 'read'
    sSlc.setAccessMode(accessMode)
    sSlc.createImage()

    objOffset = isceobj.createEstimateOffsets(name='insarapp_slcs_estoffset')
    objOffset.configure()
    if not objOffset.searchWindowSize:
        objOffset.setSearchWindowSize(self.offsetSearchWindowSize, self.sensorName)
    margin = 2*objOffset.searchWindowSize + objOffset.windowSize

    offAc = max(firstAc,-coarseAcross)+margin+1
    offDn = max(firstDown,-coarseDown)+margin+1

    mWidth = mSlc.getWidth()
    sWidth = sSlc.getWidth()
    mLength = mSlc.getLength()
    sLength = sSlc.getLength()

    offDnmax = int(coarseDown + ((prf2/prf1)-1)*mLength)
    lastAc = int(min(mWidth, sWidth-coarseAcross) - margin-1)
    lastDown = int(min(mLength, sLength-offDnmax) - margin-1)


    if not objOffset.firstSampleAcross:
        objOffset.setFirstSampleAcross(offAc)

    if not objOffset.lastSampleAcross:
        objOffset.setLastSampleAcross(lastAc)

    if not objOffset.firstSampleDown:
        objOffset.setFirstSampleDown(offDn)

    if not objOffset.lastSampleDown:
        objOffset.setLastSampleDown(lastDown)

    if not objOffset.numberLocationAcross:
        objOffset.setNumberLocationAcross(numLocationAcross)

    if not objOffset.numberLocationDown:
        objOffset.setNumberLocationDown(numLocationDown)

    if not objOffset.acrossGrossOffset:
        objOffset.setAcrossGrossOffset(coarseAcross)

    if not objOffset.downGrossOffset:
        objOffset.setDownGrossOffset(coarseDown)

    ###Always set these values
    objOffset.setFirstPRF(prf1)
    objOffset.setSecondPRF(prf2)

    # Record the inputs
    recordInputs(self._insar.procDoc,
                 objOffset,
                 "runOffsetprf",
                 logger,
                 "runOffsetprf")

    objOffset.estimateoffsets(image1=mSlc,image2=sSlc,band1=0,band2=0)

    # Record the outputs
    from isceobj.Catalog import recordOutputs
    recordOutputs(self._insar.procDoc,
                  objOffset,
                  "runOffsetprf",
                  logger,
                  "runOffsetprf")

    mSlc.finalizeImage()
    sSlc.finalizeImage()

    # save the input offset field for the record
    self._insar.setOffsetField(objOffset.getOffsetField())
    self._insar.setRefinedOffsetField(objOffset.getOffsetField())
