#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Authors: Kosal Khun, Marco Lavalle
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# Comment: Adapted from InsarProc/runOffsetprf.py
import logging
import isceobj
import sys

from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from isceobj import Constants as CN

logger = logging.getLogger('isce.isceProc.runOffsetprf')

def runOffsetprf(self):
    infos = {}
    for attribute in ['patchSize', 'numberValidPulses', 'numberPatches', 'firstSampleAcrossPrf', 'firstSampleDownPrf', 'numberLocationAcrossPrf', 'numberLocationDownPrf']:
        infos[attribute] = getattr(self._isce, attribute)
    for attribute in ['grossRg', 'grossAz', 'sensorName', 'offsetSearchWindowSize']:
        infos[attribute] = getattr(self, attribute)
    refPol = self._isce.refPol
    for sceneid1, sceneid2 in self._isce.pairsToCoreg:
        pair = (sceneid1, sceneid2)
        frame1 = self._isce.frames[sceneid1][refPol]
        orbit1 = self._isce.orbits[sceneid1][refPol]
        formSlc1 = self._isce.formSLCs[sceneid1][refPol]
        imSlc1 = self._isce.slcImages[sceneid1][refPol]
        frame2 = self._isce.frames[sceneid2][refPol]
        orbit2 = self._isce.orbits[sceneid2][refPol]
        formSlc2 = self._isce.formSLCs[sceneid2][refPol]
        imSlc2 = self._isce.slcImages[sceneid2][refPol]
        catalog = isceobj.Catalog.createCatalog(self._isce.procDoc.name)
        sid = self._isce.formatname(pair)
        offsetField = run(frame1, frame2, orbit1, orbit2, formSlc1, formSlc2, imSlc1, imSlc2, infos, catalog=catalog, sceneid=sid)
        self._isce.procDoc.addAllFromCatalog(catalog)
        self._isce.offsetFields[pair] = offsetField
        self._isce.refinedOffsetFields[pair] = offsetField


def run(frame1, frame2, orbit1, orbit2, formSlc1, formSlc2, imSlc1, imSlc2, infos, catalog=None, sceneid='NO_ID'):
    logger.info("Calculate offset between slcs: %s" % sceneid)

    prf1 = frame1.getInstrument().getPulseRepetitionFrequency()
    prf2 = frame2.getInstrument().getPulseRepetitionFrequency()
    nearRange1 = formSlc1.startingRange
    nearRange2 = formSlc2.startingRange
    fs1 = frame1.getInstrument().getRangeSamplingRate()

    ###There seems to be no other way of determining image length - Piyush
    patchSize = infos['patchSize']
    numPatches = infos['numberPatches']
    valid_az_samples = infos['numberValidPulses']
    firstAc = infos['firstSampleAcrossPrf']
    firstDown = infos['firstSampleDownPrf']
    numLocationAcross = infos['numberLocationAcrossPrf']
    numLocationDown =  infos['numberLocationDownPrf']

    widthSlc = imSlc1.getWidth()

    grossRg = infos['grossRg']
    if grossRg is not None:
        coarseAcross = grossRg
    else:
        coarseRange = (nearRange1 - nearRange2) / (CN.SPEED_OF_LIGHT / (2 * fs1))
        coarseAcross = int(coarseRange + 0.5)
        if(coarseRange <= 0):
            coarseAcross = int(coarseRange - 0.5)

    grossAz = infos['grossAz']
    if grossAz is not None:
        coarseDown = grossAz
    else:
        time1, schPosition1, schVelocity1, offset1 = orbit1._unpackOrbit()
        time2, schPosition2, schVelocity2, offset2 = orbit2._unpackOrbit()
        s1 = schPosition1[0][0]
        s1_2 = schPosition1[1][0]
        s2 = schPosition2[0][0]
        s2_2 = schPosition2[1][0]

        coarseAz = int( (s1 - s2)/(s2_2 - s2) + prf1*(1/prf1 - 1/prf2) * (patchSize - valid_az_samples) / 2 )
        coarseDown = int(coarseAz + 0.5)
        if(coarseAz <= 0):
            coarseDown = int(coarseAz - 0.5)

    coarseAcross = 0 + coarseAcross
    coarseDown = 0 + coarseDown

    logger.debug("Gross Across: %s" % (coarseAcross))
    logger.debug("Gross Down: %s" % (coarseDown))

    offAc = max(firstAc,coarseAcross)
    offDn = max(firstDown,coarseDown)
    lastAc = widthSlc - offAc
    lastDown = (numPatches * valid_az_samples) - offDn

    mSlc = isceobj.createSlcImage()
    IU.copyAttributes(imSlc1, mSlc)
    accessMode = 'read'
    mSlc.setAccessMode(accessMode)
    mSlc.createImage()

    sSlc = isceobj.createSlcImage()
    IU.copyAttributes(imSlc2, sSlc)
    accessMode = 'read'
    sSlc.setAccessMode(accessMode)
    sSlc.createImage()

    objOffset = isceobj.createEstimateOffsets()


    objOffset.configure()
    if not objOffset.searchWindowSize:
        #objOffset.setSearchWindowSize(self.offsetSearchWindowSize, self.sensorName)
        objOffset.setSearchWindowSize(infos['offsetSearchWindowSize'], infos['sensorName'])
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

    objOffset.setFirstSampleAcross(offAc)
    objOffset.setLastSampleAcross(lastAc)
    objOffset.setNumberLocationAcross(numLocationAcross)
    objOffset.setFirstSampleDown(offDn)
    objOffset.setLastSampleDown(lastDown)
    objOffset.setNumberLocationDown(numLocationDown)
    objOffset.setAcrossGrossOffset(coarseAcross)
    objOffset.setDownGrossOffset(coarseDown)
    objOffset.setFirstPRF(prf1)
    objOffset.setSecondPRF(prf2)

    if catalog is not None:
        # Record the inputs
        isceobj.Catalog.recordInputs(catalog,
                                     objOffset,
                                     "runOffsetprf.%s" % sceneid,
                                     logger,
                                     "runOffsetprf.%s" % sceneid)

    objOffset.estimateoffsets(image1=mSlc,image2=sSlc,band1=0,band2=0)

    if catalog is not None:
    # Record the outputs
        isceobj.Catalog.recordOutputs(catalog,
                                      objOffset,
                                      "runOffsetprf.%s" % sceneid,
                                      logger,
                                      "runOffsetprf.%s" % sceneid)

    mSlc.finalizeImage()
    sSlc.finalizeImage()

    return objOffset.getOffsetField()
