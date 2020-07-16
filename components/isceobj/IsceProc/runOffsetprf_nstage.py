#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2014 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Kosal Khun
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# Comment: Adapted from InsarProc/runOffsetprf_nstage.py
import logging
import isceobj
import mroipac

from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from isceobj import Constants as CN
from mroipac.ampcor.Ampcor import Ampcor

logger = logging.getLogger('isce.isceProc.runOffsetprf')

def runOffsetprf(self, nstages=2, scale=4):
    stdWriter = self._stdWriter
    infos = {}
    for attribute in ['patchSize', 'numberValidPulses', 'numberPatches', 'firstSampleAcrossPrf', 'firstSampleDownPrf', 'numberLocationAcrossPrf', 'numberLocationDownPrf']:
        infos[attribute] = getattr(self._isce, attribute)
    for attribute in ['grossRg', 'grossAz', 'sensorName', 'offsetSearchWindowSize']:
        infos[attribute] = getattr(self, attribute)
    refPol = self._isce.refPol
    for sceneid1, sceneid2 in self._isce.pairsToCoreg:
        pair = (sceneid1, sceneid2)
        frame1 = self._isce.frames[sceneid1][refPol]
        formSlc1 = self._isce.formSLCs[sceneid1][refPol]
        imSlc1 = self._isce.slcImages[sceneid1][refPol]
        frame2 = self._isce.frames[sceneid2][refPol]
        formSlc2 = self._isce.formSLCs[sceneid2][refPol]
        imSlc2 = self._isce.slcImages[sceneid2][refPol]
        catalog = isceobj.Catalog.createCatalog(self._isce.procDoc.name)
        sid = self._isce.formatname(pair)
        offsetField = run(frame1, frame2, formSlc1, formSlc2, imSlc1, imSlc2, nstages, scale, infos, stdWriter, catalog=catalog, sceneid=sid)
        self._isce.procDoc.addAllFromCatalog(catalog)
        self._isce.offsetFields[pair] = offsetField
        self._isce.refinedOffsetFields[pair] = offsetField


def run(frame1, frame2, formSlc1, formSlc2, imSlc1, imSlc2, nstages, scale, infos, stdWriter, catalog=None, sceneid='NO_ID'):
    logger.info("Calculate offset between slcs using %d stages of ampcor: %s " % (nstages, sceneid))

    prf1 = frame1.getInstrument().getPulseRepetitionFrequency()
    prf2 = frame2.getInstrument().getPulseRepetitionFrequency()
    nearRange1 = formSlc1.startingRange
    nearRange2 = formSlc2.startingRange
    fs1 = frame1.getInstrument().getRangeSamplingRate()
    fs2 = frame2.getInstrument().getRangeSamplingRate()

    ###There seems to be no other way of determining image length - Piyush
    patchSize = infos['patchSize']
    numPatches = infos['numberPatches']
    valid_az_samples = infos['numberValidPulses']
    firstAc = infos['firstSampleAcrossPrf']
    firstDown = infos['firstSampleDownPrf']
    numLocationAcross = infos['numberLocationAcrossPrf']
    numLocationDown =  infos['numberLocationDownPrf']

    delRg1 = CN.SPEED_OF_LIGHT / (2*fs1)
    delRg2 = CN.SPEED_OF_LIGHT / (2*fs2)

    grossRg = infos['grossRg']
    if grossRg is not None:
        coarseAcross = grossRg
    else:
        coarseRange = (nearRange1 - nearRange2) / delRg2
        coarseAcross = int(coarseRange + 0.5)
        if(coarseRange <= 0):
            coarseAcross = int(coarseRange - 0.5)
        
    grossAz = infos['grossAz']
    if grossAz is not None:
        coarseDown = grossAz
    else:
        s1 = formSlc1.mocompPosition[1][0]
        s1_2 = formSlc1.mocompPosition[1][1]
        s2 = formSlc2.mocompPosition[1][0]
        s2_2 = formSlc2.mocompPosition[1][1]

        coarseAz = int( (s1 - s2)/(s2_2 - s2) + prf2*(1/prf1 - 1/prf2) * (patchSize - valid_az_samples) / 2 )
        coarseDown = int(coarseAz + 0.5)
        if(coarseAz <= 0):
            coarseDown = int(coarseAz - 0.5)

    coarseAcross = 0 + coarseAcross
    coarseDown = 0 + coarseDown

    mSlc = isceobj.createSlcImage()
    IU.copyAttributes(imSlc1, mSlc)
    accessMode = 'read'
    mSlc.setAccessMode(accessMode)
    mSlc.createImage()
    referenceWidth = mSlc.getWidth()
    referenceLength = mSlc.getLength()
    
    sSlc = isceobj.createSlcImage()
    IU.copyAttributes(imSlc2, sSlc)
    accessMode = 'read'
    sSlc.setAccessMode(accessMode)
    sSlc.createImage()
    secondaryWidth = sSlc.getWidth()
    secondaryLength = sSlc.getLength()

    finalIteration = False
    for iterNum in xrange(nstages-1,-1,-1):
        ####Rewind the images
        try:
            mSlc.rewind()
            sSlc.rewind()
        except:
            print('Issues when rewinding images.') #KK shouldn't it be an error? sys.exit

        ######
        logger.debug('Starting Iteration Stage : %d'%(iterNum))
        logger.debug("Gross Across: %s" % (coarseAcross)) 
        logger.debug("Gross Down: %s" % (coarseDown))

        ####Clear objs
        objAmpcor = None
        objOff = None
        offField = None

        objAmpcor = Ampcor()
        objAmpcor.setImageDataType1('complex')
        objAmpcor.setImageDataType2('complex')
        objAmpcor.setFirstPRF(prf1)
        objAmpcor.setSecondPRF(prf2)
        objAmpcor.setFirstRangeSpacing(delRg1)
        objAmpcor.setSecondRangeSpacing(delRg2)

        #####Scale all the reference and search windows
        scaleFactor = scale**iterNum
        objAmpcor.windowSizeWidth *= scaleFactor
        objAmpcor.windowSizeHeight *= scaleFactor
        objAmpcor.searchWindowSizeWidth *= scaleFactor
        objAmpcor.searchWindowSizeHeight *= scaleFactor
        xMargin = 2*objAmpcor.searchWindowSizeWidth + objAmpcor.windowSizeWidth
        yMargin = 2*objAmpcor.searchWindowSizeHeight + objAmpcor.windowSizeHeight

        #####Set image limits for search
        offAc = max(firstAc,-coarseAcross)+xMargin
        offDn = max(firstDown,-coarseDown)+yMargin

        offAcmax = int(coarseAcross + ((fs2/fs1)-1)*referenceWidth)
        logger.debug("Gross Max Across: %s" % (offAcmax))
        lastAc = int(min(referenceWidth, secondaryWidth-offAcmax) - xMargin)

        offDnmax = int(coarseDown + ((prf2/prf1)-1)*referenceLength)
        logger.debug("Gross Max Down: %s" % (offDnmax))

        lastDn = int(min(referenceLength, secondaryLength-offDnmax)  - yMargin)

        objAmpcor.setFirstSampleAcross(offAc)
        objAmpcor.setLastSampleAcross(lastAc)
        objAmpcor.setFirstSampleDown(offDn)
        objAmpcor.setLastSampleDown(lastDn)
        objAmpcor.setAcrossGrossOffset(coarseAcross)
        objAmpcor.setDownGrossOffset(coarseDown)

        if (offAc > lastAc) or (offDn > lastDn):
            print('Search window scale is too large.')
            print('Skipping Scale: %d'%(iterNum+1))
            continue
        
        logger.debug('Looks = %d'%scaleFactor)
        logger.debug('Correlation window sizes: %d  %d'%(objAmpcor.windowSizeWidth, objAmpcor.windowSizeHeight))
        logger.debug('Search window sizes: %d %d'%(objAmpcor.searchWindowSizeWidth, objAmpcor.searchWindowSizeHeight))
        logger.debug(' Across pos: %d %d out of (%d,%d)'%(objAmpcor.firstSampleAcross, objAmpcor.lastSampleAcross, referenceWidth, secondaryWidth))
        logger.debug(' Down pos: %d %d out of (%d,%d)'%(objAmpcor.firstSampleDown, objAmpcor.lastSampleDown, referenceLength, secondaryLength))
        if (iterNum == 0) or finalIteration:
            if catalog is not None:
                # Record the inputs
                isceobj.Catalog.recordInputs(catalog,
                                             objAmpcor,
                                             "runOffsetprf.%s" % sceneid,
                                             logger,
                                             "runOffsetprf.%s" % sceneid)
            objAmpcor.setNumberLocationAcross(numLocationAcross)
            objAmpcor.setNumberLocationDown(numLocationDown)
        else:
            objAmpcor.setNumberLocationAcross(10)
            objAmpcor.setNumberLocationDown(10)
            objAmpcor.setAcrossLooks(scaleFactor)
            objAmpcor.setDownLooks(scaleFactor)
            objAmpcor.setZoomWindowSize(scale*objAmpcor.zoomWindowSize)
            objAmpcor.setOversamplingFactor(2)

        
        objAmpcor.ampcor(mSlc,sSlc)
        offField = objAmpcor.getOffsetField()

        if (iterNum == 0) or finalIteration:
            if catalog is not None:
                # Record the outputs
                isceobj.Catalog.recordOutputs(catalog,
                                              objAmpcor,
                                              "runOffsetprf.%s" % sceneid,
                                              logger,
                                              "runOffsetprf.%s" % sceneid)
        else:
            objOff = isceobj.createOffoutliers()
            objOff.wireInputPort(name='offsets', object=offField)
            objOff.setSNRThreshold(2.0)
            objOff.setDistance(10)
            objOff.setStdWriter = stdWriter.set_file_tags("nstage_offoutliers"+str(iterNum),
                                                          "log",
                                                          "err",
                                                          "out")
            objOff.offoutliers()
            coarseAcross = int(objOff.averageOffsetAcross)
            coarseDown = int(objOff.averageOffsetDown)

    mSlc.finalizeImage()
    sSlc.finalizeImage()
    objOff = None
    objAmpcor = None
    
    return offField
