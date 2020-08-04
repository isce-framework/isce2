#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging

import isceobj
from isceobj.Alos2Proc.runSwathOffset import swathOffset

logger = logging.getLogger('isce.alos2burstinsar.runSwathOffset')

def runSwathOffset(self):
    '''estimate swath offsets.
    '''
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    referenceTrack = self._insar.loadTrack(reference=True)
    secondaryTrack = self._insar.loadTrack(reference=False)



    for i, frameNumber in enumerate(self._insar.referenceFrames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        os.chdir(frameDir)

        mosaicDir = 'mosaic'
        os.makedirs(mosaicDir, exist_ok=True)
        os.chdir(mosaicDir)

        if self._insar.endingSwath-self._insar.startingSwath+1 == 1:
            os.chdir('../../')
            continue

        #compute swath offset
        offsetReference = swathOffset(referenceTrack.frames[i], os.path.join(self._insar.referenceBurstPrefix, self._insar.referenceMagnitude), self._insar.referenceSwathOffset, 
                                   crossCorrelation=self.swathOffsetMatching, numberOfAzimuthLooks=1)
        #only use geometrical offset for secondary
        offsetSecondary = swathOffset(secondaryTrack.frames[i], os.path.join(self._insar.secondaryBurstPrefix, self._insar.secondaryMagnitude), self._insar.secondarySwathOffset, 
                                  crossCorrelation=False, numberOfAzimuthLooks=1)

        #initialization
        if i == 0:
            self._insar.swathRangeOffsetGeometricalReference = []
            self._insar.swathAzimuthOffsetGeometricalReference = []
            self._insar.swathRangeOffsetGeometricalSecondary = []
            self._insar.swathAzimuthOffsetGeometricalSecondary = []
            if self.swathOffsetMatching:
                self._insar.swathRangeOffsetMatchingReference = []
                self._insar.swathAzimuthOffsetMatchingReference = []
                #self._insar.swathRangeOffsetMatchingSecondary = []
                #self._insar.swathAzimuthOffsetMatchingSecondary = []

        #append list directly, as the API support 2-d list
        self._insar.swathRangeOffsetGeometricalReference.append(offsetReference[0])
        self._insar.swathAzimuthOffsetGeometricalReference.append(offsetReference[1])
        self._insar.swathRangeOffsetGeometricalSecondary.append(offsetSecondary[0])
        self._insar.swathAzimuthOffsetGeometricalSecondary.append(offsetSecondary[1])
        if self.swathOffsetMatching:
            self._insar.swathRangeOffsetMatchingReference.append(offsetReference[2])
            self._insar.swathAzimuthOffsetMatchingReference.append(offsetReference[3])
            #self._insar.swathRangeOffsetMatchingSecondary.append(offsetSecondary[2])
            #self._insar.swathAzimuthOffsetMatchingSecondary.append(offsetSecondary[3])

        os.chdir('../../')

    catalog.printToLog(logger, "runSwathOffset")
    self._insar.procDoc.addAllFromCatalog(catalog)


