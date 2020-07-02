#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging

import isceobj
from isceobj.Alos2Proc.runFrameOffset import frameOffset

logger = logging.getLogger('isce.alos2burstinsar.runFrameOffset')

def runFrameOffset(self):
    '''estimate frame offsets.
    '''
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    referenceTrack = self._insar.loadTrack(reference=True)
    secondaryTrack = self._insar.loadTrack(reference=False)

    mosaicDir = 'insar'
    os.makedirs(mosaicDir, exist_ok=True)
    os.chdir(mosaicDir)

    if len(referenceTrack.frames) > 1:
        #here we use reference amplitude image mosaicked from extracted bursts.
        matchingMode=1

        #compute swath offset
        offsetReference = frameOffset(referenceTrack, os.path.join(self._insar.referenceBurstPrefix, self._insar.referenceMagnitude), self._insar.referenceFrameOffset, 
                                   crossCorrelation=self.frameOffsetMatching, matchingMode=matchingMode)
        #only use geometrical offset for secondary
        offsetSecondary = frameOffset(secondaryTrack, os.path.join(self._insar.secondaryBurstPrefix, self._insar.secondaryMagnitude), self._insar.secondaryFrameOffset, 
                                  crossCorrelation=False, matchingMode=matchingMode)

        self._insar.frameRangeOffsetGeometricalReference = offsetReference[0]
        self._insar.frameAzimuthOffsetGeometricalReference = offsetReference[1]
        self._insar.frameRangeOffsetGeometricalSecondary = offsetSecondary[0]
        self._insar.frameAzimuthOffsetGeometricalSecondary = offsetSecondary[1]
        if self.frameOffsetMatching:
            self._insar.frameRangeOffsetMatchingReference = offsetReference[2]
            self._insar.frameAzimuthOffsetMatchingReference = offsetReference[3]
            #self._insar.frameRangeOffsetMatchingSecondary = offsetSecondary[2]
            #self._insar.frameAzimuthOffsetMatchingSecondary = offsetSecondary[3]


    os.chdir('../')

    catalog.printToLog(logger, "runFrameOffset")
    self._insar.procDoc.addAllFromCatalog(catalog)


