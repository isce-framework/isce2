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

    masterTrack = self._insar.loadTrack(master=True)
    slaveTrack = self._insar.loadTrack(master=False)

    mosaicDir = 'insar'
    if not os.path.exists(mosaicDir):
        os.makedirs(mosaicDir)
    os.chdir(mosaicDir)

    if len(masterTrack.frames) > 1:
        #here we use master amplitude image mosaicked from extracted bursts.
        matchingMode=1

        #compute swath offset
        offsetMaster = frameOffset(masterTrack, os.path.join(self._insar.masterBurstPrefix, self._insar.masterMagnitude), self._insar.masterFrameOffset, 
                                   crossCorrelation=self.frameOffsetMatching, matchingMode=matchingMode)
        #only use geometrical offset for slave
        offsetSlave = frameOffset(slaveTrack, os.path.join(self._insar.slaveBurstPrefix, self._insar.slaveMagnitude), self._insar.slaveFrameOffset, 
                                  crossCorrelation=False, matchingMode=matchingMode)

        self._insar.frameRangeOffsetGeometricalMaster = offsetMaster[0]
        self._insar.frameAzimuthOffsetGeometricalMaster = offsetMaster[1]
        self._insar.frameRangeOffsetGeometricalSlave = offsetSlave[0]
        self._insar.frameAzimuthOffsetGeometricalSlave = offsetSlave[1]
        if self.frameOffsetMatching:
            self._insar.frameRangeOffsetMatchingMaster = offsetMaster[2]
            self._insar.frameAzimuthOffsetMatchingMaster = offsetMaster[3]
            #self._insar.frameRangeOffsetMatchingSlave = offsetSlave[2]
            #self._insar.frameAzimuthOffsetMatchingSlave = offsetSlave[3]


    os.chdir('../')

    catalog.printToLog(logger, "runFrameOffset")
    self._insar.procDoc.addAllFromCatalog(catalog)


