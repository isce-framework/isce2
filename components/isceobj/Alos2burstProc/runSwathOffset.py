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

    masterTrack = self._insar.loadTrack(master=True)
    slaveTrack = self._insar.loadTrack(master=False)



    for i, frameNumber in enumerate(self._insar.masterFrames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        os.chdir(frameDir)

        mosaicDir = 'mosaic'
        if not os.path.exists(mosaicDir):
            os.makedirs(mosaicDir)
        os.chdir(mosaicDir)

        if self._insar.endingSwath-self._insar.startingSwath+1 == 1:
            os.chdir('../../')
            continue

        #compute swath offset
        offsetMaster = swathOffset(masterTrack.frames[i], os.path.join(self._insar.masterBurstPrefix, self._insar.masterMagnitude), self._insar.masterSwathOffset, 
                                   crossCorrelation=self.swathOffsetMatching, numberOfAzimuthLooks=1)
        #only use geometrical offset for slave
        offsetSlave = swathOffset(slaveTrack.frames[i], os.path.join(self._insar.slaveBurstPrefix, self._insar.slaveMagnitude), self._insar.slaveSwathOffset, 
                                  crossCorrelation=False, numberOfAzimuthLooks=1)

        #initialization
        if i == 0:
            self._insar.swathRangeOffsetGeometricalMaster = []
            self._insar.swathAzimuthOffsetGeometricalMaster = []
            self._insar.swathRangeOffsetGeometricalSlave = []
            self._insar.swathAzimuthOffsetGeometricalSlave = []
            if self.swathOffsetMatching:
                self._insar.swathRangeOffsetMatchingMaster = []
                self._insar.swathAzimuthOffsetMatchingMaster = []
                #self._insar.swathRangeOffsetMatchingSlave = []
                #self._insar.swathAzimuthOffsetMatchingSlave = []

        #append list directly, as the API support 2-d list
        self._insar.swathRangeOffsetGeometricalMaster.append(offsetMaster[0])
        self._insar.swathAzimuthOffsetGeometricalMaster.append(offsetMaster[1])
        self._insar.swathRangeOffsetGeometricalSlave.append(offsetSlave[0])
        self._insar.swathAzimuthOffsetGeometricalSlave.append(offsetSlave[1])
        if self.swathOffsetMatching:
            self._insar.swathRangeOffsetMatchingMaster.append(offsetMaster[2])
            self._insar.swathAzimuthOffsetMatchingMaster.append(offsetMaster[3])
            #self._insar.swathRangeOffsetMatchingSlave.append(offsetSlave[2])
            #self._insar.swathAzimuthOffsetMatchingSlave.append(offsetSlave[3])

        os.chdir('../../')

    catalog.printToLog(logger, "runSwathOffset")
    self._insar.procDoc.addAllFromCatalog(catalog)


