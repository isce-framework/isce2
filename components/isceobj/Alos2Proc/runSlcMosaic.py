#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging
import numpy as np

import isceobj
from isceobj.Alos2Proc.runFrameOffset import frameOffset
from isceobj.Alos2Proc.runFrameMosaic import frameMosaic
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml

logger = logging.getLogger('isce.alos2insar.runSlcMosaic')

def runSlcMosaic(self):
    '''mosaic SLCs
    '''
    if not self.doDenseOffset:
        print('\ndense offset not requested, skip this and the remaining steps...')
        return
    if not ((self._insar.modeCombination == 0) or (self._insar.modeCombination == 1)):
        print('dense offset only support spotligh-spotlight and stripmap-stripmap pairs')
        return

    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()
    masterTrack = self._insar.loadTrack(master=True)
    slaveTrack = self._insar.loadTrack(master=False)

    denseOffsetDir = 'dense_offset'
    os.makedirs(denseOffsetDir, exist_ok=True)
    os.chdir(denseOffsetDir)


    ##################################################
    # estimate master and slave frame offsets
    ##################################################
    if len(masterTrack.frames) > 1:
        matchingMode=1

        #if master offsets from matching are not already computed
        if self.frameOffsetMatching == False:
            offsetMaster = frameOffset(masterTrack, self._insar.masterSlc, self._insar.masterFrameOffset, 
                                       crossCorrelation=True, matchingMode=matchingMode)
        offsetSlave = frameOffset(slaveTrack, self._insar.slaveSlc, self._insar.slaveFrameOffset, 
                                  crossCorrelation=True, matchingMode=matchingMode)
        if self.frameOffsetMatching == False:
            self._insar.frameRangeOffsetMatchingMaster = offsetMaster[2]
            self._insar.frameAzimuthOffsetMatchingMaster = offsetMaster[3]
        self._insar.frameRangeOffsetMatchingSlave = offsetSlave[2]
        self._insar.frameAzimuthOffsetMatchingSlave = offsetSlave[3]


    ##################################################
    # mosaic slc
    ##################################################
    numberOfFrames = len(masterTrack.frames)
    if numberOfFrames == 1:
        import shutil
        #frameDir = os.path.join('f1_{}/mosaic'.format(self._insar.masterFrames[0]))
        frameDir = os.path.join('f1_{}/s{}'.format(self._insar.masterFrames[0], self._insar.startingSwath))
        if not os.path.isfile(self._insar.masterSlc):
            if os.path.isfile(os.path.join('../', frameDir, self._insar.masterSlc)):
                os.symlink(os.path.join('../', frameDir, self._insar.masterSlc), self._insar.masterSlc)
        #shutil.copy2() can overwrite
        shutil.copy2(os.path.join('../', frameDir, self._insar.masterSlc+'.vrt'), self._insar.masterSlc+'.vrt')
        shutil.copy2(os.path.join('../', frameDir, self._insar.masterSlc+'.xml'), self._insar.masterSlc+'.xml')
        if not os.path.isfile(self._insar.slaveSlc):
            if os.path.isfile(os.path.join('../', frameDir, self._insar.slaveSlc)):
                os.symlink(os.path.join('../', frameDir, self._insar.slaveSlc), self._insar.slaveSlc)
        shutil.copy2(os.path.join('../', frameDir, self._insar.slaveSlc+'.vrt'), self._insar.slaveSlc+'.vrt')
        shutil.copy2(os.path.join('../', frameDir, self._insar.slaveSlc+'.xml'), self._insar.slaveSlc+'.xml')

        #update track parameters
        #########################################################
        #mosaic size
        masterTrack.numberOfSamples = masterTrack.frames[0].swaths[0].numberOfSamples
        masterTrack.numberOfLines = masterTrack.frames[0].swaths[0].numberOfLines
        #NOTE THAT WE ARE STILL USING SINGLE LOOK PARAMETERS HERE
        #range parameters
        masterTrack.startingRange = masterTrack.frames[0].swaths[0].startingRange
        masterTrack.rangeSamplingRate = masterTrack.frames[0].swaths[0].rangeSamplingRate
        masterTrack.rangePixelSize = masterTrack.frames[0].swaths[0].rangePixelSize
        #azimuth parameters
        masterTrack.sensingStart = masterTrack.frames[0].swaths[0].sensingStart
        masterTrack.prf = masterTrack.frames[0].swaths[0].prf
        masterTrack.azimuthPixelSize = masterTrack.frames[0].swaths[0].azimuthPixelSize
        masterTrack.azimuthLineInterval = masterTrack.frames[0].swaths[0].azimuthLineInterval

        masterTrack.dopplerVsPixel = masterTrack.frames[0].swaths[0].dopplerVsPixel

        #update track parameters, slave
        #########################################################
        #mosaic size
        slaveTrack.numberOfSamples = slaveTrack.frames[0].swaths[0].numberOfSamples
        slaveTrack.numberOfLines = slaveTrack.frames[0].swaths[0].numberOfLines
        #NOTE THAT WE ARE STILL USING SINGLE LOOK PARAMETERS HERE
        #range parameters
        slaveTrack.startingRange = slaveTrack.frames[0].swaths[0].startingRange
        slaveTrack.rangeSamplingRate = slaveTrack.frames[0].swaths[0].rangeSamplingRate
        slaveTrack.rangePixelSize = slaveTrack.frames[0].swaths[0].rangePixelSize
        #azimuth parameters
        slaveTrack.sensingStart = slaveTrack.frames[0].swaths[0].sensingStart
        slaveTrack.prf = slaveTrack.frames[0].swaths[0].prf
        slaveTrack.azimuthPixelSize = slaveTrack.frames[0].swaths[0].azimuthPixelSize
        slaveTrack.azimuthLineInterval = slaveTrack.frames[0].swaths[0].azimuthLineInterval

        slaveTrack.dopplerVsPixel = slaveTrack.frames[0].swaths[0].dopplerVsPixel

    else:
        #mosaic master slc
        #########################################################
        #choose offsets
        rangeOffsets = self._insar.frameRangeOffsetMatchingMaster
        azimuthOffsets = self._insar.frameAzimuthOffsetMatchingMaster

        #list of input files
        slcs = []
        for i, frameNumber in enumerate(self._insar.masterFrames):
            frameDir = 'f{}_{}'.format(i+1, frameNumber)
            swathDir = 's{}'.format(self._insar.startingSwath)
            slcs.append(os.path.join('../', frameDir, swathDir, self._insar.masterSlc))

        #note that track parameters are updated after mosaicking
        #parameters update is checked, it is OK.
        frameMosaic(masterTrack, slcs, self._insar.masterSlc, 
            rangeOffsets, azimuthOffsets, 1, 1, 
            updateTrack=True, phaseCompensation=True, resamplingMethod=2)
        create_xml(self._insar.masterSlc, masterTrack.numberOfSamples, masterTrack.numberOfLines, 'slc')
        masterTrack.dopplerVsPixel = computeTrackDoppler(masterTrack)

        #mosaic slave slc
        #########################################################
        #choose offsets
        rangeOffsets = self._insar.frameRangeOffsetMatchingSlave
        azimuthOffsets = self._insar.frameAzimuthOffsetMatchingSlave

        #list of input files
        slcs = []
        for i, frameNumber in enumerate(self._insar.masterFrames):
            frameDir = 'f{}_{}'.format(i+1, frameNumber)
            swathDir = 's{}'.format(self._insar.startingSwath)
            slcs.append(os.path.join('../', frameDir, swathDir, self._insar.slaveSlc))

        #note that track parameters are updated after mosaicking
        #parameters update is checked, it is OK.
        frameMosaic(slaveTrack, slcs, self._insar.slaveSlc, 
            rangeOffsets, azimuthOffsets, 1, 1, 
            updateTrack=True, phaseCompensation=True, resamplingMethod=2)
        create_xml(self._insar.slaveSlc, slaveTrack.numberOfSamples, slaveTrack.numberOfLines, 'slc')
        slaveTrack.dopplerVsPixel = computeTrackDoppler(slaveTrack)


    #save parameter file inside denseoffset directory
    self._insar.saveProduct(masterTrack, self._insar.masterTrackParameter)
    self._insar.saveProduct(slaveTrack, self._insar.slaveTrackParameter)


    os.chdir('../')
    catalog.printToLog(logger, "runSlcMosaic")
    self._insar.procDoc.addAllFromCatalog(catalog)


def computeTrackDoppler(track):
    '''
    compute doppler for a track
    '''
    numberOfFrames = len(track.frames)
    dop = np.zeros(track.numberOfSamples)
    for i in range(numberOfFrames):
        index = track.startingRange + np.arange(track.numberOfSamples) * track.rangePixelSize
        index = (index - track.frames[i].swaths[0].startingRange) / track.frames[i].swaths[0].rangePixelSize
        dop = dop + np.polyval(track.frames[i].swaths[0].dopplerVsPixel[::-1], index)
    
    index1 = np.arange(track.numberOfSamples)
    dop1 = dop/numberOfFrames
    p = np.polyfit(index1, dop1, 3)

    return [p[3], p[2], p[1], p[0]]
