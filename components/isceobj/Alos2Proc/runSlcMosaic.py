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
    referenceTrack = self._insar.loadTrack(reference=True)
    secondaryTrack = self._insar.loadTrack(reference=False)

    denseOffsetDir = 'dense_offset'
    os.makedirs(denseOffsetDir, exist_ok=True)
    os.chdir(denseOffsetDir)


    ##################################################
    # estimate reference and secondary frame offsets
    ##################################################
    if len(referenceTrack.frames) > 1:
        matchingMode=1

        #determine whether reference offset from matching is already done in previous InSAR processing.
        if hasattr(self, 'doInSAR'):
            if not self.doInSAR:
                referenceEstimated = False
            else:
                if self.frameOffsetMatching == False:
                    referenceEstimated = False
                else:
                    referenceEstimated = True
        else:
            if self.frameOffsetMatching == False:
                referenceEstimated = False
            else:
                referenceEstimated = True

        #if reference offsets from matching are not already computed
        #if self.frameOffsetMatching == False:
        if referenceEstimated == False:
            offsetReference = frameOffset(referenceTrack, self._insar.referenceSlc, self._insar.referenceFrameOffset, 
                                       crossCorrelation=True, matchingMode=matchingMode)
        offsetSecondary = frameOffset(secondaryTrack, self._insar.secondarySlc, self._insar.secondaryFrameOffset, 
                                  crossCorrelation=True, matchingMode=matchingMode)
        #if self.frameOffsetMatching == False:
        if referenceEstimated == False:
            self._insar.frameRangeOffsetMatchingReference = offsetReference[2]
            self._insar.frameAzimuthOffsetMatchingReference = offsetReference[3]
        self._insar.frameRangeOffsetMatchingSecondary = offsetSecondary[2]
        self._insar.frameAzimuthOffsetMatchingSecondary = offsetSecondary[3]


    ##################################################
    # mosaic slc
    ##################################################
    numberOfFrames = len(referenceTrack.frames)
    if numberOfFrames == 1:
        import shutil
        #frameDir = os.path.join('f1_{}/mosaic'.format(self._insar.referenceFrames[0]))
        frameDir = os.path.join('f1_{}/s{}'.format(self._insar.referenceFrames[0], self._insar.startingSwath))
        if not os.path.isfile(self._insar.referenceSlc):
            if os.path.isfile(os.path.join('../', frameDir, self._insar.referenceSlc)):
                os.symlink(os.path.join('../', frameDir, self._insar.referenceSlc), self._insar.referenceSlc)
        #shutil.copy2() can overwrite
        shutil.copy2(os.path.join('../', frameDir, self._insar.referenceSlc+'.vrt'), self._insar.referenceSlc+'.vrt')
        shutil.copy2(os.path.join('../', frameDir, self._insar.referenceSlc+'.xml'), self._insar.referenceSlc+'.xml')
        if not os.path.isfile(self._insar.secondarySlc):
            if os.path.isfile(os.path.join('../', frameDir, self._insar.secondarySlc)):
                os.symlink(os.path.join('../', frameDir, self._insar.secondarySlc), self._insar.secondarySlc)
        shutil.copy2(os.path.join('../', frameDir, self._insar.secondarySlc+'.vrt'), self._insar.secondarySlc+'.vrt')
        shutil.copy2(os.path.join('../', frameDir, self._insar.secondarySlc+'.xml'), self._insar.secondarySlc+'.xml')

        #update track parameters
        #########################################################
        #mosaic size
        referenceTrack.numberOfSamples = referenceTrack.frames[0].swaths[0].numberOfSamples
        referenceTrack.numberOfLines = referenceTrack.frames[0].swaths[0].numberOfLines
        #NOTE THAT WE ARE STILL USING SINGLE LOOK PARAMETERS HERE
        #range parameters
        referenceTrack.startingRange = referenceTrack.frames[0].swaths[0].startingRange
        referenceTrack.rangeSamplingRate = referenceTrack.frames[0].swaths[0].rangeSamplingRate
        referenceTrack.rangePixelSize = referenceTrack.frames[0].swaths[0].rangePixelSize
        #azimuth parameters
        referenceTrack.sensingStart = referenceTrack.frames[0].swaths[0].sensingStart
        referenceTrack.prf = referenceTrack.frames[0].swaths[0].prf
        referenceTrack.azimuthPixelSize = referenceTrack.frames[0].swaths[0].azimuthPixelSize
        referenceTrack.azimuthLineInterval = referenceTrack.frames[0].swaths[0].azimuthLineInterval

        referenceTrack.dopplerVsPixel = referenceTrack.frames[0].swaths[0].dopplerVsPixel

        #update track parameters, secondary
        #########################################################
        #mosaic size
        secondaryTrack.numberOfSamples = secondaryTrack.frames[0].swaths[0].numberOfSamples
        secondaryTrack.numberOfLines = secondaryTrack.frames[0].swaths[0].numberOfLines
        #NOTE THAT WE ARE STILL USING SINGLE LOOK PARAMETERS HERE
        #range parameters
        secondaryTrack.startingRange = secondaryTrack.frames[0].swaths[0].startingRange
        secondaryTrack.rangeSamplingRate = secondaryTrack.frames[0].swaths[0].rangeSamplingRate
        secondaryTrack.rangePixelSize = secondaryTrack.frames[0].swaths[0].rangePixelSize
        #azimuth parameters
        secondaryTrack.sensingStart = secondaryTrack.frames[0].swaths[0].sensingStart
        secondaryTrack.prf = secondaryTrack.frames[0].swaths[0].prf
        secondaryTrack.azimuthPixelSize = secondaryTrack.frames[0].swaths[0].azimuthPixelSize
        secondaryTrack.azimuthLineInterval = secondaryTrack.frames[0].swaths[0].azimuthLineInterval

        secondaryTrack.dopplerVsPixel = secondaryTrack.frames[0].swaths[0].dopplerVsPixel

    else:
        #in case InSAR, and therefore runSwathMosaic, was not done previously
        for i, frameNumber in enumerate(self._insar.referenceFrames):
            #update frame parameters
            #########################################################
            frame = referenceTrack.frames[i]
            #mosaic size
            frame.numberOfSamples = frame.swaths[0].numberOfSamples
            frame.numberOfLines = frame.swaths[0].numberOfLines
            #NOTE THAT WE ARE STILL USING SINGLE LOOK PARAMETERS HERE
            #range parameters
            frame.startingRange = frame.swaths[0].startingRange
            frame.rangeSamplingRate = frame.swaths[0].rangeSamplingRate
            frame.rangePixelSize = frame.swaths[0].rangePixelSize
            #azimuth parameters
            frame.sensingStart = frame.swaths[0].sensingStart
            frame.prf = frame.swaths[0].prf
            frame.azimuthPixelSize = frame.swaths[0].azimuthPixelSize
            frame.azimuthLineInterval = frame.swaths[0].azimuthLineInterval

            #update frame parameters, secondary
            #########################################################
            frame = secondaryTrack.frames[i]
            #mosaic size
            frame.numberOfSamples = frame.swaths[0].numberOfSamples
            frame.numberOfLines = frame.swaths[0].numberOfLines
            #NOTE THAT WE ARE STILL USING SINGLE LOOK PARAMETERS HERE
            #range parameters
            frame.startingRange = frame.swaths[0].startingRange
            frame.rangeSamplingRate = frame.swaths[0].rangeSamplingRate
            frame.rangePixelSize = frame.swaths[0].rangePixelSize
            #azimuth parameters
            frame.sensingStart = frame.swaths[0].sensingStart
            frame.prf = frame.swaths[0].prf
            frame.azimuthPixelSize = frame.swaths[0].azimuthPixelSize
            frame.azimuthLineInterval = frame.swaths[0].azimuthLineInterval


        #mosaic reference slc
        #########################################################
        #choose offsets
        rangeOffsets = self._insar.frameRangeOffsetMatchingReference
        azimuthOffsets = self._insar.frameAzimuthOffsetMatchingReference

        #list of input files
        slcs = []
        for i, frameNumber in enumerate(self._insar.referenceFrames):
            frameDir = 'f{}_{}'.format(i+1, frameNumber)
            swathDir = 's{}'.format(self._insar.startingSwath)
            slcs.append(os.path.join('../', frameDir, swathDir, self._insar.referenceSlc))

        #note that track parameters are updated after mosaicking
        #parameters update is checked, it is OK.
        frameMosaic(referenceTrack, slcs, self._insar.referenceSlc, 
            rangeOffsets, azimuthOffsets, 1, 1, 
            updateTrack=True, phaseCompensation=True, resamplingMethod=2)
        create_xml(self._insar.referenceSlc, referenceTrack.numberOfSamples, referenceTrack.numberOfLines, 'slc')
        referenceTrack.dopplerVsPixel = computeTrackDoppler(referenceTrack)

        #mosaic secondary slc
        #########################################################
        #choose offsets
        rangeOffsets = self._insar.frameRangeOffsetMatchingSecondary
        azimuthOffsets = self._insar.frameAzimuthOffsetMatchingSecondary

        #list of input files
        slcs = []
        for i, frameNumber in enumerate(self._insar.referenceFrames):
            frameDir = 'f{}_{}'.format(i+1, frameNumber)
            swathDir = 's{}'.format(self._insar.startingSwath)
            slcs.append(os.path.join('../', frameDir, swathDir, self._insar.secondarySlc))

        #note that track parameters are updated after mosaicking
        #parameters update is checked, it is OK.
        frameMosaic(secondaryTrack, slcs, self._insar.secondarySlc, 
            rangeOffsets, azimuthOffsets, 1, 1, 
            updateTrack=True, phaseCompensation=True, resamplingMethod=2)
        create_xml(self._insar.secondarySlc, secondaryTrack.numberOfSamples, secondaryTrack.numberOfLines, 'slc')
        secondaryTrack.dopplerVsPixel = computeTrackDoppler(secondaryTrack)


    #save parameter file inside denseoffset directory
    self._insar.saveProduct(referenceTrack, self._insar.referenceTrackParameter)
    self._insar.saveProduct(secondaryTrack, self._insar.secondaryTrackParameter)


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
