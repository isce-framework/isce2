#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging

import isceobj
from isceobj.Alos2Proc.runFrameMosaic import frameMosaic
from isceobj.Alos2Proc.runFrameMosaic import frameMosaicParameters
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml

logger = logging.getLogger('isce.alos2burstinsar.runFrameMosaic')

def runFrameMosaic(self):
    '''mosaic frames
    '''
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    referenceTrack = self._insar.loadTrack(reference=True)
    secondaryTrack = self._insar.loadTrack(reference=False)

    mosaicDir = 'insar'
    os.makedirs(mosaicDir, exist_ok=True)
    os.chdir(mosaicDir)

    numberOfFrames = len(referenceTrack.frames)
    if numberOfFrames == 1:
        import shutil
        frameDir = os.path.join('f1_{}/mosaic'.format(self._insar.referenceFrames[0]))
        if not os.path.isfile(self._insar.interferogram):
            os.symlink(os.path.join('../', frameDir, self._insar.interferogram), self._insar.interferogram)
        #shutil.copy2() can overwrite
        shutil.copy2(os.path.join('../', frameDir, self._insar.interferogram+'.vrt'), self._insar.interferogram+'.vrt')
        shutil.copy2(os.path.join('../', frameDir, self._insar.interferogram+'.xml'), self._insar.interferogram+'.xml')
        if not os.path.isfile(self._insar.amplitude):
            os.symlink(os.path.join('../', frameDir, self._insar.amplitude), self._insar.amplitude)
        shutil.copy2(os.path.join('../', frameDir, self._insar.amplitude+'.vrt'), self._insar.amplitude+'.vrt')
        shutil.copy2(os.path.join('../', frameDir, self._insar.amplitude+'.xml'), self._insar.amplitude+'.xml')

        # os.rename(os.path.join('../', frameDir, self._insar.interferogram), self._insar.interferogram)
        # os.rename(os.path.join('../', frameDir, self._insar.interferogram+'.vrt'), self._insar.interferogram+'.vrt')
        # os.rename(os.path.join('../', frameDir, self._insar.interferogram+'.xml'), self._insar.interferogram+'.xml')
        # os.rename(os.path.join('../', frameDir, self._insar.amplitude), self._insar.amplitude)
        # os.rename(os.path.join('../', frameDir, self._insar.amplitude+'.vrt'), self._insar.amplitude+'.vrt')
        # os.rename(os.path.join('../', frameDir, self._insar.amplitude+'.xml'), self._insar.amplitude+'.xml')

        #update track parameters
        #########################################################
        #mosaic size
        referenceTrack.numberOfSamples = referenceTrack.frames[0].numberOfSamples
        referenceTrack.numberOfLines = referenceTrack.frames[0].numberOfLines
        #NOTE THAT WE ARE STILL USING SINGLE LOOK PARAMETERS HERE
        #range parameters
        referenceTrack.startingRange = referenceTrack.frames[0].startingRange
        referenceTrack.rangeSamplingRate = referenceTrack.frames[0].rangeSamplingRate
        referenceTrack.rangePixelSize = referenceTrack.frames[0].rangePixelSize
        #azimuth parameters
        referenceTrack.sensingStart = referenceTrack.frames[0].sensingStart
        referenceTrack.prf = referenceTrack.frames[0].prf
        referenceTrack.azimuthPixelSize = referenceTrack.frames[0].azimuthPixelSize
        referenceTrack.azimuthLineInterval = referenceTrack.frames[0].azimuthLineInterval

        #update track parameters, secondary
        #########################################################
        #mosaic size
        secondaryTrack.numberOfSamples = secondaryTrack.frames[0].numberOfSamples
        secondaryTrack.numberOfLines = secondaryTrack.frames[0].numberOfLines
        #NOTE THAT WE ARE STILL USING SINGLE LOOK PARAMETERS HERE
        #range parameters
        secondaryTrack.startingRange = secondaryTrack.frames[0].startingRange
        secondaryTrack.rangeSamplingRate = secondaryTrack.frames[0].rangeSamplingRate
        secondaryTrack.rangePixelSize = secondaryTrack.frames[0].rangePixelSize
        #azimuth parameters
        secondaryTrack.sensingStart = secondaryTrack.frames[0].sensingStart
        secondaryTrack.prf = secondaryTrack.frames[0].prf
        secondaryTrack.azimuthPixelSize = secondaryTrack.frames[0].azimuthPixelSize
        secondaryTrack.azimuthLineInterval = secondaryTrack.frames[0].azimuthLineInterval

    else:
        #choose offsets
        if self.frameOffsetMatching:
            rangeOffsets = self._insar.frameRangeOffsetMatchingReference
            azimuthOffsets = self._insar.frameAzimuthOffsetMatchingReference
        else:
            rangeOffsets = self._insar.frameRangeOffsetGeometricalReference
            azimuthOffsets = self._insar.frameAzimuthOffsetGeometricalReference

        #list of input files
        inputInterferograms = []
        inputAmplitudes = []
        for i, frameNumber in enumerate(self._insar.referenceFrames):
            frameDir = 'f{}_{}'.format(i+1, frameNumber)
            inputInterferograms.append(os.path.join('../', frameDir, 'mosaic', self._insar.interferogram))
            inputAmplitudes.append(os.path.join('../', frameDir, 'mosaic', self._insar.amplitude))

        #note that track parameters are updated after mosaicking
        #mosaic amplitudes
        frameMosaic(referenceTrack, inputAmplitudes, self._insar.amplitude, 
            rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, 
            updateTrack=False, phaseCompensation=False, resamplingMethod=0)
        #mosaic interferograms
        (phaseDiffEst, phaseDiffUsed, phaseDiffSource, numberOfValidSamples) = frameMosaic(referenceTrack, inputInterferograms, self._insar.interferogram, 
            rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, 
            updateTrack=True, phaseCompensation=True, resamplingMethod=1)

        create_xml(self._insar.amplitude, referenceTrack.numberOfSamples, referenceTrack.numberOfLines, 'amp')
        create_xml(self._insar.interferogram, referenceTrack.numberOfSamples, referenceTrack.numberOfLines, 'int')

        catalog.addItem('frame phase diff estimated', phaseDiffEst[1:], 'runFrameMosaic')
        catalog.addItem('frame phase diff used', phaseDiffUsed[1:], 'runFrameMosaic')
        catalog.addItem('frame phase diff used source', phaseDiffSource[1:], 'runFrameMosaic')
        catalog.addItem('frame phase diff samples used', numberOfValidSamples[1:], 'runFrameMosaic')

        #update secondary parameters here
        #do not match for secondary, always use geometrical
        rangeOffsets = self._insar.frameRangeOffsetGeometricalSecondary
        azimuthOffsets = self._insar.frameAzimuthOffsetGeometricalSecondary
        frameMosaicParameters(secondaryTrack, rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1)

    os.chdir('../')
    #save parameter file
    self._insar.saveProduct(referenceTrack, self._insar.referenceTrackParameter)
    self._insar.saveProduct(secondaryTrack, self._insar.secondaryTrackParameter)



    #mosaic spectral diversity inteferograms
    mosaicDir = 'sd'
    os.makedirs(mosaicDir, exist_ok=True)
    os.chdir(mosaicDir)

    numberOfFrames = len(referenceTrack.frames)
    if numberOfFrames == 1:
        import shutil
        frameDir = os.path.join('f1_{}/mosaic'.format(self._insar.referenceFrames[0]))
        for sdFile in self._insar.interferogramSd:
            if not os.path.isfile(sdFile):
                os.symlink(os.path.join('../', frameDir, sdFile), sdFile)
            shutil.copy2(os.path.join('../', frameDir, sdFile+'.vrt'), sdFile+'.vrt')
            shutil.copy2(os.path.join('../', frameDir, sdFile+'.xml'), sdFile+'.xml')
    else:
        #choose offsets
        if self.frameOffsetMatching:
            rangeOffsets = self._insar.frameRangeOffsetMatchingReference
            azimuthOffsets = self._insar.frameAzimuthOffsetMatchingReference
        else:
            rangeOffsets = self._insar.frameRangeOffsetGeometricalReference
            azimuthOffsets = self._insar.frameAzimuthOffsetGeometricalReference

        #list of input files
        inputSd = [[], [], []]
        for i, frameNumber in enumerate(self._insar.referenceFrames):
            frameDir = 'f{}_{}'.format(i+1, frameNumber)
            for k, sdFile in enumerate(self._insar.interferogramSd):
                inputSd[k].append(os.path.join('../', frameDir, 'mosaic', sdFile))

        #mosaic spectral diversity interferograms
        for i, (inputSdList, outputSdFile) in enumerate(zip(inputSd, self._insar.interferogramSd)):
            (phaseDiffEst, phaseDiffUsed, phaseDiffSource, numberOfValidSamples) = frameMosaic(referenceTrack, inputSdList, outputSdFile, 
                rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, 
                updateTrack=False, phaseCompensation=True, resamplingMethod=1)

        catalog.addItem('sd {} frame phase diff estimated'.format(i+1), phaseDiffEst[1:], 'runFrameMosaic')
        catalog.addItem('sd {} frame phase diff used'.format(i+1), phaseDiffUsed[1:], 'runFrameMosaic')
        catalog.addItem('sd {} frame phase diff used source'.format(i+1), phaseDiffSource[1:], 'runFrameMosaic')
        catalog.addItem('sd {} frame phase diff samples used'.format(i+1), numberOfValidSamples[1:], 'runFrameMosaic')


        for sdFile in self._insar.interferogramSd:
            create_xml(sdFile, referenceTrack.numberOfSamples, referenceTrack.numberOfLines, 'int')

    os.chdir('../')


    catalog.printToLog(logger, "runFrameMosaic")
    self._insar.procDoc.addAllFromCatalog(catalog)


