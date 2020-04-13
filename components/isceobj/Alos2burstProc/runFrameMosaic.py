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

    masterTrack = self._insar.loadTrack(master=True)
    slaveTrack = self._insar.loadTrack(master=False)

    mosaicDir = 'insar'
    os.makedirs(mosaicDir, exist_ok=True)
    os.chdir(mosaicDir)

    numberOfFrames = len(masterTrack.frames)
    if numberOfFrames == 1:
        import shutil
        frameDir = os.path.join('f1_{}/mosaic'.format(self._insar.masterFrames[0]))
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
        masterTrack.numberOfSamples = masterTrack.frames[0].numberOfSamples
        masterTrack.numberOfLines = masterTrack.frames[0].numberOfLines
        #NOTE THAT WE ARE STILL USING SINGLE LOOK PARAMETERS HERE
        #range parameters
        masterTrack.startingRange = masterTrack.frames[0].startingRange
        masterTrack.rangeSamplingRate = masterTrack.frames[0].rangeSamplingRate
        masterTrack.rangePixelSize = masterTrack.frames[0].rangePixelSize
        #azimuth parameters
        masterTrack.sensingStart = masterTrack.frames[0].sensingStart
        masterTrack.prf = masterTrack.frames[0].prf
        masterTrack.azimuthPixelSize = masterTrack.frames[0].azimuthPixelSize
        masterTrack.azimuthLineInterval = masterTrack.frames[0].azimuthLineInterval

        #update track parameters, slave
        #########################################################
        #mosaic size
        slaveTrack.numberOfSamples = slaveTrack.frames[0].numberOfSamples
        slaveTrack.numberOfLines = slaveTrack.frames[0].numberOfLines
        #NOTE THAT WE ARE STILL USING SINGLE LOOK PARAMETERS HERE
        #range parameters
        slaveTrack.startingRange = slaveTrack.frames[0].startingRange
        slaveTrack.rangeSamplingRate = slaveTrack.frames[0].rangeSamplingRate
        slaveTrack.rangePixelSize = slaveTrack.frames[0].rangePixelSize
        #azimuth parameters
        slaveTrack.sensingStart = slaveTrack.frames[0].sensingStart
        slaveTrack.prf = slaveTrack.frames[0].prf
        slaveTrack.azimuthPixelSize = slaveTrack.frames[0].azimuthPixelSize
        slaveTrack.azimuthLineInterval = slaveTrack.frames[0].azimuthLineInterval

    else:
        #choose offsets
        if self.frameOffsetMatching:
            rangeOffsets = self._insar.frameRangeOffsetMatchingMaster
            azimuthOffsets = self._insar.frameAzimuthOffsetMatchingMaster
        else:
            rangeOffsets = self._insar.frameRangeOffsetGeometricalMaster
            azimuthOffsets = self._insar.frameAzimuthOffsetGeometricalMaster

        #list of input files
        inputInterferograms = []
        inputAmplitudes = []
        for i, frameNumber in enumerate(self._insar.masterFrames):
            frameDir = 'f{}_{}'.format(i+1, frameNumber)
            inputInterferograms.append(os.path.join('../', frameDir, 'mosaic', self._insar.interferogram))
            inputAmplitudes.append(os.path.join('../', frameDir, 'mosaic', self._insar.amplitude))

        #note that track parameters are updated after mosaicking
        #mosaic amplitudes
        frameMosaic(masterTrack, inputAmplitudes, self._insar.amplitude, 
            rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, 
            updateTrack=False, phaseCompensation=False, resamplingMethod=0)
        #mosaic interferograms
        frameMosaic(masterTrack, inputInterferograms, self._insar.interferogram, 
            rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, 
            updateTrack=True, phaseCompensation=True, resamplingMethod=1)

        create_xml(self._insar.amplitude, masterTrack.numberOfSamples, masterTrack.numberOfLines, 'amp')
        create_xml(self._insar.interferogram, masterTrack.numberOfSamples, masterTrack.numberOfLines, 'int')

        #update slave parameters here
        #do not match for slave, always use geometrical
        rangeOffsets = self._insar.frameRangeOffsetGeometricalSlave
        azimuthOffsets = self._insar.frameAzimuthOffsetGeometricalSlave
        frameMosaicParameters(slaveTrack, rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1)

    os.chdir('../')
    #save parameter file
    self._insar.saveProduct(masterTrack, self._insar.masterTrackParameter)
    self._insar.saveProduct(slaveTrack, self._insar.slaveTrackParameter)



    #mosaic spectral diversity inteferograms
    mosaicDir = 'sd'
    os.makedirs(mosaicDir, exist_ok=True)
    os.chdir(mosaicDir)

    numberOfFrames = len(masterTrack.frames)
    if numberOfFrames == 1:
        import shutil
        frameDir = os.path.join('f1_{}/mosaic'.format(self._insar.masterFrames[0]))
        for sdFile in self._insar.interferogramSd:
            if not os.path.isfile(sdFile):
                os.symlink(os.path.join('../', frameDir, sdFile), sdFile)
            shutil.copy2(os.path.join('../', frameDir, sdFile+'.vrt'), sdFile+'.vrt')
            shutil.copy2(os.path.join('../', frameDir, sdFile+'.xml'), sdFile+'.xml')
    else:
        #choose offsets
        if self.frameOffsetMatching:
            rangeOffsets = self._insar.frameRangeOffsetMatchingMaster
            azimuthOffsets = self._insar.frameAzimuthOffsetMatchingMaster
        else:
            rangeOffsets = self._insar.frameRangeOffsetGeometricalMaster
            azimuthOffsets = self._insar.frameAzimuthOffsetGeometricalMaster

        #list of input files
        inputSd = [[], [], []]
        for i, frameNumber in enumerate(self._insar.masterFrames):
            frameDir = 'f{}_{}'.format(i+1, frameNumber)
            for k, sdFile in enumerate(self._insar.interferogramSd):
                inputSd[k].append(os.path.join('../', frameDir, 'mosaic', sdFile))

        #mosaic spectral diversity interferograms
        for inputSdList, outputSdFile in zip(inputSd, self._insar.interferogramSd):
            frameMosaic(masterTrack, inputSdList, outputSdFile, 
                rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, 
                updateTrack=False, phaseCompensation=True, resamplingMethod=1)

        for sdFile in self._insar.interferogramSd:
            create_xml(sdFile, masterTrack.numberOfSamples, masterTrack.numberOfLines, 'int')

    os.chdir('../')


    catalog.printToLog(logger, "runFrameMosaic")
    self._insar.procDoc.addAllFromCatalog(catalog)


