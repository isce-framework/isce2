#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging

import isceobj
from isceobj.Alos2Proc.runSwathMosaic import swathMosaic
from isceobj.Alos2Proc.runSwathMosaic import swathMosaicParameters
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml

logger = logging.getLogger('isce.alos2burstinsar.runSwathMosaic')

def runSwathMosaic(self):
    '''mosaic subswaths
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
            import shutil
            swathDir = 's{}'.format(masterTrack.frames[i].swaths[0].swathNumber)
            
            if not os.path.isfile(self._insar.interferogram):
                os.symlink(os.path.join('../', swathDir, self._insar.interferogram), self._insar.interferogram)
            shutil.copy2(os.path.join('../', swathDir, self._insar.interferogram+'.vrt'), self._insar.interferogram+'.vrt')
            shutil.copy2(os.path.join('../', swathDir, self._insar.interferogram+'.xml'), self._insar.interferogram+'.xml')
            if not os.path.isfile(self._insar.amplitude):
                os.symlink(os.path.join('../', swathDir, self._insar.amplitude), self._insar.amplitude)
            shutil.copy2(os.path.join('../', swathDir, self._insar.amplitude+'.vrt'), self._insar.amplitude+'.vrt')
            shutil.copy2(os.path.join('../', swathDir, self._insar.amplitude+'.xml'), self._insar.amplitude+'.xml')

            # os.rename(os.path.join('../', swathDir, self._insar.interferogram), self._insar.interferogram)
            # os.rename(os.path.join('../', swathDir, self._insar.interferogram+'.vrt'), self._insar.interferogram+'.vrt')
            # os.rename(os.path.join('../', swathDir, self._insar.interferogram+'.xml'), self._insar.interferogram+'.xml')
            # os.rename(os.path.join('../', swathDir, self._insar.amplitude), self._insar.amplitude)
            # os.rename(os.path.join('../', swathDir, self._insar.amplitude+'.vrt'), self._insar.amplitude+'.vrt')
            # os.rename(os.path.join('../', swathDir, self._insar.amplitude+'.xml'), self._insar.amplitude+'.xml')

            #update frame parameters
            #########################################################
            frame = masterTrack.frames[i]
            infImg = isceobj.createImage()
            infImg.load(self._insar.interferogram+'.xml')
            #mosaic size
            frame.numberOfSamples = infImg.width
            frame.numberOfLines = infImg.length
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

            #update frame parameters, slave
            #########################################################
            frame = slaveTrack.frames[i]
            #mosaic size
            frame.numberOfSamples = int(frame.swaths[0].numberOfSamples/self._insar.numberRangeLooks1)
            frame.numberOfLines = int(frame.swaths[0].numberOfLines/self._insar.numberAzimuthLooks1)
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

            os.chdir('../')

            #save parameter file
            self._insar.saveProduct(masterTrack.frames[i], self._insar.masterFrameParameter)
            self._insar.saveProduct(slaveTrack.frames[i], self._insar.slaveFrameParameter)

            os.chdir('../')

            continue

        #choose offsets
        numberOfFrames = len(masterTrack.frames)
        numberOfSwaths = len(masterTrack.frames[i].swaths)
        if self.swathOffsetMatching:
            #no need to do this as the API support 2-d list
            #rangeOffsets = (np.array(self._insar.swathRangeOffsetMatchingMaster)).reshape(numberOfFrames, numberOfSwaths)
            #azimuthOffsets = (np.array(self._insar.swathAzimuthOffsetMatchingMaster)).reshape(numberOfFrames, numberOfSwaths)
            rangeOffsets = self._insar.swathRangeOffsetMatchingMaster
            azimuthOffsets = self._insar.swathAzimuthOffsetMatchingMaster

        else:
            #rangeOffsets = (np.array(self._insar.swathRangeOffsetGeometricalMaster)).reshape(numberOfFrames, numberOfSwaths)
            #azimuthOffsets = (np.array(self._insar.swathAzimuthOffsetGeometricalMaster)).reshape(numberOfFrames, numberOfSwaths)
            rangeOffsets = self._insar.swathRangeOffsetGeometricalMaster
            azimuthOffsets = self._insar.swathAzimuthOffsetGeometricalMaster

        rangeOffsets = rangeOffsets[i]
        azimuthOffsets = azimuthOffsets[i]

        #list of input files
        inputInterferograms = []
        inputAmplitudes = []
        for j, swathNumber in enumerate(range(self._insar.startingSwath, self._insar.endingSwath + 1)):
            swathDir = 's{}'.format(swathNumber)
            inputInterferograms.append(os.path.join('../', swathDir, self._insar.interferogram))
            inputAmplitudes.append(os.path.join('../', swathDir, self._insar.amplitude))

        #note that frame parameters are updated after mosaicking
        #mosaic amplitudes
        swathMosaic(masterTrack.frames[i], inputAmplitudes, self._insar.amplitude, 
            rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, resamplingMethod=0)
        #mosaic interferograms
        swathMosaic(masterTrack.frames[i], inputInterferograms, self._insar.interferogram, 
            rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, updateFrame=True, resamplingMethod=1)

        create_xml(self._insar.amplitude, masterTrack.frames[i].numberOfSamples, masterTrack.frames[i].numberOfLines, 'amp')
        create_xml(self._insar.interferogram, masterTrack.frames[i].numberOfSamples, masterTrack.frames[i].numberOfLines, 'int')

        #update slave frame parameters here
        #no matching for slave, always use geometry
        rangeOffsets = self._insar.swathRangeOffsetGeometricalSlave
        azimuthOffsets = self._insar.swathAzimuthOffsetGeometricalSlave
        rangeOffsets = rangeOffsets[i]
        azimuthOffsets = azimuthOffsets[i]
        swathMosaicParameters(slaveTrack.frames[i], rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1)

        os.chdir('../')

        #save parameter file
        self._insar.saveProduct(masterTrack.frames[i], self._insar.masterFrameParameter)
        self._insar.saveProduct(slaveTrack.frames[i], self._insar.slaveFrameParameter)

        os.chdir('../')


    #mosaic spectral diversity interferograms
    for i, frameNumber in enumerate(self._insar.masterFrames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        os.chdir(frameDir)

        mosaicDir = 'mosaic'
        if not os.path.exists(mosaicDir):
            os.makedirs(mosaicDir)
        os.chdir(mosaicDir)

        if self._insar.endingSwath-self._insar.startingSwath+1 == 1:
            import shutil
            swathDir = 's{}'.format(masterTrack.frames[i].swaths[0].swathNumber)
            
            for sdFile in self._insar.interferogramSd:
                if not os.path.isfile(sdFile):
                    os.symlink(os.path.join('../', swathDir, 'spectral_diversity', sdFile), sdFile)
                shutil.copy2(os.path.join('../', swathDir, 'spectral_diversity', sdFile+'.vrt'), sdFile+'.vrt')
                shutil.copy2(os.path.join('../', swathDir, 'spectral_diversity', sdFile+'.xml'), sdFile+'.xml')

            os.chdir('../')
            os.chdir('../')
            continue

        #choose offsets
        numberOfFrames = len(masterTrack.frames)
        numberOfSwaths = len(masterTrack.frames[i].swaths)
        if self.swathOffsetMatching:
            #no need to do this as the API support 2-d list
            #rangeOffsets = (np.array(self._insar.swathRangeOffsetMatchingMaster)).reshape(numberOfFrames, numberOfSwaths)
            #azimuthOffsets = (np.array(self._insar.swathAzimuthOffsetMatchingMaster)).reshape(numberOfFrames, numberOfSwaths)
            rangeOffsets = self._insar.swathRangeOffsetMatchingMaster
            azimuthOffsets = self._insar.swathAzimuthOffsetMatchingMaster

        else:
            #rangeOffsets = (np.array(self._insar.swathRangeOffsetGeometricalMaster)).reshape(numberOfFrames, numberOfSwaths)
            #azimuthOffsets = (np.array(self._insar.swathAzimuthOffsetGeometricalMaster)).reshape(numberOfFrames, numberOfSwaths)
            rangeOffsets = self._insar.swathRangeOffsetGeometricalMaster
            azimuthOffsets = self._insar.swathAzimuthOffsetGeometricalMaster

        rangeOffsets = rangeOffsets[i]
        azimuthOffsets = azimuthOffsets[i]

        #list of input files
        inputSd = [[], [], []]
        for j, swathNumber in enumerate(range(self._insar.startingSwath, self._insar.endingSwath + 1)):
            swathDir = 's{}'.format(swathNumber)
            for k, sdFile in enumerate(self._insar.interferogramSd):
                inputSd[k].append(os.path.join('../', swathDir, 'spectral_diversity', sdFile))

        #mosaic spectral diversity interferograms
        for inputSdList, outputSdFile in zip(inputSd, self._insar.interferogramSd):
            swathMosaic(masterTrack.frames[i], inputSdList, outputSdFile, 
                rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, updateFrame=False, phaseCompensation=True, pcRangeLooks=5, pcAzimuthLooks=5, filt=True, resamplingMethod=1)

        for sdFile in self._insar.interferogramSd:
            create_xml(sdFile, masterTrack.frames[i].numberOfSamples, masterTrack.frames[i].numberOfLines, 'int')

        os.chdir('../')
        os.chdir('../')


    catalog.printToLog(logger, "runSwathMosaic")
    self._insar.procDoc.addAllFromCatalog(catalog)


