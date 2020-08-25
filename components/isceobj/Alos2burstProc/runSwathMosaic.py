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

    referenceTrack = self._insar.loadTrack(reference=True)
    secondaryTrack = self._insar.loadTrack(reference=False)


    for i, frameNumber in enumerate(self._insar.referenceFrames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        os.chdir(frameDir)

        mosaicDir = 'mosaic'
        os.makedirs(mosaicDir, exist_ok=True)
        os.chdir(mosaicDir)

        if self._insar.endingSwath-self._insar.startingSwath+1 == 1:
            import shutil
            swathDir = 's{}'.format(referenceTrack.frames[i].swaths[0].swathNumber)
            
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
            frame = referenceTrack.frames[i]
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

            #update frame parameters, secondary
            #########################################################
            frame = secondaryTrack.frames[i]
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
            self._insar.saveProduct(referenceTrack.frames[i], self._insar.referenceFrameParameter)
            self._insar.saveProduct(secondaryTrack.frames[i], self._insar.secondaryFrameParameter)

            os.chdir('../')

            continue

        #choose offsets
        numberOfFrames = len(referenceTrack.frames)
        numberOfSwaths = len(referenceTrack.frames[i].swaths)
        if self.swathOffsetMatching:
            #no need to do this as the API support 2-d list
            #rangeOffsets = (np.array(self._insar.swathRangeOffsetMatchingReference)).reshape(numberOfFrames, numberOfSwaths)
            #azimuthOffsets = (np.array(self._insar.swathAzimuthOffsetMatchingReference)).reshape(numberOfFrames, numberOfSwaths)
            rangeOffsets = self._insar.swathRangeOffsetMatchingReference
            azimuthOffsets = self._insar.swathAzimuthOffsetMatchingReference

        else:
            #rangeOffsets = (np.array(self._insar.swathRangeOffsetGeometricalReference)).reshape(numberOfFrames, numberOfSwaths)
            #azimuthOffsets = (np.array(self._insar.swathAzimuthOffsetGeometricalReference)).reshape(numberOfFrames, numberOfSwaths)
            rangeOffsets = self._insar.swathRangeOffsetGeometricalReference
            azimuthOffsets = self._insar.swathAzimuthOffsetGeometricalReference

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
        swathMosaic(referenceTrack.frames[i], inputAmplitudes, self._insar.amplitude, 
            rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, resamplingMethod=0)
        #mosaic interferograms
        swathMosaic(referenceTrack.frames[i], inputInterferograms, self._insar.interferogram, 
            rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, updateFrame=True, resamplingMethod=1)

        create_xml(self._insar.amplitude, referenceTrack.frames[i].numberOfSamples, referenceTrack.frames[i].numberOfLines, 'amp')
        create_xml(self._insar.interferogram, referenceTrack.frames[i].numberOfSamples, referenceTrack.frames[i].numberOfLines, 'int')

        #update secondary frame parameters here
        #no matching for secondary, always use geometry
        rangeOffsets = self._insar.swathRangeOffsetGeometricalSecondary
        azimuthOffsets = self._insar.swathAzimuthOffsetGeometricalSecondary
        rangeOffsets = rangeOffsets[i]
        azimuthOffsets = azimuthOffsets[i]
        swathMosaicParameters(secondaryTrack.frames[i], rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1)

        os.chdir('../')

        #save parameter file
        self._insar.saveProduct(referenceTrack.frames[i], self._insar.referenceFrameParameter)
        self._insar.saveProduct(secondaryTrack.frames[i], self._insar.secondaryFrameParameter)

        os.chdir('../')


    #mosaic spectral diversity interferograms
    for i, frameNumber in enumerate(self._insar.referenceFrames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        os.chdir(frameDir)

        mosaicDir = 'mosaic'
        os.makedirs(mosaicDir, exist_ok=True)
        os.chdir(mosaicDir)

        if self._insar.endingSwath-self._insar.startingSwath+1 == 1:
            import shutil
            swathDir = 's{}'.format(referenceTrack.frames[i].swaths[0].swathNumber)
            
            for sdFile in self._insar.interferogramSd:
                if not os.path.isfile(sdFile):
                    os.symlink(os.path.join('../', swathDir, 'spectral_diversity', sdFile), sdFile)
                shutil.copy2(os.path.join('../', swathDir, 'spectral_diversity', sdFile+'.vrt'), sdFile+'.vrt')
                shutil.copy2(os.path.join('../', swathDir, 'spectral_diversity', sdFile+'.xml'), sdFile+'.xml')

            os.chdir('../')
            os.chdir('../')
            continue

        #choose offsets
        numberOfFrames = len(referenceTrack.frames)
        numberOfSwaths = len(referenceTrack.frames[i].swaths)
        if self.swathOffsetMatching:
            #no need to do this as the API support 2-d list
            #rangeOffsets = (np.array(self._insar.swathRangeOffsetMatchingReference)).reshape(numberOfFrames, numberOfSwaths)
            #azimuthOffsets = (np.array(self._insar.swathAzimuthOffsetMatchingReference)).reshape(numberOfFrames, numberOfSwaths)
            rangeOffsets = self._insar.swathRangeOffsetMatchingReference
            azimuthOffsets = self._insar.swathAzimuthOffsetMatchingReference

        else:
            #rangeOffsets = (np.array(self._insar.swathRangeOffsetGeometricalReference)).reshape(numberOfFrames, numberOfSwaths)
            #azimuthOffsets = (np.array(self._insar.swathAzimuthOffsetGeometricalReference)).reshape(numberOfFrames, numberOfSwaths)
            rangeOffsets = self._insar.swathRangeOffsetGeometricalReference
            azimuthOffsets = self._insar.swathAzimuthOffsetGeometricalReference

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
            swathMosaic(referenceTrack.frames[i], inputSdList, outputSdFile, 
                rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, updateFrame=False, phaseCompensation=True, pcRangeLooks=5, pcAzimuthLooks=5, filt=True, resamplingMethod=1)

        for sdFile in self._insar.interferogramSd:
            create_xml(sdFile, referenceTrack.frames[i].numberOfSamples, referenceTrack.frames[i].numberOfLines, 'int')

        os.chdir('../')
        os.chdir('../')


    catalog.printToLog(logger, "runSwathMosaic")
    self._insar.procDoc.addAllFromCatalog(catalog)


