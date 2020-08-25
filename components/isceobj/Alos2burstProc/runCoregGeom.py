#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging

import isceobj
from isceobj.Alos2Proc.runRdr2Geo import topoCPU
from isceobj.Alos2Proc.runRdr2Geo import topoGPU
from isceobj.Alos2Proc.runGeo2Rdr import geo2RdrCPU
from isceobj.Alos2Proc.runGeo2Rdr import geo2RdrGPU
from isceobj.Alos2Proc.Alos2ProcPublic import waterBodyRadar
from isceobj.Alos2Proc.Alos2ProcPublic import resampleBursts
from isceobj.Alos2Proc.Alos2ProcPublic import mosaicBurstAmplitude
from isceobj.Alos2Proc.Alos2ProcPublic import mosaicBurstInterferogram

logger = logging.getLogger('isce.alos2burstinsar.runCoregGeom')

def runCoregGeom(self):
    '''compute geometric offset
    '''
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    referenceTrack = self._insar.loadTrack(reference=True)
    secondaryTrack = self._insar.loadTrack(reference=False)

    demFile = os.path.abspath(self._insar.dem)
    wbdFile = os.path.abspath(self._insar.wbd)
###############################################################################

    for i, frameNumber in enumerate(self._insar.referenceFrames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        os.chdir(frameDir)
        for j, swathNumber in enumerate(range(self._insar.startingSwath, self._insar.endingSwath + 1)):
            swathDir = 's{}'.format(swathNumber)
            os.chdir(swathDir)

            print('processing frame {}, swath {}'.format(frameNumber, swathNumber))

            referenceSwath = referenceTrack.frames[i].swaths[j]
            secondarySwath = secondaryTrack.frames[i].swaths[j]


            ##################################################
            # compute geometric offsets
            ##################################################
            #set up track parameters just for computing offsets
            #ALL track parameters are listed here
            #reference
            #referenceTrack.passDirection = 
            #referenceTrack.pointingDirection = 
            #referenceTrack.operationMode = 
            #referenceTrack.radarWavelength = 
            referenceTrack.numberOfSamples = referenceSwath.numberOfSamples
            referenceTrack.numberOfLines = referenceSwath.numberOfLines
            referenceTrack.startingRange = referenceSwath.startingRange
            #referenceTrack.rangeSamplingRate = 
            referenceTrack.rangePixelSize = referenceSwath.rangePixelSize
            referenceTrack.sensingStart = referenceSwath.sensingStart
            #referenceTrack.prf = 
            #referenceTrack.azimuthPixelSize = 
            referenceTrack.azimuthLineInterval = referenceSwath.azimuthLineInterval
            #referenceTrack.dopplerVsPixel = 
            #referenceTrack.frames = 
            #referenceTrack.orbit = 

            #secondary
            secondaryTrack.numberOfSamples = secondarySwath.numberOfSamples
            secondaryTrack.numberOfLines = secondarySwath.numberOfLines
            secondaryTrack.startingRange = secondarySwath.startingRange
            secondaryTrack.rangePixelSize = secondarySwath.rangePixelSize
            secondaryTrack.sensingStart = secondarySwath.sensingStart
            secondaryTrack.azimuthLineInterval = secondarySwath.azimuthLineInterval

            if self.useGPU and self._insar.hasGPU():
                topoGPU(referenceTrack, 1, 1, demFile, 
                               self._insar.latitude, self._insar.longitude, self._insar.height, self._insar.los)
                geo2RdrGPU(secondaryTrack, 1, 1, 
                    self._insar.latitude, self._insar.longitude, self._insar.height, self._insar.rangeOffset, self._insar.azimuthOffset)
            else:
                topoCPU(referenceTrack, 1, 1, demFile, 
                               self._insar.latitude, self._insar.longitude, self._insar.height, self._insar.los)
                geo2RdrCPU(secondaryTrack, 1, 1, 
                    self._insar.latitude, self._insar.longitude, self._insar.height, self._insar.rangeOffset, self._insar.azimuthOffset)

            waterBodyRadar(self._insar.latitude, self._insar.longitude, wbdFile, self._insar.wbdOut)

            #clear up, leaving only range/azimuth offsets
            os.remove(self._insar.latitude)
            os.remove(self._insar.latitude+'.vrt')
            os.remove(self._insar.latitude+'.xml')
            os.remove(self._insar.longitude)
            os.remove(self._insar.longitude+'.vrt')
            os.remove(self._insar.longitude+'.xml')
            os.remove(self._insar.height)
            os.remove(self._insar.height+'.vrt')
            os.remove(self._insar.height+'.xml')
            os.remove(self._insar.los)
            os.remove(self._insar.los+'.vrt')
            os.remove(self._insar.los+'.xml')


            ##################################################
            # resample bursts
            ##################################################
            secondaryBurstResampledDir = self._insar.secondaryBurstPrefix + '_1_coreg_geom'
            #interferogramDir = self._insar.referenceBurstPrefix + '-' + self._insar.secondaryBurstPrefix + '_coreg_geom'
            interferogramDir = 'burst_interf_1_coreg_geom'
            interferogramPrefix = self._insar.referenceBurstPrefix + '-' + self._insar.secondaryBurstPrefix
            resampleBursts(referenceSwath, secondarySwath, 
                self._insar.referenceBurstPrefix, self._insar.secondaryBurstPrefix, secondaryBurstResampledDir, interferogramDir,
                self._insar.referenceBurstPrefix, self._insar.secondaryBurstPrefix, self._insar.secondaryBurstPrefix, interferogramPrefix, 
                self._insar.rangeOffset, self._insar.azimuthOffset, rangeOffsetResidual=0, azimuthOffsetResidual=0)


            ##################################################
            # mosaic burst amplitudes and interferograms
            ##################################################
            os.chdir(secondaryBurstResampledDir)
            mosaicBurstAmplitude(referenceSwath, self._insar.secondaryBurstPrefix, self._insar.secondaryMagnitude, numberOfLooksThreshold=4)
            os.chdir('../')

            #the interferogram is not good enough, do not mosaic
            mosaic=False
            if mosaic:
                os.chdir(interferogramDir)
                mosaicBurstInterferogram(referenceSwath, interferogramPrefix, self._insar.interferogram, numberOfLooksThreshold=4)
                os.chdir('../')


            os.chdir('../')
        os.chdir('../')

###############################################################################
    catalog.printToLog(logger, "runCoregGeom")
    self._insar.procDoc.addAllFromCatalog(catalog)



