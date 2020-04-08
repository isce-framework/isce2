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

    masterTrack = self._insar.loadTrack(master=True)
    slaveTrack = self._insar.loadTrack(master=False)

    demFile = os.path.abspath(self._insar.dem)
    wbdFile = os.path.abspath(self._insar.wbd)
###############################################################################

    for i, frameNumber in enumerate(self._insar.masterFrames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        os.chdir(frameDir)
        for j, swathNumber in enumerate(range(self._insar.startingSwath, self._insar.endingSwath + 1)):
            swathDir = 's{}'.format(swathNumber)
            os.chdir(swathDir)

            print('processing frame {}, swath {}'.format(frameNumber, swathNumber))

            masterSwath = masterTrack.frames[i].swaths[j]
            slaveSwath = slaveTrack.frames[i].swaths[j]


            ##################################################
            # compute geometric offsets
            ##################################################
            #set up track parameters just for computing offsets
            #ALL track parameters are listed here
            #master
            #masterTrack.passDirection = 
            #masterTrack.pointingDirection = 
            #masterTrack.operationMode = 
            #masterTrack.radarWavelength = 
            masterTrack.numberOfSamples = masterSwath.numberOfSamples
            masterTrack.numberOfLines = masterSwath.numberOfLines
            masterTrack.startingRange = masterSwath.startingRange
            #masterTrack.rangeSamplingRate = 
            masterTrack.rangePixelSize = masterSwath.rangePixelSize
            masterTrack.sensingStart = masterSwath.sensingStart
            #masterTrack.prf = 
            #masterTrack.azimuthPixelSize = 
            masterTrack.azimuthLineInterval = masterSwath.azimuthLineInterval
            #masterTrack.dopplerVsPixel = 
            #masterTrack.frames = 
            #masterTrack.orbit = 

            #slave
            slaveTrack.numberOfSamples = slaveSwath.numberOfSamples
            slaveTrack.numberOfLines = slaveSwath.numberOfLines
            slaveTrack.startingRange = slaveSwath.startingRange
            slaveTrack.rangePixelSize = slaveSwath.rangePixelSize
            slaveTrack.sensingStart = slaveSwath.sensingStart
            slaveTrack.azimuthLineInterval = slaveSwath.azimuthLineInterval

            if self.useGPU and self._insar.hasGPU():
                topoGPU(masterTrack, 1, 1, demFile, 
                               self._insar.latitude, self._insar.longitude, self._insar.height, self._insar.los)
                geo2RdrGPU(slaveTrack, 1, 1, 
                    self._insar.latitude, self._insar.longitude, self._insar.height, self._insar.rangeOffset, self._insar.azimuthOffset)
            else:
                topoCPU(masterTrack, 1, 1, demFile, 
                               self._insar.latitude, self._insar.longitude, self._insar.height, self._insar.los)
                geo2RdrCPU(slaveTrack, 1, 1, 
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
            slaveBurstResampledDir = self._insar.slaveBurstPrefix + '_1_coreg_geom'
            #interferogramDir = self._insar.masterBurstPrefix + '-' + self._insar.slaveBurstPrefix + '_coreg_geom'
            interferogramDir = 'burst_interf_1_coreg_geom'
            interferogramPrefix = self._insar.masterBurstPrefix + '-' + self._insar.slaveBurstPrefix
            resampleBursts(masterSwath, slaveSwath, 
                self._insar.masterBurstPrefix, self._insar.slaveBurstPrefix, slaveBurstResampledDir, interferogramDir,
                self._insar.masterBurstPrefix, self._insar.slaveBurstPrefix, self._insar.slaveBurstPrefix, interferogramPrefix, 
                self._insar.rangeOffset, self._insar.azimuthOffset, rangeOffsetResidual=0, azimuthOffsetResidual=0)


            ##################################################
            # mosaic burst amplitudes and interferograms
            ##################################################
            os.chdir(slaveBurstResampledDir)
            mosaicBurstAmplitude(masterSwath, self._insar.slaveBurstPrefix, self._insar.slaveMagnitude, numberOfLooksThreshold=4)
            os.chdir('../')

            #the interferogram is not good enough, do not mosaic
            mosaic=False
            if mosaic:
                os.chdir(interferogramDir)
                mosaicBurstInterferogram(masterSwath, interferogramPrefix, self._insar.interferogram, numberOfLooksThreshold=4)
                os.chdir('../')


            os.chdir('../')
        os.chdir('../')

###############################################################################
    catalog.printToLog(logger, "runCoregGeom")
    self._insar.procDoc.addAllFromCatalog(catalog)



