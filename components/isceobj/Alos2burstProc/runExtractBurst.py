#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import glob
import logging
import datetime
#import subprocess
import numpy as np

import isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
from isceobj.Alos2Proc.Alos2ProcPublic import mosaicBurstAmplitude
from contrib.alos2proc.alos2proc import extract_burst

logger = logging.getLogger('isce.alos2burstinsar.runExtractBurst')

def runExtractBurst(self):
    '''extract bursts.
    '''
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    referenceTrack = self._insar.loadTrack(reference=True)
    secondaryTrack = self._insar.loadTrack(reference=False)

    #demFile = os.path.abspath(self._insar.dem)
    #wbdFile = os.path.abspath(self._insar.wbd)
###############################################################################
    for i, frameNumber in enumerate(self._insar.referenceFrames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        os.chdir(frameDir)
        for j, swathNumber in enumerate(range(self._insar.startingSwath, self._insar.endingSwath + 1)):
            swathDir = 's{}'.format(swathNumber)
            os.chdir(swathDir)

            print('extracting bursts frame {}, swath {}'.format(frameNumber, swathNumber))

            az_ratio1 = 20.0
            for k in range(2):
                if k==0:
                    #reference
                    swath = referenceTrack.frames[i].swaths[j]
                    unsynLines = self._insar.burstUnsynchronizedTime * swath.prf
                    extractDir = self._insar.referenceBurstPrefix
                    burstPrefix = self._insar.referenceBurstPrefix
                    fullApertureSlc = self._insar.referenceSlc
                    magnitude = self._insar.referenceMagnitude
                else:
                    #secondary
                    swath = secondaryTrack.frames[i].swaths[j]
                    unsynLines = -self._insar.burstUnsynchronizedTime * swath.prf
                    extractDir = self._insar.secondaryBurstPrefix
                    burstPrefix = self._insar.secondaryBurstPrefix
                    fullApertureSlc = self._insar.secondarySlc
                    magnitude = self._insar.secondaryMagnitude

                #UPDATE SWATH PARAMETERS 1
                #########################################################################################
                if self._insar.burstSynchronization <= self.burstSynchronizationThreshold:
                    swath.burstLength -= abs(unsynLines)
                    if unsynLines < 0:
                        swath.burstStartTime += datetime.timedelta(seconds=abs(unsynLines)/swath.prf)
                #########################################################################################

                #extract burst
                os.makedirs(extractDir, exist_ok=True)
                os.chdir(extractDir)
                if os.path.isfile(os.path.join('../', fullApertureSlc)):
                    os.rename(os.path.join('../', fullApertureSlc), fullApertureSlc)
                os.rename(os.path.join('../', fullApertureSlc+'.vrt'), fullApertureSlc+'.vrt')
                os.rename(os.path.join('../', fullApertureSlc+'.xml'), fullApertureSlc+'.xml')

                extract_burst(fullApertureSlc, burstPrefix, swath.prf, swath.prfFraction, swath.burstLength, swath.burstCycleLength-swath.burstLength, \
                (swath.burstStartTime - swath.sensingStart).total_seconds() * swath.prf, swath.azimuthFmrateVsPixel, swath.dopplerVsPixel, az_ratio1, 0.0)

                #read output parameters
                with open('extract_burst.txt', 'r') as f:
                    lines = f.readlines()
                offsetFromFirstBurst = []
                for linex in lines:
                    if 'total number of bursts extracted' in linex:
                        numberOfBursts = int(linex.split(':')[1])
                    if 'output burst length' in linex:
                        burstSlcNumberOfLines = int(linex.split(':')[1])
                    if 'line number of first line of first output burst in original SLC (1.0/prf)' in linex:
                        fb_ln = float(linex.split(':')[1])
                    if 'bsl of first output burst' in linex:
                        bsl_firstburst = float(linex.split(':')[1])
                    if 'offset from first burst' in linex:
                        offsetFromFirstBurst.append(int(linex.split(',')[0].split(':')[1]))

                #time of first line of first burst raw
                firstBurstRawStartTime = swath.sensingStart + datetime.timedelta(seconds=bsl_firstburst/swath.prf)

                #time of first line of first burst slc
                #original time is at the upper edge of first line, we change it to center of first line. 
                sensingStart = swath.sensingStart + datetime.timedelta(seconds=fb_ln/swath.prf+(az_ratio1-1.0)/2.0/swath.prf)
                numberOfLines = offsetFromFirstBurst[numberOfBursts-1] + burstSlcNumberOfLines

                for ii in range(numberOfBursts):
                    burstFile = burstPrefix + '_%02d.slc'%(ii+1)
                    create_xml(burstFile, swath.numberOfSamples, burstSlcNumberOfLines, 'slc')

                #UPDATE SWATH PARAMETERS 2
                #########################################################################################
                swath.numberOfLines = numberOfLines
                #this is also the time of the first line of the first burst slc
                swath.sensingStart = sensingStart
                swath.azimuthPixelSize = az_ratio1 * swath.azimuthPixelSize
                swath.azimuthLineInterval = az_ratio1 * swath.azimuthLineInterval

                swath.numberOfBursts = numberOfBursts
                swath.firstBurstRawStartTime = firstBurstRawStartTime
                swath.firstBurstSlcStartTime = sensingStart
                swath.burstSlcFirstLineOffsets = offsetFromFirstBurst
                swath.burstSlcNumberOfSamples = swath.numberOfSamples
                swath.burstSlcNumberOfLines = burstSlcNumberOfLines
                #########################################################################################

                #create a magnitude image
                mosaicBurstAmplitude(swath, burstPrefix, magnitude, numberOfLooksThreshold=4)

                os.chdir('../')
            os.chdir('../')
        self._insar.saveProduct(referenceTrack.frames[i], self._insar.referenceFrameParameter)
        self._insar.saveProduct(secondaryTrack.frames[i], self._insar.secondaryFrameParameter)
        os.chdir('../')

###############################################################################
    catalog.printToLog(logger, "runExtractBurst")
    self._insar.procDoc.addAllFromCatalog(catalog)

