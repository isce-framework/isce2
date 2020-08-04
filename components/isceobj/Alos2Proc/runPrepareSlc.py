#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging
import datetime
import numpy as np

import isceobj
from isceobj.Constants import SPEED_OF_LIGHT
from isceobj.Alos2Proc.Alos2ProcPublic import overlapFrequency
from contrib.alos2proc.alos2proc import rg_filter
from contrib.alos2proc.alos2proc import resamp
from contrib.alos2proc.alos2proc import mbf

logger = logging.getLogger('isce.alos2insar.runPrepareSlc')

def runPrepareSlc(self):
    '''Extract images.
    '''
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    referenceTrack = self._insar.loadTrack(reference=True)
    secondaryTrack = self._insar.loadTrack(reference=False)


    ####################################################
    #1. crop slc
    ####################################################
    #for ScanSAR-stripmap interferometry, we always crop slcs
    #for other cases, up to users
    if ((self._insar.modeCombination == 31) or (self._insar.modeCombination == 32)) or (self.cropSlc):
        for i, frameNumber in enumerate(self._insar.referenceFrames):
            frameDir = 'f{}_{}'.format(i+1, frameNumber)
            os.chdir(frameDir)
            for j, swathNumber in enumerate(range(self._insar.startingSwath, self._insar.endingSwath + 1)):
                swathDir = 's{}'.format(swathNumber)
                os.chdir(swathDir)

                print('cropping frame {}, swath {}'.format(frameNumber, swathNumber))

                referenceSwath = referenceTrack.frames[i].swaths[j]
                secondarySwath = secondaryTrack.frames[i].swaths[j]
                
                #crop reference
                cropSlc(referenceTrack.orbit, referenceSwath, self._insar.referenceSlc, secondaryTrack.orbit, secondarySwath, edge=0, useVirtualFile=self.useVirtualFile)
                #crop secondary, since secondary may go through resampling, we set edge=9
                #cropSlc(secondaryTrack.orbit, secondarySwath, self._insar.secondarySlc, referenceTrack.orbit, referenceSwath, edge=9, useVirtualFile=self.useVirtualFile)
                cropSlc(secondaryTrack.orbit, secondarySwath, self._insar.secondarySlc, referenceTrack.orbit, referenceSwath, edge=0, useVirtualFile=self.useVirtualFile)

                os.chdir('../')
            os.chdir('../')


    ####################################################
    #2. range-filter slc
    ####################################################
    #compute filtering parameters, radarwavelength and range bandwidth should be the same across all swaths and frames
    centerfreq1 = SPEED_OF_LIGHT / referenceTrack.radarWavelength
    bandwidth1 = referenceTrack.frames[0].swaths[0].rangeBandwidth
    centerfreq2 = SPEED_OF_LIGHT / secondaryTrack.radarWavelength
    bandwidth2 = secondaryTrack.frames[0].swaths[0].rangeBandwidth
    overlapfreq = overlapFrequency(centerfreq1, bandwidth1, centerfreq2, bandwidth2)

    if overlapfreq == None:
        raise Exception('there is no overlap bandwidth in range')
    overlapbandwidth = overlapfreq[1] - overlapfreq[0]
    if overlapbandwidth < 3e6:
        print('overlap bandwidth: {}, percentage: {}%'.format(overlapbandwidth, 100.0*overlapbandwidth/bandwidth1))
        raise Exception('there is not enough overlap bandwidth in range')
    centerfreq = (overlapfreq[1] + overlapfreq[0]) / 2.0

    for i, frameNumber in enumerate(self._insar.referenceFrames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        os.chdir(frameDir)
        for j, swathNumber in enumerate(range(self._insar.startingSwath, self._insar.endingSwath + 1)):
            swathDir = 's{}'.format(swathNumber)
            os.chdir(swathDir)

            print('range filtering frame {}, swath {}'.format(frameNumber, swathNumber))

            referenceSwath = referenceTrack.frames[i].swaths[j]
            secondarySwath = secondaryTrack.frames[i].swaths[j]

            # #compute filtering parameters
            # centerfreq1 = SPEED_OF_LIGHT / referenceTrack.radarWavelength
            # bandwidth1 = referenceSwath.rangeBandwidth
            # centerfreq2 = SPEED_OF_LIGHT / secondaryTrack.radarWavelength
            # bandwidth2 = secondarySwath.rangeBandwidth
            # overlapfreq = overlapFrequency(centerfreq1, bandwidth1, centerfreq2, bandwidth2)

            # if overlapfreq == None:
            #     raise Exception('there is no overlap bandwidth in range')
            # overlapbandwidth = overlapfreq[1] - overlapfreq[0]
            # if overlapbandwidth < 3e6:
            #     print('overlap bandwidth: {}, percentage: {}%'.format(overlapbandwidth, 100.0*overlapbandwidth/bandwidth1))
            #     raise Exception('there is not enough overlap bandwidth in range')
            # centerfreq = (overlapfreq[1] + overlapfreq[0]) / 2.0

            #filter reference
            if abs(centerfreq1 - centerfreq) < 1.0 and (bandwidth1 - 1.0) < overlapbandwidth:
                print('no need to range filter {}'.format(self._insar.referenceSlc))
            else:
                print('range filter {}'.format(self._insar.referenceSlc))
                tmpSlc = 'tmp.slc'
                rg_filter(self._insar.referenceSlc, 1, [tmpSlc], [overlapbandwidth / referenceSwath.rangeSamplingRate], 
                    [(centerfreq - centerfreq1) / referenceSwath.rangeSamplingRate], 
                    257, 2048, 0.1, 0, 0.0)

                if os.path.isfile(self._insar.referenceSlc):
                    os.remove(self._insar.referenceSlc)
                os.remove(self._insar.referenceSlc+'.vrt')
                os.remove(self._insar.referenceSlc+'.xml')

                img = isceobj.createSlcImage()
                img.load(tmpSlc + '.xml')
                #remove original
                os.remove(tmpSlc + '.vrt')
                os.remove(tmpSlc + '.xml')
                os.rename(tmpSlc, self._insar.referenceSlc)
                #creat new
                img.setFilename(self._insar.referenceSlc)
                img.extraFilename = self._insar.referenceSlc + '.vrt'
                img.setAccessMode('READ')
                img.renderHdr()

                referenceTrack.radarWavelength = SPEED_OF_LIGHT/centerfreq
                referenceSwath.rangeBandwidth = overlapbandwidth

            #filter secondary
            if abs(centerfreq2 - centerfreq) < 1.0 and (bandwidth2 - 1.0) < overlapbandwidth:
                print('no need to range filter {}'.format(self._insar.secondarySlc))
            else:
                print('range filter {}'.format(self._insar.secondarySlc))
                tmpSlc = 'tmp.slc'
                rg_filter(self._insar.secondarySlc, 1, [tmpSlc], [overlapbandwidth / secondarySwath.rangeSamplingRate], 
                    [(centerfreq - centerfreq2) / secondarySwath.rangeSamplingRate], 
                    257, 2048, 0.1, 0, 0.0)

                if os.path.isfile(self._insar.secondarySlc):
                    os.remove(self._insar.secondarySlc)
                os.remove(self._insar.secondarySlc+'.vrt')
                os.remove(self._insar.secondarySlc+'.xml')

                img = isceobj.createSlcImage()
                img.load(tmpSlc + '.xml')
                #remove original
                os.remove(tmpSlc + '.vrt')
                os.remove(tmpSlc + '.xml')
                os.rename(tmpSlc, self._insar.secondarySlc)
                #creat new
                img.setFilename(self._insar.secondarySlc)
                img.extraFilename = self._insar.secondarySlc + '.vrt'
                img.setAccessMode('READ')
                img.renderHdr()

                secondaryTrack.radarWavelength = SPEED_OF_LIGHT/centerfreq
                secondarySwath.rangeBandwidth = overlapbandwidth

            os.chdir('../')
        os.chdir('../')

    
    ####################################################
    #3. equalize sample size
    ####################################################
    for i, frameNumber in enumerate(self._insar.referenceFrames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        os.chdir(frameDir)
        for j, swathNumber in enumerate(range(self._insar.startingSwath, self._insar.endingSwath + 1)):
            swathDir = 's{}'.format(swathNumber)
            os.chdir(swathDir)

            print('equalize sample size frame {}, swath {}'.format(frameNumber, swathNumber))

            referenceSwath = referenceTrack.frames[i].swaths[j]
            secondarySwath = secondaryTrack.frames[i].swaths[j]

            if abs(referenceSwath.rangeSamplingRate - secondarySwath.rangeSamplingRate) < 1.0 and abs(referenceSwath.prf - secondarySwath.prf) < 1.0:
                print('no need to resample {}.'.format(self._insar.secondarySlc))
            else:
                outWidth  = round(secondarySwath.numberOfSamples / secondarySwath.rangeSamplingRate * referenceSwath.rangeSamplingRate)
                outLength = round(secondarySwath.numberOfLines / secondarySwath.prf * referenceSwath.prf)
                
                tmpSlc = 'tmp.slc'
                resamp(self._insar.secondarySlc, tmpSlc, 'fake', 'fake', outWidth, outLength, secondarySwath.prf, secondarySwath.dopplerVsPixel, 
                    rgcoef=[0.0, (1.0/referenceSwath.rangeSamplingRate) / (1.0/secondarySwath.rangeSamplingRate) - 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                    azcoef=[0.0, 0.0, (1.0/referenceSwath.prf) / (1.0/secondarySwath.prf) - 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                    azpos_off=0.0)

                if os.path.isfile(self._insar.secondarySlc):
                    os.remove(self._insar.secondarySlc)
                os.remove(self._insar.secondarySlc+'.vrt')
                os.remove(self._insar.secondarySlc+'.xml')

                img = isceobj.createSlcImage()
                img.load(tmpSlc + '.xml')
                #remove original
                os.remove(tmpSlc + '.vrt')
                os.remove(tmpSlc + '.xml')
                os.rename(tmpSlc, self._insar.secondarySlc)
                #creat new
                img.setFilename(self._insar.secondarySlc)
                img.extraFilename = self._insar.secondarySlc + '.vrt'
                img.setAccessMode('READ')
                img.renderHdr()

                #update parameters
                #update doppler and azfmrate first
                index2  = np.arange(outWidth)
                index = np.arange(outWidth) * (1.0/referenceSwath.rangeSamplingRate) / (1.0/secondarySwath.rangeSamplingRate)
                dop = np.polyval(secondarySwath.dopplerVsPixel[::-1], index)
                p = np.polyfit(index2, dop, 3)
                secondarySwath.dopplerVsPixel = [p[3], p[2], p[1], p[0]]

                azfmrate = np.polyval(secondarySwath.azimuthFmrateVsPixel[::-1], index)
                p = np.polyfit(index2, azfmrate, 3)
                secondarySwath.azimuthFmrateVsPixel = [p[3], p[2], p[1], p[0]]

                secondarySwath.numberOfSamples = outWidth
                secondarySwath.numberOfLines = outLength

                secondarySwath.prf = referenceSwath.prf
                secondarySwath.rangeSamplingRate = referenceSwath.rangeSamplingRate
                secondarySwath.rangePixelSize = referenceSwath.rangePixelSize
                secondarySwath.azimuthPixelSize = referenceSwath.azimuthPixelSize
                secondarySwath.azimuthLineInterval = referenceSwath.azimuthLineInterval
                secondarySwath.prfFraction = referenceSwath.prfFraction

            os.chdir('../')
        os.chdir('../')


    ####################################################
    #4. mbf
    ####################################################
    for i, frameNumber in enumerate(self._insar.referenceFrames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        os.chdir(frameDir)
        for j, swathNumber in enumerate(range(self._insar.startingSwath, self._insar.endingSwath + 1)):
            swathDir = 's{}'.format(swathNumber)
            os.chdir(swathDir)

            print('azimuth filter frame {}, swath {}'.format(frameNumber, swathNumber))

            referenceSwath = referenceTrack.frames[i].swaths[j]
            secondarySwath = secondaryTrack.frames[i].swaths[j]

            #using Piyush's code for computing range and azimuth offsets
            midRange = referenceSwath.startingRange + referenceSwath.rangePixelSize * referenceSwath.numberOfSamples * 0.5
            midSensingStart = referenceSwath.sensingStart + datetime.timedelta(seconds = referenceSwath.numberOfLines * 0.5 / referenceSwath.prf)
            llh = referenceTrack.orbit.rdr2geo(midSensingStart, midRange)
            slvaz, slvrng = secondaryTrack.orbit.geo2rdr(llh)
            ###Translate to offsets
            #at this point, secondary range pixel size and prf should be the same as those of reference
            rgoff = ((slvrng - secondarySwath.startingRange) / referenceSwath.rangePixelSize) - referenceSwath.numberOfSamples * 0.5
            azoff = ((slvaz - secondarySwath.sensingStart).total_seconds() * referenceSwath.prf) - referenceSwath.numberOfLines * 0.5

            #filter reference
            if not ((self._insar.modeCombination == 21) and (self._insar.burstSynchronization <= self.burstSynchronizationThreshold)):
                print('no need to azimuth filter {}.'.format(self._insar.referenceSlc))
            else:
                index = np.arange(referenceSwath.numberOfSamples) + rgoff
                dop = np.polyval(secondarySwath.dopplerVsPixel[::-1], index)
                p = np.polyfit(index-rgoff, dop, 3)
                dopplerVsPixelSecondary = [p[3], p[2], p[1], p[0]]

                tmpSlc = 'tmp.slc'
                mbf(self._insar.referenceSlc, tmpSlc, referenceSwath.prf, 1.0, 
                    referenceSwath.burstLength, referenceSwath.burstCycleLength-referenceSwath.burstLength, 
                    self._insar.burstUnsynchronizedTime * referenceSwath.prf, 
                    (referenceSwath.burstStartTime - referenceSwath.sensingStart).total_seconds() * referenceSwath.prf, 
                    referenceSwath.azimuthFmrateVsPixel, referenceSwath.dopplerVsPixel, dopplerVsPixelSecondary)

                if os.path.isfile(self._insar.referenceSlc):
                    os.remove(self._insar.referenceSlc)
                os.remove(self._insar.referenceSlc+'.vrt')
                os.remove(self._insar.referenceSlc+'.xml')

                img = isceobj.createSlcImage()
                img.load(tmpSlc + '.xml')
                #remove original
                os.remove(tmpSlc + '.vrt')
                os.remove(tmpSlc + '.xml')
                os.rename(tmpSlc, self._insar.referenceSlc)
                #creat new
                img.setFilename(self._insar.referenceSlc)
                img.extraFilename = self._insar.referenceSlc + '.vrt'
                img.setAccessMode('READ')
                img.renderHdr()

            #filter secondary
            if not(
                ((self._insar.modeCombination == 21) and (self._insar.burstSynchronization <= self.burstSynchronizationThreshold)) or \
                (self._insar.modeCombination == 31)
                ):
                print('no need to azimuth filter {}.'.format(self._insar.secondarySlc))
            else:
                index = np.arange(secondarySwath.numberOfSamples) - rgoff
                dop = np.polyval(referenceSwath.dopplerVsPixel[::-1], index)
                p = np.polyfit(index+rgoff, dop, 3)
                dopplerVsPixelReference = [p[3], p[2], p[1], p[0]]

                tmpSlc = 'tmp.slc'
                mbf(self._insar.secondarySlc, tmpSlc, secondarySwath.prf, 1.0, 
                    secondarySwath.burstLength, secondarySwath.burstCycleLength-secondarySwath.burstLength, 
                    -self._insar.burstUnsynchronizedTime * secondarySwath.prf, 
                    (secondarySwath.burstStartTime - secondarySwath.sensingStart).total_seconds() * secondarySwath.prf, 
                    secondarySwath.azimuthFmrateVsPixel, secondarySwath.dopplerVsPixel, dopplerVsPixelReference)

                if os.path.isfile(self._insar.secondarySlc):
                    os.remove(self._insar.secondarySlc)
                os.remove(self._insar.secondarySlc+'.vrt')
                os.remove(self._insar.secondarySlc+'.xml')

                img = isceobj.createSlcImage()
                img.load(tmpSlc + '.xml')
                #remove original
                os.remove(tmpSlc + '.vrt')
                os.remove(tmpSlc + '.xml')
                os.rename(tmpSlc, self._insar.secondarySlc)
                #creat new
                img.setFilename(self._insar.secondarySlc)
                img.extraFilename = self._insar.secondarySlc + '.vrt'
                img.setAccessMode('READ')
                img.renderHdr()

            os.chdir('../')
        os.chdir('../')

    #in case parameters changed
    self._insar.saveTrack(referenceTrack, reference=True)
    self._insar.saveTrack(secondaryTrack, reference=False)

    catalog.printToLog(logger, "runPrepareSlc")
    self._insar.procDoc.addAllFromCatalog(catalog)


def cropSlc(orbit, swath, slc, orbit2, swath2, edge=0, useVirtualFile=True):
    from isceobj.Alos2Proc.Alos2ProcPublic import find_vrt_keyword
    from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
    '''
    orbit:  orbit of the image to be cropped
    swath:  swath of the image to be cropped
    slc:    image to be cropped
    orbit2: orbit of the other image
    swath2: swath of the other image
    '''

    #find topleft and lowerright corners
    #all indices start with 0
    corner = []
    for x in [[0, 0], [swath2.numberOfLines -1, swath2.numberOfSamples-1]]:
        line2 = x[0]
        sample2 = x[1]
        rg2 = swath2.startingRange + swath2.rangePixelSize * sample2
        az2 = swath2.sensingStart + datetime.timedelta(seconds = line2 / swath2.prf)
        llh2 = orbit2.rdr2geo(az2, rg2)
        az, rg = orbit.geo2rdr(llh2)
        line = (az - swath.sensingStart).total_seconds() * swath.prf
        sample = (rg - swath.startingRange) / swath.rangePixelSize
        corner.append([line, sample])

    #image (to be cropped) bounds
    firstLine   = 0
    lastLine    = swath.numberOfLines-1
    firstSample = 0
    lastSample  = swath.numberOfSamples-1

    #the othe image bounds in image (to be cropped)
    #add edge
    #edge = 9
    firstLine2   = int(corner[0][0] - edge)
    lastLine2    = int(corner[1][0] + edge)
    firstSample2 = int(corner[0][1] - edge)
    lastSample2  = int(corner[1][1] + edge)

    #image (to be cropped) output bounds
    firstLine3   = max(firstLine, firstLine2)
    lastLine3    = min(lastLine, lastLine2)
    firstSample3 = max(firstSample, firstSample2)
    lastSample3  = min(lastSample, lastSample2)
    numberOfSamples3 = lastSample3-firstSample3+1
    numberOfLines3 = lastLine3-firstLine3+1

    #check if there is overlap
    if lastLine3 - firstLine3 +1 < 1000:
        raise Exception('azimuth overlap < 1000 lines, not enough area for InSAR\n')
    if lastSample3 - firstSample3 +1 < 1000:
        raise Exception('range overlap < 1000 samples, not enough area for InSAR\n')

    #check if there is a need to crop image
    if abs(firstLine3-firstLine) < 100 and abs(lastLine3-lastLine) < 100 and \
       abs(firstSample3-firstSample) < 100 and abs(lastSample3-lastSample) < 100:
        print('no need to crop {}. nothing is done by crop.'.format(slc))
        return

    #crop image
    if useVirtualFile:
        #vrt
        SourceFilename = find_vrt_keyword(slc+'.vrt', 'SourceFilename')
        ImageOffset    = int(find_vrt_keyword(slc+'.vrt', 'ImageOffset'))
        PixelOffset    = int(find_vrt_keyword(slc+'.vrt', 'PixelOffset'))
        LineOffset     = int(find_vrt_keyword(slc+'.vrt', 'LineOffset'))

        #overwrite vrt and xml
        img = isceobj.createImage()
        img.load(slc+'.xml')
        img.width = numberOfSamples3
        img.length = numberOfLines3
        img.renderHdr()

        #overrite vrt
        with open(slc+'.vrt', 'w') as fid:
                fid.write('''<VRTDataset rasterXSize="{0}" rasterYSize="{1}">
    <VRTRasterBand band="1" dataType="CFloat32" subClass="VRTRawRasterBand">
        <SourceFilename relativeToVRT="0">{2}</SourceFilename>
        <ByteOrder>MSB</ByteOrder>
        <ImageOffset>{3}</ImageOffset>
        <PixelOffset>8</PixelOffset>
        <LineOffset>{4}</LineOffset>
    </VRTRasterBand>
</VRTDataset>'''.format(numberOfSamples3, 
                        numberOfLines3,
                       SourceFilename,
                       ImageOffset + firstLine3*LineOffset + firstSample3*8,
                       LineOffset))
    else:
        #read and crop data
        with open(slc, 'rb') as f:
            f.seek(firstLine3 * swath.numberOfSamples * np.dtype(np.complex64).itemsize, 0)
            data = np.fromfile(f, dtype=np.complex64, count=numberOfLines3 * swath.numberOfSamples)\
                               .reshape(numberOfLines3,swath.numberOfSamples)
            data2 = data[:, firstSample3:lastSample3+1]
        #overwrite original
        data2.astype(np.complex64).tofile(slc)
        
        #creat new vrt and xml
        os.remove(slc + '.xml')
        os.remove(slc + '.vrt')
        create_xml(slc, numberOfSamples3, numberOfLines3, 'slc')

    #update parameters
    #update doppler and azfmrate first
    dop = np.polyval(swath.dopplerVsPixel[::-1], np.arange(swath.numberOfSamples))
    dop3 = dop[firstSample3:lastSample3+1]
    p = np.polyfit(np.arange(numberOfSamples3), dop3, 3)
    swath.dopplerVsPixel = [p[3], p[2], p[1], p[0]]

    azfmrate = np.polyval(swath.azimuthFmrateVsPixel[::-1], np.arange(swath.numberOfSamples))
    azfmrate3 = azfmrate[firstSample3:lastSample3+1]
    p = np.polyfit(np.arange(numberOfSamples3), azfmrate3, 3)
    swath.azimuthFmrateVsPixel = [p[3], p[2], p[1], p[0]]

    swath.numberOfSamples = numberOfSamples3
    swath.numberOfLines = numberOfLines3

    swath.startingRange += firstSample3 * swath.rangePixelSize
    swath.sensingStart  += datetime.timedelta(seconds = firstLine3 / swath.prf)

    #no need to update frame and track, as parameters requiring changes are determined
    #in swath and frame mosaicking, which is not yet done at this point.

