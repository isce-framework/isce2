#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import glob
import logging
import datetime
import numpy as np

import isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml

logger = logging.getLogger('isce.alos2insar.runSwathMosaic')

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

        if not (
               ((self._insar.modeCombination == 21) or \
                (self._insar.modeCombination == 22) or \
                (self._insar.modeCombination == 31) or \
                (self._insar.modeCombination == 32)) 
               and
               (self._insar.endingSwath-self._insar.startingSwath+1 > 1)
               ):
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

    catalog.printToLog(logger, "runSwathMosaic")
    self._insar.procDoc.addAllFromCatalog(catalog)


def swathMosaic(frame, inputFiles, outputfile, rangeOffsets, azimuthOffsets, numberOfRangeLooks, numberOfAzimuthLooks, updateFrame=False, phaseCompensation=False, phaseDiff=None, phaseDiffFixed=None, snapThreshold=None, pcRangeLooks=1, pcAzimuthLooks=4, filt=False, resamplingMethod=0):
    '''
    mosaic swaths
    
    #PART 1. REGULAR INPUT PARAMTERS
    frame:                 frame
    inputFiles:            input file list
    outputfile:            output mosaic file
    rangeOffsets:          range offsets
    azimuthOffsets:        azimuth offsets
    numberOfRangeLooks:    number of range looks of the input files
    numberOfAzimuthLooks:  number of azimuth looks of the input files
    updateFrame:           whether update frame parameters

    #PART 2. PARAMETERS FOR COMPUTING PHASE DIFFERENCE BETWEEN SUBSWATHS
    phaseCompensation:     whether do phase compensation for each swath
    phaseDiff:             pre-computed compensation phase for each swath
    phaseDiffFixed:        if provided, the estimated value will snap to one of these values, which is nearest to the estimated one.
    snapThreshold:         this is used with phaseDiffFixed
    pcRangeLooks:          number of range looks to take when compute swath phase difference
    pcAzimuthLooks:        number of azimuth looks to take when compute swath phase difference
    filt:                  whether do filtering when compute swath phase difference

    #PART 3. RESAMPLING METHOD
    resamplingMethod:      0: amp resampling. 1: int resampling.
    '''
    from contrib.alos2proc_f.alos2proc_f import rect_with_looks
    from contrib.alos2proc.alos2proc import mosaicsubswath
    from isceobj.Alos2Proc.Alos2ProcPublic import multilook
    from isceobj.Alos2Proc.Alos2ProcPublic import cal_coherence_1
    from isceobj.Alos2Proc.Alos2ProcPublic import filterInterferogram

    numberOfSwaths = len(frame.swaths)
    swaths = frame.swaths

    rangeScale = []
    azimuthScale = []
    rectWidth = []
    rectLength = []
    for i in range(numberOfSwaths):
        rangeScale.append(swaths[0].rangePixelSize / swaths[i].rangePixelSize)
        azimuthScale.append(swaths[0].azimuthLineInterval / swaths[i].azimuthLineInterval)
        if i == 0:
            rectWidth.append( int(swaths[i].numberOfSamples / numberOfRangeLooks) )
            rectLength.append( int(swaths[i].numberOfLines / numberOfAzimuthLooks) )
        else:
            rectWidth.append( int(1.0 / rangeScale[i] * int(swaths[i].numberOfSamples / numberOfRangeLooks)) )
            rectLength.append( int(1.0 / azimuthScale[i] * int(swaths[i].numberOfLines / numberOfAzimuthLooks)) )

    #convert original offset to offset for images with looks
    #use list instead of np.array to make it consistent with the rest of the code
    rangeOffsets1 = [i/numberOfRangeLooks for i in rangeOffsets]
    azimuthOffsets1 = [i/numberOfAzimuthLooks for i in azimuthOffsets]

    #get offset relative to the first frame
    rangeOffsets2 = [0.0]
    azimuthOffsets2 = [0.0]
    for i in range(1, numberOfSwaths):
        rangeOffsets2.append(0.0)
        azimuthOffsets2.append(0.0)
        for j in range(1, i+1):
            rangeOffsets2[i] += rangeOffsets1[j]
            azimuthOffsets2[i] += azimuthOffsets1[j]

    #resample each swath
    rinfs = []
    for i, inf in enumerate(inputFiles):
        rinfs.append("{}_{}{}".format(os.path.splitext(os.path.basename(inf))[0], i, os.path.splitext(os.path.basename(inf))[1]))
        #do not resample first swath
        if i == 0:
            if os.path.isfile(rinfs[i]):
                os.remove(rinfs[i])
            os.symlink(inf, rinfs[i])
        else:
            infImg = isceobj.createImage()
            infImg.load(inf+'.xml')
            rangeOffsets2Frac = rangeOffsets2[i] - int(rangeOffsets2[i])
            azimuthOffsets2Frac = azimuthOffsets2[i] - int(azimuthOffsets2[i])


            if resamplingMethod == 0:
                rect_with_looks(inf,
                                rinfs[i],
                                infImg.width, infImg.length,
                                rectWidth[i], rectLength[i],
                                rangeScale[i], 0.0,
                                0.0,azimuthScale[i],
                                rangeOffsets2Frac * rangeScale[i], azimuthOffsets2Frac * azimuthScale[i],
                                1,1,
                                1,1,
                                'COMPLEX',
                                'Bilinear')
            elif resamplingMethod == 1:
                #decompose amplitude and phase
                phaseFile = 'phase'
                amplitudeFile = 'amplitude'
                data = np.fromfile(inf, dtype=np.complex64).reshape(infImg.length, infImg.width)
                phase = np.exp(np.complex64(1j) * np.angle(data))
                phase[np.nonzero(data==0)] = 0
                phase.astype(np.complex64).tofile(phaseFile)
                amplitude = np.absolute(data)
                amplitude.astype(np.float32).tofile(amplitudeFile)

                #resampling
                phaseRectFile = 'phaseRect'
                amplitudeRectFile = 'amplitudeRect'
                rect_with_looks(phaseFile,
                                phaseRectFile,
                                infImg.width, infImg.length,
                                rectWidth[i], rectLength[i],
                                rangeScale[i], 0.0,
                                0.0,azimuthScale[i],
                                rangeOffsets2Frac * rangeScale[i], azimuthOffsets2Frac * azimuthScale[i],
                                1,1,
                                1,1,
                                'COMPLEX',
                                'Sinc')
                rect_with_looks(amplitudeFile,
                                amplitudeRectFile,
                                infImg.width, infImg.length,
                                rectWidth[i], rectLength[i],
                                rangeScale[i], 0.0,
                                0.0,azimuthScale[i],
                                rangeOffsets2Frac * rangeScale[i], azimuthOffsets2Frac * azimuthScale[i],
                                1,1,
                                1,1,
                                'REAL',
                                'Bilinear')

                #recombine amplitude and phase
                phase = np.fromfile(phaseRectFile, dtype=np.complex64).reshape(rectLength[i], rectWidth[i])
                amplitude = np.fromfile(amplitudeRectFile, dtype=np.float32).reshape(rectLength[i], rectWidth[i])
                (phase*amplitude).astype(np.complex64).tofile(rinfs[i])

                #tidy up
                os.remove(phaseFile)
                os.remove(amplitudeFile)
                os.remove(phaseRectFile)
                os.remove(amplitudeRectFile)


    #determine output width and length
    #actually no need to calculate in range direction
    xs = []
    xe = []
    ys = []
    ye = []
    for i in range(numberOfSwaths):
        if i == 0:
            xs.append(0)
            xe.append(rectWidth[i] - 1)
            ys.append(0)
            ye.append(rectLength[i] - 1)
        else:
            xs.append(0 - int(rangeOffsets2[i]))
            xe.append(rectWidth[i] - 1 - int(rangeOffsets2[i]))
            ys.append(0 - int(azimuthOffsets2[i]))
            ye.append(rectLength[i] - 1 - int(azimuthOffsets2[i]))

    (xmin, xminIndex) = min((v,i) for i,v in enumerate(xs))
    (xmax, xmaxIndex) = max((v,i) for i,v in enumerate(xe))
    (ymin, yminIndex) = min((v,i) for i,v in enumerate(ys))
    (ymax, ymaxIndex) = max((v,i) for i,v in enumerate(ye))

    outWidth = xmax - xmin + 1
    outLength = ymax - ymin + 1

    #prepare offset for mosaicing
    rangeOffsets3 = []
    azimuthOffsets3 = []
    for i in range(numberOfSwaths):
        azimuthOffsets3.append(int(azimuthOffsets2[i]) - int(azimuthOffsets2[yminIndex]))
        if i != 0:
            rangeOffsets3.append(int(rangeOffsets2[i]) - int(rangeOffsets2[i-1]))
        else:
            rangeOffsets3.append(0)


    delta = int(30 / numberOfRangeLooks)

    #compute compensation phase for each swath
    diffMean2 = [0.0 for i in range(numberOfSwaths)]
    phaseDiffEst = [None for i in range(numberOfSwaths)]
    #True if:
    #  (1) used diff phase from input
    #  (2) used estimated diff phase after snapping to a fixed diff phase provided
    #False if:
    #  (1) used purely estimated diff phase
    phaseDiffSource  = ['estimated' for i in range(numberOfSwaths)]
    # 1. 'estimated': estimated from subswath overlap
    # 2. 'estimated+snap': estimated from subswath overlap and snap to a fixed value
    # 3. 'input': pre-computed
    # confidence level: 3 > 2 > 1
    if phaseCompensation:
        #compute swath phase offset
        diffMean = [0.0]
        for i in range(1, numberOfSwaths):

            #no need to estimate diff phase if provided from input
            #####################################################################
            if phaseDiff!=None:
                if phaseDiff[i]!=None:
                    diffMean.append(phaseDiff[i])
                    phaseDiffSource[i] = 'input'
                    print('using pre-computed phase offset given from input')
                    print('phase offset: subswath{} - subswath{}: {}'.format(frame.swaths[i-1].swathNumber, frame.swaths[i].swathNumber, phaseDiff[i]))
                    continue
            #####################################################################

            #all indexes start with zero, all the computed start/end sample/line indexes are included.
            
            #no need to add edge here, as we are going to find first/last nonzero sample/lines later
            #edge = delta
            edge = 0

            #image i-1
            startSample1 =  edge + 0 - int(rangeOffsets2[i]) + int(rangeOffsets2[i-1])
            endSample1   = -edge + rectWidth[i-1]-1
            startLine1   =  edge + max(0 - int(azimuthOffsets2[i]) + int(azimuthOffsets2[i-1]), 0)
            endLine1     = -edge + min(rectLength[i]-1 - int(azimuthOffsets2[i]) + int(azimuthOffsets2[i-1]), rectLength[i-1]-1)
            data1 = readImage(rinfs[i-1], rectWidth[i-1], rectLength[i-1], startSample1, endSample1, startLine1, endLine1)

            #image i
            startSample2 =  edge + 0
            endSample2   = -edge + rectWidth[i-1]-1 - int(rangeOffsets2[i-1]) + int(rangeOffsets2[i])
            startLine2   =  edge + max(0 - int(azimuthOffsets2[i-1]) + int(azimuthOffsets2[i]), 0)
            endLine2     = -edge + min(rectLength[i-1]-1 - int(azimuthOffsets2[i-1]) + int(azimuthOffsets2[i]), rectLength[i]-1)
            data2 = readImage(rinfs[i], rectWidth[i], rectLength[i], startSample2, endSample2, startLine2, endLine2)

            #remove edge due to incomplete covolution in resampling
            edge = 9
            (startLine0, endLine0, startSample0, endSample0) = findNonzero( np.logical_and((data1!=0), (data2!=0)) )
            data1 = data1[startLine0+edge:endLine0+1-edge, startSample0+edge:endSample0+1-edge]
            data2 = data2[startLine0+edge:endLine0+1-edge, startSample0+edge:endSample0+1-edge]

            #take looks
            data1 = multilook(data1, pcAzimuthLooks, pcRangeLooks)
            data2 = multilook(data2, pcAzimuthLooks, pcRangeLooks)

            #filter
            if filt:
                data1 /= (np.absolute(data1)+(data1==0))
                data2 /= (np.absolute(data2)+(data2==0))
                data1 = filterInterferogram(data1, 3.0, 64, 1)
                data2 = filterInterferogram(data2, 3.0, 64, 1)


            #get difference
            dataDiff = data1 * np.conj(data2)
            cor = cal_coherence_1(dataDiff, win=5)
            index = np.nonzero(np.logical_and(cor>0.85, dataDiff!=0))

            DEBUG=False
            if DEBUG:
                from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
                (length7, width7)=dataDiff.shape
                filename = 'diff_ori_s{}-s{}.int'.format(frame.swaths[i-1].swathNumber, frame.swaths[i].swathNumber)
                dataDiff.astype(np.complex64).tofile(filename)
                create_xml(filename, width7, length7, 'int')
                filename = 'cor_ori_s{}-s{}.cor'.format(frame.swaths[i-1].swathNumber, frame.swaths[i].swathNumber)
                cor.astype(np.float32).tofile(filename)
                create_xml(filename, width7, length7, 'float')

            print('\ncompute phase difference between subswaths {} and {}'.format(frame.swaths[i-1].swathNumber, frame.swaths[i].swathNumber))
            print('number of pixels with coherence > 0.85: {}'.format(index[0].size))

            #if already filtered the subswath overlap interferograms (MAI), do not filtered differential interferograms
            if (filt == False) and (index[0].size < 4000):
                #coherence too low, filter subswath overlap differential interferogram
                diffMean0 = 0.0
                breakFlag = False
                for (filterStrength, filterWinSize) in zip([3.0, 9.0], [64, 128]):
                    dataDiff = data1 * np.conj(data2)
                    dataDiff /= (np.absolute(dataDiff)+(dataDiff==0))
                    dataDiff = filterInterferogram(dataDiff, filterStrength, filterWinSize, 1)
                    cor = cal_coherence_1(dataDiff, win=7)

                    DEBUG=False
                    if DEBUG:
                        from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
                        (length7, width7)=dataDiff.shape
                        filename = 'diff_filt_s{}-s{}_strength_{}_winsize_{}.int'.format(frame.swaths[i-1].swathNumber, frame.swaths[i].swathNumber, filterStrength, filterWinSize)
                        dataDiff.astype(np.complex64).tofile(filename)
                        create_xml(filename, width7, length7, 'int')
                        filename = 'cor_filt_s{}-s{}_strength_{}_winsize_{}.cor'.format(frame.swaths[i-1].swathNumber, frame.swaths[i].swathNumber, filterStrength, filterWinSize)
                        cor.astype(np.float32).tofile(filename)
                        create_xml(filename, width7, length7, 'float')

                    for corth in [0.99999, 0.9999]:
                        index = np.nonzero(np.logical_and(cor>corth, dataDiff!=0))
                        if index[0].size > 30000:
                            breakFlag = True
                            break
                    if breakFlag:
                        break

                if index[0].size < 100:
                    diffMean0 = 0.0
                    print('\n\nWARNING: too few high coherence pixels for swath phase difference estimation')
                    print('         number of high coherence pixels: {}\n\n'.format(index[0].size))
                else:
                    print('filtered coherence threshold used: {}, number of pixels used: {}'.format(corth, index[0].size))
                    angle = np.mean(np.angle(dataDiff[index]), dtype=np.float64)
                    diffMean0 += angle
                    data2 *= np.exp(np.complex64(1j) * angle)
                    print('phase offset: %15.12f rad with filter strength: %f, window size: %3d'%(diffMean0, filterStrength, filterWinSize))
            else:
                diffMean0 = 0.0
                for k in range(30):
                    dataDiff = data1 * np.conj(data2)
                    cor = cal_coherence_1(dataDiff, win=5)
                    if filt:
                        index = np.nonzero(np.logical_and(cor>0.95, dataDiff!=0))
                    else:
                        index = np.nonzero(np.logical_and(cor>0.85, dataDiff!=0))
                    if index[0].size < 100:
                        diffMean0 = 0.0
                        print('\n\nWARNING: too few high coherence pixels for swath phase difference estimation')
                        print('         number of high coherence pixels: {}\n\n'.format(index[0].size))
                        break
                    angle = np.mean(np.angle(dataDiff[index]), dtype=np.float64)
                    diffMean0 += angle
                    data2 *= np.exp(np.complex64(1j) * angle)
                    print('phase offset: %15.12f rad after loop: %3d'%(diffMean0, k))

                    DEBUG=False
                    if DEBUG and (k==0):
                        from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
                        (length7, width7)=dataDiff.shape
                        filename = 'diff_ori_s{}-s{}_loop_{}.int'.format(frame.swaths[i-1].swathNumber, frame.swaths[i].swathNumber, k)
                        dataDiff.astype(np.complex64).tofile(filename)
                        create_xml(filename, width7, length7, 'int')
                        filename = 'cor_ori_s{}-s{}_loop_{}.cor'.format(frame.swaths[i-1].swathNumber, frame.swaths[i].swathNumber, k)
                        cor.astype(np.float32).tofile(filename)
                        create_xml(filename, width7, length7, 'float')


            #save purely estimated diff phase
            phaseDiffEst[i] = diffMean0
            
            #if fixed diff phase provided and the estimated diff phase is close enough to a fixed value, snap to it
            ############################################################################################################
            if phaseDiffFixed != None:
                phaseDiffTmp = np.absolute(np.absolute(np.array(phaseDiffFixed)) - np.absolute(diffMean0))
                phaseDiffTmpMinIndex = np.argmin(phaseDiffTmp)
                if phaseDiffTmp[phaseDiffTmpMinIndex] < snapThreshold:
                   diffMean0 = np.sign(diffMean0) * np.absolute(phaseDiffFixed[phaseDiffTmpMinIndex])
                   phaseDiffSource[i] = 'estimated+snap'
            ############################################################################################################

            diffMean.append(diffMean0)
            print('phase offset: subswath{} - subswath{}: {}'.format(frame.swaths[i-1].swathNumber, frame.swaths[i].swathNumber, diffMean0))

        for i in range(1, numberOfSwaths):
            for j in range(1, i+1):
                diffMean2[i] += diffMean[j]


    #mosaic swaths
    diffflag = 1
    oflag = [0 for i in range(numberOfSwaths)]
    mosaicsubswath(outputfile, outWidth, outLength, delta, diffflag, numberOfSwaths, 
        rinfs, rectWidth, rangeOffsets3, azimuthOffsets3, diffMean2, oflag)
    #remove tmp files
    for x in rinfs:
        os.remove(x)


    #update frame parameters
    if updateFrame:
        #mosaic size
        frame.numberOfSamples = outWidth
        frame.numberOfLines = outLength
        #NOTE THAT WE ARE STILL USING SINGLE LOOK PARAMETERS HERE
        #range parameters
        frame.startingRange = frame.swaths[0].startingRange
        frame.rangeSamplingRate = frame.swaths[0].rangeSamplingRate
        frame.rangePixelSize = frame.swaths[0].rangePixelSize
        #azimuth parameters
        azimuthTimeOffset = - max([int(x) for x in azimuthOffsets2]) * numberOfAzimuthLooks * frame.swaths[0].azimuthLineInterval
        frame.sensingStart = frame.swaths[0].sensingStart + datetime.timedelta(seconds = azimuthTimeOffset)
        frame.prf = frame.swaths[0].prf
        frame.azimuthPixelSize = frame.swaths[0].azimuthPixelSize
        frame.azimuthLineInterval = frame.swaths[0].azimuthLineInterval


    if phaseCompensation:
        # estimated phase diff, used phase diff, used phase diff source
        return (phaseDiffEst, diffMean, phaseDiffSource)

def swathMosaicParameters(frame, rangeOffsets, azimuthOffsets, numberOfRangeLooks, numberOfAzimuthLooks):
    '''
    mosaic swaths (this is simplified version of swathMosaic to update parameters only)
    
    frame:                 frame
    rangeOffsets:          range offsets
    azimuthOffsets:        azimuth offsets
    numberOfRangeLooks:    number of range looks of the input files
    numberOfAzimuthLooks:  number of azimuth looks of the input files
    '''

    numberOfSwaths = len(frame.swaths)
    swaths = frame.swaths

    rangeScale = []
    azimuthScale = []
    rectWidth = []
    rectLength = []
    for i in range(numberOfSwaths):
        rangeScale.append(swaths[0].rangePixelSize / swaths[i].rangePixelSize)
        azimuthScale.append(swaths[0].azimuthLineInterval / swaths[i].azimuthLineInterval)
        if i == 0:
            rectWidth.append( int(swaths[i].numberOfSamples / numberOfRangeLooks) )
            rectLength.append( int(swaths[i].numberOfLines / numberOfAzimuthLooks) )
        else:
            rectWidth.append( int(1.0 / rangeScale[i] * int(swaths[i].numberOfSamples / numberOfRangeLooks)) )
            rectLength.append( int(1.0 / azimuthScale[i] * int(swaths[i].numberOfLines / numberOfAzimuthLooks)) )

    #convert original offset to offset for images with looks
    #use list instead of np.array to make it consistent with the rest of the code
    rangeOffsets1 = [i/numberOfRangeLooks for i in rangeOffsets]
    azimuthOffsets1 = [i/numberOfAzimuthLooks for i in azimuthOffsets]

    #get offset relative to the first frame
    rangeOffsets2 = [0.0]
    azimuthOffsets2 = [0.0]
    for i in range(1, numberOfSwaths):
        rangeOffsets2.append(0.0)
        azimuthOffsets2.append(0.0)
        for j in range(1, i+1):
            rangeOffsets2[i] += rangeOffsets1[j]
            azimuthOffsets2[i] += azimuthOffsets1[j]

    #determine output width and length
    #actually no need to calculate in range direction
    xs = []
    xe = []
    ys = []
    ye = []
    for i in range(numberOfSwaths):
        if i == 0:
            xs.append(0)
            xe.append(rectWidth[i] - 1)
            ys.append(0)
            ye.append(rectLength[i] - 1)
        else:
            xs.append(0 - int(rangeOffsets2[i]))
            xe.append(rectWidth[i] - 1 - int(rangeOffsets2[i]))
            ys.append(0 - int(azimuthOffsets2[i]))
            ye.append(rectLength[i] - 1 - int(azimuthOffsets2[i]))

    (xmin, xminIndex) = min((v,i) for i,v in enumerate(xs))
    (xmax, xmaxIndex) = max((v,i) for i,v in enumerate(xe))
    (ymin, yminIndex) = min((v,i) for i,v in enumerate(ys))
    (ymax, ymaxIndex) = max((v,i) for i,v in enumerate(ye))

    outWidth = xmax - xmin + 1
    outLength = ymax - ymin + 1

    #update frame parameters
    #mosaic size
    frame.numberOfSamples = outWidth
    frame.numberOfLines = outLength
    #NOTE THAT WE ARE STILL USING SINGLE LOOK PARAMETERS HERE
    #range parameters
    frame.startingRange = frame.swaths[0].startingRange
    frame.rangeSamplingRate = frame.swaths[0].rangeSamplingRate
    frame.rangePixelSize = frame.swaths[0].rangePixelSize
    #azimuth parameters
    azimuthTimeOffset = - max([int(x) for x in azimuthOffsets2]) * numberOfAzimuthLooks * frame.swaths[0].azimuthLineInterval
    frame.sensingStart = frame.swaths[0].sensingStart + datetime.timedelta(seconds = azimuthTimeOffset)
    frame.prf = frame.swaths[0].prf
    frame.azimuthPixelSize = frame.swaths[0].azimuthPixelSize
    frame.azimuthLineInterval = frame.swaths[0].azimuthLineInterval


def readImage(inputfile, numberOfSamples, numberOfLines, startSample, endSample, startLine, endLine):
    '''
    read a chunk of image
    the indexes (startSample, endSample, startLine, endLine) are included and start with zero

    memmap is not used, because it is much slower
    '''
    data = np.zeros((endLine-startLine+1, endSample-startSample+1), dtype=np.complex64)
    with open(inputfile,'rb') as fp:
        #for i in range(endLine-startLine+1):
        for i in range(startLine, endLine+1):
            fp.seek((i*numberOfSamples+startSample)*8, 0)
            data[i-startLine] = np.fromfile(fp, dtype=np.complex64, count=endSample-startSample+1)
    return data


def findNonzero_v1(data):
    '''
    find the first/last non-zero line/sample
    all indexes start from zero
    '''
    indexes = np.nonzero(data)

           #first line     last line       first sample    last sample
    return (indexes[0][0], indexes[0][-1], indexes[1][0], indexes[1][-1])


def findNonzero(data, lineRatio=0.5, sampleRatio=0.5):
    '''
    find the first/last non-zero line/sample
    all indexes start from zero
    '''
    import numpy as np
    
    (length, width)=data.shape

    lineIndex = (np.nonzero(np.sum((data!=0), axis=1) > width*lineRatio))[0]
    sampleIndex = (np.nonzero(np.sum((data!=0), axis=0) > length*sampleRatio))[0]

           #first line     last line       first sample    last sample
    return (lineIndex[0], lineIndex[-1], sampleIndex[0], sampleIndex[-1])


