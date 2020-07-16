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

logger = logging.getLogger('isce.alos2insar.runFrameMosaic')

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
        frameMosaic(referenceTrack, inputInterferograms, self._insar.interferogram, 
            rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, 
            updateTrack=True, phaseCompensation=True, resamplingMethod=1)

        create_xml(self._insar.amplitude, referenceTrack.numberOfSamples, referenceTrack.numberOfLines, 'amp')
        create_xml(self._insar.interferogram, referenceTrack.numberOfSamples, referenceTrack.numberOfLines, 'int')

        #update secondary parameters here
        #do not match for secondary, always use geometrical
        rangeOffsets = self._insar.frameRangeOffsetGeometricalSecondary
        azimuthOffsets = self._insar.frameAzimuthOffsetGeometricalSecondary
        frameMosaicParameters(secondaryTrack, rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1)

    os.chdir('../')
    #save parameter file
    self._insar.saveProduct(referenceTrack, self._insar.referenceTrackParameter)
    self._insar.saveProduct(secondaryTrack, self._insar.secondaryTrackParameter)

    catalog.printToLog(logger, "runFrameMosaic")
    self._insar.procDoc.addAllFromCatalog(catalog)


def frameMosaic(track, inputFiles, outputfile, rangeOffsets, azimuthOffsets, numberOfRangeLooks, numberOfAzimuthLooks, updateTrack=False, phaseCompensation=False, resamplingMethod=0):
    '''
    mosaic frames
    
    track:                 track
    inputFiles:            input file list
    output file:           output mosaic file
    rangeOffsets:          range offsets
    azimuthOffsets:        azimuth offsets
    numberOfRangeLooks:    number of range looks of the input files
    numberOfAzimuthLooks:  number of azimuth looks of the input files
    updateTrack:           whether update track parameters
    phaseCompensation:     whether do phase compensation for each frame
    resamplingMethod:      0: amp resampling. 1: int resampling. 2: slc resampling
    '''
    import numpy as np

    from contrib.alos2proc_f.alos2proc_f import rect_with_looks
    from contrib.alos2proc.alos2proc import resamp
    from isceobj.Alos2Proc.runSwathMosaic import readImage
    from isceobj.Alos2Proc.runSwathMosaic import findNonzero
    from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
    from isceobj.Alos2Proc.Alos2ProcPublic import find_vrt_file
    from isceobj.Alos2Proc.Alos2ProcPublic import find_vrt_keyword

    numberOfFrames = len(track.frames)
    frames = track.frames

    rectWidth = []
    rectLength = []
    for i in range(numberOfFrames):
        infImg = isceobj.createImage()
        infImg.load(inputFiles[i]+'.xml')
        rectWidth.append(infImg.width)
        rectLength.append(infImg.length)

    #convert original offset to offset for images with looks
    #use list instead of np.array to make it consistent with the rest of the code
    rangeOffsets1 = [i/numberOfRangeLooks for i in rangeOffsets]
    azimuthOffsets1 = [i/numberOfAzimuthLooks for i in azimuthOffsets]

    #get offset relative to the first frame
    rangeOffsets2 = [0.0]
    azimuthOffsets2 = [0.0]
    for i in range(1, numberOfFrames):
        rangeOffsets2.append(0.0)
        azimuthOffsets2.append(0.0)
        for j in range(1, i+1):
            rangeOffsets2[i] += rangeOffsets1[j]
            azimuthOffsets2[i] += azimuthOffsets1[j]

    #resample each frame
    rinfs = []
    for i, inf in enumerate(inputFiles):
        rinfs.append("{}_{}{}".format(os.path.splitext(os.path.basename(inf))[0], i, os.path.splitext(os.path.basename(inf))[1]))
        #do not resample first frame
        if i == 0:
            rinfs[i] = inf
        else:
            infImg = isceobj.createImage()
            infImg.load(inf+'.xml')
            rangeOffsets2Frac = rangeOffsets2[i] - int(rangeOffsets2[i])
            azimuthOffsets2Frac = azimuthOffsets2[i] - int(azimuthOffsets2[i])

            if resamplingMethod == 0:
                rect_with_looks(inf,
                                rinfs[i],
                                infImg.width, infImg.length,
                                infImg.width, infImg.length,
                                1.0, 0.0,
                                0.0, 1.0,
                                rangeOffsets2Frac, azimuthOffsets2Frac,
                                1,1,
                                1,1,
                                'COMPLEX',
                                'Bilinear')
                if infImg.getImageType() == 'amp':
                    create_xml(rinfs[i], infImg.width, infImg.length, 'amp')
                else:
                    create_xml(rinfs[i], infImg.width, infImg.length, 'int')

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
                                infImg.width, infImg.length,
                                1.0, 0.0,
                                0.0, 1.0,
                                rangeOffsets2Frac, azimuthOffsets2Frac,
                                1,1,
                                1,1,
                                'COMPLEX',
                                'Sinc')
                rect_with_looks(amplitudeFile,
                                amplitudeRectFile,
                                infImg.width, infImg.length,
                                infImg.width, infImg.length,
                                1.0, 0.0,
                                0.0, 1.0,
                                rangeOffsets2Frac, azimuthOffsets2Frac,
                                1,1,
                                1,1,
                                'REAL',
                                'Bilinear')

                #recombine amplitude and phase
                phase = np.fromfile(phaseRectFile, dtype=np.complex64).reshape(infImg.length, infImg.width)
                amplitude = np.fromfile(amplitudeRectFile, dtype=np.float32).reshape(infImg.length, infImg.width)
                (phase*amplitude).astype(np.complex64).tofile(rinfs[i])

                #tidy up
                os.remove(phaseFile)
                os.remove(amplitudeFile)
                os.remove(phaseRectFile)
                os.remove(amplitudeRectFile)
                if infImg.getImageType() == 'amp':
                    create_xml(rinfs[i], infImg.width, infImg.length, 'amp')
                else:
                    create_xml(rinfs[i], infImg.width, infImg.length, 'int')
            else:
                resamp(inf,
                       rinfs[i],
                       'fake',
                       'fake',
                       infImg.width, infImg.length,
                       frames[i].swaths[0].prf,
                       frames[i].swaths[0].dopplerVsPixel,
                       [rangeOffsets2Frac, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [azimuthOffsets2Frac, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                create_xml(rinfs[i], infImg.width, infImg.length, 'slc')

    #determine output width and length
    #actually no need to calculate in azimuth direction
    xs = []
    xe = []
    ys = []
    ye = []
    for i in range(numberOfFrames):
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


    #prepare for mosaicing using numpy
    xs = [x-xmin for x in xs]
    xe = [x-xmin for x in xe]
    ys = [y-ymin for y in ys]
    ye = [y-ymin for y in ye]


    #compute phase offset
    if phaseCompensation:
        phaseOffsetPolynomials = [np.array([0.0])]
        for i in range(1, numberOfFrames):
            upperframe = np.zeros((ye[i-1]-ys[i]+1, outWidth), dtype=np.complex128)
            lowerframe = np.zeros((ye[i-1]-ys[i]+1, outWidth), dtype=np.complex128)
            #upper frame
            if os.path.isfile(rinfs[i-1]):
                upperframe[:,xs[i-1]:xe[i-1]+1] = readImage(rinfs[i-1], rectWidth[i-1], rectLength[i-1], 0, rectWidth[i-1]-1, ys[i]-ys[i-1], ye[i-1]-ys[i-1])
            else:
                upperframe[:,xs[i-1]:xe[i-1]+1] = readImageFromVrt(rinfs[i-1], 0, rectWidth[i-1]-1, ys[i]-ys[i-1], ye[i-1]-ys[i-1])
            #lower frame
            if os.path.isfile(rinfs[i]):
                lowerframe[:,xs[i]:xe[i]+1] = readImage(rinfs[i], rectWidth[i], rectLength[i], 0, rectWidth[i]-1, 0, ye[i-1]-ys[i])
            else:
                lowerframe[:,xs[i]:xe[i]+1] = readImageFromVrt(rinfs[i], 0, rectWidth[i]-1, 0, ye[i-1]-ys[i])
            #get a polynomial
            diff = np.sum(upperframe * np.conj(lowerframe), axis=0)
            (firstLine, lastLine, firstSample, lastSample) = findNonzero(np.reshape(diff, (1, outWidth)))
            #here i use mean value(deg=0) in case difference is around -pi or pi.
            deg = 0
            p = np.polyfit(np.arange(firstSample, lastSample+1), np.angle(diff[firstSample:lastSample+1]), deg)
            phaseOffsetPolynomials.append(p)


            #check fit result
            DEBUG = False
            if DEBUG:
                #create a dir and work in this dir
                diffDir = 'frame_mosaic'
                os.makedirs(diffDir, exist_ok=True)
                os.chdir(diffDir)

                #dump phase difference
                diffFilename = 'phase_difference_frame{}-frame{}.int'.format(i, i+1)
                (upperframe * np.conj(lowerframe)).astype(np.complex64).tofile(diffFilename)
                create_xml(diffFilename, outWidth, ye[i-1]-ys[i]+1, 'int')

                #plot phase difference vs range
                import matplotlib.pyplot as plt
                x = np.arange(firstSample, lastSample+1)
                y = np.angle(diff[firstSample:lastSample+1])
                plt.plot(x, y, label='original phase difference')
                plt.plot(x, np.polyval(p, x), label='fitted phase difference')
                plt.legend()

                plt.minorticks_on()
                plt.tick_params('both', length=10, which='major')
                plt.tick_params('both', length=5, which='minor')

                plt.xlabel('Range Sample Number [Samples]')
                plt.ylabel('Phase Difference [Rad]')
                plt.savefig('phase_difference_frame{}-frame{}.pdf'.format(i, i+1))

                os.chdir('../')


    #mosaic file
    outFp = open(outputfile,'wb')
    for i in range(numberOfFrames):
        print('adding frame: {}'.format(i+1))

        #phase offset in the polynomials
        if phaseCompensation:
            cJ = np.complex64(1j)
            phaseOffset = np.ones(outWidth, dtype=np.complex64)
            for j in range(i+1):
                phaseOffset *= np.exp(cJ*np.polyval(phaseOffsetPolynomials[j], np.arange(outWidth)))

        #get start line number (starts with zero)
        if i == 0:
            ys1 = 0
        else:
            ys1 = int((ye[i-1]+ys[i])/2.0) + 1 - ys[i]
        #get end line number (start with zero)
        if i == numberOfFrames-1:
            ye1 = rectLength[i] - 1
        else:
            ye1 = int((ye[i]+ys[i+1])/2.0) - ys[i]

        #get image format
        inputimage = find_vrt_file(rinfs[i]+'.vrt', 'SourceFilename', relative_path=True)
        byteorder = find_vrt_keyword(rinfs[i]+'.vrt', 'ByteOrder')
        if byteorder == 'LSB':
            swapByte = False
        else:
            swapByte = True
        imageoffset = int(find_vrt_keyword(rinfs[i]+'.vrt', 'ImageOffset'))
        lineoffset = int(find_vrt_keyword(rinfs[i]+'.vrt', 'LineOffset'))

        #read image
        with open(inputimage,'rb') as fp:
            for j in range(ys1, ye1+1):
                fp.seek(imageoffset+j*lineoffset, 0)
                data = np.zeros(outWidth, dtype=np.complex64)
                if swapByte:
                    tmp = np.fromfile(fp, dtype='>f', count=2*rectWidth[i])
                    cJ = np.complex64(1j)
                    data[xs[i]:xe[i]+1] = tmp[0::2] + cJ * tmp[1::2]
                else:
                    data[xs[i]:xe[i]+1] = np.fromfile(fp, dtype=np.complex64, count=rectWidth[i])
                if phaseCompensation:
                    data *= phaseOffset
                data.astype(np.complex64).tofile(outFp)
    outFp.close()


    #delete files. DO NOT DELETE THE FIRST ONE!!!
    for i in range(numberOfFrames):
        if i == 0:
            continue
        os.remove(rinfs[i])
        os.remove(rinfs[i]+'.vrt')
        os.remove(rinfs[i]+'.xml')


    #update frame parameters
    if updateTrack:
        #mosaic size
        track.numberOfSamples = outWidth
        track.numberOfLines = outLength
        #NOTE THAT WE ARE STILL USING SINGLE LOOK PARAMETERS HERE
        #range parameters
        track.startingRange = frames[0].startingRange + (int(rangeOffsets2[0]) - int(rangeOffsets2[xminIndex])) * numberOfRangeLooks * frames[0].rangePixelSize
        track.rangeSamplingRate = frames[0].rangeSamplingRate
        track.rangePixelSize = frames[0].rangePixelSize
        #azimuth parameters
        track.sensingStart = frames[0].sensingStart
        track.prf = frames[0].prf
        track.azimuthPixelSize = frames[0].azimuthPixelSize
        track.azimuthLineInterval = frames[0].azimuthLineInterval


def frameMosaicParameters(track, rangeOffsets, azimuthOffsets, numberOfRangeLooks, numberOfAzimuthLooks):
    '''
    mosaic frames (this simplified version of frameMosaic to only update parameters)
    
    track:                 track
    rangeOffsets:          range offsets
    azimuthOffsets:        azimuth offsets
    numberOfRangeLooks:    number of range looks of the input files
    numberOfAzimuthLooks:  number of azimuth looks of the input files
    '''

    numberOfFrames = len(track.frames)
    frames = track.frames

    rectWidth = []
    rectLength = []
    for i in range(numberOfFrames):
        rectWidth.append(frames[i].numberOfSamples)
        rectLength.append(frames[i].numberOfLines)

    #convert original offset to offset for images with looks
    #use list instead of np.array to make it consistent with the rest of the code
    rangeOffsets1 = [i/numberOfRangeLooks for i in rangeOffsets]
    azimuthOffsets1 = [i/numberOfAzimuthLooks for i in azimuthOffsets]

    #get offset relative to the first frame
    rangeOffsets2 = [0.0]
    azimuthOffsets2 = [0.0]
    for i in range(1, numberOfFrames):
        rangeOffsets2.append(0.0)
        azimuthOffsets2.append(0.0)
        for j in range(1, i+1):
            rangeOffsets2[i] += rangeOffsets1[j]
            azimuthOffsets2[i] += azimuthOffsets1[j]

    #determine output width and length
    #actually no need to calculate in azimuth direction
    xs = []
    xe = []
    ys = []
    ye = []
    for i in range(numberOfFrames):
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
    track.numberOfSamples = outWidth
    track.numberOfLines = outLength
    #NOTE THAT WE ARE STILL USING SINGLE LOOK PARAMETERS HERE
    #range parameters
    track.startingRange = frames[0].startingRange + (int(rangeOffsets2[0]) - int(rangeOffsets2[xminIndex])) * numberOfRangeLooks * frames[0].rangePixelSize
    track.rangeSamplingRate = frames[0].rangeSamplingRate
    track.rangePixelSize = frames[0].rangePixelSize
    #azimuth parameters
    track.sensingStart = frames[0].sensingStart
    track.prf = frames[0].prf
    track.azimuthPixelSize = frames[0].azimuthPixelSize
    track.azimuthLineInterval = frames[0].azimuthLineInterval


def readImageFromVrt(inputfile, startSample, endSample, startLine, endLine):
    '''
    read a chunk of image
    the indexes (startSample, endSample, startLine, endLine) are included and start with zero

    memmap is not used, because it is much slower

    tested against readImage in runSwathMosaic.py
    '''
    import os
    from isceobj.Alos2Proc.Alos2ProcPublic import find_vrt_keyword
    from isceobj.Alos2Proc.Alos2ProcPublic import find_vrt_file

    inputimage = find_vrt_file(inputfile+'.vrt', 'SourceFilename', relative_path=True)
    byteorder = find_vrt_keyword(inputfile+'.vrt', 'ByteOrder')
    if byteorder == 'LSB':
        swapByte = False
    else:
        swapByte = True
    imageoffset = int(find_vrt_keyword(inputfile+'.vrt', 'ImageOffset'))
    lineoffset = int(find_vrt_keyword(inputfile+'.vrt', 'LineOffset'))

    data = np.zeros((endLine-startLine+1, endSample-startSample+1), dtype=np.complex64)
    with open(inputimage,'rb') as fp:
        #fp.seek(imageoffset, 0)
        #for i in range(endLine-startLine+1):
        for i in range(startLine, endLine+1):
            fp.seek(imageoffset+i*lineoffset+startSample*8, 0)
            if swapByte:
                tmp = np.fromfile(fp, dtype='>f', count=2*(endSample-startSample+1))
                cJ = np.complex64(1j)
                data[i-startLine] = tmp[0::2] + cJ * tmp[1::2]
            else:
                data[i-startLine] = np.fromfile(fp, dtype=np.complex64, count=endSample-startSample+1)
    return data
