#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging
import numpy as np

import isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import resampleBursts
from isceobj.Alos2Proc.Alos2ProcPublic import mosaicBurstAmplitude
from isceobj.Alos2Proc.Alos2ProcPublic import mosaicBurstInterferogram

logger = logging.getLogger('isce.alos2burstinsar.runCoregSd')

def runCoregSd(self):
    '''coregister bursts by spectral diversity
    '''
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    referenceTrack = self._insar.loadTrack(reference=True)
    secondaryTrack = self._insar.loadTrack(reference=False)

    #demFile = os.path.abspath(self._insar.dem)
    #wbdFile = os.path.abspath(self._insar.wbd)
###############################################################################
    #self._insar.rangeResidualOffsetSd = [[] for i in range(len(referenceTrack.frames))]
    self._insar.azimuthResidualOffsetSd = [[] for i in range(len(referenceTrack.frames))]
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
            # spectral diversity or mai
            ##################################################
            sdDir = 'spectral_diversity'
            os.makedirs(sdDir, exist_ok=True)
            os.chdir(sdDir)

            interferogramDir = 'burst_interf_2_coreg_cc'
            interferogramPrefix = self._insar.referenceBurstPrefix + '-' + self._insar.secondaryBurstPrefix
            offsetSd = spectralDiversity(referenceSwath, os.path.join('../', interferogramDir), interferogramPrefix, self._insar.interferogramSd,
                numberLooksScanSAR=4, numberRangeLooks=28, numberAzimuthLooks=8, coherenceThreshold=0.85, 
                keep=True, filt=True, filtWinSizeRange=5, filtWinSizeAzimuth=5)
            #here use the number of looks for sd as filtWinSizeRange and filtWinSizeAzimuth to get the best filtering result?

            os.chdir('../')

            self._insar.azimuthResidualOffsetSd[i].append(offsetSd)
            catalog.addItem('azimuth residual offset at frame {}, swath {}'.format(frameNumber, swathNumber), '{}'.format(offsetSd), 'runCoregSd')
            

            #this small residual azimuth offset has small impact, it's not worth the time to resample secondary bursts again.
            formInterferogram=False
            if formInterferogram:
                ##################################################
                # resample bursts
                ##################################################
                secondaryBurstResampledDir = self._insar.secondaryBurstPrefix + '_3_coreg_sd'
                #interferogramDir = self._insar.referenceBurstPrefix + '-' + self._insar.secondaryBurstPrefix + '_coreg_geom'
                interferogramDir = 'burst_interf_3_coreg_sd'
                interferogramPrefix = self._insar.referenceBurstPrefix + '-' + self._insar.secondaryBurstPrefix
                resampleBursts(referenceSwath, secondarySwath, 
                    self._insar.referenceBurstPrefix, self._insar.secondaryBurstPrefix, secondaryBurstResampledDir, interferogramDir,
                    self._insar.referenceBurstPrefix, self._insar.secondaryBurstPrefix, self._insar.secondaryBurstPrefix, interferogramPrefix, 
                    self._insar.rangeOffset, self._insar.azimuthOffset, rangeOffsetResidual=self._insar.rangeResidualOffsetCc[i][j], azimuthOffsetResidual=self._insar.azimuthResidualOffsetCc[i][j]+offsetSd)


                ##################################################
                # mosaic burst amplitudes and interferograms
                ##################################################
                os.chdir(secondaryBurstResampledDir)
                mosaicBurstAmplitude(referenceSwath, self._insar.secondaryBurstPrefix, self._insar.secondaryMagnitude, numberOfLooksThreshold=4)
                os.chdir('../')

                os.chdir(interferogramDir)
                mosaicBurstInterferogram(referenceSwath, interferogramPrefix, self._insar.interferogram, numberOfLooksThreshold=4)
                os.chdir('../')


            os.chdir('../')
        os.chdir('../')

###############################################################################
    catalog.printToLog(logger, "runCoregSd")
    self._insar.procDoc.addAllFromCatalog(catalog)


def spectralDiversity(referenceSwath, interferogramDir, interferogramPrefix, outputList, numberLooksScanSAR=None, numberRangeLooks=20, numberAzimuthLooks=10, coherenceThreshold=0.85, keep=False, filt=False, filtWinSizeRange=5, filtWinSizeAzimuth=5):
    '''
    numberLooksScanSAR: number of looks of the ScanSAR system
    numberRangeLooks:   number of range looks to take
    numberAzimuthLooks: number of azimuth looks to take
    keep:               whether keep intermediate files
    '''
    import os
    import numpy as np
    from isceobj.Alos2Proc.Alos2ProcPublic import create_multi_index
    from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
    from isceobj.Alos2Proc.Alos2ProcPublic import multilook
    from isceobj.Alos2Proc.Alos2ProcPublic import cal_coherence_1

    width  = referenceSwath.numberOfSamples
    length = referenceSwath.numberOfLines
    lengthBurst = referenceSwath.burstSlcNumberOfLines
    nBurst = referenceSwath.numberOfBursts
    azsi = referenceSwath.azimuthLineInterval
    tc = referenceSwath.burstCycleLength / referenceSwath.prf

    bursts = [os.path.join(interferogramDir, interferogramPrefix+'_%02d.int'%(i+1)) for i in range(referenceSwath.numberOfBursts)]

    ####################################################
    #input parameters
    rgl = numberRangeLooks
    azl = numberAzimuthLooks
    cor_th = coherenceThreshold
    nls0 = lengthBurst / (referenceSwath.burstSlcFirstLineOffsets[nBurst-1] / (nBurst-1.0))
    print('number of looks of the ScanSAR system: {}'.format(nls0))
    if numberLooksScanSAR != None:
        nls = numberLooksScanSAR
    else:
        nls = int(nls0)
    print('number of looks to be used: {}'.format(nls))
    ####################################################

    #read burst interferograms
    inf = np.zeros((length, width, nls), dtype=np.complex64)
    cnt = np.zeros((length, width), dtype=np.int8)
    for i in range(nBurst):
        if (i+1)%5 == 0 or (i+1) == nBurst:
            print('reading burst %02d' % (i+1))

        burst = np.fromfile(bursts[i], dtype=np.complex64).reshape(lengthBurst, width)

        #subset for the burst
        cntBurst = cnt[0+referenceSwath.burstSlcFirstLineOffsets[i]:lengthBurst+referenceSwath.burstSlcFirstLineOffsets[i], :]
        infBurst = inf[0+referenceSwath.burstSlcFirstLineOffsets[i]:lengthBurst+referenceSwath.burstSlcFirstLineOffsets[i], :, :]
        
        #set number of non-zero pixels
        cntBurst[np.nonzero(burst)] += 1

        #get index
        index1 = np.nonzero(np.logical_and(burst!=0, cntBurst<=nls))
        index2 = index1 + (cntBurst[index1]-1,)

        #set values
        infBurst[index2] = burst[index1]
    
    #number of looks for each sample
    if keep:
        nlFile = 'number_of_looks.nl'
        cnt.astype(np.int8).tofile(nlFile)
        create_xml(nlFile, width, length, 'byte')        

    if filt:
        import scipy.signal as ss
        filterKernel = np.ones((filtWinSizeAzimuth,filtWinSizeRange), dtype=np.float64)
        for i in range(nls):
            print('filtering look {}'.format(i+1))
            flag = (inf[:,:,i]!=0)
            #scale = ss.fftconvolve(flag, filterKernel, mode='same')
            #inf[:,:,i] = flag*ss.fftconvolve(inf[:,:,i], filterKernel, mode='same') / (scale + (scale==0))
            #this should be faster?
            scale = ss.convolve2d(flag, filterKernel, mode='same')
            inf[:,:,i] = flag*ss.convolve2d(inf[:,:,i], filterKernel, mode='same') / (scale + (scale==0))
 
    #width and length after multilooking
    widthm = int(width/rgl)
    lengthm = int(length/azl)
    #use the convention that ka > 0
    ka = -np.polyval(referenceSwath.azimuthFmrateVsPixel[::-1], create_multi_index(width, rgl))

    #get spectral diversity inteferogram
    offset_sd=[]
    for i in range(1, nls):
        print('ouput spectral diversity inteferogram %d' % i)
        #original spectral diversity inteferogram
        sd = inf[:,:,0] * np.conj(inf[:,:,i])

        #replace original amplitude with its square root
        index = np.nonzero(sd!=0)
        sd[index] /= np.sqrt(np.absolute(sd[index]))

        sdFile = outputList[i-1]
        sd.astype(np.complex64).tofile(sdFile)
        create_xml(sdFile, width, length, 'int')

        #multi look
        sdm = multilook(sd, azl, rgl)
        cor = cal_coherence_1(sdm)

        #convert phase to offset
        offset = np.angle(sdm)/(2.0 * np.pi * ka * tc * i)[None,:] / azsi

        #compute offset using good samples
        point_index = np.nonzero(np.logical_and(cor>=cor_th, np.angle(sdm)!=0))
        npoint = round(np.size(point_index)/2)
        if npoint < 20:
            print('WARNING: too few good samples for spectral diversity at look {}: {}'.format(i, npoint))
            offset_sd.append(0)
        else:
            offset_sd.append(  np.sum(offset[point_index]*cor[point_index])/np.sum(cor[point_index])  )

        if keep:
            sdmFile = 'sd_%d_%drlks_%dalks.int' % (i, rgl, azl)
            sdm.astype(np.complex64).tofile(sdmFile)
            create_xml(sdmFile, widthm, lengthm, 'int')
            corFile = 'sd_%d_%drlks_%dalks.cor' % (i, rgl, azl)
            cor.astype(np.float32).tofile(corFile)
            create_xml(corFile, widthm, lengthm, 'float')
            offsetFile = 'sd_%d_%drlks_%dalks.off' % (i, rgl, azl)
            offset.astype(np.float32).tofile(offsetFile)
            create_xml(offsetFile, widthm, lengthm, 'float')

    offset_mean = np.sum(np.array(offset_sd) * np.arange(1, nls)) / np.sum(np.arange(1, nls))

    return offset_mean








