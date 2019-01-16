#
# Author: Heresh Fattahi
# Copyright 2017
#
import logging
import isceobj
from contrib.splitSpectrum import SplitRangeSpectrum as splitSpectrum
import numpy as np
import os
from isceobj.Constants import SPEED_OF_LIGHT
import time


logger = logging.getLogger('isce.insar.runSplitSpectrum')

def split(fullBandSlc, lowBandSlc, highBandSlc, fs, bL, bH, fL, fH):

    ss = splitSpectrum()

    ss.blocksize = 100
    ss.memsize = 512
    ss.inputDS = fullBandSlc + ".vrt"
    ss.lbDS = lowBandSlc
    ss.hbDS = highBandSlc
    ss.rangeSamplingRate = fs
    ss.lowBandWidth = bL
    ss.highBandWidth = bH
    ss.lowCenterFrequency = fL
    ss.highCenterFrequency = fH

    ss.split()

def createSlcImage(slcName, width):

    slc = isceobj.createSlcImage()
    slc.setWidth(width)
    slc.filename = slcName
    slc.setAccessMode('write')
    slc.renderHdr()
    

def adjustCenterFrequency(B,  N,  dc):

    # because of quantization, there may not be an index representing dc. We 
    # therefore adjust dc to make sure that there is an index to represent it. 
    # We find the index that is closest to nominal dc and then adjust dc to the 
    # frequency of that index.
    # B = full band-width
    # N = length of signal
    # dc = center frequency of the sub-band

    df = B/N
    if (dc < 0):
        ind = N + np.round(dc/df)
    
    else:
        ind = np.round(dc/df);
    
    dc = frequency (B, N, ind)

    return dc


def frequency (B, N, n):
# calculates frequency at a given index.
# Assumption: for indices 0 to (N-1)/2, frequency is positive 
# and for indices larger than (N-1)/2 frequency is negative

#frequency interval given B as the total bandwidth
    df = B/N
    middleIndex = int((N-1)/2)

    if (n > middleIndex):
        f = (n-N)*df

    else:
        f = n*df

    return f


def runSplitSpectrum(self):
    '''
    Generate split spectrum SLCs.
    '''

    if not self.doSplitSpectrum:
        print('Split spectrum processing not requested. Skipping ....')
        return

    masterFrame = self._insar.loadProduct( self._insar.masterSlcCropProduct)
    slaveFrame = self._insar.loadProduct( self._insar.slaveSlcCropProduct)

    masterSlc =  masterFrame.getImage().filename
    slaveSlc = slaveFrame.getImage().filename

    width1 = masterFrame.getImage().getWidth()
    width2 = slaveFrame.getImage().getWidth()

    fs_master = masterFrame.rangeSamplingRate
    pulseLength_master = masterFrame.instrument.pulseLength
    chirpSlope_master = masterFrame.instrument.chirpSlope

    #Bandwidth
    B_master = np.abs(chirpSlope_master)*pulseLength_master

    fs_slave = slaveFrame.rangeSamplingRate
    pulseLength_slave = slaveFrame.instrument.pulseLength
    chirpSlope_slave = slaveFrame.instrument.chirpSlope

    #Bandwidth
    B_slave = np.abs(chirpSlope_slave)*pulseLength_slave

    print("master image range sampling rate: {0} MHz".format(fs_master/(1.0e6)))
    print("slave image range sampling rate: {0} MHz".format(fs_slave/(1.0e6)))


    print("master image total range bandwidth: {0} MHz".format(B_master/(1.0e6)))
    print("slave image total range bandwidth: {0} MHz".format(B_slave/(1.0e6)))


    # If the bandwidth of master and slave are different, choose the smaller bandwidth 
    # for range split spectrum
    B = np.min([B_slave, B_master])
    print("Bandwidth used for split spectrum: {0} MHz".format(B/(1.e6)))

    # Dividing the total bandwidth of B to three bands and consider the sub bands on
    # the most left and right hand side as the spectrum of low band and high band SLCs
    
    # band width of the low-band 
    bL = B/3.0

    # band width of the high-band 
    bH = B/3.0

    # center frequency of the low-band
    fL = -1.0*B/3.0

    # center frequency of the high-band
    fH = B/3.0

    lowBandDir = os.path.join(self.insar.splitSpectrumDirname, self.insar.lowBandSlcDirname)
    highBandDir = os.path.join(self.insar.splitSpectrumDirname, self.insar.highBandSlcDirname)

    if os.path.isdir(lowBandDir):
        logger.info('low-band slc directory {0} already exists.'.format(lowBandDir))
    else:
        os.makedirs(lowBandDir)

    if os.path.isdir(highBandDir):
        logger.info('high-band slc directory {0} already exists.'.format(highBandDir))
    else:
        os.makedirs(highBandDir)

    masterLowBandSlc = os.path.join(lowBandDir, os.path.basename(masterSlc)) 
    masterHighBandSlc = os.path.join(highBandDir, os.path.basename(masterSlc)) 

    slaveLowBandSlc = os.path.join(lowBandDir, os.path.basename(slaveSlc))
    slaveHighBandSlc = os.path.join(highBandDir, os.path.basename(slaveSlc))

    radarWavelength = masterFrame.radarWavelegth

    print("deviation of low-band's center frequency from full-band's center frequency: {0} MHz".format(fL/1.0e6))

    print("deviation of high-band's center frequency from full-band's center frequency: {0} MHz".format(fH/1.0e6))

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("splitting the range-spectrum of master SLC")
    split(masterSlc, masterLowBandSlc, masterHighBandSlc, fs_master, bL, bH, fL, fH)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("splitting the range-spectrum of slave SLC")
    split(slaveSlc, slaveLowBandSlc, slaveHighBandSlc, fs_slave, bL, bH, fL, fH)
    ########################
    
    createSlcImage(masterLowBandSlc, width1)
    createSlcImage(masterHighBandSlc, width1)
    createSlcImage(slaveLowBandSlc, width2)
    createSlcImage(slaveHighBandSlc, width2)

    ########################

    f0 = SPEED_OF_LIGHT/radarWavelength
    fH = f0 + fH
    fL = f0 + fL
    wavelengthL = SPEED_OF_LIGHT/fL
    wavelengthH = SPEED_OF_LIGHT/fH

    self.insar.lowBandRadarWavelength = wavelengthL
    self.insar.highBandRadarWavelength = wavelengthH

    self.insar.lowBandSlc1 = masterLowBandSlc
    self.insar.lowBandSlc2 = slaveLowBandSlc

    self.insar.highBandSlc1 = masterHighBandSlc
    self.insar.highBandSlc2 = slaveHighBandSlc

    ########################
    

