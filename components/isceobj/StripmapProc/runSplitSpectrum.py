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

    referenceFrame = self._insar.loadProduct( self._insar.referenceSlcCropProduct)
    secondaryFrame = self._insar.loadProduct( self._insar.secondarySlcCropProduct)

    referenceSlc =  referenceFrame.getImage().filename
    secondarySlc = secondaryFrame.getImage().filename

    width1 = referenceFrame.getImage().getWidth()
    width2 = secondaryFrame.getImage().getWidth()

    fs_reference = referenceFrame.rangeSamplingRate
    pulseLength_reference = referenceFrame.instrument.pulseLength
    chirpSlope_reference = referenceFrame.instrument.chirpSlope

    #Bandwidth
    B_reference = np.abs(chirpSlope_reference)*pulseLength_reference

    fs_secondary = secondaryFrame.rangeSamplingRate
    pulseLength_secondary = secondaryFrame.instrument.pulseLength
    chirpSlope_secondary = secondaryFrame.instrument.chirpSlope

    #Bandwidth
    B_secondary = np.abs(chirpSlope_secondary)*pulseLength_secondary

    print("reference image range sampling rate: {0} MHz".format(fs_reference/(1.0e6)))
    print("secondary image range sampling rate: {0} MHz".format(fs_secondary/(1.0e6)))


    print("reference image total range bandwidth: {0} MHz".format(B_reference/(1.0e6)))
    print("secondary image total range bandwidth: {0} MHz".format(B_secondary/(1.0e6)))


    # If the bandwidth of reference and secondary are different, choose the smaller bandwidth 
    # for range split spectrum
    B = np.min([B_secondary, B_reference])
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

    os.makedirs(lowBandDir, exist_ok=True)
    os.makedirs(highBandDir, exist_ok=True)

    referenceLowBandSlc = os.path.join(lowBandDir, os.path.basename(referenceSlc)) 
    referenceHighBandSlc = os.path.join(highBandDir, os.path.basename(referenceSlc)) 

    secondaryLowBandSlc = os.path.join(lowBandDir, os.path.basename(secondarySlc))
    secondaryHighBandSlc = os.path.join(highBandDir, os.path.basename(secondarySlc))

    radarWavelength = referenceFrame.radarWavelegth

    print("deviation of low-band's center frequency from full-band's center frequency: {0} MHz".format(fL/1.0e6))

    print("deviation of high-band's center frequency from full-band's center frequency: {0} MHz".format(fH/1.0e6))

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("splitting the range-spectrum of reference SLC")
    split(referenceSlc, referenceLowBandSlc, referenceHighBandSlc, fs_reference, bL, bH, fL, fH)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("splitting the range-spectrum of secondary SLC")
    split(secondarySlc, secondaryLowBandSlc, secondaryHighBandSlc, fs_secondary, bL, bH, fL, fH)
    ########################
    
    createSlcImage(referenceLowBandSlc, width1)
    createSlcImage(referenceHighBandSlc, width1)
    createSlcImage(secondaryLowBandSlc, width2)
    createSlcImage(secondaryHighBandSlc, width2)

    ########################

    f0 = SPEED_OF_LIGHT/radarWavelength
    fH = f0 + fH
    fL = f0 + fL
    wavelengthL = SPEED_OF_LIGHT/fL
    wavelengthH = SPEED_OF_LIGHT/fH

    self.insar.lowBandRadarWavelength = wavelengthL
    self.insar.highBandRadarWavelength = wavelengthH

    self.insar.lowBandSlc1 = referenceLowBandSlc
    self.insar.lowBandSlc2 = secondaryLowBandSlc

    self.insar.highBandSlc1 = referenceHighBandSlc
    self.insar.highBandSlc2 = secondaryHighBandSlc

    ########################
    

