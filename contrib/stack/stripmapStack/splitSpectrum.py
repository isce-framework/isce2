#!/usr/bin/env python3
#Author: Heresh Fattahi


import numpy as np
import argparse
import os
import isce
import isceobj
import shelve
#import BurstUtils as BU
#from Sentinel1A_TOPS import Sentinel1A_TOPS
#import pyfftw
import copy
import time
#import matplotlib.pyplot as plt
from contrib.splitSpectrum import SplitRangeSpectrum as splitSpectrum
from isceobj.Constants import SPEED_OF_LIGHT
import gdal


def createParser():
    '''     
    Command line parser.
    '''
            
    parser = argparse.ArgumentParser( description='split the range spectrum of SLC')
    parser.add_argument('-s', '--slc', dest='slc', type=str, required=True,
            help='Name of the SLC image or the directory that contains the burst slcs')
    parser.add_argument('-o', '--outDir', dest='outDir', type=str, required=True,
            help='Name of the output directory')
    parser.add_argument('-L', '--dcL', dest='dcL', type=float, default=None,
            help='Low band central frequency [MHz]')
    parser.add_argument('-H', '--dcH', dest='dcH', type=float, default=None,
            help='High band central frequency [MHz]')
    parser.add_argument('-b', '--bwL', dest='bwL', type=float, default=None,
            help='band width of the low-band')
    parser.add_argument('-B', '--bwH', dest='bwH', type=float, default=None,
            help='band width of the high-band')
    parser.add_argument('-m', '--shelve', dest='shelve', type=str, default=None,
            help='shelve file used to extract metadata')
    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


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

def getShape(fileName):

    dataset = gdal.Open(fileName,gdal.GA_ReadOnly)
    return dataset.RasterYSize, dataset.RasterXSize

def main(iargs=None):
    '''
    Split the range spectrum
    '''
    #Check if the master and slave are .slc files then go ahead and split the range spectrum
    tstart = time.time()
    inps = cmdLineParse(iargs)
    print ('input full-band SLC: ', inps.slc)
    if os.path.isfile(inps.slc):

        
        with shelve.open((inps.shelve), flag='r') as db:
            frame = db['frame']
            try:
              doppler = db['doppler']
            except:
              doppler = None

        radarWavelength = frame.radarWavelegth
        fs = frame.rangeSamplingRate

        pulseLength = frame.instrument.pulseLength
        chirpSlope = frame.instrument.chirpSlope

        #Bandwidth
        totalBandwidth = np.abs(chirpSlope)*pulseLength # Hz


        ###############################################
        if not (inps.dcL and inps.dcH and inps.bwL and inps.bwH):
                # If center frequency and bandwidth of the desired sub-bands are not given,
                # let's choose the one-third of the total bandwidth at the two ends of the 
                # spectrum as low-band and high band
                #pulseLength = frame.instrument.pulseLength
                #chirpSlope = frame.instrument.chirpSlope

                #Bandwidth
                #totalBandwidth = np.abs(chirpSlope)*pulseLength # Hz

                # Dividing the total bandwidth of B to three bands and consider the sub bands on
                # the most left and right hand side as the spectrum of low band and high band SLCs

                # band width of the sub-bands 
                inps.bwL = totalBandwidth/3.0
                inps.bwH = totalBandwidth/3.0
                # center frequency of the low-band
                inps.dcL = -1.0*totalBandwidth/3.0

                # center frequency of the high-band
                inps.dcH = totalBandwidth/3.0

        print("**********************")
        print("Total range bandwidth: ", totalBandwidth)
        print("low-band bandwidth: ", inps.bwL)
        print("high-band bandwidth: ", inps.bwH)
        print("dcL: ", inps.dcL)
        print("dcH: ", inps.dcH)
        print("**********************")

        outDirH = os.path.join(inps.outDir,'HighBand')
        outDirL = os.path.join(inps.outDir,'LowBand')

        if not os.path.exists(outDirH):
           os.makedirs(outDirH)

        if not os.path.exists(outDirL):
           os.makedirs(outDirL)

        fullBandSlc = os.path.basename(inps.slc)
        lowBandSlc = os.path.join(outDirL, fullBandSlc)
        highBandSlc = os.path.join(outDirH, fullBandSlc)

        print(inps.slc, lowBandSlc, highBandSlc, fs, inps.bwL, inps.bwH, inps.dcL, inps.dcH)
        print("strat")
        split(inps.slc, lowBandSlc, highBandSlc, fs, inps.bwL, inps.bwH, inps.dcL, inps.dcH)
        print("end")
        length, width = getShape(inps.slc + ".vrt")
        createSlcImage(lowBandSlc, width)
        createSlcImage(highBandSlc, width)

        f0 = SPEED_OF_LIGHT/radarWavelength
        fH = f0 + inps.dcH
        fL = f0 + inps.dcL
        wavelengthL = SPEED_OF_LIGHT/fL
        wavelengthH = SPEED_OF_LIGHT/fH

        frameH = copy.deepcopy(frame)
        frameH.subBandRadarWavelength = wavelengthH
        frameH.image.filename = highBandSlc
        with shelve.open(os.path.join(outDirH, 'data')) as db:
            db['frame'] = frameH
            if doppler:
               db['doppler'] = doppler  

        frameL = copy.deepcopy(frame)
        frameL.subBandRadarWavelength = wavelengthL
        frameL.image.filename = lowBandSlc
        with shelve.open(os.path.join(outDirL, 'data')) as db:
            db['frame'] = frameL 
            if doppler:
               db['doppler'] = doppler
        
        print ('total processing time: ', time.time()-tstart, ' sec')

   
if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()



