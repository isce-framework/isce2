#!/usr/bin/env python3

#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import glob
import shutil
import datetime
import numpy as np
import xml.etree.ElementTree as ET

import isce, isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
from isceobj.Alos2Proc.runSwathOffset import swathOffset
from isceobj.Alos2Proc.runFrameOffset import frameOffset
from isceobj.Alos2Proc.runSwathMosaic import swathMosaic
from isceobj.Alos2Proc.runFrameMosaic import frameMosaic

from StackPulic import acquisitionModesAlos2
from StackPulic import loadTrack

def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='form interferogram')
    parser.add_argument('-ref_date_stack', dest='ref_date_stack', type=str, required=True,
            help = 'reference date of stack. format: YYMMDD')
    parser.add_argument('-ref_date', dest='ref_date', type=str, required=True,
            help = 'reference date of this pair. format: YYMMDD')
    parser.add_argument('-sec_date', dest='sec_date', type=str, required=True,
            help = 'reference date of this pair. format: YYMMDD')
    parser.add_argument('-nrlks1', dest='nrlks1', type=int, default=1,
            help = 'number of range looks 1. default: 1')
    parser.add_argument('-nalks1', dest='nalks1', type=int, default=1,
            help = 'number of azimuth looks 1. default: 1')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()


    #get user parameters from input
    dateReferenceStack = inps.ref_date_stack
    dateReference = inps.ref_date
    dateSecondary = inps.sec_date
    numberRangeLooks1 = inps.nrlks1
    numberAzimuthLooks1 = inps.nalks1
    #######################################################

    logFile = 'process.log'

    pair = '{}-{}'.format(dateReference, dateSecondary)

    ml1 = '_{}rlks_{}alks'.format(numberRangeLooks1, numberAzimuthLooks1)

    interferogram = pair + ml1 + '.int'
    amplitude = pair + ml1 + '.amp'

    spotlightModes, stripmapModes, scansarNominalModes, scansarWideModes, scansarModes = acquisitionModesAlos2()

    #use one date to find frames and swaths. any date should work, here we use dateIndexReference
    frames = sorted([x[-4:] for x in glob.glob(os.path.join('./', 'f*_*'))])
    swaths = sorted([int(x[-1]) for x in glob.glob(os.path.join('./', 'f1_*', 's*'))])

    nframe = len(frames)
    nswath = len(swaths)

    trackReferenceStack = loadTrack('./', dateReferenceStack)

    #mosaic swaths
    for i, frameNumber in enumerate(frames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        os.chdir(frameDir)

        mosaicDir = 'mosaic'
        os.makedirs(mosaicDir, exist_ok=True)
        os.chdir(mosaicDir)


        if not (swaths[-1] - swaths[0] >= 1):

            swathDir = 's{}'.format(swaths[0])
            if not os.path.isfile(interferogram):
                os.symlink(os.path.join('../', swathDir, interferogram), interferogram)
            shutil.copy2(os.path.join('../', swathDir, interferogram+'.vrt'), interferogram+'.vrt')
            shutil.copy2(os.path.join('../', swathDir, interferogram+'.xml'), interferogram+'.xml')
            if not os.path.isfile(amplitude):
                os.symlink(os.path.join('../', swathDir, amplitude), amplitude)
            shutil.copy2(os.path.join('../', swathDir, amplitude+'.vrt'), amplitude+'.vrt')
            shutil.copy2(os.path.join('../', swathDir, amplitude+'.xml'), amplitude+'.xml')

            os.chdir('../../')

        else:
            #compute swath offset using reference stack
            #geometrical offset is enough now
            offsetReferenceStack = swathOffset(trackReferenceStack.frames[i], dateReferenceStack+'.slc', 'swath_offset_' + dateReferenceStack + '.txt', 
                               crossCorrelation=False, numberOfAzimuthLooks=10)
            #we can faithfully make it integer.
            #this can also reduce the error due to floating point computation
            rangeOffsets = [float(round(x)) for x in offsetReferenceStack[0]]
            azimuthOffsets = [float(round(x)) for x in offsetReferenceStack[1]]

            #list of input files
            inputInterferograms = []
            inputAmplitudes = []
            for j, swathNumber in enumerate(range(swaths[0], swaths[-1] + 1)):
                swathDir = 's{}'.format(swathNumber)
                inputInterferograms.append(os.path.join('../', swathDir, interferogram))
                inputAmplitudes.append(os.path.join('../', swathDir, amplitude))

            #note that frame parameters do not need to be updated after mosaicking
            #mosaic amplitudes
            swathMosaic(trackReferenceStack.frames[i], inputAmplitudes, amplitude, 
                rangeOffsets, azimuthOffsets, numberRangeLooks1, numberAzimuthLooks1, resamplingMethod=0)
            #mosaic interferograms
            swathMosaic(trackReferenceStack.frames[i], inputInterferograms, interferogram, 
                rangeOffsets, azimuthOffsets, numberRangeLooks1, numberAzimuthLooks1, resamplingMethod=1)

            create_xml(amplitude, trackReferenceStack.frames[i].numberOfSamples, trackReferenceStack.frames[i].numberOfLines, 'amp')
            create_xml(interferogram, trackReferenceStack.frames[i].numberOfSamples, trackReferenceStack.frames[i].numberOfLines, 'int')

            os.chdir('../../')

        
    #mosaic frame
    mosaicDir = 'insar'
    os.makedirs(mosaicDir, exist_ok=True)
    os.chdir(mosaicDir)

    if nframe == 1:
        frameDir = os.path.join('f1_{}/mosaic'.format(frames[0]))
        if not os.path.isfile(interferogram):
            os.symlink(os.path.join('../', frameDir, interferogram), interferogram)
        #shutil.copy2() can overwrite
        shutil.copy2(os.path.join('../', frameDir, interferogram+'.vrt'), interferogram+'.vrt')
        shutil.copy2(os.path.join('../', frameDir, interferogram+'.xml'), interferogram+'.xml')
        if not os.path.isfile(amplitude):
            os.symlink(os.path.join('../', frameDir, amplitude), amplitude)
        shutil.copy2(os.path.join('../', frameDir, amplitude+'.vrt'), amplitude+'.vrt')
        shutil.copy2(os.path.join('../', frameDir, amplitude+'.xml'), amplitude+'.xml')
    else:
        if trackReferenceStack.operationMode in scansarModes:
            matchingMode=0
        else:
            matchingMode=1

        #geometrical offset is enough
        offsetReferenceStack = frameOffset(trackReferenceStack, dateReferenceStack+'.slc', 'frame_offset_' + dateReferenceStack + '.txt', 
                                   crossCorrelation=False, matchingMode=matchingMode)

        #we can faithfully make it integer.
        #this can also reduce the error due to floating point computation
        rangeOffsets = [float(round(x)) for x in offsetReferenceStack[0]]
        azimuthOffsets = [float(round(x)) for x in offsetReferenceStack[1]]

        #list of input files
        inputInterferograms = []
        inputAmplitudes = []
        for i, frameNumber in enumerate(frames):
            frameDir = 'f{}_{}'.format(i+1, frameNumber)
            inputInterferograms.append(os.path.join('../', frameDir, 'mosaic', interferogram))
            inputAmplitudes.append(os.path.join('../', frameDir, 'mosaic', amplitude))

        #note that track parameters do not need to be updated after mosaicking
        #mosaic amplitudes
        frameMosaic(trackReferenceStack, inputAmplitudes, amplitude, 
            rangeOffsets, azimuthOffsets, numberRangeLooks1, numberAzimuthLooks1, 
            updateTrack=False, phaseCompensation=False, resamplingMethod=0)
        #mosaic interferograms
        (phaseDiffEst, phaseDiffUsed, phaseDiffSource, numberOfValidSamples) = \
        frameMosaic(trackReferenceStack, inputInterferograms, interferogram, 
            rangeOffsets, azimuthOffsets, numberRangeLooks1, numberAzimuthLooks1, 
            updateTrack=False, phaseCompensation=True, resamplingMethod=1)

        create_xml(amplitude, trackReferenceStack.numberOfSamples, trackReferenceStack.numberOfLines, 'amp')
        create_xml(interferogram, trackReferenceStack.numberOfSamples, trackReferenceStack.numberOfLines, 'int')

        #if multiple frames, remove frame amplitudes/inteferograms to save space
        for x in inputAmplitudes:
            os.remove(x)
            os.remove(x+'.vrt')
            os.remove(x+'.xml')

        for x in inputInterferograms:
            os.remove(x)
            os.remove(x+'.vrt')
            os.remove(x+'.xml')

        #log output info
        log  = '{} at {}\n'.format(os.path.basename(__file__), datetime.datetime.now())
        log += '================================================================================================\n'
        log += 'frame phase diff estimated: {}\n'.format(phaseDiffEst[1:])
        log += 'frame phase diff used: {}\n'.format(phaseDiffUsed[1:])
        log += 'frame phase diff used source: {}\n'.format(phaseDiffSource[1:])
        log += 'frame phase diff samples used: {}\n'.format(numberOfValidSamples[1:])
        log += '\n'
        with open(os.path.join('../', logFile), 'a') as f:
            f.write(log)












