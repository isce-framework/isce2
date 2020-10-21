#!/usr/bin/env python3

#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import glob
import datetime
import numpy as np

import isce, isceobj

from StackPulic import loadTrack
from StackPulic import saveTrack
from StackPulic import stackDateStatistics
from StackPulic import acquisitionModesAlos2


def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='mosaic all swaths and frames to form an entire track')
    parser.add_argument('-idir', dest='idir', type=str, required=True,
            help = 'input directory where data of each date (YYMMDD) is located. only folders are recognized')
    parser.add_argument('-ref_date', dest='ref_date', type=str, required=True,
            help = 'reference date. format: YYMMDD')
    parser.add_argument('-sec_date', dest='sec_date', type=str, nargs='+', default=[],
            help = 'a number of secondary dates seperated by blanks, can also include ref_date. format: YYMMDD YYMMDD YYMMDD. If provided, only process these dates')
    parser.add_argument('-ref_frame', dest='ref_frame', type=str, default=None,
            help = 'frame number of the swath whose grid is used as reference. e.g. 2800. default: first frame')
    parser.add_argument('-ref_swath', dest='ref_swath', type=int, default=None,
            help = 'swath number of the swath whose grid is used as reference. e.g. 1. default: first swath')
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
    idir = inps.idir
    dateReference = inps.ref_date
    dateSecondary = inps.sec_date
    frameReference = inps.ref_frame
    swathReference = inps.ref_swath
    numberRangeLooks1 = inps.nrlks1
    numberAzimuthLooks1 = inps.nalks1
    #######################################################

    DEBUG=False

    spotlightModes, stripmapModes, scansarNominalModes, scansarWideModes, scansarModes = acquisitionModesAlos2()

    #get date statistics
    dateDirs,   dates,   frames,   swaths,   dateIndexReference = stackDateStatistics(idir, dateReference)
    ndate = len(dates)
    nframe = len(frames)
    nswath = len(swaths)

    #find frame and swath indexes of reference swath
    if frameReference is None:
        frameReference = frames[0]
    if swathReference is None:
        swathReference = swaths[0]


    frameReferenceIndex = frames.index(frameReference)
    swathReferenceIndex = swaths.index(swathReference)

    print('resampling all frames and swaths to frame: {} (index: {}) swath: {} (index {})'.format(
        frameReference, frameReferenceIndex, swathReference, swathReferenceIndex))


    #mosaic parameters of each date
    #strictly follow the actual image mosaicking processing of reference (after resampling adjustment in resample_common_grid.py)
    #secondary sensingStart and startingRange are OK, no need to consider other things about secondary
    os.chdir(idir)
    for idate in range(ndate):
        if dateSecondary != []:
            if dates[idate] not in dateSecondary:
                continue

        print('processing: {}'.format(dates[idate]))
        os.chdir(dates[idate])

        track = loadTrack('./', dates[idate])
        swathReference = track.frames[frameReferenceIndex].swaths[swathReferenceIndex]
        #1. mosaic swaths
        for i, frameNumber in enumerate(frames):
            startingRange = []
            sensingStart = []
            endingRange = []
            sensingEnd = []
            for j, swathNumber in enumerate(range(swaths[0], swaths[-1] + 1)):
                swath = track.frames[i].swaths[j]
                startingRange.append(swath.startingRange)
                endingRange.append(swath.startingRange+swath.rangePixelSize*swath.numberOfSamples)
                sensingStart.append(swath.sensingStart)
                sensingEnd.append(swath.sensingStart+datetime.timedelta(seconds=swath.azimuthLineInterval*swath.numberOfLines))
            
            #update frame parameters
            #########################################################
            frame = track.frames[i]
            #mosaic size
            frame.numberOfSamples = int(round((max(endingRange)-min(startingRange))/swathReference.rangePixelSize) / numberRangeLooks1)
            frame.numberOfLines = int(round((max(sensingEnd)-min(sensingStart)).total_seconds()/swathReference.azimuthLineInterval) / numberAzimuthLooks1)
            #NOTE THAT WE ARE STILL USING SINGLE LOOK PARAMETERS HERE
            #range parameters
            frame.startingRange = min(startingRange)
            frame.rangeSamplingRate = swathReference.rangeSamplingRate
            frame.rangePixelSize = swathReference.rangePixelSize
            #azimuth parameters
            frame.sensingStart = min(sensingStart)
            frame.prf = swathReference.prf
            frame.azimuthPixelSize = swathReference.azimuthPixelSize
            frame.azimuthLineInterval = swathReference.azimuthLineInterval


        #2. mosaic frames
        startingRange = []
        sensingStart = []
        endingRange = []
        sensingEnd = []
        for i, frameNumber in enumerate(frames):
            frame = track.frames[i]
            startingRange.append(frame.startingRange)
            endingRange.append(frame.startingRange+numberRangeLooks1*frame.rangePixelSize*frame.numberOfSamples)
            sensingStart.append(frame.sensingStart)
            sensingEnd.append(frame.sensingStart+datetime.timedelta(seconds=numberAzimuthLooks1*frame.azimuthLineInterval*frame.numberOfLines))


        #update track parameters
        #########################################################
        #mosaic size
        track.numberOfSamples = round((max(endingRange)-min(startingRange))/(numberRangeLooks1*swathReference.rangePixelSize))
        track.numberOfLines = round((max(sensingEnd)-min(sensingStart)).total_seconds()/(numberAzimuthLooks1*swathReference.azimuthLineInterval))
        #NOTE THAT WE ARE STILL USING SINGLE LOOK PARAMETERS HERE
        #range parameters
        track.startingRange = min(startingRange)
        track.rangeSamplingRate = swathReference.rangeSamplingRate
        track.rangePixelSize = swathReference.rangePixelSize
        #azimuth parameters
        track.sensingStart = min(sensingStart)
        track.prf = swathReference.prf
        track.azimuthPixelSize = swathReference.azimuthPixelSize
        track.azimuthLineInterval = swathReference.azimuthLineInterval

        #save mosaicking result
        saveTrack(track, dates[idate])
        os.chdir('../')
