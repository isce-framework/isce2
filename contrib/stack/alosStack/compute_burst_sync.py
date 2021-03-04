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
from StackPulic import stackDateStatistics


def computeBurstSynchronization(trackReference, trackSecondary):
    '''compute burst synchronization
    '''

    import datetime
    import numpy as np

    frames = [frame.frameNumber for frame in trackReference.frames]
    swaths = [swath.swathNumber for swath in trackReference.frames[0].swaths]
    startingSwath = swaths[0]
    endingSwath = swaths[-1]

    #burst synchronization may slowly change along a track as a result of the changing relative speed of the two flights
    #in one frame, real unsynchronized time is the same for all swaths
    unsynTime = 0
    #real synchronized time/percentage depends on the swath burst length (synTime = burstlength - abs(unsynTime))
    #synTime = 0
    synPercentage = 0

    numberOfFrames = len(frames)
    numberOfSwaths = endingSwath - startingSwath + 1
    
    unsynTimeAll = []
    synPercentageAll = []
    for i, frameNumber in enumerate(frames):
        unsynTimeAll0 = []
        synPercentageAll0 = []
        for j, swathNumber in enumerate(range(startingSwath, endingSwath + 1)):
            referenceSwath = trackReference.frames[i].swaths[j]
            secondarySwath = trackSecondary.frames[i].swaths[j]
            #using Piyush's code for computing range and azimuth offsets
            midRange = referenceSwath.startingRange + referenceSwath.rangePixelSize * referenceSwath.numberOfSamples * 0.5
            midSensingStart = referenceSwath.sensingStart + datetime.timedelta(seconds = referenceSwath.numberOfLines * 0.5 / referenceSwath.prf)
            llh = trackReference.orbit.rdr2geo(midSensingStart, midRange)
            slvaz, slvrng = trackSecondary.orbit.geo2rdr(llh)
            ###Translate to offsets
            #note that secondary range pixel size and prf might be different from reference, here we assume there is a virtual secondary with same
            #range pixel size and prf
            rgoff = ((slvrng - secondarySwath.startingRange) / referenceSwath.rangePixelSize) - referenceSwath.numberOfSamples * 0.5
            azoff = ((slvaz - secondarySwath.sensingStart).total_seconds() * referenceSwath.prf) - referenceSwath.numberOfLines * 0.5

            #compute burst synchronization
            #burst parameters for ScanSAR wide mode not estimed yet
            #if self._insar.modeCombination == 21:
            scburstStartLine = (referenceSwath.burstStartTime - referenceSwath.sensingStart).total_seconds() * referenceSwath.prf + azoff
            #secondary burst start times corresponding to reference burst start times (100% synchronization)
            scburstStartLines = np.arange(scburstStartLine - 100000*referenceSwath.burstCycleLength, \
                                          scburstStartLine + 100000*referenceSwath.burstCycleLength, \
                                          referenceSwath.burstCycleLength)
            dscburstStartLines = -((secondarySwath.burstStartTime - secondarySwath.sensingStart).total_seconds() * secondarySwath.prf - scburstStartLines)
            #find the difference with minimum absolute value
            unsynLines = dscburstStartLines[np.argmin(np.absolute(dscburstStartLines))]
            if np.absolute(unsynLines) >= secondarySwath.burstLength:
                synLines = 0
                if unsynLines > 0:
                    unsynLines = secondarySwath.burstLength
                else:
                    unsynLines = -secondarySwath.burstLength
            else:
                synLines = secondarySwath.burstLength - np.absolute(unsynLines)

            unsynTime += unsynLines / referenceSwath.prf
            synPercentage += synLines / referenceSwath.burstLength * 100.0

            unsynTimeAll0.append(unsynLines / referenceSwath.prf)
            synPercentageAll0.append(synLines / referenceSwath.burstLength * 100.0)

        unsynTimeAll.append(unsynTimeAll0)
        synPercentageAll.append(synPercentageAll0)

            ############################################################################################
            #illustration of the sign of the number of unsynchronized lines (unsynLines)     
            #The convention is the same as ampcor offset, that is,
            #              secondaryLineNumber = referenceLineNumber + unsynLines
            #
            # |-----------------------|     ------------
            # |                       |        ^
            # |                       |        |
            # |                       |        |   unsynLines < 0
            # |                       |        |
            # |                       |       \ /
            # |                       |    |-----------------------|
            # |                       |    |                       |
            # |                       |    |                       |
            # |-----------------------|    |                       |
            #        Reference Burst          |                       |
            #                              |                       |
            #                              |                       |
            #                              |                       |
            #                              |                       |
            #                              |-----------------------|
            #                                     Secondary Burst
            #
            #
            ############################################################################################
 
    #getting average
    #if self._insar.modeCombination == 21:
    unsynTime /= numberOfFrames*numberOfSwaths
    synPercentage /= numberOfFrames*numberOfSwaths

    return (unsynTimeAll, synPercentageAll)


def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='compute burst synchronization for a number of dates')
    parser.add_argument('-idir', dest='idir', type=str, required=True,
            help = 'input directory where data of each date (YYMMDD) is located. only folders are recognized')
    parser.add_argument('-burst_sync_file', dest='burst_sync_file', type=str, required=True,
            help = 'output burst synchronization file')
    parser.add_argument('-ref_date', dest='ref_date', type=str, required=True,
            help = 'reference date. format: YYMMDD')
    parser.add_argument('-sec_date', dest='sec_date', type=str, nargs='+', default=[],
            help = 'a number of secondary dates seperated by blanks. format: YYMMDD YYMMDD YYMMDD. If provided, only compute burst synchronization of these dates')

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
    burstSyncFile = inps.burst_sync_file
    dateReference = inps.ref_date
    dateSecondary = inps.sec_date
    #######################################################


    #get date statistics
    dateDirs,   dates,   frames,   swaths,   dateIndexReference = stackDateStatistics(idir, dateReference)
    ndate = len(dates)
    nframe = len(frames)
    nswath = len(swaths)


    #compute burst synchronization
    trackReference = loadTrack(dateDirs[dateIndexReference], dates[dateIndexReference])

    frames = [frame.frameNumber for frame in trackReference.frames]
    swaths = [swath.swathNumber for swath in trackReference.frames[0].swaths]
    startingSwath = swaths[0]
    endingSwath = swaths[-1]

    burstSync  = '  reference date    secondary date    frame    swath    burst UNsync time [ms]    burst sync [%]\n'
    burstSync += '==================================================================================================\n'

    #compute burst synchronization
    for i in range(ndate):
        if i == dateIndexReference:
            continue
        if dateSecondary != []:
            if dates[i] not in dateSecondary:
                continue

        trackSecondary = loadTrack(dateDirs[i], dates[i])
        unsynTimeAll, synPercentageAll = computeBurstSynchronization(trackReference, trackSecondary)

        for j in range(nframe):
            for k in range(nswath):
                if (j == 0) and (k == 0):
                    burstSync += '     %s            %s          %s      %d           %8.2f              %6.2f\n'%\
                        (dates[dateIndexReference], dates[i], frames[j], swaths[k], unsynTimeAll[j][k]*1000.0, synPercentageAll[j][k])
                else:
                    burstSync += '                                       %s      %d           %8.2f              %6.2f\n'%\
                        (frames[j], swaths[k], unsynTimeAll[j][k]*1000.0, synPercentageAll[j][k])

        burstSync += '                                                             %8.2f (mean)       %6.2f (mean)\n\n'%(np.mean(np.array(unsynTimeAll), dtype=np.float64)*1000.0, np.mean(np.array(synPercentageAll), dtype=np.float64))


    #dump burstSync
    print('\nburst synchronization')
    print(burstSync)
    with open(burstSyncFile, 'w') as f:
        f.write(burstSync)

