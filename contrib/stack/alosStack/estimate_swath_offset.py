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
from isceobj.Alos2Proc.runSwathOffset import swathOffset

from StackPulic import loadTrack
from StackPulic import acquisitionModesAlos2


def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='estimate swath offset')
    parser.add_argument('-idir', dest='idir', type=str, required=True,
            help = 'data directory')
    parser.add_argument('-date', dest='date', type=str, required=True,
            help = 'data acquisition date. format: YYMMDD')
    parser.add_argument('-output', dest='output', type=str, required=True,
            help = 'output file')
    #parser.add_argument('-match', dest='match', type=int, default=1,
    #        help = 'do matching when computing adjacent swath offset. 0: no. 1: yes (default)')
    parser.add_argument('-match', dest='match', action='store_true', default=False,
            help='do matching when computing adjacent swath offset')

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
    date = inps.date
    outputFile = inps.output
    match = inps.match
    #######################################################

    spotlightModes, stripmapModes, scansarNominalModes, scansarWideModes, scansarModes = acquisitionModesAlos2()


    frames = sorted([x[-4:] for x in glob.glob(os.path.join(idir, 'f*_*'))])
    track = loadTrack(idir, date)

    #save current dir
    dirOriginal = os.getcwd()
    os.chdir(idir)


    if (track.operationMode in scansarModes) and (len(track.frames[0].swaths) >= 2):
        for i, frameNumber in enumerate(frames):
            frameDir = 'f{}_{}'.format(i+1, frameNumber)
            os.chdir(frameDir)

            mosaicDir = 'mosaic'
            os.makedirs(mosaicDir, exist_ok=True)
            os.chdir(mosaicDir)

            #compute swath offset
            offsetReference = swathOffset(track.frames[i], date+'.slc', outputFile, 
                                       crossCorrelation=match, numberOfAzimuthLooks=10)

            os.chdir('../../')
    else:
        print('there is only one swath, no need to estimate swath offset')
