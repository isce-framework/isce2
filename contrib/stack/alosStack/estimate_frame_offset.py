#!/usr/bin/env python3

#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os

import isce, isceobj
from isceobj.Alos2Proc.runFrameOffset import frameOffset

from StackPulic import loadTrack
from StackPulic import acquisitionModesAlos2


def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='estimate frame offset')
    parser.add_argument('-idir', dest='idir', type=str, required=True,
            help = 'data directory')
    parser.add_argument('-date', dest='date', type=str, required=True,
            help = 'data acquisition date. format: YYMMDD')
    parser.add_argument('-output', dest='output', type=str, required=True,
            help = 'output file')
    #parser.add_argument('-match', dest='match', type=int, default=1,
    #        help = 'do matching when computing adjacent frame offset. 0: no. 1: yes (default)')
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


    track = loadTrack(idir, date)

    #save current dir
    dirOriginal = os.getcwd()
    os.chdir(idir)


    if len(track.frames) > 1:
        if track.operationMode in scansarModes:
            matchingMode=0
        else:
            matchingMode=1

        mosaicDir = 'insar'
        os.makedirs(mosaicDir, exist_ok=True)
        os.chdir(mosaicDir)

        #compute swath offset
        offsetReference = frameOffset(track, date+'.slc', 'frame_offset.txt', 
                                   crossCorrelation=match, matchingMode=matchingMode)

        os.chdir('../')
    else:
        print('there is only one frame, no need to estimate frame offset')

