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
from isceobj.Alos2Proc.Alos2ProcPublic import waterBodyRadar
from isceobj.Alos2Proc.runRdr2Geo import topoCPU
from isceobj.Alos2Proc.runRdr2Geo import topoGPU

from StackPulic import loadTrack
from StackPulic import hasGPU


def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='compute longitude, latitude, height and water body from radar parameters')
    parser.add_argument('-date', dest='date', type=str, required=True,
            help = 'date. format: YYMMDD')
    parser.add_argument('-dem', dest='dem', type=str, required=True,
            help = 'dem file')
    parser.add_argument('-wbd', dest='wbd', type=str, required=True,
            help = 'water body file')
    parser.add_argument('-nrlks1', dest='nrlks1', type=int, default=1,
            help = 'number of range looks 1. default: 1')
    parser.add_argument('-nalks1', dest='nalks1', type=int, default=1,
            help = 'number of azimuth looks 1. default: 1')
    #parser.add_argument('-gpu', dest='gpu', type=int, default=1,
    #        help = 'use GPU when available. 0: no. 1: yes (default)')
    parser.add_argument('-gpu', dest='gpu', action='store_true', default=False,
            help='use GPU when available')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()


    #get user parameters from input
    date = inps.date
    demFile = inps.dem
    wbdFile = inps.wbd
    numberRangeLooks1 = inps.nrlks1
    numberAzimuthLooks1 = inps.nalks1
    useGPU = inps.gpu
    #######################################################

    demFile = os.path.abspath(demFile)
    wbdFile = os.path.abspath(wbdFile)

    insarDir = 'insar'
    os.makedirs(insarDir, exist_ok=True)
    os.chdir(insarDir)

    ml1 = '_{}rlks_{}alks'.format(numberRangeLooks1, numberAzimuthLooks1)

    latitude = date + ml1 + '.lat'
    longitude = date + ml1 + '.lon'
    height = date + ml1 + '.hgt'
    los = date + ml1 + '.los'
    wbdOut = date + ml1 + '.wbd'


    track = loadTrack('../', date)
    if useGPU and hasGPU():
        topoGPU(track, numberRangeLooks1, numberAzimuthLooks1, demFile, 
                       latitude, longitude, height, los)
    else:
        snwe = topoCPU(track, numberRangeLooks1, numberAzimuthLooks1, demFile, 
                       latitude, longitude, height, los)
    waterBodyRadar(latitude, longitude, wbdFile, wbdOut)


