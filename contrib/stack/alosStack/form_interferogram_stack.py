#!/usr/bin/env python3

#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

# #examples
# #regular insar
# /home/liang/software/isce/isce_current/isce/contrib/stack/alosStack/form_interferogram_stack.py -dates_dir dates_resampled -pairs_dir pairs -pairs_user 151016-170106 151016-170217 211203-220909 -nrlks1 1 -nalks1 14 -frame_user 2900 -swath_user 1

# #ion lower
# /home/liang/software/isce/isce_current/isce/contrib/stack/alosStack/form_interferogram_stack.py -dates_dir dates_resampled -suffix _lower -pairs_dir pairs_ion -mid_path ion/lower -nrlks1 1 -nalks1 14

# #ion upper
# /home/liang/software/isce/isce_current/isce/contrib/stack/alosStack/form_interferogram_stack.py -dates_dir dates_resampled -suffix _upper -pairs_dir pairs_ion -mid_path ion/upper -nrlks1 1 -nalks1 14


import os
import glob
import shutil
import datetime
import numpy as np
import xml.etree.ElementTree as ET

import isce, isceobj

from StackPulic import datesFromPairs
from StackPulic import formInterferogramStack


def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='form a stack of interferograms')
    parser.add_argument('-dates_dir', dest='dates_dir', type=str, required=True,
            help = 'resampled slc directory')
    parser.add_argument('-suffix', dest='suffix', type=str, default=None,
            help = 'slc file suffix')
    parser.add_argument('-pairs_dir', dest='pairs_dir', type=str, required=True,
            help = 'pair directory')
    parser.add_argument('-mid_path', dest='mid_path', type=str, default=None,
            help = 'path between pair folder and frame folder')
    parser.add_argument('-pairs_user', dest='pairs_user', type=str, nargs='+', default=None,
            help = 'a number of pairs seperated by blanks. format: YYMMDD-YYMMDD YYMMDD-YYMMDD YYMMDD-YYMMDD... This argument has highest priority. When provided, only process these pairs')
    parser.add_argument('-nrlks1', dest='nrlks1', type=int, default=1,
            help = 'number of range looks 1. default: 1')
    parser.add_argument('-nalks1', dest='nalks1', type=int, default=1,
            help = 'number of azimuth looks 1. default: 1')
    parser.add_argument('-frame_user', dest='frame_user', type=str, nargs='+', default=None,
            help = 'a number of frames seperated by blanks. format: 2700 2750 2800... This argument has highest priority. When provided, only process these frames')
    parser.add_argument('-swath_user', dest='swath_user', type=int, nargs='+', default=None,
            help = 'a number of frames seperated by blanks. format: 2 3 4... This argument has highest priority. When provided, only process these swaths')


    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()


    #get user parameters from input
    dates_dir = inps.dates_dir
    suffix = inps.suffix
    if suffix is None:
        suffix = ''
    pairs_dir = inps.pairs_dir
    mid_path = inps.mid_path
    if mid_path is None:
        #os.path.join() will ignore '' 
        mid_path = ''
    pairs_user = inps.pairs_user
    numberRangeLooks1 = inps.nrlks1
    numberAzimuthLooks1 = inps.nalks1
    frame_user = inps.frame_user
    swath_user = inps.swath_user
    #######################################################

     
    dates0 = sorted([os.path.basename(x) for x in glob.glob(os.path.join(dates_dir, '*')) if os.path.isdir(x)])
    pairs = sorted([os.path.basename(x) for x in glob.glob(os.path.join(pairs_dir, '*')) if os.path.isdir(x)])
    if pairs_user is not None:
        pairs = sorted(pairs_user)

    #used dates
    dates = sorted(datesFromPairs(pairs))
    for x in dates:
        if x not in dates0:
            raise Exception('date {} used in pairs, but does not exist'.format(x))

    ndates = len(dates)
    npairs = len(pairs)

    #link dates and pairs
    date_pair_index = []
    for i in range(npairs):
        date_pair_index.append([dates.index(x) for x in pairs[i].split('-')])


    #######################################################
    ml1 = '_{}rlks_{}alks'.format(numberRangeLooks1, numberAzimuthLooks1)

    #use one date to find frames and swaths. any date should work, here we use first date. all should be the same as those of reference date of stack
    frames = sorted([x[-4:] for x in glob.glob(os.path.join(dates_dir, dates[0], 'f*_*'))])
    swaths = sorted([int(x[-1]) for x in glob.glob(os.path.join(dates_dir, dates[0], 'f1_*', 's*'))])

    nframe = len(frames)
    nswath = len(swaths)

    if frame_user is None:
        frame_user = frames
    if swath_user is None:
        swath_user = swaths

    for i, frameNumber in enumerate(frames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        #os.chdir(frameDir)
        for j, swathNumber in enumerate(range(swaths[0], swaths[-1] + 1)):
            swathDir = 's{}'.format(swathNumber)
            #os.chdir(swathDir)
            if (frameNumber not in frame_user) or (swathNumber not in swath_user):
                continue

            print('\nprocessing swath {}, frame {}'.format(swathNumber, frameNumber))
            slcs = [os.path.join(dates_dir, x, frameDir, swathDir, x+suffix+'.slc') for x in dates]
            # interferograms = [os.path.join(pairs_dir, x, mid_path, frameDir, swathDir, x+ml1+'.int') for x in pairs]
            # amplitudes = [os.path.join(pairs_dir, x, mid_path, frameDir, swathDir, x+ml1+'.amp') for x in pairs]

            interferograms = []
            amplitudes = []
            for x in pairs:
                output_dir = os.path.join(pairs_dir, x, mid_path, frameDir, swathDir)
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                interferograms.append(os.path.join(output_dir, x+ml1+'.int'))
                amplitudes.append(os.path.join(output_dir, x+ml1+'.amp'))

            formInterferogramStack(slcs, date_pair_index, interferograms, amplitudes=amplitudes, numberRangeLooks=numberRangeLooks1, numberAzimuthLooks=numberAzimuthLooks1)




















