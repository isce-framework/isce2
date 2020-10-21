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
from isceobj.Alos2Proc.Alos2ProcPublic import multilook
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml

from StackPulic import stackDateStatistics
from StackPulic import acquisitionModesAlos2
from StackPulic import formInterferogram


def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='form interferogram')
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
    dateReference = inps.ref_date
    dateSecondary = inps.sec_date
    numberRangeLooks1 = inps.nrlks1
    numberAzimuthLooks1 = inps.nalks1
    #######################################################

    pair = '{}-{}'.format(dateReference, dateSecondary)

    ml1 = '_{}rlks_{}alks'.format(numberRangeLooks1, numberAzimuthLooks1)

    #use one date to find frames and swaths. any date should work, here we use dateIndexReference
    frames = sorted([x[-4:] for x in glob.glob(os.path.join('./', 'f*_*'))])
    swaths = sorted([int(x[-1]) for x in glob.glob(os.path.join('./', 'f1_*', 's*'))])

    nframe = len(frames)
    nswath = len(swaths)

    for i, frameNumber in enumerate(frames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        os.chdir(frameDir)
        for j, swathNumber in enumerate(range(swaths[0], swaths[-1] + 1)):
            swathDir = 's{}'.format(swathNumber)
            os.chdir(swathDir)
            
            print('processing swath {}, frame {}'.format(swathNumber, frameNumber))

            slcReference = dateReference+'.slc'
            slcSecondary = dateSecondary+'.slc'
            interferogram = pair + ml1 + '.int'
            amplitude = pair + ml1 + '.amp'
            formInterferogram(slcReference, slcSecondary, interferogram, amplitude, numberRangeLooks1, numberAzimuthLooks1)

            os.chdir('../')
        os.chdir('../')




