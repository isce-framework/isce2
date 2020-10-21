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
from isceobj.Alos2Proc.runFilt import filt

from StackPulic import createObject

def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='take more looks and compute coherence')
    parser.add_argument('-idir', dest='idir', type=str, required=True,
            help = 'input directory where resampled data of each date (YYMMDD) is located. only folders are recognized')
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
    parser.add_argument('-nrlks2', dest='nrlks2', type=int, default=1,
            help = 'number of range looks 2. default: 1')
    parser.add_argument('-nalks2', dest='nalks2', type=int, default=1,
            help = 'number of azimuth looks 2. default: 1')
    parser.add_argument('-alpha', dest='alpha', type=float, default=0.3,
            help='filtering strength. default: 0.3')
    parser.add_argument('-win', dest='win', type=int, default=32,
            help = 'filter window size. default: 32')
    parser.add_argument('-step', dest='step', type=int, default=4,
            help = 'filter step size. default: 4')
    parser.add_argument('-keep_mag', dest='keep_mag', action='store_true', default=False,
            help='keep magnitude before filtering interferogram')
    parser.add_argument('-wbd_msk', dest='wbd_msk', action='store_true', default=False,
            help='mask filtered interferogram with water body')

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
    dateReferenceStack = inps.ref_date_stack
    dateReference = inps.ref_date
    dateSecondary = inps.sec_date
    numberRangeLooks1 = inps.nrlks1
    numberAzimuthLooks1 = inps.nalks1
    numberRangeLooks2 = inps.nrlks2
    numberAzimuthLooks2 = inps.nalks2
    filterStrength = inps.alpha
    filterWinsize = inps.win
    filterStepsize = inps.step
    removeMagnitudeBeforeFiltering = not inps.keep_mag
    waterBodyMaskStartingStep = inps.wbd_msk
    #######################################################

    pair = '{}-{}'.format(dateReference, dateSecondary)
    ms = pair
    ml2 = '_{}rlks_{}alks'.format(numberRangeLooks1*numberRangeLooks2, numberAzimuthLooks1*numberAzimuthLooks2)

    self = createObject()
    self._insar = createObject()

    self.filterStrength = filterStrength
    self.filterWinsize = filterWinsize
    self.filterStepsize = filterStepsize
    self.removeMagnitudeBeforeFiltering = removeMagnitudeBeforeFiltering
    self._insar.multilookDifferentialInterferogram = 'diff_' + ms + ml2 + '.int'
    self._insar.filteredInterferogram = 'filt_' + ms + ml2 + '.int'
    self._insar.multilookAmplitude = ms + ml2 + '.amp'
    self._insar.multilookPhsig = ms + ml2 + '.phsig'
    self._insar.multilookWbdOut = os.path.join(idir, dateReferenceStack, 'insar', dateReferenceStack + ml2 + '.wbd')
    if waterBodyMaskStartingStep:
        self.waterBodyMaskStartingStep='filt'
    else:
        self.waterBodyMaskStartingStep=None

    filt(self)



