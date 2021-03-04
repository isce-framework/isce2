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
from isceobj.Alos2Proc.runUnwrapSnaphu import unwrapSnaphu

from StackPulic import createObject
from StackPulic import loadProduct

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
    parser.add_argument('-wbd_msk', dest='wbd_msk', action='store_true', default=False,
            help='mask unwrapped interferogram with water body')

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
    waterBodyMaskStartingStep = inps.wbd_msk
    #######################################################

    pair = '{}-{}'.format(dateReference, dateSecondary)
    ms = pair
    ml2 = '_{}rlks_{}alks'.format(numberRangeLooks1*numberRangeLooks2, numberAzimuthLooks1*numberAzimuthLooks2)

    self = createObject()
    self._insar = createObject()

    self._insar.filteredInterferogram = 'filt_' + ms + ml2 + '.int'
    self._insar.multilookAmplitude = ms + ml2 + '.amp'
    self._insar.multilookPhsig = ms + ml2 + '.phsig'
    self._insar.unwrappedInterferogram = 'filt_' + ms + ml2 + '.unw'
    self._insar.unwrappedMaskedInterferogram = 'filt_' + ms + ml2 + '_msk.unw'
    self._insar.multilookWbdOut = os.path.join('../', idir, dateReferenceStack, 'insar', dateReferenceStack + ml2 + '.wbd')

    self._insar.numberRangeLooks1 = numberRangeLooks1
    self._insar.numberAzimuthLooks1 = numberAzimuthLooks1
    self._insar.numberRangeLooks2 = numberRangeLooks2
    self._insar.numberAzimuthLooks2 = numberAzimuthLooks2

    if waterBodyMaskStartingStep:
        self.waterBodyMaskStartingStep='unwrap'
    else:
        self.waterBodyMaskStartingStep=None

    trackReference = loadProduct('{}.track.xml'.format(dateReference))
    unwrapSnaphu(self, trackReference)



