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
from isceobj.Alos2Proc.runIonUwrap import ionUwrap

from StackPulic import loadTrack
from StackPulic import createObject
from StackPulic import stackDateStatistics

def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='unwrap subband interferograms for ionospheric correction')
    parser.add_argument('-idir', dest='idir', type=str, required=True,
            help = 'input directory where resampled data of each date (YYMMDD) is located. only folders are recognized')
    parser.add_argument('-ref_date_stack', dest='ref_date_stack', type=str, required=True,
            help = 'reference date of stack. format: YYMMDD')
    parser.add_argument('-ref_date', dest='ref_date', type=str, required=True,
            help = 'reference date of this pair. format: YYMMDD')
    parser.add_argument('-sec_date', dest='sec_date', type=str, required=True,
            help = 'reference date of this pair. format: YYMMDD')
    parser.add_argument('-wbd', dest='wbd', type=str, required=True,
            help = 'water body file')
    parser.add_argument('-nrlks1', dest='nrlks1', type=int, default=1,
            help = 'number of range looks 1. default: 1')
    parser.add_argument('-nalks1', dest='nalks1', type=int, default=1,
            help = 'number of azimuth looks 1. default: 1')
    parser.add_argument('-nrlks_ion', dest='nrlks_ion', type=int, default=1,
            help = 'number of range looks ion. default: 1')
    parser.add_argument('-nalks_ion', dest='nalks_ion', type=int, default=1,
            help = 'number of azimuth looks ion. default: 1')
    parser.add_argument('-filt', dest='filt', action='store_true', default=False,
            help='filter subband interferograms')
    parser.add_argument('-alpha', dest='alpha', type=float, default=0.3,
            help='filtering strength. default: 0.3')
    parser.add_argument('-win', dest='win', type=int, default=32,
            help = 'filter window size. default: 32')
    parser.add_argument('-step', dest='step', type=int, default=4,
            help = 'filter step size. default: 4')
    parser.add_argument('-keep_mag', dest='keep_mag', action='store_true', default=False,
            help='keep magnitude before filtering subband interferogram')

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
    wbd = inps.wbd
    numberRangeLooks1 = inps.nrlks1
    numberAzimuthLooks1 = inps.nalks1
    numberRangeLooksIon = inps.nrlks_ion
    numberAzimuthLooksIon = inps.nalks_ion
    filterSubbandInt = inps.filt
    filterStrengthSubbandInt = inps.alpha
    filterWinsizeSubbandInt = inps.win
    filterStepsizeSubbandInt = inps.step
    removeMagnitudeBeforeFilteringSubbandInt = not inps.keep_mag
    #######################################################

    pair = '{}-{}'.format(dateReference, dateSecondary)
    ms = pair
    ml1 = '_{}rlks_{}alks'.format(numberRangeLooks1, numberAzimuthLooks1)
    dateDirs,   dates,   frames,   swaths,   dateIndexReference = stackDateStatistics(idir, dateReferenceStack)
    trackReference = loadTrack('./', dateReference)

    self = createObject()
    self._insar = createObject()
    self._insar.wbd = wbd
    self._insar.numberRangeLooks1 = numberRangeLooks1
    self._insar.numberAzimuthLooks1 = numberAzimuthLooks1
    self._insar.numberRangeLooksIon = numberRangeLooksIon
    self._insar.numberAzimuthLooksIon = numberAzimuthLooksIon

    self._insar.amplitude = ms + ml1 + '.amp'
    self._insar.differentialInterferogram = 'diff_' + ms + ml1 + '.int'
    self._insar.latitude = dateReferenceStack + ml1 + '.lat'
    self._insar.longitude = dateReferenceStack + ml1 + '.lon'
    self.filterSubbandInt = filterSubbandInt
    self.filterStrengthSubbandInt = filterStrengthSubbandInt
    self.filterWinsizeSubbandInt = filterWinsizeSubbandInt
    self.filterStepsizeSubbandInt = filterStepsizeSubbandInt
    self.removeMagnitudeBeforeFilteringSubbandInt = removeMagnitudeBeforeFilteringSubbandInt

    ionUwrap(self, trackReference, latLonDir=os.path.join(idir, dates[dateIndexReference], 'insar'))
