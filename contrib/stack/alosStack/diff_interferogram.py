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
from isceobj.Alos2Proc.Alos2ProcPublic import runCmd

from StackPulic import loadProduct
from StackPulic import stackDateStatistics


def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='form interferogram')
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
    #######################################################

    pair = '{}-{}'.format(dateReference, dateSecondary)

    ml1 = '_{}rlks_{}alks'.format(numberRangeLooks1, numberAzimuthLooks1)

    dateDirs,   dates,   frames,   swaths,   dateIndexReference = stackDateStatistics(idir, dateReferenceStack)

    trackParameter = os.path.join(dateDirs[dateIndexReference], dates[dateIndexReference]+'.track.xml')
    trackReferenceStack = loadProduct(trackParameter)

    rangePixelSize = numberRangeLooks1 * trackReferenceStack.rangePixelSize
    radarWavelength = trackReferenceStack.radarWavelength

    insarDir = 'insar'
    os.makedirs(insarDir, exist_ok=True)
    os.chdir(insarDir)

    interferogram = pair + ml1 + '.int'
    differentialInterferogram = 'diff_' + pair + ml1 + '.int'

    if dateReference == dateReferenceStack:
        rectRangeOffset = os.path.join('../', idir, dateSecondary, 'insar', dateSecondary + ml1 + '_rg_rect.off')
        cmd = "imageMath.py -e='a*exp(-1.0*J*b*4.0*{}*{}/{})*(b!=0)' --a={} --b={} -o {} -t cfloat".format(np.pi, rangePixelSize, radarWavelength, interferogram, rectRangeOffset, differentialInterferogram)
    elif dateSecondary == dateReferenceStack:
        rectRangeOffset = os.path.join('../', idir, dateReference, 'insar', dateReference + ml1 + '_rg_rect.off')
        cmd = "imageMath.py -e='a*exp(1.0*J*b*4.0*{}*{}/{})*(b!=0)' --a={} --b={} -o {} -t cfloat".format(np.pi, rangePixelSize, radarWavelength, interferogram, rectRangeOffset, differentialInterferogram)
    else:
        rectRangeOffset1 = os.path.join('../', idir, dateReference, 'insar', dateReference + ml1 + '_rg_rect.off')
        rectRangeOffset2 = os.path.join('../', idir, dateSecondary, 'insar', dateSecondary + ml1 + '_rg_rect.off')
        cmd = "imageMath.py -e='a*exp(1.0*J*(b-c)*4.0*{}*{}/{})*(b!=0)*(c!=0)' --a={} --b={} --c={} -o {} -t cfloat".format(np.pi, rangePixelSize, radarWavelength, interferogram, rectRangeOffset1, rectRangeOffset2, differentialInterferogram)
    runCmd(cmd)


    os.chdir('../')
