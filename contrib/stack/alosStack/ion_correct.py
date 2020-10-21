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
from isceobj.Alos2Proc.Alos2ProcPublic import renameFile
from isceobj.Alos2Proc.Alos2ProcPublic import runCmd

def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='ionospheric correction')
    parser.add_argument('-ion_dir', dest='ion_dir', type=str, required=True,
            help = 'directory of ionospheric phase for each date')
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

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()


    #get user parameters from input
    ion_dir = inps.ion_dir
    dateReference = inps.ref_date
    dateSecondary = inps.sec_date
    numberRangeLooks1 = inps.nrlks1
    numberAzimuthLooks1 = inps.nalks1
    numberRangeLooks2 = inps.nrlks2
    numberAzimuthLooks2 = inps.nalks2
    #######################################################

    pair = '{}-{}'.format(dateReference, dateSecondary)
    ms = pair
    ml2 = '_{}rlks_{}alks'.format(numberRangeLooks1*numberRangeLooks2, numberAzimuthLooks1*numberAzimuthLooks2)

    multilookDifferentialInterferogram = 'diff_' + ms + ml2 + '.int'
    multilookDifferentialInterferogramOriginal = 'diff_' + ms + ml2 + '_ori.int'
    
    ionosphereReference = os.path.join('../', ion_dir, 'filt_ion_'+dateReference+ml2+'.ion')
    ionosphereSecondary = os.path.join('../', ion_dir, 'filt_ion_'+dateSecondary+ml2+'.ion')


    insarDir = 'insar'
    #os.makedirs(insarDir, exist_ok=True)
    os.chdir(insarDir)

    if not os.path.isfile(ionosphereReference):
        raise Exception('ionospheric phase file: {} of reference date does not exist in {}.\n'.format(os.path.basename(ionosphereReference), ion_dir))
    if not os.path.isfile(ionosphereSecondary):
        raise Exception('ionospheric phase file: {} of secondary date does not exist in {}.\n'.format(os.path.basename(ionosphereSecondary), ion_dir))

    #correct interferogram
    if os.path.isfile(multilookDifferentialInterferogramOriginal):
        print('original interferogram: {} is already here, do not rename: {}'.format(multilookDifferentialInterferogramOriginal, multilookDifferentialInterferogram))
    else:
        print('renaming {} to {}'.format(multilookDifferentialInterferogram, multilookDifferentialInterferogramOriginal))
        renameFile(multilookDifferentialInterferogram, multilookDifferentialInterferogramOriginal)

    cmd = "imageMath.py -e='a*exp(-1.0*J*(b-c))' --a={} --b={} --c={} -s BIP -t cfloat -o {}".format(
        multilookDifferentialInterferogramOriginal,
        ionosphereReference,
        ionosphereSecondary,
        multilookDifferentialInterferogram)
    runCmd(cmd)

    os.chdir('../')
