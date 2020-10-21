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
from contrib.alos2proc.alos2proc import look
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
from isceobj.Alos2Proc.Alos2ProcPublic import runCmd
from isceobj.Alos2Proc.runCoherence import coherence

from StackPulic import loadProduct
from StackPulic import stackDateStatistics


def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='take more looks and compute coherence')
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
    dateReference = inps.ref_date
    dateSecondary = inps.sec_date
    numberRangeLooks1 = inps.nrlks1
    numberAzimuthLooks1 = inps.nalks1
    numberRangeLooks2 = inps.nrlks2
    numberAzimuthLooks2 = inps.nalks2
    #######################################################

    pair = '{}-{}'.format(dateReference, dateSecondary)

    ml1 = '_{}rlks_{}alks'.format(numberRangeLooks1, numberAzimuthLooks1)
    ml2 = '_{}rlks_{}alks'.format(numberRangeLooks1*numberRangeLooks2, numberAzimuthLooks1*numberAzimuthLooks2)

    insarDir = 'insar'
    os.makedirs(insarDir, exist_ok=True)
    os.chdir(insarDir)

    amplitude = pair + ml1 + '.amp'
    differentialInterferogram = 'diff_' + pair + ml1 + '.int'
    multilookAmplitude = pair + ml2 + '.amp'
    multilookDifferentialInterferogram = 'diff_' + pair + ml2 + '.int'
    multilookCoherence = pair + ml2 + '.cor'

    amp = isceobj.createImage()
    amp.load(amplitude+'.xml')
    width = amp.width
    length = amp.length
    width2 = int(width / numberRangeLooks2)
    length2 = int(length / numberAzimuthLooks2)


    if not ((numberRangeLooks2 == 1) and (numberAzimuthLooks2 == 1)):
        #take looks
        look(differentialInterferogram, multilookDifferentialInterferogram, width, numberRangeLooks2, numberAzimuthLooks2, 4, 0, 1)
        look(amplitude, multilookAmplitude, width, numberRangeLooks2, numberAzimuthLooks2, 4, 1, 1)
        #creat xml
        create_xml(multilookDifferentialInterferogram, width2, length2, 'int')
        create_xml(multilookAmplitude, width2, length2, 'amp')



    if (numberRangeLooks1*numberRangeLooks2*numberAzimuthLooks1*numberAzimuthLooks2 >= 9):
        cmd = "imageMath.py -e='sqrt(b_0*b_1);abs(a)/(b_0+(b_0==0))/(b_1+(b_1==0))*(b_0!=0)*(b_1!=0)' --a={} --b={} -o {} -t float -s BIL".format(
            multilookDifferentialInterferogram,
            multilookAmplitude,
            multilookCoherence)
        runCmd(cmd)
    else:
        #estimate coherence using a moving window
        coherence(multilookAmplitude, multilookDifferentialInterferogram, multilookCoherence, 
            method="cchz_wave", windowSize=5)


    os.chdir('../')
