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
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
from contrib.alos2proc.alos2proc import look
from isceobj.Alos2Proc.Alos2ProcPublic import runCmd
from isceobj.Alos2Proc.Alos2ProcPublic import waterBodyRadar


def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='take more looks')
    parser.add_argument('-date', dest='date', type=str, required=True,
            help = 'date. format: YYMMDD')
    parser.add_argument('-wbd', dest='wbd', type=str, required=True,
            help = 'water body file')
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
    date = inps.date
    wbdFile = inps.wbd
    numberRangeLooks1 = inps.nrlks1
    numberAzimuthLooks1 = inps.nalks1
    numberRangeLooks2 = inps.nrlks2
    numberAzimuthLooks2 = inps.nalks2
    #######################################################

    #pair = '{}-{}'.format(dateReference, dateSecondary)

    ml1 = '_{}rlks_{}alks'.format(numberRangeLooks1, numberAzimuthLooks1)
    ml2 = '_{}rlks_{}alks'.format(numberRangeLooks1*numberRangeLooks2, numberAzimuthLooks1*numberAzimuthLooks2)


    latitude = date + ml1 + '.lat'
    longitude = date + ml1 + '.lon'
    height = date + ml1 + '.hgt'
    los = date + ml1 + '.los'

    multilookLatitude = date + ml2 + '.lat'
    multilookLongitude = date + ml2 + '.lon'
    multilookHeight = date + ml2 + '.hgt'
    multilookLos = date + ml2 + '.los'
    multilookWbdOut = date + ml2 + '.wbd'

    wbdFile = os.path.abspath(wbdFile)

    insarDir = 'insar'
    os.makedirs(insarDir, exist_ok=True)
    os.chdir(insarDir)


    img = isceobj.createImage()
    img.load(latitude+'.xml')
    width = img.width
    length = img.length
    width2 = int(width / numberRangeLooks2)
    length2 = int(length / numberAzimuthLooks2)

    if not ((numberRangeLooks2 == 1) and (numberAzimuthLooks2 == 1)):
        #take looks
        look(latitude, multilookLatitude, width, numberRangeLooks2, numberAzimuthLooks2, 3, 0, 1)
        look(longitude, multilookLongitude, width, numberRangeLooks2, numberAzimuthLooks2, 3, 0, 1)
        look(height, multilookHeight, width, numberRangeLooks2, numberAzimuthLooks2, 3, 0, 1)
        #creat xml
        create_xml(multilookLatitude, width2, length2, 'double')
        create_xml(multilookLongitude, width2, length2, 'double')
        create_xml(multilookHeight, width2, length2, 'double')
        #los has two bands, use look program in isce instead
        #cmd = "looks.py -i {} -o {} -r {} -a {}".format(self._insar.los, self._insar.multilookLos, self._insar.numberRangeLooks2, self._insar.numberAzimuthLooks2)
        #runCmd(cmd)

        #replace the above system call with function call
        from mroipac.looks.Looks import Looks
        from isceobj.Image import createImage
        inImage = createImage()
        inImage.load(los+'.xml')

        lkObj = Looks()
        lkObj.setDownLooks(numberAzimuthLooks2)
        lkObj.setAcrossLooks(numberRangeLooks2)
        lkObj.setInputImage(inImage)
        lkObj.setOutputFilename(multilookLos)
        lkObj.looks()

        #water body
        #this looking operation has no problems where there is only water and land, but there is also possible no-data area
        #look(self._insar.wbdOut, self._insar.multilookWbdOut, width, self._insar.numberRangeLooks2, self._insar.numberAzimuthLooks2, 0, 0, 1)
        #create_xml(self._insar.multilookWbdOut, width2, length2, 'byte')
        #use waterBodyRadar instead to avoid the problems of no-data pixels in water body
        waterBodyRadar(multilookLatitude, multilookLongitude, wbdFile, multilookWbdOut)


    os.chdir('../')