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
from isceobj.Alos2Proc.runGeocode import geocode
from isceobj.Alos2Proc.Alos2ProcPublic import getBboxGeo

from StackPulic import loadProduct

def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='geocode')
    parser.add_argument('-ref_date_stack_track', dest='ref_date_stack_track', type=str, required=True,
            help = 'track parameter of reference date of stack. format: YYMMDD.track.xml')
    parser.add_argument('-dem', dest='dem', type=str, required=True,
            help = 'dem file used for geocoding')
    parser.add_argument('-input', dest='input', type=str, required=True,
            help='input file to be geocoded')
    parser.add_argument('-bbox', dest='bbox', type=str, default=None,
            help = 'user input bounding box, format: s/n/w/e. default: bbox of ref_date_stack_track')
    parser.add_argument('-interp_method', dest='interp_method', type=str, default='nearest',
            help = 'interpolation method: sinc, bilinear, bicubic, nearest. default: nearest')
    parser.add_argument('-nrlks', dest='nrlks', type=int, default=1,
            help = 'total number of range looks = number of range looks 1 * number of range looks 2.  default: 1')
    parser.add_argument('-nalks', dest='nalks', type=int, default=1,
            help = 'total number of azimuth looks = number of azimuth looks 1 * number of azimuth looks 2. default: 1')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()


    #get user parameters from input
    ref_date_stack_track = inps.ref_date_stack_track
    demGeo = inps.dem
    inputFile = inps.input
    bbox = inps.bbox
    geocodeInterpMethod = inps.interp_method
    numberRangeLooks = inps.nrlks
    numberAzimuthLooks = inps.nalks
    #######################################################

    demFile = os.path.abspath(demGeo)
    trackReferenceStack = loadProduct(ref_date_stack_track)

    #compute bounding box for geocoding
    if bbox is not None:
        bbox = [float(x) for x in bbox.split('/')]
        if len(bbox)!=4:
            raise Exception('user input bbox must have four elements')
    else:
        img = isceobj.createImage()
        img.load(inputFile+'.xml')
        bbox = getBboxGeo(trackReferenceStack, useTrackOnly=True, numberOfSamples=img.width, numberOfLines=img.length, numberRangeLooks=numberRangeLooks, numberAzimuthLooks=numberAzimuthLooks)
    print('=====================================================================================================')
    print('geocode bounding box: {}'.format(bbox))
    print('=====================================================================================================')

    interpMethod = geocodeInterpMethod
    geocode(trackReferenceStack, demFile, inputFile, bbox, numberRangeLooks, numberAzimuthLooks, interpMethod, 0, 0)



