#!/usr/bin/env python3

#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import glob
import datetime
import numpy as np

import isce, isceobj
from isceobj.Alos2Proc.runGeo2Rdr import geo2RdrCPU
from isceobj.Alos2Proc.runGeo2Rdr import geo2RdrGPU

from StackPulic import loadTrack
from StackPulic import hasGPU


def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='compute range and azimuth offsets')
    parser.add_argument('-date', dest='date', type=str, required=True,
            help = 'date. format: YYMMDD')
    parser.add_argument('-date_par_dir', dest='date_par_dir', type=str, default='./',
            help = 'date parameter directory. default: ./')
    parser.add_argument('-lat', dest='lat', type=str, required=True,
            help = 'latitude file')
    parser.add_argument('-lon', dest='lon', type=str, required=True,
            help = 'longtitude file')
    parser.add_argument('-hgt', dest='hgt', type=str, required=True,
            help = 'height file')
    parser.add_argument('-nrlks1', dest='nrlks1', type=int, default=1,
            help = 'number of range looks 1. default: 1')
    parser.add_argument('-nalks1', dest='nalks1', type=int, default=1,
            help = 'number of azimuth looks 1. default: 1')
    #parser.add_argument('-gpu', dest='gpu', type=int, default=1,
    #        help = 'use GPU when available. 0: no. 1: yes (default)')
    parser.add_argument('-gpu', dest='gpu', action='store_true', default=False,
            help='use GPU when available')

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
    dateParDir = os.path.join('../', inps.date_par_dir)
    latitude = os.path.join('../', inps.lat)
    longitude = os.path.join('../', inps.lon)
    height = os.path.join('../', inps.hgt)
    numberRangeLooks1 = inps.nrlks1
    numberAzimuthLooks1 = inps.nalks1
    useGPU = inps.gpu
    #######################################################

    insarDir = 'insar'
    os.makedirs(insarDir, exist_ok=True)
    os.chdir(insarDir)

    ml1 = '_{}rlks_{}alks'.format(numberRangeLooks1, numberAzimuthLooks1)

    rangeOffset = date + ml1 + '_rg.off'
    azimuthOffset = date + ml1 + '_az.off'

    
    if not os.path.isfile(os.path.basename(latitude)):
        latitudeLink = True
        os.symlink(latitude, os.path.basename(latitude))
        os.symlink(latitude+'.vrt', os.path.basename(latitude)+'.vrt')
        os.symlink(latitude+'.xml', os.path.basename(latitude)+'.xml')
    else:
        latitudeLink = False

    if not os.path.isfile(os.path.basename(longitude)):
        longitudeLink = True
        os.symlink(longitude, os.path.basename(longitude))
        os.symlink(longitude+'.vrt', os.path.basename(longitude)+'.vrt')
        os.symlink(longitude+'.xml', os.path.basename(longitude)+'.xml')
    else:
        longitudeLink = False

    if not os.path.isfile(os.path.basename(height)):
        heightLink = True
        os.symlink(height, os.path.basename(height))
        os.symlink(height+'.vrt', os.path.basename(height)+'.vrt')
        os.symlink(height+'.xml', os.path.basename(height)+'.xml')
    else:
        heightLink = False



    track = loadTrack(dateParDir, date)
    if useGPU and hasGPU():
        geo2RdrGPU(track, numberRangeLooks1, numberAzimuthLooks1, 
            latitude, longitude, height, rangeOffset, azimuthOffset)
    else:
        geo2RdrCPU(track, numberRangeLooks1, numberAzimuthLooks1, 
            latitude, longitude, height, rangeOffset, azimuthOffset)



    if latitudeLink == True:
        os.remove(os.path.basename(latitude))
        os.remove(os.path.basename(latitude)+'.vrt')
        os.remove(os.path.basename(latitude)+'.xml')

    if longitudeLink == True:
        os.remove(os.path.basename(longitude))
        os.remove(os.path.basename(longitude)+'.vrt')
        os.remove(os.path.basename(longitude)+'.xml')

    if heightLink == True:
        os.remove(os.path.basename(height))
        os.remove(os.path.basename(height)+'.vrt')
        os.remove(os.path.basename(height)+'.xml')

