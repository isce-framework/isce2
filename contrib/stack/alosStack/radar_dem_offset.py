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
from isceobj.Alos2Proc.runRdrDemOffset import rdrDemOffset

from StackPulic import loadProduct
from StackPulic import createObject

def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='estimate offset between radar and dem')
    parser.add_argument('-track', dest='track', type=str, required=True,
            help = 'track parameter file')
    parser.add_argument('-dem', dest='dem', type=str, required=True,
            help = 'dem used for geometrical coregistration')
    parser.add_argument('-wbd', dest='wbd', type=str, required=True,
            help = 'water body in radar coordinate')
    parser.add_argument('-hgt', dest='hgt', type=str, required=True,
            help = 'height in radar coordinate computed in geometrical coregistration')
    parser.add_argument('-amp', dest='amp', type=str, required=True,
            help = 'amplitude image')
    parser.add_argument('-output', dest='output', type=str, required=True,
            help = 'output file for saving the affine transformation paramters')
    parser.add_argument('-nrlks1', dest='nrlks1', type=int, default=1,
            help = 'number of range looks 1. default: 1')
    parser.add_argument('-nalks1', dest='nalks1', type=int, default=1,
            help = 'number of azimuth looks 1. default: 1')
    parser.add_argument('-nrlks_sim', dest='nrlks_sim', type=int, default=None,
            help = 'number of range looks when simulating radar image')
    parser.add_argument('-nalks_sim', dest='nalks_sim', type=int, default=None,
            help = 'number of azimuth looks when simulating radar image')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()


    #get user parameters from input
    trackParameter = inps.track
    demFile = inps.dem
    wbdOut = inps.wbd
    height = inps.hgt
    amplitude = inps.amp
    output = inps.output
    numberRangeLooks1 = inps.nrlks1
    numberAzimuthLooks1 = inps.nalks1
    numberRangeLooksSim = inps.nrlks_sim
    numberAzimuthLooksSim = inps.nalks_sim
    #######################################################

    #prepare amplitude image
    insarDir = 'insar'
    os.makedirs(insarDir, exist_ok=True)
    os.chdir(insarDir)
    if not os.path.isfile(os.path.basename(amplitude)):
        os.symlink(os.path.join('../', amplitude), os.path.basename(amplitude))
    if not os.path.isfile(os.path.basename(amplitude)+'.vrt'):
        os.symlink(os.path.join('../', amplitude)+'.vrt', os.path.basename(amplitude)+'.vrt')
    if not os.path.isfile(os.path.basename(amplitude)+'.xml'):
        os.symlink(os.path.join('../', amplitude)+'.xml', os.path.basename(amplitude)+'.xml')
    os.chdir('../')


    ml1 = '_{}rlks_{}alks'.format(numberRangeLooks1, numberAzimuthLooks1)
    simFile = 'radar_{}.sim'.format(ml1)

    self = createObject()
    self._insar = createObject()

    self._insar.dem = demFile
    self._insar.numberRangeLooksSim = numberRangeLooksSim
    self._insar.numberRangeLooks1 = numberRangeLooks1
    self._insar.numberAzimuthLooksSim = numberAzimuthLooksSim
    self._insar.numberAzimuthLooks1 = numberAzimuthLooks1
    self._insar.height = os.path.basename(height)
    self._insar.sim = simFile
    self._insar.amplitude = os.path.basename(amplitude)
    self._insar.wbdOut = os.path.basename(wbdOut)
    self._insar.radarDemAffineTransform = None

    referenceTrack = loadProduct(trackParameter)
    rdrDemOffset(self, referenceTrack, catalog=None)

    os.chdir(insarDir)
    #save the result
    with open(output, 'w') as f:
        f.write('{} {}\n{}'.format(self._insar.numberRangeLooksSim, self._insar.numberAzimuthLooksSim, self._insar.radarDemAffineTransform))

    #remove amplitude image
    os.remove(os.path.basename(amplitude))
    os.remove(os.path.basename(amplitude)+'.vrt')
    os.remove(os.path.basename(amplitude)+'.xml')
    os.chdir('../')