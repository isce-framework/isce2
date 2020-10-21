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
from contrib.alos2proc_f.alos2proc_f import rect_with_looks
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml

from StackPulic import createObject

def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='rectify range offset')
    parser.add_argument('-aff', dest='aff', type=str, required=True,
            help = 'affine transform paramter file')
    parser.add_argument('-input', dest='input', type=str, default='./',
            help = 'input file')
    parser.add_argument('-output', dest='output', type=str, required=True,
            help = 'output file')
    parser.add_argument('-nrlks1', dest='nrlks1', type=int, default=1,
            help = 'number of range looks 1 . default: 1')
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
    aff = inps.aff
    rangeOffset = inps.input
    rectRangeOffset = inps.output
    numberRangeLooks1 = inps.nrlks1
    numberAzimuthLooks1 = inps.nalks1
    #######################################################

    DEBUG=False

    self = createObject()
    self._insar = createObject()

    self._insar.rangeOffset = rangeOffset
    self._insar.rectRangeOffset = rectRangeOffset
    self._insar.numberRangeLooks1 = numberRangeLooks1
    self._insar.numberAzimuthLooks1 = numberAzimuthLooks1

    #read affine transform parameters
    with open(aff, 'r') as f:
        lines = f.readlines()
    self._insar.numberRangeLooksSim = int(lines[0].split()[0])
    self._insar.numberAzimuthLooksSim = int(lines[0].split()[1])
    self._insar.radarDemAffineTransform = [float(x) for x in lines[1].strip('[').strip(']').split(',')]
    if DEBUG:
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('{} {}\n{}'.format(self._insar.numberRangeLooksSim, self._insar.numberAzimuthLooksSim, self._insar.radarDemAffineTransform))
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    #rectify
    rgoff = isceobj.createImage()
    rgoff.load(self._insar.rangeOffset+'.xml')

    if self._insar.radarDemAffineTransform == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]:
        if not os.path.isfile(self._insar.rectRangeOffset):
            os.symlink(self._insar.rangeOffset, self._insar.rectRangeOffset)
            create_xml(self._insar.rectRangeOffset, rgoff.width, rgoff.length, 'float')
    else:
        rect_with_looks(self._insar.rangeOffset,
                        self._insar.rectRangeOffset,
                        rgoff.width, rgoff.length,
                        rgoff.width, rgoff.length,
                        self._insar.radarDemAffineTransform[0], self._insar.radarDemAffineTransform[1],
                        self._insar.radarDemAffineTransform[2], self._insar.radarDemAffineTransform[3],
                        self._insar.radarDemAffineTransform[4], self._insar.radarDemAffineTransform[5],
                        self._insar.numberRangeLooksSim*self._insar.numberRangeLooks1, self._insar.numberAzimuthLooksSim*self._insar.numberAzimuthLooks1,
                        self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1,
                        'REAL',
                        'Bilinear')
        create_xml(self._insar.rectRangeOffset, rgoff.width, rgoff.length, 'float')

