#!/usr/bin/env python3

# Author: Cunren Liang
# Copyright 2021

import os
import copy
import argparse
import numpy as np

import isce
import isceobj
from isceobj.Constants import SPEED_OF_LIGHT
from isceobj.TopsProc.runIon import computeIonosphere
from isceobj.Alos2Proc.runIonFilt import reformatMaskedAreas

from Stack import ionParam


def createParser():
    parser = argparse.ArgumentParser(description='compute ionosphere using lower and upper band interferograms')
    parser.add_argument('-l', '--lower', dest='lower', type=str, required=True,
            help='lower band interferogram')
    parser.add_argument('-u', '--upper', dest='upper', type=str, required=True,
            help='upper band interferogram')
    parser.add_argument('-c', '--coherence', dest='coherence', type=str, required=True,
            help='input coherence')
    parser.add_argument('-i', '--ionosphere', dest='ionosphere', type=str, required=True,
            help='output ionosphere')
    parser.add_argument('-o', '--coherence_output', dest='coherence_output', type=str, required=True,
            help='output coherence file name. simply copy input coherence')
    parser.add_argument('-m', '--masked_areas', dest='masked_areas', type=int, nargs='+', action='append', default=None,
            help='This is a 2-d list. Each element in the 2-D list is a four-element list: [firstLine, lastLine, firstColumn, lastColumn], with line/column numbers starting with 1. If one of the four elements is specified with -1, the program will use firstLine/lastLine/firstColumn/lastColumn instead. e.g. two areas masked out: --masked_areas 10 20 10 20 --masked_areas 110 120 110 120')
    #parser.add_argument('-m', '--masked_areas', dest='masked_areas', type=int, nargs='+', default=None,
    #        help='This is a 2-d list. Each element in the 2-D list is a four-element list: [firstLine, lastLine, firstColumn, lastColumn], with line/column numbers starting with 1. If one of the four elements is specified with -1, the program will use firstLine/lastLine/firstColumn/lastColumn instead. e.g. two areas masked out: --masked_areas 10 20 10 20 110 120 110 120')

    return parser


def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def main(iargs=None):
    '''
    '''
    inps = cmdLineParse(iargs)

    # #convert 1-d list to 2-d list
    # if len(inps.masked_areas) % 4 != 0:
    #     raise Exception('each maksed area must have four elements')
    # else:
    #     masked_areas = []
    #     n = np.int32(len(inps.masked_areas)/4)
    #     for i in range(n):
    #         masked_areas.append([inps.masked_areas[i*4+0], inps.masked_areas[i*4+1], inps.masked_areas[i*4+2], inps.masked_areas[i*4+3]])
    #     inps.masked_areas = masked_areas

    ###################################
    #SET PARAMETERS HERE
    #THESE SHOULD BE GOOD ENOUGH, NO NEED TO SET IN setup(self)
    corThresholdAdj = 0.85
    ###################################

    print('computing ionosphere')
    #get files
    lowerUnwfile = inps.lower
    upperUnwfile = inps.upper
    corfile = inps.coherence

    #use image size from lower unwrapped interferogram
    img = isceobj.createImage()
    img.load(lowerUnwfile + '.xml')
    width = img.width
    length = img.length

    lowerUnw = (np.fromfile(lowerUnwfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    upperUnw = (np.fromfile(upperUnwfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    #lowerAmp = (np.fromfile(lowerUnwfile, dtype=np.float32).reshape(length*2, width))[0:length*2:2, :]
    #upperAmp = (np.fromfile(upperUnwfile, dtype=np.float32).reshape(length*2, width))[0:length*2:2, :]
    cor = (np.fromfile(corfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    #amp = np.sqrt(lowerAmp**2+upperAmp**2)
    amp = (np.fromfile(corfile, dtype=np.float32).reshape(length*2, width))[0:length*2:2, :]

    #masked out user-specified areas
    if inps.masked_areas != None:
        maskedAreas = reformatMaskedAreas(inps.masked_areas, length, width)
        for area in maskedAreas:
            lowerUnw[area[0]:area[1], area[2]:area[3]] = 0
            upperUnw[area[0]:area[1], area[2]:area[3]] = 0
            cor[area[0]:area[1], area[2]:area[3]] = 0

    ionParamObj=ionParam()
    ionParamObj.configure()

    #compute ionosphere
    fl = SPEED_OF_LIGHT / ionParamObj.radarWavelengthLower
    fu = SPEED_OF_LIGHT / ionParamObj.radarWavelengthUpper
    adjFlag = 1
    ionos = computeIonosphere(lowerUnw, upperUnw, cor, fl, fu, adjFlag, corThresholdAdj, 0)

    #dump ionosphere
    outFilename = inps.ionosphere
    os.makedirs(os.path.dirname(inps.ionosphere), exist_ok=True)

    ion = np.zeros((length*2, width), dtype=np.float32)
    ion[0:length*2:2, :] = amp
    ion[1:length*2:2, :] = ionos
    ion.astype(np.float32).tofile(outFilename)
    img.filename = outFilename
    img.extraFilename = outFilename + '.vrt'
    img.renderHdr()

    #dump coherence
    outFilename = inps.coherence_output
    os.makedirs(os.path.dirname(inps.coherence_output), exist_ok=True)

    ion[1:length*2:2, :] = cor
    ion.astype(np.float32).tofile(outFilename)
    img.filename = outFilename
    img.extraFilename = outFilename + '.vrt'
    img.renderHdr()



if __name__ == '__main__':
    '''
    Main driver.
    '''
    # Main Driver
    main()



