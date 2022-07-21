#!/usr/bin/env python3

# Author: Cunren Liang
# Copyright 2021

import os
import copy
import glob
import shutil
import argparse
import numpy as np

import isce
import isceobj
from isceobj.TopsProc.runIon import adaptive_gaussian
from isceobj.TopsProc.runIon import weight_fitting


def createParser():
    parser = argparse.ArgumentParser(description='filtering ionosphere')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
            help='input ionosphere')
    parser.add_argument('-c', '--coherence', dest='coherence', type=str, required=True,
            help='coherence')
    parser.add_argument('-o', '--output', dest='output', type=str, required=True,
            help='output ionosphere')
    parser.add_argument('-a', '--win_min', dest='win_min', type=int, default=100, 
            help='minimum filtering window size')
    parser.add_argument('-b', '--win_max', dest='win_max', type=int, default=200, 
            help='maximum filtering window size')
    #parser.add_argument('-m', '--masked_areas', dest='masked_areas', type=int, nargs='+', action='append', default=None,
    #        help='This is a 2-d list. Each element in the 2-D list is a four-element list: [firstLine, lastLine, firstColumn, lastColumn], with line/column numbers starting with 1. If one of the four elements is specified with -1, the program will use firstLine/lastLine/firstColumn/lastColumn instead. e.g. two areas masked out: --masked_areas 10 20 10 20 --masked_areas 110 120 110 120')

    return parser


def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def main(iargs=None):
    '''
    check overlap among all acquistions, only keep the bursts that in the common overlap,
    and then renumber the bursts.
    '''
    inps = cmdLineParse(iargs)

    '''
    This function filters image using gaussian filter

    we projected the ionosphere value onto the ionospheric layer, and the indexes are integers.
    this reduces the number of samples used in filtering
    a better method is to project the indexes onto the ionospheric layer. This way we have orginal
    number of samples used in filtering. but this requries more complicated operation in filtering
    currently not implemented.
    a less accurate method is to use ionsphere without any projection
    '''

    #################################################
    #SET PARAMETERS HERE
    #if applying polynomial fitting
    #False: no fitting, True: with fitting
    fit = True
    #gaussian filtering window size
    size_max = inps.win_max
    size_min = inps.win_min

    #THESE SHOULD BE GOOD ENOUGH, NO NEED TO SET IN setup(self)
    corThresholdIon = 0.85
    #################################################

    print('filtering ionosphere')
    #I find it's better to use ionosphere that is not projected, it's mostly slowlying changing anyway.
    #this should also be better for operational use.
    ionfile = inps.input
    #since I decide to use ionosphere that is not projected, I should also use coherence that is not projected.
    corfile = inps.coherence

    #use ionosphere and coherence that are projected.
    #ionfile = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname, ionParam.ionRaw)
    #corfile = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname, ionParam.ionCor)

    outfile = inps.output

    img = isceobj.createImage()
    img.load(ionfile + '.xml')
    width = img.width
    length = img.length
    ion = (np.fromfile(ionfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    cor = (np.fromfile(corfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    amp = (np.fromfile(ionfile, dtype=np.float32).reshape(length*2, width))[0:length*2:2, :]

    ########################################################################################
    #AFTER COHERENCE IS RESAMPLED AT grd2ion, THERE ARE SOME WIRED VALUES
    cor[np.nonzero(cor<0)] = 0.0
    cor[np.nonzero(cor>1)] = 0.0
    ########################################################################################

    ion_fit = weight_fitting(ion, cor, width, length, 1, 1, 1, 1, 2, corThresholdIon)

    #no fitting
    if fit == False:
        ion_fit *= 0

    ion -= ion_fit * (ion!=0)
    
    #minimize the effect of low coherence pixels
    #cor[np.nonzero( (cor<0.85)*(cor!=0) )] = 0.00001
    #filt = adaptive_gaussian(ion, cor, size_max, size_min)
    #cor**14 should be a good weight to use. 22-APR-2018
    filt = adaptive_gaussian(ion, cor**14, size_max, size_min)

    filt += ion_fit * (filt!=0)

    #do not mask now as there is interpolation after least squares estimation of each date ionosphere
    #filt *= (amp!=0)

    ion = np.zeros((length*2, width), dtype=np.float32)
    ion[0:length*2:2, :] = amp
    ion[1:length*2:2, :] = filt
    ion.astype(np.float32).tofile(outfile)
    img.filename = outfile
    img.extraFilename = outfile + '.vrt'
    img.renderHdr()




if __name__ == '__main__':
    '''
    Main driver.
    '''
    # Main Driver
    main()



