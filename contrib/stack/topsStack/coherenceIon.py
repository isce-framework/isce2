#!/usr/bin/env python3

# Author: Cunren Liang
# Copyright 2021

import os
import copy
import argparse
import numpy as np

import isce
import isceobj
from isceobj.TopsProc.runIon import cal_coherence
from isceobj.TopsProc.runIon import multilook


def createParser():
    parser = argparse.ArgumentParser(description='compute coherence using only differential interferograms')
    parser.add_argument('-l', '--lower', dest='lower', type=str, required=True,
            help='lower band interferogram')
    parser.add_argument('-u', '--upper', dest='upper', type=str, required=True,
            help='upper band interferogram')
    parser.add_argument('-c', '--coherence', dest='coherence', type=str, required=True,
            help='output coherence')
    parser.add_argument('-r', '--nrlks', dest='nrlks', type=int, default=1, 
            help='number of range looks. Default: 1')
    parser.add_argument('-a', '--nalks', dest='nalks', type=int, default=1, 
            help='number of azimuth looks. Default: 1')

    return parser


def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def main(iargs=None):
    '''
    '''
    inps = cmdLineParse(iargs)

    os.makedirs(os.path.dirname(inps.coherence), exist_ok=True)

    #The orginal coherence calculated by topsApp.py is not good at all, use the following coherence instead
    #lowerintfile = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, ionParam.mergedDirname, self._insar.mergedIfgname)
    #upperintfile = os.path.join(ionParam.ionDirname, ionParam.upperDirname, ionParam.mergedDirname, self._insar.mergedIfgname)
    #corfile = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, ionParam.mergedDirname, self._insar.correlationFilename)

    img = isceobj.createImage()
    img.load(inps.lower + '.xml')
    width = img.width
    length = img.length
    lowerint = np.fromfile(inps.lower, dtype=np.complex64).reshape(length, width)
    upperint = np.fromfile(inps.upper, dtype=np.complex64).reshape(length, width)

    if (inps.nrlks != 1) or (inps.nalks != 1):
        width = np.int(width/inps.nrlks)
        length = np.int(length/inps.nalks)
        lowerint = multilook(lowerint, inps.nalks, inps.nrlks)
        upperint = multilook(upperint, inps.nalks, inps.nrlks)

    #compute coherence only using interferogram
    #here I use differential interferogram of lower and upper band interferograms
    #so that coherence is not affected by fringes
    cord = cal_coherence(lowerint*np.conjugate(upperint), win=3, edge=4)
    cor = np.zeros((length*2, width), dtype=np.float32)
    cor[0:length*2:2, :] = np.sqrt( (np.absolute(lowerint)+np.absolute(upperint))/2.0 )
    cor[1:length*2:2, :] = cord
    cor.astype(np.float32).tofile(inps.coherence)

    #create xml and vrt
    #img.scheme = 'BIL'
    #img.bands = 2
    #img.filename = corfile
    #img.renderHdr()

    #img = isceobj.Image.createUnwImage()
    img = isceobj.createOffsetImage()
    img.setFilename(inps.coherence)
    img.extraFilename = inps.coherence + '.vrt'
    img.setWidth(width)
    img.setLength(length)
    img.renderHdr()


if __name__ == '__main__':
    '''
    Main driver.
    '''
    # Main Driver
    main()



