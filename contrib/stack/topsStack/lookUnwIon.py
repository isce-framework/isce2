#!/usr/bin/env python3

# Author: Cunren Liang
# Copyright 2021

import os
import copy
import argparse
import numpy as np

import isce
import isceobj
from isceobj.TopsProc.runIon import multilook


def createParser():
    parser = argparse.ArgumentParser(description='multilook unwrapped interferograms')
    parser.add_argument('-u', '--unw', dest='unw', type=str, required=True,
            help='input unwrapped interferogram')
    parser.add_argument('-c', '--cor', dest='cor', type=str, required=True,
            help='input coherence')
    parser.add_argument('-o', '--output', dest='output', type=str, required=True,
            help='output multi-look unwrapped interferogram')
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

    nrlks = inps.nrlks
    nalks = inps.nalks

    if (nrlks == 1) and (nalks == 1):
        img = isceobj.createImage()
        img.load(inps.unw + '.xml')
        img.setFilename(inps.output)
        img.extraFilename = inps.output+'.vrt'
        img.renderHdr()

        os.symlink(os.path.abspath(inps.unw), os.path.abspath(inps.output))
    else:
        #use coherence to compute weight
        corName0 = inps.cor
        corimg = isceobj.createImage()
        corimg.load(corName0 + '.xml')
        width = corimg.width
        length = corimg.length
        widthNew = int(width / nrlks)
        lengthNew = int(length / nalks)
        cor0 = (np.fromfile(corName0, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
        wgt = cor0**2
        a = multilook(wgt, nalks, nrlks)
        d = multilook((cor0!=0).astype(int), nalks, nrlks)

        #unwrapped file
        unwrapName0 = inps.unw
        unwimg = isceobj.createImage()
        unwimg.load(unwrapName0 + '.xml')
        unw0 = (np.fromfile(unwrapName0, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
        amp0 = (np.fromfile(unwrapName0, dtype=np.float32).reshape(length*2, width))[0:length*2:2, :]
        e = multilook(unw0*wgt, nalks, nrlks)
        f = multilook(amp0**2, nalks, nrlks)
        unw = np.zeros((lengthNew*2, widthNew), dtype=np.float32)
        unw[0:lengthNew*2:2, :] = np.sqrt(f / (d + (d==0)))
        unw[1:lengthNew*2:2, :] = e / (a + (a==0))

        #output file
        os.makedirs(os.path.dirname(inps.output), exist_ok=True)
        unwrapName = inps.output
        unw.astype(np.float32).tofile(unwrapName)
        unwimg.setFilename(unwrapName)
        unwimg.extraFilename = unwrapName + '.vrt'
        unwimg.setWidth(widthNew)
        unwimg.setLength(lengthNew)
        unwimg.renderHdr()


if __name__ == '__main__':
    '''
    Main driver.
    '''
    # Main Driver
    main()



