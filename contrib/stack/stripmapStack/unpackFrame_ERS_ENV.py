#!/usr/bin/env python3

import isce
from isceobj.Sensor import createSensor
import shelve
import argparse
import glob
from isceobj.Util import Poly1D
from isceobj.Planet.AstronomicalHandbook import Const
import os
import datetime
import numpy as np

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Unpack ERS(ESA) SLC data and store metadata in pickle file.')
    parser.add_argument('-i','--input', dest='datadir', type=str,
            required=True, help='Input ERS files path')
    parser.add_argument('-o', '--output', dest='slcdir', type=str,
            required=True, help='Output SLC directory')
    parser.add_argument('--orbitdir', dest='orbitdir', type=str,required=True, help='Orbit directory')

    return parser.parse_args()

def get_Date(file):
    yyyymmdd=file[14:22]
    return yyyymmdd

def unpack(fname, slcname, orbitdir):
    '''
    Unpack .E* file to binary SLC file.
    '''

    obj = createSensor('ERS_ENviSAT_SLC')
    obj._imageFileName = fname
    obj._orbitDir = orbitdir
    obj._orbitType = 'ODR'
    #obj.instrumentDir = '/Users/agram/orbit/INS_DIR'
    obj.output = os.path.join(slcname,os.path.basename(slcname)+'.slc')
    obj.extractImage()
    obj.frame.getImage().renderHdr()

    ####  computation of "poly" adapted from line 339 - line 353 of  Components/isceobj/Sensor/ERS_EnviSAT_SLC.py ###
    coeffs = obj.dopplerRangeTime
    dr = obj.frame.getInstrument().getRangePixelSize()
    rref = 0.5 * Const.c * obj.rangeRefTime
    r0 = obj.frame.getStartingRange()
    norm = 0.5 * Const.c / dr

    dcoeffs = []
    for ind, val in enumerate(coeffs):
        dcoeffs.append( val / (norm**ind))
    
    poly = Poly1D.Poly1D()
    poly.initPoly(order=len(coeffs)-1)
    poly.setMean( (rref - r0)/dr - 1.0)
    poly.setCoeffs(dcoeffs)


    pickName = os.path.join(slcname, 'data')
    with shelve.open(pickName) as db:
        db['frame'] = obj.frame
        db['doppler'] = poly 


if __name__ == '__main__':
    '''
    Main driver.
    '''

    inps = cmdLineParse()
    if inps.slcdir.endswith('/'):
        inps.slcdir = inps.slcdir[:-1]
    if not os.path.isdir(inps.slcdir):
        os.mkdir(inps.slcdir)
    for fname in glob.glob(os.path.join(inps.datadir, '*.E*')):
        date = get_Date(os.path.basename(fname))
        slcname = os.path.join(inps.slcdir, date)
        if not os.path.isdir(slcname):
            os.mkdir(slcname)
        unpack(fname, slcname, inps.orbitdir)
