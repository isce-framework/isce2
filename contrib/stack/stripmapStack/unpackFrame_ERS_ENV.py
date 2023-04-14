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
            required=True, help='Input ERS directory')
    parser.add_argument('-o', '--output', dest='slcdir', type=str,
            required=True, help='Output SLC directory')
    parser.add_argument('--orbitdir', dest='orbitdir', type=str,required=True, help='Orbit directory')

    return parser.parse_args()


def unpack(hdf5, slcname, orbitdir):
    '''
    Unpack HDF5 to binary SLC file.
    '''

    fname = glob.glob(os.path.join(hdf5,'SAR*.E*'))[0]
    if not os.path.isdir(slcname):
        os.mkdir(slcname)

    date = os.path.basename(slcname)

    obj = createSensor('ERS_ENviSAT_SLC')
    obj._imageFileName = fname
    obj.orbitDir = orbitdir
    #obj.instrumentDir = '/Users/agram/orbit/INS_DIR'
    obj.output = os.path.join(slcname, date+'.slc')

    obj.extractImage()
    obj.frame.getImage().renderHdr()

    ####  computation of "poly" adapted from line 339 - line 353 of isceobj/Sensor/ERS_EnviSAT_SLC.py ###
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

    if inps.datadir.endswith('/'):
        inps.datadir = inps.datadir[:-1]

    unpack(inps.datadir, inps.slcdir, inps.orbitdir)
