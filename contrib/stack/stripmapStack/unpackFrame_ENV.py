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

    parser = argparse.ArgumentParser(description='Unpack Envisat SLC data and store metadata in pickle file.')
    parser.add_argument('-i','--input', dest='h5dir', type=str,
            required=True, help='Input Envisat directory')
    parser.add_argument('-o', '--output', dest='slcdir', type=str,
            required=True, help='Output SLC directory')

    return parser.parse_args()


def unpack(hdf5, slcname):
    '''
    Unpack HDF5 to binary SLC file.
    '''

    fname = glob.glob(os.path.join(hdf5,'ASA*.N1'))[0]
    if not os.path.isdir(slcname):
        os.mkdir(slcname)

    date = os.path.basename(slcname)

    obj = createSensor('ENVISAT_SLC')
    obj._imageFileName = fname
    obj.orbitDir = '/Users/agram/orbit/VOR'
    obj.instrumentDir = '/Users/agram/orbit/INS_DIR'
    obj.output = os.path.join(slcname, date+'.slc')

    obj.extractImage()
    obj.frame.getImage().renderHdr()


    obj.extractDoppler()

    pickName = os.path.join(slcname, 'data')
    with shelve.open(pickName) as db:
        db['frame'] = obj.frame


if __name__ == '__main__':
    '''
    Main driver.
    '''

    inps = cmdLineParse()
    if inps.slcdir.endswith('/'):
        inps.slcdir = inps.slcdir[:-1]

    if inps.h5dir.endswith('/'):
        inps.h5dir = inps.h5dir[:-1]

    unpack(inps.h5dir, inps.slcdir)
