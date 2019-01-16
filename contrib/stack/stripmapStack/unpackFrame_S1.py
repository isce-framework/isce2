#!/usr/bin/env python3

import isce
from isceobj.Sensor import createSensor
import shelve
import argparse
import glob
from isceobj.Util import Poly1D
from isceobj.Planet.AstronomicalHandbook import Const
import os
import numpy as np

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Unpack CSK SLC data and store metadata in pickle file.')
    parser.add_argument('-i','--input', dest='safe', type=str,
            required=True, help='Input SAFE directory/zip')
    parser.add_argument('-o', '--output', dest='slcdir', type=str,
            required=True, help='Output SLC directory')
    parser.add_argument('-p', '--pol', dest='polarization', type=str,
            default='vv', help='Polarization')
    parser.add_argument('-b', '--orbdir', dest='orbdir', type=str,
            required=True, help='Orbit directory')

    return parser.parse_args()


def unpack(hdf5, slcname, pol, orbdir):
    '''
    Unpack SAFE to binary SLC file.
    '''

    if not os.path.isdir(slcname):
        os.mkdir(slcname)

    date = os.path.basename(slcname)

    obj = createSensor('SENTINEL1')
    obj.safe = hdf5
    obj.polarization = pol
    obj.orbitDir=orbdir
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

    if inps.safe.endswith('/'):
        inps.safe = inps.h5dir[:-1]

    unpack(inps.safe, inps.slcdir, inps.polarization, inps.orbdir)
