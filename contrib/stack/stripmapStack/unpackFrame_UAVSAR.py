#!/usr/bin/env python3

import isce
from isceobj.Sensor import createSensor
import shelve
import argparse
import glob
from isceobj.Util import Poly1D
from isceobj.Planet.AstronomicalHandbook import Const
import os

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Unpack CSK SLC data and store metadata in pickle file.')
    parser.add_argument('-i','--input', dest='h5dir', type=str,
            required=True, help='Input CSK directory')
    parser.add_argument('-d','--dop_file', dest='dopFile', type=str,
            default=None, help='Doppler file')
    parser.add_argument('-o', '--output', dest='slcdir', type=str,
            required=True, help='Output SLC directory')
    return parser.parse_args()


def unpack(hdf5, slcname, dopFile, parse=False):
    '''
    Unpack HDF5 to binary SLC file.
    '''

    obj = createSensor('UAVSAR_STACK')
    obj.configure()
    obj.metadataFile = hdf5
    obj.dopplerFile = dopFile
    obj.parse()

    if not os.path.isdir(slcname):
        os.mkdir(slcname)

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

    unpack(inps.h5dir, inps.slcdir, inps.dopFile)
