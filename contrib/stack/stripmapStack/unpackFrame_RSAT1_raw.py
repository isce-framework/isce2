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
    parser.add_argument('-o', '--output', dest='slcdir', type=str,
            required=True, help='Output SLC directory')

    return parser.parse_args()


def unpack(hdf5, slcname):
    '''
    Unpack HDF5 to binary SLC file.
    '''

    try:
        imgname = glob.glob(os.path.join(hdf5, '*.raw'))[0]
    except:
        try:
            imgname = glob.glob(os.path.join(hdf5, 'DAT*'))[0]
        except:
            raise Exception('Cant find .raw of DAT. file in dir')

    try:
        ldrname = glob.glob(os.path.join(hdf5, '*.ldr'))[0]
    except:
        try:
            ldrname = glob.glob(os.path.join(hdf5, 'LEA*'))[0]
        except:
            raise Exception('Cant find .ldr or LEA. file in dir')

    parname = glob.glob(os.path.join(hdf5, '*.par'))[0]

    if not os.path.isdir(slcname):
        os.mkdir(slcname)

    date = os.path.basename(slcname)
    obj = createSensor('RADARSAT1')
    obj.configure()
    obj._leaderFile = ldrname
    obj._imageFile = imgname
    obj._parFile = parname
    obj.output = os.path.join(slcname, date+'.raw')

    print(obj._leaderFile)
    print(obj._imageFile)
    print(obj._parFile)
    print(obj.output)
    obj.extractImage()
    obj.frame.getImage().renderHdr()
    obj.extractDoppler()

    pickName = os.path.join(slcname, 'raw')
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
