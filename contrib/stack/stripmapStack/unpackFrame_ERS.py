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

    imgname = glob.glob(os.path.join(hdf5,'DAT*'))[0]
    ldrname = glob.glob(os.path.join(hdf5, 'LEA*'))[0]
    if not os.path.isdir(slcname):
        os.mkdir(slcname)

    date = os.path.basename(slcname)
    obj = createSensor('ERS_SLC')
    obj.configure()
    obj._leaderFile = ldrname
    obj._imageFile = imgname
    obj._orbitType = 'ODR'
    obj._orbitDir = '/Users/agram/orbit/ODR/ERS2'
    obj.output = os.path.join(slcname, date+'.slc')

    print(obj._leaderFile)
    print(obj._imageFile)
    print(obj.output)
    obj.extractImage()
    obj.frame.getImage().renderHdr()


    coeffs = obj.doppler_coeff
#    coeffs = [0.,0.]
    dr = obj.frame.getInstrument().getRangePixelSize()
    r0 = obj.frame.getStartingRange()

    print(coeffs)
    poly = Poly1D.Poly1D()
    poly.initPoly(order=len(coeffs)-1)
    poly.setCoeffs(coeffs)


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

    if inps.h5dir.endswith('/'):
        inps.h5dir = inps.h5dir[:-1]

    unpack(inps.h5dir, inps.slcdir)
