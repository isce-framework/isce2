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


    ######Numpy polynomial manipulation
    pc = obj._dopplerCoeffs[::-1]
    
    inds = np.linspace(0, obj.frame.numberOfSamples-1, len(pc) + 1)+1
    rng = obj.frame.getStartingRange() + inds * obj.frame.instrument.getRangePixelSize()
    dops = np.polyval(pc, 2*rng/Const.c-obj._dopplerTime)

    print('Near range doppler: ', dops[0])
    print('Far range doppler: ', dops[-1])
   
    dopfit = np.polyfit(inds, dops, len(pc)-1)
    
    poly = Poly1D.Poly1D()
    poly.initPoly(order=len(pc)-1)
    poly.setCoeffs(dopfit[::-1])

    print('Poly near range doppler: ', poly(1))
    print('Poly far range doppler: ', poly(obj.frame.numberOfSamples))

#    width = obj.frame.getImage().getWidth()
#    midrange = r0 + 0.5 * width * dr
#    dt = datetime.timedelta(seconds = midrange / Const.c)

#    obj.frame.sensingStart = obj.frame.sensingStart - dt
#    obj.frame.sensingStop = obj.frame.sensingStop - dt
#    obj.frame.sensingMid = obj.frame.sensingMid - dt


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
