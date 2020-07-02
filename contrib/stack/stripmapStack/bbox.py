#!/usr/bin/env python3

import numpy as np 
import argparse
import os
import isce
import isceobj
from isceobj.Planet.Planet import Planet
import datetime
import shelve
from isceobj.Constants import SPEED_OF_LIGHT
from isceobj.Util.Poly2D import Poly2D 


def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser( description='Generate offset field between two Sentinel swaths')
    parser.add_argument('-i', type=str, dest='reference', required=True,
            help='Directory with the reference image')
    parser.add_argument('-n', action='store_true', dest='isnative', default=False,
            help='If product is native doppler')
    parser.add_argument('-m', type=float, dest='margin', default=0.05,
            help='Margin in degrees')
    return parser.parse_args()


if __name__ == '__main__':
    '''
    Generate offset fields burst by burst.
    '''

    inps = cmdLineParse()

    try:
        mdb = shelve.open( os.path.join(inps.reference, 'data'), flag='r')

    except:
        print('SLC not found ... trying RAW data')
        mdb = shelve.open( os.path.join(inps.reference, 'raw'), flag='r')
        inps.isnative = True

    frame = mdb['frame']

    lookSide = frame.instrument.platform.pointingDirection
    planet = Planet(pname='Earth')
    wvl = frame.instrument.getRadarWavelength()
    zrange = [-500., 9000.]


    if inps.isnative:
        ####If geometry is in native doppler / raw 
        ####You need doppler as a function of range to do
        ####geometry mapping correctly
        ###Currently doppler is saved as function of pixel number - old ROIPAC style
        ###Transform to function of slant range
        coeff = frame._dopplerVsPixel
        doppler = Poly2D()
        doppler._meanRange = frame.startingRange
        doppler._normRange = frame.instrument.rangePixelSize
        doppler.initPoly(azimuthOrder=0, rangeOrder=len(coeff)-1, coeffs=[coeff])
    else:
        ###Zero doppler system
        doppler = Poly2D()
        doppler.initPoly(azimuthOrder=0, rangeOrder=0, coeffs=[[0.]])

         
    
    llh = []
    for z in zrange:
        for taz in [frame.sensingStart, frame.sensingMid, frame.sensingStop]:
            for rng in [frame.startingRange, frame.getFarRange()]:
                pt = frame.orbit.rdr2geo(taz, rng, doppler=doppler, height=z,
                                        wvl=wvl, side=lookSide)
                llh.append(pt)

    llh = np.array(llh)
    minLat = np.min(llh[:,0]) - inps.margin 
    maxLat = np.max(llh[:,0]) + inps.margin
    minLon = np.min(llh[:,1]) - inps.margin
    maxLon = np.max(llh[:,1]) + inps.margin

    print('Lat limits: ', minLat, maxLat)
    print('Lon limits: ', minLon, maxLon)
