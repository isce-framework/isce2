#!/usr/bin/env python3

import isce
from isceobj.Sensor import createSensor
import shelve
import argparse
import os
from isceobj.Util import Poly1D
from isceobj.Planet.AstronomicalHandbook import Const

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Unpack RISAT SLC data and store metadata in pickle file.')
    parser.add_argument('-i','--input', dest='indir', type=str,
            required=True, help='Input CSK frame')
    parser.add_argument('-o', '--output', dest='slc', type=str,
            required=True, help='Output SLC file')
    parser.add_argument('-p', '--polar', dest='polar', type=str,
            default='RH', help='Polarization to extract')
    
    parser.add_argument('-f', '--float', dest='isfloat', action='store_true',
            default = False, help='If float SLC is provided') 

    return parser.parse_args()


def unpack(hdf5, slcname, polar='RH', isfloat=False):
    '''
    Unpack HDF5 to binary SLC file.
    '''

    obj = createSensor('RISAT1_SLC')
    obj._imageFile = os.path.join(hdf5, 'scene_'+polar, 'dat_01.001')
    obj._leaderFile = os.path.join(hdf5, 'scene_'+polar,'lea_01.001')

    if isfloat:
        obj._dataType = 'float'
    else:
        obj._dataType = 'short'

    if not os.path.isdir(slcname):
        os.mkdir(slcname)

    date = os.path.basename(slcname)
    obj.output = os.path.join(slcname, date + '.slc')

    obj.extractImage()
    obj.frame.getImage().renderHdr()


    coeffs = obj.doppler_coeff
    dr = obj.frame.getInstrument().getRangePixelSize()
    r0 = obj.frame.getStartingRange()


    poly = Poly1D.Poly1D()
    poly.initPoly(order=len(coeffs)-1)
    poly.setCoeffs(coeffs)


    fcoeffs = obj.azfmrate_coeff
    fpoly = Poly1D.Poly1D()
    fpoly.initPoly(order=len(fcoeffs)-1)
    fpoly.setCoeffs(fcoeffs)

    pickName = os.path.join(slcname, 'data')
    with shelve.open(pickName) as db:
        db['frame'] = obj.frame
        db['doppler'] = poly
        db['fmrate'] = fpoly

    print(poly._coeffs)
    print(fpoly._coeffs)

if __name__ == '__main__':
    '''
    Main driver.
    '''

    inps = cmdLineParse()
    unpack(inps.indir, inps.slc, polar=inps.polar, isfloat=inps.isfloat)
