#!/usr/bin/env python3

import isce
from isceobj.Sensor import createSensor
import shelve
import argparse
import os
from isceobj.Util import Poly1D
from isceobj.Planet.AstronomicalHandbook import Const
from mroipac.dopiq.DopIQ import DopIQ
import copy


def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Unpack RISAT raw data and store metadata in pickle file.')
    parser.add_argument('-i','--input', dest='indir', type=str,
            required=True, help='Input CSK frame')
    parser.add_argument('-o', '--output', dest='slc', type=str,
            required=True, help='Output SLC file')
    parser.add_argument('-p', '--polar', dest='polar', type=str,
            default='RH', help='Polarization to extract')

    return parser.parse_args()


def unpack(hdf5, slcname, polar='RH'):
    '''
    Unpack HDF5 to binary SLC file.
    '''

    obj = createSensor('RISAT1')
    obj._imageFile = os.path.join(hdf5, 'scene_'+polar, 'dat_01.001')
    obj._leaderFile = os.path.join(hdf5, 'scene_'+polar,'lea_01.001')
    if not os.path.isdir(slcname):
        os.mkdir(slcname)

    date = os.path.basename(slcname)
    obj.output = os.path.join(slcname, date + '.raw')

    obj.extractImage()

    obj.frame.getImage().renderHdr()

    #####Estimate doppler
    dop = DopIQ()
    dop.configure()

    img = copy.deepcopy(obj.frame.getImage())
    img.setAccessMode('READ')

    dop.wireInputPort('frame', object=obj.frame)
    dop.wireInputPort('instrument', object=obj.frame.instrument)
    dop.wireInputPort('image', object=img)
    dop.calculateDoppler()
    dop.fitDoppler()
    fit = dop.quadratic
    coef = [fit['a'], fit['b'], fit['c']]

    print(coef)
    obj.frame._dopplerVsPixel = [x*obj.frame.PRF for x in coef]

    pickName = os.path.join(slcname, 'raw')
    with shelve.open(pickName) as db:
        db['frame'] = obj.frame

if __name__ == '__main__':
    '''
    Main driver.
    '''

    inps = cmdLineParse()
    unpack(inps.indir, inps.slc, polar=inps.polar)
