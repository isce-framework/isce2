#!/usr/bin/env python3

import isce
from isceobj.Sensor import createSensor
import shelve
import argparse
import glob
from isceobj.Util import Poly1D
from isceobj.Planet.AstronomicalHandbook import Const
import os
from mroipac.dopiq.DopIQ import DopIQ
import copy

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Unpack raw data and store metadata in pickle file.')
    parser.add_argument('-i','--input', dest='rawfile', type=str,
            required=True, help='Input ROI_PAC file')
    parser.add_argument('-r','--hdr', dest='hdrfile', type=str,
            required=True, help='Input hdr (orbit) file')
    parser.add_argument('-o', '--output', dest='slcdir', type=str,
            required=True, help='Output data directory')

    return parser.parse_args()


def unpack(rawname, hdrname, slcname):
    '''
    Unpack raw to binary file.
    '''

    if not os.path.isdir(slcname):
        os.mkdir(slcname)

    date = os.path.basename(slcname)
    obj = createSensor('ROI_PAC')
    obj.configure()
    obj._rawFile = rawname
    obj._hdrFile = hdrname
    obj.output = os.path.join(slcname, date+'.raw')

    print(obj._rawFile)
    print(obj._hdrFile)
    print(obj.output)
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
    if inps.slcdir.endswith('/'):
        inps.slcdir = inps.slcdir[:-1]

    unpack(inps.rawfile, inps.hdrfile, inps.slcdir)
