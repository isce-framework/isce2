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
import pprint

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Unpack ENV raw data and store metadata in shelf file.')
    parser.add_argument('-i','--input', dest='h5dir', type=str,
            required=True, help='Input CSK directory')
    parser.add_argument('-o', '--output', dest='slcdir', type=str,
            required=True, help='Output SLC directory')
    parser.add_argument('-m', '--mult', dest='multiple', action='store_true',
            default=False, help='Stitch multiple frames together')
    return parser.parse_args()


def unpack(hdf5, slcname, multiple=False):
    '''
    Unpack HDF5 to binary SLC file.
    '''

    if multiple:
        print('Trying multiple sub-dirs ....')
        imgname = glob.glob( os.path.join(hdf5, '*', 'ASA*'))

        if len(imgname) == 0:
            print('No ASA files found in sub-dirs. Trying same dir ...')
            imgname = glob.glob(os.path.join(hdf5, 'ASA*'))
    else:
        imgname = [glob.glob(os.path.join(hdf5,'ASA*'))[0]]

    if not os.path.isdir(slcname):
        os.mkdir(slcname)

    date = os.path.basename(slcname)
    obj = createSensor('ENVISAT')
    obj.configure()
    obj._imageryFileList = imgname
    obj.instrumentDir = '/Users/agram/orbit/INS_DIR'
    obj.orbitDir = '/Users/agram/orbit/VOR'
    obj.output = os.path.join(slcname, date+'.raw')

    obj.extractImage()
    obj.frame.getImage().renderHdr()

#    print('Beam number: ', obj._imageryFileData['antennaBeamSetNumber']-1)
#    pprint.pprint(obj._instrumentFileData)

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

    if inps.h5dir.endswith('/'):
        inps.h5dir = inps.h5dir[:-1]

    unpack(inps.h5dir, inps.slcdir, multiple=inps.multiple)
