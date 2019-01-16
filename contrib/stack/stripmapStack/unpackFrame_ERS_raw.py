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

    parser = argparse.ArgumentParser(description='Unpack CSK SLC data and store metadata in pickle file.')
    parser.add_argument('-i','--input', dest='h5dir', type=str,
            required=True, help='Input CSK directory')
    parser.add_argument('-o', '--output', dest='slcdir', type=str,
            required=True, help='Output SLC directory')
    parser.add_argument('-m', '--mult', dest='multiple', action='store_true',
            default=False, help='If stitching multiple frames')
    parser.add_Argument('-t', '--type', dest='orbtype', type=str, default='PRC',
            help='Orbit Type - PRC or ODR')

    return parser.parse_args()


def unpack(hdf5, slcname,multiple=False,orbtype='PRC'):
    '''
    Unpack HDF5 to binary SLC file.
    '''

    if multiple:
        print('Trying multiple sub-dirs - ESA convention ...')
        imgname = glob.glob(os.path.join(hdf5, '*', 'DAT*'))
        ldrname = glob.glob(os.path.join(hdf5, '*', 'LEA*'))

        if (len(imgname)==0) or (len(ldrname) == 0):
            print('Did not find ESA style files in sub-dirs. Trying RPAC style in sub-dirs ....')
            imgname = glob.glob(os.path.join(hdf5, '*', 'IMA*'))
            ldrname = glob.glob(os.path.join(hdf5, '*', 'SAR*'))

            if (len(imgname)==0) or (len(ldrname) == 0):
                print('Did not find RPAC style files in sub-dirs. Trying RPAC style in same-dir ....')
                imgname = glob.glob(os.path.join(hdf5, 'IMA*'))
                ldrname = glob.glob(os.path.join(hdf5, 'SAR*'))

    else:
        imgname = [glob.glob(os.path.join(hdf5,'DAT*'))[0]]
        ldrname = [glob.glob(os.path.join(hdf5,'LEA*'))[0]]

    if not os.path.isdir(slcname):
        os.mkdir(slcname)

    date = os.path.basename(slcname)
    obj = createSensor('ERS')
    obj.configure()
    obj._imageFileList = imgname
    obj._leaderFileList = ldrname

    ####Need to figure out 1/2 automatically also
    if orbtype == 'ODR':
        obj._orbitType = 'ODR'
        obj._orbitDir = '/Users/agram/orbit/ODR/ERS1'

    if orbtype == 'PRC':
        obj._orbitType = 'PRC'
        obj._orbitDir = '/Users/agram/orbit/PRC/ERS1'

    obj.output = os.path.join(slcname, date+'.raw')

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

    if inps.h5dir.endswith('/'):
        inps.h5dir = inps.h5dir[:-1]

    unpack(inps.h5dir, inps.slcdir,
            multiple=inps.multiple, orbtype=inps.orbtype)
