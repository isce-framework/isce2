#!/usr/bin/env python3

import isce
from isceobj.Sensor import createSensor
import isceobj
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

    parser = argparse.ArgumentParser(description='Unpack ERS(ESA) SLC data and store metadata in pickle file.')
    parser.add_argument('-i','--input', dest='datadir', type=str,
            required=True, help='Input ERS files path')
    parser.add_argument('-b', '--box' ,dest='bbox', type=float, nargs=4, default=None,
            help='Bbox (SNWE in degrees)')
    parser.add_argument('-o', '--output', dest='slcdir', type=str,
            required=True, help='Output SLC directory')
    parser.add_argument('--orbitdir', dest='orbitdir', type=str, required=True, help='Orbit directory')
    parser.add_argument('--orbittype',dest='orbittype', type = str, default='ODR', help='ODR, PDS, PDS')
    return parser.parse_args()

def get_Date(file):
    yyyymmdd=file[14:22]
    return yyyymmdd

def write_xml(shelveFile, slcFile):
    with shelve.open(shelveFile,flag='r') as db:
        frame = db['frame']

    length = frame.numberOfLines 
    width = frame.numberOfSamples
    print (width,length)

    slc = isceobj.createSlcImage()
    slc.setWidth(width)
    slc.setLength(length)
    slc.filename = slcFile
    slc.setAccessMode('write')
    slc.renderHdr()
    slc.renderVRT()

def unpack(fname, slcname, orbitdir, orbittype):
    '''
    Unpack .E* file to binary SLC file.
    '''

    obj = createSensor('ERS_ENviSAT_SLC')
    obj._imageFileName = fname
    obj._orbitDir = orbitdir
    obj._orbitType = orbittype
    #obj.instrumentDir = '/Users/agram/orbit/INS_DIR'
    obj.output = os.path.join(slcname,os.path.basename(slcname)+'.slc')
    obj.extractImage()
    obj.frame.getImage().renderHdr()
    obj.extractDoppler()

    # ####  computation of "poly" adapted from line 339 - line 353 of  Components/isceobj/Sensor/ERS_EnviSAT_SLC.py ###
    #######  removed this section and added  obj.extractDoppler() above instead. Doesn't seem to change anything in the processing. The latter is required for cropFrame.
    # coeffs = obj.dopplerRangeTime
    # dr = obj.frame.getInstrument().getRangePixelSize()
    # rref = 0.5 * Const.c * obj.rangeRefTime
    # r0 = obj.frame.getStartingRange()
    # norm = 0.5 * Const.c / dr

    # dcoeffs = []
    # for ind, val in enumerate(coeffs):
    #     dcoeffs.append( val / (norm**ind))
    
    # poly = Poly1D.Poly1D()
    # poly.initPoly(order=len(coeffs)-1)
    # poly.setMean( (rref - r0)/dr - 1.0)
    # poly.setCoeffs(dcoeffs)


    pickName = os.path.join(slcname, 'data')
    with shelve.open(pickName) as db:
        db['frame'] = obj.frame
        # db['doppler'] = poly 


if __name__ == '__main__':
    '''
    Main driver.
    '''

    inps = cmdLineParse()
    if inps.slcdir.endswith('/'):
        inps.slcdir = inps.slcdir[:-1]
    if not os.path.isdir(inps.slcdir):
        os.mkdir(inps.slcdir)
    for fname in glob.glob(os.path.join(inps.datadir, '*.E*')):
        date = get_Date(os.path.basename(fname))
        slcname = os.path.abspath(os.path.join(inps.slcdir, date))
        os.makedirs(slcname, exist_ok=True)

        print(fname)
        unpack(fname, slcname, inps.orbitdir, inps.orbittype)

        slcFile = os.path.abspath(os.path.join(slcname, date+'.slc'))

        shelveFile = os.path.join(slcname, 'data')
        write_xml(shelveFile,slcFile)

        if inps.bbox is not None:
            slccropname = os.path.abspath(os.path.join(inps.slcdir+'_crop',date))
            os.makedirs(slccropname, exist_ok=True)
            cmd = 'cropFrame.py -i {} -o {} -b {}'.format(slcname, slccropname, ' '.join([str(x) for x in inps.bbox]))
            print(cmd)
            os.system(cmd)
    
