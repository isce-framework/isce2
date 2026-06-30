#!/usr/bin/env python3

##Francisco Delgado, Universidad de Chile, 2023/06/21

import isce
from isceobj.Sensor import createSensor
import shelve
import argparse
import glob
from isceobj.Util import Poly1D
from isceobj.Planet.AstronomicalHandbook import Const
import os
import numpy as np

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Unpack SAOCOM SLC data and store metadata in pickle file.')
    parser.add_argument('-i','--input', dest='IMAGEFILE', type=str,
            required=True, help='IMAGEFILE')
    parser.add_argument('-o', '--output', dest='slcdir', type=str,
            required=True, help='Output SLC directory')

    return parser.parse_args()


def unpack(fname, slcname):
    '''
    Unpack SAOCOM data to binary SLC file.
    '''

    if not os.path.isdir(slcname):
        os.mkdir(slcname)

    date = os.path.basename(slcname)

    imgname = glob.glob(os.path.join(fname,'S1*/Data/slc*-hh'))[0]
    xmlname = glob.glob(os.path.join(fname,'S1*/Data/slc*-hh.xml'))[0]
    xemtname = glob.glob(os.path.join(fname,'S1*.xemt'))[0]
    #print(imgname)
    #print(xmlname)
    #print(xemtname)

    obj = createSensor('SAOCOM_SLC')
    obj._imageFileName = imgname
    obj.xmlFile = xmlname
    obj.xemtFile = xemtname
    obj.output = os.path.join(slcname, date+'.slc')

    obj.extractImage()
    obj.frame.getImage().renderHdr()


    obj.extractDoppler()

    pickName = os.path.join(slcname, 'data')
    with shelve.open(pickName) as db:
        db['frame'] = obj.frame


if __name__ == '__main__':
    '''
    Main driver.
    '''

    inps = cmdLineParse()
    if inps.IMAGEFILE.endswith('/'):
        inps.IMAGEFILE = inps.slcdir[:-1]
        
    unpack(inps.IMAGEFILE, inps.slcdir)
