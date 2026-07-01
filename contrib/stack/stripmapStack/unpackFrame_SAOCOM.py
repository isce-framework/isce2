#!/usr/bin/env python3

import isce
import shelve
import argparse
import glob
import os
from isceobj.Sensor import createSensor

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Unpack SAOCOM SLC data and store metadata in pickle file.')
    parser.add_argument('-i', '--input', dest='inputdir', type=str, required=True, help='Input directory')
    parser.add_argument('-o', '--output', dest='outputdir', type=str, required=True, help='Output directory')

    return parser.parse_args()

def unpack(fname, outputdir):
    '''
    Unpack SAOCOM data to binary SLC file.
    '''

    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)

    date = os.path.basename(outputdir)
    
    imgname = glob.glob(os.path.join(fname,'S1*/Data/slc*-vv'))[0]
    xmlname = glob.glob(os.path.join(fname,'S1*/Data/slc*-vv.xml'))[0]
    xemtname = glob.glob(os.path.join(fname,'S1*.xemt'))[0]

    obj = createSensor('SAOCOM_SLC')
    obj._imageFileName = imgname
    obj.xmlFile = xmlname
    obj.xemtFile = xemtname
    obj.output = os.path.join(outputdir, date+'.slc')

    obj.extractImage()
    obj.frame.getImage().renderHdr()
    obj.extractDoppler()

    pickName = os.path.join(outputdir, 'data')
    with shelve.open(pickName) as db:
        db['frame'] = obj.frame

if __name__ == '__main__':
    '''
    Main driver.
    '''

    inps = cmdLineParse()

    if inps.inputdir.endswith('/'):
        inps.inputdir = inps.outputdir[:-1]

    unpack(inps.inputdir, inps.outputdir)
