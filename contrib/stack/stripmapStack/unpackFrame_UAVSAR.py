#!/usr/bin/env python3
# modified to pass the segment number to UAVSAR_STACK sensor EJF 2020/08/02

import os
import glob
import argparse
import shelve
import isce
from isceobj.Sensor import createSensor
from isceobj.Util import Poly1D
from isceobj.Planet.AstronomicalHandbook import Const


def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Unpack UAVSAR SLC data and store metadata in pickle file.')
    parser.add_argument('-i','--input', dest='metaFile', type=str,
            required=True, help='metadata file')
    parser.add_argument('-d','--dop_file', dest='dopFile', type=str,
            default=None, help='Doppler file')
    parser.add_argument('-s','--segment', dest='stackSegment', type=int,
            default=1, help='stack segment')
    parser.add_argument('-o', '--output', dest='slcDir', type=str,
            required=True, help='Output SLC directory')
    return parser.parse_args()


def unpack(metaFile, slcDir, dopFile, stackSegment, parse=False):
    '''
    Prepare shelve/pickle file for the binary SLC file.
    '''

    obj = createSensor('UAVSAR_STACK')
    obj.configure()
    obj.metadataFile = metaFile
    obj.dopplerFile = dopFile
    obj.segment_index = stackSegment
    obj.parse()

    if not os.path.isdir(slcDir):
        os.mkdir(slcDir)

    pickName = os.path.join(slcDir, 'data')
    with shelve.open(pickName) as db:
        db['frame'] = obj.frame


if __name__ == '__main__':
    '''
    Main driver.
    '''

    inps = cmdLineParse()
    inps.slcDir = inps.slcDir.rstrip('/')
    inps.metaFile = os.path.abspath(inps.metaFile)
    inps.dopFile = os.path.abspath(inps.dopFile)
    inps.slcDir = os.path.abspath(inps.slcDir)

    unpack(inps.metaFile, inps.slcDir, inps.dopFile, inps.stackSegment)
