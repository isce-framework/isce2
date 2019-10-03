#!/usr/bin/env python3

import isce
from isceobj.Sensor import createSensor
import shelve
import argparse
import glob
from isceobj.Util import Poly1D
from isceobj.Planet.AstronomicalHandbook import Const
import os
from datetime import datetime

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Unpack CSK SLC data and store metadata in pickle file.')
    parser.add_argument('-i','--input', dest='h5dir', type=str,
            required=True, help='Input CSK directory')
    parser.add_argument('-o', '--output', dest='slcdir', type=str,
            required=True, help='Output SLC directory')

    return parser.parse_args()

def unpack(hdf5, slcname):
    '''
    Unpack HDF5 to binary SLC file.
    '''

    if os.listdir(hdf5)[0].endswith('.E1'): #ERS1
        fname = glob.glob(os.path.join(hdf5,'SAR*.E1'))[0]
        orbitDir = '/home/mgovorcin/Working_dir/Ston_Slano1996/INSAR/orbits/ODR/ERS1/dgm-e04/' #ERS1

    elif os.listdir(hdf5)[0].endswith('.E2'):
        fname = glob.glob(os.path.join(hdf5,'SAR*.E2'))[0] #ERS2
        if datetime.strptime(os.path.basename(fname)[14:22],'%Y%m%d').date()>datetime(2003,8,8).date():
            orbitDir = '/home/mgovorcin/Working_dir/Ston_Slano1996/INSAR/orbits/ODR/ERS2/' #ERS2
        else:
            orbitDir = '/home/mgovorcin/Working_dir/Ston_Slano1996/INSAR/orbits/ODR/ERS2/dgm-e04/' #ERS2
 
    print(fname)

    if not os.path.isdir(slcname):
        os.mkdir(slcname)

    date = os.path.basename(slcname)
    obj = createSensor('ERS_ENVISAT_SLC')
    obj.configure()
    obj._imageFileName = fname
    obj._orbitDir = orbitDir
    obj._orbitType = 'ODR'
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
    if inps.slcdir.endswith('/'):
        inps.slcdir = inps.slcdir[:-1]

    if inps.h5dir.endswith('/'):
        inps.h5dir = inps.h5dir[:-1]

    unpack(inps.h5dir, inps.slcdir)
