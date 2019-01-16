#!/usr/bin/env python3

import os
import shelve
import isce
import argparse
import datetime

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Fix master metadata')
    parser.add_argument('-m', '--master', dest='master', type=str, required=True,
            help='Directory with master SLC')
    parser.add_argument('-o', '--offset', dest='offset', type=str, required=True,
            help='Pickle with offsets')

    return parser.parse_args()

if __name__ == '__main__':
    '''
    Main driver.
    '''
    
    inps = cmdLineParse()

    mdb = shelve.open( os.path.join(inps.master, 'data'), writeback=True)
    odb = shelve.open( inps.offset, flag='r')

    field = odb['cull_field']
    aa, rr = field.getFitPolynomials(azimuthOrder=0, rangeOrder=0, usenumpy=True)

    meanaz = aa._coeffs[0][0]
    meanrg = rr._coeffs[0][0]

    frame = mdb['frame']
    
    r0 = frame.getStartingRange()
    dr = frame.getInstrument().getRangePixelSize()
    frame.setStartingRange( r0 + meanrg* dr)

    prf = frame.getInstrument().getPulseRepetitionFrequency()
    delta = datetime.timedelta(seconds = meanaz / prf) 

    print('Range: ', meanrg*dr)
    print('Azimuth: ', delta)
    frame.setSensingStart( frame.getSensingStart() + delta)
    frame.setSensingMid( frame.getSensingMid() + delta)
    frame.setSensingStop( frame.getSensingStop() + delta)
   
    odb.close()
    mdb.close()
