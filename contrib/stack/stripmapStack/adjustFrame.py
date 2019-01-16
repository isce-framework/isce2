#!/usr/bin/env python3

import numpy as np 
import argparse
import os
import isce
import isceobj
import datetime
import shelve
import matplotlib.pyplot as plt

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser( description='Generate offset field between two Sentinel swaths')
    parser.add_argument('-m', type=str, dest='master', required=True,
            help='Directory with the master image')

    parser.add_argument('-a', '--az', type=float, dest='azshift', default=0.0,
            help='Azimuth shift to add in lines')

    parser.add_argument('-r', '--rg', type=float, dest='rgshift', default=0.0,
            help='Range shift to add in pixels')
    
    return parser.parse_args()


if __name__ == '__main__':
    '''
    Generate offset fields burst by burst.
    '''

    inps = cmdLineParse()

    mdb = shelve.open( os.path.join(inps.master, 'data'), flag='r')
    mFrame = mdb['frame']
    mdb.close()

    print('Before: ')
    print('t0: ', mFrame.sensingStart)
    print('r0: ', mFrame.startingRange)


    deltat = datetime.timedelta(seconds = inps.azshift/ mFrame.PRF)

    mFrame.sensingStart += deltat
    mFrame.sensingMid  += deltat
    mFrame.sensingStop += deltat


    deltar = inps.rgshift * mFrame.instrument.rangePixelSize
    mFrame.startingRange += deltar

    mdb =  shelve.open( os.path.join(inps.master, 'data'), writeback=True)
    mdb['frame'] = mFrame
    mdb.close()

    mdb = shelve.open(os.path.join(inps.master, 'data'), flag='r')
    mFrame = mdb['frame']
    mdb.close()

    print('After: ')
    print('t0: ', mFrame.sensingStart)
    print('r0: ', mFrame.startingRange)

