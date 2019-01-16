#!/usr/bin/env python3

import numpy as np 
import isce
import pickle
import argparse
import datetime

def cmdLineParse():

    parser =argparse.ArgumentParser(description='Fix SLC pickle file.')
    parser.add_argument('-i', dest='infile', type=str, required=True,
            help='Pickle file to fix')
    parser.add_argument('-a', dest='azoffset', type=float, required=True,
            help='Azimuth offset in pixels')
    parser.add_argument('-r', dest='rgoffset', type=float, required=True,
            help='Range offset in pixels')

    return parser.parse_args()

if __name__ == '__main__':

    inps = cmdLineParse()

    with open(inps.infile, 'rb') as f:
        data = pickle.load(f)

    prf = data.getInstrument().getPulseRepetitionFrequency()
    deltat = datetime.timedelta(seconds = inps.azoffset/prf)

    data.sensingStart += deltat
    data.sensingStop += deltat
    data.sensingMid += deltat

    
    dr = data.getInstrument().getRangePixelSize()

    print(prf, dr)
    data.startingRange += inps.rgoffset * dr
   

    with open(inps.infile, 'wb') as f:
        pickle.dump(data,f)
