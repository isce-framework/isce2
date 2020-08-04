#!/usr/bin/env python3

import numpy as np 
import argparse
import os
import isce
import isceobj
from mroipac.baseline.Baseline import Baseline
from isceobj.Planet.Planet import Planet
import datetime
import shelve

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser( description='Generate offset field between two Sentinel swaths')
    parser.add_argument('-m', type=str, dest='reference', required=True,
            help='Directory with the reference image')
    parser.add_argument('-s', type=str, dest='secondary', required=True,
            help='Directory with the secondary image')

    return parser.parse_args()


if __name__ == '__main__':
    '''
    Generate offset fields burst by burst.
    '''

    inps = cmdLineParse()

    try:
        mdb = shelve.open( os.path.join(inps.reference, 'data'), flag='r')
    except:
        mdb = shelve.open( os.path.join(inps.reference, 'raw'), flag='r')

    mFrame = mdb['frame']

    try:
        sdb = shelve.open( os.path.join(inps.secondary, 'data'), flag='r')
    except:
        sdb = shelve.open( os.path.join(inps.secondary, 'raw'), flag='r')


    sFrame = sdb['frame']


    bObj = Baseline()
    bObj.configure()
    bObj.wireInputPort(name='referenceFrame', object=mFrame)
    bObj.wireInputPort(name='secondaryFrame', object=sFrame)

    bObj.baseline()

    print('Baseline at top/bottom: %f %f'%(bObj.pBaselineTop,bObj.pBaselineBottom))

    mdb.close()
    sdb.close()
