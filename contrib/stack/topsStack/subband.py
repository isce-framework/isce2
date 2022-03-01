#!/usr/bin/env python3

# Author: Cunren Liang
# Copyright 2021

import isce
import isceobj
import stdproc
from stdproc.stdproc import crossmul
import numpy as np
from isceobj.Util.Poly2D import Poly2D
import argparse
import os
import copy
import s1a_isce_utils as ut
from isceobj.Sensor.TOPS import createTOPSSwathSLCProduct

#it should be OK that function name is the same as script name
from subband_and_resamp import subband


def createParser():
    parser = argparse.ArgumentParser( description='bandpass filtering burst by burst SLCs ')

    parser.add_argument('-d', '--directory', dest='directory', type=str, required=True,
            help='Directory with acquisition')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def main(iargs=None):
    '''
    Create subband burst SLCs.
    '''
    inps = cmdLineParse(iargs)
    swathList = ut.getSwathList(inps.directory)

    for swath in swathList:
        acquisition = ut.loadProduct( os.path.join(inps.directory , 'IW{0}.xml'.format(swath)))
        for burst in acquisition.bursts:

            print("processing swath {}, burst {}".format(swath, os.path.basename(burst.image.filename)))

            outname = burst.image.filename
            outnameLower = os.path.splitext(outname)[0]+'_lower.slc'
            outnameUpper = os.path.splitext(outname)[0]+'_upper.slc'
            if os.path.exists(outnameLower) and os.path.exists(outnameLower+'.vrt') and os.path.exists(outnameLower+'.xml') and \
               os.path.exists(outnameUpper) and os.path.exists(outnameUpper+'.vrt') and os.path.exists(outnameUpper+'.xml'):
                print('burst {} already processed, skip...'.format(os.path.basename(burst.image.filename)))
                continue

            #subband filtering
            from Stack import ionParam
            from isceobj.Constants import SPEED_OF_LIGHT
            rangeSamplingRate = SPEED_OF_LIGHT / (2.0 * burst.rangePixelSize)

            ionParamObj=ionParam()
            ionParamObj.configure()
            outputfile = [outnameLower, outnameUpper]
            bw = [ionParamObj.rgBandwidthSub / rangeSamplingRate, ionParamObj.rgBandwidthSub / rangeSamplingRate]
            bc = [-ionParamObj.rgBandwidthForSplit / 3.0 / rangeSamplingRate, ionParamObj.rgBandwidthForSplit / 3.0 / rangeSamplingRate]
            rgRef = ionParamObj.rgRef
            subband(burst, 2, outputfile, bw, bc, rgRef, True)



if __name__ == '__main__':
    '''
    Main driver.
    '''
    # Main Driver
    main()



