#!/usr/bin/env python3

#Author: Heresh Fattahi

import isce
import isceobj
from contrib.demUtils.SWBDStitcher import SWBDStitcher
from iscesys.DataManager import createManager
import argparse
import configparser
from numpy import round

def createParser():
    '''
    Create command line parser.
    '''
    
    parser = argparse.ArgumentParser( description='extracts the overlap geometry between master bursts')
   # parser.add_argument('-b', '--bbox', dest='bbox', type=str, default=None, 
    #                  help='Lat/Lon Bounding SNWE')
    parser.add_argument('-b', '--bbox', type = int, default = None, nargs = '+', dest = 'bbox', help = 'Defines the spatial region in the format south north west east.\
                        The values should be integers from (-90,90) for latitudes and (0,360) or (-180,180) for longitudes.')
    parser.add_argument('-l', '--lat_file', dest='latName', type=str, default=None, 
                      help='pixel by pixel lat file in radar coordinate')
    parser.add_argument('-L', '--lon_file', dest='lonName', type=str, default=None, 
                      help='pixel by pixel lat file in radar coordinate')
    return parser

def cmdLineParse(iargs = None):
    '''
    Command line parser.
    '''

    parser = createParser()
    inps = parser.parse_args(args=iargs)
    #inps.bbox = [int(round(val)) for val in inps.bbox.split()]
    return inps


def download_waterMask(inps):

    sw = createManager('wbd')
    sw.configure()
    inps.waterBodyGeo = sw.defaultName(inps.bbox)
    sw._noFilling = False
    #sw._fillingValue = -1.0
    sw._fillingValue = 0.0
    sw.stitch(inps.bbox[0:2],inps.bbox[2:])

    return inps

def geo2radar(inps):
    inps.waterBodyRadar = inps.waterBodyGeo + '.rdr'
    sw = SWBDStitcher()
    sw.toRadar(inps.waterBodyGeo, inps.latName, inps.lonName, inps.waterBodyRadar)

#looks.py -i watermask.msk -r 4 -a 14 -o 'waterMask.14alks_4rlks.msk'

#imageMath.py -e='a*b' --a=filt_20100911_20101027.int --b=watermask.14alks_4rlks.msk -o filt_20100911_20101027_masked.int -t cfloat -s BIL

def main(iargs=None):

  inps = cmdLineParse(iargs)
  inps = download_waterMask(inps)    
  if inps.latName and inps.lonName:
     inps = geo2radar(inps)

if __name__ == '__main__' :
  ''' 
  creates a water mask and transforms to radar Coord
  '''
  main()



