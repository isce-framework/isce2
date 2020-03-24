#!/usr/bin/env python3

#Author: Heresh Fattahi

import os
import argparse
import configparser
import numpy as np
import isce
import isceobj
from iscesys.DataManager import createManager
from contrib.demUtils.SWBDStitcher import SWBDStitcher


EXAMPLE = """example:
  createWaterMask.py -b 31 33 130 132
  createWaterMask.py -b 31 33 130 132 -l lat.rdr -L lon.rdr -o waterMask.rdr
  createWaterMask.py -d ../DEM/demLat_N31_N33_Lon_E130_E132.dem.wgs84 -l lat.rdr -L lon.rdr -o waterMask.rdr
"""

def createParser():
    '''
    Create command line parser.
    '''
    
    parser = argparse.ArgumentParser(description='Create water body mask in geo and/or radar coordinates',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)
    parser.add_argument('-b', '--bbox', dest='bbox', type=int, default=None, nargs=4, metavar=('S','N','W','E'), 
                      help = 'Defines the spatial region in the format south north west east.\n'
                             'The values should be integers from (-90,90) for latitudes '
                             'and (0,360) or (-180,180) for longitudes.')
    parser.add_argument('-d','--dem_file', dest='demName', type=str, default=None,
                      help='DEM file in geo coordinates, i.e. demLat*.dem.wgs84.')

    parser.add_argument('-l', '--lat_file', dest='latName', type=str, default=None, 
                      help='pixel by pixel lat file in radar coordinate')
    parser.add_argument('-L', '--lon_file', dest='lonName', type=str, default=None, 
                      help='pixel by pixel lat file in radar coordinate')

    parser.add_argument('--fill', dest='fillValue', type=int, default=-1, choices={-1,0},
                      help='fill value for pixels with missing data. Default: -1.\n'
                           '-1 for water body\n'
                           ' 0 for land')
    parser.add_argument('-o', '--output', dest='outfile', type=str,
                      help='output filename of water mask in radar coordinates')
    return parser


def cmdLineParse(iargs = None):
    '''
    Command line parser.
    '''

    parser = createParser()
    inps = parser.parse_args(args=iargs)

    if not inps.bbox and not inps.demName:
        parser.print_usage()
        raise SystemExit('ERROR: no --bbox/--dem_file input, at least one is required.')

    if not inps.outfile and (inps.latName and inps.lonName):
        inps.outfile = os.path.join(os.path.dirname(inps.latName), 'waterMask.rdr')

    return inps


def dem2bbox(dem_file):
    """Grab bbox from DEM file in geo coordinates"""
    demImage = isceobj.createDemImage()
    demImage.load(dem_file + '.xml')
    demImage.setAccessMode('read')
    N = demImage.getFirstLatitude()
    W = demImage.getFirstLongitude()
    S = N + demImage.getDeltaLatitude() * demImage.getLength()
    E = W + demImage.getDeltaLongitude() * demImage.getWidth()
    bbox = [np.floor(S).astype(int), np.ceil(N).astype(int),
            np.floor(W).astype(int), np.ceil(E).astype(int)]
    return bbox


def download_waterMask(bbox, dem_file, fill_value=-1):
    out_dir = os.getcwd()
    # update out_dir and/or bbox if dem_file is input
    if dem_file:
        out_dir = os.path.dirname(dem_file)
        if not bbox:
            bbox = dem2bbox(dem_file)

    sw = createManager('wbd')
    sw.configure()
    #inps.waterBodyGeo = sw.defaultName(inps.bbox)
    sw.outputFile = os.path.join(out_dir, sw.defaultName(bbox))
    sw._noFilling = False
    sw._fillingValue = fill_value
    sw.stitch(bbox[0:2], bbox[2:])
    return sw.outputFile


def geo2radar(geo_file, rdr_file, lat_file, lon_file):
    #inps.waterBodyRadar = inps.waterBodyGeo + '.rdr'
    sw = SWBDStitcher()
    sw.toRadar(geo_file, lat_file, lon_file, rdr_file)
    return rdr_file

#looks.py -i watermask.msk -r 4 -a 14 -o 'waterMask.14alks_4rlks.msk'

#imageMath.py -e='a*b' --a=filt_20100911_20101027.int --b=watermask.14alks_4rlks.msk -o filt_20100911_20101027_masked.int -t cfloat -s BIL


def main(iargs=None):

    inps = cmdLineParse(iargs)
    geo_file = download_waterMask(inps.bbox, inps.demName, inps.fillValue)
    if inps.latName and inps.lonName:
        geo2radar(geo_file, inps.outfile, inps.latName, inps.lonName)
    return


if __name__ == '__main__' :
  ''' 
  creates a water mask and transforms to radar Coord
  '''
  main()



