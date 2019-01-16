#!/usr/bin/env python3

import numpy as np 
from osgeo import gdal
import os
import argparse
import pyproj
from geocodeGdal import write_xml

def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser(description='Convert geocoded files to Antarctica grid')
    parser.add_argument('-i', '--input', dest='infile', type=str, required=True,
            help='Input file to geocode')
    parser.add_argument('-r', '--resamp', dest='resampmethod', type=str, default='near',
            help='Resampling method')
    parser.add_argument('-f', '--format', dest='outformat', type=str, default='GTiff',
            help='Output GDAL format. If ENVI, ISCE XML is also written.')
    
    return parser


def cmdLineParse(iargs=None):
    '''
    Command line parser.
    '''

    parser = createParser()
    inps = parser.parse_args(args=iargs)

    inps.infile = inps.infile.split()

    if inps.outformat not in ['ENVI', 'GTiff']:
        raise Exception('Format can be ENVI or GTiff')

    return inps

def getGridLimits(geofile=None, latfile=None, lonfile=None):
    '''
    Get limits corresponding to Alex's grid.
    '''

    xmin = -2678407.5
    xmax = 2816632.5
    Nx = 22896

    ymin = -2154232.5
    ymax = 2259847.5
    Ny = 18392

    delx = 240.
    dely = 240.


    spolar = pyproj.Proj("+init=EPSG:3031")

    minyy = np.inf
    minxx = np.inf
    maxxx = -np.inf
    maxyy = -np.inf

    samples = 20

    if geofile is None:
        latds = gdal.Open(latfile, gdal.GA_ReadOnly)
        londs = gdal.Open(lonfile, gdal.GA_ReadOnly)

        width = latds.RasterXSize
        lgth  = latds.RasterYSize

        xs = np.linspace(0, width-1, num=samples).astype(np.int)
        ys = np.linspace(0, lgth-1, num=samples).astype(np.int)

        for line in range(samples):

            lats = latds.GetRasterBand(1).ReadAsArray(0, ys[line], width, 1)
            lons = londs.GetRasterBand(1).ReadAsArray(0, ys[line], width, 1)

            llat = lats[xs]
            llon = lats[ys]

            xx, yy = spolar(llon, llat)

            minxx = min(minxx, xx.min())
            maxxx = max(maxxx, xx.max())

            minyy = min(minyy, yy.min())
            maxyy = max(maxyy, yy.max())

        latds = None
        londs = None

    elif (latfile is None) and (lonfile is None):
       
        ds = gdal.Open(geofile, gdal.GA_ReadOnly)
        trans = ds.GetGeoTransform()

        width = ds.RasterXSize
        lgth  = ds.RasterYSize

        ds = None
        xs = np.linspace(0, width-1, num=samples)
        ys = np.linspace(0, lgth-1, num=samples)

        lons = trans[0] + xs * trans[1]

        for line in range(samples):
            lats = (trans[3] + ys[line] * trans[5]) * np.ones(samples)

            xx, yy = spolar(lons, lats)

            minxx = min(minxx, xx.min())
            maxxx = max(maxxx, xx.max())

            minyy = min(minyy, yy.min())
            maxyy = max(maxyy, yy.max())

    else:
        raise Exception('Either geofile is provided (or) latfile and lonfile. All 3 inputs cannot be provided') 


    ii0 =  max(np.int((ymax - maxyy - dely/2.0) / dely ), 0)
    ii1 =  min(np.int((ymax - minyy + dely/2.0) / dely ) + 1, Ny)

    jj0 = max(np.int((minxx - xmin - delx/2.0)/delx), 0)
    jj1 = min(np.int((maxxx - xmin + delx/2.0)/delx) + 1, Nx)


    ylim = ymax - np.array([ii1,ii0]) * dely
    xlim = xmin + np.array([jj0,jj1]) * delx

    return ylim, xlim


def runGeo(inps, ylim, xlim, method='near', fmt='GTiff'):


    WSEN = str(xlim[0]) + ' ' + str(ylim[0]) + ' ' + str(xlim[1]) + ' ' + str(ylim[1])

    if fmt == 'ENVI':
        ext = '.ant'
    else:
        ext = '.tif'

    for infile in inps:
        infile = os.path.abspath(infile)
        print('geocoding: ' + infile)

        cmd = 'gdalwarp -of ' + fmt + ' -t_srs EPSG:3031 -te ' + WSEN + ' -tr 240.0 240.0 -srcnodata 0 -dstnodata 0 -r ' + method + ' ' + infile + ' ' + infile + ext

        status = os.system(cmd)
        if status:
            raise Exception('Command {0} Failed'.format(cmd))

def main(iargs=None):
    '''
    Main driver.
    '''

    inps = cmdLineParse(iargs)

    ylim, xlim = getGridLimits(geofile=inps.infile[0])

    print('YLim: ', ylim, (ylim[1]-ylim[0])/240. + 1)
    print('XLim: ', xlim, (xlim[1]-xlim[0])/240. + 1)

    runGeo(inps.infile, ylim, xlim, 
            method=inps.resampmethod, fmt=inps.outformat)

if __name__ == '__main__':

    main()
