#!/usr/bin/env python3

import numpy as np
import os
import argparse
import tempfile

try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    raise Exception('gdal python bindings are needed for this script to work.')

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Generate graphics from ISCE products using gdal')
    parser.add_argument('-i', dest='infile', type=str, required=True,
            help='Input ISCE product file')
    parser.add_argument('-o', dest='outfile', type=str, required=True,
            help='Output GEOTIFF file')
    parser.add_argument('-b', dest='band', type=int, default=0,
            help='Band number to use if input image is multiband. Default: 0')
    parser.add_argument('-c', dest='clim', type=float, nargs=2, required=True,
            help='Color limits for the graphics')
    parser.add_argument('-m', dest='cmap', type=str, default='jet',
            help='Matplotlib colormap to use')
    parser.add_argument('-t', dest='table', type=str, default=None,
            help='Color table to use')
    parser.add_argument('-n', dest='ncolors', type=int, default=64,
            help='Number of colors')
    inps = parser.parse_args()

    return inps


def get_cmap(mapname, N, clim):
    '''
    Get the colormap from matplotlib.
    '''

    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import matplotlib.cm as cmx
    except ImportError:
        raise Exception('Matplotlib is needed if user-defined color table is not provided.')

    cmap = plt.get_cmap(mapname)
    cNorm = colors.Normalize(vmin = clim[0], vmax = clim[1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
   
    vals = np.linspace(inps.clim[0], inps.clim[1], endpoint=True)

    outname = mapname + '.cpt'
    
    with open(outname, 'w') as fid:
        for val in vals:
            cval = scalarMap.to_rgba(val)
            fid.write('{0} {1} {2} {3} \n'.format(val,int(cval[0]*255), int(cval[1]*255), int(cval[2]*255)))
            
        fid.write('nv 0 0 0 0 \n')

    return outname

if __name__ == '__main__':
    '''
    Main driver.
    '''

    #Parse command line
    inps = cmdLineParse()


    ####Convert to a gdal format if not already done
    try:
        ds = gdal.Open(inps.infile)
        ds = None
    except:
        cmd = 'isce2gis.py envi -i {0}'.format(inps.infile)
        flag = os.system(cmd)

        if flag:
            raise Exception('Failed: {0}'.format(cmd))

    ####Set up the color table
    if inps.table is None: ####No custom color map has been provided
        cmap = get_cmap(inps.cmap, inps.ncolors, inps.clim)
        plt_cmap = True
    else:
        cmap = inps.table
        plt_cmap = False


    #####Build VRT
    vrtname = inps.outfile+'.vrt'
    if os.path.exists(vrtname):
        print('VRT file already exists. Cleaning it ....')
        os.remove(vrtname)

    cmd = 'gdaldem color-relief {0} {1} {2} -alpha -b {3} -of VRT'.format(inps.infile, cmap, vrtname, inps.band+1)
    
    flag = os.system(cmd)
    if flag:
        raise Exception('Failed: %s'%(cmd))

    ###Build geotiff
    cmd = 'gdal_translate {0} {1}'.format(vrtname, inps.outfile)

    flag = os.system(cmd)

    if flag:
        raise Exception('Failed: %s'%(cmd))

