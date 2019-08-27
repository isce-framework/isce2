#!/usr/bin/env python3
# modified by Marin Govorcin, 05.2018, mgovorcin@geof.hr
# Added option to export Wrapped interferogram to geotiff
# Added option to clip interferogram tiff with water shapefile mask
# Added export to KML
# Usage isce2geotiff.py -i /fullpath/filt_topophase_unw(unwrapped)/flat(wrapped).geo -o /fullpath/output_name -c 0 3.13 (color limits) -wrap (flag when converting wrapped intfg) -mask /fullpath/mask.shp (use water shapefile to mask water areas) -kml /fullpath/kml_output_name -b 1 (band 1 when using filt_topophase.unw.geo)

import numpy as np
import os
import argparse
import tempfile
#import pdb; pdb.set_trace()

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
    parser.add_argument('-wrap', dest='wrap', action='store_true',
            help='Use wrapped input ISCE product file')
    parser.add_argument('-mask', dest='mask', type=str, default=None,
            help='Use water mask to crop tif')
    parser.add_argument('-kml', dest='kml', type=str, default=None,
            help='Create kml file')
    inps = parser.parse_args()

    if inps.table is not None:
        if inps.reversemap:
            raise Exception('Only matplotlib colormaps can be reversed')

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
        inname = inps.infile+'.vrt'
        ds = gdal.Open(inname)
        print(inname)
        ds = None
    except:
        dirpath = os.getcwd()
        folderpath = os.path.basename(dirpath)
        print(folderpath)
        inname = inps.infile
        if folderpath == 'interferogram':
            cmd = 'cd ..; isce2gis.py vrt -i {0}'.format(inname)
        else:
            cmd = 'isce2gis.py vrt -i {0}'.format(inname)
        print(cmd)
        flag = os.system(cmd)

        if flag:
            raise Exception('Failed: {0}'.format(cmd))

    ####Set up the color table
    if inps.table is None: ####No custom color map has been provided
        cmap = get_cmap(inps.cmap, inps.ncolors, inps.clim)
        print(cmap)
        plt_cmap = True
    else:
        cmap = table
        plt_cmap = False

    #####Build Wrapped VRT
    if inps.wrap:
        vrtname = inps.infile+'.vrt'
        invrtname = inps.outfile+'.phs.tif'
        outvrtname = inps.outfile+'.vrt'
        tifoutname = inps.outfile+'.wrapped.tif'
        cmd = 'gdal_calc.py --type Float32 -A {0} --calc="numpy.angle(A)" --outfile={1} --NoDataValue=0.0 --overwrite'.format(vrtname, invrtname)
        print(cmd)
        
        flag = os.system(cmd)

        if flag:
           raise Exception('Failed: %s'%(cmd))

    else:
    #####Build Unwrapped VRT 
        invrtname = inps.infile+'.vrt'
        outvrtname = inps.outfile+'.vrt'
        tifoutname = inps.outfile+'.unwrapped.tif'
        if os.path.exists(outvrtname):
           print('VRT file already exists. Cleaning it ....')
           os.remove(outvrtname)

    ###Build geotiff
    if os.path.exists(tifoutname):
        print('TIF file already exists. Cleaning it ....')
        os.remove(tifoutname)
    cmd = 'gdaldem color-relief {0} {1} {2} -alpha -b {3} -of VRT'.format(invrtname, cmap, outvrtname, inps.band+1)
    print(cmd)
    print(inps.band)

    flag = os.system(cmd)

    if flag:
        raise Exception('Failed: %s'%(cmd))

    ###Build geotiff
    if os.path.exists(tifoutname):
        print('TIF file already exists. Cleaning it ....')
        os.remove(tifoutname)
    cmd = 'gdal_translate {0} {1}'.format(outvrtname, tifoutname)

    flag = os.system(cmd)

    if flag:
        raise Exception('Failed: %s'%(cmd))

   ####Use water mask to Crop RGB tiff
    if inps.mask is None:
        mask = None 
    else:
        maskinput = inps.mask
        cmd = 'gdal_rasterize -b 1 -b 2 -b 3 -b 4 -burn 0 -burn 0 -burn 0 -burn 0 {0} {1}'.format(maskinput,tifoutname)
        print(cmd)
        flag = os.system(cmd)
        if flag:
           raise Exception('Failed: {0}'.format(cmd))

    ####Create KML png file
    if inps.kml is None:
        kml = None
    else:
        kmloutname = inps.kml+'.kmz'
        cmd = 'gdal_translate -of KMLSUPEROVERLAY {0} {1} -co format=png'.format(tifoutname,kmloutname)
        print(cmd)
        flag = os.system(cmd)
        if flag:
           raise Exception('Failed: {0}'.format(cmd))

        


