#!/usr/bin/env python3

#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import glob
import shutil
import datetime
import numpy as np
import xml.etree.ElementTree as ET

import isce, isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import runCmd


def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='check ionospheric correction results')
    parser.add_argument('-idir', dest='idir', type=str, required=True,
            help = 'input directory where each date (YYYYMMDD.ion) is located. only files *.ion are recognized')
    parser.add_argument('-odir', dest='odir', type=str, required=True,
            help = 'output directory')
    parser.add_argument('-dates', dest='dates', type=str, nargs='+', default=None,
            help = 'a number of dates seperated by blanks. format: YYYYMMDD YYYYMMDD YYYYMMDD... This argument has highest priority. When provided, only process these dates')

    # parser.add_argument('-nrlks', dest='nrlks', type=int, default=1,
    #         help = 'number of range looks 1 * number of range looks ion. default: 1')
    # parser.add_argument('-nalks', dest='nalks', type=int, default=1,
    #         help = 'number of azimuth looks 1 * number of azimuth looks ion. default: 1')


    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()


    #get user parameters from input
    idir = inps.idir
    odir = inps.odir
    datesUser = inps.dates
    #######################################################

    if shutil.which('montage') is None:
        raise Exception('this command requires montage in ImageMagick\n')


    #get date folders
    dateDirs = sorted(glob.glob(os.path.join(os.path.abspath(idir), '*.ion')))
    dateDirs = [os.path.splitext(os.path.basename(x))[0] for x in dateDirs if os.path.isfile(x)]
    if datesUser is not None:
        pairs = datesUser
    else:
        pairs = dateDirs

    os.makedirs(odir, exist_ok=True)

    img = isceobj.createImage()
    img.load(os.path.join(idir, pairs[0] + '.ion.xml'))
    width = img.width
    length = img.length

    widthMax = 600
    if width >= widthMax:
        ratio = widthMax / width
        resize = ' -resize {}%'.format(ratio*100.0)
    else:
        ratio = 1.0
        resize = ''

    for ipair in pairs:
        ion = os.path.join(idir, ipair + '.ion')
        runCmd('mdx {} -s {} -cmap cmy -wrap 6.283185307179586 -addr -3.141592653589793 -P -workdir {}'.format(ion, width, odir))
        runCmd("montage -pointsize {} -label '{}' {} -geometry +{} -compress LZW{} {}.tif".format(
            int((ratio*width)/111*9+0.5),
            ipair,
            os.path.join(odir, 'out.ppm'),
            int((ratio*width)/111*5+0.5),
            resize,
            os.path.join(odir, ipair)))
        runCmd('rm {}'.format(os.path.join(odir, 'out.ppm')))


    #create colorbar
    width_colorbar = 100
    length_colorbar = 20
    colorbar = np.ones((length_colorbar, width_colorbar), dtype=np.float32) * \
               (np.linspace(-np.pi, np.pi, num=width_colorbar,endpoint=True,dtype=np.float32))[None,:]
    colorbar.astype(np.float32).tofile(os.path.join(odir, 'colorbar'))
    runCmd('mdx {} -s {} -cmap cmy -wrap 6.283185307179586 -addr -3.141592653589793 -P -workdir {}'.format(os.path.join(odir, 'colorbar'), width_colorbar, odir))
    runCmd('convert {} -compress LZW -resize 100% {}'.format(os.path.join(odir, 'out.ppm'), os.path.join(odir, 'colorbar_-pi_pi.tiff')))
    runCmd('rm {} {}'.format(
        os.path.join(odir, 'colorbar'),
        os.path.join(odir, 'out.ppm')))




