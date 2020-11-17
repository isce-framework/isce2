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
            help = 'input directory where each pair (YYMMDD-YYMMDD) is located. only folders are recognized')
    parser.add_argument('-odir', dest='odir', type=str, required=True,
            help = 'output directory for estimated ionospheric phase of each date')
    parser.add_argument('-pairs', dest='pairs', type=str, nargs='+', default=None,
            help = 'a number of pairs seperated by blanks. format: YYMMDD-YYMMDD YYMMDD-YYMMDD YYMMDD-YYMMDD... This argument has highest priority. When provided, only process these pairs')
    parser.add_argument('-wbd_msk', dest='wbd_msk', action='store_true',
            help='apply water body mask in the output image')

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
    pairsUser = inps.pairs
    wbdMsk = inps.wbd_msk
    #######################################################

    if shutil.which('montage') is None:
        raise Exception('this command requires montage in ImageMagick\n')


    #get date folders
    dateDirs = sorted(glob.glob(os.path.join(os.path.abspath(idir), '*')))
    dateDirs = [os.path.basename(x) for x in dateDirs if os.path.isdir(x)]
    if pairsUser is not None:
        pairs = pairsUser
    else:
        pairs = dateDirs

    os.makedirs(odir, exist_ok=True)

    img = isceobj.createImage()
    img.load(glob.glob(os.path.join(idir, pairs[0], 'ion', 'ion_cal', 'filt_ion_*rlks_*alks.ion'))[0] + '.xml')
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
        diffOriginal = glob.glob(os.path.join(idir, ipair, 'ion', 'ion_cal', 'diff_{}_*rlks_*alks_ori.int'.format(ipair)))[0]
        ion = glob.glob(os.path.join(idir, ipair, 'ion', 'ion_cal', 'filt_ion_*rlks_*alks.ion'))[0]
        diff = glob.glob(os.path.join(idir, ipair, 'ion', 'ion_cal', 'diff_{}_*rlks_*alks.int'.format(ipair)))[0]

        if wbdMsk:
            wbd = glob.glob(os.path.join(idir, ipair, 'ion', 'ion_cal', 'wbd_*rlks_*alks.wbd'))[0]
            wbdArguments = ' {} -s {} -i1 -cmap grey -percent 100'.format(wbd, width)
        else:
            wbdArguments = ''

        runCmd('mdx {} -s {} -c8pha -cmap cmy -wrap 6.283185307179586 -addr -3.141592653589793{} -P -workdir {}'.format(diffOriginal, width, wbdArguments, odir))
        runCmd('mv {} {}'.format(os.path.join(odir, 'out.ppm'), os.path.join(odir, 'out1.ppm')))
        runCmd('mdx {} -s {} -cmap cmy -wrap 6.283185307179586 -addr -3.141592653589793{} -P -workdir {}'.format(ion, width, wbdArguments, odir))
        runCmd('mv {} {}'.format(os.path.join(odir, 'out.ppm'), os.path.join(odir, 'out2.ppm')))
        runCmd('mdx {} -s {} -c8pha -cmap cmy -wrap 6.283185307179586 -addr -3.141592653589793{} -P -workdir {}'.format(diff, width, wbdArguments, odir))
        runCmd('mv {} {}'.format(os.path.join(odir, 'out.ppm'), os.path.join(odir, 'out3.ppm')))
        runCmd("montage -pointsize {} -label 'original' {} -label 'ionosphere' {} -label 'corrected' {} -geometry +{} -compress LZW{} {}.tif".format(
            int((ratio*width)/111*18+0.5),
            os.path.join(odir, 'out1.ppm'),
            os.path.join(odir, 'out2.ppm'),
            os.path.join(odir, 'out3.ppm'),
            int((ratio*width)/111*5+0.5),
            resize,
            os.path.join(odir, ipair)))
        runCmd('rm {} {} {}'.format(
            os.path.join(odir, 'out1.ppm'),
            os.path.join(odir, 'out2.ppm'),
            os.path.join(odir, 'out3.ppm')))


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




