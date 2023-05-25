#!/usr/bin/env python3


import os
import argparse
import glob
import isce
import isceobj
from isceobj.Util.ImageUtil import ImageLib as IML


def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Fixes pathnames in ISCE image XML files. Can be used to do more things in the future.')
    parser.add_argument('-i', '--input', type=str, nargs='+', required=True, dest='infile',
            help = 'Input image for which the XML file needs to be fixed.')

    fname = parser.add_mutually_exclusive_group(required=True)
    fname.add_argument('-f', '--full', action='store_true',
            help = 'Replace filename with full path including dir in which file is located')
    fname.add_argument('-b', '--base', action='store_true',
            help = 'Replace filename with basename to use in current directory')

    inps = parser.parse_args()
    return inps


if __name__ == '__main__':
    '''
    Main driver.
    '''
    inps = cmdLineParse()

    for fname in inps.infile:
        if fname.endswith('.xml'):
            fname = os.path.splitext(fname)[0]
        print('fixing xml file path for file: {}'.format(fname))

        if inps.full:
            fdir = os.path.dirname(fname)
            fname = os.path.abspath(os.path.join(fdir, os.path.basename(fname)))
        else:
            fname = os.path.basename(os.path.basename(fname))

        img = IML.loadImage(fname)[0]
        img.filename = fname
        img.setAccessMode('READ')
        img.renderHdr()

