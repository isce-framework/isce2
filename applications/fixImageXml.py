#!/usr/bin/env python3


import os
import argparse
import isce
import isceobj
from isceobj.Util.ImageUtil import ImageLib as IML


def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Fixes pathnames in ISCE image XML files. Can be used to do more things in the future.')
    parser.add_argument('-i', '--input', type=str, required=True, dest='infile',
            help = 'Input image for which the XML file needs to be fixed.')

    fname = parser.add_mutually_exclusive_group(required=True)
    fname.add_argument('-f', '--full', action='store_false',
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

    if inps.infile.endswith('.xml'):
        inps.infile = os.path.splitext(inps.infile)[0]

    dirname  = os.path.dirname(inps.infile)

    img = IML.loadImage(inps.infile)[0]

    if inps.full:
        fname = os.path.abspath( os.path.join(dirname, os.path.basename(inps.infile)))
    else:
        fname = os.path.basename( os.path.basename(inps.infile))

    img.filename = fname
    img.setAccessMode('READ')
    img.renderHdr()


