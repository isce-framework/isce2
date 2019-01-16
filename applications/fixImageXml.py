#!/usr/bin/env python3

import isce
import isceobj
import argparse
import os

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Fixes pathnames in ISCE image XML files. Can be used to do more things in the future.')
    parser.add_argument('-i', '--input', type=str, required=True, dest='infile',
            help = 'Input image for which the XML file needs to be fixed.')
    parser.add_argument('-f', '--full', action='store_true', default=False, dest='full',
            help = 'Replace filename with full path including dir in which file is located')
    parser.add_argument('-b', '--base', action='store_true', default=False, dest='base',
            help = 'Replace filename with basename to use in current directory')

    inps = parser.parse_args()

    if (inps.full and inps.base):
        raise Exception('User requested to use both full path and basename')

    if (not inps.full) and (not inps.base):
        raise Exception('User did not request any change')

    return inps


if __name__ == '__main__':
    '''
    Main driver.
    '''
    from imageMath import IML

    inps = cmdLineParse()

    if inps.infile.endswith('.xml'):
        inps.infile = os.path.splitext(inps.infile)[0]

    dirname  = os.path.dirname(inps.infile)

    img, dataname, metaName = IML.loadImage(inps.infile)

    if inps.full:
        fname = os.path.abspath( os.path.join(dirname, os.path.basename(inps.infile)))
    elif inps.base:
        fname = os.path.basename( os.path.basename(inps.infile))
    else:
        raise Exception('Unknown state in {0}'.format(os.path.basename(__file__)))

    img.filename = fname
    img.setAccessMode('READ')
    img.renderHdr()
