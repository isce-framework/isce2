#!/usr/bin/env python3

import isce
from isceobj.Sensor import createSensor
import shelve
import argparse
import glob
from isceobj.Util import Poly1D
from isceobj.Util.decorators import use_api
import os


def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Unpack Capella SLC data and store metadata in pickle file.')
    parser.add_argument('-i', '--input', dest='tiffFile', type=str,
                        required=True, help='Input Capella SLC GeoTIFF file')
    parser.add_argument('-o', '--output', dest='slcdir', type=str,
                        required=True, help='Output unpacked SLC directory')

    return parser.parse_args()


@use_api
def unpack(tiffFile, slcname):
    '''
    Unpack Capella data to binary SLC file.
    '''

    # Create output SLC directory if needed
    if not os.path.isdir(slcname):
        os.mkdir(slcname)

    date = os.path.basename(slcname)

    # Create a Capella sensor object and configure it
    obj = createSensor('Capella')
    obj.configure()
    obj.tiff = tiffFile
    obj.output = os.path.join(slcname, date + '.slc')

    # Extract the image and write the XML file for the SLC
    obj.extractImage()
    obj.frame.getImage().renderHdr()

    # Save the doppler polynomial (Capella provides doppler_centroid_polynomial)
    coeffs = obj.doppler_coeff
    poly = Poly1D.Poly1D()
    poly.initPoly(order=len(coeffs) - 1)
    poly.setCoeffs(coeffs)

    # Save required metadata for further use
    # Note: Capella does not provide FM rate polynomial, so we don't save it
    pickName = os.path.join(slcname, 'data')
    with shelve.open(pickName) as db:
        db['frame'] = obj.frame
        db['doppler'] = poly


if __name__ == '__main__':
    '''
    Main driver.
    '''

    inps = cmdLineParse()
    if inps.slcdir.endswith('/'):
        inps.slcdir = inps.slcdir[:-1]

    if inps.tiffFile.endswith('/'):
        inps.tiffFile = inps.tiffFile[:-1]

    unpack(inps.tiffFile, inps.slcdir)
