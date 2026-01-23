#!/usr/bin/env python3

import isce
from isceobj.Sensor import createSensor
import shelve
import argparse
import glob
from isceobj.Util import Poly1D
from isceobj.Planet.AstronomicalHandbook import Const
from isceobj.Util.decorators import use_api
import os


def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Unpack Capella SLC data and store metadata in pickle file.')
    parser.add_argument('-i', '--input', dest='capellaDir', type=str,
                        required=True, help='Input Capella SLC directory')
    parser.add_argument('-o', '--output', dest='slcdir', type=str,
                        required=True, help='Output unpacked SLC directory')

    return parser.parse_args()


@use_api
def unpack(capellaDir, slcname):
    '''
    Unpack Capella data to binary SLC file.
    '''

    # Search for imagery (GeoTIFF) and metadata (JSON) files in input directory
    # Capella file naming: CAPELLA_<sat>_<mode>_SLC_<pol>_<start>_<stop>.tif
    # and CAPELLA_<sat>_<mode>_SLC_<pol>_<start>_<stop>_extended.json
    tiffiles = glob.glob(os.path.join(capellaDir, 'CAPELLA*.tif'))
    if not tiffiles:
        tiffiles = glob.glob(os.path.join(capellaDir, '*.tif'))

    if not tiffiles:
        raise FileNotFoundError(f'No TIFF files found in {capellaDir}')

    imgname = tiffiles[0]

    # Look for the extended JSON metadata file
    jsonfiles = glob.glob(os.path.join(capellaDir, '*_extended.json'))
    if not jsonfiles:
        # Try any JSON file
        jsonfiles = glob.glob(os.path.join(capellaDir, '*.json'))

    if not jsonfiles:
        raise FileNotFoundError(f'No JSON metadata files found in {capellaDir}')

    metaname = jsonfiles[0]

    # Create output SLC directory if needed
    if not os.path.isdir(slcname):
        os.mkdir(slcname)

    date = os.path.basename(slcname)

    # Create a Capella sensor object and configure it
    obj = createSensor('Capella')
    obj.configure()
    obj.metadataFile = metaname
    obj.tiff = imgname
    obj.output = os.path.join(slcname, date + '.slc')

    # Extract the image and write the XML file for the SLC
    obj.extractImage()
    obj.frame.getImage().renderHdr()

    # Save the doppler polynomial
    coeffs = obj.doppler_coeff
    poly = Poly1D.Poly1D()
    poly.initPoly(order=len(coeffs) - 1)
    poly.setCoeffs(coeffs)

    # Save the FM rate polynomial
    fcoeffs = obj.azfmrate_coeff
    fpoly = Poly1D.Poly1D()
    fpoly.initPoly(order=len(fcoeffs) - 1)
    fpoly.setCoeffs(fcoeffs)

    # Save required metadata for further use
    # All data is output to a shelve file
    pickName = os.path.join(slcname, 'data')
    with shelve.open(pickName) as db:
        db['frame'] = obj.frame
        db['doppler'] = poly
        db['fmrate'] = fpoly


if __name__ == '__main__':
    '''
    Main driver.
    '''

    inps = cmdLineParse()
    if inps.slcdir.endswith('/'):
        inps.slcdir = inps.slcdir[:-1]

    if inps.capellaDir.endswith('/'):
        inps.capellaDir = inps.capellaDir[:-1]

    unpack(inps.capellaDir, inps.slcdir)
