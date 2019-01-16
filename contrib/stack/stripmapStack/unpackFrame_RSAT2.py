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

    parser = argparse.ArgumentParser(description='Unpack RADARSAT2 SLC data and store metadata in pickle file.')
    parser.add_argument('-i','--input', dest='RSATdir', type=str,
            required=True, help='Input RADARSAT2 SLC directory')
    parser.add_argument('-o', '--output', dest='slcdir', type=str,
            required=True, help='Output unpacked SLC directory')

    return parser.parse_args()



@use_api
def unpack(RSATdir, slcname):
    '''
    Unpack RADARSAT2 data to binary SLC file. assume HH only for now
    '''

    ###Search for imagery and XML files in input directory
    imgname = glob.glob(os.path.join(RSATdir,'imagery*.tif'))[0]
    xmlname = glob.glob(os.path.join(RSATdir, 'product.xml'))[0]

    ####Create output SLC directory if needed
    if not os.path.isdir(slcname):
        os.mkdir(slcname)

    date = os.path.basename(slcname)

    #####Create an Radarsat2 object and wire it
    obj = createSensor('Radarsat2')
    obj.configure()
    obj.xml = xmlname
    obj.tiff = imgname
    obj.output = os.path.join(slcname, date+'.slc')

    ####Extract the image and write the XML file for the SLC
    obj.extractImage()
    obj.frame.getImage().renderHdr()


    ####Save the doppler polynomial
    ####CEOS already provides doppler polynomial
    ####as a function of range pixel
    coeffs = obj.doppler_coeff
    poly = Poly1D.Poly1D()
    poly.initPoly(order=len(coeffs)-1)
    poly.setCoeffs(coeffs)


    ####Save the FMrate polynomial
    ####CEOS already provides FMrate polynomial
    ####as a function of range pixel
    fcoeffs = obj.azfmrate_coeff
#    fcoeffs = [0.0, 0.0, 0.0]  # zero-Doppler geometry, so this is not used
    fpoly = Poly1D.Poly1D()
    fpoly.initPoly(order=len(fcoeffs)-1)
    fpoly.setCoeffs(fcoeffs)


    ####Save required metadata for further use
    ####All data is output to a shelve file
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

    if inps.RSATdir.endswith('/'):
        inps.RSATdir = inps.RSATdir[:-1]

    unpack(inps.RSATdir, inps.slcdir)
