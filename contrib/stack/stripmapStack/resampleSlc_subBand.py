#!/usr/bin/env python3

import isce
import isceobj
import stdproc
from stdproc.stdproc import crossmul
import numpy as np
from isceobj.Util.Poly2D import Poly2D
import argparse
import os
import shelve
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
import json
import logging
from isceobj.Util.decorators import use_api

def createParser():
    parser = argparse.ArgumentParser( description='Use polynomial offsets and create burst by burst interferograms')

    parser.add_argument('-m', '--reference', dest='reference', type=str, default=None,
            help = 'Directory with reference acquisition for reference')
    parser.add_argument('-s', '--secondary', dest='secondary', type=str, required=True,
            help='Directory with secondary acquisition')

    parser.add_argument('-p', '--poly',dest='poly', type=str, default=None,
            help='Pickle file with the resampling polynomials')

    parser.add_argument('-o','--coreg', dest='coreg', type=str, default=None,
            help='Directory with coregistered SLC')

    parser.add_argument('-f', '--offsets', dest='offsets', type=str, default=None,
            help='Directory with the offset files')

    parser.add_argument('-z', dest='zero', action='store_true', default=False,
            help='Resample without using azimuth carrier')

    parser.add_argument('--noflat', dest='noflat', action='store_true', default=False,
            help='To turn off flattening')

    parser.add_argument('-d', '--dims', dest='dims', nargs=2, type=int, default=None,
            help='Dimensions if using directly with poly')

    #inps = parser.parse_args()

    #if inps.secondary.endswith('/'):
    #    inps.secondary = inps.secondary[:-1]

    #if inps.coreg is None:
    #    inps.coreg = os.path.join('coreg', os.path.basename(inps.secondary))

    #return inps

    return parser
    
def cmdLineParse(iargs = None):
    parser = createParser()
    #return parser.parse_args(args=iargs)
    
    inps =  parser.parse_args(args=iargs)
    
    #inps = parser.parse_args()
    
    if inps.secondary.endswith('/'):
        inps.secondary = inps.secondary[:-1]
    
    if inps.coreg is None:
        inps.coreg = os.path.join('coreg', os.path.basename(inps.secondary))
    
    return inps

@use_api
def resampSecondary(burst, offdir, outname, doppler, azpoly, rgpoly, 
        reference=None, flatten=False, zero=False, dims=None):
    '''
    Resample burst by burst.
    '''
  

    if offdir is not None:
        rgname = os.path.join(offdir, 'range.off')
        azname = os.path.join(offdir, 'azimuth.off')

        rngImg = isceobj.createImage()
        rngImg.load(rgname + '.xml')
        rngImg.setAccessMode('READ')

        aziImg = isceobj.createImage()
        aziImg.load(azname + '.xml')
        aziImg.setAccessMode('READ')
        
        width = rngImg.getWidth()
        length = rngImg.getLength()

    else:
        rngImg = None
        aziImg = None
        if dims is None:
            raise Exception('No offset image / dims provided.')

        width = dims[1]
        length = dims[0]


    inimg = isceobj.createSlcImage()
    inimg.load(burst.getImage().filename + '.xml')
    inimg.setAccessMode('READ')

    prf = burst.getInstrument().getPulseRepetitionFrequency()

    if zero:
        factor = 0.0
    else:
        factor = 1.0

    try:
        print('Polynomial doppler provided')
        coeffs = [factor * 2*np.pi*val/prf for val in doppler._coeffs]
    except:
        print('List of coefficients provided')
        coeffs = [factor * 2*np.pi*val/prf for val in doppler]


    zcoeffs = [0. for val in coeffs]
    dpoly = Poly2D()
#    dpoly.initPoly(rangeOrder=len(coeffs)-1, azimuthOrder=1, coeffs=[zcoeffs,coeffs])
    dpoly.initPoly(rangeOrder=len(coeffs)-1, azimuthOrder=0, coeffs=[coeffs])

    rObj = stdproc.createResamp_slc()
    rObj.slantRangePixelSpacing = burst.getInstrument().getRangePixelSize()
   # rObj.radarWavelength = burst.getInstrument().getRadarWavelength()
    rObj.radarWavelength = burst.subBandRadarWavelength 
    rObj.dopplerPoly = dpoly
   
    rObj.azimuthOffsetsPoly = azpoly
    rObj.rangeOffsetsPoly = rgpoly
    rObj.imageIn = inimg
    imgOut = isceobj.createSlcImage()
    imgOut.setWidth(width)

    outdir = os.path.dirname(outname)
    os.makedirs(outdir, exist_ok=True)

    if zero:
        imgOut.filename = os.path.join(outname)
    else:
        imgOut.filename = os.path.join(outname)
    imgOut.setAccessMode('write')
    
    rObj.flatten = flatten
    rObj.outputWidth = width
    rObj.outputLines = length
    rObj.residualRangeImage = rngImg
    rObj.residualAzimuthImage = aziImg

    if reference is not None:
        rObj.startingRange = burst.startingRange
        rObj.referenceStartingRange = reference.startingRange
        rObj.referenceSlantRangePixelSpacing = reference.getInstrument().getRangePixelSize()
      #  rObj.referenceWavelength = reference.getInstrument().getRadarWavelength()
        rObj.referenceWavelength = reference.subBandRadarWavelength

    rObj.resamp_slc(imageOut=imgOut)

    imgOut.renderHdr()
    return imgOut


def main(iargs=None):
    '''
    Main driver.
    '''
    inps = cmdLineParse(iargs)

    outfile = os.path.join(inps.coreg,os.path.basename(os.path.dirname(inps.coreg))+'.slc')
    outDir = inps.coreg
    os.makedirs(outDir, exist_ok=True)

    referenceShelveDir = os.path.join(outDir, 'referenceShelve')
    secondaryShelveDir = os.path.join(outDir, 'secondaryShelve')

    if inps.reference is not None:
       os.makedirs(referenceShelveDir, exist_ok=True)

    os.makedirs(secondaryShelveDir, exist_ok=True)

    cmd = 'cp '+ inps.secondary + '/data* ' + secondaryShelveDir
    print (cmd)
    os.system(cmd)

    if inps.reference is not None:
       cmd = 'cp '+ inps.reference + '/data* ' + referenceShelveDir
       os.system(cmd)

   # with shelve.open(os.path.join(inps.secondary, 'data'), flag='r') as sdb:
    with shelve.open(os.path.join(secondaryShelveDir, 'data'), flag='r') as sdb:
        secondary = sdb['frame']
        try:
            doppler = sdb['doppler']
        except:
            doppler = secondary._dopplerVsPixel

    if inps.poly is not None:
        with shelve.open(inps.poly, flag='r') as db:
            azpoly = db['azpoly']
            rgpoly = db['rgpoly']


    else:
        azpoly = None
        rgpoly = None


    if inps.reference is not None:
       # with shelve.open(os.path.join(inps.reference, 'data'), flag='r') as mdb:
        with shelve.open(os.path.join(referenceShelveDir, 'data'), flag='r') as mdb:
            reference = mdb['frame']
    else:
        reference = None

    resampSecondary(secondary, inps.offsets, outfile,
            doppler, azpoly,rgpoly, 
            flatten=(not inps.noflat), zero=inps.zero,
            dims = inps.dims,
            reference = reference)

#    flattenSLC(secondary, inps.coreg, rgpoly)

if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()

