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

    parser.add_argument('-m', '--master', dest='master', type=str, default=None,
            help = 'Directory with master acquisition for reference')
    parser.add_argument('-s', '--slave', dest='slave', type=str, required=True,
            help='Directory with slave acquisition')

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

    #if inps.slave.endswith('/'):
    #    inps.slave = inps.slave[:-1]

    #if inps.coreg is None:
    #    inps.coreg = os.path.join('coreg', os.path.basename(inps.slave))

    #return inps

    return parser
    
def cmdLineParse(iargs = None):
    parser = createParser()
    #return parser.parse_args(args=iargs)
    
    inps =  parser.parse_args(args=iargs)
    
    #inps = parser.parse_args()
    
    if inps.slave.endswith('/'):
        inps.slave = inps.slave[:-1]
    
    if inps.coreg is None:
        inps.coreg = os.path.join('coreg', os.path.basename(inps.slave))
    
    return inps

@use_api
def resampSlave(burst, offdir, outname, doppler, azpoly, rgpoly, 
        master=None, flatten=False, zero=False, dims=None):
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
    if not os.path.exists(outdir):
        os.makedirs(outdir)

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

    if master is not None:
        rObj.startingRange = burst.startingRange
        rObj.referenceStartingRange = master.startingRange
        rObj.referenceSlantRangePixelSpacing = master.getInstrument().getRangePixelSize()
      #  rObj.referenceWavelength = master.getInstrument().getRadarWavelength()
        rObj.referenceWavelength = master.subBandRadarWavelength

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
    if not os.path.exists(outDir):
       os.makedirs(outDir)

    masterShelveDir = os.path.join(outDir, 'masterShelve')
    slaveShelveDir = os.path.join(outDir, 'slaveShelve')

    if (not os.path.exists(masterShelveDir)) and (inps.master is not None ):
       os.makedirs(masterShelveDir)

    if not os.path.exists(slaveShelveDir):
       os.makedirs(slaveShelveDir)

    cmd = 'cp '+ inps.slave + '/data* ' + slaveShelveDir
    print (cmd)
    os.system(cmd)

    if inps.master is not None:
       cmd = 'cp '+ inps.master + '/data* ' + masterShelveDir
       os.system(cmd)

   # with shelve.open(os.path.join(inps.slave, 'data'), flag='r') as sdb:
    with shelve.open(os.path.join(slaveShelveDir, 'data'), flag='r') as sdb:
        slave = sdb['frame']
        try:
            doppler = sdb['doppler']
        except:
            doppler = slave._dopplerVsPixel

    if inps.poly is not None:
        with shelve.open(inps.poly, flag='r') as db:
            azpoly = db['azpoly']
            rgpoly = db['rgpoly']


    else:
        azpoly = None
        rgpoly = None


    if inps.master is not None:
       # with shelve.open(os.path.join(inps.master, 'data'), flag='r') as mdb:
        with shelve.open(os.path.join(masterShelveDir, 'data'), flag='r') as mdb:
            master = mdb['frame']
    else:
        master = None

    resampSlave(slave, inps.offsets, outfile,
            doppler, azpoly,rgpoly, 
            flatten=(not inps.noflat), zero=inps.zero,
            dims = inps.dims,
            master = master)

#    flattenSLC(slave, inps.coreg, rgpoly)

if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()

