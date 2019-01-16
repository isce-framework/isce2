#!/usr/bin/env python3

import isce
import isceobj
import stdproc
import numpy as np
from isceobj.Util.Poly1D import Poly1D
from isceobj.Util.Poly2D import Poly2D
import argparse
import os
import shelve
import matplotlib.pyplot as plt
import datetime

def cmdLineParse():
    parser = argparse.ArgumentParser( description='Use polynomial offsets and create burst by burst interferograms')

    parser.add_argument('-i', dest='master', type=str, required=True,
            help='Directory with master acquisition')

    inps = parser.parse_args()

    if inps.master.endswith('/'):
        inps.master = inps.master[:-1]
    return inps


def resampSlave(master, burst, doppler, azpoly, flatten=False):
    '''
    Resample burst by burst.
    '''
  

    inimg = isceobj.createSlcImage()
    base = os.path.basename(master)
    inimg.load(os.path.join(master, base+ '_orig.slc.xml'))
    inimg.setAccessMode('READ')
    width = inimg.getWidth()
    length = inimg.getLength()

    prf = burst.getInstrument().getPulseRepetitionFrequency()

    coeffs = [2*np.pi*val/prf for val in doppler._coeffs]
    zcoeffs = [0. for val in coeffs]

    dpoly = Poly2D()
    dpoly.initPoly(rangeOrder=doppler._order, azimuthOrder=1, coeffs=[zcoeffs,coeffs])

    apoly = Poly2D()
    apoly.setMeanRange(azpoly._mean)
    apoly.setNormRange(azpoly._norm)
    apoly.initPoly(rangeOrder=azpoly._order, azimuthOrder=0, coeffs=[azpoly._coeffs])

    print('Shifts: ', apoly(1,1), apoly(10240,10240))

    rObj = stdproc.createResamp_slc()
    rObj.slantRangePixelSpacing = burst.getInstrument().getRangePixelSize()
    rObj.radarWavelength = burst.getInstrument().getRadarWavelength()
    rObj.azimuthCarrierPoly = dpoly
   
    rObj.azimuthOffsetsPoly = apoly
    rObj.imageIn = inimg


    imgOut = isceobj.createSlcImage()
    imgOut.setWidth(width)
    imgOut.filename = os.path.join(master, base+'.slc')
    imgOut.setAccessMode('write')
    
    rObj.flatten = flatten
    rObj.outputWidth = width
    rObj.outputLines = length

    rObj.resamp_slc(imageOut=imgOut)

    imgOut.renderHdr()
    return imgOut


def estimateAzShift(frame, dpoly, fpoly):
    '''
    Estimate azimuth shift polynomial.
    '''
    width = frame.getNumberOfSamples()
    prf = frame.getInstrument().getPulseRepetitionFrequency()

    print('Dopplers: ', dpoly(0), dpoly(width-1))
    print('FMrates: ', fpoly(0), fpoly(width-1))

    x = np.linspace(0,width, num=100)
    dt = -prf * dpoly(x) / fpoly(x)

    print('Shifts: ', dt[0], dt[-1])

    dt0 = dt[0] ####Account for shift to near range

#    dt = dt-dt0
    shift = Poly1D()
    shift.initPoly(order=4)
    shift.polyfit(x,dt)
    y = shift(x)

    print('Est shifts: ', y[0], y[-1])

    if False:
        plt.plot(x, dt, 'b')
        plt.plot(x, y, 'r')
        plt.show()

    return shift, dt0/prf 


if __name__ == '__main__':
    '''
    Main driver.
    '''
    inps = cmdLineParse()

    db = shelve.open(os.path.join(inps.master, 'original'), flag='r')
    frame = db['frame']
    doppler = db['doppler']
    fmrate = db['fmrate']
    db.close()

    azpoly, dt0 = estimateAzShift(frame, doppler, fmrate)

    imgout = resampSlave(inps.master, frame, doppler, azpoly)
    
    imgout.setAccessMode('READ')
    frame.image = imgout

    delta = datetime.timedelta(seconds=dt0)
    
    print('Time shift: ', -delta.total_seconds())

#    frame.sensingStart -= delta
#    frame.sensingMid -= delta
#    frame.sensingStop -= delta

    db = shelve.open(os.path.join(inps.master, 'data'))
    db['frame'] = frame
    db['doppler'] = doppler
    db['fmrate'] = fmrate
    db.close()

