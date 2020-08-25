#
# Author: Joshua Cohen
# Copyright 2016

import numpy as np
import argparse
import os,sys
import isce
import isceobj
import shelve
import datetime
from isceobj.Location.Offset import OffsetField,Offset
from iscesys.StdOEL.StdOELPy import create_writer
from GPUampcor import PyAmpcor
import pickle

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser( description='Generate offset field between two Sentinel swaths')
    parser.add_argument('-m', type=str, dest='reference', required=True, help='Reference image')
    parser.add_argument('-s', type=str, dest='secondary', required=True, help='Secondary image')
    parser.add_argument('-o', type=str, dest='outfile', required=True, help='Misregistration in subpixels')

    inps = parser.parse_args()
    return inps


def estimateOffsetField(reference, secondary, azoffset=0, rgoffset=0):
    '''
    Estimate offset field between burst and simamp.
    '''
    print('Creating images')
    sim = isceobj.createSlcImage()
    sim.load(secondary+'.xml')
    sim.setAccessMode('READ')
    sim.createImage()

    sar = isceobj.createSlcImage()
    sar.load(reference+'.xml')
    sar.setAccessMode('READ')
    sar.createImage()

    print('Configuring parameters')
    width = sar.getWidth()
    length = sar.getLength()

    ###     CONFIG PARAMS   ###
    refChipWidth = 128
    refChipHeight = 128
    schMarginX = 40
    schMarginY = 40
    schWinWidth = refChipWidth + (2 * schMarginX)
    schWinHeight = refChipHeight + (2 * schMarginY)
    offAc = 309
    offDn = 309
    numberLocationAcross = 40
    numberLocationDown = 40
    ###                     ###

    lastAc = int(min(width,(sim.getWidth()-offAc)) - schWinWidth)
    lastDn = int(min(length,(sim.getLength()-offDn)) - schWinHeight)

    band1 = 0 # bands are 0-indexed
    band2 = 0
    slcAccessor1 = sar.getImagePointer()
    slcAccessor2 = sim.getImagePointer()
    lineLength1 = sar.getWidth()
    fileLength1 = sar.getLength()
    lineLength2 = sim.getWidth()
    fileLength2 = sim.getLength()
    if sar.getDataType().upper().startswith('C'):
        imageDataType1 = 'complex'
    else:
        imageDataType1 = 'real'
    if sim.getDataType().upper().startswith('C'):
        imageDataType2 = 'complex'
    else:
        imageDataType2 = 'real'
    
    scaleFactorY = 1.0 # == prf2 / prf1
    scaleFactorX = 1.0 # == rangeSpacing1 / rangeSpacing2, but these are never set so I'm assuming....
    print('Scale Factor in Range: ', scaleFactorX)
    print('Scale Factor in Azimuth: ', scaleFactorY)
    offAcmax = int(rgoffset + (scaleFactorX-1)*lineLength1)
    offDnmax = int(azoffset + (scaleFactorY-1)*fileLength1)
    
    skipSampleDown = int((lastDn - offDn) / (numberLocationDown - 1.))
    print('Skip Sample Down: %d'%(skipSampleDown))
    skipSampleAcross = int((lastAc - offAc) / (numberLocationAcross - 1.))
    print('Skip Sample Across: %d'%(skipSampleAcross))

    ### setState
    objOffset = PyAmpcor()
    
    objOffset.imageBand1 = band1
    objOffset.imageBand2 = band2
    objOffset.imageAccessor1 = slcAccessor1
    objOffset.imageAccessor2 = slcAccessor2
    objOffset.datatype1 = imageDataType1
    objOffset.datatype2 = imageDataType2
    objOffset.lineLength1 = lineLength1
    objOffset.lineLength2 = lineLength2
    objOffset.firstSampleAcross = offAc
    objOffset.lastSampleAcross = lastAc
    objOffset.skipSampleAcross = skipSampleAcross
    objOffset.firstSampleDown = offDn
    objOffset.lastSampleDown = lastDn
    objOffset.skipSampleDown = skipSampleDown
    objOffset.acrossGrossOffset = rgoffset
    objOffset.downGrossOffset = azoffset
    objOffset.debugFlag = False
    objOffset.displayFlag = False
    objOffset.windowSizeWidth = refChipWidth
    objOffset.windowSizeHeight = refChipHeight
    objOffset.searchWindowSizeWidth = schMarginX
    objOffset.searchWindowSizeHeight = schMarginY
    objOffset.zoomWindowSize = 8
    objOffset.oversamplingFactor = 16
    objOffset.thresholdSNR = .001
    objOffset.thresholdCov = 1000.
    objOffset.scaleFactorX = scaleFactorX
    objOffset.scaleFactorY = scaleFactorY
    objOffset.acrossLooks = 1
    objOffset.downLooks = 1
    
    objOffset.runAmpcor()
    
    numElem = objOffset.numElem # numRowTable
    locationAcross = np.zeros(numElem, dtype=int)
    locationAcrossOffset = np.zeros(numElem, dtype=np.float32)
    locationDown = np.zeros(numElem, dtype=int)
    locationDownOffset = np.zeros(numElem, dtype=np.float32)
    snrRet = np.zeros(numElem, dtype=np.float32)
    cov1Ret = np.zeros(numElem, dtype=np.float32)
    cov2Ret = np.zeros(numElem, dtype=np.float32)
    cov3Ret = np.zeros(numElem, dtype=np.float32)
    for i in range(numElem):
        locationAcross[i] = objOffset.getLocationAcrossAt(i)
        locationAcrossOffset[i] = objOffset.getLocationAcrossOffsetAt(i)
        locationDown[i] = objOffset.getLocationDownAt(i)
        locationDownOffset[i] = objOffset.getLocationDownOffsetAt(i)
        snrRet[i] = objOffset.getSNRAt(i)
        cov1Ret[i] = objOffset.getCov1At(i)
        cov2Ret[i] = objOffset.getCov2At(i)
        cov3Ret[i] = objOffset.getCov3At(i)

    ### Back to refineSecondaryTiming.py from Ampcor.py
    sar.finalizeImage()
    sim.finalizeImage()
    
    result = OffsetField()
    for i in range(numElem):
        across = locationAcross[i]
        down = locationDown[i]
        acrossOffset = locationAcrossOffset[i]
        downOffset = locationDownOffset[i]
        snr = snrRet[i]
        sigx = cov1Ret[i]
        sigy = cov2Ret[i]
        sigxy = cov3Ret[i]
        offset = Offset()
        offset.setCoordinate(across,down)
        offset.setOffset(acrossOffset,downOffset)
        offset.setSignalToNoise(snr)
        offset.setCovariance(sigx,sigy,sigxy)
        result.addOffset(offset)
    
    return result

def fitOffsets(field, azrgOrder=0, azazOrder=0, rgrgOrder=0, rgazOrder=0, snr=5.0):
    '''
    Estimate constant range and azimuth shifts.
    '''

    stdWriter = create_writer("log","",True,filename='off.log')

    for distance in [10,5,3,1]:
        inpts = len(field._offsets)

        objOff = isceobj.createOffoutliers()
        objOff.wireInputPort(name='offsets', object=field)
        objOff.setSNRThreshold(snr)
        objOff.setDistance(distance)
        objOff.setStdWriter(stdWriter)

        objOff.offoutliers()

        field = objOff.getRefinedOffsetField()
        outputs = len(field._offsets)

        print('%d points left'%(len(field._offsets)))

    aa, dummy = field.getFitPolynomials(azimuthOrder=azazOrder, rangeOrder=azrgOrder, usenumpy=True)
    dummy, rr = field.getFitPolynomials(azimuthOrder=rgazOrder, rangeOrder=rgrgOrder, usenumpy=True)
    
    azshift = aa._coeffs[0][0]
    rgshift = rr._coeffs[0][0]
    print('Estimated az shift: ', azshift)
    print('Estimated rg shift: ', rgshift)

    return (aa, rr), field


if __name__ == '__main__':
    '''
    Generate offset fields burst by burst.
    '''

    print('Starting')

    inps = cmdLineParse()
    
    print('Estimating offset field')
    field = estimateOffsetField(inps.reference, inps.secondary)

    if os.path.exists(inps.outfile):
        os.remove(inps.outfile)

    rgratio = 1.0
    azratio = 1.0

    # section that gets [rg/az]ratio from metareference/metasecondary?
    odb = shelve.open(inps.outfile)
    odb['raw_field'] = field
    '''
    shifts, cull = fitOffsets(field)
    odb['cull_field'] = cull

    for row in shifts[0]._coeffs:
        for ind,val in enumerate(row):
            row[ind] = val * azratio
    for row in shifts[1]._coeffs:
        for ind,val in enumerate(row):
            row[ind] = val * rgratio

    odb['azpoly'] = shifts[0]
    odb['rgpoly'] = shifts[1]
    '''
    odb.close()

