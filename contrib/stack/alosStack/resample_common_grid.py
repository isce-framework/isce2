#!/usr/bin/env python3

#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import glob
import datetime
import numpy as np

import isce, isceobj, stdproc
from isceobj.Util.Poly2D import Poly2D
from isceobj.Location.Offset import OffsetField, Offset

from isceobj.Alos2Proc.Alos2ProcPublic import readOffset
from isceobj.Alos2Proc.runSwathOffset import swathOffset

from contrib.alos2proc.alos2proc import rg_filter

from StackPulic import loadTrack
from StackPulic import saveTrack
from StackPulic import subbandParameters
from StackPulic import stackDateStatistics
from StackPulic import acquisitionModesAlos2


def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='resample data to a common grid')
    parser.add_argument('-idir', dest='idir', type=str, required=True,
            help = 'input directory where data of each date (YYMMDD) is located. only folders are recognized')
    parser.add_argument('-odir', dest='odir', type=str, required=True,
            help = 'output directory where resampled version of each date is output')
    parser.add_argument('-ref_date', dest='ref_date', type=str, required=True,
            help = 'reference date. format: YYMMDD')
    parser.add_argument('-sec_date', dest='sec_date', type=str, nargs='+', default=[],
            help = 'a number of secondary dates seperated by blanks, can also include ref_date. format: YYMMDD YYMMDD YYMMDD. If provided, only resample these dates')
    parser.add_argument('-ref_frame', dest='ref_frame', type=str, default=None,
            help = 'frame number of the swath whose grid is used as reference. e.g. 2800. default: first frame')
    parser.add_argument('-ref_swath', dest='ref_swath', type=int, default=None,
            help = 'swath number of the swath whose grid is used as reference. e.g. 1. default: first swath')
    parser.add_argument('-nrlks1', dest='nrlks1', type=int, default=1,
            help = 'range offsets between swaths/frames should be integer multiples of -nrlks1. default: 1 ')
    parser.add_argument('-nalks1', dest='nalks1', type=int, default=14,
            help = 'azimuth offsets between swaths/frames should be integer multiples of -nalks1. default: 14')
    parser.add_argument('-subband', dest='subband', action='store_true', default=False,
            help='create and resample subband SLCs')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()


    #get user parameters from input
    idir = inps.idir
    odir = inps.odir
    dateReference = inps.ref_date
    dateSecondary = inps.sec_date
    frameReference = inps.ref_frame
    swathReference = inps.ref_swath
    nRange = inps.nrlks1
    nAzimuth = inps.nalks1
    subbandFlag = inps.subband
    #######################################################

    DEBUG=False

    spotlightModes, stripmapModes, scansarNominalModes, scansarWideModes, scansarModes = acquisitionModesAlos2()

    #get date statistics
    dateDirs,   dates,   frames,   swaths,   dateIndexReference = stackDateStatistics(idir, dateReference)
    ndate = len(dates)
    nframe = len(frames)
    nswath = len(swaths)

    if frameReference is None:
        frameReference = frames[0]
    else:
        if frameReference not in frames:
            raise Exception('specified -ref_frame {} not in frame list {}'.format(frameReference, frames))
    if swathReference is None:
        swathReference = swaths[0]
    else:
        if swathReference not in swaths:
            raise Exception('specified -ref_swath {} not in swath list {}'.format(swathReference, swaths))

    #find frame and swath indexes of reference swath
    frameReferenceIndex = frames.index(frameReference)
    swathReferenceIndex = swaths.index(swathReference)

    print('resampling all frames and swaths to frame: {} (index: {}) swath: {} (index {})'.format(
        frameReference, frameReferenceIndex, swathReference, swathReferenceIndex))


    #read swath offsets and save in 2-d lists
    swathRangeOffsetGeometrical = []
    swathAzimuthOffsetGeometrical = []
    swathRangeOffsetMatching = []
    swathAzimuthOffsetMatching = []
    for i, frameNumber in enumerate(frames):

        swathRangeOffsetGeometrical0 = []
        swathAzimuthOffsetGeometrical0 = []
        swathRangeOffsetMatching0 = []
        swathAzimuthOffsetMatching0 = []

        if nswath >= 2:
            frameDir = 'f{}_{}'.format(i+1, frameNumber)
            with open(os.path.join(idir, dateReference, frameDir, 'mosaic/swath_offset.txt'), 'r') as f:
                lines = f.readlines()

            for linex in lines:
                if 'range offset' in linex:
                    swathRangeOffsetGeometrical0.append(float(linex.split()[3]))
                    swathRangeOffsetMatching0.append(float(linex.split()[4]))
                if 'azimuth offset' in linex:
                    swathAzimuthOffsetGeometrical0.append(float(linex.split()[3]))
                    swathAzimuthOffsetMatching0.append(float(linex.split()[4]))
        else:
            swathRangeOffsetGeometrical0.append(0.0)
            swathRangeOffsetMatching0.append(0.0)
            swathAzimuthOffsetGeometrical0.append(0.0)
            swathAzimuthOffsetMatching0.append(0.0)

        swathRangeOffsetGeometrical.append(swathRangeOffsetGeometrical0)
        swathAzimuthOffsetGeometrical.append(swathAzimuthOffsetGeometrical0)
        swathRangeOffsetMatching.append(swathRangeOffsetMatching0)
        swathAzimuthOffsetMatching.append(swathAzimuthOffsetMatching0)


    #read frame offsets and save in 1-d list
    frameRangeOffsetGeometrical = []
    frameAzimuthOffsetGeometrical = []
    frameRangeOffsetMatching = []
    frameAzimuthOffsetMatching = []

    if nframe >= 2:
        with open(os.path.join(idir, dateReference, 'insar/frame_offset.txt'), 'r') as f:
            lines = f.readlines()
        for linex in lines:
            if 'range offset' in linex:
                frameRangeOffsetGeometrical.append(float(linex.split()[3]))
                frameRangeOffsetMatching.append(float(linex.split()[4]))
            if 'azimuth offset' in linex:
                frameAzimuthOffsetGeometrical.append(float(linex.split()[3]))
                frameAzimuthOffsetMatching.append(float(linex.split()[4]))
    else:
        frameRangeOffsetGeometrical.append(0.0)
        frameRangeOffsetMatching.append(0.0)
        frameAzimuthOffsetGeometrical.append(0.0)
        frameAzimuthOffsetMatching.append(0.0)


    #compute accurate starting range and sensing start using offset file for reference date
    #swath offset is computed between adjacent swaths within a frame, offset unit: first swath sample size
    #frame offset is computed between first swaths of adjacent frames, offset unit: first swath sample size
    startingRangeAll = [[None for j in range(nswath)] for i in range(nframe)]
    sensingStartAll = [[None for j in range(nswath)] for i in range(nframe)]

    trackReference = loadTrack(dateDirs[dateIndexReference], dates[dateIndexReference])
    for i, frameNumber in enumerate(frames):
        #startingRange and sensingStart of first swath of current frame
        # for i1 in range(i+1):
        #     startingRangeFirst = trackReference.frames[0].swaths[0].startingRange - \
        #                          frameRangeOffsetMatching[i1] * trackReference.frames[0].swaths[0].rangePixelSize
        #     sensingStartFirst  = trackReference.frames[0].swaths[0].sensingStart - \
        #                          datetime.timedelta(seconds = frameAzimuthOffsetMatching[i1] * trackReference.frames[0].swaths[0].azimuthLineInterval)

        startingRangeFirst = trackReference.frames[0].swaths[0].startingRange - \
                             sum(frameRangeOffsetMatching[0:i+1]) * trackReference.frames[0].swaths[0].rangePixelSize
        sensingStartFirst  = trackReference.frames[0].swaths[0].sensingStart - \
                             datetime.timedelta(seconds = sum(frameAzimuthOffsetMatching[0:i+1]) * trackReference.frames[0].swaths[0].azimuthLineInterval)

        #startingRange and sensingStart of each swath of current frame
        for j, swathNumber in enumerate(range(swaths[0], swaths[-1] + 1)):
            # for j1 in range(j+1):
            #     startingRangeAll[i][j] = startingRangeFirst - \
            #         swathRangeOffsetMatching[i][j1] * trackReference.frames[i].swaths[0].rangePixelSize
            #     sensingStartAll[i][j] = sensingStartFirst - \
            #         datetime.timedelta(seconds = swathAzimuthOffsetMatching[i][j1] * trackReference.frames[i].swaths[0].azimuthLineInterval)

            startingRangeAll[i][j] = startingRangeFirst - \
                sum(swathRangeOffsetMatching[i][0:j+1]) * trackReference.frames[i].swaths[0].rangePixelSize
            sensingStartAll[i][j] = sensingStartFirst - \
                datetime.timedelta(seconds = sum(swathAzimuthOffsetMatching[i][0:j+1]) * trackReference.frames[i].swaths[0].azimuthLineInterval)

    #check computation result
    if DEBUG:
        for i, frameNumber in enumerate(frames):
            for j, swathNumber in enumerate(range(swaths[0], swaths[-1] + 1)):
                print(i, j, (trackReference.frames[i].swaths[j].startingRange-startingRangeAll[i][j])/trackReference.frames[0].swaths[0].rangePixelSize, 
                      (trackReference.frames[i].swaths[j].sensingStart-sensingStartAll[i][j]).total_seconds()/trackReference.frames[0].swaths[0].azimuthLineInterval)

    #update startingRange and sensingStart of reference track
    for i, frameNumber in enumerate(frames):
        for j, swathNumber in enumerate(range(swaths[0], swaths[-1] + 1)):
            trackReference.frames[i].swaths[j].startingRange = startingRangeAll[i][j]
            trackReference.frames[i].swaths[j].sensingStart = sensingStartAll[i][j]


    ##find minimum startingRange and sensingStart
    startingRangeMinimum = trackReference.frames[0].swaths[0].startingRange
    sensingStartMinimum  = trackReference.frames[0].swaths[0].sensingStart
    for i, frameNumber in enumerate(frames):
        for j, swathNumber in enumerate(range(swaths[0], swaths[-1] + 1)):
            if trackReference.frames[i].swaths[j].startingRange < startingRangeMinimum:
                startingRangeMinimum = trackReference.frames[i].swaths[j].startingRange
            if trackReference.frames[i].swaths[j].sensingStart < sensingStartMinimum:
                sensingStartMinimum = trackReference.frames[i].swaths[j].sensingStart
    print('startingRangeMinimum (m): {}'.format(startingRangeMinimum))
    print('sensingStartMinimum: {}'.format(sensingStartMinimum))


    #adjust each swath of each frame to minimum startingRange and sensingStart
    #load reference track again for saving track parameters of resampled
    trackReferenceResampled = loadTrack(dateDirs[dateIndexReference], dates[dateIndexReference])
    for i, frameNumber in enumerate(frames):
        for j, swathNumber in enumerate(range(swaths[0], swaths[-1] + 1)):
            #current swath
            swathReference = trackReference.frames[i].swaths[j]
            #swath of reference sample size
            swathReferenceReference = trackReference.frames[frameReferenceIndex].swaths[swathReferenceIndex]
            #current swath resampled
            swathReferenceResampled = trackReferenceResampled.frames[i].swaths[j]

            #update startingRange and sensingStart
            offsetRange = (swathReference.startingRange - startingRangeMinimum) / (swathReferenceReference.rangePixelSize*nRange)
            offsetAzimuth = (swathReference.sensingStart - sensingStartMinimum).total_seconds() / (swathReferenceReference.azimuthLineInterval*nAzimuth)

            swathReferenceResampled.startingRange = startingRangeMinimum + round(offsetRange) * (swathReferenceReference.rangePixelSize*nRange)
            swathReferenceResampled.sensingStart  = sensingStartMinimum + datetime.timedelta(seconds = round(offsetAzimuth) *
                                                                             (swathReferenceReference.azimuthLineInterval*nAzimuth))

            #update other parameters
            swathReferenceResampled.numberOfSamples     = round(swathReference.numberOfSamples * swathReference.rangePixelSize / swathReferenceReference.rangePixelSize)
            swathReferenceResampled.numberOfLines       = round(swathReference.numberOfLines * swathReference.azimuthLineInterval / swathReferenceReference.azimuthLineInterval)
            swathReferenceResampled.rangeSamplingRate   = swathReferenceReference.rangeSamplingRate
            swathReferenceResampled.rangePixelSize      = swathReferenceReference.rangePixelSize
            swathReferenceResampled.prf                 = swathReferenceReference.prf
            swathReferenceResampled.azimuthPixelSize    = swathReferenceReference.azimuthPixelSize
            swathReferenceResampled.azimuthLineInterval = swathReferenceReference.azimuthLineInterval
            #should also update dopplerVsPixel, azimuthFmrateVsPixel?
            #if hasattr(swathReference, 'burstLength'):
            if swathReference.burstLength is not None:
                swathReferenceResampled.burstLength        *= (swathReference.burstLength * swathReference.azimuthLineInterval / swathReferenceReference.azimuthLineInterval)
            #if hasattr(swathReference, 'burstCycleLength'):
            if swathReference.burstCycleLength is not None:
                swathReferenceResampled.burstCycleLength   *= (swathReference.burstCycleLength * swathReference.azimuthLineInterval / swathReferenceReference.azimuthLineInterval)
            #no need to update parameters for ScanSAR burst-by-burst processing, since we are not doing such burst-by-burst processing.


    #resample each date
    os.makedirs(odir, exist_ok=True)
    os.chdir(odir)
    for idate in range(ndate):
        if dateSecondary != []:
            if dates[idate] not in dateSecondary:
                continue

        os.makedirs(dates[idate], exist_ok=True)
        os.chdir(dates[idate])

        trackSecondary = loadTrack(dateDirs[idate], dates[idate])
        for i, frameNumber in enumerate(frames):
            frameDir = 'f{}_{}'.format(i+1, frameNumber)
            os.makedirs(frameDir, exist_ok=True)
            os.chdir(frameDir)
            for j, swathNumber in enumerate(range(swaths[0], swaths[-1] + 1)):
                swathDir = 's{}'.format(swathNumber)
                os.makedirs(swathDir, exist_ok=True)
                os.chdir(swathDir)

                #current swath
                swathReference = trackReference.frames[i].swaths[j]
                #swath of reference sample size
                swathReferenceReference = trackReference.frames[frameReferenceIndex].swaths[swathReferenceIndex]
                #current swath resampled
                swathReferenceResampled = trackReferenceResampled.frames[i].swaths[j]

                #current swath to be resampled
                swathSecondary = trackSecondary.frames[i].swaths[j]


                #current slc to be processed
                slc = os.path.join(dateDirs[idate], frameDir, swathDir, dates[idate]+'.slc')


                #0. create subband SLCs
                if subbandFlag:
                    subbandRadarWavelength, subbandBandWidth, subbandFrequencyCenter, subbandPrefix = subbandParameters(trackReference)

                    slcLower = dates[idate]+'_{}_tmp.slc'.format(subbandPrefix[0])
                    slcUpper = dates[idate]+'_{}_tmp.slc'.format(subbandPrefix[1])
                    rg_filter(slc, 2, 
                        [slcLower, slcUpper], 
                        subbandBandWidth, 
                        subbandFrequencyCenter, 
                        257, 2048, 0.1, 0, 0.0)
                    slcList = [slc, slcLower, slcUpper]
                    slcListResampled = [dates[idate]+'.slc', dates[idate]+'_{}.slc'.format(subbandPrefix[0]), dates[idate]+'_{}.slc'.format(subbandPrefix[1])]
                    slcListRemoved = [slcLower, slcUpper]
                else:
                    slcList = [slc]
                    slcListResampled = [dates[idate]+'.slc']
                    slcListRemoved = []


                #1. compute offset polynomial
                if idate == dateIndexReference:
                    rangePoly = Poly2D()
                    rangePoly.initPoly(rangeOrder=1,azimuthOrder=0,coeffs=[[
                        (swathReferenceResampled.startingRange - swathReference.startingRange) / swathReference.rangePixelSize, 
                        swathReferenceResampled.rangePixelSize / swathReference.rangePixelSize - 1.0]])
                
                    azimuthPoly = Poly2D()
                    azimuthPoly.initPoly(rangeOrder=0,azimuthOrder=1,coeffs=[
                        [(swathReferenceResampled.sensingStart - swathReference.sensingStart).total_seconds() / swathReference.azimuthLineInterval], 
                        [swathReferenceResampled.azimuthLineInterval / swathReference.azimuthLineInterval - 1.0]])

                    if DEBUG:
                        print()
                        print('rangePoly.getCoeffs(): {}'.format(rangePoly.getCoeffs()))
                        print('azimuthPoly.getCoeffs(): {}'.format(azimuthPoly.getCoeffs()))
                        print('rangePoly._meanRange: {}'.format(rangePoly._meanRange))
                        print('rangePoly._normRange: {}'.format(rangePoly._normRange))
                        print('rangePoly._meanAzimuth: {}'.format(rangePoly._meanAzimuth))
                        print('rangePoly._normAzimuth: {}'.format(rangePoly._normAzimuth))
                        print('azimuthPoly._meanRange: {}'.format(azimuthPoly._meanRange))
                        print('azimuthPoly._normRange: {}'.format(azimuthPoly._normRange))
                        print('azimuthPoly._meanAzimuth: {}'.format(azimuthPoly._meanAzimuth))
                        print('azimuthPoly._normAzimuth: {}'.format(azimuthPoly._normAzimuth))
                        print()

                else:
                    offsets = readOffset(os.path.join(dateDirs[idate], frameDir, swathDir, 'cull.off'))
                    #                                                   x1                      x2                 x3
                    #                                                   y1                      y2                 y3
                    #create new offset field to save offsets: swathReferenceResampled --> swathReference --> swathSecondary
                    offsetsUpdated = OffsetField()

                    for offset in offsets:
                        offsetUpdate = Offset()

                        x1 = offset.x * swathReference.rangePixelSize / swathReferenceResampled.rangePixelSize + \
                             (swathReference.startingRange - swathReferenceResampled.startingRange) / swathReferenceResampled.rangePixelSize
                        y1 = offset.y * swathReference.azimuthLineInterval / swathReferenceResampled.azimuthLineInterval + \
                             (swathReference.sensingStart - swathReferenceResampled.sensingStart).total_seconds() / swathReferenceResampled.azimuthLineInterval

                        x3 = offset.x + offset.dx
                        y3 = offset.y + offset.dy

                        dx = x3 - x1
                        dy = y3 - y1

                        offsetUpdate.setCoordinate(x1, y1)
                        offsetUpdate.setOffset(dx, dy)
                        offsetUpdate.setSignalToNoise(offset.snr)
                        offsetUpdate.setCovariance(offset.sigmax, offset.sigmay, offset.sigmaxy)
                        offsetsUpdated.addOffset(offsetUpdate)

                    azimuthPoly, rangePoly = offsetsUpdated.getFitPolynomials(rangeOrder=2,azimuthOrder=2,maxOrder=True, usenumpy=False)

                    #check polynomial accuracy
                    if DEBUG:
                        print()
                        print('       x            y            dx          dy         dx(poly)    dy(poly)    dx - dx(poly)  dy - dy(poly)')
                        print('==============================================================================================================')
                        for offset in offsetsUpdated:
                            print('%11.3f  %11.3f  %11.3f  %11.3f  %11.3f  %11.3f  %11.3f  %11.3f'%(offset.x, offset.y, 
                                offset.dx, offset.dy, 
                                rangePoly(offset.y, offset.x), azimuthPoly(offset.y, offset.x), 
                                offset.dx - rangePoly(offset.y, offset.x), offset.dy - azimuthPoly(offset.y, offset.x)))
                        print()

                    if DEBUG:
                        print()
                        print('rangePoly.getCoeffs(): {}'.format(rangePoly.getCoeffs()))
                        print('azimuthPoly.getCoeffs(): {}'.format(azimuthPoly.getCoeffs()))
                        print('rangePoly._meanRange: {}'.format(rangePoly._meanRange))
                        print('rangePoly._normRange: {}'.format(rangePoly._normRange))
                        print('rangePoly._meanAzimuth: {}'.format(rangePoly._meanAzimuth))
                        print('rangePoly._normAzimuth: {}'.format(rangePoly._normAzimuth))
                        print('azimuthPoly._meanRange: {}'.format(azimuthPoly._meanRange))
                        print('azimuthPoly._normRange: {}'.format(azimuthPoly._normRange))
                        print('azimuthPoly._meanAzimuth: {}'.format(azimuthPoly._meanAzimuth))
                        print('azimuthPoly._normAzimuth: {}'.format(azimuthPoly._normAzimuth))
                        print()


                #2. carrier phase
                dpoly = Poly2D()
                order = len(swathSecondary.dopplerVsPixel) - 1
                coeffs = [2*np.pi*val*swathSecondary.azimuthLineInterval for val in swathSecondary.dopplerVsPixel]
                dpoly.initPoly(rangeOrder=order, azimuthOrder=0)
                dpoly.setCoeffs([coeffs])

                #azCarrPoly = Poly2D()
                #azCarrPoly.initPoly(rangeOrder=0,azimuthOrder=0,coeffs=[[0.]])


                #3. resample images
                #checked: offset computation results using azimuthPoly/rangePoly and in resamp_slc.f90
                #checked: no flattenning
                #checked: no reading of range and azimuth images
                #checked: range/azimuth carrier values: 0, 0
                #checked: doppler no problem
                #         but doppler is computed using reference's coordinate in:
                #         isce/components/stdproc/stdproc/resamp_slc/src/resamp_slc.f90
                #         I have fixed it.

                
                for slcInput, slcOutput in zip(slcList, slcListResampled):
                    inimg = isceobj.createSlcImage()
                    inimg.load(slcInput + '.xml')
                    inimg.filename = slcInput
                    inimg.extraFilename = slcInput+'.vrt'
                    inimg.setAccessMode('READ')

                    rObj = stdproc.createResamp_slc()
                    #the following two items are actually not used, since we are not flattenning?
                    #but need to set these otherwise the program complains
                    rObj.slantRangePixelSpacing = swathSecondary.rangePixelSize
                    rObj.radarWavelength = trackSecondary.radarWavelength
                    #rObj.azimuthCarrierPoly = azCarrPoly
                    rObj.dopplerPoly = dpoly
                   
                    rObj.azimuthOffsetsPoly = azimuthPoly
                    rObj.rangeOffsetsPoly = rangePoly
                    rObj.imageIn = inimg

                    ####Setting reference values
                    #the following four items are actually not used, since we are not flattenning?
                    #but need to set these otherwise the program complains
                    rObj.startingRange = swathSecondary.startingRange
                    rObj.referenceSlantRangePixelSpacing = swathReferenceResampled.rangePixelSize
                    rObj.referenceStartingRange = swathReferenceResampled.startingRange
                    rObj.referenceWavelength = trackReferenceResampled.radarWavelength


                    width = swathReferenceResampled.numberOfSamples
                    length = swathReferenceResampled.numberOfLines
                    imgOut = isceobj.createSlcImage()
                    imgOut.setWidth(width)
                    imgOut.filename = slcOutput
                    imgOut.setAccessMode('write')

                    rObj.outputWidth = width
                    rObj.outputLines = length
                    #rObj.residualRangeImage = rngImg
                    #rObj.residualAzimuthImage = aziImg

                    rObj.resamp_slc(imageOut=imgOut)

                    imgOut.renderHdr()
                
                for x in slcListRemoved:
                    os.remove(x)
                    os.remove(x + '.vrt')
                    os.remove(x + '.xml')

                os.chdir('../')
            os.chdir('../')
        os.chdir('../')


    #dump resampled reference paramter files, only do this when reference is resampled
    dumpFlag = True
    if dateSecondary != []:
        if dates[dateIndexReference] not in dateSecondary:
            dumpFlag = False
    if dumpFlag:
        #we are still in directory 'odir'
        os.chdir(dates[dateIndexReference])
        saveTrack(trackReferenceResampled, dates[dateIndexReference])











