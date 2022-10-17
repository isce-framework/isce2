#!/usr/bin/env python3

# Author: Cunren Liang
# Copyright 2022

import os
import copy
import glob
import shutil
import argparse
import numpy as np

import isce
import isceobj
import s1a_isce_utils as ut
from VRTManager import Swath
from isceobj.TopsProc.runIon import computeDopplerOffset


def createParser():
    parser = argparse.ArgumentParser(description='compute burst phase ramps using azimuth ionospheric shift')
    parser.add_argument('-k', '--reference_stack', type=str, dest='reference_stack', required=True,
            help='Directory with the reference image of the stack')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
            help='input azimuth ionospheric shift')
    parser.add_argument('-o', '--output', dest='output', type=str, required=True,
            help='Directory with output')
    parser.add_argument('-r', '--nrlks', dest='nrlks', type=int, default=1, 
            help='number of range looks of azimuth ionospheric shift. Default: 1')
    parser.add_argument('-a', '--nalks', dest='nalks', type=int, default=1, 
            help='number of azimuth looks of azimuth ionospheric shift. Default: 1')
    parser.add_argument('-t', '--ion_height', dest='ion_height', type=float, default=200.0, 
            help='height of ionospheric layer above the Earth surface in km. Default: 200.0')

    return parser


def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def main(iargs=None):
    '''
    compute burst phase ramps using azimuth ionospheric shift
    both regular and subband bursts are merged in a box defined by reference of stack

    '''
    inps = cmdLineParse(iargs)

    #ionospheric layer height (m)
    ionHeight = inps.ion_height * 1000.0
    #earth's radius (m)
    earthRadius = 6371 * 1000.0

    img = isceobj.createImage()
    img.load(inps.input + '.xml')
    width = img.width
    length = img.length
    ionShift = np.fromfile(inps.input, dtype=np.float32).reshape(length, width)

    swathList = ut.getSwathList(inps.reference_stack)
    frameReferenceAll = [ut.loadProduct(os.path.join(inps.reference_stack, 'IW{0}.xml'.format(swath))) for swath in swathList]

    refSwaths = [Swath(x) for x in frameReferenceAll]
    topSwath = min(refSwaths, key = lambda x: x.sensingStart)
    botSwath = max(refSwaths, key = lambda x: x.sensingStop)
    leftSwath = min(refSwaths, key = lambda x: x.nearRange)
    rightSwath = max(refSwaths, key = lambda x: x.farRange)

    totalWidth  = int(np.round((rightSwath.farRange - leftSwath.nearRange)/leftSwath.dr + 1))
    totalLength = int(np.round((botSwath.sensingStop - topSwath.sensingStart).total_seconds()/topSwath.dt + 1 ))

    #!!!should CHECK if input ionospheric shift is really in this geometry
    nearRange = leftSwath.nearRange
    sensingStart = topSwath.sensingStart
    dr = leftSwath.dr
    dt = topSwath.dt

    for swath in swathList:
        frameReference = ut.loadProduct(os.path.join(inps.reference_stack, 'IW{0}.xml'.format(swath)))
        swathObj = Swath(frameReference)

        minBurst = frameReference.bursts[0].burstNumber
        maxBurst = frameReference.bursts[-1].burstNumber
        #!!!should CHECK if input ionospheric shift is really in this geometry
        if minBurst==maxBurst:
            print('Skipping processing of swath {0}'.format(swath))
            continue

        midBurstIndex = round((minBurst + maxBurst) / 2.0) - minBurst
        midBurst = frameReference.bursts[midBurstIndex]

        outputDir = os.path.join(inps.output, 'IW{0}'.format(swath))
        os.makedirs(outputDir, exist_ok=True)

        #compute mean offset: indices start from 0
        firstLine   = int(np.round((swathObj.sensingStart - sensingStart).total_seconds()/(dt*inps.nalks))) + 1
        lastLine    = int(np.round((swathObj.sensingStop - sensingStart).total_seconds()/(dt*inps.nalks))) - 1
        firstColumn = int(np.round((swathObj.nearRange - nearRange)/(dr*inps.nrlks)))     + 1
        lastColumn  = int(np.round((swathObj.farRange - nearRange)/(dr*inps.nrlks)))     - 1
        ionShiftSwath = ionShift[firstLine:lastLine+1, firstColumn:lastColumn+1]
        ionShiftSwathValid = ionShiftSwath[np.nonzero(ionShiftSwath!=0)]

        if ionShiftSwathValid.size == 0:
            ionShiftSwathMean = 0.0
            print('mean azimuth ionospheric shift of swath {}: {} single look azimuth lines'.format(swath, ionShiftSwathMean))
        else:
            ionShiftSwathMean = np.mean(ionShiftSwathValid, dtype=np.float64)
            print('mean azimuth ionospheric shift of swath {}: {} single look azimuth lines'.format(swath, ionShiftSwathMean))

        for burst in frameReference.bursts:
            #compute mean offset: indices start from 0
            firstLine   = int(np.round((burst.sensingStart - sensingStart).total_seconds()/(dt*inps.nalks))) + 1
            lastLine    = int(np.round((burst.sensingStop - sensingStart).total_seconds()/(dt*inps.nalks))) - 1
            firstColumn = int(np.round((burst.startingRange - nearRange)/(dr*inps.nrlks))) + 1
            lastColumn  = int(np.round((burst.startingRange + (burst.numberOfSamples - 1) * dr - nearRange)/(dr*inps.nrlks))) - 1
            ionShiftBurst = ionShift[firstLine:lastLine+1, firstColumn:lastColumn+1]
            
            ionShiftBurstValid = ionShiftBurst[np.nonzero(ionShiftBurst!=0)]
            if ionShiftBurstValid.size < (lastLine - firstLine + 1) * (lastColumn - firstColumn + 1) / 2.0:
                ionShiftBurstMean = 0.0
                print('mean azimuth ionospheric shift of burst {}: 0.0 single look azimuth lines'.format(burst.burstNumber))
            else:
                #ionShiftBurstMean should use both phaseRamp1 and phaseRamp2, while
                #ionShiftSwathMean should use phaseRamp2 only as in (to be consistent with) previous ESD
                #The above is tested against runIon.py in topsApp.py
                #ionShiftBurstMean = np.mean(ionShiftBurstValid, dtype=np.float64) - ionShiftSwathMean
                ionShiftBurstMean = np.mean(ionShiftBurstValid, dtype=np.float64)
                print('mean azimuth ionospheric shift of burst {}: {} single look azimuth lines'.format(burst.burstNumber, ionShiftBurstMean))

            #compute burst phase ramps
            (dopplerOffset, Ka) = computeDopplerOffset(burst, 1, burst.numberOfLines, 1, burst.numberOfSamples, nrlks=1, nalks=1)

            #satellite height
            satHeight = np.linalg.norm(burst.orbit.interpolateOrbit(burst.sensingMid, method='hermite').getPosition())
            #orgininal doppler offset should be multiplied by this ratio
            ratio = ionHeight/(satHeight-earthRadius)

            #phaseRamp1 and phaseRamp2 have same sign according to Liang et. al., 2019.
            #phase ramp due to imaging geometry
            phaseRamp1 = dopplerOffset * burst.azimuthTimeInterval * ratio * (burst.azimuthTimeInterval * Ka[None,:] * 4.0 * np.pi)
            #phase ramp due to non-zero doppler centroid frequency, ESD
            phaseRamp2 = dopplerOffset * burst.azimuthTimeInterval * Ka[None,:] * 2.0 * np.pi * burst.azimuthTimeInterval
            phaseRamp = (phaseRamp1 + phaseRamp2) * ionShiftBurstMean - phaseRamp2 * ionShiftSwathMean

            outfile = os.path.join(outputDir, '%s_%02d.float'%('burst', burst.burstNumber))
            
            phaseRamp.astype(np.float32).tofile(outfile)

            image = isceobj.createImage()
            image.setDataType('FLOAT')
            image.setFilename(outfile)
            image.extraFilename = outfile + '.vrt'
            image.setWidth(burst.numberOfSamples)
            image.setLength(burst.numberOfLines)
            image.renderHdr()



if __name__ == '__main__':
    '''
    Main driver.
    '''
    # Main Driver
    main()



