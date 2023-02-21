#!/usr/bin/env python3

#Author: Heresh Fattahi

import os
import argparse
import glob
import numpy as np
from osgeo import gdal
import isce
import isceobj
from isceobj.Sensor.TOPS import createTOPSSwathSLCProduct
from mroipac.correlation.correlation import Correlation
import s1a_isce_utils as ut


def createParser():
    parser = argparse.ArgumentParser( description='Extract valid overlap region for the stack')

    parser.add_argument('-m', '--reference', dest='reference', type=str, required=True,
            help='Directory with reference acquisition')

    parser.add_argument('-s', '--secondary', dest='secondary', type=str, required=True,
            help='Directory with secondary acquisition')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)

def updateValidRegion(topReference, secondaryPath, swath):

    #secondarySwathList = ut.getSwathList(secondary)
    #swathList = list(sorted(set(referenceSwathList+secondarySwathList)))

    #for swath in swathList:
        #IWstr = 'IW{0}'.format(swath)
    ####Load relevant products
        #topReference = ut.loadProduct(os.path.join(inps.reference , 'IW{0}.xml'.format(swath)))

    print(secondaryPath)
    topCoreg = ut.loadProduct(os.path.join(secondaryPath , 'IW{0}.xml'.format(swath)))

    topIfg = ut.coregSwathSLCProduct()
    topIfg.configure()


    minReference = topReference.bursts[0].burstNumber
    maxReference = topReference.bursts[-1].burstNumber

    minSecondary = topCoreg.bursts[0].burstNumber
    maxSecondary = topCoreg.bursts[-1].burstNumber

    minBurst = max(minSecondary, minReference)
    maxBurst = min(maxSecondary, maxReference)
    print ('minSecondary,maxSecondary',minSecondary, maxSecondary)
    print ('minReference,maxReference',minReference, maxReference)
    print ('minBurst, maxBurst: ', minBurst, maxBurst)

    for ii in range(minBurst, maxBurst + 1):

            ####Process the top bursts
        reference = topReference.bursts[ii-minReference]
        secondary  = topCoreg.bursts[ii-minSecondary]
        ut.adjustCommonValidRegion(reference,secondary)
        #topReference.bursts[ii-minReference].firstValidLine = reference.firstValidLine

    return topReference

def dropSecondarysWithDifferentNumberOfBursts(secondaryList, reference, swathList):
    '''Drop secondary acquisitions that have different number of bursts
    than the reference acquisition.
    '''
    print('checking the number of bursts in coreg_secondarys against the one in reference')
    secondaryList2Drop = []
    for swath in swathList:
        prodReference = ut.loadProduct(os.path.join(reference, 'IW{0}.xml'.format(swath)))
        numBursts = len(prodReference.bursts)

        for secondary in secondaryList:
            prodSecondary = ut.loadProduct(os.path.join(secondary, 'IW{0}.xml'.format(swath)))
            if len(prodSecondary.bursts) != numBursts:
                msg = 'WARNING: {} has different number of bursts ({}) than the reference {} ({}) for swath {}'.format(
                    os.path.basename(secondary), len(prodSecondary.bursts),
                    os.path.basename(reference), numBursts, swath)
                msg += ' --> exclude it for common region calculation'
                print(msg)
                secondaryList2Drop.append(secondary)

    secondaryList2Drop = list(sorted(set(secondaryList2Drop)))
    if len(secondaryList2Drop) == 0:
        print('all secondary images have the same number of bursts as the reference')

    secondaryList = list(sorted(set(secondaryList) - set(secondaryList2Drop)))

    return secondaryList


def main(iargs=None):
    '''extract common valid overlap region for the stack.
    '''
    inps=cmdLineParse(iargs)

    stackDir = os.path.join(os.path.dirname(inps.reference),'stack')
    if not os.path.exists(stackDir):
        print('creating ', stackDir)
        os.makedirs(stackDir)
    elif len(glob.glob(os.path.join(stackDir, '*.xml'))) > 0:
        print(stackDir , ' already exists.')
        print('Replacing reference with existing stack.')
        inps.reference = stackDir
        print('updating the valid overlap region of:')
        print(stackDir)

    referenceSwathList = ut.getSwathList(inps.reference)
    secondaryList = glob.glob(os.path.join(inps.secondary,'2*'))
    secondarySwathList = ut.getSwathList(secondaryList[0]) # assuming all secondarys have the same swaths
    swathList = list(sorted(set(referenceSwathList+secondarySwathList)))
    # discard secondarys with different number of bursts than the reference
    secondaryList = dropSecondarysWithDifferentNumberOfBursts(secondaryList, inps.reference, swathList)

    for swath in swathList:
        print('******************')
        print('swath: ', swath)
        ####Load relevant products
        topReference = ut.loadProduct(os.path.join(inps.reference , 'IW{0}.xml'.format(swath)))
        #print('reference.firstValidLine: ', topReference.bursts[4].firstValidLine)
        for secondary in secondaryList:
            topReference = updateValidRegion(topReference, secondary, swath)

        print('writing ', os.path.join(stackDir , 'IW{0}.xml'.format(swath)))
        ut.saveProduct(topReference, os.path.join(stackDir , 'IW{0}.xml'.format(swath)))
        os.makedirs(os.path.join(stackDir ,'IW{0}'.format(swath)), exist_ok=True)


if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()


#swathList = ut.getSwathList(reference)
#swathList[2]
#frames = []
#for swath in swathList:
#        ifg = ut.loadProduct(os.path.join(inps.reference , 'IW{0}.xml'.format(swath)))

#        if inps.isaligned:
#            reference = ifg.reference
#        else:
#            reference = ifg

#        minBurst = ifg.bursts[0].burstNumber
#        maxBurst = ifg.bursts[-1].burstNumber

#        if minBurst==maxBurst:
#            print('Skipping processing of swath {0}'.format(swath))
#            continue

#        frames.append(ifg)


#swaths = [Swath(x) for x in frame]

'''

slcPath = '/home/hfattahi/PROCESSDIR/MexicoCity_Test/TestStack_offsets/reference'
swath = ut.loadProduct(os.path.join(slcPath , 'IW{0}.xml'.format(2)))

tref = swath.sensingStart
rref = swath.bursts[0].startingRange
dt = swath.bursts[0].azimuthTimeInterval
dr = swath.bursts[0].rangePixelSize



print (slcPath)
for ind, burst in enumerate(swath.bursts):

    xoff = int(np.round( (burst.startingRange - rref)/dr))
    yoff = int(np.round( (burst.sensingStart - tref).total_seconds() / dt))
    tyoff = int(burst.firstValidLine)
    txoff = int(burst.firstValidSample)
    wysize = int(burst.numValidLines)
    wxsize = int(burst.numValidSamples)
    fyoff = int(yoff + burst.firstValidLine)
    fxoff = int(xoff + burst.firstValidSample)

    #print(xoff, fxoff)
    print(yoff, fyoff)
'''
