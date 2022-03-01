#!/usr/bin/env python3

# Author: Cunren Liang
# Copyright 2021


import os
import glob
import argparse
import numpy as np 

import isce
import isceobj

from isceobj.TopsProc.runMergeBursts import mergeBox
from isceobj.TopsProc.runMergeBursts import adjustValidWithLooks
from isceobj.TopsProc.runMergeBursts import mergeBurstsVirtual
from isceobj.TopsProc.runMergeBursts import multilook as multilook2

from Stack import ionParam
import s1a_isce_utils as ut

def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser(description='merge bursts for ionosphere estimation')
    parser.add_argument('-i', '--reference', type=str, dest='reference', required=True,
            help='directory with the reference image. will be merged in a box defined by reference')
    parser.add_argument('-s', '--stack', type=str, dest='stack', default = None,
                help='directory with the stack xml files which includes the common valid region of each burst in the stack')
    parser.add_argument('-d', '--dirname', type=str, dest='dirname', required=True,
                help='directory with products to merge')
    parser.add_argument('-n', '--name_pattern', type=str, dest='name_pattern', required=True,
                help = 'a name pattern of burst products that will be merged. e.g.: fine_*.int')
    parser.add_argument('-o', '--outfile', type=str, dest='outfile', required=True,
            help='output merged file')
    parser.add_argument('-r', '--nrlks', type=int, dest='nrlks', default=1,
            help = 'number of range looks')
    parser.add_argument('-a', '--nalks', type=int, dest='nalks', default=1,
            help = 'number of azimuth looks')
    parser.add_argument('-u', '--nrlks0', type=int, dest='nrlks0', default=1,
            help = 'number of range looks 0')
    parser.add_argument('-v', '--nalks0', type=int, dest='nalks0', default=1,
            help = 'number of azimuth looks 0')
    parser.add_argument('-x', '--rvalid', type=int, dest='rvalid', default=None,
            help = 'number of valid samples in a multilook window in range, 1<=rvalid<=nrlks. default: nrlks')
    parser.add_argument('-y', '--avalid', type=int, dest='avalid', default=None,
            help = 'number of valid lines in a multilook window in azimuth, 1<=avalid<=nalks. default: nalks')
    parser.add_argument('-w', '--swath', type=int, dest='swath', default=None,
            help = 'swaths to merge, 1 or 2 or 3. default: all swaths')

    return parser


def cmdLineParse(iargs = None):
    '''
    Command line parser.
    '''

    parser = createParser()
    inps = parser.parse_args(args=iargs)

    return inps


def updateValid(frame1, frame2):
    '''
    update frame1 valid with frame2 valid
    '''

    min1 = frame1.bursts[0].burstNumber
    max1 = frame1.bursts[-1].burstNumber

    min2 = frame2.bursts[0].burstNumber
    max2 = frame2.bursts[-1].burstNumber

    minBurst = max(min1, min2)
    maxBurst = min(max1, max2)

    for ii in range(minBurst, maxBurst + 1):
        frame1.bursts[ii-min1].firstValidLine   = frame2.bursts[ii-min2].firstValidLine
        frame1.bursts[ii-min1].firstValidSample = frame2.bursts[ii-min2].firstValidSample
        frame1.bursts[ii-min1].numValidLines    = frame2.bursts[ii-min2].numValidLines
        frame1.bursts[ii-min1].numValidSamples  = frame2.bursts[ii-min2].numValidSamples


    return


def main(iargs=None):
    '''
    merge bursts
    '''
    inps=cmdLineParse(iargs)

    if inps.rvalid is None:
        inps.rvalid = 'strict'
    else:
        if not (1 <= inps.rvalid <= inps.nrlks):
            raise Exception('1<=rvalid<=nrlks')
    if inps.avalid is None:
        inps.avalid = 'strict'
    else:
        if not (1 <= inps.avalid <= inps.nalks):
            raise Exception('1<=avalid<=nalks')

    namePattern = inps.name_pattern.split('*')

    frameReferenceList=[]
    frameProductList=[]
    burstList = []
    swathList = ut.getSwathList(inps.reference)
    for swath in swathList:
        frameReference = ut.loadProduct(os.path.join(inps.reference, 'IW{0}.xml'.format(swath)))

        minBurst = frameReference.bursts[0].burstNumber
        maxBurst = frameReference.bursts[-1].burstNumber
        if minBurst==maxBurst:
            print('Skipping processing of swath {0}'.format(swath))
            continue

        frameProduct = ut.loadProduct(os.path.join(inps.dirname, 'IW{0}.xml'.format(swath)))
        minBurst = frameProduct.bursts[0].burstNumber
        maxBurst = frameProduct.bursts[-1].burstNumber

        if inps.stack is not None:
            print('Updating the valid region of each burst to the common valid region of the stack')
            frameStack = ut.loadProduct(os.path.join(inps.stack, 'IW{0}.xml'.format(swath)))
            updateValid(frameReference, frameStack)
            updateValid(frameProduct, frameStack)


        frameReferenceList.append(frameReference)

        if inps.swath is not None:
            if swath == inps.swath:
                frameProductList.append(frameProduct)
                burstList.append([os.path.join(inps.dirname, 'IW{0}'.format(swath), namePattern[0]+'%02d'%(x)+namePattern[1]) for x in range(minBurst, maxBurst+1)])
        else:
            frameProductList.append(frameProduct)
            burstList.append([os.path.join(inps.dirname, 'IW{0}'.format(swath), namePattern[0]+'%02d'%(x)+namePattern[1]) for x in range(minBurst, maxBurst+1)])

    os.makedirs(os.path.dirname(inps.outfile), exist_ok=True)
    suffix = '.full'
    if (inps.nrlks0 == 1) and (inps.nalks0 == 1):
        suffix=''

    box = mergeBox(frameReferenceList)
    #adjust valid with looks, 'frames' ARE CHANGED AFTER RUNNING THIS
    #here numberRangeLooks, instead of numberRangeLooks0, is used, since we need to do next step multilooking after unwrapping. same for numberAzimuthLooks.
    (burstValidBox, burstValidBox2, message) = adjustValidWithLooks(frameProductList, box, inps.nalks, inps.nrlks, edge=0, avalid=inps.avalid, rvalid=inps.rvalid)
    mergeBurstsVirtual(frameProductList, burstList, box, inps.outfile+suffix)
    if suffix not in ['',None]:
        multilook2(inps.outfile+suffix,
          outname = inps.outfile,
          alks = inps.nalks0, rlks=inps.nrlks0)
    #this is never used for ionosphere correction
    else:
        print('Skipping multi-looking ....')


if __name__ == '__main__' :
    '''
    Merge products burst-by-burst.
    '''

    main()
