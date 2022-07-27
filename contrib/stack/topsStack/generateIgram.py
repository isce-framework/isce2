#!/usr/bin/env python3

# Author: Piyush Agram
# Heresh Fattahi: Adopted for stack

import isce
import isceobj
import numpy as np
from isceobj.Util.Poly2D import Poly2D
import argparse
import os
import copy
from isceobj.Sensor.TOPS import createTOPSSwathSLCProduct 
from mroipac.correlation.correlation import Correlation
import s1a_isce_utils as ut
from osgeo import gdal


def createParser():
    parser = argparse.ArgumentParser( description='Use polynomial offsets and create burst by burst interferograms')

    parser.add_argument('-m', '--reference', dest='reference', type=str, required=True,
            help='Directory with reference acquisition')

    parser.add_argument('-x', '--reference_suffix', dest='reference_suffix', type=str, default=None,
            help='reference burst file name suffix')

    parser.add_argument('-s', '--secondary', dest='secondary', type=str, required=True,
            help='Directory with secondary acquisition')

    parser.add_argument('-y', '--secondary_suffix', dest='secondary_suffix', type=str, default=None,
            help='secondary burst file name suffix')

    parser.add_argument('-f', '--flatten', dest='flatten', action='store_true', default=False,
            help='Flatten the interferograms with offsets if needed')

    parser.add_argument('-i', '--interferogram', dest='interferogram', type=str, default='interferograms',
            help='Path for the interferogram')

    parser.add_argument('-p', '--interferogram_prefix', dest='intprefix', type=str, default='int',
            help='Prefix for the interferogram')
    parser.add_argument('-v', '--overlap', dest='overlap', action='store_true', default=False,
            help='Flatten the interferograms with offsets if needed')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def multiply(masname, slvname, outname, rngname1, rngname2, fact, referenceFrame,
        flatten=False):

    print('multiply')
    masImg = isceobj.createSlcImage()
    masImg.load( masname + '.xml')

    width = masImg.getWidth()
    length = masImg.getLength()

    ds = gdal.Open(masname + '.vrt', gdal.GA_ReadOnly)
    reference = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    ds = gdal.Open(slvname + '.vrt', gdal.GA_ReadOnly)
    secondary = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    print('read') 
    #reference = np.memmap(masname, dtype=np.complex64, mode='r', shape=(length,width))
    #secondary = np.memmap(slvname, dtype=np.complex64, mode='r', shape=(length, width))

    if os.path.exists(rngname1):
        rng1 = np.memmap(rngname1, dtype=np.float32, mode='r', shape=(length,width))
    else:
        print('No range offsets provided')
        rng1 = np.zeros((length,width))

    if os.path.exists(rngname2):
        rng2 = np.memmap(rngname2, dtype=np.float32, mode='r', shape=(length,width))
    else:
        print('No range offsets provided')
        rng2 = np.zeros((length,width))

    rng12 = rng2 - rng1

    cJ = np.complex64(-1j)

    #Zero out anytging outside the valid region:
    ifg = np.memmap(outname, dtype=np.complex64, mode='w+', shape=(length,width))
    firstS = referenceFrame.firstValidSample
    lastS = referenceFrame.firstValidSample + referenceFrame.numValidSamples -1
    firstL = referenceFrame.firstValidLine
    lastL = referenceFrame.firstValidLine + referenceFrame.numValidLines - 1
    for kk in range(firstL,lastL + 1):
        ifg[kk,firstS:lastS + 1] = reference[kk,firstS:lastS + 1] * np.conj(secondary[kk,firstS:lastS + 1])
        if flatten:
            phs = np.exp(cJ*fact*rng12[kk,firstS:lastS + 1])
            ifg[kk,firstS:lastS + 1] *= phs


    ####
    reference=None
    secondary=None
    ifg = None

    objInt = isceobj.createIntImage()
    objInt.setFilename(outname)
    objInt.setWidth(width)
    objInt.setLength(length)
    objInt.setAccessMode('READ')
    #objInt.createImage()
    #objInt.finalizeImage()
    objInt.renderHdr()
    objInt.renderVRT()
    return objInt


def main(iargs=None):
    '''Create overlap interferograms.
    '''
    inps=cmdLineParse(iargs)

    if inps.overlap:
        referenceSwathList = ut.getSwathList(os.path.join(inps.reference, 'overlap'))
        secondarySwathList = ut.getSwathList(os.path.join(inps.secondary, 'overlap'))
    else:
        referenceSwathList = ut.getSwathList(inps.reference)
        secondarySwathList = ut.getSwathList(inps.secondary)
    swathList = list(sorted(set(referenceSwathList+secondarySwathList)))

    for swath in swathList:
        IWstr = 'IW{0}'.format(swath)
        if inps.overlap:
            ifgdir = os.path.join(inps.interferogram, 'overlap', IWstr)
        else:
            ifgdir = os.path.join(inps.interferogram, IWstr)

        os.makedirs(ifgdir, exist_ok=True)

    ####Load relevant products
        if inps.overlap:
            topReference = ut.loadProduct(os.path.join(inps.reference , 'overlap','IW{0}_top.xml'.format(swath)))
            botReference = ut.loadProduct(os.path.join(inps.reference ,'overlap', 'IW{0}_bottom.xml'.format(swath)))
            topCoreg = ut.loadProduct(os.path.join(inps.secondary, 'overlap', 'IW{0}_top.xml'.format(swath)))
            botCoreg = ut.loadProduct(os.path.join(inps.secondary, 'overlap', 'IW{0}_bottom.xml'.format(swath)))

        else:
            topReference = ut.loadProduct(os.path.join(inps.reference , 'IW{0}.xml'.format(swath)))
            topCoreg = ut.loadProduct(os.path.join(inps.secondary , 'IW{0}.xml'.format(swath)))

        if inps.overlap:
            coregdir = os.path.join(inps.secondary, 'overlap', 'IW{0}'.format(swath))
        else:
            coregdir = os.path.join(inps.secondary,'IW{0}'.format(swath))
    
        topIfg = ut.coregSwathSLCProduct()
        topIfg.configure()

        if inps.overlap:
            botIfg = ut.coregSwathSLCProduct()
            botIfg.configure()

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

            print('matching burst numbers: ',reference.burstNumber, secondary.burstNumber)

            referencename = reference.image.filename
            secondaryname = secondary.image.filename

            if inps.reference_suffix is not None:
                referencename = os.path.splitext(referencename)[0] + inps.reference_suffix + os.path.splitext(referencename)[1]
            if inps.secondary_suffix is not None:
                secondaryname = os.path.splitext(secondaryname)[0] + inps.secondary_suffix + os.path.splitext(secondaryname)[1]

            if inps.overlap:
                rdict = { 'rangeOff1' : os.path.join(inps.reference, 'overlap', IWstr, 'range_top_%02d_%02d.off'%(ii,ii+1)),
                     'rangeOff2' : os.path.join(inps.secondary, 'overlap', IWstr, 'range_top_%02d_%02d.off'%(ii,ii+1)),
                     'azimuthOff': os.path.join(inps.secondary, 'overlap', IWstr, 'azimuth_top_%02d_%02d.off'%(ii,ii+1))}

                intname = os.path.join(ifgdir, '%s_top_%02d_%02d.int'%(inps.intprefix,ii,ii+1))
        
            else:

                rdict = {'rangeOff1' : os.path.join(inps.reference, IWstr, 'range_%02d.off'%(ii)),
                     'rangeOff2' : os.path.join(inps.secondary, IWstr, 'range_%02d.off'%(ii)),
                     'azimuthOff1': os.path.join(inps.secondary, IWstr, 'azimuth_%02d.off'%(ii))}
            
                intname = os.path.join(ifgdir, '%s_%02d.int'%(inps.intprefix,ii))


            ut.adjustCommonValidRegion(reference,secondary)
            fact = 4 * np.pi * secondary.rangePixelSize / secondary.radarWavelength
            intimage = multiply(referencename, secondaryname, intname,
                        rdict['rangeOff1'], rdict['rangeOff2'], fact, reference, flatten=inps.flatten)

            burst = copy.deepcopy(reference)
            burst.image = intimage
            burst.burstNumber = ii
            topIfg.bursts.append(burst)


            if inps.overlap:
                ####Process the bottom bursts
                reference = botReference.bursts[ii-minReference]
                secondary = botCoreg.bursts[ii-minSecondary]


                referencename =  reference.image.filename
                secondaryname = secondary.image.filename
#            rdict = {'rangeOff' : os.path.join(coregdir, 'range_bot_%02d_%02d.off'%(ii,ii+1)),
#                   'azimuthOff': os.path.join(coregdir, 'azimuth_bot_%02d_%02d.off'%(ii,ii+1))}

                rdict = { 'rangeOff1' : os.path.join(inps.reference, 'overlap', IWstr, 'range_bot_%02d_%02d.off'%(ii,ii+1)),
                     'rangeOff2' : os.path.join(inps.secondary, 'overlap', IWstr, 'range_bot_%02d_%02d.off'%(ii,ii+1)),
                    'azimuthOff': os.path.join(inps.secondary, 'overlap', IWstr, 'azimuth_bot_%02d_%02d.off'%(ii,ii+1))}


                print ('rdict: ', rdict)

                ut.adjustCommonValidRegion(reference,secondary)
                intname = os.path.join(ifgdir, '%s_bot_%02d_%02d.int'%(inps.intprefix,ii,ii+1))
                fact = 4 * np.pi * secondary.rangePixelSize / secondary.radarWavelength

            #intimage = multiply(referencename, secondaryname, intname,
            #        rdict['rangeOff'], fact, reference, flatten=True)

                intimage = multiply(referencename, secondaryname, intname,
                        rdict['rangeOff1'], rdict['rangeOff2'], fact, reference, flatten=inps.flatten)

                burst = copy.deepcopy(reference)
                burst.burstNumber = ii
                burst.image = intimage
                botIfg.bursts.append(burst)


        topIfg.numberOfBursts = len(topIfg.bursts)
        if hasattr(topCoreg, 'reference'):
            topIfg.reference = topCoreg.reference
        else:
            topIfg.reference = topReference.reference

        print('Type: ',type(topIfg.reference))

        if inps.overlap:
            ut.saveProduct(topIfg, ifgdir +  '_top.xml')
            botIfg.numberOfBursts = len(botIfg.bursts)
            botIfg.reference = botCoreg.reference
            print(botIfg.reference)
            ut.saveProduct(botIfg, ifgdir + '_bottom.xml')
        else:
            ut.saveProduct(topIfg, ifgdir + '.xml')    
        


if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()

