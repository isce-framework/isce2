#
# Author: Piyush Agram
# Copyright 2016
#

import isceobj
import stdproc
from stdproc.stdproc import crossmul
import numpy as np
from isceobj.Util.Poly2D import Poly2D
import argparse
import os
import copy
from isceobj.Sensor.TOPS import createTOPSSwathSLCProduct
from mroipac.correlation.correlation import Correlation

def loadVirtualArray(fname):
    from osgeo import gdal

    ds = gdal.Open(fname, gdal.GA_ReadOnly)
    data = ds.GetRasterBand(1).ReadAsArray()

    ds = None
    return data

def multiply(masname, slvname, outname, rngname, fact, referenceFrame,
        flatten=True, alks=3, rlks=7, virtual=True):


    masImg = isceobj.createSlcImage()
    masImg.load( masname + '.xml')

    width = masImg.getWidth()
    length = masImg.getLength()


    if not virtual:
        reference = np.memmap(masname, dtype=np.complex64, mode='r', shape=(length,width))
    else:
        reference = loadVirtualArray(masname + '.vrt')
    
    secondary = np.memmap(slvname, dtype=np.complex64, mode='r', shape=(length, width))
   
    if os.path.exists(rngname):
        rng2 = np.memmap(rngname, dtype=np.float32, mode='r', shape=(length,width))
    else:
        print('No range offsets provided')
        rng2 = np.zeros((length,width))

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
            phs = np.exp(cJ*fact*rng2[kk,firstS:lastS + 1])
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
    objInt.renderHdr()


    try:
        takeLooks(objInt, alks, rlks)
    except:
        raise Exception('Failed to multilook ifg: {0}'.format(objInt.filename))

    return objInt


def takeLooks(inimg, alks, rlks):
    '''
    Take looks.
    '''

    from mroipac.looks.Looks import Looks

    spl = os.path.splitext(inimg.filename)
    ext = '.{0}alks_{1}rlks'.format(alks, rlks)
    outfile = spl[0] + ext + spl[1]


    lkObj = Looks()
    lkObj.setDownLooks(alks)
    lkObj.setAcrossLooks(rlks)
    lkObj.setInputImage(inimg)
    lkObj.setOutputFilename(outfile)
    lkObj.looks()

    return outfile

def computeCoherence(slc1name, slc2name, corname, virtual=True):
                          
    slc1 = isceobj.createImage()
    slc1.load( slc1name + '.xml')
    slc1.createImage()


    slc2 = isceobj.createImage()
    slc2.load( slc2name + '.xml')
    slc2.createImage()

    cohImage = isceobj.createOffsetImage()
    cohImage.setFilename(corname)
    cohImage.setWidth(slc1.getWidth())
    cohImage.setAccessMode('write')
    cohImage.createImage()

    cor = Correlation()
    cor.configure()
    cor.wireInputPort(name='slc1', object=slc1)
    cor.wireInputPort(name='slc2', object=slc2)
    cor.wireOutputPort(name='correlation', object=cohImage)
    cor.coregisteredSlcFlag = True
    cor.calculateCorrelation()

    cohImage.finalizeImage()
    slc1.finalizeImage()
    slc2.finalizeImage()
    return


def adjustValidLineSample(reference,secondary):

    reference_lastValidLine = reference.firstValidLine + reference.numValidLines - 1
    reference_lastValidSample = reference.firstValidSample + reference.numValidSamples - 1
    secondary_lastValidLine = secondary.firstValidLine + secondary.numValidLines - 1
    secondary_lastValidSample = secondary.firstValidSample + secondary.numValidSamples - 1

    igram_lastValidLine = min(reference_lastValidLine, secondary_lastValidLine)
    igram_lastValidSample = min(reference_lastValidSample, secondary_lastValidSample)

    reference.firstValidLine = max(reference.firstValidLine, secondary.firstValidLine)
    reference.firstValidSample = max(reference.firstValidSample, secondary.firstValidSample)

    reference.numValidLines = igram_lastValidLine - reference.firstValidLine + 1
    reference.numValidSamples = igram_lastValidSample - reference.firstValidSample + 1

def runBurstIfg(self):
    '''Create burst interferograms.
    '''

    virtual = self.useVirtualFiles

    swathList = self._insar.getValidSwathList(self.swaths)


    for swath in swathList:

        minBurst, maxBurst = self._insar.commonReferenceBurstLimits(swath-1)
        nBurst = maxBurst - minBurst

        if nBurst == 0:
            continue
    
        ifgdir = os.path.join(self._insar.fineIfgDirname, 'IW{0}'.format(swath))
        os.makedirs(ifgdir, exist_ok=True)

        ####Load relevant products
        reference = self._insar.loadProduct( os.path.join(self._insar.referenceSlcProduct, 'IW{0}.xml'.format(swath)))
        secondary = self._insar.loadProduct( os.path.join(self._insar.fineCoregDirname, 'IW{0}.xml'.format(swath)))

        coregdir = os.path.join(self._insar.fineOffsetsDirname, 'IW{0}'.format(swath))

        fineIfg =  createTOPSSwathSLCProduct()
        fineIfg.configure()

        for ii in range(minBurst, maxBurst):
    
            jj = ii - minBurst


            ####Process the top bursts
            masBurst = reference.bursts[ii] 
            slvBurst = secondary.bursts[jj]

            referencename = masBurst.image.filename
            secondaryname = slvBurst.image.filename
            rdict = {'rangeOff' : os.path.join(coregdir, 'range_%02d.off'%(ii+1)),
                     'azimuthOff': os.path.join(coregdir, 'azimuth_%02d.off'%(ii+1))}
            
            
            adjustValidLineSample(masBurst,slvBurst)
       
           
            if self.doInSAR:
                intname = os.path.join(ifgdir, '%s_%02d.int'%('burst',ii+1))
                fact = 4 * np.pi * slvBurst.rangePixelSize / slvBurst.radarWavelength
                intimage = multiply(referencename, secondaryname, intname,
                        rdict['rangeOff'], fact, masBurst, flatten=True,
                        alks = self.numberAzimuthLooks, rlks=self.numberRangeLooks,
                        virtual=virtual)

            burst = masBurst.clone()

            if self.doInSAR:
                burst.image = intimage

            fineIfg.bursts.append(burst)


            if self.doInSAR:
                ####Estimate coherence
                corname =  os.path.join(ifgdir, '%s_%02d.cor'%('burst',ii+1))
                computeCoherence(referencename, secondaryname, corname) 


        fineIfg.numberOfBursts = len(fineIfg.bursts)
        self._insar.saveProduct(fineIfg, os.path.join(self._insar.fineIfgDirname, 'IW{0}.xml'.format(swath)))
