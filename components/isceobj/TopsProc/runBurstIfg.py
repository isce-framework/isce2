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

def multiply(masname, slvname, outname, rngname, fact, masterFrame,
        flatten=True, alks=3, rlks=7, virtual=True):


    masImg = isceobj.createSlcImage()
    masImg.load( masname + '.xml')

    width = masImg.getWidth()
    length = masImg.getLength()


    if not virtual:
        master = np.memmap(masname, dtype=np.complex64, mode='r', shape=(length,width))
    else:
        master = loadVirtualArray(masname + '.vrt')
    
    slave = np.memmap(slvname, dtype=np.complex64, mode='r', shape=(length, width))
   
    if os.path.exists(rngname):
        rng2 = np.memmap(rngname, dtype=np.float32, mode='r', shape=(length,width))
    else:
        print('No range offsets provided')
        rng2 = np.zeros((length,width))

    cJ = np.complex64(-1j)
    
    #Zero out anytging outside the valid region:
    ifg = np.memmap(outname, dtype=np.complex64, mode='w+', shape=(length,width))
    firstS = masterFrame.firstValidSample
    lastS = masterFrame.firstValidSample + masterFrame.numValidSamples -1
    firstL = masterFrame.firstValidLine
    lastL = masterFrame.firstValidLine + masterFrame.numValidLines - 1
    for kk in range(firstL,lastL + 1):
        ifg[kk,firstS:lastS + 1] = master[kk,firstS:lastS + 1] * np.conj(slave[kk,firstS:lastS + 1])
        if flatten:
            phs = np.exp(cJ*fact*rng2[kk,firstS:lastS + 1])
            ifg[kk,firstS:lastS + 1] *= phs

    ####
    master=None
    slave=None
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


def adjustValidLineSample(master,slave):

    master_lastValidLine = master.firstValidLine + master.numValidLines - 1
    master_lastValidSample = master.firstValidSample + master.numValidSamples - 1
    slave_lastValidLine = slave.firstValidLine + slave.numValidLines - 1
    slave_lastValidSample = slave.firstValidSample + slave.numValidSamples - 1

    igram_lastValidLine = min(master_lastValidLine, slave_lastValidLine)
    igram_lastValidSample = min(master_lastValidSample, slave_lastValidSample)

    master.firstValidLine = max(master.firstValidLine, slave.firstValidLine)
    master.firstValidSample = max(master.firstValidSample, slave.firstValidSample)

    master.numValidLines = igram_lastValidLine - master.firstValidLine + 1
    master.numValidSamples = igram_lastValidSample - master.firstValidSample + 1

def runBurstIfg(self):
    '''Create burst interferograms.
    '''

    virtual = self.useVirtualFiles

    swathList = self._insar.getValidSwathList(self.swaths)


    for swath in swathList:

        minBurst, maxBurst = self._insar.commonMasterBurstLimits(swath-1)
        nBurst = maxBurst - minBurst

        if nBurst == 0:
            continue
    
        ifgdir = os.path.join(self._insar.fineIfgDirname, 'IW{0}'.format(swath))
        if not os.path.exists(ifgdir):
            os.makedirs(ifgdir)

        ####Load relevant products
        master = self._insar.loadProduct( os.path.join(self._insar.masterSlcProduct, 'IW{0}.xml'.format(swath)))
        slave = self._insar.loadProduct( os.path.join(self._insar.fineCoregDirname, 'IW{0}.xml'.format(swath)))

        coregdir = os.path.join(self._insar.fineOffsetsDirname, 'IW{0}'.format(swath))

        fineIfg =  createTOPSSwathSLCProduct()
        fineIfg.configure()

        for ii in range(minBurst, maxBurst):
    
            jj = ii - minBurst


            ####Process the top bursts
            masBurst = master.bursts[ii] 
            slvBurst = slave.bursts[jj]

            mastername = masBurst.image.filename
            slavename = slvBurst.image.filename
            rdict = {'rangeOff' : os.path.join(coregdir, 'range_%02d.off'%(ii+1)),
                     'azimuthOff': os.path.join(coregdir, 'azimuth_%02d.off'%(ii+1))}
            
            
            adjustValidLineSample(masBurst,slvBurst)
       
           
            if self.doInSAR:
                intname = os.path.join(ifgdir, '%s_%02d.int'%('burst',ii+1))
                fact = 4 * np.pi * slvBurst.rangePixelSize / slvBurst.radarWavelength
                intimage = multiply(mastername, slavename, intname,
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
                computeCoherence(mastername, slavename, corname) 


        fineIfg.numberOfBursts = len(fineIfg.bursts)
        self._insar.saveProduct(fineIfg, os.path.join(self._insar.fineIfgDirname, 'IW{0}.xml'.format(swath)))
