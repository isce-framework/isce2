#
# Author: Piyush Agram
# Copyright 2016
#


import numpy as np 
import os
import isceobj
import logging
import datetime
from isceobj.Location.Offset import OffsetField, Offset

logger = logging.getLogger('isce.topsinsar.rangecoreg')

def runAmpcor(reference, secondary):
    '''
    Run one ampcor process.
    '''
    import isceobj
    from mroipac.ampcor.Ampcor import Ampcor

    mImg = isceobj.createSlcImage()
    mImg.load(reference + '.xml')
    mImg.setAccessMode('READ')
    mImg.createImage()

    sImg = isceobj.createSlcImage()
    sImg.load(secondary + '.xml')
    sImg.setAccessMode('READ')
    sImg.createImage()

    objAmpcor = Ampcor('ampcor_burst')
    objAmpcor.configure()
    objAmpcor.setImageDataType1('mag')
    objAmpcor.setImageDataType2('mag')


    if objAmpcor.acrossGrossOffset is None:
        coarseAcross = 0

    if objAmpcor.downGrossOffset is None:
        coarseDown = 0

    objAmpcor.windowSizeWidth = 64
    objAmpcor.windowSizeHeight = 32
    objAmpcor.searchWindowSizeWidth = 16
    objAmpcor.searchWindowSizeHeight = 16
    objAmpcor.oversamplingFactor = 32

    xMargin = 2*objAmpcor.searchWindowSizeWidth + objAmpcor.windowSizeWidth
    yMargin = 2*objAmpcor.searchWindowSizeHeight + objAmpcor.windowSizeHeight

    firstAc = 1000

    #####Compute image positions

    offDn = objAmpcor.windowSizeHeight//2 + 1
    offAc = firstAc+xMargin

    offDnmax = mImg.getLength() - objAmpcor.windowSizeHeight//2 - 1
    lastAc = int(mImg.width - 1000 - xMargin)

    if not objAmpcor.firstSampleAcross:
        objAmpcor.setFirstSampleAcross(offAc)

    if not objAmpcor.lastSampleAcross:
        objAmpcor.setLastSampleAcross(lastAc)

    if not objAmpcor.numberLocationAcross:
        objAmpcor.setNumberLocationAcross(40)

    if not objAmpcor.firstSampleDown:
        objAmpcor.setFirstSampleDown(offDn)

    if not objAmpcor.lastSampleDown:
        objAmpcor.setLastSampleDown(offDnmax)

    ###Since we are only dealing with overlaps
    objAmpcor.setNumberLocationDown(20)

    #####Override gross offsets if not provided
    if not objAmpcor.acrossGrossOffset:
        objAmpcor.setAcrossGrossOffset(coarseAcross)

    if not objAmpcor.downGrossOffset:
        objAmpcor.setDownGrossOffset(coarseDown)


    objAmpcor.setImageDataType1('mag')
    objAmpcor.setImageDataType2('mag')

    objAmpcor.setFirstPRF(1.0)
    objAmpcor.setSecondPRF(1.0)
    objAmpcor.setFirstRangeSpacing(1.0)
    objAmpcor.setSecondRangeSpacing(1.0)
    objAmpcor(mImg, sImg)

    mImg.finalizeImage()
    sImg.finalizeImage()

    return objAmpcor.getOffsetField()


def runRangeCoreg(self, debugPlot=True):
    '''
    Estimate constant offset in range.
    '''

    if not self.doESD:
        return 

    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)

    swathList = self._insar.getValidSwathList(self.swaths)

    rangeOffsets = []
    snr = []

    for swath in swathList:

        if self._insar.numberOfCommonBursts[swath-1] < 2:
            print('Skipping range coreg for swath IW{0}'.format(swath))
            continue

        minBurst, maxBurst = self._insar.commonReferenceBurstLimits(swath-1)

        maxBurst = maxBurst - 1  ###For overlaps 
    
        referenceTop = self._insar.loadProduct( os.path.join(self._insar.referenceSlcOverlapProduct, 'top_IW{0}.xml'.format(swath)))
        referenceBottom  = self._insar.loadProduct( os.path.join(self._insar.referenceSlcOverlapProduct , 'bottom_IW{0}.xml'.format(swath)))

        secondaryTop = self._insar.loadProduct( os.path.join(self._insar.coregOverlapProduct , 'top_IW{0}.xml'.format(swath)))
        secondaryBottom = self._insar.loadProduct( os.path.join(self._insar.coregOverlapProduct, 'bottom_IW{0}.xml'.format(swath)))

        for pair in [(referenceTop,secondaryTop), (referenceBottom,secondaryBottom)]:
            for ii in range(minBurst,maxBurst):
                mFile = pair[0].bursts[ii-minBurst].image.filename
                sFile = pair[1].bursts[ii-minBurst].image.filename
            
                field = runAmpcor(mFile, sFile)

                for offset in field:
                    rangeOffsets.append(offset.dx)
                    snr.append(offset.snr)

    ###Cull 
    mask = np.logical_and(np.array(snr) >  self.offsetSNRThreshold, np.abs(rangeOffsets) < 1.2)
    val = np.array(rangeOffsets)[mask]

    medianval = np.median(val)
    meanval = np.mean(val)
    stdval = np.std(val)

    hist, bins = np.histogram(val, 50, normed=1)
    center = 0.5*(bins[:-1] + bins[1:])


    try:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
    except:
        print('Matplotlib could not be imported. Skipping debug plot ...')
        debugPlot = False

    if debugPlot:

        try:
            ####Plotting
            plt.figure()
            plt.bar(center, hist, align='center', width = 0.7*(bins[1] - bins[0]))
            plt.xlabel('Range shift in pixels')
            plt.savefig( os.path.join(self._insar.esdDirname, 'rangeMisregistration.jpg'))
            plt.close()
        except:
            print('Looks like matplotlib could not save image to JPEG, continuing .....')
            print('Install Pillow to ensure debug plots for Residual range offsets are generated.')
            pass


    catalog.addItem('Median', medianval, 'esd')
    catalog.addItem('Mean', meanval, 'esd')
    catalog.addItem('Std', stdval, 'esd')
    catalog.addItem('snr threshold', self.offsetSNRThreshold, 'esd')
    catalog.addItem('number of coherent points', val.size, 'esd')

    catalog.printToLog(logger, "runRangeCoreg")
    self._insar.procDoc.addAllFromCatalog(catalog)

    self._insar.secondaryRangeCorrection = meanval * referenceTop.bursts[0].rangePixelSize
