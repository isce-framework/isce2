#
#

import isce
import isceobj
import stdproc
from isceobj.Util.Poly2D import Poly2D
import logging
from isceobj.Util.decorators import use_api

import os
import numpy as np
import shelve

logger = logging.getLogger('isce.insar.runResampleSlc')

def runResampleSlc(self, kind='coarse'):
    '''
    Kind can either be coarse, refined or fine.
    '''

    if kind not in ['coarse', 'refined', 'fine']:
        raise Exception('Unknown operation type {0} in runResampleSlc'.format(kind))

    if kind == 'fine':
        if not (self.doRubbersheetingRange | self.doRubbersheetingAzimuth): # Modified by V. Brancato 10.10.2019
            print('Rubber sheeting not requested, skipping resampling ....')
            return

    logger.info("Resampling secondary SLC")

    secondaryFrame = self._insar.loadProduct( self._insar.secondarySlcCropProduct)
    referenceFrame = self._insar.loadProduct( self._insar.referenceSlcCropProduct)

    inimg = isceobj.createSlcImage()
    inimg.load(secondaryFrame.getImage().filename + '.xml')
    inimg.setAccessMode('READ')

    prf = secondaryFrame.PRF

    doppler = secondaryFrame._dopplerVsPixel
    coeffs = [2*np.pi*val/prf for val in doppler]
    
    dpoly = Poly2D()
    dpoly.initPoly(rangeOrder=len(coeffs)-1, azimuthOrder=0, coeffs=[coeffs])

    rObj = stdproc.createResamp_slc()
    rObj.slantRangePixelSpacing = secondaryFrame.getInstrument().getRangePixelSize()
    rObj.radarWavelength = secondaryFrame.getInstrument().getRadarWavelength() 
    rObj.dopplerPoly = dpoly 

    # for now let's start with None polynomial. Later this should change to
    # the misregistration polynomial

    misregFile = os.path.join(self.insar.misregDirname, self.insar.misregFilename)
    if ((kind in ['refined','fine']) and os.path.exists(misregFile+'_az.xml')):
        azpoly = self._insar.loadProduct(misregFile + '_az.xml')
        rgpoly = self._insar.loadProduct(misregFile + '_rg.xml')
    else:
        print(misregFile , " does not exist.")
        azpoly = None
        rgpoly = None

    rObj.azimuthOffsetsPoly = azpoly
    rObj.rangeOffsetsPoly = rgpoly
    rObj.imageIn = inimg

    #Since the app is based on geometry module we expect pixel-by-pixel offset
    #field
    offsetsDir = self.insar.offsetsDirname 
    
    # Modified by V. Brancato 10.10.2019
    #rgname = os.path.join(offsetsDir, self.insar.rangeOffsetFilename)
    
    if kind in ['coarse', 'refined']:
        azname = os.path.join(offsetsDir, self.insar.azimuthOffsetFilename)
        rgname = os.path.join(offsetsDir, self.insar.rangeOffsetFilename)
        flatten = True
    else:
        azname = os.path.join(offsetsDir, self.insar.azimuthRubbersheetFilename)
        if self.doRubbersheetingRange:
           print('Rubbersheeting in range is turned on, taking the cross-correlation offsets') 
           print('Setting Flattening to False') 
           rgname = os.path.join(offsetsDir, self.insar.rangeRubbersheetFilename) 
           flatten=False
        else:
           print('Rubbersheeting in range is turned off, taking range geometric offsets')
           rgname = os.path.join(offsetsDir, self.insar.rangeOffsetFilename)
           flatten=True
    
    rngImg = isceobj.createImage()
    rngImg.load(rgname + '.xml')
    rngImg.setAccessMode('READ')

    aziImg = isceobj.createImage()
    aziImg.load(azname + '.xml')
    aziImg.setAccessMode('READ')

    width = rngImg.getWidth()
    length = rngImg.getLength()

# Modified by V. Brancato 10.10.2019
    #flatten = True
    rObj.flatten = flatten
    rObj.outputWidth = width
    rObj.outputLines = length
    rObj.residualRangeImage = rngImg
    rObj.residualAzimuthImage = aziImg

    if referenceFrame is not None:
        rObj.startingRange = secondaryFrame.startingRange
        rObj.referenceStartingRange = referenceFrame.startingRange
        rObj.referenceSlantRangePixelSpacing = referenceFrame.getInstrument().getRangePixelSize()
        rObj.referenceWavelength = referenceFrame.getInstrument().getRadarWavelength()

    
    # preparing the output directory for coregistered secondary slc
    coregDir = self.insar.coregDirname

    os.makedirs(coregDir, exist_ok=True)

    # output file name of the coregistered secondary slc
    img = secondaryFrame.getImage()

    if kind  == 'coarse':
        coregFilename = os.path.join(coregDir , self._insar.coarseCoregFilename)
    elif kind == 'refined':
        coregFilename = os.path.join(coregDir, self._insar.refinedCoregFilename)
    elif kind == 'fine':
        coregFilename = os.path.join(coregDir, self._insar.fineCoregFilename)
    else:
        print('Exception: Should not have gotten to this stage')

    imgOut = isceobj.createSlcImage()
    imgOut.setWidth(width)
    imgOut.filename = coregFilename
    imgOut.setAccessMode('write')

    rObj.resamp_slc(imageOut=imgOut)

    imgOut.renderHdr()

    return

