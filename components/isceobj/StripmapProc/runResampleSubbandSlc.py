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

logger = logging.getLogger('isce.insar.runResampleSubbandSlc')

def resampleSlc(masterFrame, slaveFrame, imageSlc2, radarWavelength, coregDir,
                azoffname, rgoffname, azpoly = None, rgpoly = None, misreg=False):
    logger.info("Resampling slave SLC")

    imageSlc1 =  masterFrame.getImage().filename

    inimg = isceobj.createSlcImage()
    inimg.load(imageSlc2 + '.xml')
    inimg.setAccessMode('READ')

    prf = slaveFrame.PRF

    doppler = slaveFrame._dopplerVsPixel
    factor = 1.0 # this should be zero for zero Doppler SLC.
    coeffs = [factor * 2*np.pi*val/prf/prf for val in doppler]

    dpoly = Poly2D()
    dpoly.initPoly(rangeOrder=len(coeffs)-1, azimuthOrder=0, coeffs=[coeffs])

    rObj = stdproc.createResamp_slc()
    rObj.slantRangePixelSpacing = slaveFrame.getInstrument().getRangePixelSize()
    #rObj.radarWavelength = slaveFrame.getInstrument().getRadarWavelength()
    rObj.radarWavelength = radarWavelength
    rObj.dopplerPoly = dpoly 

    # for now let's start with None polynomial. Later this should change to
    # the misregistration polynomial
    rObj.azimuthOffsetsPoly = azpoly
    rObj.rangeOffsetsPoly = rgpoly
    rObj.imageIn = inimg

    rngImg = isceobj.createImage()
    rngImg.load(rgoffname + '.xml')
    rngImg.setAccessMode('READ')

    aziImg = isceobj.createImage()
    aziImg.load(azoffname + '.xml')
    aziImg.setAccessMode('READ')

    width = rngImg.getWidth()
    length = rngImg.getLength()


    flatten = True
    rObj.flatten = flatten
    rObj.outputWidth = width
    rObj.outputLines = length
    rObj.residualRangeImage = rngImg
    rObj.residualAzimuthImage = aziImg

    if masterFrame is not None:
        rObj.startingRange = slaveFrame.startingRange
        rObj.referenceStartingRange = masterFrame.startingRange
        rObj.referenceSlantRangePixelSpacing = masterFrame.getInstrument().getRangePixelSize()
        rObj.referenceWavelength = radarWavelength
    
    # preparing the output directory for coregistered slave slc
    #coregDir = self.insar.coregDirname

    if os.path.isdir(coregDir):
        logger.info('Geometry directory {0} already exists.'.format(coregDir))
    else:
        os.makedirs(coregDir)

    # output file name of the coregistered slave slc
    img = slaveFrame.getImage() 
    coregFilename = os.path.join(coregDir , os.path.basename(img.filename))

    imgOut = isceobj.createSlcImage()
    imgOut.setWidth(width)
    imgOut.filename = coregFilename
    imgOut.setAccessMode('write')

    rObj.resamp_slc(imageOut=imgOut)

    imgOut.renderHdr()

    return coregFilename


def runResampleSubbandSlc(self, misreg=False):
    '''Run method for split spectrum.
    '''

    if not self.doSplitSpectrum:
        print('Split spectrum not requested. Skipping...')
        return
    
    masterFrame = self._insar.loadProduct( self._insar.masterSlcCropProduct)
    slaveFrame = self._insar.loadProduct( self._insar.slaveSlcCropProduct)

    if self.doRubbersheeting:
        print('Using rubber sheeted offsets for resampling sub-bands')
        azoffname = os.path.join( self.insar.offsetsDirname, self.insar.azimuthRubbersheetFilename)

    else:
        print('Using refined offsets for resampling sub-bands')
        azoffname = os.path.join( self.insar.offsetsDirname, self.insar.azimuthOffsetFilename)
    
    rgoffname = os.path.join( self.insar.offsetsDirname, self.insar.rangeOffsetFilename)
    azpoly = self.insar.loadProduct( os.path.join(self.insar.misregDirname, self.insar.misregFilename) + '_az.xml')
    rgpoly = self.insar.loadProduct( os.path.join(self.insar.misregDirname, self.insar.misregFilename) + '_rg.xml')


    imageSlc2 = os.path.join(self.insar.splitSpectrumDirname, self.insar.lowBandSlcDirname, 
                        os.path.basename(slaveFrame.getImage().filename))

    wvlL = self.insar.lowBandRadarWavelength
    coregDir = os.path.join(self.insar.coregDirname, self.insar.lowBandSlcDirname)
    
    lowbandCoregFilename = resampleSlc(masterFrame, slaveFrame, imageSlc2, wvlL, coregDir,
                azoffname, rgoffname, azpoly=azpoly, rgpoly=rgpoly,misreg=False)

    imageSlc2 = os.path.join(self.insar.splitSpectrumDirname, self.insar.highBandSlcDirname,
                        os.path.basename(slaveFrame.getImage().filename))
    wvlH = self.insar.highBandRadarWavelength
    coregDir = os.path.join(self.insar.coregDirname, self.insar.highBandSlcDirname)

    highbandCoregFilename = resampleSlc(masterFrame, slaveFrame, imageSlc2, wvlH, coregDir, 
                    azoffname, rgoffname, azpoly=azpoly, rgpoly=rgpoly, misreg=False)

    self.insar.lowBandSlc2 = lowbandCoregFilename
    self.insar.highBandSlc2 = highbandCoregFilename
    
