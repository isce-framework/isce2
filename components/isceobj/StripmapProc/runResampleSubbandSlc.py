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

# Modified by V. Brancato 10.14.2019 added "self" as input parameter of resampleSLC
def resampleSlc(self,referenceFrame, secondaryFrame, imageSlc2, radarWavelength, coregDir,
                azoffname, rgoffname, azpoly = None, rgpoly = None, misreg=False):
    logger.info("Resampling secondary SLC")

    imageSlc1 =  referenceFrame.getImage().filename

    inimg = isceobj.createSlcImage()
    inimg.load(imageSlc2 + '.xml')
    inimg.setAccessMode('READ')

    prf = secondaryFrame.PRF

    doppler = secondaryFrame._dopplerVsPixel
    factor = 1.0 # this should be zero for zero Doppler SLC.
    coeffs = [factor * 2*np.pi*val/prf/prf for val in doppler]

    dpoly = Poly2D()
    dpoly.initPoly(rangeOrder=len(coeffs)-1, azimuthOrder=0, coeffs=[coeffs])

    rObj = stdproc.createResamp_slc()
    rObj.slantRangePixelSpacing = secondaryFrame.getInstrument().getRangePixelSize()
    #rObj.radarWavelength = secondaryFrame.getInstrument().getRadarWavelength()
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

# Modified by V. Brancato on 10.14.2019 (if Rubbersheeting in range is turned on, flatten the interferogram during cross-correlation)
    if not self.doRubbersheetingRange:
       print('Rubber sheeting in range is turned off, flattening the interferogram during resampling')
       flatten = True
       print(flatten)
    else:
       print('Rubber sheeting in range is turned on, flattening the interferogram during interferogram formation')
       flatten=False
       print(flatten)
# end of Modification
       
    rObj.flatten = flatten
    rObj.outputWidth = width
    rObj.outputLines = length
    rObj.residualRangeImage = rngImg
    rObj.residualAzimuthImage = aziImg

    if referenceFrame is not None:
        rObj.startingRange = secondaryFrame.startingRange
        rObj.referenceStartingRange = referenceFrame.startingRange
        rObj.referenceSlantRangePixelSpacing = referenceFrame.getInstrument().getRangePixelSize()
        rObj.referenceWavelength = radarWavelength
    
    # preparing the output directory for coregistered secondary slc
    #coregDir = self.insar.coregDirname

    os.makedirs(coregDir, exist_ok=True)

    # output file name of the coregistered secondary slc
    img = secondaryFrame.getImage() 
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
    
    referenceFrame = self._insar.loadProduct( self._insar.referenceSlcCropProduct)
    secondaryFrame = self._insar.loadProduct( self._insar.secondarySlcCropProduct)

# Modified by V. Brancato 10.14.2019

    if self.doRubbersheetingAzimuth:
        print('Using rubber in azimuth sheeted offsets for resampling sub-bands')
        azoffname = os.path.join( self.insar.offsetsDirname, self.insar.azimuthRubbersheetFilename)

    else:
        print('Using refined offsets for resampling sub-bands')
        azoffname = os.path.join( self.insar.offsetsDirname, self.insar.azimuthOffsetFilename)
    
    if self.doRubbersheetingRange:
       print('Using rubber in range sheeted offsets for resampling sub-bands')
       rgoffname = os.path.join( self.insar.offsetsDirname, self.insar.rangeRubbersheetFilename)
    else:
       print('Using refined offsets for resampling sub-bands')
       rgoffname = os.path.join( self.insar.offsetsDirname, self.insar.rangeOffsetFilename)
# ****************** End of Modification
     
   # rgoffname = os.path.join( self.insar.offsetsDirname, self.insar.rangeOffsetFilename)
    azpoly = self.insar.loadProduct( os.path.join(self.insar.misregDirname, self.insar.misregFilename) + '_az.xml')
    rgpoly = self.insar.loadProduct( os.path.join(self.insar.misregDirname, self.insar.misregFilename) + '_rg.xml')


    imageSlc2 = os.path.join(self.insar.splitSpectrumDirname, self.insar.lowBandSlcDirname, 
                        os.path.basename(secondaryFrame.getImage().filename))

    wvlL = self.insar.lowBandRadarWavelength
    coregDir = os.path.join(self.insar.coregDirname, self.insar.lowBandSlcDirname)
    
    lowbandCoregFilename = resampleSlc(self,referenceFrame, secondaryFrame, imageSlc2, wvlL, coregDir,
                azoffname, rgoffname, azpoly=azpoly, rgpoly=rgpoly,misreg=False)

    imageSlc2 = os.path.join(self.insar.splitSpectrumDirname, self.insar.highBandSlcDirname,
                        os.path.basename(secondaryFrame.getImage().filename))
    wvlH = self.insar.highBandRadarWavelength
    coregDir = os.path.join(self.insar.coregDirname, self.insar.highBandSlcDirname)

    highbandCoregFilename = resampleSlc(self,referenceFrame, secondaryFrame, imageSlc2, wvlH, coregDir, 
                    azoffname, rgoffname, azpoly=azpoly, rgpoly=rgpoly, misreg=False)

    self.insar.lowBandSlc2 = lowbandCoregFilename
    self.insar.highBandSlc2 = highbandCoregFilename
    
