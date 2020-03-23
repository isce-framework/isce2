
#
# Author: Heresh Fattahi, 2017
# Modified by V. Brancato (10.2019)
#         (Included flattening when rubbersheeting in range is turned on

import isceobj
import logging
from components.stdproc.stdproc import crossmul
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
import os
import gdal
import numpy as np

logger = logging.getLogger('isce.insar.runInterferogram')

# Added by V. Brancato 10.09.2019
def write_xml(fileName,width,length,bands,dataType,scheme):

    img = isceobj.createImage()
    img.setFilename(fileName)
    img.setWidth(width)
    img.setLength(length)
    img.setAccessMode('READ')
    img.bands = bands
    img.dataType = dataType
    img.scheme = scheme
    img.renderHdr()
    img.renderVRT()
    
    return None

    	    
def compute_FlatEarth(self,ifgFilename,width,length,radarWavelength):
    from imageMath import IML
    import logging
    
    # If rubbersheeting has been performed add back the range sheet offsets
    
    info = self._insar.loadProduct(self._insar.slaveSlcCropProduct)
    #radarWavelength = info.getInstrument().getRadarWavelength() 
    rangePixelSize = info.getInstrument().getRangePixelSize()
    fact = 4 * np.pi* rangePixelSize / radarWavelength

    cJ = np.complex64(-1j)

    # Open the range sheet offset
    rngOff = os.path.join(self.insar.offsetsDirname, self.insar.rangeOffsetFilename )
    
    print(rngOff)
    if os.path.exists(rngOff):
       rng2 = np.memmap(rngOff, dtype=np.float64, mode='r', shape=(length,width))
    else:
       print('No range offsets provided')
       rng2 = np.zeros((length,width))
    
    # Open the interferogram
    #ifgFilename= os.path.join(self.insar.ifgDirname, self.insar.ifgFilename)
    intf = np.memmap(ifgFilename,dtype=np.complex64,mode='r+',shape=(length,width))
   
    for ll in range(length):
        intf[ll,:] *= np.exp(cJ*fact*rng2[ll,:])
    
    del rng2
    del intf
       
    return 
    
    

def multilook(infile, outname=None, alks=5, rlks=15):
    '''
    Take looks.
    '''

    from mroipac.looks.Looks import Looks

    print('Multilooking {0} ...'.format(infile))

    inimg = isceobj.createImage()
    inimg.load(infile + '.xml')

    if outname is None:
        spl = os.path.splitext(inimg.filename)
        ext = '.{0}alks_{1}rlks'.format(alks, rlks)
        outname = spl[0] + ext + spl[1]

    lkObj = Looks()
    lkObj.setDownLooks(alks)
    lkObj.setAcrossLooks(rlks)
    lkObj.setInputImage(inimg)
    lkObj.setOutputFilename(outname)
    lkObj.looks()

    return outname

def computeCoherence(slc1name, slc2name, corname, virtual=True):
    from mroipac.correlation.correlation import Correlation
                          
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

# Modified by V. Brancato on 10.09.2019 (added self)
# Modified by V. Brancato on 11.13.2019 (added radar wavelength for low and high band flattening
def generateIgram(self,imageSlc1, imageSlc2, resampName, azLooks, rgLooks,radarWavelength):
    objSlc1 = isceobj.createSlcImage()
    IU.copyAttributes(imageSlc1, objSlc1)
    objSlc1.setAccessMode('read')
    objSlc1.createImage()

    objSlc2 = isceobj.createSlcImage()
    IU.copyAttributes(imageSlc2, objSlc2)
    objSlc2.setAccessMode('read')
    objSlc2.createImage()

    slcWidth = imageSlc1.getWidth()
    
    
    if not self.doRubbersheetingRange:
     intWidth = int(slcWidth/rgLooks)    # Modified by V. Brancato intWidth = int(slcWidth / rgLooks)
    else:
     intWidth = int(slcWidth)
    
    lines = min(imageSlc1.getLength(), imageSlc2.getLength())

    if '.flat' in resampName:
        resampAmp = resampName.replace('.flat', '.amp')
    elif '.int' in resampName:
        resampAmp = resampName.replace('.int', '.amp')
    else:
        resampAmp += '.amp'

    if not self.doRubbersheetingRange:
        resampInt = resampName
    else:
        resampInt = resampName + ".full"

    objInt = isceobj.createIntImage()
    objInt.setFilename(resampInt)
    objInt.setWidth(intWidth)
    imageInt = isceobj.createIntImage()
    IU.copyAttributes(objInt, imageInt)
    objInt.setAccessMode('write')
    objInt.createImage()

    objAmp = isceobj.createAmpImage()
    objAmp.setFilename(resampAmp)
    objAmp.setWidth(intWidth)
    imageAmp = isceobj.createAmpImage()
    IU.copyAttributes(objAmp, imageAmp)
    objAmp.setAccessMode('write')
    objAmp.createImage()
    
    if not self.doRubbersheetingRange:
       print('Rubbersheeting in range is off, interferogram is already flattened')
       objCrossmul = crossmul.createcrossmul()
       objCrossmul.width = slcWidth
       objCrossmul.length = lines
       objCrossmul.LooksDown = azLooks
       objCrossmul.LooksAcross = rgLooks

       objCrossmul.crossmul(objSlc1, objSlc2, objInt, objAmp)
    else:
     # Modified by V. Brancato 10.09.2019 (added option to add Range Rubber sheet Flat-earth back)
       print('Rubbersheeting in range is on, removing flat-Earth phase')
       objCrossmul = crossmul.createcrossmul()
       objCrossmul.width = slcWidth
       objCrossmul.length = lines
       objCrossmul.LooksDown = 1
       objCrossmul.LooksAcross = 1
       objCrossmul.crossmul(objSlc1, objSlc2, objInt, objAmp)
       
       # Remove Flat-Earth component
       compute_FlatEarth(self,resampInt,intWidth,lines,radarWavelength)
       
       # Perform Multilook
       multilook(resampInt, outname=resampName, alks=azLooks, rlks=rgLooks)  #takeLooks(objAmp,azLooks,rgLooks)
       multilook(resampAmp, outname=resampAmp.replace(".full",""), alks=azLooks, rlks=rgLooks)  #takeLooks(objInt,azLooks,rgLooks)
       
       #os.system('rm ' + resampInt+'.full* ' + resampAmp + '.full* ')
       # End of modification 
    for obj in [objInt, objAmp, objSlc1, objSlc2]:
        obj.finalizeImage()

    return imageInt, imageAmp


def subBandIgram(self, masterSlc, slaveSlc, subBandDir,radarWavelength):

    img1 = isceobj.createImage()
    img1.load(masterSlc + '.xml')

    img2 = isceobj.createImage()
    img2.load(slaveSlc + '.xml')

    azLooks = self.numberAzimuthLooks
    rgLooks = self.numberRangeLooks

    ifgDir = os.path.join(self.insar.ifgDirname, subBandDir)

    if os.path.isdir(ifgDir):
        logger.info('Interferogram directory {0} already exists.'.format(ifgDir))
    else:
        os.makedirs(ifgDir)

    interferogramName = os.path.join(ifgDir , self.insar.ifgFilename)

    generateIgram(self,img1, img2, interferogramName, azLooks, rgLooks,radarWavelength)
    
    return interferogramName

def runSubBandInterferograms(self):
    
    logger.info("Generating sub-band interferograms")

    masterFrame = self._insar.loadProduct( self._insar.masterSlcCropProduct)
    slaveFrame = self._insar.loadProduct( self._insar.slaveSlcCropProduct)

    azLooks, rgLooks = self.insar.numberOfLooks( masterFrame, self.posting,
                                        self.numberAzimuthLooks, self.numberRangeLooks)

    self.numberAzimuthLooks = azLooks
    self.numberRangeLooks = rgLooks

    print("azimuth and range looks: ", azLooks, rgLooks)

    ###########
    masterSlc =  masterFrame.getImage().filename
    lowBandDir = os.path.join(self.insar.splitSpectrumDirname, self.insar.lowBandSlcDirname)
    highBandDir = os.path.join(self.insar.splitSpectrumDirname, self.insar.highBandSlcDirname)
    masterLowBandSlc = os.path.join(lowBandDir, os.path.basename(masterSlc))
    masterHighBandSlc = os.path.join(highBandDir, os.path.basename(masterSlc))
    ##########
    slaveSlc = slaveFrame.getImage().filename
    coregDir = os.path.join(self.insar.coregDirname, self.insar.lowBandSlcDirname) 
    slaveLowBandSlc = os.path.join(coregDir , os.path.basename(slaveSlc))
    coregDir = os.path.join(self.insar.coregDirname, self.insar.highBandSlcDirname)
    slaveHighBandSlc = os.path.join(coregDir , os.path.basename(slaveSlc))
    ##########

    interferogramName = subBandIgram(self, masterLowBandSlc, slaveLowBandSlc, self.insar.lowBandSlcDirname,self.insar.lowBandRadarWavelength)

    interferogramName = subBandIgram(self, masterHighBandSlc, slaveHighBandSlc, self.insar.highBandSlcDirname,self.insar.highBandRadarWavelength)
    
def runFullBandInterferogram(self):
    logger.info("Generating interferogram")

    masterFrame = self._insar.loadProduct( self._insar.masterSlcCropProduct)
    masterSlc =  masterFrame.getImage().filename
   
    if (self.doRubbersheetingRange | self.doRubbersheetingAzimuth):    
        slaveSlc = os.path.join(self._insar.coregDirname, self._insar.fineCoregFilename)
    else:
        slaveSlc = os.path.join(self._insar.coregDirname, self._insar.refinedCoregFilename)

    img1 = isceobj.createImage()
    img1.load(masterSlc + '.xml')

    img2 = isceobj.createImage()
    img2.load(slaveSlc + '.xml')

    azLooks, rgLooks = self.insar.numberOfLooks( masterFrame, self.posting, 
                            self.numberAzimuthLooks, self.numberRangeLooks) 

    self.numberAzimuthLooks = azLooks
    self.numberRangeLooks = rgLooks

    print("azimuth and range looks: ", azLooks, rgLooks)
    ifgDir = self.insar.ifgDirname

    if os.path.isdir(ifgDir):
        logger.info('Interferogram directory {0} already exists.'.format(ifgDir))
    else:
        os.makedirs(ifgDir)

    interferogramName = os.path.join(ifgDir , self.insar.ifgFilename)
    
    info = self._insar.loadProduct(self._insar.slaveSlcCropProduct)
    radarWavelength = info.getInstrument().getRadarWavelength()
    
    generateIgram(self,img1, img2, interferogramName, azLooks, rgLooks,radarWavelength)


    ###Compute coherence
    cohname = os.path.join(self.insar.ifgDirname, self.insar.correlationFilename)
    computeCoherence(masterSlc, slaveSlc, cohname+'.full')
    multilook(cohname+'.full', outname=cohname, alks=azLooks, rlks=rgLooks)


    ##Multilook relevant geometry products
    for fname in [self.insar.latFilename, self.insar.lonFilename, self.insar.losFilename]:
        inname =  os.path.join(self.insar.geometryDirname, fname)
        multilook(inname + '.full', outname= inname, alks=azLooks, rlks=rgLooks)

def runInterferogram(self, igramSpectrum = "full"):

    logger.info("igramSpectrum = {0}".format(igramSpectrum))

    if igramSpectrum == "full":
        runFullBandInterferogram(self)


    elif igramSpectrum == "sub":
        if not self.doDispersive:
            print('Estimating dispersive phase not requested ... skipping sub-band interferograms')
            return
        runSubBandInterferograms(self) 

