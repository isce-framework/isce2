#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging
import numpy as np

import isceobj
from isceobj.Util.decorators import use_api

logger = logging.getLogger('isce.alos2insar.runDenseOffset')

def runDenseOffset(self):
    '''estimate offset fied
    '''
    if not self.doDenseOffset:
        return
    if not ((self._insar.modeCombination == 0) or (self._insar.modeCombination == 1)):
        return

    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    denseOffsetDir = 'dense_offset'
    os.makedirs(denseOffsetDir, exist_ok=True)
    os.chdir(denseOffsetDir)

    #referenceTrack = self._insar.loadProduct(self._insar.referenceTrackParameter)
    #secondaryTrack = self._insar.loadProduct(self._insar.secondaryTrackParameter)

#########################################################################################

    if self.useGPU and self._insar.hasGPU():
        runDenseOffsetGPU(self)
        #define null value. Lijun said there is actually no such null value in GPU ampcor.
        nullValue = -10000.0
    else:
        runDenseOffsetCPU(self)
        #define null value
        nullValue = -10000.0

    #null value set to zero
    img = isceobj.createImage()
    img.load(self._insar.denseOffset+'.xml')
    width = img.width
    length = img.length
    offset=np.memmap(self._insar.denseOffset, dtype='float32', mode='r+', shape=(length*2, width))
    snr=np.memmap(self._insar.denseOffsetSnr, dtype='float32', mode='r+', shape=(length, width))
    offsetband1 = offset[0:length*2:2, :]
    offsetband2 = offset[1:length*2:2, :]
    index = np.nonzero(np.logical_or(offsetband1==nullValue, offsetband2==nullValue))
    offsetband1[index] = 0
    offsetband2[index] = 0
    snr[index] = 0
    del offset, offsetband1, offsetband2, snr

    #areas covered by water body set to zero
    if self.maskOffsetWithWbd:
        img = isceobj.createImage()
        img.load('wbd.rdr.xml')
        width0 = img.width
        length0 = img.length

        img = isceobj.createImage()
        img.load(self._insar.denseOffset+'.xml')
        width = img.width
        length = img.length

        #get water body mask
        wbd0=np.memmap('wbd.rdr', dtype=np.int8, mode='r', shape=(length0, width0))
        wbd0=wbd0[0+self._insar.offsetImageTopoffset:length0:self.offsetSkipHeight, 
                   0+self._insar.offsetImageLeftoffset:width0:self.offsetSkipWidth]
        wbd = np.zeros((length+100, width+100), dtype=np.int8)
        wbd[0:wbd0.shape[0], 0:wbd0.shape[1]]=wbd0

        #mask offset and snr
        offset=np.memmap(self._insar.denseOffset, dtype='float32', mode='r+', shape=(length*2, width))
        snr=np.memmap(self._insar.denseOffsetSnr, dtype='float32', mode='r+', shape=(length, width))
        (offset[0:length*2:2, :])[np.nonzero(wbd[0:length, 0:width]==-1)]=0
        (offset[1:length*2:2, :])[np.nonzero(wbd[0:length, 0:width]==-1)]=0
        snr[np.nonzero(wbd[0:length, 0:width]==-1)]=0

        del wbd0, wbd, offset, snr


#########################################################################################

    os.chdir('../')
    catalog.printToLog(logger, "runDenseOffset")
    self._insar.procDoc.addAllFromCatalog(catalog)


#@use_api
def runDenseOffsetCPU(self):
    '''
    Estimate dense offset field between a pair of SLCs.
    '''
    from mroipac.ampcor.DenseAmpcor import DenseAmpcor
    from isceobj.Alos2Proc.Alos2ProcPublic import runCmd

    ####For this module currently, we need to create an actual file on disk
    for infile in [self._insar.referenceSlc, self._insar.secondarySlcCoregistered]:
        if os.path.isfile(infile):
            continue
        cmd = 'gdal_translate -of ENVI {0}.vrt {0}'.format(infile)
        runCmd(cmd)

    m = isceobj.createSlcImage()
    m.load(self._insar.referenceSlc + '.xml')
    m.setAccessMode('READ')

    s = isceobj.createSlcImage()
    s.load(self._insar.secondarySlcCoregistered + '.xml')
    s.setAccessMode('READ')

    #objOffset.numberThreads = 1
    print('\n************* dense offset estimation parameters *************')
    print('reference SLC: %s' % (self._insar.referenceSlc))
    print('secondary SLC: %s' % (self._insar.secondarySlcCoregistered))
    print('dense offset estimation window width: %d' % (self.offsetWindowWidth))
    print('dense offset estimation window hight: %d' % (self.offsetWindowHeight))
    print('dense offset search window width: %d' % (self.offsetSearchWindowWidth))
    print('dense offset search window hight: %d' % (self.offsetSearchWindowHeight))
    print('dense offset skip width: %d' % (self.offsetSkipWidth))
    print('dense offset skip hight: %d' % (self.offsetSkipHeight))
    print('dense offset covariance surface oversample factor: %d' % (self.offsetCovarianceOversamplingFactor))
    print('dense offset covariance surface oversample window size: %d\n' % (self.offsetCovarianceOversamplingWindowsize))


    objOffset = DenseAmpcor(name='dense')
    objOffset.configure()

    if m.dataType.startswith('C'):
        objOffset.setImageDataType1('complex')
    else:
        objOffset.setImageDataType1('real')
    if s.dataType.startswith('C'):
        objOffset.setImageDataType2('complex')
    else:
        objOffset.setImageDataType2('real')

    objOffset.offsetImageName = self._insar.denseOffset
    objOffset.snrImageName = self._insar.denseOffsetSnr
    objOffset.covImageName = self._insar.denseOffsetCov

    objOffset.setWindowSizeWidth(self.offsetWindowWidth)
    objOffset.setWindowSizeHeight(self.offsetWindowHeight)
    #NOTE: actual number of resulting correlation pixels: self.offsetSearchWindowWidth*2+1
    objOffset.setSearchWindowSizeWidth(self.offsetSearchWindowWidth)
    objOffset.setSearchWindowSizeHeight(self.offsetSearchWindowHeight)
    objOffset.setSkipSampleAcross(self.offsetSkipWidth)
    objOffset.setSkipSampleDown(self.offsetSkipHeight)
    objOffset.setOversamplingFactor(self.offsetCovarianceOversamplingFactor)
    objOffset.setZoomWindowSize(self.offsetCovarianceOversamplingWindowsize)
    objOffset.setAcrossGrossOffset(0)
    objOffset.setDownGrossOffset(0)
    #these are azimuth scaling factor
    #Matching Scale for Sample/Line Directions (-) = 1.000000551500 1.000002373200
    objOffset.setFirstPRF(1.0)
    objOffset.setSecondPRF(1.0)

    objOffset.denseampcor(m, s)

    ### Store params for later
    self._insar.offsetImageTopoffset = objOffset.locationDown[0][0]
    self._insar.offsetImageLeftoffset = objOffset.locationAcross[0][0]

    #change band order
    width=objOffset.offsetCols
    length=objOffset.offsetLines

    offset1 = np.fromfile(self._insar.denseOffset, dtype=np.float32).reshape(length*2, width)
    offset2 = np.zeros((length*2, width), dtype=np.float32)
    offset2[0:length*2:2, :] = offset1[1:length*2:2, :]
    offset2[1:length*2:2, :] = offset1[0:length*2:2, :]

    os.remove(self._insar.denseOffset)
    os.remove(self._insar.denseOffset+'.vrt')
    os.remove(self._insar.denseOffset+'.xml')

    offset2.astype(np.float32).tofile(self._insar.denseOffset)
    outImg = isceobj.createImage()
    outImg.setDataType('FLOAT')
    outImg.setFilename(self._insar.denseOffset)
    outImg.setBands(2)
    outImg.scheme = 'BIL'
    outImg.setWidth(width)
    outImg.setLength(length)
    outImg.addDescription('two-band pixel offset file. 1st band: range offset, 2nd band: azimuth offset')
    outImg.setAccessMode('read')
    outImg.renderHdr()

    return (objOffset.offsetCols, objOffset.offsetLines)


def runDenseOffsetGPU(self):
    '''
    Estimate dense offset field between a pair of SLCs.
    '''
    from contrib.PyCuAmpcor import PyCuAmpcor
    from isceobj.Alos2Proc.Alos2ProcPublic import runCmd
    from isceobj.Alos2Proc.Alos2ProcPublic import create_xml

    ############################################################################################
    # #different from minyan's script: cuDenseOffsets.py: deramp method (0: mag, 1: complex)
    # objOffset.derampMethod = 2 #
    # #varying-gross-offset parameters not set

    # #not set in minyan's script: cuDenseOffsets.py
    # objOffset.corrSurfaceZoomInWindow
    # objOffset.grossOffsetAcrossStatic = 0
    # objOffset.grossOffsetDownStatic = 0
    ############################################################################################


    ####For this module currently, we need to create an actual file on disk
    for infile in [self._insar.referenceSlc, self._insar.secondarySlcCoregistered]:
        if os.path.isfile(infile):
            continue
        cmd = 'gdal_translate -of ENVI {0}.vrt {0}'.format(infile)
        runCmd(cmd)

    m = isceobj.createSlcImage()
    m.load(self._insar.referenceSlc + '.xml')
    m.setAccessMode('READ')

    s = isceobj.createSlcImage()
    s.load(self._insar.secondarySlcCoregistered + '.xml')
    s.setAccessMode('READ')

    print('\n************* dense offset estimation parameters *************')
    print('reference SLC: %s' % (self._insar.referenceSlc))
    print('secondary SLC: %s' % (self._insar.secondarySlcCoregistered))
    print('dense offset estimation window width: %d' % (self.offsetWindowWidth))
    print('dense offset estimation window hight: %d' % (self.offsetWindowHeight))
    print('dense offset search window width: %d' % (self.offsetSearchWindowWidth))
    print('dense offset search window hight: %d' % (self.offsetSearchWindowHeight))
    print('dense offset skip width: %d' % (self.offsetSkipWidth))
    print('dense offset skip hight: %d' % (self.offsetSkipHeight))
    print('dense offset covariance surface oversample factor: %d' % (self.offsetCovarianceOversamplingFactor))
    print('dense offset covariance surface oversample window size: %d\n' % (self.offsetCovarianceOversamplingWindowsize))


    objOffset = PyCuAmpcor.PyCuAmpcor()
    objOffset.algorithm = 0
    objOffset.deviceID = -1
    objOffset.nStreams = 2
    #original ampcor program in roi_pac uses phase gradient to deramp
    objOffset.derampMethod = 2
    objOffset.referenceImageName = self._insar.referenceSlc
    objOffset.referenceImageHeight = m.length
    objOffset.referenceImageWidth = m.width
    objOffset.secondaryImageName = self._insar.secondarySlcCoregistered
    objOffset.secondaryImageHeight = s.length
    objOffset.secondaryImageWidth = s.width
    objOffset.offsetImageName = self._insar.denseOffset
    objOffset.snrImageName = self._insar.denseOffsetSnr

    objOffset.windowSizeWidth = self.offsetWindowWidth
    objOffset.windowSizeHeight = self.offsetWindowHeight
    #objOffset.halfSearchRangeAcross = int(self.offsetSearchWindowWidth / 2 + 0.5)
    #objOffset.halfSearchRangeDown = int(self.offsetSearchWindowHeight / 2 + 0.5)
    objOffset.halfSearchRangeAcross = self.offsetSearchWindowWidth
    objOffset.halfSearchRangeDown = self.offsetSearchWindowHeight
    objOffset.skipSampleDown = self.offsetSkipHeight
    objOffset.skipSampleAcross = self.offsetSkipWidth
    #Oversampling method for correlation surface(0=fft,1=sinc)
    objOffset.corrSufaceOverSamplingMethod = 0
    objOffset.corrSurfaceOverSamplingFactor = self.offsetCovarianceOversamplingFactor
    objOffset.corrSurfaceZoomInWindow = self.offsetCovarianceOversamplingWindowsize
    objOffset.grossOffsetAcrossStatic = 0
    objOffset.grossOffsetDownStatic = 0

    objOffset.referenceStartPixelDownStatic = self.offsetWindowHeight//2
    objOffset.referenceStartPixelAcrossStatic = self.offsetWindowWidth//2

    objOffset.numberWindowDown = (m.length - 2*self.offsetSearchWindowHeight - self.offsetWindowHeight) // self.offsetSkipHeight
    objOffset.numberWindowAcross = (m.width - 2*self.offsetSearchWindowWidth - self.offsetWindowWidth) // self.offsetSkipWidth

    # generic control
    objOffset.numberWindowDownInChunk = 8
    objOffset.numberWindowAcrossInChunk = 8
    objOffset.mmapSize = 16

    objOffset.setupParams()
    objOffset.setConstantGrossOffset(0, 0)
    objOffset.checkPixelInImageRange()
    objOffset.runAmpcor()

    ### Store params for later
    self._insar.offsetImageTopoffset = objOffset.halfSearchRangeDown
    self._insar.offsetImageLeftoffset = objOffset.halfSearchRangeAcross

    
    width = objOffset.numberWindowAcross
    length = objOffset.numberWindowDown
    offsetBIP = np.fromfile(objOffset.offsetImageName.decode('utf-8'), dtype=np.float32).reshape(length, width*2)
    offsetBIL = np.zeros((length*2, width), dtype=np.float32)
    offsetBIL[0:length*2:2, :] = offsetBIP[:, 1:width*2:2]
    offsetBIL[1:length*2:2, :] = offsetBIP[:, 0:width*2:2]
    os.remove(objOffset.offsetImageName.decode('utf-8'))
    offsetBIL.astype(np.float32).tofile(objOffset.offsetImageName.decode('utf-8'))

    outImg = isceobj.createImage()
    outImg.setDataType('FLOAT')
    outImg.setFilename(objOffset.offsetImageName.decode('utf-8'))
    outImg.setBands(2)
    outImg.scheme = 'BIL'
    outImg.setWidth(objOffset.numberWindowAcross)
    outImg.setLength(objOffset.numberWindowDown)
    outImg.addDescription('two-band pixel offset file. 1st band: range offset, 2nd band: azimuth offset')
    outImg.setAccessMode('read')
    outImg.renderHdr()

    snrImg = isceobj.createImage()
    snrImg.setFilename( objOffset.snrImageName.decode('utf8'))
    snrImg.setDataType('FLOAT')
    snrImg.setBands(1)
    snrImg.setWidth(objOffset.numberWindowAcross)
    snrImg.setLength(objOffset.numberWindowDown)
    snrImg.setAccessMode('read')
    snrImg.renderHdr()

    return (objOffset.numberWindowAcross, objOffset.numberWindowDown)
