#

#
import isce
import isceobj
from mroipac.ampcor.DenseAmpcor import DenseAmpcor
from isceobj.Util.decorators import use_api
import os
import logging

logger = logging.getLogger('isce.insar.runDenseOffsets')

@use_api
def estimateOffsetField(master, slave, denseOffsetFileName,
                        ww=64, wh=64,
                        sw=20, shh=20,
                        kw=32, kh=32):
    '''
    Estimate offset field between burst and simamp.
    '''

    ###Loading the slave image object
    sim = isceobj.createSlcImage()
    sim.load(slave+'.xml')
    sim.setAccessMode('READ')
    sim.createImage()

    ###Loading the master image object
    sar = isceobj.createSlcImage()
    sar.load(master + '.xml')
    sar.setAccessMode('READ')
    sar.createImage()

    width = sar.getWidth()
    length = sar.getLength()

    objOffset = DenseAmpcor(name='dense')
    objOffset.configure()

#   objOffset.numberThreads = 6
    objOffset.setWindowSizeWidth(ww) #inps.winwidth)
    objOffset.setWindowSizeHeight(wh) #inps.winhgt)
    objOffset.setSearchWindowSizeWidth(sw) #inps.srcwidth)
    objOffset.setSearchWindowSizeHeight(shh) #inps.srchgt)
    objOffset.skipSampleAcross = kw #inps.skipwidth
    objOffset.skipSampleDown = kh #inps.skiphgt
    objOffset.margin = 50 #inps.margin
    objOffset.oversamplingFactor = 32  #inps.oversample

    objOffset.setAcrossGrossOffset(0) #inps.rgshift)
    objOffset.setDownGrossOffset(0) #inps.azshift)

    objOffset.setFirstPRF(1.0)
    objOffset.setSecondPRF(1.0)
    if sar.dataType.startswith('C'):
        objOffset.setImageDataType1('mag')
    else:
        objOffset.setImageDataType1('real')

    if sim.dataType.startswith('C'):
        objOffset.setImageDataType2('mag')
    else:
        objOffset.setImageDataType2('real')


    objOffset.offsetImageName = denseOffsetFileName + '.bil'
    objOffset.snrImageName = denseOffsetFileName +'_snr.bil'
    objOffset.covImageName = denseOffsetFileName +'_cov.bil'

    objOffset.denseampcor(sar, sim)

    sar.finalizeImage()
    sim.finalizeImage()
    return (objOffset.locationDown[0][0], objOffset.locationAcross[0][0])

def runDenseOffsets(self):

    if self.doDenseOffsets or self.doRubbersheetingAzimuth:
        if self.doDenseOffsets:
            print('Dense offsets explicitly requested')

        if self.doRubbersheetingAzimuth:
            print('Generating offsets as rubber sheeting requested')
    else:
        return

    masterFrame = self.insar.loadProduct( self._insar.masterSlcCropProduct)
    masterSlc =  masterFrame.getImage().filename

    slaveSlc = os.path.join(self.insar.coregDirname, self._insar.refinedCoregFilename )

    dirname = self.insar.denseOffsetsDirname
    if os.path.isdir(dirname):
        logger.info('dense offsets directory {0} already exists.'.format(dirname))
    else:
        os.makedirs(dirname)

    denseOffsetFilename = os.path.join(dirname , self.insar.denseOffsetFilename)

    field = estimateOffsetField(masterSlc, slaveSlc, denseOffsetFilename,
                                ww = self.denseWindowWidth,
                                wh = self.denseWindowHeight,
                                sw = self.denseSearchWidth,
                                shh = self.denseSearchHeight,
                                kw = self.denseSkipWidth,
                                kh = self.denseSkipHeight)

    self._insar.offset_top = field[0]
    self._insar.offset_left = field[1]

    return None
