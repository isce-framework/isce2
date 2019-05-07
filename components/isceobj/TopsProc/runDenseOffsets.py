#
# Author: Joshua Cohen
# Copyright 2016
# Based on Piyush Agram's denseOffsets.py script
#

import os
import isce
import isceobj
import logging
from isceobj.Util.decorators import use_api

logger = logging.getLogger('isce.insar.DenseOffsets')

def runDenseOffsets(self):
    '''
    Run CPU / GPU version depending on user choice and availability.
    '''

    if not self.doDenseOffsets:
        print('Dense offsets not requested. Skipping ....')
        return

    hasGPU = self.useGPU and self._insar.hasGPU()
    if hasGPU:
        runDenseOffsetsGPU(self)
    else:
        runDenseOffsetsCPU(self)



@use_api
def runDenseOffsetsCPU(self):
    '''
    Estimate dense offset field between merged master bursts and slave bursts.
    '''
    from mroipac.ampcor.DenseAmpcor import DenseAmpcor

    os.environ['VRT_SHARED_SOURCE'] = "0"

    print('\n============================================================')
    print('Configuring DenseAmpcor object for processing...\n')

    ### Determine appropriate filenames
    mf = 'master.slc'
    sf = 'slave.slc'

    if not ((self.numberRangeLooks == 1) and (self.numberAzimuthLooks==1)):
        mf += '.full'
        sf += '.full'
    master = os.path.join(self._insar.mergedDirname, mf)
    slave = os.path.join(self._insar.mergedDirname, sf)
    
    ####For this module currently, we need to create an actual file on disk
    for infile in [master,slave]:
        if os.path.isfile(infile):
            continue
        cmd = 'gdal_translate -of ENVI {0}.vrt {0}'.format(infile)
        status = os.system(cmd)
        if status:
            raise Exception('{0} could not be executed'.format(status))
    


    ### Load the master object
    m = isceobj.createSlcImage()
    m.load(master + '.xml')
    m.setAccessMode('READ')
#    m.createImage()

    ### Load the slave object
    s = isceobj.createSlcImage()
    s.load(slave + '.xml')
    s.setAccessMode('READ')
#    s.createImage()
    
    width = m.getWidth()
    length = m.getLength()

    objOffset = DenseAmpcor(name='dense')
    objOffset.configure()

#    objOffset.numberThreads = 1
    ### Configure dense Ampcor object
    print('\nMaster frame: %s' % (mf))
    print('Slave frame: %s' % (sf))
    print('Main window size width: %d' % (self.winwidth))
    print('Main window size height: %d' % (self.winhgt))
    print('Search window size width: %d' % (self.srcwidth))
    print('Search window size height: %d' % (self.srchgt))
    print('Skip sample across: %d' % (self.skipwidth))
    print('Skip sample down: %d' % (self.skiphgt))
    print('Field margin: %d' % (self.margin))
    print('Oversampling factor: %d' % (self.oversample))
    print('Gross offset across: %d' % (self.rgshift))
    print('Gross offset down: %d\n' % (self.azshift))

    objOffset.setWindowSizeWidth(self.winwidth)
    objOffset.setWindowSizeHeight(self.winhgt)
    objOffset.setSearchWindowSizeWidth(self.srcwidth)
    objOffset.setSearchWindowSizeHeight(self.srchgt)
    objOffset.skipSampleAcross = self.skipwidth
    objOffset.skipSampleDown = self.skiphgt
    objOffset.oversamplingFactor = self.oversample
    objOffset.setAcrossGrossOffset(self.rgshift)
    objOffset.setDownGrossOffset(self.azshift)
    
    objOffset.setFirstPRF(1.0)
    objOffset.setSecondPRF(1.0)
    if m.dataType.startswith('C'):
        objOffset.setImageDataType1('mag')
    else:
        objOffset.setImageDataType1('real')
    if s.dataType.startswith('C'):
        objOffset.setImageDataType2('mag')
    else:
        objOffset.setImageDataType2('real')

    objOffset.offsetImageName = os.path.join(self._insar.mergedDirname, self._insar.offsetfile)
    objOffset.snrImageName = os.path.join(self._insar.mergedDirname, self._insar.snrfile)
    objOffset.covImageName = os.path.join(self._insar.mergedDirname, self._insar.covfile)

    print('Output dense offsets file name: %s' % (objOffset.offsetImageName))
    print('Output SNR file name: %s' % (objOffset.snrImageName))
    print('Output covariance file name: %s' % (objOffset.covImageName))
    print('\n======================================')
    print('Running dense ampcor...')
    print('======================================\n')
    
    objOffset.denseampcor(m, s) ### Where the magic happens...

    ### Store params for later
    self._insar.offset_width = objOffset.offsetCols
    self._insar.offset_length = objOffset.offsetLines
    self._insar.offset_top = objOffset.locationDown[0][0]
    self._insar.offset_left = objOffset.locationAcross[0][0]


def runDenseOffsetsGPU(self):
    '''
    Estimate dense offset field between merged master bursts and slave bursts.
    '''
    
    from contrib.PyCuAmpcor import PyCuAmpcor

    print('\n============================================================')
    print('Configuring PyCuAmpcor object for processing...\n')

    ### Determine appropriate filenames
    mf = 'master.slc'
    sf = 'slave.slc'
    if not ((self.numberRangeLooks == 1) and (self.numberAzimuthLooks==1)):
        mf += '.full'
        sf += '.full'
    master = os.path.join(self._insar.mergedDirname, mf)
    slave = os.path.join(self._insar.mergedDirname, sf)

    ####For this module currently, we need to create an actual file on disk
    
    for infile in [master,slave]:
        if os.path.isfile(infile):
            continue

        cmd = 'gdal_translate -of ENVI {0}.vrt {0}'.format(infile)
        status = os.system(cmd)
        if status:
            raise Exception('{0} could not be executed'.format(status))

    ### Load the master object
    m = isceobj.createSlcImage()
    m.load(master + '.xml')
    m.setAccessMode('READ')
#    m.createImage()

    ### Load the slave object
    s = isceobj.createSlcImage()
    s.load(slave + '.xml')
    s.setAccessMode('READ')
#    s.createImage()
    
    width = m.getWidth()
    length = m.getLength()

    objOffset = PyCuAmpcor.PyCuAmpcor()
    objOffset.algorithm = 0
    objOffset.deviceID = -1
    objOffset.nStreams = 2
    objOffset.derampMethod = 0
    objOffset.masterImageName = master
    objOffset.masterImageHeight = length
    objOffset.masterImageWidth = width
    objOffset.slaveImageName = slave
    objOffset.slaveImageHeight = length
    objOffset.slaveImageWidth = width

    objOffset.numberWindowDown = (length-100-self.winhgt)//self.skiphgt
    objOffset.numberWindowAcross = (width-100-self.winwidth)//self.skipwidth

    objOffset.windowSizeHeight = self.winhgt
    objOffset.windowSizeWidth = self.winwidth

    objOffset.halfSearchRangeDown = self.srchgt
    objOffset.halfSearchRangeAcross = self.srcwidth

    objOffset.masterStartPixelDownStatic = 50
    objOffset.masterStartPixelAcrossStatic = 50

    objOffset.skipSampleDown = self.skiphgt
    objOffset.skipSampleAcross = self.skipwidth

    objOffset.corrSufaceOverSamplingMethod = 0
    objOffset.corrSurfaceOverSamplingFactor = self.oversample

    # generic control
    objOffset.numberWindowDownInChunk = 10
    objOffset.numberWindowAcrossInChunk = 10
    objOffset.mmapSize = 16


    objOffset.setupParams()

    objOffset.setConstantGrossOffset(self.azshift,self.rgshift)

#    objOffset.numberThreads = 1
    ### Configure dense Ampcor object
    print('\nMaster frame: %s' % (mf))
    print('Slave frame: %s' % (sf))
    print('Main window size width: %d' % (self.winwidth))
    print('Main window size height: %d' % (self.winhgt))
    print('Search window size width: %d' % (self.srcwidth))
    print('Search window size height: %d' % (self.srchgt))
    print('Skip sample across: %d' % (self.skipwidth))
    print('Skip sample down: %d' % (self.skiphgt))
    print('Field margin: %d' % (self.margin))
    print('Oversampling factor: %d' % (self.oversample))
    print('Gross offset across: %d' % (self.rgshift))
    print('Gross offset down: %d\n' % (self.azshift))

    #Modify BIL in filename to BIP if needed and store for future use
    prefix, ext = os.path.splitext(self._insar.offsetfile)
    if ext == '.bil':
        ext = '.bip'
        self._insar.offsetfile = prefix + ext

    objOffset.offsetImageName = os.path.join(self._insar.mergedDirname, self._insar.offsetfile)
    objOffset.snrImageName = os.path.join(self._insar.mergedDirname, self._insar.snrfile)

    print('Output dense offsets file name: %s' % (objOffset.offsetImageName))
    print('Output SNR file name: %s' % (objOffset.snrImageName))
    print('\n======================================')
    print('Running dense ampcor...')
    print('======================================\n')
   

    objOffset.checkPixelInImageRange()
    objOffset.runAmpcor()

    #objOffset.denseampcor(m, s) ### Where the magic happens...

    ### Store params for later
    self._insar.offset_width = objOffset.numberWindowAcross
    self._insar.offset_length = objOffset.numberWindowDown
    self._insar.offset_top = 50
    self._insar.offset_left = 50

    outImg = isceobj.createImage()
    outImg.setDataType('FLOAT')
    outImg.setFilename(objOffset.offsetImageName.decode('utf-8'))
    outImg.setBands(2)
    outImg.scheme = 'BIP'
    outImg.setWidth(objOffset.numberWindowAcross)
    outImg.setLength(objOffset.numberWindowDown)
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
        


if __name__ == '__main__' :
    '''
    Default routine to plug master.slc.full/slave.slc.full into
    Dense Offsets Ampcor module.
    '''

    main()
