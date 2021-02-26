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
    Estimate dense offset field between merged reference bursts and secondary bursts.
    '''
    from mroipac.ampcor.DenseAmpcor import DenseAmpcor

    os.environ['VRT_SHARED_SOURCE'] = "0"

    print('\n============================================================')
    print('Configuring DenseAmpcor object for processing...\n')

    ### Determine appropriate filenames
    mf = 'reference.slc'
    sf = 'secondary.slc'

    if not ((self.numberRangeLooks == 1) and (self.numberAzimuthLooks==1)):
        mf += '.full'
        sf += '.full'
    reference = os.path.join(self._insar.mergedDirname, mf)
    secondary = os.path.join(self._insar.mergedDirname, sf)

    ####For this module currently, we need to create an actual file on disk
    for infile in [reference,secondary]:
        if os.path.isfile(infile):
            continue
        cmd = 'gdal_translate -of ENVI {0}.vrt {0}'.format(infile)
        status = os.system(cmd)
        if status:
            raise Exception('{0} could not be executed'.format(status))



    ### Load the reference object
    m = isceobj.createSlcImage()
    m.load(reference + '.xml')
    m.setAccessMode('READ')
#    m.createImage()

    ### Load the secondary object
    s = isceobj.createSlcImage()
    s.load(secondary + '.xml')
    s.setAccessMode('READ')
#    s.createImage()

    width = m.getWidth()
    length = m.getLength()

    objOffset = DenseAmpcor(name='dense')
    objOffset.configure()

#    objOffset.numberThreads = 1
    ### Configure dense Ampcor object
    print('\nReference frame: %s' % (mf))
    print('Secondary frame: %s' % (sf))
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
    Estimate dense offset field between merged reference bursts and secondary bursts.
    '''

    from contrib.PyCuAmpcor import PyCuAmpcor

    print('\n============================================================')
    print('Configuring PyCuAmpcor object for processing...\n')

    ### Determine appropriate filenames
    mf = 'reference.slc'
    sf = 'secondary.slc'
    if not ((self.numberRangeLooks == 1) and (self.numberAzimuthLooks==1)):
        mf += '.full'
        sf += '.full'
    reference = os.path.join(self._insar.mergedDirname, mf)
    secondary = os.path.join(self._insar.mergedDirname, sf)

    ####For this module currently, we need to create an actual file on disk
    for infile in [reference,secondary]:
        if os.path.isfile(infile):
            continue

        print('Creating actual file {} ...\n'.format(infile))
        cmd = 'gdal_translate -of ENVI {0}.vrt {0}'.format(infile)
        status = os.system(cmd)
        if status:
            raise Exception('{0} could not be executed'.format(status))

    ### Load the reference object
    m = isceobj.createSlcImage()
    m.load(reference + '.xml')
    m.setAccessMode('READ')
    # re-create vrt in terms of merged full slc
    m.renderHdr()

    ### Load the secondary object
    s = isceobj.createSlcImage()
    s.load(secondary + '.xml')
    s.setAccessMode('READ')
    # re-create vrt in terms of merged full slc
    s.renderHdr()

    # get the dimension
    width = m.getWidth()
    length = m.getLength()

    ### create the GPU processor
    objOffset = PyCuAmpcor.PyCuAmpcor()

    ### Set parameters
    # cross-correlation method, 0=Frequency domain, 1= Time domain
    objOffset.algorithm = 0
    # deramping method: 0 to take magnitude (fixed for Tops)
    objOffset.derampMethod = 0
    objOffset.referenceImageName = reference + '.vrt'
    objOffset.referenceImageHeight = length
    objOffset.referenceImageWidth = width
    objOffset.secondaryImageName = secondary + '.vrt'
    objOffset.secondaryImageHeight = length
    objOffset.secondaryImageWidth = width

    # adjust the margin
    margin = max(self.margin, abs(self.azshift), abs(self.rgshift))

    # set the start pixel in the reference image
    objOffset.referenceStartPixelDownStatic = margin + self.srchgt
    objOffset.referenceStartPixelAcrossStatic = margin + self.srcwidth

    # compute the number of windows
    objOffset.numberWindowDown = (length-2*margin-2*self.srchgt-self.winhgt)//self.skiphgt
    objOffset.numberWindowAcross = (width-2*margin-2*self.srcwidth-self.winwidth)//self.skipwidth

    # set the template window size
    objOffset.windowSizeHeight = self.winhgt
    objOffset.windowSizeWidth = self.winwidth

    # set the (half) search range
    objOffset.halfSearchRangeDown = self.srchgt
    objOffset.halfSearchRangeAcross = self.srcwidth

    # set the skip distance between windows
    objOffset.skipSampleDown = self.skiphgt
    objOffset.skipSampleAcross = self.skipwidth

    # correlation surface oversampling method, # 0=FFT, 1=Sinc
    objOffset.corrSurfaceOverSamplingMethod = 0
    # oversampling factor
    objOffset.corrSurfaceOverSamplingFactor = self.oversample

    ### gpu control
    objOffset.deviceID = 0
    objOffset.nStreams = 2
    # number of windows in a chunk/batch
    objOffset.numberWindowDownInChunk = 1
    objOffset.numberWindowAcrossInChunk = 64
    # memory map cache size in GB
    objOffset.mmapSize = 16

    # Modify BIL in filename to BIP if needed and store for future use
    prefix, ext = os.path.splitext(self._insar.offsetfile)
    if ext == '.bil':
        ext = '.bip'
        self._insar.offsetfile = prefix + ext

    # set the output file name
    objOffset.offsetImageName = os.path.join(self._insar.mergedDirname, self._insar.offsetfile)
    objOffset.grossOffsetImageName = os.path.join(self._insar.mergedDirname, self._insar.offsetfile + ".gross")
    objOffset.snrImageName = os.path.join(self._insar.mergedDirname, self._insar.snrfile)
    objOffset.covImageName = os.path.join(self._insar.mergedDirname, self._insar.covfile)

    # merge gross offset to final offset
    objOffset.mergeGrossOffset = 1

    ### print the settings
    print('\nReference frame: %s' % (mf))
    print('Secondary frame: %s' % (sf))
    print('Main window size width: %d' % (self.winwidth))
    print('Main window size height: %d' % (self.winhgt))
    print('Search window size width: %d' % (self.srcwidth))
    print('Search window size height: %d' % (self.srchgt))
    print('Skip sample across: %d' % (self.skipwidth))
    print('Skip sample down: %d' % (self.skiphgt))
    print('Field margin: %d' % (margin))
    print('Oversampling factor: %d' % (self.oversample))
    print('Gross offset across: %d' % (self.rgshift))
    print('Gross offset down: %d\n' % (self.azshift))
    print('Output dense offsets file name: %s' % (objOffset.offsetImageName))
    print('Output gross offsets file name: %s' % (objOffset.grossOffsetImageName))
    print('Output SNR file name: %s' % (objOffset.snrImageName))
    print('Output COV file name: %s' % (objOffset.covImageName))

    # pass the parameters to C++ programs
    objOffset.setupParams()
    # set the (static) gross offset
    objOffset.setConstantGrossOffset(self.azshift,self.rgshift)
    # make sure all pixels are in range
    objOffset.checkPixelInImageRange()

    print('\n======================================')
    print('Running PyCuAmpcor...')
    print('======================================\n')

    # run ampcor
    objOffset.runAmpcor()

    ### Store params for later
    # offset width x length, also number of windows
    self._insar.offset_width = objOffset.numberWindowAcross
    self._insar.offset_length = objOffset.numberWindowDown
    # the center of the first reference window
    self._insar.offset_top = objOffset.referenceStartPixelDownStatic + (objOffset.windowSizeHeight-1)//2
    self._insar.offset_left = objOffset.referenceStartPixelAcrossStatic + (objOffset.windowSizeWidth-1)//2

    # generate description files for output images
    outImg = isceobj.createImage()
    outImg.setDataType('FLOAT')
    outImg.setFilename(objOffset.offsetImageName)
    outImg.setBands(2)
    outImg.scheme = 'BIP'
    outImg.setWidth(objOffset.numberWindowAcross)
    outImg.setLength(objOffset.numberWindowDown)
    outImg.setAccessMode('read')
    outImg.renderHdr()

    # gross offset
    goutImg = isceobj.createImage()
    goutImg.setDataType('FLOAT')
    goutImg.setFilename(objOffset.grossOffsetImageName)
    goutImg.setBands(2)
    goutImg.scheme = 'BIP'
    goutImg.setWidth(objOffset.numberWindowAcross)
    goutImg.setLength(objOffset.numberWindowDown)
    goutImg.setAccessMode('read')
    goutImg.renderHdr()

    snrImg = isceobj.createImage()
    snrImg.setFilename(objOffset.snrImageName)
    snrImg.setDataType('FLOAT')
    snrImg.setBands(1)
    snrImg.setWidth(objOffset.numberWindowAcross)
    snrImg.setLength(objOffset.numberWindowDown)
    snrImg.setAccessMode('read')
    snrImg.renderHdr()

    covImg = isceobj.createImage()
    covImg.setFilename(objOffset.covImageName)
    covImg.setDataType('FLOAT')
    covImg.setBands(3)
    covImg.scheme = 'BIP'
    covImg.setWidth(objOffset.numberWindowAcross)
    covImg.setLength(objOffset.numberWindowDown)
    covImg.setAccessMode('read')
    covImg.renderHdr()


if __name__ == '__main__' :
    '''
    Default routine to plug reference.slc.full/secondary.slc.full into
    Dense Offsets Ampcor module.
    '''

    main()
