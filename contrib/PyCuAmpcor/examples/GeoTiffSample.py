#!/usr/bin/env python3
#
# Test program to run ampcor with GPU
# For two GeoTiff images
#

import argparse
import numpy as np
from PyCuAmpcor import PyCuAmpcor


def main():
    '''
    main program
    '''

    objOffset = PyCuAmpcor() # create the processor

    objOffset.algorithm = 0 # cross-correlation method 0=freq 1=time
    objOffset.deviceID = 0  # GPU device id to be used
    objOffset.nStreams = 2  # cudaStreams; multiple streams to overlap data transfer with gpu calculations
    objOffset.masterImageName = "master.tif"
    objOffset.masterImageHeight = 16480 # RasterYSize
    objOffset.masterImageWidth = 17000 # RasterXSize
    objOffset.slaveImageName = "slave.tif"
    objOffset.slaveImageHeight = 16480
    objOffset.slaveImageWidth = 17000
    objOffset.windowSizeWidth = 64 # template window size
    objOffset.windowSizeHeight = 64
    objOffset.halfSearchRangeDown = 20 # search range
    objOffset.halfSearchRangeAcross = 20
    objOffset.derampMethod = 1 # deramping for complex signal, set to 1 for real images

    objOffset.skipSampleDown = 128 # strides between windows
    objOffset.skipSampleAcross = 64
    # gpu processes several windows in one batch/Chunk
    # total windows in Chunk = numberWindowDownInChunk*numberWindowAcrossInChunk
    # the max number of windows depending on gpu memory and type
    objOffset.numberWindowDownInChunk = 1
    objOffset.numberWindowAcrossInChunk = 10
    objOffset.corrSurfaceOverSamplingFactor = 8 # oversampling factor for correlation surface
    objOffset.corrSurfaceZoomInWindow = 16  # area in correlation surface to be oversampled
    objOffset.corrSufaceOverSamplingMethod = 1 # fft or sinc oversampler
    objOffset.useMmap = 1 # default using memory map as buffer, if having troubles, set to 0
    objOffset.mmapSize = 1 # mmap or buffer size used for transferring data from file to gpu, in GB

    objOffset.numberWindowDown = 40 # number of windows to be processed
    objOffset.numberWindowAcross = 100
    # if to process the whole image; some math needs to be done
    # margin = 0 # margins to be neglected
    #objOffset.numberWindowDown = (objOffset.slaveImageHeight - 2*margin - 2*objOffset.halfSearchRangeDown - objOffset.windowSizeHeight) // objOffset.skipSampleDown
    #objOffset.numberWindowAcross = (objOffset.slaveImageWidth - 2*margin - 2*objOffset.halfSearchRangeAcross - objOffset.windowSizeWidth) // objOffset.skipSampleAcross

    objOffset.setupParams()
    objOffset.masterStartPixelDownStatic = objOffset.halfSearchRangeDown # starting pixel offset
    objOffset.masterStartPixelAcrossStatic = objOffset.halfSearchRangeDown
    objOffset.setConstantGrossOffset(0, 0) # gross offset between master and slave images
    objOffset.checkPixelInImageRange() # check whether there is something wrong with
    objOffset.runAmpcor()


if __name__ == '__main__':

