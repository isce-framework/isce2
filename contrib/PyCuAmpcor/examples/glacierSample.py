#!/usr/bin/env python3
#
# test_cuAmpcor.py
# Test program to run ampcor with GPU
#
#

import argparse
import numpy as np
from PyCuAmpcor import PyCuAmpcor


def main():
    '''
    main program
    '''

    objOffset = PyCuAmpcor()

    objOffset.algorithm = 0
    objOffset.deviceID = 0  # -1:let system find the best GPU
    objOffset.nStreams = 2  #cudaStreams
    objOffset.referenceImageName = "20131213.slc.vrt"
    objOffset.referenceImageHeight = 43008
    objOffset.referenceImageWidth = 24320
    objOffset.secondaryImageName = "20131221.slc.vrt"
    objOffset.secondaryImageHeight = 43008
    objOffset.secondaryImageWidth = 24320
    objOffset.windowSizeWidth = 64
    objOffset.windowSizeHeight = 64
    objOffset.halfSearchRangeDown = 20
    objOffset.halfSearchRangeAcross = 20
    objOffset.derampMethod = 1
    objOffset.numberWindowDown = 300
    objOffset.numberWindowAcross = 30
    objOffset.skipSampleDown = 128
    objOffset.skipSampleAcross = 64
    objOffset.numberWindowDownInChunk = 10
    objOffset.numberWindowAcrossInChunk = 10
    objOffset.corrSurfaceOverSamplingFactor = 8
    objOffset.corrSurfaceZoomInWindow = 16
    objOffset.corrSufaceOverSamplingMethod = 1
    objOffset.useMmap = 1
    objOffset.mmapSize = 8

    objOffset.setupParams()
    objOffset.referenceStartPixelDownStatic = 1000
    objOffset.referenceStartPixelAcrossStatic = 1000
    objOffset.setConstantGrossOffset(642, -30)
    objOffset.checkPixelInImageRange()
    objOffset.runAmpcor()


if __name__ == '__main__':

    main()
