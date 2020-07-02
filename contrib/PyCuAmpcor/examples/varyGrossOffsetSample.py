#!/usr/bin/env python3
#

from PyCuAmpcor import PyCuAmpcor
import numpy as np

def main():
    '''
    Set parameters manually and run ampcor
    '''
    objOffset = PyCuAmpcor()

    #step 1 set constant parameters
    objOffset.referenceImageName = "reference.slc.vrt"
    objOffset.referenceImageHeight = 128
    objOffset.referenceImageWidth = 128
    objOffset.secondaryImageName = "secondary.slc.vrt"
    objOffset.referenceImageHeight = 128
    objOffset.referenceImageWidth = 128
    objOffset.skipSampleDown = 2
    objOffset.skipSampleAcross = 2
    objOffset.windowSizeHeight = 16
    objOffset.windowSizeWidth = 16
    objOffset.halfSearchRangeDown = 20
    objOffset.halfSearchRangeAcross = 20
    objOffset.numberWindowDown = 2
    objOffset.numberWindowAcross = 2
    objOffset.numberWindowDownInChunk = 2
    objOffset.numberWindowAcrossInChunk = 2
    # 2 set other dependent parameters and allocate aray parameters
    objOffset.setupParams()

    #3 set gross offsets: constant or varying
    objOffset.referenceStartPixelDownStatic = objOffset.halfSearchRangeDown
    objOffset.referenceStartPixelAcrossStatic = objOffset.halfSearchRangeAcross
    vD = np.random.randint(0, 10, size =objOffset.numberWindows, dtype=np.int32)
    vA = np.random.randint(0, 1, size = objOffset.numberWindows, dtype=np.int32)
    objOffset.setVaryingGrossOffset(vD, vA)

    objOffset.checkPixelInImageRange()
    #4 run ampcor
    objOffset.runAmpcor()



if __name__ == '__main__':
    main()
