#!/usr/bin/env python3
# Heresh Fattahi

import numpy as np
import argparse
import os
import isce
import isceobj
import shelve
import gdal
import osr
from gdalconst import GA_ReadOnly
from scipy import ndimage


GDAL2NUMPY_DATATYPE = {

1 : np.uint8,
2 : np.uint16,
3 : np.int16,
4 : np.uint32,
5 : np.int32,
6 : np.float32,
7 : np.float64,
10: np.complex64,
11: np.complex128,

}


def createParser():
    '''     
    Command line parser.
    '''

    parser = argparse.ArgumentParser( description='interpolates and adds to the targetFile')
    parser.add_argument('-i', '--input', dest='input', type=str, default=None,
            help='input file')
    parser.add_argument('-t', '--target_file', dest='targetFile', type=str, default=None,
            help='the reference file that the input will be interpolated to its size and added to it')
    parser.add_argument('-o', '--output', dest='output', type=str, default=None,
            help='output file')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)



def getShape(file):

    dataset = gdal.Open(file,GA_ReadOnly)
    return dataset.RasterYSize, dataset.RasterXSize

def resampleOffset(maskedFiltOffset, geometryOffset, resampledOffset, outName):

    length, width = getShape(geometryOffset)
    print('oversampling the filtered and masked offsets to the width and length:', width, ' ', length )
    cmd = 'gdal_translate -of ENVI  -outsize  ' + str(width) + ' ' + str(length) + ' ' + maskedFiltOffset + ' ' + resampledOffset
    os.system(cmd)

    img = isceobj.createImage()
    img.setFilename(resampledOffset)
    img.setWidth(width)
    img.setLength(length)
    img.setAccessMode('READ')
    img.bands = 1
    img.dataType = 'FLOAT'
    img.scheme = 'BIP'
    img.renderHdr()
    img.renderVRT()

    print ('Adding the dense offsets to the geometry offsets. Output: ', outName)
    cmd = "gdal_calc.py -A " + geometryOffset + " -B " + resampledOffset + " --outfile=" + outName + ' --calc="A+B" --format=ENVI --type=Float64 --quiet --overwrite'
    print (cmd)
    os.system(cmd)

def main(iargs=None):

    inps = cmdLineParse(iargs)
    resampledDenseOffset = inps.input + '.resampled'
    resampleOffset(inps.input, inps.targetFile, resampledDenseOffset, inps.output)

if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()

