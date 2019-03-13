#!/usr/bin/env python3
#
# Author: Bekaert David
# Year: 2017

import isce
import isceobj
import sys
from osgeo import gdal
import argparse
import os 

# command line parsing of input file
def cmdLineParse():
    '''
    Command line parser.
    '''
    parser = argparse.ArgumentParser(description='Generate ISCE xml from gdal products')
    parser.add_argument('-i','--input', dest='fname', type=str, required=True, help='Input filename (GDAL supported)')
    return parser.parse_args()


# main script
if __name__ == '__main__':
    '''
    Main driver.
    '''
 
    # Parse command line
    inps = cmdLineParse()
    # check if the input file exist
    if not os.path.isfile(inps.fname):
       raise Exception('Input file is not found ....')

    # open the GDAL file and get typical data informationi
    GDAL2ISCE_DATATYPE = {
       1 : 'BYTE',
       2 : 'uint16',
       3 : 'SHORT',
       4 : 'uint32',
       5 : 'INT',
       6 : 'FLOAT',
       7 : 'DOUBLE',
       10: 'CFLOAT',
       11: 'complex128',
    }
#    GDAL2NUMPY_DATATYPE = {
#       1 : np.uint8,
#       2 : np.uint16,
#       3 : np.int16,
#       4 : np.uint32,
#       5 : np.int32,
#       6 : np.float32,
#       7 : np.float64,
#       10: np.complex64,
#       11: np.complex128,
#     }

    # check if the input file is a vrt
    filename, file_extension = os.path.splitext(inps.fname)
    print(file_extension)
    if file_extension == ".vrt":
        inps.outname = filename
    else:
        inps.outname = inps.fname
    print(inps.outname)

    # open the GDAL file and get typical data informationi
    data =  gdal.Open(inps.fname, gdal.GA_ReadOnly)
    width = data.RasterXSize
    length = data.RasterYSize
    bands = data.RasterCount
    # output to user
    print("width:    " + "\t" + str(width))
    print("length:   " + "\t" + str(length))
    print("nof bands:" + "\t" + str(bands))

    # getting the datatype information
    raster = data.GetRasterBand(1)
    dataTypeGdal = raster.DataType
    # user look-up dictionary from gdal to isce format
    dataType= GDAL2ISCE_DATATYPE[dataTypeGdal]
    # output to user
    print("dataType: " + "\t" + str(dataType))


    # transformation contains gridcorners (lines/pixels or lonlat and the spacing 1/-1 or deltalon/deltalat)
    transform = data.GetGeoTransform()
    # if a complex data type, then create complex image
    # if a real data type, then create a regular image

    img = isceobj.createImage()
    img.setFilename(os.path.abspath(inps.outname))
    img.setWidth(width)
    img.setLength(length)
    img.setAccessMode('READ')
    img.bands = bands
    img.dataType = dataType
    
    md = data.GetMetadata('IMAGE_STRUCTURE')
    sch = md.get('INTERLEAVE', None)
    if sch == 'LINE':
        img.scheme = 'BIL'
    elif sch == 'PIXEL':
        img.scheme = 'BIP'
    elif sch == 'BAND':
        img.scheme = 'BSQ'
    else:
        print('Unrecognized interleaving scheme, {}'.format(sch))
        print('Assuming default, BIP')
        img.scheme = 'BIP'


    img.firstLongitude = transform[0]
    img.firstLatitude = transform[3] 
    img.deltaLatitude = transform[5] 
    img.deltaLongitude = transform[1] 
    img.dump(inps.outname + ".xml")

