#!/usr/bin/env python3
##############################################################################
#Author: Heresh Fattahi
# Copyright 2016
###############################################################################

            
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
           
    parser = argparse.ArgumentParser( description='filters the densOffset, oversamples it and adds back to the geometry offset')
    parser.add_argument('-a', '--geometry_azimuth_offset', dest='geometryAzimuthOffset', type=str, default=None,
            help='The azimuth offsets file obtained with geometry')
    parser.add_argument('-r', '--geometry_range_offset', dest='geometryRangeOffset', type=str, default=None,
            help='The range offsets file obtained with geometry')
    parser.add_argument('-d', '--dense_offset', dest='denseOffset', type=str, required=True,
            help='The dense offsets file obtained from cross correlation or any other approach')
    parser.add_argument('-s', '--snr', dest='snr', type=str, required=True,
            help='The SNR of the dense offsets obtained from cross correlation or any other approach')
    parser.add_argument('-n', '--filter_size', dest='filterSize', type=int, default=8,
            help='The size of the median filter')
    parser.add_argument('-t', '--snr_threshold', dest='snrThreshold', type=float, default=5,
            help='The snr threshold used to mask the offset')
    parser.add_argument('-A', '--output_azimuth_offset', dest='outAzimuth', type=str, default='azimuth_rubberSheet.off',
            help='The azimuth offsets after rubber sheeting')
    parser.add_argument('-R', '--output_range_offset', dest='outRange', type=str, default='range_rubberSheet.off',
            help='The range offsets after rubber sheeting')

    parser.add_argument('-p', '--plot', dest='plot', type=str, default='False',
            help='plot the offsets before and after masking and filtering')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def read(file, processor='ISCE' , bands=None , dataType=None):
    ''' raeder based on GDAL.
       
    Args:

        * file      -> File name to be read

    Kwargs:

        * processor -> the processor used for the InSAR processing. default: ISCE
        * bands     -> a list of bands to be extracted. If not specified all bands will be extracted. 
        * dataType  -> if not specified, it will be extracted from the data itself
    Returns:
        * data : A numpy array with dimensions : number_of_bands * length * width
    '''

    if processor == 'ISCE':
        cmd = 'isce2gis.py envi -i ' + file
        os.system(cmd)

    dataset = gdal.Open(file,GA_ReadOnly)

    ######################################
    # if the bands have not been specified, all bands will be extracted
    if bands is None:
        bands = range(1,dataset.RasterCount+1)
    ######################################
    # if dataType is not known let's get it from the data:    
    if dataType is None:
        band = dataset.GetRasterBand(1)
        dataType =  GDAL2NUMPY_DATATYPE[band.DataType]

    ######################################
    # Form a numpy array of zeros with the the shape of (number of bands * length * width) and a given data type
    data = np.zeros((len(bands), dataset.RasterYSize, dataset.RasterXSize),dtype=dataType)
    ######################################
    # Fill the array with the Raster bands
    idx=0
    for i in bands:
       band=dataset.GetRasterBand(i)
       data[idx,:,:] = band.ReadAsArray()
       idx+=1

    dataset = None
    return data


def write(raster, fileName, nbands, bandType):

    ############
    # Create the file
    driver = gdal.GetDriverByName( 'ENVI' )
    dst_ds = driver.Create(fileName, raster.shape[1], raster.shape[0], nbands, bandType )
    dst_ds.GetRasterBand(1).WriteArray( raster, 0 ,0 )
    
    dst_ds = None


def mask_filter(inps, band, outName, plot=False):
    #masking and Filtering
    Offset = read(inps.denseOffset, bands=band) 
    Offset = Offset[0,:,:]

    snr = read(inps.snr, bands=[1])
    snr = snr[0,:,:]

    # Masking the dense offsets based on SNR
    Offset[snr<inps.snrThreshold] = 0

    # Median filtering the masked offsets
    Offset_filt = ndimage.median_filter(Offset, size=inps.filterSize)
    width = Offset_filt.shape[1]

    # writing the masked and filtered offsets to a file
    write(Offset_filt, outName, 1, 6)

    # write the xml file
    img = isceobj.createImage()
    img.setFilename(outName)
    img.setWidth(width)
    img.setAccessMode('READ')
    img.bands = 1
    img.dataType = 'FLOAT'
    img.scheme = 'BIP'
    img.createImage()
    img.renderHdr()
    img.finalizeImage()

    ################################
    if plot:
       import matplotlib.pyplot as plt
       fig = plt.figure()

       ax=fig.add_subplot(1,2,1)
       # cax=ax.imshow(azOffset[800:,:], vmin=-2, vmax=4)
       cax=ax.imshow(Offset, vmin=-2, vmax=4)
       ax.set_title('''Offset''')
    
       ax=fig.add_subplot(1,2,2)
    #ax.imshow(azOffset_filt[800:,:], vmin=-2, vmax=4)
       ax.imshow(Offset_filt, vmin=-2, vmax=4)
       ax.set_title('''Offset filt''')
       plt.show()

def getShape(file):
    
    dataset = gdal.Open(file,GA_ReadOnly) 
    return dataset.RasterYSize, dataset.RasterXSize

def resampleOffset(maskedFiltOffset, geometryOffset, resampledOffset, outName):
    
    length, width = getShape(geometryOffset)
    
    cmd = 'gdal_translate -of ENVI  -outsize  ' + str(width) + ' ' + str(length) + ' ' + maskedFiltOffset + ' ' + resampledOffset
    os.system(cmd)

    img = isceobj.createImage()
    img.setFilename(resampledOffset)
    img.setWidth(width)
    img.setAccessMode('READ')
    img.bands = 1
    img.dataType = 'FLOAT'
    img.scheme = 'BIP'
    img.createImage()
    img.renderHdr()
    img.finalizeImage()

    cmd = "imageMath.py -e='a+b' -o " + outName + " -t float  --a=" + geometryOffset+ " --b=" + resampledOffset
    os.system(cmd)


def main(iargs=None):

    inps = cmdLineParse(iargs)
    if inps.geometryAzimuthOffset:
        #######################
        # working on the azimuth offsets
        #######################
        cmd = 'isce2gis.py envi -i ' + inps.geometryAzimuthOffset
        os.system(cmd)

        #######################
        # masking the dense offsets based on SNR and median filter the masked offsets
        maskedFiltOffset = 'filtAzOff.bil'
        mask_filter(inps, band=[1], outName = maskedFiltOffset, plot=inps.plot)

        cmd = 'isce2gis.py envi -i ' + maskedFiltOffset
        os.system(cmd)
        #######################
        # resampling the masked and filtered dense offsets to the same grid size of the geometry offsets

        resampledDenseOffset = 'filtAzOff_resampled.bil'
        resampleOffset(maskedFiltOffset, inps.geometryAzimuthOffset, resampledDenseOffset, inps.outAzimuth)
        
        #######################

    if inps.geometryRangeOffset:
        #######################
        # working on the range offsets
        #######################
        cmd = 'isce2gis.py envi -i ' + inps.geometryRangeOffset
        os.system(cmd)

        #######################
        # masking the dense offsets based on SNR and median filter the masked offsets

        maskedFiltOffset = 'filtRngOff.bil'
        mask_filter(inps, band=[2], outName = maskedFiltOffset, plot=inps.plot)

        cmd = 'isce2gis.py envi -i ' + maskedFiltOffset
        os.system(cmd)
        #######################
        # resampling the masked and filtered dense offsets to the same grid size of the geometry offsets
        resampledDenseOffset = 'filtRngOff_resampled.bil'
        resampleOffset(maskedFiltOffset, inps.geometryRangeOffset, resampledDenseOffset, inps.outRange)


if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()


