#
# Author: Heresh Fattahi
# Copyright 2017
#
# Modified by V. Brancato
# Included offset filtering with no SNR
#

import isce
import isceobj
from osgeo import gdal
import numpy as np
import os

def mask_filterNoSNR(denseOffsetFile,filterSize,outName):
    # Masking the offsets with a data-based approach

    from scipy import ndimage

    # Open the offsets
    ds = gdal.Open(denseOffsetFile+'.vrt',gdal.GA_ReadOnly)
    off_az = ds.GetRasterBand(1).ReadAsArray()
    off_rg = ds.GetRasterBand(2).ReadAsArray()
    ds = None

    # Remove missing values from ampcor
    off_rg[np.where(off_rg < -9999)]=0
    off_az[np.where(off_az < -9999)]=0


    # Store the offsets in a complex variable
    off = off_rg + 1j*off_az

    # Mask the azimuth offsets based on the MAD
    mask = off_masking(off,filterSize,thre=3)

    xoff_masked = np.ma.array(off.real,mask=mask)
    yoff_masked = np.ma.array(off.imag,mask=mask)

    # Delete unused variables
    mask = None
    off = None

    # Remove residual noisy spots with a median filter on the azimuth offmap
    yoff_masked.mask = yoff_masked.mask | \
            (ndimage.median_filter(xoff_masked.filled(fill_value=0),3) == 0) | \
            (ndimage.median_filter(yoff_masked.filled(fill_value=0),3) == 0)

    # Fill the data by iteratively using smoothed values
    data = yoff_masked.data
    data[yoff_masked.mask]=np.nan

    off_az_filled = fill_with_smoothed(data,filterSize)

    # Apply median filter to smooth the azimuth offset map
    off_az_filled = ndimage.median_filter(off_az_filled,filterSize)

    # Save the filtered offsets
    length, width = off_az_filled.shape

    # writing the masked and filtered offsets to a file
    print ('writing masked and filtered offsets to: ', outName)

    ##Write array to offsetfile
    off_az_filled.tofile(outName)

    # write the xml file
    img = isceobj.createImage()
    img.setFilename(outName)
    img.setWidth(width)
    img.setAccessMode('READ')
    img.bands = 1
    img.dataType = 'FLOAT'
    img.scheme = 'BIP'
    img.renderHdr()


    return


def off_masking(off,filterSize,thre=2):

    from scipy import ndimage

    # Define the mask to fill the offsets
    vram = ndimage.median_filter(off.real, filterSize)
    vazm = ndimage.median_filter(off.imag, filterSize)

    mask =  (np.abs(off.real-vram) > thre) | (np.abs(off.imag-vazm) > thre) | (off.imag == 0) | (off.real == 0)

    return mask

def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'.
                 data value are replaced where invalid is True
                 If None (default), use: invalid  = np.isnan(data)

    Output:
        Return a filled array.
    """
    from scipy import ndimage

    if invalid is None: invalid = np.isnan(data)

    ind = ndimage.distance_transform_edt(invalid,
                                    return_distances=False,
                                    return_indices=True)
    return data[tuple(ind)]


def mask_filter(denseOffsetFile, snrFile, band, snrThreshold, filterSize, outName):
    #masking and Filtering

    from scipy import ndimage

    ##Read in the offset file
    ds = gdal.Open(denseOffsetFile + '.vrt', gdal.GA_ReadOnly)
    Offset = ds.GetRasterBand(band).ReadAsArray()
    ds = None

    ##Read in the SNR file
    ds = gdal.Open(snrFile + '.vrt', gdal.GA_ReadOnly)
    snr = ds.GetRasterBand(1).ReadAsArray()
    ds = None

    # Masking the dense offsets based on SNR
    print ('masking the dense offsets with SNR threshold: ', snrThreshold)
    Offset[snr<snrThreshold]=np.nan

    # Fill the masked region using valid neighboring pixels
    Offset = fill(Offset)
    ############

    # Median filtering the masked offsets
    print ('Filtering with median filter with size : ', filterSize)
    Offset = ndimage.median_filter(Offset, size=filterSize)
    length, width = Offset.shape

    # writing the masked and filtered offsets to a file
    print ('writing masked and filtered offsets to: ', outName)

    ##Write array to offsetfile
    Offset.tofile(outName)

    # write the xml file
    img = isceobj.createImage()
    img.setFilename(outName)
    img.setWidth(width)
    img.setAccessMode('READ')
    img.bands = 1
    img.dataType = 'FLOAT'
    img.scheme = 'BIP'
    img.renderHdr()

    return None

def fill_with_smoothed(off,filterSize):

    from astropy.convolution import convolve

    off_2filt=np.copy(off)
    kernel = np.ones((filterSize,filterSize),np.float32)/(filterSize*filterSize)
    loop = 0
    cnt2=1

    while (cnt2!=0 & loop<100):
       loop += 1
       idx2= np.isnan(off_2filt)
       cnt2 = np.sum(np.count_nonzero(np.isnan(off_2filt)))
       print(cnt2)
       if cnt2 != 0:
          off_filt= convolve(off_2filt,kernel,boundary='extend',nan_treatment='interpolate')
          off_2filt[idx2]=off_filt[idx2]
          idx3 = np.where(off_filt == 0)
          off_2filt[idx3]=np.nan
          off_filt=None

    return off_2filt

def resampleOffset(maskedFiltOffset, geometryOffset, outName):
    '''
    Oversample offset and add.
    '''
    from imageMath import IML
    import logging

    resampledOffset = maskedFiltOffset + ".resampled"

    inimg = isceobj.createImage()
    inimg.load(geometryOffset + '.xml')
    length = inimg.getLength()
    width = inimg.getWidth()

    ###Currently making the assumption that top left of dense offsets and interfeorgrams are the same.
    ###This is not true for now. We need to update DenseOffsets to have the ability to have same top left
    ###As the input images. Once that is implemente, the math here should all be consistent.
    ###However, this is not too far off since the skip for doing dense offsets is generally large.
    ###The offset is not too large to worry about right now. If the skip is decreased, this could be an issue.

    print('oversampling the filtered and masked offsets to the width and length:', width, ' ', length )
    cmd = 'gdal_translate -of ENVI -ot Float64  -outsize  ' + str(width) + ' ' + str(length) + ' ' + maskedFiltOffset + '.vrt ' + resampledOffset
    print(cmd)
    os.system(cmd)

    img = isceobj.createImage()
    img.setFilename(resampledOffset)
    img.setWidth(width)
    img.setLength(length)
    img.setAccessMode('READ')
    img.bands = 1
    img.dataType = 'DOUBLE'
    img.scheme = 'BIP'
    img.renderHdr()


    ###Adding the geometry offset and oversampled offset
    geomoff = IML.mmapFromISCE(geometryOffset, logging)
    osoff = IML.mmapFromISCE(resampledOffset, logging)

    fid = open(outName, 'w')

    for ll in range(length):
        val = geomoff.bands[0][ll,:] + osoff.bands[0][ll,:]
        val.tofile(fid)

    fid.close()

    img = isceobj.createImage()
    img.setFilename(outName)
    img.setWidth(width)
    img.setLength(length)
    img.setAccessMode('READ')
    img.bands = 1
    img.dataType = 'DOUBLE'
    img.scheme = 'BIP'
    img.renderHdr()



    return None

def runRubbersheetAzimuth(self):

    if not self.doRubbersheetingAzimuth:
        print('Rubber sheeting in azimuth not requested ... skipping')
        return

    # denseOffset file name computeed from cross-correlation
    denseOffsetFile = os.path.join(self.insar.denseOffsetsDirname , self.insar.denseOffsetFilename)
    snrFile = denseOffsetFile + "_snr.bil"
    denseOffsetFile = denseOffsetFile + ".bil"

    # we want the azimuth offsets only which are the first band
    band = [1]
    snrThreshold = self.rubberSheetSNRThreshold
    filterSize = self.rubberSheetFilterSize
    filtAzOffsetFile = os.path.join(self.insar.denseOffsetsDirname, self._insar.filtAzimuthOffsetFilename)

    # masking and median filtering the dense offsets
    if not self.doRubbersheetingRange:
       print('Rubber sheeting in range is off, filtering the offsets with a SNR-based mask')
       mask_filter(denseOffsetFile, snrFile, band[0], snrThreshold, filterSize, filtAzOffsetFile)
    else:
       print('Rubber sheeting in range is on, filtering the offsets with data-based mask')
       mask_filterNoSNR(denseOffsetFile, filterSize, filtAzOffsetFile)

    # azimuth offsets computed from geometry
    offsetsDir = self.insar.offsetsDirname
    geometryAzimuthOffset = os.path.join(offsetsDir, self.insar.azimuthOffsetFilename)
    sheetOffset = os.path.join(offsetsDir, self.insar.azimuthRubbersheetFilename)

    # oversampling the filtAzOffsetFile to the same size of geometryAzimuthOffset
    # and then update the geometryAzimuthOffset by adding the oversampled
    # filtAzOffsetFile to it.
    resampleOffset(filtAzOffsetFile, geometryAzimuthOffset, sheetOffset)

    return None
