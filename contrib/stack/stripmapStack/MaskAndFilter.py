#!/usr/bin/env python3
#
# Author: Heresh Fattahi
# Copyright 2016
#
            
import os   
import argparse
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import gdal
from gdalconst import GA_ReadOnly

# suppress the DEBUG message
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import isce
import isceobj
from isceobj.Util.ImageUtil import ImageLib as IML


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


EXAMPLE = '''example:
  MaskAndFilter.py -d offset.bip -s offset_snr.bip
  MaskAndFilter.py -d offset.bip -s offset_snr.bip --plot
'''


def createParser():
    '''     
    Command line parser.
    '''
           
    parser = argparse.ArgumentParser(description='Mask and filter the densOffset',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)
    parser.add_argument('-d', '--dense_offset', dest='denseOffset', type=str, required=True,
            help='The dense offsets file obtained from cross correlation or any other approach')
    parser.add_argument('-s', '--snr', dest='snr', type=str, required=True,
            help='The SNR of the dense offsets obtained from cross correlation or any other approach')
    parser.add_argument('-n', '--filter_size', dest='filterSize', type=int, default=8,
            help='Size of the median filter (default: %(default)s).')
    parser.add_argument('-t', '--snr_threshold', dest='snrThreshold', type=float, default=5,
            help='Min SNR used in the offset (default: %(default)s).')

    # output
    parser.add_argument('-A', '--output_azimuth_offset', dest='outAzimuth', type=str, default='filtAzimuth.off',
            help='File name of the azimuth offsets after rubber sheeting (default: %(default)s).')
    parser.add_argument('-R', '--output_range_offset', dest='outRange', type=str, default='filtRange.off',
            help='File name of the range offsets after rubber sheeting (default: %(default)s).')
    parser.add_argument('-o', '--output_directory', dest='outDir', type=str, default='./',
            help='Output directory (default: %(default)s).')

    # plot
    plot = parser.add_argument_group('plot')
    plot.add_argument('-p', '--plot', dest='plot', action='store_true', default=False,
                      help='plot the offsets before and after masking and filtering')
    plot.add_argument('-v', dest='vlim', nargs=2, type=float, default=(-0.05, 0.05),
                      help='display range for offset (default: %(default)s).')
    plot.add_argument('--v-snr', dest='vlim_snr', nargs=2, type=float, default=(0, 100),
                      help='display range for offset SNR (default: %(default)s).')
    plot.add_argument('--figsize', dest='figsize', nargs=2, type=float, default=(18, 5),
                      help='figure size in inch (default: %(default)s).')
    plot.add_argument('--save', dest='fig_name', type=str, default=None,
                      help='save figure as file')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def read(file, processor='ISCE', bands=None, dataType=None):
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
    # generate ENVI hdr file and fix the file path in xml
    file = os.path.abspath(file)
    if processor == 'ISCE':
        img, dataname, metaname = IML.loadImage(file)
        img.filename = file
        img.setAccessMode('READ')
        img.renderHdr()

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
    # Create the file
    driver = gdal.GetDriverByName( 'ENVI' )
    dst_ds = driver.Create(fileName, raster.shape[1], raster.shape[0], nbands, bandType )
    dst_ds.GetRasterBand(1).WriteArray( raster, 0 ,0 )
    dst_ds = None
    return


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
    if invalid is None: invalid = np.isnan(data)

    ind = ndimage.distance_transform_edt(invalid,
                                         return_distances=False,
                                         return_indices=True)
    return data[tuple(ind)]


def mask_filter(inps, band, outName):
    """masking and Filtering"""

    # read offset
    offset = read(inps.denseOffset, bands=band) 
    offset = offset[0,:,:]

    # read SNR
    snr = read(inps.snr, bands=[1])
    snr = snr[0,:,:]
    snr[np.isnan(snr)] = 0

    # mask the offset based on SNR
    print('masking the dense offsets with SNR threshold: {}'.format(inps.snrThreshold))
    offset1 = np.array(offset)
    offset1[snr < inps.snrThreshold] = np.nan

    # fill the hole in offset with nearest data
    print('fill the masked out region with nearest data')
    offset2 = fill(offset1)

    # median filtering
    print('filtering with median filter with size : ', inps.filterSize)
    offset3 = ndimage.median_filter(offset2, size=inps.filterSize)
    length, width = offset3.shape

    # write data to file
    print('writing masked and filtered offsets to: ', outName)
    write(offset3, outName, 1, 6)

    # write the xml/vrt/hdr file
    img = isceobj.createImage()
    img.setFilename(outName)
    img.setWidth(width)
    img.setLength(length)
    img.setAccessMode('READ')
    img.bands = 1
    img.dataType = 'FLOAT'
    img.scheme = 'BIP'
    #img.createImage()
    img.renderHdr()
    img.renderVRT()
    #img.finalizeImage()

    return [snr, offset, offset1, offset2, offset3]


def plot_mask_and_filtering(az_list, rg_list, inps=None):

    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=inps.figsize, sharex=True, sharey=True)
    titles = ['SNR', 'offset', 'offset (mask)', 'offset (mask/fill)', 'offset (mask/fill/filter)']

    # plot SNR
    im0 = axs[0,0].imshow(az_list[0], vmin=inps.vlim_snr[0], vmax=inps.vlim_snr[1], cmap='RdBu')
    im0 = axs[1,0].imshow(rg_list[0], vmin=inps.vlim_snr[0], vmax=inps.vlim_snr[1], cmap='RdBu')
    axs[0,0].set_title('SNR', fontsize=12)

    # label
    axs[0,0].set_ylabel('azimuth', fontsize=12)
    axs[1,0].set_ylabel('range', fontsize=12)

    # plot offset
    for i in range(1,len(az_list)):
        im1 = axs[0,i].imshow(az_list[i], vmin=inps.vlim[0], vmax=inps.vlim[1], cmap='jet')
        im1 = axs[1,i].imshow(rg_list[i], vmin=inps.vlim[0], vmax=inps.vlim[1], cmap='jet')
        axs[0,i].set_title(titles[i], fontsize=12)
    fig.tight_layout()

    # colorbar
    fig.subplots_adjust(bottom=0.15)
    cax0 = fig.add_axes([0.09, 0.1, 0.08, 0.015])
    cbar0 = plt.colorbar(im0, cax=cax0, orientation='horizontal')
    cax0.yaxis.set_ticks_position('left')

    #fig.subplots_adjust(right=0.93)
    cax1 = fig.add_axes([0.57, 0.1, 0.08, 0.015])
    cbar1 = plt.colorbar(im1, cax=cax1,  orientation='horizontal')
    cbar1.set_label('pixel', fontsize=12)

    # save figure to file
    if inps.fig_name is not None:
        inps.fig_name = os.path.abspath(inps.fig_name)
        print('save figure to file {}'.format(inps.fig_name))
        plt.savefig(inps.fig_name, bbox_inches='tight', transparent=True, dpi=300)
    plt.show()
    return


def main(iargs=None):

    inps = cmdLineParse(iargs)

    if not os.path.exists(inps.outDir):
        os.makedirs(inps.outD)

    #######################
    # masking the dense offsets based on SNR and median filter the masked offs

    # azimuth offsets
    inps.outAzimuth = os.path.join(inps.outDir, inps.outAzimuth)
    az_list = mask_filter(inps, band=[1], outName=inps.outAzimuth)

    # range offsets
    inps.outRange = os.path.join(inps.outDir, inps.outRange)
    rg_list = mask_filter(inps, band=[2], outName=inps.outRange)

    # plot result
    if inps.plot:
        plot_mask_and_filtering(az_list, rg_list, inps)
    return


if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()

