#!/usr/bin/env python3

# Author: Minyan Zhong, Lijun Zhu


import os
import argparse
import numpy as np

import isce
import isceobj
from isceobj.Util.decorators import use_api
from contrib.PyCuAmpcor.PyCuAmpcor import PyCuAmpcor


EXAMPLE = '''example
  cuDenseOffsets.py -m ./merged/SLC/20151120/20151120.slc.full -s ./merged/SLC/20151214/20151214.slc.full
      --referencexml ./reference/IW1.xml --outprefix ./merged/offsets/20151120_20151214/offset
      --ww 256 --wh 256 --oo 32 --kw 300 --kh 100 --nwac 100 --nwdc 1 --sw 8 --sh 8 --gpuid 2
'''


def createParser():
    '''
    Command line parser.
    '''


    parser = argparse.ArgumentParser(description='Generate offset field between two Sentinel slc',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)
    parser.add_argument('-m','--reference', type=str, dest='reference', required=True,
                        help='Reference image')
    parser.add_argument('-s', '--secondary',type=str, dest='secondary', required=True,
                        help='Secondary image')
    parser.add_argument('-l', '--lat',type=str, dest='lat', required=False,
                        help='Latitude')
    parser.add_argument('-L', '--lon',type=str, dest='lon', required=False,
                        help='Longitude')
    parser.add_argument('--los',type=str, dest='los', required=False,
                        help='Line of Sight')
    parser.add_argument('-x', '--referencexml',type=str, dest='referencexml', required=False,
                        help='Reference Image XML File')

    parser.add_argument('--op','--outprefix','--output-prefix', type=str, dest='outprefix',
                        default='offset', required=True,
                        help='Output prefix, default: offset.')
    parser.add_argument('--os','--outsuffix', type=str, dest='outsuffix', default='',
                        help='Output suffix, default:.')
    parser.add_argument('--ww', type=int, dest='winwidth', default=64,
                        help='Window width (default: %(default)s).')
    parser.add_argument('--wh', type=int, dest='winhgt', default=64,
                        help='Window height (default: %(default)s).')

    parser.add_argument('--sw', type=int, dest='srcwidth', default=20, choices=range(8, 33),
                        help='Search window width (default: %(default)s).')
    parser.add_argument('--sh', type=int, dest='srchgt', default=20, choices=range(8, 33),
                        help='Search window height (default: %(default)s).')
    parser.add_argument('--mm', type=int, dest='margin', default=50,
                        help='Margin (default: %(default)s).')

    parser.add_argument('--kw', type=int, dest='skipwidth', default=64,
                        help='Skip across (default: %(default)s).')
    parser.add_argument('--kh', type=int, dest='skiphgt', default=64,
                        help='Skip down (default: %(default)s).')

    parser.add_argument('--raw-osf','--raw-over-samp-factor', type=int, dest='raw_oversample',
                        default=2, choices=range(2,5),
                        help='raw data oversampling factor (default: %(default)s).')

    gross = parser.add_argument_group('Initial gross offset')
    gross.add_argument('-g','--gross', type=int, dest='gross', default=0,
                       help='Use gross offset or not')
    gross.add_argument('--aa', type=int, dest='azshift', default=0,
                       help='Gross azimuth offset (default: %(default)s).')
    gross.add_argument('--rr', type=int, dest='rgshift', default=0,
                       help='Gross range offset (default: %(default)s).')

    corr = parser.add_argument_group('Correlation surface')
    corr.add_argument('--corr-win-size', type=int, dest='corr_win_size', default=-1,
                      help='Zoom-in window size of the correlation surface for oversampling (default: %(default)s).')
    corr.add_argument('--corr-osf', '--oo', '--corr-over-samp-factor', type=int, dest='corr_oversample', default=32,
                      help = 'Oversampling factor of the zoom-in correlation surface (default: %(default)s).')

    parser.add_argument('--nwa', type=int, dest='numWinAcross', default=-1,
                        help='Number of window across (default: %(default)s).')
    parser.add_argument('--nwd', type=int, dest='numWinDown', default=-1,
                        help='Number of window down (default: %(default)s).')

    parser.add_argument('--nwac', type=int, dest='numWinAcrossInChunk', default=1,
                        help='Number of window across in chunk (default: %(default)s).')
    parser.add_argument('--nwdc', type=int, dest='numWinDownInChunk', default=1,
                        help='Number of window down in chunk (default: %(default)s).')
    parser.add_argument('-r', '--redo', dest='redo', action='store_true',
                        help='To redo by force (ignore the existing offset fields).')

    parser.add_argument('--drmp', '--deramp', dest='deramp', type=int, default=0,
                        help='deramp method (0: mag, 1: complex) (default: %(default)s).')

    parser.add_argument('--gpuid', '--gid', '--gpu-id', dest='gpuid', type=int, default=-1,
                        help='GPU ID (default: %(default)s).')

    return parser


def cmdLineParse(iargs = None):
    parser = createParser()
    inps =  parser.parse_args(args=iargs)

    # check oversampled window size
    if (inps.winwidth + 2 * inps.srcwidth) * inps.raw_oversample > 1024:
        msg = 'input oversampled window size in the across/range direction '
        msg += 'exceeds the current implementaion limit of 1024!'
        raise ValueError(msg)

    return inps


@use_api
def estimateOffsetField(reference, secondary, inps=None):


    ###Loading the secondary image object
    sim = isceobj.createSlcImage()
    sim.load(secondary+'.xml')
    sim.setAccessMode('READ')
    sim.createImage()

    ###Loading the reference image object
    sar = isceobj.createSlcImage()
    sar.load(reference+'.xml')

    sar.setAccessMode('READ')
    sar.createImage()

    width = sar.getWidth()
    length = sar.getLength()

    objOffset = PyCuAmpcor()

    objOffset.algorithm = 0
    objOffset.deviceID = inps.gpuid  # -1:let system find the best GPU
    objOffset.nStreams = 2 #cudaStreams
    objOffset.derampMethod = inps.deramp
    print('deramp method (0 for magnitude, 1 for complex): ', objOffset.derampMethod)


    objOffset.referenceImageName = reference+'.vrt'
    objOffset.referenceImageHeight = length
    objOffset.referenceImageWidth = width
    objOffset.secondaryImageName = secondary+'.vrt'
    objOffset.secondaryImageHeight = length
    objOffset.secondaryImageWidth = width

    print("image length:",length)
    print("image width:",width)

    objOffset.numberWindowDown = (length-2*inps.margin-2*inps.srchgt-inps.winhgt)//inps.skiphgt
    objOffset.numberWindowAcross = (width-2*inps.margin-2*inps.srcwidth-inps.winwidth)//inps.skipwidth

    if (inps.numWinDown != -1):
        objOffset.numberWindowDown = inps.numWinDown
    if (inps.numWinAcross != -1):
        objOffset.numberWindowAcross = inps.numWinAcross
    print("offset field length: ",objOffset.numberWindowDown)
    print("offset field width: ",objOffset.numberWindowAcross)

    # window size
    objOffset.windowSizeHeight = inps.winhgt
    objOffset.windowSizeWidth = inps.winwidth
    print('cross correlation window size: {} by {}'.format(objOffset.windowSizeHeight, objOffset.windowSizeWidth))

    # search range
    objOffset.halfSearchRangeDown = inps.srchgt
    objOffset.halfSearchRangeAcross = inps.srcwidth
    print('half search range: {} by {}'.format(inps.srchgt, inps.srcwidth))

    # starting pixel

    objOffset.referenceStartPixelDownStatic = inps.margin
    objOffset.referenceStartPixelAcrossStatic = inps.margin
 
    # skip size
    
    objOffset.skipSampleDown = inps.skiphgt
    objOffset.skipSampleAcross = inps.skipwidth
    print('search step: {} by {}'.format(inps.skiphgt, inps.skipwidth))

    # oversample raw data (SLC)
    objOffset.rawDataOversamplingFactor = inps.raw_oversample
    print('raw data oversampling factor:', inps.raw_oversample)

    # correlation surface
    if inps.corr_win_size == -1:
        corr_win_size_orig = min(inps.srchgt, inps.srcwidth) * inps.raw_oversample + 1
        inps.corr_win_size = np.power(2, int(np.log2(corr_win_size_orig)))
        objOffset.corrSurfaceZoomInWindow = inps.corr_win_size
        print('correlation surface zoom-in window size:', inps.corr_win_size)

    objOffset.corrSufaceOverSamplingMethod = 0
    objOffset.corrSurfaceOverSamplingFactor = inps.corr_oversample
    print('correlation surface oversampling factor:', inps.corr_oversample)

    # output filenames
    objOffset.offsetImageName = str(inps.outprefix) + str(inps.outsuffix) + '.bip'
    objOffset.grossOffsetImageName = str(inps.outprefix) + str(inps.outsuffix) + '_gross.bip'
    objOffset.snrImageName = str(inps.outprefix) + str(inps.outsuffix) + '_snr.bip'
    objOffset.covImageName = str(inps.outprefix) + str(inps.outsuffix) + '_cov.bip'
    print("offsetfield: ",objOffset.offsetImageName)
    print("gross offsetfield: ",objOffset.grossOffsetImageName)
    print("snr: ",objOffset.snrImageName)
    print("cov: ",objOffset.covImageName)

    offsetImageName = objOffset.offsetImageName.decode('utf8')
    grossOffsetImageName = objOffset.grossOffsetImageName.decode('utf8')
    snrImageName = objOffset.snrImageName.decode('utf8')
    covImageName = objOffset.covImageName.decode('utf8')

    print(offsetImageName)
    print(inps.redo)
    if os.path.exists(offsetImageName) and not inps.redo:
        print('offsetfield file exists')
        return 0

    # generic control
    objOffset.numberWindowDownInChunk = inps.numWinDownInChunk
    objOffset.numberWindowAcrossInChunk = inps.numWinAcrossInChunk
    objOffset.useMmap = 0
    objOffset.mmapSize = 8
    objOffset.setupParams()

    ## Set Gross Offset ###
    if inps.gross == 0:
        print("Set constant grossOffset")
        print("By default, the gross offsets are zero")
        print("You can override the default values here")
        objOffset.setConstantGrossOffset(0, 0)

    else:
        print("Set varying grossOffset")
        print("By default, the gross offsets are zero")
        print("You can override the default grossDown and grossAcross arrays here")
        objOffset.setVaryingGrossOffset(np.zeros(shape=grossDown.shape,dtype=np.int32),
                                        np.zeros(shape=grossAcross.shape,dtype=np.int32))

    # check
    objOffset.checkPixelInImageRange()

    # Run the code
    print('Running PyCuAmpcor')

    objOffset.runAmpcor()
    print('Finished')

    sar.finalizeImage()
    sim.finalizeImage()

    # Finalize the results
    # offsetfield
    outImg = isceobj.createImage()
    outImg.setDataType('FLOAT')
    outImg.setFilename(offsetImageName)
    outImg.setBands(2)
    outImg.scheme = 'BIP'
    outImg.setWidth(objOffset.numberWindowAcross)
    outImg.setLength(objOffset.numberWindowDown)
    outImg.setAccessMode('read')
    outImg.renderHdr()

    # gross offsetfield
    outImg = isceobj.createImage()
    outImg.setDataType('FLOAT')
    outImg.setFilename(grossOffsetImageName)
    outImg.setBands(2)
    outImg.scheme = 'BIP'
    outImg.setWidth(objOffset.numberWindowAcross)
    outImg.setLength(objOffset.numberWindowDown)
    outImg.setAccessMode('read')
    outImg.renderHdr()

    # snr
    snrImg = isceobj.createImage()
    snrImg.setFilename(snrImageName)
    snrImg.setDataType('FLOAT')
    snrImg.setBands(1)
    snrImg.setWidth(objOffset.numberWindowAcross)
    snrImg.setLength(objOffset.numberWindowDown)
    snrImg.setAccessMode('read')
    snrImg.renderHdr()

    # cov
    covImg = isceobj.createImage()
    covImg.setFilename(covImageName)
    covImg.setDataType('FLOAT')
    covImg.setBands(3)
    covImg.scheme = 'BIP'
    covImg.setWidth(objOffset.numberWindowAcross)
    covImg.setLength(objOffset.numberWindowDown)
    covImg.setAccessMode('read')
    covImg.renderHdr()

    return


def main(iargs=None):

    inps = cmdLineParse(iargs)
    outDir = os.path.dirname(inps.outprefix)
    print(inps.outprefix)

    os.makedirs(outDir, exist_ok=True)

    estimateOffsetField(inps.reference, inps.secondary, inps)
    return



if __name__ == '__main__':

    main()
