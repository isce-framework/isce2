#!/usr/bin/env python3

# Author: Minyan Zhong, Lijun Zhu


import os
import sys
import time
import argparse
import numpy as np
from osgeo import gdal

import isce
import isceobj
from isceobj.Util.decorators import use_api
from isceobj.Util.ImageUtil import ImageLib as IML
from contrib.PyCuAmpcor.PyCuAmpcor import PyCuAmpcor


EXAMPLE = '''example
  cuDenseOffsets.py -r ./SLC/20151120/20151120.slc.full -s ./SLC/20151214/20151214.slc.full
  cuDenseOffsets.py -r ./SLC/20151120/20151120.slc.full -s ./SLC/20151214/20151214.slc.full --outprefix ./offsets/20151120_20151214/offset --ww 256 --wh 256 --sw 8 --sh 8 --oo 32 --kw 300 --kh 100 --nwac 100 --nwdc 1 --gpuid 2

  # offset and its geometry
  # tip: re-run with --full/out-geom and without --redo to generate geometry only
  cuDenseOffsets.py -r ./SLC/20151120/20151120.slc.full -s ./SLC/20151214/20151214.slc.full --outprefix ./offsets/20151120_20151214/offset --ww 256 --wh 256 --sw 8 --sh 8 --oo 32 --kw 300 --kh 100 --nwac 100 --nwdc 1 --gpuid 2 --full-geom ./geom_reference --out-geom ./offset/geom_reference
'''


def createParser():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Generate offset field between two SLCs',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)

    # input/output
    parser.add_argument('-r','--reference', type=str, dest='reference', required=True,
                        help='Reference image')
    parser.add_argument('-s', '--secondary',type=str, dest='secondary', required=True,
                        help='Secondary image')
    parser.add_argument('--fix-xml','--fix-image-xml', dest='fixImageXml', action='store_true',
                        help='Fix the image file path in the XML file. Enable this if input files havee been moved.')

    parser.add_argument('--op','--outprefix','--output-prefix', type=str, dest='outprefix',
                        default='offset', required=True,
                        help='Output prefix (default: %(default)s).')
    parser.add_argument('--os','--outsuffix', type=str, dest='outsuffix', default='',
                        help='Output suffix (default: %(default)s).')

    # window size settings
    parser.add_argument('--ww', type=int, dest='winwidth', default=64,
                        help='Window width (default: %(default)s).')
    parser.add_argument('--wh', type=int, dest='winhgt', default=64,
                        help='Window height (default: %(default)s).')
    parser.add_argument('--sw', type=int, dest='srcwidth', default=20,
                        help='Half search range along width, (default: %(default)s, recommend: 4-32).')
    parser.add_argument('--sh', type=int, dest='srchgt', default=20,
                        help='Half search range along height (default: %(default)s, recommend: 4-32).')
    parser.add_argument('--kw', type=int, dest='skipwidth', default=64,
                        help='Skip across (default: %(default)s).')
    parser.add_argument('--kh', type=int, dest='skiphgt', default=64,
                        help='Skip down (default: %(default)s).')

    # determine the number of windows
    # either specify the starting pixel and the number of windows,
    # or by setting them to -1, let the script to compute these parameters
    parser.add_argument('--mm', type=int, dest='margin', default=0,
                        help='Margin (default: %(default)s).')
    parser.add_argument('--nwa', type=int, dest='numWinAcross', default=-1,
                        help='Number of window across (default: %(default)s to be auto-determined).')
    parser.add_argument('--nwd', type=int, dest='numWinDown', default=-1,
                        help='Number of window down (default: %(default)s).')
    parser.add_argument('--startpixelac', dest='startpixelac', type=int, default=-1,
                        help='Starting Pixel across of the reference image.' +
                             'Default: %(default)s to be determined by margin and search range.')
    parser.add_argument('--startpixeldw', dest='startpixeldw', type=int, default=-1,
                        help='Starting Pixel down of the reference image.' +
                             'Default: %(default)s to be determined by margin and search range.')

    # cross-correlation algorithm
    parser.add_argument('--alg', '--algorithm', dest='algorithm', type=int, default=0,
                        help='cross-correlation algorithm (0 = frequency domain, 1 = time domain) (default: %(default)s).')
    parser.add_argument('--raw-osf','--raw-over-samp-factor', type=int, dest='raw_oversample',
                        default=2, choices=range(2,5),
                        help='anti-aliasing oversampling factor, equivalent to i_ovs in RIOPAC (default: %(default)s).')
    parser.add_argument('--drmp', '--deramp', dest='deramp', type=int, default=0,
                        help='deramp method (0: mag for TOPS, 1:complex with linear ramp) (default: %(default)s).')

    # gross offset
    gross = parser.add_argument_group('Initial gross offset')
    gross.add_argument('-g','--gross', type=int, dest='gross', default=0,
                       help='Use varying gross offset or not')
    gross.add_argument('--aa', type=int, dest='azshift', default=0,
                       help='Gross azimuth offset (default: %(default)s).')
    gross.add_argument('--rr', type=int, dest='rgshift', default=0,
                       help='Gross range offset (default: %(default)s).')
    gross.add_argument('--gf', '--gross-file', type=str, dest='gross_offset_file',
                       help='Varying gross offset input file')
    gross.add_argument('--mg', '--merge-gross-offset', type=int, dest='merge_gross_offset', default=0,
                       help='Whether to merge gross offset to the output offset image (default: %(default)s).')

    corr = parser.add_argument_group('Correlation surface')
    corr.add_argument('--corr-stat-size', type=int, dest='corr_stat_win_size', default=21,
                      help='Zoom-in window size of the correlation surface for statistics(snr/variance) (default: %(default)s).')
    corr.add_argument('--corr-srch-size', type=int, dest='corr_srch_size', default=4,
                      help='(half) Zoom-in window size of the correlation surface for oversampling, ' \
                      'equivalent to i_srcp in RIOPAC (default: %(default)s).')
    corr.add_argument('--corr-osf', '--oo', '--corr-over-samp-factor', type=int, dest='corr_oversample', default=32,
                      help = 'Oversampling factor of the zoom-in correlation surface (default: %(default)s).')
    corr.add_argument('--corr-osm', '--corr-over-samp-method', type=int, dest='corr_oversamplemethod', default=0,
                      help = 'Oversampling method for the correlation surface 0=fft, 1=sinc (default: %(default)s).')

    geom = parser.add_argument_group('Geometry', 'generate corresponding geometry datasets ')
    geom.add_argument('--full-geom', dest='full_geometry_dir', type=str,
                      help='(Input) Directory of geometry files in full resolution.')
    geom.add_argument('--out-geom', dest='out_geometry_dir', type=str,
                      help='(Output) Directory of geometry files corresponding to the offset field.')

    # gpu settings
    proc = parser.add_argument_group('Processing parameters')
    proc.add_argument('--gpuid', '--gid', '--gpu-id', dest='gpuid', type=int, default=0,
                        help='GPU ID (default: %(default)s).')
    proc.add_argument('--nstreams', dest='nstreams', type=int, default=2,
                        help='Number of cuda streams (default: %(default)s).')
    proc.add_argument('--usemmap', dest='usemmap', type=int, default=1,
                        help='Whether to use memory map for loading image files (default: %(default)s).')
    proc.add_argument('--mmapsize', dest='mmapsize', type=int, default=8,
                        help='The memory map buffer size in GB (default: %(default)s).')
    proc.add_argument('--nwac', type=int, dest='numWinAcrossInChunk', default=10,
                        help='Number of window across in a chunk/batch (default: %(default)s).')
    proc.add_argument('--nwdc', type=int, dest='numWinDownInChunk', default=1,
                        help='Number of window down in a chunk/batch (default: %(default)s).')

    proc.add_argument('--redo', dest='redo', action='store_true',
                        help='To redo by force (ignore the existing offset fields).')

    return parser


def cmdLineParse(iargs = None):
    parser = createParser()
    inps = parser.parse_args(args=iargs)

    return inps


@use_api
def estimateOffsetField(reference, secondary, inps=None):
    """Estimte offset field using PyCuAmpcor.
    Parameters: reference - str, path of the reference SLC file
                secondary - str, path of the secondary SLC file
                inps      - Namespace, input configuration
    Returns:    objOffset - PyCuAmpcor object
                geomDict  - dict, geometry location info of the offset field
    """

    # update file path in xml file
    if inps.fixImageXml:
        for fname in [reference, secondary]:
            fname = os.path.abspath(fname)
            img = IML.loadImage(fname)[0]
            img.filename = fname
            img.setAccessMode('READ')
            img.renderHdr()

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

    # create a PyCuAmpcor instance
    objOffset = PyCuAmpcor()

    objOffset.algorithm = inps.algorithm
    objOffset.deviceID = inps.gpuid
    objOffset.nStreams = inps.nstreams #cudaStreams
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

    # if using gross offset, adjust the margin
    margin = max(inps.margin, abs(inps.azshift), abs(inps.rgshift))

    # determine the number of windows down and across
    # that's also the size of the output offset field
    objOffset.numberWindowDown = inps.numWinDown if inps.numWinDown > 0 \
        else (length-2*margin-2*inps.srchgt-inps.winhgt)//inps.skiphgt
    objOffset.numberWindowAcross = inps.numWinAcross if inps.numWinAcross > 0 \
        else (width-2*margin-2*inps.srcwidth-inps.winwidth)//inps.skipwidth
    print('the number of windows: {} by {}'.format(objOffset.numberWindowDown, objOffset.numberWindowAcross))

    # window size
    objOffset.windowSizeHeight = inps.winhgt
    objOffset.windowSizeWidth = inps.winwidth
    print('window size for cross-correlation: {} by {}'.format(objOffset.windowSizeHeight, objOffset.windowSizeWidth))

    # search range
    objOffset.halfSearchRangeDown = inps.srchgt
    objOffset.halfSearchRangeAcross = inps.srcwidth
    print('initial search range: {} by {}'.format(inps.srchgt, inps.srcwidth))

    # starting pixel
    objOffset.referenceStartPixelDownStatic = inps.startpixeldw if inps.startpixeldw != -1 \
        else margin + objOffset.halfSearchRangeDown    # use margin + halfSearchRange instead
    objOffset.referenceStartPixelAcrossStatic = inps.startpixelac if inps.startpixelac != -1 \
        else margin + objOffset.halfSearchRangeAcross

    print('the first pixel in reference image is: ({}, {})'.format(
        objOffset.referenceStartPixelDownStatic, objOffset.referenceStartPixelAcrossStatic))

    # skip size
    objOffset.skipSampleDown = inps.skiphgt
    objOffset.skipSampleAcross = inps.skipwidth
    print('search step: {} by {}'.format(inps.skiphgt, inps.skipwidth))

    # oversample raw data (SLC)
    objOffset.rawDataOversamplingFactor = inps.raw_oversample

    # correlation surface
    objOffset.corrStatWindowSize = inps.corr_stat_win_size

    corr_win_size = 2*inps.corr_srch_size*inps.raw_oversample
    objOffset.corrSurfaceZoomInWindow = corr_win_size
    print('correlation surface zoom-in window size:', corr_win_size)

    objOffset.corrSurfaceOverSamplingMethod = inps.corr_oversamplemethod
    objOffset.corrSurfaceOverSamplingFactor = inps.corr_oversample
    print('correlation surface oversampling factor:', inps.corr_oversample)

    # output filenames
    fbase = '{}{}'.format(inps.outprefix, inps.outsuffix)
    objOffset.offsetImageName = fbase + '.bip'
    objOffset.grossOffsetImageName = fbase + '_gross.bip'
    objOffset.snrImageName = fbase + '_snr.bip'
    objOffset.covImageName = fbase + '_cov.bip'
    print("offsetfield: ",objOffset.offsetImageName)
    print("gross offsetfield: ",objOffset.grossOffsetImageName)
    print("snr: ",objOffset.snrImageName)
    print("cov: ",objOffset.covImageName)

    # whether to include the gross offset in offsetImage
    objOffset.mergeGrossOffset = inps.merge_gross_offset

    try:
        offsetImageName = objOffset.offsetImageName.decode('utf8')
        grossOffsetImageName = objOffset.grossOffsetImageName.decode('utf8')
        snrImageName = objOffset.snrImageName.decode('utf8')
        covImageName = objOffset.covImageName.decode('utf8')
    except:
        offsetImageName = objOffset.offsetImageName
        grossOffsetImageName = objOffset.grossOffsetImageName
        snrImageName = objOffset.snrImageName
        covImageName = objOffset.covImageName

    # generic control
    objOffset.numberWindowDownInChunk = inps.numWinDownInChunk
    objOffset.numberWindowAcrossInChunk = inps.numWinAcrossInChunk
    objOffset.useMmap = inps.usemmap
    objOffset.mmapSize = inps.mmapsize

    # setup and check parameters
    objOffset.setupParams()

    ## Set Gross Offset ###
    if inps.gross == 0: # use static grossOffset
        print('Set constant grossOffset ({}, {})'.format(inps.azshift, inps.rgshift))
        objOffset.setConstantGrossOffset(inps.azshift, inps.rgshift)

    else: # use varying offset
        print("Set varying grossOffset from file {}".format(inps.gross_offset_file))
        grossOffset = np.fromfile(inps.gross_offset_file, dtype=np.int32)
        numberWindows = objOffset.numberWindowDown*objOffset.numberWindowAcross
        if grossOffset.size != 2*numberWindows :
            print(('WARNING: The input gross offsets do not match the number of windows:'
                   ' {} by {} in int32 type').format(objOffset.numberWindowDown,
                                                     objOffset.numberWindowAcross))
            return 0

        grossOffset = grossOffset.reshape(numberWindows, 2)
        grossAzimuthOffset = grossOffset[:, 0]
        grossRangeOffset = grossOffset[:, 1]
        # enforce C-contiguous flag
        grossAzimuthOffset = grossAzimuthOffset.copy(order='C')
        grossRangeOffset = grossRangeOffset.copy(order='C')
        # set varying gross offset
        objOffset.setVaryingGrossOffset(grossAzimuthOffset, grossRangeOffset)

    # check
    objOffset.checkPixelInImageRange()

    # save output geometry location info
    geomDict = {
        'x_start'   : objOffset.referenceStartPixelAcrossStatic + int(objOffset.windowSizeWidth / 2.),
        'y_start'   : objOffset.referenceStartPixelDownStatic + int(objOffset.windowSizeHeight / 2.),
        'x_step'    : objOffset.skipSampleAcross,
        'y_step'    : objOffset.skipSampleDown,
        'x_win_num' : objOffset.numberWindowAcross,
        'y_win_num' : objOffset.numberWindowDown,
    }

    # check redo
    print('redo: ', inps.redo)
    if not inps.redo:
        offsetImageName = '{}{}.bip'.format(inps.outprefix, inps.outsuffix)
        if os.path.exists(offsetImageName):
            print('offset field file: {} exists and w/o redo, skip re-estimation.'.format(offsetImageName))
            return objOffset, geomDict

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

    return objOffset, geomDict


def prepareGeometry(full_dir, out_dir, x_start, y_start, x_step, y_step, x_win_num, y_win_num,
                    fbases=['hgt','lat','lon','los','shadowMask','waterMask']):
    """Generate multilooked geometry datasets in the same grid as the estimated offset field
    from the full resolution geometry datasets.
    Parameters: full_dir    - str, path of input  geometry directory in full resolution
                out_dir     - str, path of output geometry directory
                x/y_start   - int, starting column/row number
                x/y_step    - int, output pixel step in column/row direction
                x/y_win_num - int, number of columns/rows
    """
    full_dir = os.path.abspath(full_dir)
    out_dir = os.path.abspath(out_dir)

    # grab the file extension for full resolution file
    full_exts = ['.rdr.full','.rdr'] if full_dir != out_dir else ['.rdr.full']
    full_exts = [e for e in full_exts if os.path.isfile(os.path.join(full_dir, '{f}{e}'.format(f=fbases[0], e=e)))]
    if len(full_exts) == 0:
        raise ValueError('No full resolution {}.rdr* file found in: {}'.format(fbases[0], full_dir))
    full_ext = full_exts[0]

    print('-'*50)
    print('generate the corresponding multi-looked geometry datasets using gdal ...')
    # input files
    in_files = [os.path.join(full_dir, '{f}{e}'.format(f=f, e=full_ext)) for f in fbases]
    in_files = [i for i in in_files if os.path.isfile(i)]
    fbases = [os.path.basename(i).split('.')[0] for i in in_files]

    # output files
    out_files = [os.path.join(out_dir, '{}.rdr'.format(i)) for i in fbases]
    os.makedirs(out_dir, exist_ok=True)

    for i in range(len(in_files)):
        in_file = in_files[i]
        out_file = out_files[i]

        # input file size
        ds = gdal.Open(in_file, gdal.GA_ReadOnly)
        in_wid = ds.RasterXSize
        in_len = ds.RasterYSize

        # starting column/row and column/row number
        src_win = [x_start, y_start, x_win_num * x_step, y_win_num * y_step]
        print('read {} from file: {}'.format(src_win, in_file))

        # write binary data file
        print('write file: {}'.format(out_file))
        opts = gdal.TranslateOptions(format='ENVI',
                                     width=x_win_num,
                                     height=y_win_num,
                                     srcWin=src_win,
                                     noData=0)
        gdal.Translate(out_file, ds, options=opts)
        ds = None

        # write VRT file
        print('write file: {}'.format(out_file+'.vrt'))
        ds = gdal.Open(out_file, gdal.GA_ReadOnly)
        gdal.Translate(out_file+'.vrt', ds, options=gdal.TranslateOptions(format='VRT'))
        ds = None

    return


def main(iargs=None):
    inps = cmdLineParse(iargs)
    start_time = time.time()

    print(inps.outprefix)
    outDir = os.path.dirname(inps.outprefix)
    os.makedirs(outDir, exist_ok=True)

    # estimate offset
    geomDict = estimateOffsetField(inps.reference, inps.secondary, inps)[1]

    # generate geometry
    if inps.full_geometry_dir and inps.out_geometry_dir:
        prepareGeometry(inps.full_geometry_dir, inps.out_geometry_dir,
                        x_start=geomDict['x_start'],
                        y_start=geomDict['y_start'],
                        x_step=geomDict['x_step'],
                        y_step=geomDict['y_step'],
                        x_win_num=geomDict['x_win_num'],
                        y_win_num=geomDict['y_win_num'])

    m, s = divmod(time.time() - start_time, 60)
    print('time used: {:02.0f} mins {:02.1f} secs.\n'.format(m, s))
    return

if __name__ == '__main__':
    main(sys.argv[1:])
