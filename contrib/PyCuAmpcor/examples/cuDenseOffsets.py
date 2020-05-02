#!/usr/bin/env python3

# Author: Minyan Zhong, Lijun Zhu 

import argparse
import os
import isce
import isceobj
from isceobj.Util.decorators import use_api

import numpy as np 
from contrib.PyCuAmpcor.PyCuAmpcor import PyCuAmpcor

def createParser():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser( description='Generate offset field between two Sentinel slc')
    parser.add_argument('-m','--master', type=str, dest='master', required=True,
            help='Master image')
    parser.add_argument('-s', '--slave',type=str, dest='slave', required=True,
            help='Slave image')
    parser.add_argument('-l', '--lat',type=str, dest='lat', required=False,
           help='Latitude')
    parser.add_argument('-L', '--lon',type=str, dest='lon', required=False,
           help='Longitude')
    parser.add_argument('--los',type=str, dest='los', required=False,
           help='Line of Sight')
    parser.add_argument('--masterxml',type=str, dest='masterxml', required=False,
           help='Master Image Xml File')
    parser.add_argument('--ww', type=int, dest='winwidth', default=64,
            help='Window Width')
    parser.add_argument('--wh', type=int, dest='winhgt', default=64,
            help='Window height')
    parser.add_argument('--sw', type=int, dest='srcwidth', default=20,
            help='Search window width')
    parser.add_argument('--sh', type=int, dest='srchgt', default=20,
            help='Search window height')
    parser.add_argument('--mm', type=int, dest='margin', default=50,
            help='Margin')
    parser.add_argument('--kw', type=int, dest='skipwidth', default=64,
            help='Skip across')
    parser.add_argument('--kh', type=int, dest='skiphgt', default=64,
            help='Skip down')

    parser.add_argument('--nwa', type=int, dest='numWinAcross', default=-1,
            help='Number of Window Across')
    parser.add_argument('--nwd', type=int, dest='numWinDown', default=-1,
            help='Number of Window Down')

    parser.add_argument('--nwac', type=int, dest='numWinAcrossInChunk', default=1,
            help='Number of Window Across in Chunk')
    parser.add_argument('--nwdc', type=int, dest='numWinDownInChunk', default=1,
            help='Number of Window Down in Chunk')

    parser.add_argument('-op','--outprefix', type=str, dest='outprefix', default='dense_ampcor', required=True,
            help='Output prefix')

    parser.add_argument('-os','--outsuffix', type=str, dest='outsuffix',default='dense_ampcor',
            help='Output suffix')

    parser.add_argument('-g','--gross', type=int, dest='gross', default=0,
            help='Use gross offset or not')

    parser.add_argument('--aa', type=int, dest='azshift', default=0,
            help='Gross azimuth offset')

    parser.add_argument('--rr', type=int, dest='rgshift', default=0,
            help='Gross range offset')

    parser.add_argument('--oo', type=int, dest='oversample', default=32,
            help = 'Oversampling factor')

    parser.add_argument('-r', '--redo', dest='redo', type=int, default=0
       , help='To redo or not')

    parser.add_argument('-drmp', '--deramp', dest='deramp', type=int, default=0
       , help='deramp method (0: mag, 1: complex)')

    parser.add_argument('-gid', '--gpuid', dest='gpuid', type=int, default=-1
       , help='GPU ID')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    inps =  parser.parse_args(args=iargs)

    return inps

@use_api
def estimateOffsetField(master, slave, inps=None):

    import pathlib

    ###Loading the slave image object
    sim = isceobj.createSlcImage()
    sim.load(pathlib.Path(slave).with_suffix('.xml'))
    sim.setAccessMode('READ')
    sim.createImage()


    ###Loading the master image object
    sar = isceobj.createSlcImage()
    sar.load(pathlib.Path(master).with_suffix('.xml'))
    sar.setAccessMode('READ')
    sar.createImage()

    width = sar.getWidth()
    length = sar.getLength()

    objOffset = PyCuAmpcor()
    
    objOffset.algorithm = 0
    objOffset.deviceID = inps.gpuid  # -1:let system find the best GPU
    objOffset.nStreams =   1 #cudaStreams 
    objOffset.derampMethod = inps.deramp

    objOffset.masterImageName = master
    objOffset.masterImageHeight = length
    objOffset.masterImageWidth = width
    objOffset.slaveImageName = slave
    objOffset.slaveImageHeight = length
    objOffset.slaveImageWidth = width

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
    
    # search range
    objOffset.halfSearchRangeDown = inps.srchgt
    objOffset.halfSearchRangeAcross = inps.srcwidth

    # starting pixel
    objOffset.masterStartPixelDownStatic = inps.margin
    objOffset.masterStartPixelAcrossStatic = inps.margin
 
    # skip size
    objOffset.skipSampleDown = inps.skiphgt
    objOffset.skipSampleAcross = inps.skipwidth

    # oversampling
    objOffset.corrSufaceOverSamplingMethod = 0
    objOffset.corrSurfaceOverSamplingFactor = inps.oversample

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

    if os.path.exists(offsetImageName) and inps.redo==0:

        print('offsetfield file exists')
        exit()

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
        objOffset.setVaryingGrossOffset(np.zeros(shape=grossDown.shape,dtype=np.int32), np.zeros(shape=grossAcross.shape,dtype=np.int32))
   
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

    return 0
            
def main(iargs=None):        

    inps = cmdLineParse(iargs)
    outDir = os.path.dirname(inps.outprefix)
    print(inps.outprefix)
    if not os.path.exists(outDir):
         os.makedirs(outDir)
    
    estimateOffsetField(inps.master, inps.slave, inps)

if __name__ == '__main__':
    
    main()
