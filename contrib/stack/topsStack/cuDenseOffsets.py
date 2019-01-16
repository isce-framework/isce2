#!/usr/bin/env python3

# author: Minyan Zhong 

import numpy as np 
import argparse
import os
import isce
import isceobj
import shelve
import datetime
from isceobj.Location.Offset import OffsetField
from iscesys.StdOEL.StdOELPy import create_writer
#from mroipac.ampcor.DenseAmpcor import DenseAmpcor

from PyCuAmpcor import PyCuAmpcor
from grossOffsets import grossOffsets

#from isceobj.Utils.denseoffsets import denseoffsets
from isceobj.Util.decorators import use_api

from pprint import pprint

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

    parser.add_argument('-op','--outprefix', type=str, dest='outprefix', default='dense_ampcor',
            help='Output prefix')

    parser.add_argument('-os','--outsuffix', type=str, dest='outsuffix', default='dense_ampcor',
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

    ###Loading the slave image object
    sim = isceobj.createSlcImage()
    sim.load(slave+'.xml')
    sim.setAccessMode('READ')
    sim.createImage()


    ###Loading the master image object
    sar = isceobj.createSlcImage()
    sar.load(master + '.xml')
    sar.setAccessMode('READ')
    sar.createImage()


    width = sar.getWidth()
    length = sar.getLength()

    objOffset = PyCuAmpcor()
    
    objOffset.algorithm = 0
    objOffset.deviceID = inps.gpuid  # -1:let system find the best GPU
    objOffset.nStreams =   2 #cudaStreams 
    objOffset.derampMethod = inps.deramp
    print(objOffset.derampMethod)

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


    print("nlines: ",objOffset.numberWindowDown)
    print("ncolumns: ",objOffset.numberWindowAcross)


    # window size
    objOffset.windowSizeHeight = inps.winhgt
    objOffset.windowSizeWidth = inps.winwidth
    
    print(objOffset.windowSizeHeight)
    print(objOffset.windowSizeWidth)
 
    # search range
    objOffset.halfSearchRangeDown = inps.srchgt
    objOffset.halfSearchRangeAcross = inps.srcwidth
    print(inps.srchgt,inps.srcwidth)

    # starting pixel
    objOffset.masterStartPixelDownStatic = inps.margin
    objOffset.masterStartPixelAcrossStatic = inps.margin
 
    # skip size
    objOffset.skipSampleDown = inps.skiphgt
    objOffset.skipSampleAcross = inps.skipwidth

    # oversampling
    objOffset.corrSufaceOverSamplingMethod = 0
    objOffset.corrSurfaceOverSamplingFactor = inps.oversample
    #objOffset.rawDataOversamplingFactor = 4
    
    # output filenames
    objOffset.offsetImageName = str(inps.outprefix) + str(inps.outsuffix) + '.bip'
    objOffset.grossOffsetImageName = str(inps.outprefix) + str(inps.outsuffix) + '_gross.bip'
    objOffset.snrImageName = str(inps.outprefix) + str(inps.outsuffix) + '_snr.bip'

    print("offsetfield: ",objOffset.offsetImageName)
    print("gross offsetfield: ",objOffset.grossOffsetImageName)
    print("snr: ",objOffset.snrImageName)

    offsetImageName = objOffset.offsetImageName.decode('utf8')
    #print(type(offsetImageName))
    #print(offsetImageName)
    #print(type(objOffset.numberWindowAcross))
    grossOffsetImageName = objOffset.grossOffsetImageName.decode('utf8')
    snrImageName = objOffset.snrImageName.decode('utf8')

    print(offsetImageName)
    print(inps.redo)
    if os.path.exists(offsetImageName) and inps.redo==0:
        print('offsetfield file exists')
    else:
        # generic control
        objOffset.numberWindowDownInChunk = 5
        objOffset.numberWindowAcrossInChunk = 5
        objOffset.mmapSize = 16
    
        objOffset.setupParams()
        
        ## Set Gross Offset ###
    
        if inps.gross == 0:
            objOffset.setConstantGrossOffset(0, 0)
        else:
    
            print("Setting up grossOffset...")
    
            objGrossOff = grossOffsets()
            
            objGrossOff.setXSize(width)
            objGrossOff.setYize(length)
            objGrossOff.setMargin(inps.margin)
            objGrossOff.setWinSizeHgt(inps.winhgt)
            objGrossOff.setWinSizeWidth(inps.winwidth)
            objGrossOff.setSearchSizeHgt(inps.srchgt)
            objGrossOff.setSearchSizeWidth(inps.srcwidth)
            objGrossOff.setSkipSizeHgt(inps.skiphgt)
            objGrossOff.setSkipSizeWidth(inps.skipwidth)
            objGrossOff.setLatFile(inps.lat)
            objGrossOff.setLonFile(inps.lon)
            objGrossOff.setLosFile(inps.los)
            objGrossOff.setMasterFile(inps.masterxml)
            objGrossOff.setbTemp(inps.bTemp)
            

     
            grossDown, grossAcross = objGrossOff.runGrossOffsets()
        
            # change nan to 0
            grossDown = np.nan_to_num(grossDown)
            grossAcross = np.nan_to_num(grossAcross)
        
            print("Before plotting the gross offsets (min and max): ", np.nanmin(grossDown),np.nanmax(grossDown))
            print("Before plotting the gross offsets (min and max): ", np.rint(np.nanmin(grossDown)),np.rint(np.nanmax(grossDown)))
        
            grossDown = np.int32(np.rint(grossDown.ravel()))
            grossAcross = np.int32(np.rint(grossAcross.ravel()))
        
            print(np.amin(grossDown), np.amax(grossDown))
            print(np.amin(grossAcross), np.amax(grossAcross))
        
            print(grossDown.shape)
            print(grossDown.shape)
        
            objOffset.setVaryingGrossOffset(grossDown, grossAcross)
            #objOffset.setVaryingGrossOffset(np.zeros(shape=grossDown.shape,dtype=np.int32), np.zeros(shape=grossAcross.shape,dtype=np.int32))
       
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

    return objOffset
            
def main(iargs=None):        

    inps = cmdLineParse(iargs)
    outDir = os.path.dirname(inps.outprefix)
    print(inps.outprefix)
    if not os.path.exists(outDir):
         os.makedirs(outDir)
    
    objOffset = estimateOffsetField(inps.master, inps.slave, inps)

if __name__ == '__main__':
    
    main()
