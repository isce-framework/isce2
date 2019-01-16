#!/usr/bin/env python3

import numpy as np 
import argparse
import os
import isce
import isceobj
import shelve
import datetime
from isceobj.Location.Offset import OffsetField
from iscesys.StdOEL.StdOELPy import create_writer
from mroipac.ampcor.DenseAmpcor import DenseAmpcor
#from isceobj.Utils.denseoffsets import denseoffsets
#import pickle
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

    parser.add_argument('-o','--outprefix', type=str, dest='outprefix', default='dense_ampcor',
            help='Output prefix')

    parser.add_argument('--aa', type=int, dest='azshift', default=0,
            help='Gross azimuth offset')

    parser.add_argument('--rr', type=int, dest='rgshift', default=0,
            help='Gross range offset')
    parser.add_argument('--oo', type=int, dest='oversample', default=32,
            help = 'Oversampling factor')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    inps =  parser.parse_args(args=iargs)

    return inps

@use_api
def estimateOffsetField(master, slave, inps=None):
    '''
    Estimate offset field between burst and simamp.
    '''

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

    objOffset = DenseAmpcor(name='dense')
    objOffset.configure()

#   objOffset.numberThreads = 6
    objOffset.setWindowSizeWidth(inps.winwidth)
    objOffset.setWindowSizeHeight(inps.winhgt)
    objOffset.setSearchWindowSizeWidth(inps.srcwidth)
    objOffset.setSearchWindowSizeHeight(inps.srchgt)
    objOffset.skipSampleAcross = inps.skipwidth
    objOffset.skipSampleDown = inps.skiphgt
    objOffset.margin = inps.margin 
    objOffset.oversamplingFactor = inps.oversample

    objOffset.setAcrossGrossOffset(inps.rgshift)
    objOffset.setDownGrossOffset(inps.azshift)

##  For Debug
#    print(vars(inps))
#    pprint(vars(inps))
#    print(stop)

    objOffset.setFirstPRF(1.0)
    objOffset.setSecondPRF(1.0)

#    print(sar.dataType)
    if sar.dataType.startswith('C'):
        objOffset.setImageDataType1('mag')
    else:
        objOffset.setImageDataType1('real')

    if sim.dataType.startswith('C'):
        objOffset.setImageDataType2('mag') 
    else:
        objOffset.setImageDataType2('real')

    objOffset.offsetImageName = inps.outprefix + '.bil'
    objOffset.snrImageName = inps.outprefix +'_snr.bil'


    objOffset.denseampcor(sar, sim)

    sar.finalizeImage()
    sim.finalizeImage()
    return objOffset



def main(iargs=None):        
    '''
    Generate offset fields burst by burst.
    '''

    inps = cmdLineParse(iargs)
    outDir = os.path.dirname(inps.outprefix)
    if not os.path.exists(outDir):
         os.makedirs(outDir)
    
    objOffset = estimateOffsetField(inps.master, inps.slave, inps)

    
    print('Top left corner of offset image: ', objOffset.locationDown[0][0],objOffset.locationAcross[0][0])

if __name__ == '__main__':
    
    main()
