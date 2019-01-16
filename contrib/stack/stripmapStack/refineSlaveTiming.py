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
from mroipac.ampcor.Ampcor import Ampcor
import pickle


def createParser():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser( description='Generate offset field between two Sentinel swaths')
    parser.add_argument('-m','--master', type=str, dest='master', required=True,
            help='Master image')
    parser.add_argument('--mm', type=str, dest='metamaster', default=None,
            help='Master meta data dir')
    parser.add_argument('-s', '--slave', type=str, dest='slave', required=True,
            help='Slave image')
    parser.add_argument('--ss', type=str, dest='metaslave', default=None,
            help='Slave meta data dir')
    parser.add_argument('-o', '--outfile',type=str, required=True, dest='outfile',
            help='Misregistration in subpixels')

    parser.add_argument('--aa', dest='azazorder', type=int, default=0,
            help = 'Azimuth order of azimuth offsets')
    parser.add_argument('--ar', dest='azrgorder', type=int, default=0,
            help = 'Range order of azimuth offsets')

    parser.add_argument('--ra', dest='rgazorder', type=int, default=0,
            help = 'Azimuth order of range offsets')
    parser.add_argument('--rr', dest='rgrgorder', type=int, default=0,
            help = 'Range order of range offsets')
    parser.add_argument('--ao', dest='azoff', type=int, default=0,
            help='Azimuth gross offset')
    parser.add_argument('--ro', dest='rgoff', type=int, default=0,
            help='Range gross offset')
    parser.add_argument('-t', '--thresh', dest='snrthresh', type=float, default=5.0,
            help='SNR threshold')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def estimateOffsetField(master, slave, azoffset=0, rgoffset=0):
    '''
    Estimate offset field between burst and simamp.
    '''


    sim = isceobj.createSlcImage()
    sim.load(slave+'.xml')
    sim.setAccessMode('READ')
    sim.createImage()

    sar = isceobj.createSlcImage()
    sar.load(master + '.xml')
    sar.setAccessMode('READ')
    sar.createImage()

    width = sar.getWidth()
    length = sar.getLength()

    objOffset = Ampcor(name='master_offset1')
    objOffset.configure()
    objOffset.setAcrossGrossOffset(rgoffset)
    objOffset.setDownGrossOffset(azoffset)
    objOffset.setWindowSizeWidth(128)
    objOffset.setWindowSizeHeight(128)
    objOffset.setSearchWindowSizeWidth(40)
    objOffset.setSearchWindowSizeHeight(40)
    margin = 2*objOffset.searchWindowSizeWidth + objOffset.windowSizeWidth

    nAcross = 60
    nDown = 60

   
    offAc = max(101,-rgoffset)+margin
    offDn = max(101,-azoffset)+margin

    
    lastAc = int( min(width, sim.getWidth() - offAc) - margin)
    lastDn = int( min(length, sim.getLength() - offDn) - margin)

#    print('Across: ', offAc, lastAc, width, sim.getWidth(), margin)
#    print('Down: ', offDn, lastDn, length, sim.getLength(), margin)

    if not objOffset.firstSampleAcross:
        objOffset.setFirstSampleAcross(offAc)

    if not objOffset.lastSampleAcross:
        objOffset.setLastSampleAcross(lastAc)

    if not objOffset.firstSampleDown:
        objOffset.setFirstSampleDown(offDn)

    if not objOffset.lastSampleDown:
        objOffset.setLastSampleDown(lastDn)

    if not objOffset.numberLocationAcross:
        objOffset.setNumberLocationAcross(nAcross)

    if not objOffset.numberLocationDown:
        objOffset.setNumberLocationDown(nDown)        

    objOffset.setFirstPRF(1.0)
    objOffset.setSecondPRF(1.0)
    objOffset.setImageDataType1('complex')
    objOffset.setImageDataType2('complex') 

    objOffset.ampcor(sar, sim)

    sar.finalizeImage()
    sim.finalizeImage()

    result = objOffset.getOffsetField()
    return result


def fitOffsets(field,azrgOrder=0,azazOrder=0,
        rgrgOrder=0,rgazOrder=0,snr=5.0):
    '''
    Estimate constant range and azimith shifs.
    '''


    stdWriter = create_writer("log","",True,filename='off.log')

    for distance in [10,5,3,1]:
        inpts = len(field._offsets)

        objOff = isceobj.createOffoutliers()
        objOff.wireInputPort(name='offsets', object=field)
        objOff.setSNRThreshold(snr)
        objOff.setDistance(distance)
        objOff.setStdWriter(stdWriter)

        objOff.offoutliers()

        field = objOff.getRefinedOffsetField()
        outputs = len(field._offsets)

        print('%d points left'%(len(field._offsets)))

    
    aa, dummy = field.getFitPolynomials(azimuthOrder=azazOrder, rangeOrder=azrgOrder, usenumpy=True)
    dummy, rr = field.getFitPolynomials(azimuthOrder=rgazOrder, rangeOrder=rgrgOrder, usenumpy=True)

    azshift = aa._coeffs[0][0]
    rgshift = rr._coeffs[0][0]
    print('Estimated az shift: ', azshift)
    print('Estimated rg shift: ', rgshift)

    return (aa, rr), field


def main(iargs=None):
    '''
    Generate offset fields burst by burst.
    '''

    inps = cmdLineParse(iargs)


    field = estimateOffsetField(inps.master, inps.slave,
            azoffset=inps.azoff, rgoffset=inps.rgoff)

    if os.path.exists(inps.outfile):
        os.remove(inps.outfile)

    outDir = os.path.dirname(inps.outfile)
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    if inps.metamaster is not None:
        masterShelveDir = os.path.join(outDir, 'masterShelve')
        if not os.path.exists(masterShelveDir):
            os.makedirs(masterShelveDir)

        cmd = 'cp ' + inps.metamaster + '/data* ' + masterShelveDir
        os.system(cmd)
        

    if inps.metaslave is not None:
        slaveShelveDir = os.path.join(outDir, 'slaveShelve')
        if not os.path.exists(slaveShelveDir):
            os.makedirs(slaveShelveDir)
        cmd = 'cp ' + inps.metaslave + '/data* ' + slaveShelveDir
        os.system(cmd)

    rgratio = 1.0
    azratio = 1.0

    if (inps.metamaster is not None) and (inps.metaslave is not None):
        
       # with shelve.open( os.path.join(inps.metamaster, 'data'), 'r') as db:
        with shelve.open( os.path.join(masterShelveDir, 'data'), 'r') as db:
            mframe = db['frame']

       # with shelve.open( os.path.join(inps.metaslave, 'data'), 'r') as db:
        with shelve.open( os.path.join(slaveShelveDir, 'data'), 'r') as db:
            sframe = db['frame']

        rgratio = mframe.instrument.getRangePixelSize()/sframe.instrument.getRangePixelSize()
        azratio = sframe.PRF / mframe.PRF

    print ('*************************************')
    print ('rgratio, azratio: ', rgratio, azratio)
    print ('*************************************')       

    odb = shelve.open(inps.outfile)
    odb['raw_field']  = field
    shifts, cull = fitOffsets(field,azazOrder=inps.azazorder,
            azrgOrder=inps.azrgorder,
            rgazOrder=inps.rgazorder,
            rgrgOrder=inps.rgrgorder,
            snr=inps.snrthresh)
    odb['cull_field'] = cull

    ####Scale by ratio
    for row in shifts[0]._coeffs:
        for ind, val in  enumerate(row):
            row[ind] = val * azratio

    for row in shifts[1]._coeffs:
        for ind, val in enumerate(row):
            row[ind] = val * rgratio
    

    odb['azpoly'] = shifts[0]
    odb['rgpoly'] = shifts[1]
    odb.close()

if __name__ == '__main__':
    main()



