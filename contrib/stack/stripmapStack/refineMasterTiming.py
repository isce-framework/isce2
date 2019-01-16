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


def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser( description='Generate offset field between two Sentinel swaths')
    parser.add_argument('-m', type=str, dest='master', required=True,
            help='Directory with the master image')
    parser.add_argument('-g', type=str, dest='geom', default=None,
            help='Directory with geometry products. If not provided: geometry_master')
    parser.add_argument('-o', type=str, default='mastershift.json', dest='outfile',
            help='Misregistration in subpixels')

    inps = parser.parse_args()

    if inps.master.endswith('/'):
        inps.master = inps.master[:-1]

    if inps.geom is None:
        inps.geom = 'geometry_' + os.path.basename(inps.master)

    return inps


def estimateOffsetField(burst, simfile,offset=0.0):
    '''
    Estimate offset field between burst and simamp.
    '''


    sim = isceobj.createImage()
    sim.load(simfile+'.xml')
    sim.setAccessMode('READ')
    sim.createImage()

    sar = isceobj.createSlcImage()
    sar.load(burst.getImage().filename + '.xml')
    sar.setAccessMode('READ')
    sar.createImage()

    width = sar.getWidth()
    length = sar.getLength()

    objOffset = Ampcor(name='master_offset')
    objOffset.configure()
    objOffset.setWindowSizeWidth(128)
    objOffset.setWindowSizeHeight(128)
    objOffset.setSearchWindowSizeWidth(16)
    objOffset.setSearchWindowSizeHeight(16)
    margin = 2*objOffset.searchWindowSizeWidth + objOffset.windowSizeWidth

    nAcross = 40
    nDown = 40

    if not objOffset.firstSampleAcross:
        objOffset.setFirstSampleAcross(margin+101)

    if not objOffset.lastSampleAcross:
        objOffset.setLastSampleAcross(width-margin-101)

    if not objOffset.firstSampleDown:
        objOffset.setFirstSampleDown(margin+offset+101)

    if not objOffset.lastSampleDown:
        objOffset.setLastSampleDown(length - margin-101)

    if not objOffset.acrossGrossOffset:
        objOffset.setAcrossGrossOffset(0.0)

    if not objOffset.downGrossOffset:
        objOffset.setDownGrossOffset(offset)

    if not objOffset.numberLocationAcross:
        objOffset.setNumberLocationAcross(nAcross)

    if not objOffset.numberLocationDown:
        objOffset.setNumberLocationDown(nDown)        

    objOffset.setFirstPRF(1.0)
    objOffset.setSecondPRF(1.0)
    objOffset.setImageDataType1('complex')
    objOffset.setImageDataType2('real') 

    objOffset.ampcor(sar, sim)

    sar.finalizeImage()
    sim.finalizeImage()

    result = objOffset.getOffsetField()
    return result


def fitOffsets(field):
    '''
    Estimate constant range and azimith shifs.
    '''


    stdWriter = create_writer("log","",True,filename='off.log')

    for distance in [10,5,3]:
        inpts = len(field._offsets)

        objOff = isceobj.createOffoutliers()
        objOff.wireInputPort(name='offsets', object=field)
        objOff.setSNRThreshold(2.0)
        objOff.setDistance(distance)
        objOff.setStdWriter(stdWriter)

        objOff.offoutliers()

        field = objOff.getRefinedOffsetField()
        outputs = len(field._offsets)

        print('%d points left'%(len(field._offsets)))

            
        wt = np.array([x.snr for x in field])
        dx = np.array([x.dx for x in field])
        dy = np.array([y.dy for y in field])

        azshift = np.dot(wt,dy) / np.sum(wt)
        rgshift = np.dot(wt,dx) / np.sum(wt)

        print('Estimated az shift: ', azshift)
        print('Estimated rg shift: ', rgshift)

    return (azshift, rgshift), field

        
if __name__ == '__main__':
    '''
    Generate offset fields burst by burst.
    '''

    inps = cmdLineParse()

    db = shelve.open( os.path.join(inps.master, 'data'), flag='r')

    frame = db['frame']

    outfile = os.path.join(inps.geom, 'simamp.rdr')
    infile = os.path.join(inps.geom, 'z.rdr')

#    runSimamp(infile, outfile)

    field = estimateOffsetField(frame, outfile)

    odb = shelve.open('masterOffset')
    odb['raw_field']  = field

    shifts, cull = fitOffsets(field)
    odb['cull_field'] = cull

    db.close()
    odb.close()
