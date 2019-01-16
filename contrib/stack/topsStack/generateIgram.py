#!/usr/bin/env python3

# Author: Piyush Agram
# Heresh Fattahi: Adopted for stack

import isce
import isceobj
import numpy as np
from isceobj.Util.Poly2D import Poly2D
import argparse
import os
import copy
from isceobj.Sensor.TOPS import createTOPSSwathSLCProduct 
from mroipac.correlation.correlation import Correlation
import s1a_isce_utils as ut
import gdal


def createParser():
    parser = argparse.ArgumentParser( description='Use polynomial offsets and create burst by burst interferograms')

    parser.add_argument('-m', '--master', dest='master', type=str, required=True,
            help='Directory with master acquisition')

    parser.add_argument('-s', '--slave', dest='slave', type=str, required=True,
            help='Directory with slave acquisition')

    parser.add_argument('-f', '--flatten', dest='flatten', action='store_true', default=False,
            help='Flatten the interferograms with offsets if needed')

    parser.add_argument('-i', '--interferogram', dest='interferogram', type=str, default='interferograms',
            help='Path for the interferogram')

    parser.add_argument('-p', '--interferogram_prefix', dest='intprefix', type=str, default='int',
            help='Prefix for the interferogram')
    parser.add_argument('-v', '--overlap', dest='overlap', action='store_true', default=False,
            help='Flatten the interferograms with offsets if needed')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def multiply(masname, slvname, outname, rngname1, rngname2, fact, masterFrame,
        flatten=False):

    print('multiply')
    masImg = isceobj.createSlcImage()
    masImg.load( masname + '.xml')

    width = masImg.getWidth()
    length = masImg.getLength()

    ds = gdal.Open(masname + '.vrt', gdal.GA_ReadOnly)
    master = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    ds = gdal.Open(slvname + '.vrt', gdal.GA_ReadOnly)
    slave = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    print('read') 
    #master = np.memmap(masname, dtype=np.complex64, mode='r', shape=(length,width))
    #slave = np.memmap(slvname, dtype=np.complex64, mode='r', shape=(length, width))

    if os.path.exists(rngname1):
        rng1 = np.memmap(rngname1, dtype=np.float32, mode='r', shape=(length,width))
    else:
        print('No range offsets provided')
        rng1 = np.zeros((length,width))

    if os.path.exists(rngname2):
        rng2 = np.memmap(rngname2, dtype=np.float32, mode='r', shape=(length,width))
    else:
        print('No range offsets provided')
        rng2 = np.zeros((length,width))

    rng12 = rng2 - rng1

    cJ = np.complex64(-1j)

    #Zero out anytging outside the valid region:
    ifg = np.memmap(outname, dtype=np.complex64, mode='w+', shape=(length,width))
    firstS = masterFrame.firstValidSample
    lastS = masterFrame.firstValidSample + masterFrame.numValidSamples -1
    firstL = masterFrame.firstValidLine
    lastL = masterFrame.firstValidLine + masterFrame.numValidLines - 1
    for kk in range(firstL,lastL + 1):
        ifg[kk,firstS:lastS + 1] = master[kk,firstS:lastS + 1] * np.conj(slave[kk,firstS:lastS + 1])
        if flatten:
            phs = np.exp(cJ*fact*rng12[kk,firstS:lastS + 1])
            ifg[kk,firstS:lastS + 1] *= phs


    ####
    master=None
    slave=None
    ifg = None

    objInt = isceobj.createIntImage()
    objInt.setFilename(outname)
    objInt.setWidth(width)
    objInt.setLength(length)
    objInt.setAccessMode('READ')
    #objInt.createImage()
    #objInt.finalizeImage()
    objInt.renderHdr()
    objInt.renderVRT()
    return objInt


def main(iargs=None):
    '''Create overlap interferograms.
    '''
    inps=cmdLineParse(iargs)

    if inps.overlap:
        masterSwathList = ut.getSwathList(os.path.join(inps.master, 'overlap'))
        slaveSwathList = ut.getSwathList(os.path.join(inps.slave, 'overlap'))
    else:
        masterSwathList = ut.getSwathList(inps.master)
        slaveSwathList = ut.getSwathList(inps.slave)
    swathList = list(sorted(set(masterSwathList+slaveSwathList)))

    for swath in swathList:
        IWstr = 'IW{0}'.format(swath)
        if inps.overlap:
            ifgdir = os.path.join(inps.interferogram, 'overlap', 'IW{0}'.format(swath))
        else:
            ifgdir = os.path.join(inps.interferogram, 'IW{0}'.format(swath))
            
        if not os.path.exists(ifgdir):
                os.makedirs(ifgdir)

    ####Load relevant products
        if inps.overlap:
            topMaster = ut.loadProduct(os.path.join(inps.master , 'overlap','IW{0}_top.xml'.format(swath)))
            botMaster = ut.loadProduct(os.path.join(inps.master ,'overlap', 'IW{0}_bottom.xml'.format(swath)))
            topCoreg = ut.loadProduct(os.path.join(inps.slave, 'overlap', 'IW{0}_top.xml'.format(swath)))
            botCoreg = ut.loadProduct(os.path.join(inps.slave, 'overlap', 'IW{0}_bottom.xml'.format(swath)))

        else:
            topMaster = ut.loadProduct(os.path.join(inps.master , 'IW{0}.xml'.format(swath)))
            topCoreg = ut.loadProduct(os.path.join(inps.slave , 'IW{0}.xml'.format(swath)))

        if inps.overlap:
            coregdir = os.path.join(inps.slave, 'overlap', 'IW{0}'.format(swath))
        else:
            coregdir = os.path.join(inps.slave,'IW{0}'.format(swath))
    
        topIfg = ut.coregSwathSLCProduct()
        topIfg.configure()

        if inps.overlap:
            botIfg = ut.coregSwathSLCProduct()
            botIfg.configure()

        minMaster = topMaster.bursts[0].burstNumber
        maxMaster = topMaster.bursts[-1].burstNumber

        minSlave = topCoreg.bursts[0].burstNumber
        maxSlave = topCoreg.bursts[-1].burstNumber

        minBurst = max(minSlave, minMaster)
        maxBurst = min(maxSlave, maxMaster)
        print ('minSlave,maxSlave',minSlave, maxSlave)
        print ('minMaster,maxMaster',minMaster, maxMaster)
        print ('minBurst, maxBurst: ', minBurst, maxBurst)

        for ii in range(minBurst, maxBurst + 1):

            ####Process the top bursts
            master = topMaster.bursts[ii-minMaster]
            slave  = topCoreg.bursts[ii-minSlave]

            print('matching burst numbers: ',master.burstNumber, slave.burstNumber)

            mastername = master.image.filename
            slavename = slave.image.filename

            if inps.overlap:
                rdict = { 'rangeOff1' : os.path.join(inps.master, 'overlap', IWstr, 'range_top_%02d_%02d.off'%(ii,ii+1)),
                     'rangeOff2' : os.path.join(inps.slave, 'overlap', IWstr, 'range_top_%02d_%02d.off'%(ii,ii+1)),
                     'azimuthOff': os.path.join(inps.slave, 'overlap', IWstr, 'azimuth_top_%02d_%02d.off'%(ii,ii+1))}

                intname = os.path.join(ifgdir, '%s_top_%02d_%02d.int'%(inps.intprefix,ii,ii+1))
        
            else:

                rdict = {'rangeOff1' : os.path.join(inps.master, IWstr, 'range_%02d.off'%(ii)),
                     'rangeOff2' : os.path.join(inps.slave, IWstr, 'range_%02d.off'%(ii)),
                     'azimuthOff1': os.path.join(inps.slave, IWstr, 'azimuth_%02d.off'%(ii))}
            
                intname = os.path.join(ifgdir, '%s_%02d.int'%(inps.intprefix,ii))


            ut.adjustCommonValidRegion(master,slave)
            fact = 4 * np.pi * slave.rangePixelSize / slave.radarWavelength
            intimage = multiply(mastername, slavename, intname,
                        rdict['rangeOff1'], rdict['rangeOff2'], fact, master, flatten=inps.flatten)

            burst = copy.deepcopy(master)
            burst.image = intimage
            burst.burstNumber = ii
            topIfg.bursts.append(burst)


            if inps.overlap:
                ####Process the bottom bursts
                master = botMaster.bursts[ii-minMaster]
                slave = botCoreg.bursts[ii-minSlave]


                mastername =  master.image.filename
                slavename = slave.image.filename
#            rdict = {'rangeOff' : os.path.join(coregdir, 'range_bot_%02d_%02d.off'%(ii,ii+1)),
#                   'azimuthOff': os.path.join(coregdir, 'azimuth_bot_%02d_%02d.off'%(ii,ii+1))}

                rdict = { 'rangeOff1' : os.path.join(inps.master, 'overlap', IWstr, 'range_bot_%02d_%02d.off'%(ii,ii+1)),
                     'rangeOff2' : os.path.join(inps.slave, 'overlap', IWstr, 'range_bot_%02d_%02d.off'%(ii,ii+1)),
                    'azimuthOff': os.path.join(inps.slave, 'overlap', IWstr, 'azimuth_bot_%02d_%02d.off'%(ii,ii+1))}


                print ('rdict: ', rdict)

                ut.adjustCommonValidRegion(master,slave)
                intname = os.path.join(ifgdir, '%s_bot_%02d_%02d.int'%(inps.intprefix,ii,ii+1))
                fact = 4 * np.pi * slave.rangePixelSize / slave.radarWavelength

            #intimage = multiply(mastername, slavename, intname,
            #        rdict['rangeOff'], fact, master, flatten=True)

                intimage = multiply(mastername, slavename, intname,
                        rdict['rangeOff1'], rdict['rangeOff2'], fact, master, flatten=inps.flatten)

                burst = copy.deepcopy(master)
                burst.burstNumber = ii
                burst.image = intimage
                botIfg.bursts.append(burst)


        topIfg.numberOfBursts = len(topIfg.bursts)
        if hasattr(topCoreg, 'reference'):
            topIfg.reference = topCoreg.reference
        else:
            topIfg.reference = topMaster.reference

        print('Type: ',type(topIfg.reference))

        if inps.overlap:
            ut.saveProduct(topIfg, ifgdir +  '_top.xml')
            botIfg.numberOfBursts = len(botIfg.bursts)
            botIfg.reference = botCoreg.reference
            print(botIfg.reference)
            ut.saveProduct(botIfg, ifgdir + '_bottom.xml')
        else:
            ut.saveProduct(topIfg, ifgdir + '.xml')    
        


if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()

