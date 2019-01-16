#!/usr/bin/env python3

import numpy as np
import argparse
import os
import isce
import isceobj
import copy
import logging
import scipy.signal as SS
from isceobj.Util.ImageUtil import ImageLib as IML
import s1a_isce_utils as ut


def createParser():
    parser = argparse.ArgumentParser( description='Generate offset field between two Sentinel swaths')

    parser.add_argument('-i', '--interferogram', type=str, dest='interferogram',required=True,
            help='Directory with the overlap interferogram')
    parser.add_argument('-m', '--master_dir', type=str, dest='master', required=True,
            help='Directory with the slave image')
    parser.add_argument('-s', '--slave_dir', type=str, dest='slave', required=True,
            help='Directory with the slave image')
    parser.add_argument('-d', '--overlap_dir', type=str, dest='overlap', required=True,
            help='Directory with overlap products')

    parser.add_argument('-a', '--esdAzimuthLooks', type=int, dest='esdAzimuthLooks', default = 5,
            help='ESD azimuth looks')
    
    parser.add_argument('-r', '--esdRangeLooks', type=int, dest='esdRangeLooks', default = 15,
            help='ESD range looks')
    return parser



def cmdLineParse(iargs=None):
    '''
    Command line parser.
    '''
    parser = createParser()
    return parser.parse_args(args=iargs)



def multilook(intName, alks=5, rlks=15):
    cmd = 'looks.py -i {0} -a {1} -r {2}'.format(intName,alks,rlks)
    flag = os.system(cmd)

    if flag:
        raise Exception('Failed to multilook %s'%(intName))

    spl = os.path.splitext(intName)
    return '{0}.{1}alks_{2}rlks{3}'.format(spl[0],alks,rlks,spl[1])



def overlapSpectralSeparation(topBurstIfg, botBurstIfg, masterTop, masterBot, slaveTop, slaveBot, azMasTop, rgMasTop, azMasBot, rgMasBot, azSlvTop, rgSlvTop, azSlvBot, rgSlvBot , misreg=0.0):
    # Added by Heresh Fattahi
    '''
    Estimate separation in frequency due to unit pixel misregistration.
    '''
    ''' 
    dt = topBurstIfg.azimuthTimeInterval
    topStart = int(np.round((topBurstIfg.sensingStart - masterTop.sensingStart).total_seconds() / dt))
    overlapLen = topBurstIfg.numberOfLines
    botStart = int(np.round((botBurstIfg.sensingStart - masterBot.sensingStart).total_seconds() / dt))

    print(topBurstIfg.sensingStart, masterTop.sensingStart)
    print(botBurstIfg.sensingStart, masterBot.sensingStart)
    print(topStart, botStart, overlapLen)
    '''
    print ('++++++++++++++++++++++')
    dt = topBurstIfg.azimuthTimeInterval
    topStart = int ( np.round( (masterBot.sensingStart - masterTop.sensingStart).total_seconds()/dt))+ masterBot.firstValidLine
    overlapLen = topBurstIfg.numberOfLines
    botStart = masterBot.firstValidLine
    print(topStart, botStart, overlapLen)
    #print(Debug)
    
    ##############
    # master top : m1



    y = np.arange(topStart, topStart+overlapLen)[:,None] * np.ones((overlapLen, topBurstIfg.numberOfSamples))
    x = np.ones((overlapLen, topBurstIfg.numberOfSamples)) * np.arange(topBurstIfg.numberOfSamples)[None,:]

    if os.path.exists(azMasTop) and os.path.exists(rgMasTop):
          yy = np.memmap( azMasTop, dtype=np.float32, mode='r',
                 shape=(topBurstIfg.numberOfLines, topBurstIfg.numberOfSamples))
          xx = np.memmap( rgMasTop, dtype=np.float32, mode='r',
                 shape=(topBurstIfg.numberOfLines, topBurstIfg.numberOfSamples))
    else:
          yy = 0.0
          xx = 0.0


    azi = y + yy
    rng = x + xx

    Vs = np.linalg.norm(masterTop.orbit.interpolateOrbit(masterTop.sensingMid, method='hermite').getVelocity())
    Ks =   2 * Vs * masterTop.azimuthSteeringRate / masterTop.radarWavelength
    rng = masterTop.startingRange + masterTop.rangePixelSize * rng
    Ka = masterTop.azimuthFMRate(rng)

    Ktm1 = Ks / (1.0 - Ks / Ka)
    tm1 = (azi - (masterTop.numberOfLines//2)) * masterTop.azimuthTimeInterval

    fm1 = masterTop.doppler(rng)

    ##############
    # master bottom : m2
    y = np.arange(botStart, botStart + overlapLen)[:,None] * np.ones((overlapLen, botBurstIfg.numberOfSamples))
    x = np.ones((overlapLen, botBurstIfg.numberOfSamples)) * np.arange(botBurstIfg.numberOfSamples)[None,:]

    if os.path.exists(azMasBot) and os.path.exists(rgMasBot):
          yy = np.memmap( azMasBot, dtype=np.float32, mode='r',
                shape=(botBurstIfg.numberOfLines, botBurstIfg.numberOfSamples))
          xx = np.memmap( rgMasBot, dtype=np.float32, mode='r',
                shape=(botBurstIfg.numberOfLines, botBurstIfg.numberOfSamples))
    else:
          yy = 0.0
          xx = 0.0

    azi = y + yy
    rng = x + xx

    Vs = np.linalg.norm(masterBot.orbit.interpolateOrbit(masterBot.sensingMid, method='hermite').getVelocity())
    Ks =   2 * Vs * masterBot.azimuthSteeringRate / masterBot.radarWavelength
    rng = masterBot.startingRange + masterBot.rangePixelSize * rng
    Ka = masterBot.azimuthFMRate(rng)

    Ktm2 = Ks / (1.0 - Ks / Ka)
    tm2 = (azi - (masterBot.numberOfLines//2)) * masterBot.azimuthTimeInterval
    fm2 = masterBot.doppler(rng)


    ##############
    # slave top : s1
    y = np.arange(topStart, topStart+overlapLen)[:,None] * np.ones((overlapLen, topBurstIfg.numberOfSamples))
    x = np.ones((overlapLen, topBurstIfg.numberOfSamples)) * np.arange(topBurstIfg.numberOfSamples)[None,:]

    if os.path.exists(azSlvTop) and os.path.exists(rgSlvTop):
          yy = np.memmap( azSlvTop, dtype=np.float32, mode='r',
                 shape=(topBurstIfg.numberOfLines, topBurstIfg.numberOfSamples))
          xx = np.memmap( rgSlvTop, dtype=np.float32, mode='r',
                 shape=(topBurstIfg.numberOfLines, topBurstIfg.numberOfSamples))
    else:
          yy = 0.0
          xx = 0.0
    

    azi = y + yy + misreg
    rng = x + xx

#    print('Azi top: ', azi[0,0], azi[-1,-1])
#    print('YY  top: ', yy[0,0], yy[-1,-1])
#    print('Rng top: ', rng[0,0], azi[-1,-1])
#    print('XX  top: ', xx[0,0], xx[-1,-1])

    Vs = np.linalg.norm(slaveTop.orbit.interpolateOrbit(slaveTop.sensingMid, method='hermite').getVelocity())
    Ks =   2 * Vs * slaveTop.azimuthSteeringRate / slaveTop.radarWavelength
    rng = slaveTop.startingRange + slaveTop.rangePixelSize * rng
    Ka = slaveTop.azimuthFMRate(rng)

    Kts1 = Ks / (1.0 - Ks / Ka)
    ts1 = (azi - (slaveTop.numberOfLines//2)) * slaveTop.azimuthTimeInterval
    fs1 = slaveTop.doppler(rng)



    ##############
    # slave bot : s2
    y = np.arange(botStart, botStart + overlapLen)[:,None] * np.ones((overlapLen, botBurstIfg.numberOfSamples))
    x = np.ones((overlapLen, botBurstIfg.numberOfSamples)) * np.arange(botBurstIfg.numberOfSamples)[None,:]

    ####Bottom slave
    if os.path.exists(azSlvBot) and os.path.exists(rgSlvBot):
          yy = np.memmap( azSlvBot, dtype=np.float32, mode='r',
                shape=(botBurstIfg.numberOfLines, botBurstIfg.numberOfSamples))
          xx = np.memmap( rgSlvBot, dtype=np.float32, mode='r',
                shape=(botBurstIfg.numberOfLines, botBurstIfg.numberOfSamples))
    else:
          yy = 0.0
          xx = 0.0

    azi = y + yy + misreg
    rng = x + xx

#    print('Azi bot: ', azi[0,0], azi[-1,-1])
#    print('YY  bot: ', yy[0,0], yy[-1,-1])
#    print('Rng bot: ', rng[0,0], azi[-1,-1])
#    print('XX  bot: ', xx[0,0], xx[-1,-1])

    Vs = np.linalg.norm(slaveBot.orbit.interpolateOrbit(slaveBot.sensingMid, method='hermite').getVelocity())
    Ks =   2 * Vs * slaveBot.azimuthSteeringRate / slaveBot.radarWavelength
    rng = slaveBot.startingRange + slaveBot.rangePixelSize * rng
    Ka = slaveBot.azimuthFMRate(rng)
    Kts2 = Ks / (1.0 - Ks / Ka)

    ts2 = (azi - (slaveBot.numberOfLines//2)) * slaveBot.azimuthTimeInterval
    fs2 = slaveBot.doppler(rng)

    ##############
    frequencySeparation =  -Ktm2*tm2 + Ktm1*tm1  + Kts1*ts1 - Kts2*ts2 +  fm2 - fm1 + fs1 -fs2
    #print(frequencySeparation)
    #print(tm2)
    #print(tm1)
    #print('*********')
    #print(ts1)
    #print(ts2)
    #print(Debug)
    return frequencySeparation


def createCoherence(intfile, win=5):
    '''
    Compute coherence using scipy convolve 2D.
    '''

    corfile = os.path.splitext(intfile)[0] + '.cor'
    filt = np.ones((win,win))/ (1.0*win*win)

    inimg = IML.mmapFromISCE(intfile + '.xml', logging)
    cJ = np.complex64(1.0j)
    angle = np.exp(cJ * np.angle(inimg.bands[0]))

    res = SS.convolve2d(angle, filt, mode='same')
    res[0:win-1,:] = 0.0
    res[-win+1:,:] = 0.0
    res[:,0:win-1] = 0.0
    res[:,-win+1:] = 0.0

    res = np.abs(res)

    with open(corfile, 'wb') as f:
        res.astype(np.float32).tofile(f)

    img = isceobj.createImage()
    img.setFilename(corfile)
    img.setWidth(res.shape[1])
    img.dataType='FLOAT'
    img.setAccessMode('READ')
    img.renderHdr()
    img.renderVRT()
  #  img.createImage()
  #  img.finalizeImage()

    return corfile

def main(iargs=None):
    '''
    Create additional layers for performing ESD.
    '''

    inps = cmdLineParse(iargs)
    inps.interferogram = os.path.join(inps.interferogram,'overlap')
    inps.master = os.path.join(inps.master,'overlap')
    inps.slave = os.path.join(inps.slave,'overlap')

    masterSwathList = ut.getSwathList(inps.master)
    slaveSwathList = ut.getSwathList(inps.slave)

    swathList = list(sorted(set(masterSwathList+slaveSwathList)))

    for swath in swathList:
        IWstr = 'IW{0}'.format(swath)
        masterTop = ut.loadProduct(os.path.join(inps.master, IWstr + '_top.xml'))
        masterBot = ut.loadProduct(os.path.join(inps.master , IWstr + '_bottom.xml'))
    
        slaveTop = ut.loadProduct(os.path.join(inps.slave, IWstr + '_top.xml'))
        slaveBot = ut.loadProduct(os.path.join(inps.slave, IWstr + '_bottom.xml'))    


        ####Load metadata for burst IFGs
        ifgTop = ut.loadProduct(os.path.join(inps.interferogram , IWstr + '_top.xml'))
        ifgBottom = ut.loadProduct(os.path.join(inps.interferogram, IWstr + '_bottom.xml'))

        ####Create ESD output directory
        esddir = os.path.join(inps.overlap, IWstr)
        if not os.path.isdir(esddir):
            os.makedirs(esddir)

        ####Overlap offsets directory
        masterOffdir = os.path.join(inps.master, IWstr)
        slaveOffdir = os.path.join(inps.slave,IWstr)

        #########
        minMaster = masterTop.bursts[0].burstNumber
        maxMaster = masterTop.bursts[-1].burstNumber

        minSlave = slaveTop.bursts[0].burstNumber
        maxSlave = slaveTop.bursts[-1].burstNumber

        minBurst = ifgTop.bursts[0].burstNumber
        maxBurst = ifgTop.bursts[-1].burstNumber
        print ('minSlave,maxSlave',minSlave, maxSlave)
        print ('minMaster,maxMaster',minMaster, maxMaster)
        print ('minBurst, maxBurst: ', minBurst, maxBurst)

    #########


        ifglist = []
        factorlist = []
        offsetlist = []
        cohlist = []

        for ii in range(minBurst, maxBurst + 1):
            ind = ii - minBurst            ###Index into overlaps
            mind = ii - minMaster  ### Index into master
            sind = ii - minSlave   ###Index into slave

            topBurstIfg = ifgTop.bursts[ind]
            botBurstIfg = ifgBottom.bursts[ind]

            ###############
            '''stackMasterTop = ifgTop.source.bursts[mind]
            stackMasterBot = ifgBottom.source.bursts[mind]

            dt = stackMasterTop.azimuthTimeInterval
            topStart = int(np.round((stackMasterBot.sensingStart - stackMasterTop.sensingStart).total_seconds() / dt))
            #overlapLen = .numberOfLines
            botStart = stackMasterBot.firstValidLine #int(np.round((.sensingStart - masterBot.sensingStart).total_seconds() / dt))
            print('+++++++++++++++++++')
            print(topStart, botStart)
            print('+++++++++++++++++++') '''
            ###############

            ####Double difference interferograms
            topInt = np.memmap( topBurstIfg.image.filename,
                    dtype=np.complex64, mode='r',
                    shape = (topBurstIfg.numberOfLines, topBurstIfg.numberOfSamples))

            botInt = np.memmap( botBurstIfg.image.filename,
                    dtype=np.complex64, mode='r',
                    shape = (botBurstIfg.numberOfLines, botBurstIfg.numberOfSamples))

            intName = os.path.join(esddir, 'overlap_%02d.int'%(ii))
            freqName = os.path.join(esddir, 'freq_%02d.bin'%(ii))

            with open(intName, 'wb') as fid:
                fid.write( topInt * np.conj(botInt))

            img = isceobj.createIntImage()
            img.setFilename(intName)
            img.setWidth(topBurstIfg.numberOfSamples)
            img.setLength(topBurstIfg.numberOfLines)
            img.setAccessMode('READ')
            img.renderHdr()
            img.renderVRT()
            img.createImage()
            img.finalizeImage()

            multIntName= multilook(intName, alks = inps.esdAzimuthLooks, rlks=inps.esdRangeLooks)
            ifglist.append(multIntName)


            ####Estimate coherence of double different interferograms
            multCor = createCoherence(multIntName)
            cohlist.append(multCor)

            ####Estimate the frequency difference 
            azMasTop = os.path.join(masterOffdir, 'azimuth_top_%02d_%02d.off'%(ii,ii+1))
            rgMasTop = os.path.join(masterOffdir, 'range_top_%02d_%02d.off'%(ii,ii+1))
            azMasBot = os.path.join(masterOffdir, 'azimuth_bot_%02d_%02d.off'%(ii,ii+1))
            rgMasBot = os.path.join(masterOffdir, 'range_bot_%02d_%02d.off'%(ii,ii+1))

            azSlvTop = os.path.join(slaveOffdir, 'azimuth_top_%02d_%02d.off'%(ii,ii+1))
            rgSlvTop = os.path.join(slaveOffdir, 'range_top_%02d_%02d.off'%(ii,ii+1))
            azSlvBot = os.path.join(slaveOffdir, 'azimuth_bot_%02d_%02d.off'%(ii,ii+1))
            rgSlvBot = os.path.join(slaveOffdir, 'range_bot_%02d_%02d.off'%(ii,ii+1))

            mFullTop = masterTop.source.bursts[mind]
            mFullBot = masterBot.source.bursts[mind+1]
            sFullTop = slaveTop.source.bursts[sind]
            sFullBot = slaveBot.source.bursts[sind+1]

            freqdiff = overlapSpectralSeparation(topBurstIfg, botBurstIfg, mFullTop, mFullBot, sFullTop, sFullBot, 
              azMasTop, rgMasTop, azMasBot, rgMasBot, azSlvTop, rgSlvTop, azSlvBot, rgSlvBot)

            with open(freqName, 'wb') as fid:
                (freqdiff * 2 * np.pi * mFullTop.azimuthTimeInterval).astype(np.float32).tofile(fid)

            img = isceobj.createImage()
            img.setFilename(freqName)
            img.setWidth(topBurstIfg.numberOfSamples)
            img.setLength(topBurstIfg.numberOfLines)
            img.setAccessMode('READ')
            img.bands = 1
            img.dataType = 'FLOAT'
           # img.createImage()
            img.renderHdr()
            img.renderVRT()
            img.createImage()
            img.finalizeImage()

            multConstName = multilook(freqName, alks = inps.esdAzimuthLooks, rlks = inps.esdRangeLooks)
            factorlist.append(multConstName)

if __name__ == '__main__':
    '''
    Main driver.
    '''
    # Main Driver
    main()



