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
    parser.add_argument('-m', '--reference_dir', type=str, dest='reference', required=True,
            help='Directory with the secondary image')
    parser.add_argument('-s', '--secondary_dir', type=str, dest='secondary', required=True,
            help='Directory with the secondary image')
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



def overlapSpectralSeparation(topBurstIfg, botBurstIfg, referenceTop, referenceBot, secondaryTop, secondaryBot, azMasTop, rgMasTop, azMasBot, rgMasBot, azSlvTop, rgSlvTop, azSlvBot, rgSlvBot , misreg=0.0):
    # Added by Heresh Fattahi
    '''
    Estimate separation in frequency due to unit pixel misregistration.
    '''
    ''' 
    dt = topBurstIfg.azimuthTimeInterval
    topStart = int(np.round((topBurstIfg.sensingStart - referenceTop.sensingStart).total_seconds() / dt))
    overlapLen = topBurstIfg.numberOfLines
    botStart = int(np.round((botBurstIfg.sensingStart - referenceBot.sensingStart).total_seconds() / dt))

    print(topBurstIfg.sensingStart, referenceTop.sensingStart)
    print(botBurstIfg.sensingStart, referenceBot.sensingStart)
    print(topStart, botStart, overlapLen)
    '''
    print ('++++++++++++++++++++++')
    dt = topBurstIfg.azimuthTimeInterval
    topStart = int ( np.round( (referenceBot.sensingStart - referenceTop.sensingStart).total_seconds()/dt))+ referenceBot.firstValidLine
    overlapLen = topBurstIfg.numberOfLines
    botStart = referenceBot.firstValidLine
    print(topStart, botStart, overlapLen)
    #print(Debug)
    
    ##############
    # reference top : m1



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

    Vs = np.linalg.norm(referenceTop.orbit.interpolateOrbit(referenceTop.sensingMid, method='hermite').getVelocity())
    Ks =   2 * Vs * referenceTop.azimuthSteeringRate / referenceTop.radarWavelength
    rng = referenceTop.startingRange + referenceTop.rangePixelSize * rng
    Ka = referenceTop.azimuthFMRate(rng)

    Ktm1 = Ks / (1.0 - Ks / Ka)
    tm1 = (azi - (referenceTop.numberOfLines//2)) * referenceTop.azimuthTimeInterval

    fm1 = referenceTop.doppler(rng)

    ##############
    # reference bottom : m2
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

    Vs = np.linalg.norm(referenceBot.orbit.interpolateOrbit(referenceBot.sensingMid, method='hermite').getVelocity())
    Ks =   2 * Vs * referenceBot.azimuthSteeringRate / referenceBot.radarWavelength
    rng = referenceBot.startingRange + referenceBot.rangePixelSize * rng
    Ka = referenceBot.azimuthFMRate(rng)

    Ktm2 = Ks / (1.0 - Ks / Ka)
    tm2 = (azi - (referenceBot.numberOfLines//2)) * referenceBot.azimuthTimeInterval
    fm2 = referenceBot.doppler(rng)


    ##############
    # secondary top : s1
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

    Vs = np.linalg.norm(secondaryTop.orbit.interpolateOrbit(secondaryTop.sensingMid, method='hermite').getVelocity())
    Ks =   2 * Vs * secondaryTop.azimuthSteeringRate / secondaryTop.radarWavelength
    rng = secondaryTop.startingRange + secondaryTop.rangePixelSize * rng
    Ka = secondaryTop.azimuthFMRate(rng)

    Kts1 = Ks / (1.0 - Ks / Ka)
    ts1 = (azi - (secondaryTop.numberOfLines//2)) * secondaryTop.azimuthTimeInterval
    fs1 = secondaryTop.doppler(rng)



    ##############
    # secondary bot : s2
    y = np.arange(botStart, botStart + overlapLen)[:,None] * np.ones((overlapLen, botBurstIfg.numberOfSamples))
    x = np.ones((overlapLen, botBurstIfg.numberOfSamples)) * np.arange(botBurstIfg.numberOfSamples)[None,:]

    ####Bottom secondary
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

    Vs = np.linalg.norm(secondaryBot.orbit.interpolateOrbit(secondaryBot.sensingMid, method='hermite').getVelocity())
    Ks =   2 * Vs * secondaryBot.azimuthSteeringRate / secondaryBot.radarWavelength
    rng = secondaryBot.startingRange + secondaryBot.rangePixelSize * rng
    Ka = secondaryBot.azimuthFMRate(rng)
    Kts2 = Ks / (1.0 - Ks / Ka)

    ts2 = (azi - (secondaryBot.numberOfLines//2)) * secondaryBot.azimuthTimeInterval
    fs2 = secondaryBot.doppler(rng)

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
    inps.reference = os.path.join(inps.reference,'overlap')
    inps.secondary = os.path.join(inps.secondary,'overlap')

    referenceSwathList = ut.getSwathList(inps.reference)
    secondarySwathList = ut.getSwathList(inps.secondary)

    swathList = list(sorted(set(referenceSwathList+secondarySwathList)))

    for swath in swathList:
        IWstr = 'IW{0}'.format(swath)
        referenceTop = ut.loadProduct(os.path.join(inps.reference, IWstr + '_top.xml'))
        referenceBot = ut.loadProduct(os.path.join(inps.reference , IWstr + '_bottom.xml'))
    
        secondaryTop = ut.loadProduct(os.path.join(inps.secondary, IWstr + '_top.xml'))
        secondaryBot = ut.loadProduct(os.path.join(inps.secondary, IWstr + '_bottom.xml'))    


        ####Load metadata for burst IFGs
        ifgTop = ut.loadProduct(os.path.join(inps.interferogram , IWstr + '_top.xml'))
        ifgBottom = ut.loadProduct(os.path.join(inps.interferogram, IWstr + '_bottom.xml'))

        ####Create ESD output directory
        esddir = os.path.join(inps.overlap, IWstr)
        os.makedirs(esddir, exist_ok=True)

        ####Overlap offsets directory
        referenceOffdir = os.path.join(inps.reference, IWstr)
        secondaryOffdir = os.path.join(inps.secondary,IWstr)

        #########
        minReference = referenceTop.bursts[0].burstNumber
        maxReference = referenceTop.bursts[-1].burstNumber

        minSecondary = secondaryTop.bursts[0].burstNumber
        maxSecondary = secondaryTop.bursts[-1].burstNumber

        minBurst = ifgTop.bursts[0].burstNumber
        maxBurst = ifgTop.bursts[-1].burstNumber
        print ('minSecondary,maxSecondary',minSecondary, maxSecondary)
        print ('minReference,maxReference',minReference, maxReference)
        print ('minBurst, maxBurst: ', minBurst, maxBurst)

    #########


        ifglist = []
        factorlist = []
        offsetlist = []
        cohlist = []

        for ii in range(minBurst, maxBurst + 1):
            ind = ii - minBurst            ###Index into overlaps
            mind = ii - minReference  ### Index into reference
            sind = ii - minSecondary   ###Index into secondary

            topBurstIfg = ifgTop.bursts[ind]
            botBurstIfg = ifgBottom.bursts[ind]

            ###############
            '''stackReferenceTop = ifgTop.source.bursts[mind]
            stackReferenceBot = ifgBottom.source.bursts[mind]

            dt = stackReferenceTop.azimuthTimeInterval
            topStart = int(np.round((stackReferenceBot.sensingStart - stackReferenceTop.sensingStart).total_seconds() / dt))
            #overlapLen = .numberOfLines
            botStart = stackReferenceBot.firstValidLine #int(np.round((.sensingStart - referenceBot.sensingStart).total_seconds() / dt))
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
            azMasTop = os.path.join(referenceOffdir, 'azimuth_top_%02d_%02d.off'%(ii,ii+1))
            rgMasTop = os.path.join(referenceOffdir, 'range_top_%02d_%02d.off'%(ii,ii+1))
            azMasBot = os.path.join(referenceOffdir, 'azimuth_bot_%02d_%02d.off'%(ii,ii+1))
            rgMasBot = os.path.join(referenceOffdir, 'range_bot_%02d_%02d.off'%(ii,ii+1))

            azSlvTop = os.path.join(secondaryOffdir, 'azimuth_top_%02d_%02d.off'%(ii,ii+1))
            rgSlvTop = os.path.join(secondaryOffdir, 'range_top_%02d_%02d.off'%(ii,ii+1))
            azSlvBot = os.path.join(secondaryOffdir, 'azimuth_bot_%02d_%02d.off'%(ii,ii+1))
            rgSlvBot = os.path.join(secondaryOffdir, 'range_bot_%02d_%02d.off'%(ii,ii+1))

            mFullTop = referenceTop.source.bursts[mind]
            mFullBot = referenceBot.source.bursts[mind+1]
            sFullTop = secondaryTop.source.bursts[sind]
            sFullBot = secondaryBot.source.bursts[sind+1]

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



