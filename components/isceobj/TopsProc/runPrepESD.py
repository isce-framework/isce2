#
# Author: Piyush Agram
# Copyright 2016
#


import numpy as np 
import os
import isceobj
import logging
from isceobj.Util.ImageUtil import ImageLib as IML
import datetime
import pprint
from .runFineResamp import getRelativeShifts

def multilook(intname, alks=5, rlks=15):
    '''
    Take looks.
    '''
    from mroipac.looks.Looks import Looks

    inimg = isceobj.createImage()
    inimg.load(intname + '.xml')


    spl = os.path.splitext(intname)
    ext = '.{0}alks_{1}rlks'.format(alks, rlks)
    outFile = spl[0] + ext + spl[1]


    lkObj = Looks()
    lkObj.setDownLooks(alks)
    lkObj.setAcrossLooks(rlks)
    lkObj.setInputImage(inimg)
    lkObj.setOutputFilename(outFile)
    lkObj.looks()

    print('Output: ', outFile)
    return outFile


def multilook_old(intName, alks=5, rlks=15):
    cmd = 'looks.py -i {0} -a {1} -r {2}'.format(intName,alks,rlks)
    flag = os.system(cmd)

    if flag:
        raise Exception('Failed to multilook %s'%(intName))

    spl = os.path.splitext(intName)
    return '{0}.{1}alks_{2}rlks{3}'.format(spl[0],alks,rlks,spl[1])



def overlapSpectralSeparation(topBurstIfg, botBurstIfg, masterTop, masterBot, slaveTop, slaveBot, azTop, rgTop, azBot, rgBot, misreg=0.0):    
    '''
    Estimate separation in frequency due to unit pixel misregistration.
    '''


    dt = topBurstIfg.azimuthTimeInterval
    topStart = int(np.round((topBurstIfg.sensingStart - masterTop.sensingStart).total_seconds() / dt))
    overlapLen = topBurstIfg.numberOfLines
    botStart = int(np.round((botBurstIfg.sensingStart - masterBot.sensingStart).total_seconds() / dt))
    

    ##############
    # master top : m1

    azi = np.arange(topStart, topStart+overlapLen)[:,None] * np.ones((overlapLen, topBurstIfg.numberOfSamples))
    rng = np.ones((overlapLen, topBurstIfg.numberOfSamples)) * np.arange(topBurstIfg.numberOfSamples)[None,:]

    Vs = np.linalg.norm(masterTop.orbit.interpolateOrbit(masterTop.sensingMid, method='hermite').getVelocity())
    Ks =   2 * Vs * masterTop.azimuthSteeringRate / masterTop.radarWavelength
    rng = masterTop.startingRange + masterTop.rangePixelSize * rng
    Ka = masterTop.azimuthFMRate(rng)

    Ktm1 = Ks / (1.0 - Ks / Ka)
    tm1 = (azi - (masterTop.numberOfLines//2)) * masterTop.azimuthTimeInterval

    fm1 = masterTop.doppler(rng)
    
    ##############
    # master bottom : m2
    azi = np.arange(botStart, botStart + overlapLen)[:,None] * np.ones((overlapLen, botBurstIfg.numberOfSamples))
    rng = np.ones((overlapLen, botBurstIfg.numberOfSamples)) * np.arange(botBurstIfg.numberOfSamples)[None,:]

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

    yy = np.memmap( azTop, dtype=np.float32, mode='r',
                 shape=(topBurstIfg.numberOfLines, topBurstIfg.numberOfSamples))
    xx = np.memmap( rgTop, dtype=np.float32, mode='r',
                 shape=(topBurstIfg.numberOfLines, topBurstIfg.numberOfSamples))

    
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
    yy = np.memmap( azBot, dtype=np.float32, mode='r',
                shape=(botBurstIfg.numberOfLines, botBurstIfg.numberOfSamples))
    xx = np.memmap( rgBot, dtype=np.float32, mode='r',
                shape=(botBurstIfg.numberOfLines, botBurstIfg.numberOfSamples))

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

#    print('Ktm1: ', Ktm1[0,0], Ktm1[-1,-1])
#    print('Ktm2: ', Ktm2[0,0], Ktm2[-1,-1])
#    print('tm1 : ', tm1[0,0], tm1[-1,-1])
#    print('tm2 : ', tm2[0,0], tm2[-1,-1])
#    print('Kts1: ', Kts1[0,0], Kts1[-1,-1])
#    print('Kts2: ', Kts2[0,0], Kts2[-1,-1])
#    print('ts1 : ', ts1[0,0], ts2[-1,-1])
#    print('ts2 : ', ts2[0,0], ts2[-1,-1])
#    print('fm1 : ', fm1[0,0], fm1[-1,-1])
#    print('fm2 : ', fm2[0,0], fm2[-1,-1])
#    print('fs1 : ', fs1[0,0], fs1[-1,-1])
#    print('fs2 : ', fs2[0,0], fs2[-1,-1])


    return frequencySeparation


def createCoherence(intfile, win=5):
    '''
    Compute coherence using scipy convolve 2D.
    '''
    import scipy.signal as SS

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
    img.setLength(res.shape[0])
    img.dataType='FLOAT'
    img.setAccessMode('READ')
    img.renderHdr()

    return corfile

def runPrepESD(self):
    '''
    Create additional layers for performing ESD.
    '''

    if not self.doESD:
        return


    swathList = self._insar.getValidSwathList(self.swaths)


    for swath in swathList:
        if self._insar.numberOfCommonBursts[swath-1] < 2:
            print('Skipping prepesd for swath IW{0}'.format(swath))
            continue

        minBurst, maxBurst = self._insar.commonMasterBurstLimits(swath-1)
        slaveBurstStart, slaveBurstEnd = self._insar.commonSlaveBurstLimits(swath-1)


        ####Load full products
        master = self._insar.loadProduct( os.path.join(self._insar.masterSlcProduct, 'IW{0}.xml'.format(swath)))
        slave = self._insar.loadProduct( os.path.join(self._insar.slaveSlcProduct, 'IW{0}.xml'.format(swath)))

        ####Estimate relative shifts
        relShifts = getRelativeShifts(master, slave, minBurst, maxBurst, slaveBurstStart)
        maxBurst = maxBurst - 1 ###For overlaps

        ####Load metadata for burst IFGs
        ifgTop = self._insar.loadProduct( os.path.join(self._insar.coarseIfgOverlapProduct, 'top_IW{0}.xml'.format(swath)))
        ifgBottom = self._insar.loadProduct( os.path.join(self._insar.coarseIfgOverlapProduct, 'bottom_IW{0}.xml'.format(swath)))


        print('Relative shifts for swath {0}:'.format(swath))
        pprint.pprint(relShifts)

        ####Create ESD output directory
        esddir = self._insar.esdDirname
        if not os.path.isdir(esddir):
            os.makedirs(esddir)

        ####Overlap offsets directory
        offdir = os.path.join( self._insar.coarseOffsetsDirname, self._insar.overlapsSubDirname, 'IW{0}'.format(swath))

        ifglist = []
        factorlist = []
        offsetlist = []
        cohlist = []

        for ii in range(minBurst, maxBurst):
            ind = ii - minBurst            ###Index into overlaps
            sind = slaveBurstStart + ind   ###Index into slave

            topShift = relShifts[sind]
            botShift = relShifts[sind+1]


            topBurstIfg = ifgTop.bursts[ind]
            botBurstIfg = ifgBottom.bursts[ind]


            ####Double difference interferograms
            topInt = np.memmap( topBurstIfg.image.filename,
                    dtype=np.complex64, mode='r',
                    shape = (topBurstIfg.numberOfLines, topBurstIfg.numberOfSamples))

            botInt = np.memmap( botBurstIfg.image.filename,
                    dtype=np.complex64, mode='r',
                    shape = (botBurstIfg.numberOfLines, botBurstIfg.numberOfSamples))

            intName = os.path.join(esddir, 'overlap_IW%d_%02d.int'%(swath,ii+1))
            freqName = os.path.join(esddir, 'freq_IW%d_%02d.bin'%(swath,ii+1))

            with open(intName, 'wb') as fid:
                fid.write( topInt * np.conj(botInt))

            img = isceobj.createIntImage()
            img.setFilename(intName)
            img.setLength(topBurstIfg.numberOfLines)
            img.setWidth(topBurstIfg.numberOfSamples)
            img.setAccessMode('READ')
            img.renderHdr()

            multIntName= multilook(intName, alks = self.esdAzimuthLooks, rlks=self.esdRangeLooks)
            ifglist.append(multIntName)


            ####Estimate coherence of double different interferograms
            multCor = createCoherence(multIntName)
            cohlist.append(multCor)

            ####Estimate the frequency difference 
            azTop = os.path.join(offdir, 'azimuth_top_%02d_%02d.off'%(ii+1,ii+2))
            rgTop = os.path.join(offdir, 'range_top_%02d_%02d.off'%(ii+1,ii+2))
            azBot = os.path.join(offdir, 'azimuth_bot_%02d_%02d.off'%(ii+1,ii+2))
            rgBot = os.path.join(offdir, 'range_bot_%02d_%02d.off'%(ii+1,ii+2))

            mFullTop = master.bursts[ii]
            mFullBot = master.bursts[ii+1]
            sFullTop = slave.bursts[sind]
            sFullBot = slave.bursts[sind+1]

            freqdiff = overlapSpectralSeparation(topBurstIfg, botBurstIfg, mFullTop, mFullBot, sFullTop, sFullBot, azTop, rgTop, azBot, rgBot)

            with open(freqName, 'wb') as fid:
                (freqdiff * 2 * np.pi * mFullTop.azimuthTimeInterval).astype(np.float32).tofile(fid)

            img = isceobj.createImage()
            img.setFilename(freqName)
            img.setWidth(topBurstIfg.numberOfSamples)
            img.setLength(topBurstIfg.numberOfLines)
            img.setAccessMode('READ')
            img.bands = 1
            img.dataType = 'FLOAT'
            img.renderHdr()

            multConstName = multilook(freqName, alks = self.esdAzimuthLooks, rlks = self.esdRangeLooks)
            factorlist.append(multConstName)

        
        

