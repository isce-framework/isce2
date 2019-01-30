#
# Author: Piyush Agram
# Copyright 2016
#

import isce
import isceobj
import stdproc
from stdproc.stdproc import crossmul
import numpy as np
from isceobj.Util.Poly2D import Poly2D
import os
import copy
from isceobj.Sensor.TOPS import createTOPSSwathSLCProduct

def resampSlave(mas, slv, rdict, outname ):
    '''
    Resample burst by burst.
    '''
   
    azpoly = rdict['azpoly']
    rgpoly = rdict['rgpoly']
    azcarrpoly = rdict['carrPoly']
    dpoly = rdict['doppPoly']

    rngImg = isceobj.createImage()
    rngImg.load(rdict['rangeOff'] + '.xml')
    rngImg.setAccessMode('READ')

    aziImg = isceobj.createImage()
    aziImg.load(rdict['azimuthOff'] + '.xml')
    aziImg.setAccessMode('READ')

    inimg = isceobj.createSlcImage()
    inimg.load(slv.image.filename + '.xml')
    inimg.setAccessMode('READ')


    rObj = stdproc.createResamp_slc()
    rObj.slantRangePixelSpacing = slv.rangePixelSize
    rObj.radarWavelength = slv.radarWavelength
    rObj.azimuthCarrierPoly = azcarrpoly
    rObj.dopplerPoly = dpoly
   
    rObj.azimuthOffsetsPoly = azpoly
    rObj.rangeOffsetsPoly = rgpoly
    rObj.imageIn = inimg

    ####Setting reference values
    rObj.startingRange = slv.startingRange
    rObj.referenceSlantRangePixelSpacing = mas.rangePixelSize
    rObj.referenceStartingRange = mas.startingRange
    rObj.referenceWavelength = mas.radarWavelength


    width = mas.numberOfSamples
    length = mas.numberOfLines
    imgOut = isceobj.createSlcImage()
    imgOut.setWidth(width)
    imgOut.filename = outname
    imgOut.setAccessMode('write')

    rObj.outputWidth = width
    rObj.outputLines = length
    rObj.residualRangeImage = rngImg
    rObj.residualAzimuthImage = aziImg

    rObj.resamp_slc(imageOut=imgOut)

    imgOut.renderHdr()
    return imgOut


def getRelativeShifts(mFrame, sFrame, minBurst, maxBurst, slaveBurstStart):
    '''
    Estimate the relative shifts between the start of the bursts.
    '''
    
    azMasterOff = {}
    azSlaveOff = {}
    azRelOff = {}
    tm = mFrame.bursts[minBurst].sensingStart
    dt = mFrame.bursts[minBurst].azimuthTimeInterval
    ts = sFrame.bursts[slaveBurstStart].sensingStart
    
    for index in range(minBurst, maxBurst):
        burst = mFrame.bursts[index]
        azMasterOff[index] = int(np.round((burst.sensingStart - tm).total_seconds() / dt))

        burst = sFrame.bursts[slaveBurstStart + index - minBurst]
        azSlaveOff[slaveBurstStart + index - minBurst] =  int(np.round((burst.sensingStart - ts).total_seconds() / dt))

        azRelOff[slaveBurstStart + index - minBurst] = azSlaveOff[slaveBurstStart + index - minBurst] - azMasterOff[index]


    return azRelOff



def adjustValidSampleLine(master, slave, minAz=0, maxAz=0, minRng=0, maxRng=0):
    ####Adjust valid samples and first valid sample here
    print ("Adjust valid samples")
    print('Before: ', master.firstValidSample, master.numValidSamples)
    print('Offsets : ', minRng, maxRng)

    if (minRng > 0) and (maxRng > 0):
        master.firstValidSample = slave.firstValidSample - int(np.floor(maxRng)-4)
        lastValidSample = master.firstValidSample - 8 + slave.numValidSamples

        if lastValidSample < master.numberOfSamples:
            master.numValidSamples = slave.numValidSamples - 8
        else:
            master.numValidSamples = master.numberOfSamples - master.firstValidSample

    elif (minRng < 0) and (maxRng < 0):
            master.firstValidSample = slave.firstValidSample - int(np.floor(minRng) - 4)
            lastValidSample = master.firstValidSample + slave.numValidSamples  - 8
            if lastValidSample < master.numberOfSamples:
               master.numValidSamples = slave.numValidSamples - 8
            else:
               master.numValidSamples = master.numberOfSamples - master.firstValidSample
    elif (minRng < 0) and (maxRng > 0):
            master.firstValidSample = slave.firstValidSample - int(np.floor(minRng) - 4)
            lastValidSample = master.firstValidSample + slave.numValidSamples + int(np.floor(minRng) - 8) - int(np.ceil(maxRng))
            if lastValidSample < master.numberOfSamples:
               master.numValidSamples = slave.numValidSamples + int(np.floor(minRng) - 8) - int(np.ceil(maxRng))
            else:
               master.numValidSamples = master.numberOfSamples - master.firstValidSample

    master.firstValidSample = np.max([0, master.firstValidSample])
    ###Adjust valid lines and first valid line here
    print ("Adjust valid lines")
    print('Before: ', master.firstValidLine, master.numValidLines)
    print('Offsets : ', minAz, maxAz)
    if (minAz > 0) and (maxAz > 0):

            master.firstValidLine = slave.firstValidLine - int(np.floor(maxAz) - 4)
            lastValidLine = master.firstValidLine - 8 + slave.numValidLines

            if lastValidLine < master.numberOfLines:
                master.numValidLines = slave.numValidLines - 8
            else:
                master.numValidLines = master.numberOfLines - master.firstValidLine

    elif (minAz < 0) and  (maxAz < 0):
            master.firstValidLine = slave.firstValidLine - int(np.floor(minAz) - 4)
            lastValidLine = master.firstValidLine + slave.numValidLines  - 8
            if lastValidLine < master.numberOfLines:
               master.numValidLines = slave.numValidLines - 8
            else:
               master.numValidLines = master.numberOfLines - master.firstValidLine
    
    elif (minAz < 0) and (maxAz > 0):
            master.firstValidLine = slave.firstValidLine - int(np.floor(minAz) - 4)
            lastValidLine = master.firstValidLine + slave.numValidLines + int(np.floor(minAz) - 8) - int(np.ceil(maxAz))
            if lastValidLine < master.numberOfLines:
               master.numValidLines = slave.numValidLines + int(np.floor(minAz) - 8) - int(np.ceil(maxAz))
            else:
               master.numValidLines = master.numberOfLines - master.firstValidLine



def getValidLines(slave, rdict, inname, misreg_az=0.0, misreg_rng=0.0):
    '''
    Looks at the master, slave and azimuth offsets and gets the Interferogram valid lines 
    '''
    dimg = isceobj.createSlcImage()
    dimg.load(inname + '.xml')
    shp = (dimg.length, dimg.width)
    az = np.fromfile(rdict['azimuthOff'], dtype=np.float32).reshape(shp)
    az += misreg_az
    aa = np.zeros(az.shape)
    aa[:,:] = az
    aa[aa < -10000.0] = np.nan
    amin = np.nanmin(aa)
    amax = np.nanmax(aa)

    rng = np.fromfile(rdict['rangeOff'], dtype=np.float32).reshape(shp)
    rng += misreg_rng
    rr = np.zeros(rng.shape)
    rr[:,:] = rng
    rr[rr < -10000.0] = np.nan
    rmin = np.nanmin(rr)
    rmax = np.nanmax(rr)

    return amin, amax, rmin, rmax



def runFineResamp(self):
    '''
    Create coregistered overlap slaves.
    '''


    swathList = self._insar.getValidSwathList(self.swaths)


    for swath in swathList:
        ####Load slave metadata
        master = self._insar.loadProduct( os.path.join(self._insar.masterSlcProduct, 'IW{0}.xml'.format(swath)))
        slave = self._insar.loadProduct( os.path.join(self._insar.slaveSlcProduct, 'IW{0}.xml'.format(swath)))

        dt = slave.bursts[0].azimuthTimeInterval
        dr = slave.bursts[0].rangePixelSize


        ###Output directory for coregistered SLCs
        outdir = os.path.join(self._insar.fineCoregDirname, 'IW{0}'.format(swath))
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

    
        ###Directory with offsets
        offdir = os.path.join(self._insar.fineOffsetsDirname, 'IW{0}'.format(swath))

        ####Indices w.r.t master
        minBurst, maxBurst = self._insar.commonMasterBurstLimits(swath-1)
        slaveBurstStart, slaveBurstEnd = self._insar.commonSlaveBurstLimits(swath-1)

        if minBurst == maxBurst:
            print('Skipping processing of swath {0}'.format(swath))
            continue

        relShifts = getRelativeShifts(master, slave, minBurst, maxBurst, slaveBurstStart)
        print('Shifts IW-{0}: '.format(swath), relShifts) 
   
        ####Can corporate known misregistration here
        apoly = Poly2D()
        apoly.initPoly(rangeOrder=0,azimuthOrder=0,coeffs=[[0.]])

        rpoly = Poly2D()
        rpoly.initPoly(rangeOrder=0,azimuthOrder=0,coeffs=[[0.]])


        misreg_az = self._insar.slaveTimingCorrection / dt
        misreg_rg = self._insar.slaveRangeCorrection / dr


        coreg = createTOPSSwathSLCProduct()
        coreg.configure()

        for ii in range(minBurst, maxBurst):
            jj = slaveBurstStart + ii - minBurst 
    
            masBurst = master.bursts[ii]
            slvBurst = slave.bursts[jj]

            try:
                offset = relShifts[jj]
            except:
                raise Exception('Trying to access shift for slave burst index {0}, which may not overlap with master for swath {1}'.format(jj, swath))

            outname = os.path.join(outdir, 'burst_%02d.slc'%(ii+1))
        
            ####Setup initial polynomials
            ### If no misregs are given, these are zero
            ### If provided, can be used for resampling without running to geo2rdr again for fast results
            rdict = {'azpoly' : apoly,
                     'rgpoly' : rpoly,
                     'rangeOff' : os.path.join(offdir, 'range_%02d.off'%(ii+1)),
                    'azimuthOff': os.path.join(offdir, 'azimuth_%02d.off'%(ii+1))}


            ###For future - should account for azimuth and range misreg here .. ignoring for now.
            azCarrPoly, dpoly = slave.estimateAzimuthCarrierPolynomials(slvBurst, offset = -1.0 * offset)

            rdict['carrPoly'] = azCarrPoly
            rdict['doppPoly'] = dpoly

            outimg = resampSlave(masBurst, slvBurst, rdict, outname)
            minAz, maxAz, minRg, maxRg = getValidLines(slvBurst, rdict, outname,
                    misreg_az = misreg_az - offset, misreg_rng = misreg_rg)


#            copyBurst = copy.deepcopy(masBurst)
            copyBurst = masBurst.clone()
            adjustValidSampleLine(copyBurst, slvBurst,
                                         minAz=minAz, maxAz=maxAz,
                                         minRng=minRg, maxRng=maxRg)
            copyBurst.image.filename = outimg.filename
            print('After: ', copyBurst.firstValidLine, copyBurst.numValidLines)
            coreg.bursts.append(copyBurst)
            #######################################################

        coreg.numberOfBursts = len(coreg.bursts)

        self._insar.saveProduct(coreg, os.path.join(self._insar.fineCoregDirname, 'IW{0}.xml'.format(swath)))

