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
from .runFineResamp import getRelativeShifts, adjustValidSampleLine

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


def runCoarseResamp(self):
    '''
    Create coregistered overlap slaves.
    '''

    if not self.doESD:
        return


    swathList = self._insar.getValidSwathList(self.swaths)

    for swath in swathList:

        if self._insar.numberOfCommonBursts[swath-1] < 2:
            print('Skipping coarse resamp for swath IW{0}'.format(swath))
            continue

        ####Load slave metadata
        master = self._insar.loadProduct( os.path.join(self._insar.masterSlcProduct, 'IW{0}.xml'.format(swath)))
        slave = self._insar.loadProduct( os.path.join(self._insar.slaveSlcProduct, 'IW{0}.xml'.format(swath)))
        masterTop = self._insar.loadProduct( os.path.join(self._insar.masterSlcOverlapProduct, 'top_IW{0}.xml'.format(swath)))
        masterBottom = self._insar.loadProduct( os.path.join(self._insar.masterSlcOverlapProduct, 'bottom_IW{0}.xml'.format(swath)))


        dt = slave.bursts[0].azimuthTimeInterval
        dr = slave.bursts[0].rangePixelSize


        ###Output directory for coregistered SLCs
        outdir = os.path.join(self._insar.coarseCoregDirname, self._insar.overlapsSubDirname, 'IW{0}'.format(swath))
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

    
        ###Directory with offsets
        offdir = os.path.join(self._insar.coarseOffsetsDirname, self._insar.overlapsSubDirname, 'IW{0}'.format(swath))

        ####Indices w.r.t master
        minBurst, maxBurst = self._insar.commonMasterBurstLimits(swath-1)
        slaveBurstStart, slaveBurstEnd = self._insar.commonSlaveBurstLimits(swath-1)


        relShifts = getRelativeShifts(master, slave, minBurst, maxBurst, slaveBurstStart)
        maxBurst = maxBurst - 1 ###For overlaps

        print('Shifts for swath IW-{0}: {1}'.format(swath,relShifts))
   
        ####Can corporate known misregistration here

        apoly = Poly2D()
        apoly.initPoly(rangeOrder=0,azimuthOrder=0,coeffs=[[0.]])

        rpoly = Poly2D()
        rpoly.initPoly(rangeOrder=0,azimuthOrder=0,coeffs=[[0.]])


        topCoreg = createTOPSSwathSLCProduct()
        topCoreg.configure()

        botCoreg = createTOPSSwathSLCProduct()
        botCoreg.configure()

        for ii in range(minBurst, maxBurst):
            jj = slaveBurstStart + ii - minBurst 
        
            topBurst = masterTop.bursts[ii-minBurst]
            botBurst = masterBottom.bursts[ii-minBurst]
            slvBurst = slave.bursts[jj]



            #####Top burst processing
            try:
                offset = relShifts[jj]
            except:
                raise Exception('Trying to access shift for slave burst index {0}, which may not overlap with master - IW-{1}'.format(jj, swath))

            outname = os.path.join(outdir, 'burst_top_%02d_%02d.slc'%(ii+1,ii+2))
        
            ####Setup initial polynomials
            ### If no misregs are given, these are zero
            ### If provided, can be used for resampling without running to geo2rdr again for fast results
            rdict = {'azpoly' : apoly,
                     'rgpoly' : rpoly,
                     'rangeOff' : os.path.join(offdir, 'range_top_%02d_%02d.off'%(ii+1,ii+2)),
                    'azimuthOff': os.path.join(offdir, 'azimuth_top_%02d_%02d.off'%(ii+1,ii+2))}


            ###For future - should account for azimuth and range misreg here .. ignoring for now.
            azCarrPoly, dpoly = slave.estimateAzimuthCarrierPolynomials(slvBurst, offset = -1.0 * offset)

            rdict['carrPoly'] = azCarrPoly
            rdict['doppPoly'] = dpoly

            outimg = resampSlave(topBurst, slvBurst, rdict, outname)

            copyBurst = topBurst.clone()
            adjustValidSampleLine(copyBurst, slvBurst)
            copyBurst.image.filename = outimg.filename
            print('After: ', copyBurst.firstValidLine, copyBurst.numValidLines)
            topCoreg.bursts.append(copyBurst)
            #######################################################


            slvBurst = slave.bursts[jj+1]
            outname = os.path.join(outdir, 'burst_bot_%02d_%02d.slc'%(ii+1,ii+2))

            ####Setup initial polynomials
            ### If no misregs are given, these are zero
            ### If provided, can be used for resampling without running to geo2rdr again for fast results
            rdict = {'azpoly' : apoly,
                     'rgpoly' : rpoly,
                     'rangeOff' : os.path.join(offdir, 'range_bot_%02d_%02d.off'%(ii+1,ii+2)),
                    'azimuthOff': os.path.join(offdir, 'azimuth_bot_%02d_%02d.off'%(ii+1,ii+2))}

            azCarrPoly, dpoly = slave.estimateAzimuthCarrierPolynomials(slvBurst, offset = -1.0 * offset)

            rdict['carrPoly'] = azCarrPoly
            rdict['doppPoly'] = dpoly

            outimg = resampSlave(botBurst, slvBurst, rdict, outname)

            copyBurst = botBurst.clone()
            adjustValidSampleLine(copyBurst, slvBurst)
            copyBurst.image.filename = outimg.filename
            print('After: ', copyBurst.firstValidLine, copyBurst.numValidLines)
            botCoreg.bursts.append(copyBurst)
            #######################################################


        topCoreg.numberOfBursts = len(topCoreg.bursts)
        botCoreg.numberOfBursts = len(botCoreg.bursts)

        self._insar.saveProduct(topCoreg, os.path.join(self._insar.coregOverlapProduct, 'top_IW{0}.xml'.format(swath)))
        self._insar.saveProduct(botCoreg, os.path.join(self._insar.coregOverlapProduct, 'bottom_IW{0}.xml'.format(swath)))

