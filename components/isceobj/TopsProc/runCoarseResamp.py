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

def resampSecondary(mas, slv, rdict, outname ):
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
    Create coregistered overlap secondarys.
    '''

    if not self.doESD:
        return


    swathList = self._insar.getValidSwathList(self.swaths)

    for swath in swathList:

        if self._insar.numberOfCommonBursts[swath-1] < 2:
            print('Skipping coarse resamp for swath IW{0}'.format(swath))
            continue

        ####Load secondary metadata
        reference = self._insar.loadProduct( os.path.join(self._insar.referenceSlcProduct, 'IW{0}.xml'.format(swath)))
        secondary = self._insar.loadProduct( os.path.join(self._insar.secondarySlcProduct, 'IW{0}.xml'.format(swath)))
        referenceTop = self._insar.loadProduct( os.path.join(self._insar.referenceSlcOverlapProduct, 'top_IW{0}.xml'.format(swath)))
        referenceBottom = self._insar.loadProduct( os.path.join(self._insar.referenceSlcOverlapProduct, 'bottom_IW{0}.xml'.format(swath)))


        dt = secondary.bursts[0].azimuthTimeInterval
        dr = secondary.bursts[0].rangePixelSize


        ###Output directory for coregistered SLCs
        outdir = os.path.join(self._insar.coarseCoregDirname, self._insar.overlapsSubDirname, 'IW{0}'.format(swath))
        os.makedirs(outdir, exist_ok=True)

    
        ###Directory with offsets
        offdir = os.path.join(self._insar.coarseOffsetsDirname, self._insar.overlapsSubDirname, 'IW{0}'.format(swath))

        ####Indices w.r.t reference
        minBurst, maxBurst = self._insar.commonReferenceBurstLimits(swath-1)
        secondaryBurstStart, secondaryBurstEnd = self._insar.commonSecondaryBurstLimits(swath-1)


        relShifts = getRelativeShifts(reference, secondary, minBurst, maxBurst, secondaryBurstStart)
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
            jj = secondaryBurstStart + ii - minBurst 
        
            topBurst = referenceTop.bursts[ii-minBurst]
            botBurst = referenceBottom.bursts[ii-minBurst]
            slvBurst = secondary.bursts[jj]



            #####Top burst processing
            try:
                offset = relShifts[jj]
            except:
                raise Exception('Trying to access shift for secondary burst index {0}, which may not overlap with reference - IW-{1}'.format(jj, swath))

            outname = os.path.join(outdir, 'burst_top_%02d_%02d.slc'%(ii+1,ii+2))
        
            ####Setup initial polynomials
            ### If no misregs are given, these are zero
            ### If provided, can be used for resampling without running to geo2rdr again for fast results
            rdict = {'azpoly' : apoly,
                     'rgpoly' : rpoly,
                     'rangeOff' : os.path.join(offdir, 'range_top_%02d_%02d.off'%(ii+1,ii+2)),
                    'azimuthOff': os.path.join(offdir, 'azimuth_top_%02d_%02d.off'%(ii+1,ii+2))}


            ###For future - should account for azimuth and range misreg here .. ignoring for now.
            azCarrPoly, dpoly = secondary.estimateAzimuthCarrierPolynomials(slvBurst, offset = -1.0 * offset)

            rdict['carrPoly'] = azCarrPoly
            rdict['doppPoly'] = dpoly

            outimg = resampSecondary(topBurst, slvBurst, rdict, outname)

            copyBurst = topBurst.clone()
            adjustValidSampleLine(copyBurst, slvBurst)
            copyBurst.image.filename = outimg.filename
            print('After: ', copyBurst.firstValidLine, copyBurst.numValidLines)
            topCoreg.bursts.append(copyBurst)
            #######################################################


            slvBurst = secondary.bursts[jj+1]
            outname = os.path.join(outdir, 'burst_bot_%02d_%02d.slc'%(ii+1,ii+2))

            ####Setup initial polynomials
            ### If no misregs are given, these are zero
            ### If provided, can be used for resampling without running to geo2rdr again for fast results
            rdict = {'azpoly' : apoly,
                     'rgpoly' : rpoly,
                     'rangeOff' : os.path.join(offdir, 'range_bot_%02d_%02d.off'%(ii+1,ii+2)),
                    'azimuthOff': os.path.join(offdir, 'azimuth_bot_%02d_%02d.off'%(ii+1,ii+2))}

            azCarrPoly, dpoly = secondary.estimateAzimuthCarrierPolynomials(slvBurst, offset = -1.0 * offset)

            rdict['carrPoly'] = azCarrPoly
            rdict['doppPoly'] = dpoly

            outimg = resampSecondary(botBurst, slvBurst, rdict, outname)

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

