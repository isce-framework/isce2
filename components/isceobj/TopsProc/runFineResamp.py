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


def getRelativeShifts(mFrame, sFrame, minBurst, maxBurst, secondaryBurstStart):
    '''
    Estimate the relative shifts between the start of the bursts.
    '''
    
    azReferenceOff = {}
    azSecondaryOff = {}
    azRelOff = {}
    tm = mFrame.bursts[minBurst].sensingStart
    dt = mFrame.bursts[minBurst].azimuthTimeInterval
    ts = sFrame.bursts[secondaryBurstStart].sensingStart
    
    for index in range(minBurst, maxBurst):
        burst = mFrame.bursts[index]
        azReferenceOff[index] = int(np.round((burst.sensingStart - tm).total_seconds() / dt))

        burst = sFrame.bursts[secondaryBurstStart + index - minBurst]
        azSecondaryOff[secondaryBurstStart + index - minBurst] =  int(np.round((burst.sensingStart - ts).total_seconds() / dt))

        azRelOff[secondaryBurstStart + index - minBurst] = azSecondaryOff[secondaryBurstStart + index - minBurst] - azReferenceOff[index]


    return azRelOff



def adjustValidSampleLine(reference, secondary, minAz=0, maxAz=0, minRng=0, maxRng=0):
    ####Adjust valid samples and first valid sample here
    print ("Adjust valid samples")
    print('Before: ', reference.firstValidSample, reference.numValidSamples)
    print('Offsets : ', minRng, maxRng)

    if (minRng > 0) and (maxRng > 0):
        reference.firstValidSample = secondary.firstValidSample - int(np.floor(maxRng)-4)
        lastValidSample = reference.firstValidSample - 8 + secondary.numValidSamples

        if lastValidSample < reference.numberOfSamples:
            reference.numValidSamples = secondary.numValidSamples - 8
        else:
            reference.numValidSamples = reference.numberOfSamples - reference.firstValidSample

    elif (minRng < 0) and (maxRng < 0):
            reference.firstValidSample = secondary.firstValidSample - int(np.floor(minRng) - 4)
            lastValidSample = reference.firstValidSample + secondary.numValidSamples  - 8
            if lastValidSample < reference.numberOfSamples:
               reference.numValidSamples = secondary.numValidSamples - 8
            else:
               reference.numValidSamples = reference.numberOfSamples - reference.firstValidSample
    elif (minRng < 0) and (maxRng > 0):
            reference.firstValidSample = secondary.firstValidSample - int(np.floor(minRng) - 4)
            lastValidSample = reference.firstValidSample + secondary.numValidSamples + int(np.floor(minRng) - 8) - int(np.ceil(maxRng))
            if lastValidSample < reference.numberOfSamples:
               reference.numValidSamples = secondary.numValidSamples + int(np.floor(minRng) - 8) - int(np.ceil(maxRng))
            else:
               reference.numValidSamples = reference.numberOfSamples - reference.firstValidSample

    reference.firstValidSample = np.max([0, reference.firstValidSample])
    ###Adjust valid lines and first valid line here
    print ("Adjust valid lines")
    print('Before: ', reference.firstValidLine, reference.numValidLines)
    print('Offsets : ', minAz, maxAz)
    if (minAz > 0) and (maxAz > 0):

            reference.firstValidLine = secondary.firstValidLine - int(np.floor(maxAz) - 4)
            lastValidLine = reference.firstValidLine - 8 + secondary.numValidLines

            if lastValidLine < reference.numberOfLines:
                reference.numValidLines = secondary.numValidLines - 8
            else:
                reference.numValidLines = reference.numberOfLines - reference.firstValidLine

    elif (minAz < 0) and  (maxAz < 0):
            reference.firstValidLine = secondary.firstValidLine - int(np.floor(minAz) - 4)
            lastValidLine = reference.firstValidLine + secondary.numValidLines  - 8
            if lastValidLine < reference.numberOfLines:
               reference.numValidLines = secondary.numValidLines - 8
            else:
               reference.numValidLines = reference.numberOfLines - reference.firstValidLine
    
    elif (minAz < 0) and (maxAz > 0):
            reference.firstValidLine = secondary.firstValidLine - int(np.floor(minAz) - 4)
            lastValidLine = reference.firstValidLine + secondary.numValidLines + int(np.floor(minAz) - 8) - int(np.ceil(maxAz))
            if lastValidLine < reference.numberOfLines:
               reference.numValidLines = secondary.numValidLines + int(np.floor(minAz) - 8) - int(np.ceil(maxAz))
            else:
               reference.numValidLines = reference.numberOfLines - reference.firstValidLine



def getValidLines(secondary, rdict, inname, misreg_az=0.0, misreg_rng=0.0):
    '''
    Looks at the reference, secondary and azimuth offsets and gets the Interferogram valid lines 
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
    Create coregistered overlap secondarys.
    '''


    swathList = self._insar.getValidSwathList(self.swaths)


    for swath in swathList:
        ####Load secondary metadata
        reference = self._insar.loadProduct( os.path.join(self._insar.referenceSlcProduct, 'IW{0}.xml'.format(swath)))
        secondary = self._insar.loadProduct( os.path.join(self._insar.secondarySlcProduct, 'IW{0}.xml'.format(swath)))

        dt = secondary.bursts[0].azimuthTimeInterval
        dr = secondary.bursts[0].rangePixelSize


        ###Output directory for coregistered SLCs
        outdir = os.path.join(self._insar.fineCoregDirname, 'IW{0}'.format(swath))
        os.makedirs(outdir, exist_ok=True)

    
        ###Directory with offsets
        offdir = os.path.join(self._insar.fineOffsetsDirname, 'IW{0}'.format(swath))

        ####Indices w.r.t reference
        minBurst, maxBurst = self._insar.commonReferenceBurstLimits(swath-1)
        secondaryBurstStart, secondaryBurstEnd = self._insar.commonSecondaryBurstLimits(swath-1)

        if minBurst == maxBurst:
            print('Skipping processing of swath {0}'.format(swath))
            continue

        relShifts = getRelativeShifts(reference, secondary, minBurst, maxBurst, secondaryBurstStart)
        print('Shifts IW-{0}: '.format(swath), relShifts) 
   
        ####Can corporate known misregistration here
        apoly = Poly2D()
        apoly.initPoly(rangeOrder=0,azimuthOrder=0,coeffs=[[0.]])

        rpoly = Poly2D()
        rpoly.initPoly(rangeOrder=0,azimuthOrder=0,coeffs=[[0.]])


        misreg_az = self._insar.secondaryTimingCorrection / dt
        misreg_rg = self._insar.secondaryRangeCorrection / dr


        coreg = createTOPSSwathSLCProduct()
        coreg.configure()

        for ii in range(minBurst, maxBurst):
            jj = secondaryBurstStart + ii - minBurst 
    
            masBurst = reference.bursts[ii]
            slvBurst = secondary.bursts[jj]

            try:
                offset = relShifts[jj]
            except:
                raise Exception('Trying to access shift for secondary burst index {0}, which may not overlap with reference for swath {1}'.format(jj, swath))

            outname = os.path.join(outdir, 'burst_%02d.slc'%(ii+1))
        
            ####Setup initial polynomials
            ### If no misregs are given, these are zero
            ### If provided, can be used for resampling without running to geo2rdr again for fast results
            rdict = {'azpoly' : apoly,
                     'rgpoly' : rpoly,
                     'rangeOff' : os.path.join(offdir, 'range_%02d.off'%(ii+1)),
                    'azimuthOff': os.path.join(offdir, 'azimuth_%02d.off'%(ii+1))}


            ###For future - should account for azimuth and range misreg here .. ignoring for now.
            azCarrPoly, dpoly = secondary.estimateAzimuthCarrierPolynomials(slvBurst, offset = -1.0 * offset)

            rdict['carrPoly'] = azCarrPoly
            rdict['doppPoly'] = dpoly

            outimg = resampSecondary(masBurst, slvBurst, rdict, outname)
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

