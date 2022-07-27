#
# Author: Piyush Agram
# Copyright 2016
#
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
import logging

logger = logging.getLogger('isce.topsinsar.fineresamp')

def resampSecondaryCPU(reference, secondary, rdict, outname):
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
    inimg.load(secondary.image.filename + '.xml')
    inimg.setAccessMode('READ')

    rObj = stdproc.createResamp_slc()
    rObj.slantRangePixelSpacing = secondary.rangePixelSize
    rObj.radarWavelength = secondary.radarWavelength
    rObj.azimuthCarrierPoly = azcarrpoly
    rObj.dopplerPoly = dpoly

    rObj.azimuthOffsetsPoly = azpoly
    rObj.rangeOffsetsPoly = rgpoly
    rObj.imageIn = inimg

    ####Setting reference values
    rObj.startingRange = secondary.startingRange
    rObj.referenceSlantRangePixelSpacing = reference.rangePixelSize
    rObj.referenceStartingRange = reference.startingRange
    rObj.referenceWavelength = reference.radarWavelength


    width = reference.numberOfSamples
    length = reference.numberOfLines
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

def convertPoly2D(poly):
    '''
    Convert a isceobj.Util.Poly2D {poly} into zerodop.GPUresampslc.GPUresampslc.PyPloy2d
    '''
    from zerodop.GPUresampslc.GPUresampslc import PyPoly2d
    import itertools

    # get parameters from poly
    azimuthOrder = poly.getAzimuthOrder()
    rangeOrder = poly.getRangeOrder()
    azimuthMean = poly.getMeanAzimuth()
    rangeMean = poly.getMeanRange()
    azimuthNorm = poly.getNormAzimuth()
    rangeNorm = poly.getNormRange()

    # create the PyPoly2d object
    pPoly = PyPoly2d(azimuthOrder, rangeOrder, azimuthMean, rangeMean, azimuthNorm, rangeNorm)
    # copy the coeffs, need to flatten into 1d list
    pPoly.coeffs = list(itertools.chain.from_iterable(poly.getCoeffs()))

    # all done
    return pPoly

def resampSecondaryGPU(reference, secondary, rdict, outname):
    '''
    Resample burst by burst with GPU
    '''

    # import the GPU module
    import zerodop.GPUresampslc

    # get Poly2D objects from rdict and convert them into PyPoly2d objects
    azpoly = convertPoly2D(rdict['azpoly'])
    rgpoly = convertPoly2D(rdict['rgpoly'])

    azcarrpoly = convertPoly2D(rdict['carrPoly'])
    dpoly = convertPoly2D(rdict['doppPoly'])

    rngImg = isceobj.createImage()
    rngImg.load(rdict['rangeOff'] + '.xml')
    rngImg.setCaster('read', 'FLOAT')
    rngImg.createImage()

    aziImg = isceobj.createImage()
    aziImg.load(rdict['azimuthOff'] + '.xml')
    aziImg.setCaster('read', 'FLOAT')
    aziImg.createImage()

    inimg = isceobj.createSlcImage()
    inimg.load(secondary.image.filename + '.xml')
    inimg.setAccessMode('READ')
    inimg.createImage()

    # create a GPU resample processor
    rObj = zerodop.GPUresampslc.createResampSlc()

    # set parameters
    rObj.slr = secondary.rangePixelSize
    rObj.wvl = secondary.radarWavelength

    # set polynomials
    rObj.azCarrier = azcarrpoly
    rObj.dopplerPoly = dpoly
    rObj.azOffsetsPoly = azpoly
    rObj.rgOffsetsPoly = rgpoly
    # need to create an empty rgCarrier poly
    rgCarrier = Poly2D()
    rgCarrier.initPoly(rangeOrder=0, azimuthOrder=0, coeffs=[[0.]])
    rgCarrier = convertPoly2D(rgCarrier)
    rObj.rgCarrier = rgCarrier

    # input secondary image
    rObj.slcInAccessor = inimg.getImagePointer()
    rObj.inWidth = inimg.getWidth()
    rObj.inLength = inimg.getLength()

    ####Setting reference values
    rObj.r0 = secondary.startingRange
    rObj.refr0 = reference.rangePixelSize
    rObj.refslr = reference.startingRange
    rObj.refwvl = reference.radarWavelength

    # set output image
    width = reference.numberOfSamples
    length = reference.numberOfLines

    imgOut = isceobj.createSlcImage()
    imgOut.setWidth(width)
    imgOut.filename = outname
    imgOut.setAccessMode('write')
    imgOut.createImage()
    rObj.slcOutAccessor = imgOut.getImagePointer()

    rObj.outWidth = width
    rObj.outLength = length
    rObj.residRgAccessor = rngImg.getImagePointer()
    rObj.residAzAccessor = aziImg.getImagePointer()

    # need to specify data type, only complex is currently supported
    rObj.isComplex = (inimg.dataType == 'CFLOAT')
    # run resampling
    rObj.resamp_slc()

    # finalize images
    inimg.finalizeImage()
    imgOut.finalizeImage()
    rngImg.finalizeImage()
    aziImg.finalizeImage()

    imgOut.renderHdr()
    return imgOut

def getRelativeShifts(referenceFrame, secondaryFrame, minBurst, maxBurst, secondaryBurstStart):
    '''
    Estimate the relative shifts between the start of the bursts.
    '''

    azReferenceOff = {}
    azSecondaryOff = {}
    azRelOff = {}
    tm = referenceFrame.bursts[minBurst].sensingStart
    dt = referenceFrame.bursts[minBurst].azimuthTimeInterval
    ts = secondaryFrame.bursts[secondaryBurstStart].sensingStart

    for index in range(minBurst, maxBurst):
        burst = referenceFrame.bursts[index]
        azReferenceOff[index] = int(np.round((burst.sensingStart - tm).total_seconds() / dt))

        burst = secondaryFrame.bursts[secondaryBurstStart + index - minBurst]
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
    Create coregistered overlap secondary image.
    '''

    # decide whether to use CPU or GPU
    hasGPU = self.useGPU and self._insar.hasGPU()

    if hasGPU:
        resampSecondary = resampSecondaryGPU
        print('Using GPU for fineresamp')
    else:
        resampSecondary = resampSecondaryCPU


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

            referenceBurst = reference.bursts[ii]
            secondaryBurst = secondary.bursts[jj]

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
            azCarrPoly, dpoly = secondary.estimateAzimuthCarrierPolynomials(secondaryBurst, offset = -1.0 * offset)

            rdict['carrPoly'] = azCarrPoly
            rdict['doppPoly'] = dpoly

            outimg = resampSecondary(referenceBurst, secondaryBurst, rdict, outname)

            minAz, maxAz, minRg, maxRg = getValidLines(secondaryBurst, rdict, outname,
                    misreg_az = misreg_az - offset, misreg_rng = misreg_rg)

#            copyBurst = copy.deepcopy(referenceBurst)
            copyBurst = referenceBurst.clone()
            adjustValidSampleLine(copyBurst, secondaryBurst,
                                         minAz=minAz, maxAz=maxAz,
                                         minRng=minRg, maxRng=maxRg)
            copyBurst.image.filename = outimg.filename
            print('After: ', copyBurst.firstValidLine, copyBurst.numValidLines)
            coreg.bursts.append(copyBurst)
            #######################################################

        coreg.numberOfBursts = len(coreg.bursts)

        self._insar.saveProduct(coreg, os.path.join(self._insar.fineCoregDirname, 'IW{0}.xml'.format(swath)))
