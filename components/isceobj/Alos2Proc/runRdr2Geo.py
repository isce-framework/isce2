#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging

import isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import waterBodyRadar

logger = logging.getLogger('isce.alos2insar.runRdr2Geo')

def runRdr2Geo(self):
    '''compute lat/lon/hgt
    '''
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    referenceTrack = self._insar.loadTrack(reference=True)
    demFile = os.path.abspath(self._insar.dem)
    wbdFile = os.path.abspath(self._insar.wbd)

    insarDir = 'insar'
    os.makedirs(insarDir, exist_ok=True)
    os.chdir(insarDir)


    if self.useGPU and self._insar.hasGPU():
        topoGPU(referenceTrack, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, demFile, 
                       self._insar.latitude, self._insar.longitude, self._insar.height, self._insar.los)
    else:
        snwe = topoCPU(referenceTrack, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, demFile, 
                       self._insar.latitude, self._insar.longitude, self._insar.height, self._insar.los)
    waterBodyRadar(self._insar.latitude, self._insar.longitude, wbdFile, self._insar.wbdOut)

    os.chdir('../')

    catalog.printToLog(logger, "runRdr2Geo")
    self._insar.procDoc.addAllFromCatalog(catalog)


def topoCPU(referenceTrack, numberRangeLooks, numberAzimuthLooks, demFile, latFile, lonFile, hgtFile, losFile):
    import datetime
    import isceobj
    from zerodop.topozero import createTopozero
    from isceobj.Planet.Planet import Planet

    pointingDirection = {'right': -1, 'left' :1}

    demImage = isceobj.createDemImage()
    demImage.load(demFile + '.xml')
    demImage.setAccessMode('read')

    planet = Planet(pname='Earth')

    topo = createTopozero()
    topo.slantRangePixelSpacing = numberRangeLooks * referenceTrack.rangePixelSize
    topo.prf = 1.0 / (numberAzimuthLooks*referenceTrack.azimuthLineInterval)
    topo.radarWavelength = referenceTrack.radarWavelength
    topo.orbit = referenceTrack.orbit
    topo.width = referenceTrack.numberOfSamples
    topo.length = referenceTrack.numberOfLines
    topo.wireInputPort(name='dem', object=demImage)
    topo.wireInputPort(name='planet', object=planet)
    topo.numberRangeLooks = 1 #must be set as 1
    topo.numberAzimuthLooks = 1 #must be set as 1 Cunren
    topo.lookSide = pointingDirection[referenceTrack.pointingDirection]
    topo.sensingStart = referenceTrack.sensingStart + datetime.timedelta(seconds=(numberAzimuthLooks-1.0)/2.0*referenceTrack.azimuthLineInterval)
    topo.rangeFirstSample = referenceTrack.startingRange + (numberRangeLooks-1.0)/2.0*referenceTrack.rangePixelSize
    topo.demInterpolationMethod='BIQUINTIC'

    topo.latFilename = latFile
    topo.lonFilename = lonFile
    topo.heightFilename = hgtFile
    topo.losFilename = losFile
    #topo.incFilename = incName
    #topo.maskFilename = mskName

    topo.topo()

    return list(topo.snwe)


def topoGPU(referenceTrack, numberRangeLooks, numberAzimuthLooks, demFile, latFile, lonFile, hgtFile, losFile):
    '''
    Try with GPU module.
    '''
    import datetime
    import numpy as np
    from isceobj.Planet.Planet import Planet
    from zerodop.GPUtopozero.GPUtopozero import PyTopozero
    from isceobj.Util.Poly2D import Poly2D
    from iscesys import DateTimeUtil as DTU

    pointingDirection = {'right': -1, 'left' :1}

    #creat poynomials
    polyDoppler = Poly2D(name='topsApp_dopplerPoly')
    polyDoppler.setWidth(referenceTrack.numberOfSamples)
    polyDoppler.setLength(referenceTrack.numberOfLines)
    polyDoppler.setNormRange(1.0)
    polyDoppler.setNormAzimuth(1.0)
    polyDoppler.setMeanRange(0.0)
    polyDoppler.setMeanAzimuth(0.0)
    polyDoppler.initPoly(rangeOrder=0,azimuthOrder=0, coeffs=[[0.]])
    polyDoppler.createPoly2D()

    slantRangeImage = Poly2D()
    slantRangeImage.setWidth(referenceTrack.numberOfSamples)
    slantRangeImage.setLength(referenceTrack.numberOfLines)
    slantRangeImage.setNormRange(1.0)
    slantRangeImage.setNormAzimuth(1.0)
    slantRangeImage.setMeanRange(0.)
    slantRangeImage.setMeanAzimuth(0.)
    slantRangeImage.initPoly(rangeOrder=1,azimuthOrder=0,
        coeffs=[[referenceTrack.startingRange + (numberRangeLooks-1.0)/2.0*referenceTrack.rangePixelSize,numberRangeLooks * referenceTrack.rangePixelSize]])
    slantRangeImage.createPoly2D()

    #creat images
    latImage = isceobj.createImage()
    latImage.initImage(latFile, 'write', referenceTrack.numberOfSamples, 'DOUBLE')
    latImage.createImage()

    lonImage = isceobj.createImage()
    lonImage.initImage(lonFile, 'write', referenceTrack.numberOfSamples, 'DOUBLE')
    lonImage.createImage()

    losImage = isceobj.createImage()
    losImage.initImage(losFile, 'write', referenceTrack.numberOfSamples, 'FLOAT', bands=2, scheme='BIL')
    losImage.setCaster('write', 'DOUBLE')
    losImage.createImage()

    heightImage = isceobj.createImage()
    heightImage.initImage(hgtFile, 'write', referenceTrack.numberOfSamples, 'DOUBLE')
    heightImage.createImage()

    demImage = isceobj.createDemImage()
    demImage.load(demFile + '.xml')
    demImage.setCaster('read', 'FLOAT')
    demImage.createImage()

    #compute a few things
    t0 = referenceTrack.sensingStart + datetime.timedelta(seconds=(numberAzimuthLooks-1.0)/2.0*referenceTrack.azimuthLineInterval)
    orb = referenceTrack.orbit
    pegHdg = np.radians( orb.getENUHeading(t0))
    elp = Planet(pname='Earth').ellipsoid

    #call gpu topo
    topo = PyTopozero()
    topo.set_firstlat(demImage.getFirstLatitude())
    topo.set_firstlon(demImage.getFirstLongitude())
    topo.set_deltalat(demImage.getDeltaLatitude())
    topo.set_deltalon(demImage.getDeltaLongitude())
    topo.set_major(elp.a)
    topo.set_eccentricitySquared(elp.e2)
    topo.set_rSpace(numberRangeLooks * referenceTrack.rangePixelSize)
    topo.set_r0(referenceTrack.startingRange + (numberRangeLooks-1.0)/2.0*referenceTrack.rangePixelSize)
    topo.set_pegHdg(pegHdg)
    topo.set_prf(1.0 / (numberAzimuthLooks*referenceTrack.azimuthLineInterval))
    topo.set_t0(DTU.seconds_since_midnight(t0))
    topo.set_wvl(referenceTrack.radarWavelength)
    topo.set_thresh(.05)
    topo.set_demAccessor(demImage.getImagePointer())
    topo.set_dopAccessor(polyDoppler.getPointer())
    topo.set_slrngAccessor(slantRangeImage.getPointer())
    topo.set_latAccessor(latImage.getImagePointer())
    topo.set_lonAccessor(lonImage.getImagePointer())
    topo.set_losAccessor(losImage.getImagePointer())
    topo.set_heightAccessor(heightImage.getImagePointer())
    topo.set_incAccessor(0)
    topo.set_maskAccessor(0)
    topo.set_numIter(25)
    topo.set_idemWidth(demImage.getWidth())
    topo.set_idemLength(demImage.getLength())
    topo.set_ilrl(pointingDirection[referenceTrack.pointingDirection])
    topo.set_extraIter(10)
    topo.set_length(referenceTrack.numberOfLines)
    topo.set_width(referenceTrack.numberOfSamples)
    topo.set_nRngLooks(1)
    topo.set_nAzLooks(1)
    topo.set_demMethod(5) # BIQUINTIC METHOD
    topo.set_orbitMethod(0) # HERMITE

    # Need to simplify orbit stuff later
    nvecs = len(orb._stateVectors)
    topo.set_orbitNvecs(nvecs)
    topo.set_orbitBasis(1) # Is this ever different?
    topo.createOrbit() # Initializes the empty orbit to the right allocated size
    count = 0
    for sv in orb._stateVectors:
        td = DTU.seconds_since_midnight(sv.getTime())
        pos = sv.getPosition()
        vel = sv.getVelocity()
        topo.set_orbitVector(count,td,pos[0],pos[1],pos[2],vel[0],vel[1],vel[2])
        count += 1

    topo.runTopo()

    #tidy up
    latImage.addDescription('Pixel-by-pixel latitude in degrees.')
    latImage.finalizeImage()
    latImage.renderHdr()

    lonImage.addDescription('Pixel-by-pixel longitude in degrees.')
    lonImage.finalizeImage()
    lonImage.renderHdr()

    heightImage.addDescription('Pixel-by-pixel height in meters.')
    heightImage.finalizeImage()
    heightImage.renderHdr()

    descr = '''Two channel Line-Of-Sight geometry image (all angles in degrees). Represents vector drawn from target to platform.
            Channel 1: Incidence angle measured from vertical at target (always +ve).
            Channel 2: Azimuth angle measured from North in Anti-clockwise direction.'''
    losImage.setImageType('bil')
    losImage.addDescription(descr)
    losImage.finalizeImage()
    losImage.renderHdr()

    demImage.finalizeImage()

    if slantRangeImage:
        try:
            slantRangeImage.finalizeImage()
        except:
            pass


