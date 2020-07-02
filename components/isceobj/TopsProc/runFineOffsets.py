#
# Author: Piyush Agram
# Copyright 2016
#


import numpy as np 
import os
import isceobj
import datetime
import sys
import logging

logger = logging.getLogger('isce.topsinsar.fineoffsets')

def runGeo2rdrCPU(info, rdict, misreg_az=0.0, misreg_rg=0.0):
    from zerodop.geo2rdr import createGeo2rdr
    from isceobj.Planet.Planet import Planet

    latImage = isceobj.createImage()
    latImage.load(rdict['lat'] + '.xml')
    latImage.setAccessMode('READ')
    latImage.createImage()

    lonImage = isceobj.createImage()
    lonImage.load(rdict['lon'] + '.xml')
    lonImage.setAccessMode('READ')
    lonImage.createImage()

    demImage = isceobj.createImage()
    demImage.load(rdict['hgt'] + '.xml')
    demImage.setAccessMode('READ')
    demImage.createImage()

    delta = datetime.timedelta(seconds=misreg_az)
    logger.info('Additional time offset applied in geo2rdr: {0} secs'.format(misreg_az))
    logger.info('Additional range offset applied in geo2rdr: {0} m'.format(misreg_rg))


    #####Run Geo2rdr
    planet = Planet(pname='Earth')
    grdr = createGeo2rdr()
    grdr.configure()

    grdr.slantRangePixelSpacing = info.rangePixelSize
    grdr.prf = 1.0 / info.azimuthTimeInterval
    grdr.radarWavelength = info.radarWavelength
    grdr.orbit = info.orbit
    grdr.width = info.numberOfSamples
    grdr.length = info.numberOfLines
    grdr.demLength = demImage.getLength()
    grdr.demWidth = demImage.getWidth()
    grdr.wireInputPort(name='planet', object=planet)
    grdr.numberRangeLooks = 1
    grdr.numberAzimuthLooks = 1
    grdr.lookSide = -1  
    grdr.setSensingStart(info.sensingStart - delta)
    grdr.rangeFirstSample = info.startingRange - misreg_rg
    grdr.dopplerCentroidCoeffs = [0.]  ###Zero doppler

    grdr.rangeOffsetImageName = rdict['rangeOffName']
    grdr.azimuthOffsetImageName = rdict['azOffName']
    grdr.demImage = demImage
    grdr.latImage = latImage
    grdr.lonImage = lonImage

    grdr.geo2rdr()

    return

def runGeo2rdrGPU(info, rdict, misreg_az=0.0, misreg_rg=0.0):
    from zerodop.GPUgeo2rdr.GPUgeo2rdr import PyGeo2rdr
    from isceobj.Planet.Planet import Planet
    from iscesys import DateTimeUtil as DTU

    latImage = isceobj.createImage()
    latImage.load(rdict['lat'] + '.xml')
    latImage.setAccessMode('READ')
    latImage.createImage()

    lonImage = isceobj.createImage()
    lonImage.load(rdict['lon'] + '.xml')
    lonImage.setAccessMode('READ')
    lonImage.createImage()

    demImage = isceobj.createImage()
    demImage.load(rdict['hgt'] + '.xml')
    demImage.setAccessMode('READ')
    demImage.createImage()

    delta = datetime.timedelta(seconds=misreg_az)
    logger.info('Additional time offset applied in geo2rdr: {0} secs'.format(misreg_az))
    logger.info('Additional range offset applied in geo2rdr: {0} m'.format(misreg_rg))


    #####Run Geo2rdr
    planet = Planet(pname='Earth')
    grdr = PyGeo2rdr()

    grdr.setRangePixelSpacing(info.rangePixelSize)
    grdr.setPRF(1.0 / info.azimuthTimeInterval)
    grdr.setRadarWavelength(info.radarWavelength)

    grdr.createOrbit(0, len(info.orbit.stateVectors.list))
    count = 0
    for sv in info.orbit.stateVectors.list:
        td = DTU.seconds_since_midnight(sv.getTime())
        pos = sv.getPosition()
        vel = sv.getVelocity()

        grdr.setOrbitVector(count, td, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2])
        count += 1

    grdr.setOrbitMethod(0)
    grdr.setWidth(info.numberOfSamples)
    grdr.setLength(info.numberOfLines)
    grdr.setSensingStart(DTU.seconds_since_midnight(info.sensingStart -delta))
    grdr.setRangeFirstSample(info.startingRange - misreg_rg)
    grdr.setNumberRangeLooks(1)
    grdr.setNumberAzimuthLooks(1)
    grdr.setEllipsoidMajorSemiAxis(planet.ellipsoid.a)
    grdr.setEllipsoidEccentricitySquared(planet.ellipsoid.e2)


    grdr.createPoly(0, 0., 1.)
    grdr.setPolyCoeff(0, 0.)

    grdr.setDemLength(demImage.getLength())
    grdr.setDemWidth(demImage.getWidth())
    grdr.setBistaticFlag(0)

    rangeOffsetImage = isceobj.createImage()
    rangeOffsetImage.setFilename(rdict['rangeOffName'])
    rangeOffsetImage.setAccessMode('write')
    rangeOffsetImage.setDataType('FLOAT')
    rangeOffsetImage.setCaster('write', 'DOUBLE')
    rangeOffsetImage.setWidth(demImage.width)
    rangeOffsetImage.createImage()
    
    azimuthOffsetImage = isceobj.createImage()
    azimuthOffsetImage.setFilename(rdict['azOffName'])
    azimuthOffsetImage.setAccessMode('write')
    azimuthOffsetImage.setDataType('FLOAT')
    azimuthOffsetImage.setCaster('write', 'DOUBLE')
    azimuthOffsetImage.setWidth(demImage.width)
    azimuthOffsetImage.createImage()
    
    grdr.setLatAccessor(latImage.getImagePointer())
    grdr.setLonAccessor(lonImage.getImagePointer())
    grdr.setHgtAccessor(demImage.getImagePointer())
    grdr.setAzAccessor(0)
    grdr.setRgAccessor(0)
    grdr.setAzOffAccessor(azimuthOffsetImage.getImagePointer())
    grdr.setRgOffAccessor(rangeOffsetImage.getImagePointer())

    grdr.geo2rdr()

    rangeOffsetImage.finalizeImage()
    rangeOffsetImage.renderHdr()
    
    azimuthOffsetImage.finalizeImage()
    azimuthOffsetImage.renderHdr()
    latImage.finalizeImage()
    lonImage.finalizeImage()
    demImage.finalizeImage()

    return

    pass

def runFineOffsets(self):
    '''
    Estimate offsets using geometry
    '''

    hasGPU = self.useGPU and self._insar.hasGPU()
    
    if hasGPU:
        runGeo2rdr = runGeo2rdrGPU
    else:
        runGeo2rdr = runGeo2rdrCPU


    ##Catalog
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)

    misreg_az = self._insar.secondaryTimingCorrection
    catalog.addItem('Initial secondary azimuth timing correction', misreg_az, 'fineoff')

    misreg_rg = self._insar.secondaryRangeCorrection
    catalog.addItem('Initial secondary range timing correction', misreg_rg, 'fineoff')

    swathList = self._insar.getValidSwathList(self.swaths)

    for swath in swathList:

        ##Load secondary metadata
        secondary = self._insar.loadProduct( os.path.join(self._insar.secondarySlcProduct, 'IW{0}.xml'.format(swath)))

        ###Offsets output directory
        outdir = os.path.join(self._insar.fineOffsetsDirname, 'IW{0}'.format(swath))

        os.makedirs(outdir, exist_ok=True)


        ###Burst indices w.r.t reference
        minBurst, maxBurst = self._insar.commonReferenceBurstLimits(swath-1)
        geomDir = os.path.join(self._insar.geometryDirname, 'IW{0}'.format(swath))

        if minBurst == maxBurst:
            print('Skipping processing of swath {0}'.format(swath))
            continue


        secondaryBurstStart = self._insar.commonBurstStartSecondaryIndex[swath-1]

        catalog.addItem('Number of bursts - IW{0}'.format(swath), maxBurst - minBurst, 'fineoff')

        for mBurst in range(minBurst, maxBurst):

            ###Corresponding secondary burst
            sBurst = secondaryBurstStart + (mBurst - minBurst)
            burst = secondary.bursts[sBurst]

            logger.info('IW{3} - Burst {1} of reference matched with Burst {2} of secondary'.format(mBurst-minBurst, mBurst, sBurst, swath))
            ####Generate offsets for top burst
            rdict = {'lat': os.path.join(geomDir,'lat_%02d.rdr'%(mBurst+1)),
                     'lon': os.path.join(geomDir,'lon_%02d.rdr'%(mBurst+1)),
                     'hgt': os.path.join(geomDir,'hgt_%02d.rdr'%(mBurst+1)),
                     'rangeOffName': os.path.join(outdir, 'range_%02d.off'%(mBurst+1)),
                     'azOffName': os.path.join(outdir, 'azimuth_%02d.off'%(mBurst+1))}
        
            runGeo2rdr(burst, rdict, misreg_az=misreg_az, misreg_rg=misreg_rg)

