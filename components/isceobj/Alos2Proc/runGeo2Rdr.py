#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging

import isceobj

logger = logging.getLogger('isce.alos2insar.runGeo2Rdr')

def runGeo2Rdr(self):
    '''compute range and azimuth offsets
    '''
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    slaveTrack = self._insar.loadTrack(master=False)

    insarDir = 'insar'
    if not os.path.exists(insarDir):
        os.makedirs(insarDir)
    os.chdir(insarDir)


    hasGPU= self.useGPU and self._insar.hasGPU()
    if hasGPU:
        geo2RdrGPU(slaveTrack, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, 
            self._insar.latitude, self._insar.longitude, self._insar.height, self._insar.rangeOffset, self._insar.azimuthOffset)
    else:
        geo2RdrCPU(slaveTrack, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, 
            self._insar.latitude, self._insar.longitude, self._insar.height, self._insar.rangeOffset, self._insar.azimuthOffset)

    os.chdir('../')

    catalog.printToLog(logger, "runGeo2Rdr")
    self._insar.procDoc.addAllFromCatalog(catalog)


def geo2RdrCPU(slaveTrack, numberRangeLooks, numberAzimuthLooks, latFile, lonFile, hgtFile, rangeOffsetFile, azimuthOffsetFile):
    import datetime
    from zerodop.geo2rdr import createGeo2rdr
    from isceobj.Planet.Planet import Planet

    pointingDirection = {'right': -1, 'left' :1}

    latImage = isceobj.createImage()
    latImage.load(latFile + '.xml')
    latImage.setAccessMode('read')

    lonImage = isceobj.createImage()
    lonImage.load(lonFile + '.xml')
    lonImage.setAccessMode('read')

    demImage = isceobj.createDemImage()
    demImage.load(hgtFile + '.xml')
    demImage.setAccessMode('read')

    planet = Planet(pname='Earth')

    topo = createGeo2rdr()
    topo.configure()
    #set parameters
    topo.slantRangePixelSpacing = numberRangeLooks * slaveTrack.rangePixelSize
    topo.prf = 1.0 / (numberAzimuthLooks*slaveTrack.azimuthLineInterval)
    topo.radarWavelength = slaveTrack.radarWavelength
    topo.orbit = slaveTrack.orbit
    topo.width = slaveTrack.numberOfSamples
    topo.length = slaveTrack.numberOfLines
    topo.demLength = demImage.length
    topo.demWidth = demImage.width
    topo.wireInputPort(name='planet', object=planet)
    topo.numberRangeLooks = 1 #
    topo.numberAzimuthLooks = 1 # must be set to be 1
    topo.lookSide = pointingDirection[slaveTrack.pointingDirection]
    topo.setSensingStart(slaveTrack.sensingStart + datetime.timedelta(seconds=(numberAzimuthLooks-1.0)/2.0*slaveTrack.azimuthLineInterval))
    topo.rangeFirstSample = slaveTrack.startingRange + (numberRangeLooks-1.0)/2.0*slaveTrack.rangePixelSize
    topo.dopplerCentroidCoeffs = [0.] # we are using zero doppler geometry
    #set files
    topo.latImage = latImage
    topo.lonImage = lonImage
    topo.demImage = demImage
    topo.rangeOffsetImageName = rangeOffsetFile
    topo.azimuthOffsetImageName = azimuthOffsetFile
    #run it
    topo.geo2rdr()

    return


def geo2RdrGPU(slaveTrack, numberRangeLooks, numberAzimuthLooks, latFile, lonFile, hgtFile, rangeOffsetFile, azimuthOffsetFile):
    '''
    currently we cannot set left/right looking.
    works for right looking, but left looking probably not supported.
    '''

    import datetime
    from zerodop.GPUgeo2rdr.GPUgeo2rdr import PyGeo2rdr
    from isceobj.Planet.Planet import Planet
    from iscesys import DateTimeUtil as DTU

    latImage = isceobj.createImage()
    latImage.load(latFile + '.xml')
    latImage.setAccessMode('READ')
    latImage.createImage()

    lonImage = isceobj.createImage()
    lonImage.load(lonFile + '.xml')
    lonImage.setAccessMode('READ')
    lonImage.createImage()

    demImage = isceobj.createImage()
    demImage.load(hgtFile + '.xml')
    demImage.setAccessMode('READ')
    demImage.createImage()

    #####Run Geo2rdr
    planet = Planet(pname='Earth')
    grdr = PyGeo2rdr()

    grdr.setRangePixelSpacing(numberRangeLooks * slaveTrack.rangePixelSize)
    grdr.setPRF(1.0 / (numberAzimuthLooks*slaveTrack.azimuthLineInterval))
    grdr.setRadarWavelength(slaveTrack.radarWavelength)

    #CHECK IF THIS WORKS!!!
    grdr.createOrbit(0, len(slaveTrack.orbit.stateVectors.list))
    count = 0
    for sv in slaveTrack.orbit.stateVectors.list:
        td = DTU.seconds_since_midnight(sv.getTime())
        pos = sv.getPosition()
        vel = sv.getVelocity()

        grdr.setOrbitVector(count, td, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2])
        count += 1

    grdr.setOrbitMethod(0)
    grdr.setWidth(slaveTrack.numberOfSamples)
    grdr.setLength(slaveTrack.numberOfLines)
    grdr.setSensingStart(DTU.seconds_since_midnight(slaveTrack.sensingStart + datetime.timedelta(seconds=(numberAzimuthLooks-1.0)/2.0*slaveTrack.azimuthLineInterval)))
    grdr.setRangeFirstSample(slaveTrack.startingRange + (numberRangeLooks-1.0)/2.0*slaveTrack.rangePixelSize)
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
    rangeOffsetImage.setFilename(rangeOffsetFile)
    rangeOffsetImage.setAccessMode('write')
    rangeOffsetImage.setDataType('FLOAT')
    rangeOffsetImage.setCaster('write', 'DOUBLE')
    rangeOffsetImage.setWidth(demImage.width)
    rangeOffsetImage.createImage()
    
    azimuthOffsetImage = isceobj.createImage()
    azimuthOffsetImage.setFilename(azimuthOffsetFile)
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
