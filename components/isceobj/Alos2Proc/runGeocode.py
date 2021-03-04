#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import glob
import logging
import numpy as np

import isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import getBboxGeo

logger = logging.getLogger('isce.alos2insar.runGeocode')

def runGeocode(self):
    '''geocode final products
    '''
    if hasattr(self, 'doInSAR'):
        if not self.doInSAR:
            return

    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    referenceTrack = self._insar.loadTrack(reference=True)
    #secondaryTrack = self._insar.loadTrack(reference=False)

    demFile = os.path.abspath(self._insar.demGeo)

    insarDir = 'insar'
    os.makedirs(insarDir, exist_ok=True)
    os.chdir(insarDir)

    #compute bounding box for geocoding
    if self.bbox == None:
        bbox = getBboxGeo(referenceTrack)
    else:
        bbox = self.bbox
    catalog.addItem('geocode bounding box', bbox, 'runGeocode')

    if self.geocodeList == None:
        geocodeList = [self._insar.unwrappedInterferogram,
        self._insar.unwrappedMaskedInterferogram,
        self._insar.multilookCoherence,
        self._insar.multilookLos]
        if self.doIon:
            geocodeList.append(self._insar.multilookIon)
    else:
        geocodeList = []
        for xxx in self.geocodeList:
            geocodeList += glob.glob(xxx)

    numberRangeLooks = self._insar.numberRangeLooks1 * self._insar.numberRangeLooks2
    numberAzimuthLooks = self._insar.numberAzimuthLooks1 * self._insar.numberAzimuthLooks2

    for inputFile in geocodeList:
        if self.geocodeInterpMethod == None:
            img = isceobj.createImage()
            img.load(inputFile + '.xml')
            if img.dataType.upper() == 'CFLOAT':
                interpMethod = 'sinc'
            else:
                interpMethod = 'bilinear'
        else:
            interpMethod = self.geocodeInterpMethod.lower()

        geocode(referenceTrack, demFile, inputFile, bbox, numberRangeLooks, numberAzimuthLooks, interpMethod, 0, 0)


    os.chdir('../')

    catalog.printToLog(logger, "runGeocode")
    self._insar.procDoc.addAllFromCatalog(catalog)


def geocode(track, demFile, inputFile, bbox, numberRangeLooks, numberAzimuthLooks, interpMethod, topShift, leftShift, addMultilookOffset=True):
    import datetime
    from zerodop.geozero import createGeozero
    from isceobj.Planet.Planet import Planet

    pointingDirection = {'right': -1, 'left' :1}

    demImage = isceobj.createDemImage()
    demImage.load(demFile + '.xml')
    demImage.setAccessMode('read')

    inImage = isceobj.createImage()
    inImage.load(inputFile + '.xml')
    inImage.setAccessMode('read')

    planet = Planet(pname='Earth')

    topo = createGeozero()
    topo.configure()
    topo.slantRangePixelSpacing = numberRangeLooks * track.rangePixelSize
    topo.prf = 1.0 / (numberAzimuthLooks*track.azimuthLineInterval)
    topo.radarWavelength = track.radarWavelength
    topo.orbit = track.orbit
    topo.width = inImage.width
    topo.length = inImage.length
    topo.wireInputPort(name='dem', object=demImage)
    topo.wireInputPort(name='planet', object=planet)
    topo.wireInputPort(name='tobegeocoded', object=inImage)
    topo.numberRangeLooks = 1
    topo.numberAzimuthLooks = 1
    topo.lookSide = pointingDirection[track.pointingDirection]
    sensingStart = track.sensingStart + datetime.timedelta(seconds=topShift*track.azimuthLineInterval)
    rangeFirstSample = track.startingRange + leftShift * track.rangePixelSize
    if addMultilookOffset:
        sensingStart += datetime.timedelta(seconds=(numberAzimuthLooks-1.0)/2.0*track.azimuthLineInterval)
        rangeFirstSample += (numberRangeLooks-1.0)/2.0*track.rangePixelSize
    topo.setSensingStart(sensingStart)
    topo.rangeFirstSample = rangeFirstSample
    topo.method=interpMethod
    topo.demCropFilename = 'crop.dem'
    #looks like this does not work
    #topo.geoFilename = outputName
    topo.dopplerCentroidCoeffs = [0.]
    #snwe list <class 'list'>
    topo.snwe = bbox

    topo.geocode()

    print('South: ', topo.minimumGeoLatitude)
    print('North: ', topo.maximumGeoLatitude)
    print('West:  ', topo.minimumGeoLongitude)
    print('East:  ', topo.maximumGeoLongitude)
    
    return
