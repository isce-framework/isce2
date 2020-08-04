#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging
import numpy as np

import isceobj
from isceobj.Alos2Proc.runGeocode import geocode
from isceobj.Alos2Proc.Alos2ProcPublic import getBboxGeo

logger = logging.getLogger('isce.alos2insar.runGeocodeOffset')

def runGeocodeOffset(self):
    '''geocode offset fied
    '''
    if not self.doDenseOffset:
        return
    if not ((self._insar.modeCombination == 0) or (self._insar.modeCombination == 1)):
        return

    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    #use original track object to determine bbox
    if self.bbox == None:
        referenceTrack = self._insar.loadTrack(reference=True)
        bbox = getBboxGeo(referenceTrack)
    else:
        bbox = self.bbox
    catalog.addItem('geocode bounding box', bbox, 'runGeocodeOffset')

    demFile = os.path.abspath(self._insar.demGeo)

    denseOffsetDir = 'dense_offset'
    os.makedirs(denseOffsetDir, exist_ok=True)
    os.chdir(denseOffsetDir)

    referenceTrack = self._insar.loadProduct(self._insar.referenceTrackParameter)
    #secondaryTrack = self._insar.loadProduct(self._insar.secondaryTrackParameter)

#########################################################################################
    #compute bounding box for geocoding
    #if self.bbox == None:
    #    bbox = getBboxGeo(referenceTrack)
    #else:
    #    bbox = self.bbox
    #catalog.addItem('geocode bounding box', bbox, 'runGeocodeOffset')

    geocodeList = [self._insar.denseOffset, self._insar.denseOffsetSnr]
    if self.doOffsetFiltering:
        geocodeList.append(self._insar.denseOffsetFilt)

    for inputFile in geocodeList:
        interpMethod = 'nearest'
        geocode(referenceTrack, demFile, inputFile, bbox, self.offsetSkipWidth, self.offsetSkipHeight, interpMethod, self._insar.offsetImageTopoffset, self._insar.offsetImageLeftoffset, addMultilookOffset=False)
#########################################################################################

    os.chdir('../')
    catalog.printToLog(logger, "runGeocodeOffset")
    self._insar.procDoc.addAllFromCatalog(catalog)
