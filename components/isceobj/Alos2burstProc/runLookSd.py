#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging

import isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
from contrib.alos2proc.alos2proc import look
from isceobj.Alos2Proc.Alos2ProcPublic import waterBodyRadar

logger = logging.getLogger('isce.alos2burstinsar.runLookSd')

def runLookSd(self):
    '''take looks
    '''
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    #referenceTrack = self._insar.loadTrack(reference=True)
    #secondaryTrack = self._insar.loadTrack(reference=False)
    wbdFile = os.path.abspath(self._insar.wbd)

    sdDir = 'sd'
    os.makedirs(sdDir, exist_ok=True)
    os.chdir(sdDir)

    sd = isceobj.createImage()
    sd.load(self._insar.interferogramSd[0]+'.xml')
    width = sd.width
    length = sd.length
    width2 = int(width / self._insar.numberRangeLooksSd)
    length2 = int(length / self._insar.numberAzimuthLooksSd)

    if not ((self._insar.numberRangeLooksSd == 1) and (self._insar.numberAzimuthLooksSd == 1)):
        #take looks
        for sd, sdMultilook in zip(self._insar.interferogramSd, self._insar.multilookInterferogramSd):
            look(sd, sdMultilook, width, self._insar.numberRangeLooksSd, self._insar.numberAzimuthLooksSd, 4, 0, 1)
            create_xml(sdMultilook, width2, length2, 'int')
        look(os.path.join('../insar', self._insar.latitude), self._insar.multilookLatitudeSd, width, 
            self._insar.numberRangeLooksSd, self._insar.numberAzimuthLooksSd, 3, 0, 1)
        look(os.path.join('../insar', self._insar.longitude), self._insar.multilookLongitudeSd, width, 
            self._insar.numberRangeLooksSd, self._insar.numberAzimuthLooksSd, 3, 0, 1)
        create_xml(self._insar.multilookLatitudeSd, width2, length2, 'double')
        create_xml(self._insar.multilookLongitudeSd, width2, length2, 'double')
        #water body
        waterBodyRadar(self._insar.multilookLatitudeSd, self._insar.multilookLongitudeSd, wbdFile, self._insar.multilookWbdOutSd)

    os.chdir('../')

    catalog.printToLog(logger, "runLookSd")
    self._insar.procDoc.addAllFromCatalog(catalog)


