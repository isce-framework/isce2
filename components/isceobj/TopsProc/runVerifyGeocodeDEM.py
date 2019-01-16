#
# Author: Piyush Agram
# Copyright 2016
#

import logging
import isceobj
import mroipac
from isceobj.Util.ImageUtil import DemImageLib
import os
import numpy as np

logger = logging.getLogger('isce.insar.VerifyGeocodeDEM')

class INFO:
    def __init__(self, snwe):
        self.extremes = snwe
    def getExtremes(self, x):
        return self.extremes

def runVerifyGeocodeDEM(self):
    '''
    Make sure that a DEM is available for processing the given data.
    '''

    self.demStitcher.noFilling = False

    ###If provided in the input XML file
    if self.geocodeDemFilename not in ['',None]:
        demimg = isceobj.createDemImage()
        demimg.load(self.geocodeDemFilename + '.xml')
        if not os.path.exists(self.geocodeDemFilename + '.vrt'):
            demimg.renderVRT()

        if demimg.reference.upper() == 'EGM96':
            wgsdemname  = self.geocodeDemFilename + '.wgs84'

            if os.path.exists(wgsdemname) and os.path.exists(wgsdemname + '.xml'):
                demimg = isceobj.createDemImage()
                demimg.load(wgsdemname + '.xml')

                if demimg.reference.upper() == 'EGM96':
                    raise Exception('WGS84 version of dem found by reference set to EGM96')

            else:
                demimg = self.demStitcher.correct(demimg)

        elif demimg.reference.upper() != 'WGS84':
            raise Exception('Unknown reference system for DEM: {0}'.format(demimg.reference))

    ##Fall back to DEM used for running topo
    else:

        self.geocodeDemFilename = self.verifyDEM()
        demimg = isceobj.createDemImage()
        demimg.load(self.geocodeDemFilename + '.xml')

        if demimg.reference.upper() == 'EGM96':
            raise Exception('EGM96 DEM returned by verifyDEM')


    return demimg.filename
