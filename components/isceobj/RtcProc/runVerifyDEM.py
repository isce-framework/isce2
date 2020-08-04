#
# Author: Piyush Agram
# Copyright 2016
#

import logging
import isceobj
import mroipac
import os
import numpy as np
from isceobj.Util.decorators import use_api

logger = logging.getLogger('isce.insar.VerifyDEM')

def runVerifyDEM(self):
    '''
    Make sure that a DEM is available for processing the given data.
    '''
    
    self.demStitcher.noFilling = False

    ###If provided in the input XML file
    if self.demFilename not in ['',None]:
        demimg = isceobj.createDemImage()
        demimg.load(self.demFilename + '.xml')
        if not os.path.exists(self.demFilename + '.vrt'):
            demimg.renderVRT()

        if demimg.reference.upper() == 'EGM96':
            wgsdemname  = self.demFilename + '.wgs84'

            if os.path.exists(wgsdemname) and os.path.exists(wgsdemname + '.xml'):
                demimg = isceobj.createDemImage()
                demimg.load(wgsdemname + '.xml')

                if demimg.reference.upper() == 'EGM96':
                    raise Exception('WGS84 version of dem found by reference set to EGM96')

            else:
                demimg = self.demStitcher.correct(demimg)

        elif demimg.reference.upper() != 'WGS84':
            raise Exception('Unknown reference system for DEM: {0}'.format(demimg.reference))

    else:

        refPol = self._grd.polarizations[0]
        
        reference = self._grd.loadProduct( os.path.join(self._grd.outputFolder, 'beta_{0}.xml'.format(refPol)))
        bbox = reference.getBbox()

        ####Truncate to integers
        tbox = [np.floor(bbox[0]), np.ceil(bbox[1]),
                np.floor(bbox[2]), np.ceil(bbox[3])]


        filename = self.demStitcher.defaultName(tbox)
        wgsfilename = filename + '.wgs84'

        ####Check if WGS84 file exists
        if os.path.exists(wgsfilename) and os.path.exists(wgsfilename + '.xml'):
            demimg = isceobj.createDemImage()
            demimg.load(wgsfilename + '.xml')

            if not os.path.exists(wgsfilename + '.vrt'):
                demimg.renderVRT()

        ####Check if EGM96 file exists
        elif os.path.exists(filename) and os.path.exists(filename + '.xml'):
            inimg = isceobj.createDemImage()
            inimg.load(filename + '.xml')

            if not os.path.exists(filename + '.xml'):
                inimg.renderVRT()

            demimg = self.demStitcher.correct(inimg)

        else:
            stitchOk = self.demStitcher.stitch(tbox[0:2], tbox[2:4])

            if not stitchOk:
                logger.error("Cannot form the DEM for the region of interest. If you have one, set the appropriate DEM component in the input file.")
                raise Exception

            inimg = isceobj.createDemImage()
            inimg.load(filename + '.xml')
            if not os.path.exists(filename):
                inimg.renderVRT()

            demimg = self.demStitcher.correct(inimg)

        #get water mask
#        self.runCreateWbdMask(info)

    return demimg.filename
