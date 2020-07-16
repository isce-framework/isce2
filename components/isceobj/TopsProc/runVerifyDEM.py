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

logger = logging.getLogger('isce.insar.VerifyDEM')

class INFO:
    def __init__(self, snwe):
        self.extremes = snwe
    def getExtremes(self, x):
        return self.extremes

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

        swathList = self._insar.getValidSwathList(self.swaths)
        bboxes = []
        for swath in swathList:
            if self._insar.numberOfCommonBursts[swath-1] > 0:
                reference = self._insar.loadProduct( os.path.join(self._insar.referenceSlcProduct, 'IW{0}.xml'.format(swath)))

                secondary  = self._insar.loadProduct( os.path.join(self._insar.secondarySlcProduct,  'IW{0}.xml'.format(swath)))

                ####Merges orbit as needed for multi-stitched frames
                mOrb = self._insar.getMergedOrbit([reference])
                sOrb = self._insar.getMergedOrbit([secondary])

                mbox = reference.getBbox()
                sbox = secondary.getBbox()

                ####Union of bounding boxes
                bbox = [min(mbox[0], sbox[0]), max(mbox[1], sbox[1]),
                        min(mbox[2], sbox[2]), max(mbox[3], sbox[3])]

                bboxes.append(bbox)


        if len(bboxes) == 0 :
            raise Exception('Something went wrong in determining bboxes')

        else:
            bbox = [min([x[0] for x in bboxes]),
                    max([x[1] for x in bboxes]),
                    min([x[2] for x in bboxes]),
                    max([x[3] for x in bboxes])]


        ####Truncate to integers
        tbox = [np.floor(bbox[0]), np.ceil(bbox[1]),
                np.floor(bbox[2]), np.ceil(bbox[3])]

        #EMG
        info = INFO(tbox)
        self.useZeroTiles = True
        DemImageLib.createDem(tbox, info, self, self.demStitcher,
            self.useHighResolutionDemOnly, self.useZeroTiles)

        # createDem puts the dem image in self. Put a reference in
        # local variable demimg to return the filename in the same
        # way as done in the "if" clause above
        demimg = self.demImage

    return demimg.filename
