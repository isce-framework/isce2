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
import datetime

logger = logging.getLogger('isce.insar.VerifyDEM')

class INFO:
    def __init__(self, snwe):
        self.extremes = snwe
    def getExtremes(self, x):
        return self.extremes


def getBbox(frame, zrange=[-500., 9000.],geom='zero doppler', margin=0.05):
    '''
    Get bounding box.
    '''
    from isceobj.Util.Poly2D import Poly2D

    #### Reference box
    r0 = frame.startingRange
    rmax = frame.getFarRange()
    t0 = frame.getSensingStart()
    t1 = frame.getSensingStop()
    tdiff = (t1-t0).total_seconds()
    wvl = frame.instrument.getRadarWavelength()
    lookSide = frame.instrument.platform.pointingDirection
    tarr = []

    for ind in range(11):
        tarr.append(t0 + datetime.timedelta(seconds=ind*tdiff/10.0))

    if geom.lower().startswith('native'):
        coeff = frame._dopplerVsPixel
        doppler = Poly2D()
        doppler._meanRange = r0
        doppler._normRange = frame.instrument.rangePixelSize
        doppler.initPoly(azimuthOrder=0, rangeOrder=len(coeff)-1, coeffs=[coeff])
        print('Using native doppler information for bbox estimation')
    else:
        doppler = Poly2D()
        doppler.initPoly(azimuthOrder=0, rangeOrder=0, coeffs=[[0.]])

    llh = []

    for z in zrange:
        for taz in tarr:
            for rng in [r0, rmax]:
                pt = frame.orbit.rdr2geo(taz, rng, doppler=doppler, height=z,
                                        wvl=wvl, side=lookSide)
                ###If nan, use nadir point
                if np.sum(np.isnan(pt)):
                    sv = frame.orbit.interpolateOrbit(taz, method='hermite')
                    pt = frame.ellipsoid.xyz_to_llh(sv.getPosition())

                llh.append(pt)

    llh = np.array(llh)
    minLat = np.min(llh[:,0]) - margin
    maxLat = np.max(llh[:,0]) + margin
    minLon = np.min(llh[:,1]) - margin
    maxLon = np.max(llh[:,1]) + margin

    return [minLat, maxLat, minLon, maxLon]
    


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

        reference = self._insar.loadProduct(self._insar.referenceSlcCropProduct)
        secondary  = self._insar.loadProduct(self._insar.secondarySlcCropProduct)

        bboxes = []

        ###Add region of interest for good measure
        if self.regionOfInterest is not None:
            bboxes.append(self.regionOfInterest)

        if self.heightRange is not None:
            zrange = self.heightRange
        else:
            zrange = [-500., 9000.]
        mbox = getBbox(reference, geom=self._insar.referenceGeometrySystem,
                               zrange = zrange)

        sbox = getBbox(secondary, geom=self._insar.secondaryGeometrySystem,
                                zrange = zrange)

        bboxes.append(mbox)
        bboxes.append(sbox)

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
