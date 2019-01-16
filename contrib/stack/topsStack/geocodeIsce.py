#!/usr/bin/env python3
# Author: Piyush Agram
# Copyright 2016
# Heresh Fattahi, adopted for stack processor

import argparse
import isce
import isceobj
import logging
from zerodop.geozero import createGeozero
from stdproc.rectify.geocode.Geocodable import Geocodable
import isceobj
import iscesys
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from isceobj.Planet.Planet import Planet
from isceobj.Orbit.Orbit import Orbit
import os
import datetime
import s1a_isce_utils as ut
from baselineGrid import getMergedOrbit

logger = logging.getLogger('isce.topsinsar.runGeocode')
posIndx = 1

def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser( description='Create DEM simulation for merged images')
    parser.add_argument('-f', '--filelist', dest='prodlist', type=str, required=True,
            help='Input file to be geocoded')
    parser.add_argument('-b', '--bbox', dest='bbox', type=str, default='',
            help='Bounding box (SNWE)')
    parser.add_argument('-d', '--demfilename', dest='demfilename', type=str, required=True,
            help='DEM filename')

    parser.add_argument('-m', '--master', dest='master', type=str, required=True,
            help='Directory with master acquisition')

    parser.add_argument('-s', '--slave', dest='slave', type=str, required=True,
            help='Directory with slave acquisition')

    parser.add_argument('-r', '--numberRangeLooks', dest='numberRangeLooks', type=int, required=True,
            help='number range looks')

    parser.add_argument('-a', '--numberAzimuthLooks', dest='numberAzimuthLooks', type=int, required=True,
            help='number range looks')

    return parser

def cmdLineParse(iargs = None):
    '''
    Command line parser.
    '''

    parser = createParser()
    inps =  parser.parse_args(args = iargs)

    return inps


def runGeocode(inps, prodlist, bbox, demfilename, is_offset_mode=False):
    '''Generalized geocoding of all the files listed above.'''
    from isceobj.Catalog import recordInputsAndOutputs
    logger.info("Geocoding Image")
    #insar = self._insar

    #if (not self.doInSAR) and (not is_offset_mode):
    #    print('Skipping geocoding as InSAR processing has not been requested ....')
    #    return

    #elif (not self.doDenseOffsets) and (is_offset_mode):
    #    print('Skipping geocoding as Dense Offsets has not been requested ....')
    #    return


    if isinstance(prodlist,str):
        from isceobj.Util.StringUtils import StringUtils as SU
        tobeGeocoded = SU.listify(prodlist)
    else:
        tobeGeocoded = prodlist

    
    #remove files that have not been processed
    newlist=[]
    for toGeo in tobeGeocoded:
        if os.path.exists(toGeo):
            newlist.append(toGeo)

    
    tobeGeocoded = newlist
    print('Number of products to geocode: ', len(tobeGeocoded))

    if len(tobeGeocoded) == 0:
        print('No products found to geocode')
        return


    #swathList = self._insar.getValidSwathList(self.swaths)

    masterSwathList = ut.getSwathList(inps.master)
    slaveSwathList = ut.getSwathList(inps.slave)
    swathList = list(sorted(set(masterSwathList+slaveSwathList)))

    frames = []
    for swath in swathList:
        #topMaster = ut.loadProduct(os.path.join(inps.master , 'IW{0}.xml'.format(swath)))
        referenceProduct = ut.loadProduct(os.path.join(inps.slave , 'IW{0}.xml'.format(swath)))
        #referenceProduct = insar.loadProduct( os.path.join(insar.fineCoregDirname, 'IW{0}.xml'.format(swath)))
        frames.append(referenceProduct)

    orb = getMergedOrbit(frames)

    if bbox is None:
        bboxes = []

        for frame in frames:
            bboxes.append(frame.getBbox())

        snwe = [min([x[0] for x in bboxes]),
                max([x[1] for x in bboxes]),
                min([x[2] for x in bboxes]),
                max([x[3] for x in bboxes])]

    else:
        snwe = list(bbox)
        if len(snwe) != 4:
            raise ValueError('Bounding box should be a list/tuple of length 4')


    ###Identify the 4 corners and dimensions
    topSwath = min(frames, key = lambda x: x.sensingStart)
    leftSwath = min(frames, key = lambda x: x.startingRange)


    ####Get required values from product
    burst = frames[0].bursts[0]
    t0 = topSwath.sensingStart
    dtaz = burst.azimuthTimeInterval
    r0 = leftSwath.startingRange
    dr = burst.rangePixelSize
    wvl = burst.radarWavelength
    planet = Planet(pname='Earth')
    
    ###Setup DEM
    #demfilename = self.verifyGeocodeDEM()
    demImage = isceobj.createDemImage()
    demImage.load(demfilename + '.xml')

    ###Catalog for tracking
    #catalog = isceobj.Catalog.createCatalog(insar.procDoc.name)
    #catalog.addItem('Dem Used', demfilename, 'geocode')

    #####Geocode one by one
    first = False
    ge = Geocodable()
    for prod in tobeGeocoded:
        objGeo = createGeozero()
        objGeo.configure()

        ####IF statements to check for user configuration
        objGeo.snwe = snwe
        objGeo.demImage = demImage
        objGeo.demCropFilename = os.path.join(os.path.dirname(demfilename), "dem.crop") 
        if is_offset_mode:  ### If using topsOffsetApp, image has been "pre-looked" by the
            objGeo.numberRangeLooks = inps.skipwidth    ### skips in runDenseOffsets
            objGeo.numberAzimuthLooks = inps.skiphgt
        else:
            objGeo.numberRangeLooks = inps.numberRangeLooks
            objGeo.numberAzimuthLooks = inps.numberAzimuthLooks
        objGeo.lookSide = -1 #S1A is currently right looking only

        #create the instance of the input image and the appropriate
        #geocode method
        inImage,method = ge.create(prod)
        objGeo.method = method

        objGeo.slantRangePixelSpacing = dr
        objGeo.prf = 1.0 / dtaz
        objGeo.orbit = orb 
        objGeo.width = inImage.getWidth()
        objGeo.length = inImage.getLength()
        objGeo.dopplerCentroidCoeffs = [0.]
        objGeo.radarWavelength = wvl

        if is_offset_mode:  ### If using topsOffsetApp, as above, the "pre-looking" adjusts the range/time start
            objGeo.rangeFirstSample = r0 + (inps.offset_left-1) * dr
            objGeo.setSensingStart( t0 + datetime.timedelta(seconds=((inps.offset_top-1)*dtaz)))
        else:
            objGeo.rangeFirstSample = r0 + ((inps.numberRangeLooks-1)/2.0) * dr
            objGeo.setSensingStart( t0 + datetime.timedelta(seconds=(((inps.numberAzimuthLooks-1)/2.0)*dtaz)))
        objGeo.wireInputPort(name='dem', object=demImage)
        objGeo.wireInputPort(name='planet', object=planet)
        objGeo.wireInputPort(name='tobegeocoded', object=inImage)

        objGeo.geocode()

        print('Geocoding: ', inImage.filename, 'geocode')
        print('Output file: ', inImage.filename + '.geo', 'geocode')
        print('Width', inImage.width, 'geocode')
        print('Length', inImage.length, 'geocode')
        print('Range looks', inps.numberRangeLooks, 'geocode')
        print('Azimuth looks', inps.numberAzimuthLooks, 'geocode')
        print('South' , objGeo.minimumGeoLatitude, 'geocode')
        print('North', objGeo.maximumGeoLatitude, 'geocode')
        print('West', objGeo.minimumGeoLongitude, 'geocode')
        print('East', objGeo.maximumGeoLongitude, 'geocode')

    #catalog.printToLog(logger, "runGeocode")
    #self._insar.procDoc.addAllFromCatalog(catalog)

def main(iargs=None):
    '''
    Main driver.
    '''
    inps = cmdLineParse(iargs)
    bbox = [float(val) for val in inps.bbox.split()] 
    runGeocode(inps, inps.prodlist, bbox, inps.demfilename, is_offset_mode=False)

if __name__ == '__main__':

    main()

