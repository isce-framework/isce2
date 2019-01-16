#
# Author: Piyush Agram
# Copyright 2016
#
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

logger = logging.getLogger('isce.topsinsar.runGeocode')
posIndx = 1

def runGeocode(self, prodlist, unwrapflag, bbox, is_offset_mode=False):
    '''Generalized geocoding of all the files listed above.'''
    from isceobj.Catalog import recordInputsAndOutputs
    logger.info("Geocoding Image")
    insar = self._insar

    if (not self.doInSAR) and (not is_offset_mode):
        print('Skipping geocoding as InSAR processing has not been requested ....')
        return

    elif (not self.doDenseOffsets) and (is_offset_mode):
        print('Skipping geocoding as Dense Offsets has not been requested ....')
        return


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


    swathList = self._insar.getValidSwathList(self.swaths)

    frames = []
    for swath in swathList:
        referenceProduct = insar.loadProduct( os.path.join(insar.fineCoregDirname, 'IW{0}.xml'.format(swath)))
        frames.append(referenceProduct)

    orb = self._insar.getMergedOrbit(frames)

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
    demfilename = self.verifyGeocodeDEM()
    demImage = isceobj.createDemImage()
    demImage.load(demfilename + '.xml')

    ###Catalog for tracking
    catalog = isceobj.Catalog.createCatalog(insar.procDoc.name)
    catalog.addItem('Dem Used', demfilename, 'geocode')

    #####Geocode one by one
    first = False
    ge = Geocodable()
    for prod in tobeGeocoded:
        objGeo = createGeozero()
        objGeo.configure()

        ####IF statements to check for user configuration
        objGeo.snwe = snwe
        objGeo.demCropFilename = os.path.join(insar.mergedDirname, insar.demCropFilename)
        if is_offset_mode:  ### If using topsOffsetApp, image has been "pre-looked" by the
            objGeo.numberRangeLooks = self.skipwidth    ### skips in runDenseOffsets
            objGeo.numberAzimuthLooks = self.skiphgt
        else:
            objGeo.numberRangeLooks = self.numberRangeLooks
            objGeo.numberAzimuthLooks = self.numberAzimuthLooks
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
            objGeo.rangeFirstSample = r0 + (self._insar.offset_left-1) * dr
            objGeo.setSensingStart( t0 + datetime.timedelta(seconds=((self._insar.offset_top-1)*dtaz)))
        else:
            objGeo.rangeFirstSample = r0 + ((self.numberRangeLooks-1)/2.0) * dr
            objGeo.setSensingStart( t0 + datetime.timedelta(seconds=(((self.numberAzimuthLooks-1)/2.0)*dtaz)))
        objGeo.wireInputPort(name='dem', object=demImage)
        objGeo.wireInputPort(name='planet', object=planet)
        objGeo.wireInputPort(name='tobegeocoded', object=inImage)

        objGeo.geocode()

        catalog.addItem('Geocoding: ', inImage.filename, 'geocode')
        catalog.addItem('Output file: ', inImage.filename + '.geo', 'geocode')
        catalog.addItem('Width', inImage.width, 'geocode')
        catalog.addItem('Length', inImage.length, 'geocode')
        catalog.addItem('Range looks', self.numberRangeLooks, 'geocode')
        catalog.addItem('Azimuth looks', self.numberAzimuthLooks, 'geocode')
        catalog.addItem('South' , objGeo.minimumGeoLatitude, 'geocode')
        catalog.addItem('North', objGeo.maximumGeoLatitude, 'geocode')
        catalog.addItem('West', objGeo.minimumGeoLongitude, 'geocode')
        catalog.addItem('East', objGeo.maximumGeoLongitude, 'geocode')

    catalog.printToLog(logger, "runGeocode")
    self._insar.procDoc.addAllFromCatalog(catalog)
