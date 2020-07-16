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
from isceobj.Constants import SPEED_OF_LIGHT
import os
import datetime
import numpy as np

logger = logging.getLogger('isce.topsinsar.runGeocode')
posIndx = 1

def runGeocode(self, prodlist, bbox, is_offset_mode=False):
    '''Generalized geocoding of all the files listed above.'''
    from isceobj.Catalog import recordInputsAndOutputs
    logger.info("Geocoding Image")
    insar = self._insar

    if (not self.doDenseOffsets) and (is_offset_mode):
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


    ###Read in the product
    burst = self._insar.loadProduct( self._insar.referenceSlcCropProduct)

    ####Get required values from product
    t0 = burst.sensingStart
    prf = burst.PRF
    r0 = burst.startingRange
    dr = 0.5* SPEED_OF_LIGHT/ burst.rangeSamplingRate
    wvl = burst.radarWavelegth
    side= burst.getInstrument().getPlatform().pointingDirection
    orb = burst.orbit
    planet = Planet(pname='Earth')
     
    if (bbox is None):
        snwe = self._insar.estimatedBbox
    else:
        snwe = bbox
        if len(snwe) != 4 :
            raise Exception('Bbox must be 4 floats in SNWE order.')

    if is_offset_mode:  ### If using topsOffsetApp, image has been "pre-looked" by the
        numberRangeLooks = self.denseSkipWidth    ### skips in runDenseOffsets
        numberAzimuthLooks = self.denseSkipHeight
        rangeFirstSample = r0 + (self._insar.offset_left-1) * dr
        sensingStart =  t0 + datetime.timedelta(seconds=((self._insar.offset_top-1)/prf))
    else:
        ###Resolve number of looks
        azLooks, rgLooks = self.insar.numberOfLooks(burst, self.posting, self.numberAzimuthLooks, self.numberRangeLooks)

        numberRangeLooks = rgLooks
        numberAzimuthLooks = azLooks
        rangeFirstSample = r0 + ((numberRangeLooks-1)/2.0) * dr
        sensingStart = t0 + datetime.timedelta(seconds=(((numberAzimuthLooks-1)/2.0)/prf))


    ###Ughhh!! Doppler handling
    if self._insar.referenceGeometrySystem.lower().startswith('native'):
        ###Need to fit polynomials
        ###Geozero fortran assumes that starting range for radar image and polynomial are same
        ###Also assumes that the polynomial spacing is original spacing at full looks
        ###This is not true for multilooked data. Need to fix this with good datastruct in ISCEv3
        ###Alternate method is to modify the mean and norm of a Poly1D structure such that the 
        ###translation is accounted for correctly.
        poly = burst._dopplerVsPixel

        if len(poly) != 1:
            slcPix = np.linspace(0., burst.numberOfSamples, len(poly)+1)
            dopplers = np.polyval(poly[::-1], slcPix)

            newPix = slcPix - (rangeFirstSample - r0)/dr
            nppoly = np.polyfit(newPix, dopplers, len(poly)-1)

            dopplercoeff = list(nppoly[::-1])
        else:
            dopplercoeff = poly

    else:
        dopplercoeff = [0.]

    ##Scale by PRF since the module needs it
    dopplercoeff = [x/prf for x in dopplercoeff]

    ###Setup DEM
    demfilename = self.verifyDEM()
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
        objGeo.demCropFilename = insar.demCropFilename


        objGeo.dopplerCentroidCoeffs = dopplercoeff
        objGeo.lookSide = side

        #create the instance of the input image and the appropriate
        #geocode method
        inImage,method = ge.create(prod)
        objGeo.method = method

        objGeo.slantRangePixelSpacing = dr
        objGeo.prf = prf
        objGeo.orbit = orb 
        objGeo.width = inImage.getWidth()
        objGeo.length = inImage.getLength()
        objGeo.dopplerCentroidCoeffs = dopplercoeff
        objGeo.radarWavelength = wvl
        objGeo.rangeFirstSample = rangeFirstSample
        objGeo.setSensingStart(sensingStart)
        objGeo.numberRangeLooks = numberRangeLooks
        objGeo.numberAzimuthLooks = numberAzimuthLooks


        objGeo.wireInputPort(name='dem', object=demImage)
        objGeo.wireInputPort(name='planet', object=planet)
        objGeo.wireInputPort(name='tobegeocoded', object=inImage)

        objGeo.geocode()

        catalog.addItem('Geocoding: ', inImage.filename, 'geocode')
        catalog.addItem('Output file: ', inImage.filename + '.geo', 'geocode')
        catalog.addItem('Width', inImage.width, 'geocode')
        catalog.addItem('Length', inImage.length, 'geocode')
        catalog.addItem('Range looks', objGeo.numberRangeLooks, 'geocode')
        catalog.addItem('Azimuth looks', objGeo.numberAzimuthLooks, 'geocode')
        catalog.addItem('South' , objGeo.minimumGeoLatitude, 'geocode')
        catalog.addItem('North', objGeo.maximumGeoLatitude, 'geocode')
        catalog.addItem('West', objGeo.minimumGeoLongitude, 'geocode')
        catalog.addItem('East', objGeo.maximumGeoLongitude, 'geocode')

    catalog.printToLog(logger, "runGeocode")
    self._insar.procDoc.addAllFromCatalog(catalog)
