#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# United States Government Sponsorship acknowledged. This software is subject to
# U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
# (No [Export] License Required except when exporting to an embargoed country,
# end user, or in support of a prohibited end use). By downloading this software,
# the user agrees to comply with all applicable U.S. export laws and regulations.
# The user has the responsibility to obtain export licenses, or other export
# authority as may be required before exporting this software to any 'EAR99'
# embargoed foreign country or citizen of those countries.
#
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



from __future__ import print_function
import math
from isceobj.Image import createDemImage,createIntImage,createImage
from isceobj import Constants as CN
from iscesys.Component.Component import Component, Port
from zerodop.geozero import geozero
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from iscesys import DateTimeUtil as DTU
from isceobj.Util import combinedlibmodule
from isceobj.Util.Poly1D import Poly1D
import os
import datetime


INTERPOLATION_METHOD = Component.Parameter('method',
        public_name = 'INTERPOLATION_METHOD',
        default = None,
        type = str,
        mandatory = True,
        doc = 'Interpolation method. Can be sinc/ bilinear/ bicubic/ nearest')

MINIMUM_LATITUDE = Component.Parameter('minimumLatitude',
        public_name = 'MINIMUM_LATITUDE',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Minimum Latitude to geocode')

MAXIMUM_LATITUDE = Component.Parameter('maximumLatitude',
        public_name = 'MAXIMUM_LATITUDE',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Maximum Latitude to geocode')

MINIMUM_LONGITUDE = Component.Parameter('minimumLongitude',
        public_name = 'MINIMUM_LONGITUDE',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Minimum Longitude to geocode')

MAXIMUM_LONGITUDE = Component.Parameter('maximumLongitude',
        public_name = 'MAXIMUM_LONGITUDE',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Maximum Longitude to geocode')

ELLIPSOID_MAJOR_SEMIAXIS = Component.Parameter('ellipsoidMajorSemiAxis',
        public_name = 'ELLIPSOID_MAJOR_SEMIAXIS',
        default = CN.EarthMajorSemiAxis,
        type = float,
        mandatory = True,
        doc = 'Ellipsoid Major Semi Axis of planet for geocoding')

ELLIPSOID_ECCENTRICITY_SQUARED = Component.Parameter('ellipsoidEccentricitySquared',
        public_name = 'ELLIPSOID_ECCENTRICITY_SQUARED',
        default = CN.EarthEccentricitySquared,
        type = float,
        mandatory = True,
        doc = 'Ellipsoid Eccentricity Squared of planet for geocoding')

SLANT_RANGE_PIXEL_SPACING = Component.Parameter('slantRangePixelSpacing',
        public_name = 'SLANT_RANGE_PIXEL_SPACING',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Slant Range Pixel Spacing (single look) in meters')

RANGE_FIRST_SAMPLE = Component.Parameter('rangeFirstSample',
        public_name = 'RANGE_FIRST_SAMPLE',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Range to first sample')

PRF = Component.Parameter('prf',
        public_name = 'PRF',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Pulse repetition frequency')

RADAR_WAVELENGTH = Component.Parameter('radarWavelength',
        public_name = 'RADAR_WAVELENGTH',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Radar wavelength')


SENSING_START = Component.Parameter('sensingStart',
        public_name = 'SENSING_START',
        default = None,
        type=float,
        doc = 'Sensing start time for the first line')

NUMBER_RANGE_LOOKS = Component.Parameter('numberRangeLooks',
        public_name = 'NUMBER_RANGE_LOOKS',
        default = None,
        type = int,
        mandatory = True,
        doc = 'Number of range looks used to generate radar image')

NUMBER_AZIMUTH_LOOKS = Component.Parameter('numberAzimuthLooks',
        public_name = 'NUMBER_AZIMUTH_LOOKS',
        default = None,
        type = int,
        mandatory = True,
        doc = 'Number of azimuth looks used to generate radar image')

DEM_CROP_FILENAME = Component.Parameter('demCropFilename',
        public_name = 'DEM_CROP_FILENAME',
        default = None,
        type = str,
        mandatory = True,
        doc = 'Filename for the cropped DEM output')

GEO_FILENAME = Component.Parameter('geoFilename',
        public_name = 'GEO_FILENAME',
        default = None,
        type = str,
        mandatory = True,
        doc = 'Output geocoded file name')

LOOK_SIDE = Component.Parameter('lookSide',
        public_name = 'LOOK_SIDE',
        default = None,
        type = int,
        mandatory = True,
        doc = 'Right (-1) / Left (1) . Look direction of the radar platform')

class Geocode(Component):

    interp_methods = { 'sinc' : 0,
                       'bilinear' : 1,
                       'bicubic'  : 2,
                       'nearest'  : 3}

    family = 'geocode'
    logging_name = 'isce.zerodop.geocode'


    parameter_list = (INTERPOLATION_METHOD,
                      MINIMUM_LATITUDE,
                      MAXIMUM_LATITUDE,
                      MINIMUM_LONGITUDE,
                      MAXIMUM_LONGITUDE,
                      SLANT_RANGE_PIXEL_SPACING,
                      ELLIPSOID_ECCENTRICITY_SQUARED,
                      ELLIPSOID_MAJOR_SEMIAXIS,
                      RANGE_FIRST_SAMPLE,
                      SENSING_START,
                      NUMBER_RANGE_LOOKS,
                      NUMBER_AZIMUTH_LOOKS,
                      PRF,
                      RADAR_WAVELENGTH,
                      DEM_CROP_FILENAME,
                      GEO_FILENAME,
                      LOOK_SIDE)


    #####Actual geocoding
    def geocode(self, demImage=None, inputImage=None, method=None):
        self.activateInputPorts()

        if demImage is not None:
            self.demImage = demImage
        if inputImage is not None:
            self.inputImage = inputImage
        if method is not None:
            self.method = method


        if self.orbit is None:
            raise Exception('No orbit provided for geocoding')

        self.setDefaults()
        self.createImages()
        self.setState()
        #this inits the image in the c++ bindings

        if not self.inputImage.dataType.upper().count('FLOAT'):
            self.inputImage.setCaster('read', 'FLOAT')

        self.inputImage.createImage()
        self.demImage.setCaster('read','FLOAT')
        self.demImage.createImage()
        demAccessor = self.demImage.getImagePointer()

        inputAccessor = self.inputImage.getImagePointer()
        complexFlag = self.inputImage.dataType.upper().startswith('C')
        nBands = self.inputImage.getBands()

        cOrbit = self.orbit.exportToC(reference=self.sensingStart)
        geozero.setOrbit_Py(cOrbit)

        #####Output cropped DEM for first band
        inband=0
        outband=0
        geozero.geozero_Py(demAccessor,
                           inputAccessor,
                           self.demCropAccessor,
                           self.geoAccessor,inband,
                           outband,int(complexFlag),
                           int(self.interp_methods[self.method]))

        #####Supress cropped DEM output for other bands
        for kk in range(1,nBands):
            self.demImage.rewind()
            self.inputImage.rewind()
            self.demCropImage.rewind()
            self.geoImage.rewind()

            inband = kk
            outband = kk
            demCropAcc = 0
            geozero.geozero_Py(demAccessor, inputAccessor, demCropAcc,
                    self.geoAccessor, inband, outband,
                    int(complexFlag), int(self.interp_methods[self.method]))

        combinedlibmodule.freeCOrbit(cOrbit)
        self.getState()

        self.demImage.finalizeImage()
        self.inputImage.finalizeImage()
        self.destroyImages()
        self.geoImage.setWidth(geozero.getGeoWidth_Py())
        self.geoImage.trueDataType = self.geoImage.getDataType()
#        self.geoImage.description = "DEM-flattened interferogram orthorectified to an equi-angular latitude, longitude grid"
        self.geoImage.coord2.coordDescription = 'Latitude'
        self.geoImage.coord2.coordUnits = 'degree'
        self.geoImage.coord2.coordStart = self.maximumGeoLatitude
        self.geoImage.coord2.coordDelta = self.deltaLatitude
        self.geoImage.coord1.coordDescription = 'Longitude'
        self.geoImage.coord1.coordUnits = 'degree'
        self.geoImage.coord1.coordStart = self.minimumGeoLongitude
        self.geoImage.coord1.coordDelta = self.deltaLongitude

        descr = self.inputImage.getDescription()
        if descr not in [None, '']:
            self.geoImage.addDescription(descr)

        self.geoImage.renderHdr()
        return None

    def setDefaults(self):
        if self.polyDoppler is None:
            self.polyDoppler = Poly1D(name=self.name+'_geozeroPoly')
            self.polyDoppler.setMean(0.0)
            self.polyDoppler.initPoly(order=len(self.dopplerCentroidCoeffs)-1,
                coeffs = self.dopplerCentroidCoeffs)
        pass


    def destroyImages(self):
        from isceobj.Util import combinedlibmodule as CL
        if self.demCropImage is not None:
            self.demCropImage.renderHdr()
            self.demCropImage.finalizeImage()

        self.geoImage.finalizeImage()

        #####Clean out polynomial object
        CL.freeCPoly1D(self.polyDopplerAccessor)
        self.polyDopplerAccessor = None

    def createImages(self):
        if self.demCropFilename:
            self.demCropImage = createDemImage()
            demAccessMode = 'write'
            demWidth = self.computeGeoImageWidth()
            self.demCropImage.initImage(self.demCropFilename,demAccessMode,demWidth)
            self.demCropImage.createImage()
            self.demCropAccessor = self.demCropImage.getImagePointer()
        else:
            self.demCropAccessor = 0

        if self.geoFilename is None:
            raise ValueError('Output geoFilename not specified')

        #the topophase files have the same format as the int file. just reuse the previous info
        self.geoImage = createIntImage()
        IU.copyAttributes(self.inputImage, self.geoImage)
        self.geoImage.imageType = self.inputImage.imageType
        self.geoImage.setFilename(self.geoFilename)
        self.geoImage.setAccessMode('write')
        self.geoImage.setWidth(demWidth)

        if not self.geoImage.dataType.upper().count('FLOAT'):
            self.geoImage.setCaster('write', 'FLOAT')

        self.geoImage.createImage()
        self.geoAccessor = self.geoImage.getImagePointer()

        self.polyDopplerAccessor = self.polyDoppler.exportToC()

    def computeGeoImageWidth(self):
        deg2rad = math.pi/180.0
        dlon = self.deltaLongitude*deg2rad
        lon_first = self.firstLongitude*deg2rad
        min_lon = deg2rad*self.minimumLongitude
        max_lon = deg2rad*self.maximumLongitude
        min_lon_idx = int( (min_lon - lon_first) / dlon)
        max_lon_idx = int( (max_lon - lon_first) / dlon)
        geo_wid = max_lon_idx - min_lon_idx + 1
        return geo_wid

    def setState(self):
        geozero.setMinimumLatitude_Py(float(self.minimumLatitude))
        geozero.setMinimumLongitude_Py(float(self.minimumLongitude))
        geozero.setMaximumLatitude_Py(float(self.maximumLatitude))
        geozero.setMaximumLongitude_Py(float(self.maximumLongitude))
        geozero.setEllipsoidMajorSemiAxis_Py(float(self.ellipsoidMajorSemiAxis))
        geozero.setEllipsoidEccentricitySquared_Py(float(self.ellipsoidEccentricitySquared))
        geozero.setRangePixelSpacing_Py(float(self.slantRangePixelSpacing))
        geozero.setRangeFirstSample_Py(float(self.rangeFirstSample))
        geozero.setDopplerAccessor_Py(self.polyDopplerAccessor)
        geozero.setPRF_Py(float(self.prf))
        geozero.setRadarWavelength_Py(float(self.radarWavelength))
        geozero.setSensingStart_Py(DTU.seconds_since_midnight(self.sensingStart))
        geozero.setFirstLatitude_Py(float(self.firstLatitude))
        geozero.setFirstLongitude_Py(float(self.firstLongitude))
        geozero.setDeltaLatitude_Py(float(self.deltaLatitude))
        geozero.setDeltaLongitude_Py(float(self.deltaLongitude))
        geozero.setLength_Py(int(self.length))
        geozero.setWidth_Py(int(self.width))
        geozero.setNumberRangeLooks_Py(int(self.numberRangeLooks))
        geozero.setNumberAzimuthLooks_Py(int(self.numberAzimuthLooks))
        geozero.setDemWidth_Py(int(self.demWidth))
        geozero.setDemLength_Py(int(self.demLength))
        geozero.setLookSide_Py(self.lookSide)


    def setMinimumLatitude(self,var):
        self.minimumLatitude = float(var)

    def setMinimumLongitude(self,var):
        self.minimumLongitude = float(var)

    def setMaximumLatitude(self,var):
        self.maximumLatitude = float(var)

    def setMaximumLongitude(self,var):
        self.maximumLongitude = float(var)

    def setEllipsoidMajorSemiAxis(self,var):
        self.ellipsoidMajorSemiAxis = float(var)

    def setEllipsoidEccentricitySquared(self,var):
        self.ellipsoidEccentricitySquared = float(var)

    def setRangePixelSpacing(self,var):
        self.slantRangePixelSpacing = float(var)

    def setRangeFirstSample(self,var):
        self.rangeFirstSample = float(var)

    def setPRF(self,var):
        self.prf = float(var)

    def setRadarWavelength(self,var):
        self.radarWavelength = float(var)

    def setSensingStart(self,var):
        self.sensingStart = var

    def setFirstLatitude(self,var):
        self.firstLatitude = float(var)

    def setFirstLongitude(self,var):
        self.firstLongitude = float(var)

    def setDeltaLatitude(self,var):
        self.deltaLatitude = float(var)

    def setDeltaLongitude(self,var):
        self.deltaLongitude = float(var)

    def setLength(self,var):
        self.length = int(var)

    def setWidth(self,var):
        self.width = int(var)

    def setNumberRangeLooks(self,var):
        self.numberRangeLooks = int(var)

    def setNumberAzimuthLooks(self,var):
        self.numberAzimuthLooks = int(var)

    def setDemWidth(self,var):
        self.demWidth = int(var)

    def setDemLength(self,var):
        self.demLength = int(var)

    def setLookSide(self,var):
        self.lookSide = int(var)

    def setOrbit(self,var):
        self.orbit = var

    def setDemCropFilename(self,var):
        self.demCropFilename = var

    def setPolyDoppler(self,var):
        self.polyDoppler = var

    ## pattern is broken here
    def setGeocodeFilename(self,var):
        self.geoFilename = var

    def getState(self):
        self.geoWidth = geozero.getGeoWidth_Py()
        self.geoLength = geozero.getGeoLength_Py()
        self.minimumGeoLatitude = geozero.getMinimumGeoLatitude_Py()
        self.minimumGeoLongitude = geozero.getMinimumGeoLongitude_Py()
        self.maximumGeoLatitude = geozero.getMaximumGeoLatitude_Py()
        self.maximumGeoLongitude = geozero.getMaximumGeoLongitude_Py()

    def getGeoWidth(self):
        return self.geoWidth

    def getGeoLength(self):
        return self.geoLength

    def getLatitudeSpacing(self):
        return self.latitudeSpacing

    def getLongitudeSpacing(self):
        return self.longitudeSpacing

    def getMinimumGeoLatitude(self):
        return self.minimumGeoLatitude

    def getMinimumGeoLongitude(self):
        return self.minimumGeoLongitude

    def getMaximumGeoLatitude(self):
        return self.maximumGeoLatitude

    def getMaximumGeoLongitude(self):
        return self.maximumGeoLongitude

    def addPlanet(self):
        planet = self._inputPorts.getPort(name='planet').getObject()
        if (planet):
            try:
                ellipsoid = planet.get_elp()
                self.ellipsoidMajorSemiAxis = ellipsoid.get_a()
                self.ellipsoidEccentricitySquared = ellipsoid.get_e2()
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError

    def addFrame(self):
        frame = self._inputPorts.getPort(name='frame').getObject()
        if (frame):
            try:
                #                self.rangeFirstSample = frame.getStartingRange() - Piyush
                instrument = frame.getInstrument()
                self.lookSide = instrument.getPlatform().pointingDirection
                self.slantRangePixelSpacing = instrument.getRangePixelSize()
                self.prf = instrument.getPulseRepetitionFrequency()
                self.radarWavelength = instrument.getRadarWavelength()
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError

    def addMasterSlc(self):   #Piyush
        formslc = self._inputPorts.getPort(name='masterslc').getObject()
        if(formslc):
            try:
                self.rangeFirstSample = formslc.startingRange
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError

            self.dopplerCentroidCoeffs = formslc.dopplerCentroidCoefficients

    def addDem(self):
        dem = self._inputPorts.getPort(name='dem').getObject()
        if (dem):
            try:
                self.demImage = dem
                self.demWidth = dem.getWidth()
                self.demLength = dem.getLength()
                self.firstLatitude = dem.getFirstLatitude()
                self.firstLongitude = dem.getFirstLongitude()
                self.deltaLatitude = dem.getDeltaLatitude()   # This should be removed once we fail-safe the ordering of addDem, addGeoPosting
                self.deltaLongitude = dem.getDeltaLongitude()
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError

    def addRadarImage(self):
        ifg = self._inputPorts.getPort(name='tobegeocoded').getObject()
        if (ifg):
            try:
                self.inputImage = ifg
                self.width = ifg.getWidth()
                self.length = ifg.getLength()

                inName = ifg.getFilename()
                self.geoFilename = os.path.join(os.path.dirname(inName),
                        os.path.basename(inName)+'.geo')
                print('Output: ' , self.geoFilename)
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError





    ## South, North, West, East boundaries
    @property
    def snwe(self):
        return (self.minimumLatitude,
                self.maximumLatitude,
                self.minimumLongitude,
                self.maximumLongitude)

    @snwe.setter
    def snwe(self, snwe):
        (self.minimumLatitude, self.maximumLatitude,
        self.minimumLongitude, self.maximumLongitude) = snwe


    logging_name = 'isce.stdproc.geocode'

    def __init__(self, name='') :
        super(Geocode, self).__init__(self.__class__.family, name)

        # Dem information
        self.demImage = None
        self.demWidth = None
        self.demLength = None
        self.firstLatitude = None
        self.firstLongitude = None
        self.deltaLatitude = None
        self.deltaLongitude = None

        # Interferogram information
        self.inputImage = None
        self.length = None
        self.width = None

        # Output
        self.demCropImage = None
        self.demCropAccessor = None

        #Doppler information
        self.polyDoppler = None
        self.polyDopplerAccessor = None
        self.dopplerCentroidCoeffs = None

        self.geoImage = None
        self.geoAccessor = None
        self.geoWidth = None
        self.geoLength = None

        self.orbit = None
        self.latitudeSpacing = None
        self.longitudeSpacing = None
        self.minimumGeoLatitude = None
        self.minimumGeoLongitude = None
        self.maximumGeoLatitude = None
        self.maximumGeoLongitude = None


        self.dictionaryOfOutputVariables = {
            'GEO_WIDTH' : 'self.geoWidth',
            'GEO_LENGTH' : 'self.geoLength',
            'LATITUDE_SPACING' : 'self.latitudeSpacing',
            'LONGITUDE_SPACING' : 'self.longitudeSpacing',
            'MINIMUM_GEO_LATITUDE' : 'self.minimumGeoLatitude',
            'MINIMUM_GEO_LONGITUDE' : 'self.minimumGeoLongitude',
            'MAXIMUM_GEO_LATITUDE' : 'self.maximumGeoLatitude',
            'MAXIMUM_GEO_LONGITUDE' : 'self.maximumGeoLongitude'
            }

        return None


    def createPorts(self):
        framePort = Port(name='frame',method=self.addFrame)
        planetPort = Port(name='planet', method=self.addPlanet)
        demPort = Port(name='dem',method=self.addDem)
        ifgPort = Port(name='tobegeocoded',method=self.addRadarImage)
        slcPort = Port(name='masterslc',method=self.addMasterSlc)    #Piyush

        self._inputPorts.add(framePort)
        self._inputPorts.add(planetPort)
        self._inputPorts.add(demPort)
        self._inputPorts.add(ifgPort)
        self._inputPorts.add(slcPort)      #Piyush
        return None
