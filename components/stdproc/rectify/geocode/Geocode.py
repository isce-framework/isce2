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
from stdproc.rectify.geocode import geocode
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from isceobj.Util.Poly1D import Poly1D
import os



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

PEG_LATITUDE = Component.Parameter('pegLatitude',
        public_name = 'PEG_LATITUDE',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Peg point latitude in radians')

PEG_LONGITUDE = Component.Parameter('pegLongitude',
        public_name = 'PEG_LONGITUDE',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Peg point longitude in radians')

PEG_HEADING = Component.Parameter('pegHeading',
        public_name = 'PEG_HEADING',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Peg Point Heading in radians')

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

SPACECRAFT_HEIGHT = Component.Parameter('spacecraftHeight',
        public_name = 'SPACECRAFT_HEIGHT',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Height of the ideal mocomp orbit')

PLANET_LOCAL_RADIUS = Component.Parameter('planetLocalRadius',
        public_name = 'PLANET_LOCAL_RADIUS',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Local radius used for the ideal mocomp orbit')

BODY_FIXED_VELOCITY = Component.Parameter('bodyFixedVelocity',
        public_name = 'BODY_FIXED_VELOCITY',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Constant S velocity used for ideal mocomp orbit')

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

S_COORDINATE_FIRST_LINE = Component.Parameter('sCoordinateFirstLine',
        public_name = 'S_COORDINATE_FIRST_LINE',
        default = 1,
        type = int,
        mandatory = True,
        doc = 'S coordinate of the first line')

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

#Named it to something meaningful - Piyush
FIRST_INDEX_MOCOMP_ORBIT = Component.Parameter('isMocomp',
        public_name = 'FIRST_INDEX_MOCOMP_ORBIT',
        default = None,
        type = int,
        mandatory =True,
        doc = 'Index of first line in the mocomp orbit array')


DEM_CROP_FILENAME = Component.Parameter('demCropFilename',
        public_name = 'DEM_CROP_FILENAME',
        default = None,
        type = str,
        mandatory = True,
        doc = 'Filename for the cropped DEM output')

LOS_FILENAME = Component.Parameter('losFilename',
        public_name = 'LOS_FILENAME',
        default = None,
        type = str,
        mandatory = True,
        doc = 'Filename for LOS in geocoded coordinates')

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

NUMBER_POINTS_PER_DEM_POST = Component.Parameter('numberPointsPerDemPost',
        public_name = 'NUMBER_POINTS_PER_DEM_POST',
        default = 1,
        type = int,
        mandatory = True,
        doc = 'Number of points per DEM pixel incase posting at different resolution')

GEO_LENGTH = Component.Parameter(
    'geoLength',
    public_name='GEO_LENGTH',
    default=None,
    type=float,
    mandatory=False,
    intent='output',
    doc='Length of the geocoded image'
)


GEO_WIDTH = Component.Parameter(
    'geoWidth',
    public_name='GEO_WIDTH',
    default=None,
    type=int,
    mandatory=False,
    intent='output',
    doc='Width of the geocoded image'
)


LATITUDE_SPACING = Component.Parameter(
    'latitudeSpacing',
    public_name='LATITUDE_SPACING',
    default=None,
    type=float,
    mandatory=False,
    intent='output',
    doc='Latitude spacing'
)


LONGITUDE_SPACING = Component.Parameter(
    'longitudeSpacing',
    public_name='LONGITUDE_SPACING',
    default=None,
    type=float,
    mandatory=False,
    intent='output',
    doc='Longitude spacing'
)


MAXIMUM_GEO_LATITUDE = Component.Parameter(
    'maximumGeoLatitude',
    public_name='MAXIMUM_GEO_LATITUDE',
    default=None,
    type=float,
    mandatory=False,
    intent='output',
    doc='Maximum latitude of geocoded image'
)


MAXIMUM_GEO_LONGITUDE = Component.Parameter(
    'maximumGeoLongitude',
    public_name='MAXIMUM_GEO_LONGITUDE',
    default=None,
    type=float,
    mandatory=False,
    intent='output',
    doc='Minimum longitude of geocoded image'
)


MINIMUM_GEO_LATITUDE = Component.Parameter(
    'minimumGeoLatitude',
    public_name='MINIMUM_GEO_LATITUDE',
    default=None,
    type=float,
    mandatory=False,
    intent='output',
    doc='Minimum latitude of geocoded image'
)


MINIMUM_GEO_LONGITUDE = Component.Parameter(
    'minimumGeoLongitude',
    public_name='MINIMUM_GEO_LONGITUDE',
    default=None,
    type=float,
    mandatory=False,
    intent='output',
    doc='Minimum longitude of geocoded image'
)
class Geocode(Component):

    interp_methods = { 'sinc' : 0,
                       'bilinear' : 1,
                       'bicubic'  : 2,
                       'nearest'  : 3}

    family = 'geocode'
    logging_name = 'isce.stdproc.geocode'


    parameter_list = (INTERPOLATION_METHOD,
                      MINIMUM_LATITUDE,
                      MAXIMUM_LATITUDE,
                      MINIMUM_LONGITUDE,
                      MAXIMUM_LONGITUDE,
                      SLANT_RANGE_PIXEL_SPACING,
                      PEG_LATITUDE,
                      PEG_LONGITUDE,
                      PEG_HEADING,
                      ELLIPSOID_ECCENTRICITY_SQUARED,
                      ELLIPSOID_MAJOR_SEMIAXIS,
                      RANGE_FIRST_SAMPLE,
                      SPACECRAFT_HEIGHT,
                      PLANET_LOCAL_RADIUS,
                      BODY_FIXED_VELOCITY,
                      NUMBER_RANGE_LOOKS,
                      NUMBER_AZIMUTH_LOOKS,
                      PRF,
                      RADAR_WAVELENGTH,
                      S_COORDINATE_FIRST_LINE,
                      FIRST_INDEX_MOCOMP_ORBIT,
                      DEM_CROP_FILENAME,
                      LOS_FILENAME,
                      GEO_FILENAME,
                      LOOK_SIDE,
                      NUMBER_POINTS_PER_DEM_POST,
                      LONGITUDE_SPACING,
                      MINIMUM_GEO_LONGITUDE,
                      GEO_LENGTH,
                      MAXIMUM_GEO_LATITUDE,
                      LATITUDE_SPACING,
                      MAXIMUM_GEO_LONGITUDE,
                      GEO_WIDTH,
                      MINIMUM_GEO_LATITUDE
                      )


    #####Actual geocoding
    def geocode(self, demImage=None, inputImage=None, method=None):
        self.activateInputPorts()

        if demImage is not None:
            self.demImage = demImage
        if inputImage is not None:
            self.inputImage = inputImage
        if method is not None:
            self.method = method


        if self.referenceOrbit is None:
            raise Exception('No reference orbit provided for geocoding')

        self.setDefaults()
        self.createImages()
        self.allocateArray()
        self.setState()
        #this inits the image in the c++ bindings
        #allow geocoding for non float imaages
        if not self.inputImage.dataType.upper().count('FLOAT'):
            self.inputImage.setCaster('read','FLOAT')
        self.inputImage.createImage()


        self.demImage.setCaster('read','FLOAT')
        self.demImage.createImage()
        demAccessor = self.demImage.getImagePointer()

        inputAccessor = self.inputImage.getImagePointer()
        complexFlag = self.inputImage.dataType.upper().startswith('C')
        nBands = self.inputImage.getBands()

        #####Output cropped DEM for first band
        inband=0
        outband=0
        geocode.geocode_Py(demAccessor,
                           inputAccessor,
                           self.demCropAccessor,
                           self.losAccessor,
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
            geocode.geocode_Py(demAccessor, inputAccessor, demCropAcc,
                    self.losAccessor,
                    self.geoAccessor, inband, outband,
                    int(complexFlag), int(self.interp_methods[self.method]))

        self.getState()

        self.demImage.finalizeImage()
        self.inputImage.finalizeImage()
        self.deallocateArray()
        self.destroyImages()
        self.geoImage.setWidth(geocode.getGeoWidth_Py())
        self.geoImage.trueDataType = self.geoImage.getDataType()
#        self.geoImage.description = "DEM-flattened interferogram orthorectified to an equi-angular latitude, longitude grid"
        self.geoImage.coord2.coordDescription = 'Latitude'
        self.geoImage.coord2.coordUnits = 'degree'
        self.geoImage.coord2.coordStart = self.minimumGeoLatitude
        self.geoImage.coord2.coordDelta = self.deltaLatitude/self.numberPointsPerDemPost
        self.geoImage.coord1.coordDescription = 'Longitude'
        self.geoImage.coord1.coordUnits = 'degree'
        self.geoImage.coord1.coordStart = self.minimumGeoLongitude
        self.geoImage.coord1.coordDelta = self.deltaLongitude/self.numberPointsPerDemPost

        descr = self.inputImage.getDescription()
        if descr not in [None, '']:
            self.geoImage.addDescription(descr)

        self.geoImage.renderHdr()
        return None

    def setDefaults(self):
        if self.polyDoppler is None:
            self.polyDoppler = Poly1D(name=self.name+'_geocodePoly')
            self.polyDoppler.setNorm(1.0/(1.0*self.numberRangeLooks))
            self.polyDoppler.setMean(0.0)
            self.polyDoppler.initPoly(order=len(self.dopplerCentroidCoeffs)-1,
                coeffs = self.dopplerCentroidCoeffs)
        pass


    def destroyImages(self):
        from isceobj.Util import combinedlibmodule as CL
        if self.demCropImage is not None:
            self.demCropImage.finalizeImage()
            self.demCropImage.renderHdr()

        self.geoImage.finalizeImage()

        if self.losImage is not None:
            descr = '''Two channel Line-Of-Sight geometry image (all angles in degrees). Represents vector drawn from target to platform.                   Channel 1: Incidence angle measured from vertical at target (always +ve).
    Channel 2: Azimuth angle measured from North in Anti-clockwise direction.'''
            self.losImage.addDescription(descr)
            self.losImage.finalizeImage()
            self.losImage.renderHdr()

        #####Clean out polynomial object
        CL.freeCPoly1D(self.polyDopplerAccessor)
        self.polyDopplerAccessor = None

    def createImages(self):
        demWidth = self.computeGeoImageWidth()
        demLength = self.computeGeoImageLength()

        if self.demCropFilename:
            self.demCropImage = createDemImage()
            demAccessMode = 'write'
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
        self.geoImage.coord1.coordEnd = None
        self.geoImage.coord2.coordEnd = None

        if not self.geoImage.dataType.upper().count('FLOAT'):
            self.geoImage.setCaster('write','FLOAT')
        self.geoImage.createImage()
        self.geoImage.createFile(demLength)

        self.geoAccessor = self.geoImage.getImagePointer()
        if (self.losImage == None and self.losFilename not in ('',None)):
            self.losImage = createImage()
            accessMode= 'write'
            dataType = 'FLOAT'
            bands = 2
            scheme = 'BIL'
            width = demWidth
            self.losImage.initImage(self.losFilename,accessMode,
                    width,dataType,bands=bands,scheme=scheme)
            self.losImage.createImage()
            self.losAccessor = self.losImage.getImagePointer()

        self.polyDopplerAccessor = self.polyDoppler.exportToC()

    def computeGeoImageWidth(self):
        deg2rad = math.pi/180
        dlon = self.deltaLongitude*deg2rad
        dlon_out = abs(dlon/float(self.numberPointsPerDemPost))
        min_lon = deg2rad*self.minimumLongitude
        max_lon = deg2rad*self.maximumLongitude
        geo_wid = math.ceil((max_lon-min_lon)/dlon_out) + 1
        return geo_wid
    def computeGeoImageLength(self):
        deg2rad = math.pi/180
        dlat = self.deltaLatitude*deg2rad
        dlat_out = abs(dlat/float(self.numberPointsPerDemPost))
        min_lat = deg2rad*self.minimumLatitude
        max_lat = deg2rad*self.maximumLatitude
        geo_wid = math.ceil((max_lat-min_lat)/dlat_out) + 1
        return geo_wid
    def setState(self):
        geocode.setStdWriter_Py(int(self.stdWriter))
        geocode.setMinimumLatitude_Py(float(self.minimumLatitude))
        geocode.setMinimumLongitude_Py(float(self.minimumLongitude))
        geocode.setMaximumLatitude_Py(float(self.maximumLatitude))
        geocode.setMaximumLongitude_Py(float(self.maximumLongitude))
        geocode.setEllipsoidMajorSemiAxis_Py(float(self.ellipsoidMajorSemiAxis))
        geocode.setEllipsoidEccentricitySquared_Py(float(self.ellipsoidEccentricitySquared))
        geocode.setPegLatitude_Py(float(self.pegLatitude))
        geocode.setPegLongitude_Py(float(self.pegLongitude))
        geocode.setPegHeading_Py(float(self.pegHeading))
        geocode.setRangePixelSpacing_Py(float(self.slantRangePixelSpacing))
        geocode.setRangeFirstSample_Py(float(self.rangeFirstSample))
        geocode.setHeight_Py(float(self.spacecraftHeight))
        geocode.setPlanetLocalRadius_Py(float(self.planetLocalRadius))
        geocode.setVelocity_Py(float(self.bodyFixedVelocity))
        geocode.setDopplerAccessor_Py(self.polyDopplerAccessor)
        geocode.setPRF_Py(float(self.prf))
        geocode.setRadarWavelength_Py(float(self.radarWavelength))
        geocode.setSCoordinateFirstLine_Py(float(self.sCoordinateFirstLine))
        geocode.setFirstLatitude_Py(float(self.firstLatitude))
        geocode.setFirstLongitude_Py(float(self.firstLongitude))
        geocode.setDeltaLatitude_Py(float(self.deltaLatitude))
        geocode.setDeltaLongitude_Py(float(self.deltaLongitude))
        geocode.setLength_Py(int(self.length))
        geocode.setWidth_Py(int(self.width))
        geocode.setNumberRangeLooks_Py(int(self.numberRangeLooks))
        geocode.setNumberAzimuthLooks_Py(int(self.numberAzimuthLooks))
        geocode.setNumberPointsPerDemPost_Py(int(self.numberPointsPerDemPost))
        geocode.setISMocomp_Py(int(self.isMocomp))
        geocode.setDemWidth_Py(int(self.demWidth))
        geocode.setDemLength_Py(int(self.demLength))
        geocode.setReferenceOrbit_Py(self.referenceOrbit, self.dim1_referenceOrbit)
        geocode.setLookSide_Py(self.lookSide)


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

    def setPegLatitude(self,var):
        self.pegLatitude = float(var)

    def setPegLongitude(self,var):
        self.pegLongitude = float(var)

    def setPegHeading(self,var):
        self.pegHeading = float(var)

    def setRangePixelSpacing(self,var):
        self.slantRangePixelSpacing = float(var)

    def setRangeFirstSample(self,var):
        self.rangeFirstSample = float(var)

    def setSpacecraftHeight(self,var):
        self.spacecraftHeight = float(var)

    def setPlanetLocalRadius(self,var):
        self.planetLocalRadius = float(var)

    def setBodyFixedVelocity(self,var):
        self.bodyFixedVelocity = float(var)

    def setPRF(self,var):
        self.prf = float(var)

    def setRadarWavelength(self,var):
        self.radarWavelength = float(var)

    def setSCoordinateFirstLine(self,var):
        self.sCoordinateFirstLine = float(var)

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

    def setNumberPointsPerDemPost(self,var):
        self.numberPointsPerDemPost = int(var)

    def setISMocomp(self,var):
        self.isMocomp = int(var)

    def setDemWidth(self,var):
        self.demWidth = int(var)

    def setDemLength(self,var):
        self.demLength = int(var)

    def setLookSide(self,var):
        self.lookSide = int(var)

    def setReferenceOrbit(self,var):
        self.referenceOrbit = var

    def setDemCropFilename(self,var):
        self.demCropFilename = var

    def setPolyDoppler(self,var):
        self.polyDoppler = var

    ## pattern is broken here
    def setGeocodeFilename(self,var):
        self.geoFilename = var

    def getState(self):
        self.geoWidth = geocode.getGeoWidth_Py()
        self.geoLength = geocode.getGeoLength_Py()
        self.latitudeSpacing = geocode.getLatitudeSpacing_Py()
        self.longitudeSpacing = geocode.getLongitudeSpacing_Py()
        self.minimumGeoLatitude = geocode.getMinimumGeoLatitude_Py()
        self.minimumGeoLongitude = geocode.getMinimumGeoLongitude_Py()
        self.maximumGeoLatitude = geocode.getMaximumGeoLatitude_Py()
        self.maximumGeoLongitude = geocode.getMaxmumGeoLongitude_Py()

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

    def getMaxmumGeoLongitude(self):
        return self.maximumGeoLongitude

    def allocateArray(self):
        if (self.dim1_referenceOrbit == None):
            self.dim1_referenceOrbit = len(self.referenceOrbit)

        if (not self.dim1_referenceOrbit):
            self.logger.error("Trying to allocate zero size array")
            raise Exception

        geocode.allocate_s_mocomp_Py(self.dim1_referenceOrbit)

    def deallocateArray(self):
        geocode.deallocate_s_mocomp_Py()

    def addPeg(self):
        peg = self._inputPorts.getPort(name='peg').getObject()
        if (peg):
            try:
                self.pegLatitude = math.radians(peg.getLatitude())
                self.pegLongitude = math.radians(peg.getLongitude())
                self.pegHeading = math.radians(peg.getHeading())
                self.planetLocalRadius = peg.getRadiusOfCurvature()
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError

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

    def addReferenceSlc(self):   #Piyush
        formslc = self._inputPorts.getPort(name='referenceslc').getObject()
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

    def addGeoPosting(self):
        posting = self._inputPorts.getPort(name='geoPosting').getObject()
        print("addGeoPosting: posting = %r" % (posting,))
        if not self.demImage:
            import sys
            print("dem port needs to be wired before addGeoPosting")
            sys.exit(1)
        if(posting):
            try:
                self.deltaLatitude = posting
                self.deltaLongitude = posting
                ipts = int(max(self.demImage.deltaLatitude/posting,self.demImage.deltaLongitude/posting))
                if ipts < 1:
                    self.logger.info("numberPointsPerDemPost < 1, resetting to 1")
                self.numberPointsPerDemPost = max(ipts,1)
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError
            self.logger.info("Geocode, input geoImage posting = %f" % (posting,))
            self.logger.info("Geocode::deltaLatitude = %f" % (self.deltaLatitude,))
            self.logger.info("Geocode::deltaLongitude = %f" % (self.deltaLongitude,))
            self.logger.info("Geocode::numberPointsPerDemPost = %d" % (self.numberPointsPerDemPost,))
        else:
            self.deltaLatitude = self.demImage.deltaLatitude
            self.deltaLongitude = self.demImage.deltaLongitude
            self.numberPointsPerDemPost = 1
        return

    def addInterferogram(self):
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
        self.dopplerCentroidConstantTerm=None

        self.geoImage = None
        self.geoAccessor = None

        self.losImage = None
        self.losAccessor = 0

        self.referenceOrbit = []
        self.dim1_referenceOrbit = None


        return None


    def createPorts(self):
        framePort = Port(name='frame',method=self.addFrame)
        pegPort = Port(name='peg', method=self.addPeg)
        planetPort = Port(name='planet', method=self.addPlanet)
        demPort = Port(name='dem',method=self.addDem)
        ifgPort = Port(name='tobegeocoded',method=self.addInterferogram)
        geoPort = Port(name='geoPosting',method=self.addGeoPosting)
        slcPort = Port(name='referenceslc',method=self.addReferenceSlc)    #Piyush

        self._inputPorts.add(framePort)
        self._inputPorts.add(pegPort)
        self._inputPorts.add(planetPort)
        self._inputPorts.add(demPort)
        self._inputPorts.add(ifgPort)
        self._inputPorts.add(geoPort)
        self._inputPorts.add(slcPort)      #Piyush
        return None
