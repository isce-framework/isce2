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
from iscesys.Component.Component import Component, Port
from isceobj import Constants as CN
from iscesys.Compatibility import Compatibility
import isceobj.Image as IF #load image factories
from stdproc.stdproc.topo import topo
from isceobj.Util import Polynomial, Poly2D
from iscesys import DateTimeUtil as DTU
from isceobj.Util import combinedlibmodule
import datetime

demInterpolationMethod = 'BIQUINTIC'

PRF = Component.Parameter(
    'prf',
    public_name='PRF',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='Radar pulse repetition frequency'
)


DOPPLER_CENTROID_CONSTANT_TERM = Component.Parameter(
    'dopplerCentroidConstantTerm',
    public_name='DOPPLER_CENTROID_CONSTANT_TERM',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='Constant term of the expansion of the doppler centroid'
)


PEG_HEADING = Component.Parameter(
    'pegHeading',
    public_name='PEG_HEADING',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='Peg heading'
)


DELTA_LONGITUDE = Component.Parameter(
    'deltaLongitude',
    public_name='DELTA_LONGITUDE',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='DEM longitude resolution'
)


FIRST_LONGITUDE = Component.Parameter(
    'firstLongitude',
    public_name='FIRST_LONGITUDE',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='DEM starting longitude value'
)



DEM_LENGTH = Component.Parameter(
    'demLength',
    public_name='DEM_LENGTH',
    default=None,
    type=int,
    mandatory=True,
    intent='input',
    doc='Number of lines in the DEM image'
)


PEG_LATITUDE = Component.Parameter(
    'pegLatitude',
    public_name='PEG_LATITUDE',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='Peg latitude'
)


FIRST_LATITUDE = Component.Parameter(
    'firstLatitude',
    public_name='FIRST_LATITUDE',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='DEM starting latitude value'
)


ELLIPSOID_MAJOR_SEMIAXIS = Component.Parameter(
    'ellipsoidMajorSemiAxis',
    public_name='ELLIPSOID_MAJOR_SEMIAXIS',
    default=None,
    type=float,
    mandatory=False,
    intent='input',
    doc='Ellipsoid major semiaxis'
)


IS_MOCOMP = Component.Parameter(
    'isMocomp',
    public_name='IS_MOCOMP',
    default=None,
    type=int,
    mandatory=False,
    intent='input',
    doc=''
)


BODY_FIXED_VELOCITY = Component.Parameter(
    'bodyFixedVelocity',
    public_name='BODY_FIXED_VELOCITY',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='Platform body fix velocity'
)


NUMBER_RANGE_LOOKS = Component.Parameter(
    'numberRangeLooks',
    public_name='NUMBER_RANGE_LOOKS',
    default=None,
    type=int,
    mandatory=True,
    intent='input',
    doc='Number of range looks'
)


NUMBER_ITERATIONS = Component.Parameter(
    'numberIterations',
    public_name='NUMBER_ITERATIONS',
    default=25,
    type=int,
    mandatory=False,
    intent='input',
    doc='Number of iterations'
)


ELLIPSOID_ECCENTRICITY_SQUARED = Component.Parameter(
    'ellipsoidEccentricitySquared',
    public_name='ELLIPSOID_ECCENTRICITY_SQUARED',
    default=None,
    type=float,
    mandatory=False,
    intent='input',
    doc='Squared value of the ellipsoid eccentricity'
)


REFERENCE_ORBIT = Component.Parameter(
    'referenceOrbit',
    public_name='REFERENCE_ORBIT',
    default=[],
    container=list,
    type=float,
    mandatory=True,
    intent='input',
    doc='Reference orbit'
)


SLANT_RANGE_PIXEL_SPACING = Component.Parameter(
    'slantRangePixelSpacing',
    public_name='SLANT_RANGE_PIXEL_SPACING',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='Slant range pixel spacing'
)


SPACECRAFT_HEIGHT = Component.Parameter(
    'spacecraftHeight',
    public_name='SPACECRAFT_HEIGHT',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='Spacecraft height'
)


RADAR_WAVELENGTH = Component.Parameter(
    'radarWavelength',
    public_name='RADAR_WAVELENGTH',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='Radar wavelength'
)


PEG_LONGITUDE = Component.Parameter(
    'pegLongitude',
    public_name='PEG_LONGITUDE',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='Peg longitude'
)


DEM_WIDTH = Component.Parameter(
    'demWidth',
    public_name='DEM_WIDTH',
    default=None,
    type=int,
    mandatory=True,
    intent='input',
    doc='DEM width'
)


NUMBER_AZIMUTH_LOOKS = Component.Parameter(
    'numberAzimuthLooks',
    public_name='NUMBER_AZIMUTH_LOOKS',
    default=None,
    type=int,
    mandatory=True,
    intent='input',
    doc='Number of azimuth looks'
)


RANGE_FIRST_SAMPLE = Component.Parameter(
    'rangeFirstSample',
    public_name='RANGE_FIRST_SAMPLE',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='Range of the first sample'
)


LENGTH = Component.Parameter(
    'length',
    public_name='LENGTH',
    default=None,
    type=int,
    mandatory=True,
    intent='input',
    doc='Number of lines in the Interferogram'
)


PLANET_LOCAL_RADIUS = Component.Parameter(
    'planetLocalRadius',
    public_name='PLANET_LOCAL_RADIUS',
    default=None,
    type=float,
    mandatory=True,
    intent='inoutput',
    doc='Planet local radius'
)


WIDTH = Component.Parameter(
    'width',
    public_name='WIDTH',
    default=None,
    type=int,
    mandatory=True,
    intent='input',
    doc='Interferogram width'
)


DELTA_LATITUDE = Component.Parameter(
    'deltaLatitude',
    public_name='DELTA_LATITUDE',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='DEM latitude resolution'
)


S_COORDINATE_LAST_LINE = Component.Parameter(
    'sCoordinateLastLine',
    public_name='S_COORDINATE_LAST_LINE',
    default=None,
    type=float,
    mandatory=False,
    intent='output',
    doc='S coordinate last line'
)


S_COORDINATE_FIRST_LINE = Component.Parameter(
    'sCoordinateFirstLine',
    public_name='S_COORDINATE_FIRST_LINE',
    default=None,
    type=float,
    mandatory=False,
    intent='output',
    doc='S coordinate last line'
)


MAXIMUM_LONGITUDE = Component.Parameter(
    'maximumLongitude',
    public_name='MAXIMUM_LONGITUDE',
    default=None,
    type=float,
    mandatory=False,
    intent='output',
    doc='Maximum longitude of the resulting image'
)


MINIMUM_LONGITUDE = Component.Parameter(
    'minimumLongitude',
    public_name='MINIMUM_LONGITUDE',
    default=None,
    type=float,
    mandatory=False,
    intent='output',
    doc='Minimum longitude of the resulting image'
)


AZIMUTH_SPACING = Component.Parameter(
    'azimuthSpacing',
    public_name='AZIMUTH_SPACING',
    default=None,
    type=float,
    mandatory=False,
    intent='output',
    doc=''
)


MINIMUM_LATITUDE = Component.Parameter(
    'minimumLatitude',
    public_name='MINIMUM_LATITUDE',
    default=None,
    type=float,
    mandatory=False,
    intent='output',
    doc='Maximum longitude of the resulting image'
)


MAXIMUM_LATITUDE = Component.Parameter(
    'maximumLatitude',
    public_name='MAXIMUM_LATITUDE',
    default=None,
    type=float,
    mandatory=False,
    intent='output',
    doc='Maximum latitude of the resulting image'
)


SQUINT_SHIFT = Component.Parameter(
    'squintshift',
    public_name='SQUINT_SHIFT',
    default=[],
    container=list,
    type=float,
    mandatory=False,
    intent='output',
    doc='Squint shift'
)

ORBIT = Component.Facility(
        'orbit',
        public_name = 'MOCOMP_ORBIT',
        module = 'isceobj.Orbit',
        args=(),
        factory='createOrbit',
        mandatory=True,
        doc='Mocomp orbit to be used for geometry.')

SENSING_START = Component.Parameter(
        'sensingStart',
        public_name='SENSING_START',
        default=None,
        type=datetime.datetime,
        mandatory=True,
        doc='Sensing start time for 1st line of input image')


class Topo(Component):


    parameter_list = (
                      PRF,
                      DOPPLER_CENTROID_CONSTANT_TERM,
                      PEG_HEADING,
                      DELTA_LONGITUDE,
                      FIRST_LONGITUDE,
                      DEM_LENGTH,
                      PEG_LATITUDE,
                      FIRST_LATITUDE,
                      ELLIPSOID_MAJOR_SEMIAXIS,
                      IS_MOCOMP,
                      BODY_FIXED_VELOCITY,
                      NUMBER_RANGE_LOOKS,
                      NUMBER_ITERATIONS,
                      ELLIPSOID_ECCENTRICITY_SQUARED,
                      REFERENCE_ORBIT,
                      SLANT_RANGE_PIXEL_SPACING,
                      SPACECRAFT_HEIGHT,
                      RADAR_WAVELENGTH,
                      PEG_LONGITUDE,
                      DEM_WIDTH,
                      NUMBER_AZIMUTH_LOOKS,
                      RANGE_FIRST_SAMPLE,
                      LENGTH,
                      PLANET_LOCAL_RADIUS,
                      WIDTH,
                      DELTA_LATITUDE,
                      S_COORDINATE_LAST_LINE,
                      S_COORDINATE_FIRST_LINE,
                      MAXIMUM_LONGITUDE,
                      MINIMUM_LONGITUDE,
                      AZIMUTH_SPACING,
                      MINIMUM_LATITUDE,
                      MAXIMUM_LATITUDE,
                      SQUINT_SHIFT,
                      SENSING_START,
                     )

    facility_list = (
                      ORBIT,
                    )


    interpolationMethods = { 'SINC' : 0,
                             'BILINEAR' : 1,
                             'BICUBIC' : 2,
                             'NEAREST' : 3,
                             'AKIMA' : 4,
                             'BIQUINTIC' : 5}
    ## South, North, West, East boundaries
    ## see geocode and topo to much resued code.
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


    def topo(self, demImage=None, intImage=None):
        for port in self._inputPorts:
            port()

        if demImage is not None:
            self.demImage = demImage

        #another way of passing width and length if not using the ports
        if intImage is not None:
            self.intImage = intImage
            #if width or length not defined get 'em  from intImage ince they
            # are needed to create the output images
            if self.width is None:
                self.width = self.intImage.getWidth()
            if self.length is None:
                self.length = self.intImage.getLength()

        self.setDefaults()
        self.createImages()
        #not all the quantities could be set before. now that we have the
        # images set the remaining defaults if necessary (such as width, length)
        self.updateDefaults()

        self.squintshift = [0]*self.width #preallocate
        self.demAccessor = self.demImage.getImagePointer()
        self.latAccessor = self.latImage.getImagePointer()
        self.lonAccessor = self.lonImage.getImagePointer()
        self.heightRAccessor = self.heightRImage.getImagePointer()
        self.heightSchAccessor = self.heightSchImage.getImagePointer()
        self.losAccessor = self.losImage.getImagePointer()

        if self.incImage:
            self.incAccessor = self.incImage.getImagePointer()
        else:
            self.incAccessor = 0

        ####Doppler accessor
        self.polyDoppler.createPoly2D()
        self.polyDopplerAccessor = self.polyDoppler.getPointer()

        self.allocateArrays()
        self.setState()

        corb = self.orbit.exportToC()
        topo.setOrbit_Py(corb)
        topo.topo_Py(self.demAccessor, self.polyDopplerAccessor)
        combinedlibmodule.freeCOrbit(corb)
        self.getState()
        self.deallocateArrays()
        self.destroyImages()

        return None



    def setDefaults(self):
        if self.ellipsoidMajorSemiAxis is None:
            self.ellipsoidMajorSemiAxis = CN.EarthMajorSemiAxis

        if self.ellipsoidEccentricitySquared is None:
            self.ellipsoidEccentricitySquared = CN.EarthEccentricitySquared

        if self.isMocomp is None:
            self.isMocomp = (8192-2048)/2

        if self.numberIterations is None:
            self.numberIterations = 25

        if self.heightRFilename == '':
            self.heightRFilename = 'z.rdr'
            self.logger.warning('The real height file has been given the default name %s' % (self.heightRFilename))
        if self.heightSchFilename == '':
            self.heightSchFilename = 'zsch.rdr'
            self.logger.warning('The sch height file has been given the default name %s' % (self.heightSchFilename))
        if self.latFilename == '':
            self.latFilename = 'lat.rdr'
            self.logger.warning('The latitude file has been given the default name %s' % (self.latFilename))

        if self.lonFilename == '':
            self.lonFilename = 'lon.rdr'
            self.logger.warning('The longitude file has been given the default name %s' % (self.lonFilename))

        if self.losFilename == '':
            self.losFilename = 'los.rdr'
            self.logger.warning('The los file has been given the default name %s' % (self.losFilename))

        if self.polyDoppler is None:
            self.polyDoppler = Poly2D.Poly2D(name=self.name + '_topoPoly')
            self.polyDoppler.setWidth(self.width)
            self.polyDoppler.setLength(self.length)
            self.polyDoppler.setNormRange(1.0/(1.0*self.numberRangeLooks))
            self.polyDoppler.setNormAzimuth(1.0/(1.0*self.numberAzimuthLooks))
            self.polyDoppler.setMeanRange(0.)
            self.polyDoppler.setMeanAzimuth(0.)
            self.polyDoppler.initPoly(
                rangeOrder=len(self.dopplerCentroidCoeffs)-1,
                azimuthOrder=0,
                coeffs=[self.dopplerCentroidCoeffs])

        if self.demInterpolationMethod is None:
            self.demInterpolationMethod = 'BILINEAR'

        else:
            if self.demInterpolationMethod.upper() not in list(self.interpolationMethods.keys()):
                raise Exception ('Interpolation method must be one of ' + str(list(self.interpolationMethods.keys())))

    def updateDefaults(self):
        if self.demLength is None:
            self.demLength = self.demImage.getLength()

        if self.demWidth is None:
            self.demWidth = self.demImage.getWidth()

    def destroyImages(self):
        self.latImage.addDescription('Pixel-by-pixel latitude in degrees.')
        self.latImage.finalizeImage()
        self.latImage.renderHdr()

        self.lonImage.addDescription('Pixel-by-pixel longitude in degrees.')
        self.lonImage.finalizeImage()
        self.lonImage.renderHdr()


        self.heightRImage.addDescription('Pixel-by-pixel height in meters.')
        self.heightRImage.finalizeImage()
        self.heightRImage.renderHdr()
        self.heightSchImage.addDescription('Pixel-by-pixel height above local sphere in meters.')
        self.heightSchImage.finalizeImage()
        self.heightSchImage.renderHdr()

        descr = '''Two channel Line-Of-Sight geometry image (all angles in degrees). Represents vector drawn from target to platform.
                Channel 1: Incidence angle measured from vertical at target (always +ve).
                Channel 2: Azimuth angle measured from North in Anti-clockwise direction.'''
        self.losImage.setImageType('bil')
        self.losImage.addDescription(descr)
        self.losImage.finalizeImage()
        self.losImage.renderHdr()

        #finalizing of the images handled here
        self.demImage.finalizeImage()
        #self.intImage.finalizeImage()
        if self.incImage:
            self.incImage.finalizeImage()
            self.incImage.renderHdr()

        self.polyDoppler.finalize()


    def createImages(self):

        #assume that even if an image is passed, the createImage and finalizeImage are called here
        if self.demImage is None and not self.demFilename == '':
            self.demImage = IF.createDemImage()
            demAccessMode = 'read'
            demWidth = self.demWidth
            self.demImage.initImage(self.demFilename,demAccessMode,demWidth)
        elif self.demImage is None:#this should never happen, atleast when using the  correct method. same for other images

            self.logger.error('Must either pass the demImage in the call or set self.demFilename.')
            raise Exception

        if(self.latImage == None and not self.latFilename == ''):
            self.latImage = IF.createImage()
            accessMode = 'write'
            dataType = 'DOUBLE'
            width = self.width
            self.latImage.initImage(self.latFilename,accessMode,width,dataType)
        elif(self.latImage == None):
            self.logger.error('Must either pass the latImage in the call or set self.latFilename.')
            raise Exception

        if(self.lonImage == None and not self.lonFilename == ''):
            self.lonImage = IF.createImage()
            accessMode = 'write'
            dataType = 'DOUBLE'
            width = self.width
            self.lonImage.initImage(self.lonFilename,accessMode,width,dataType)
        elif(self.lonImage == None):
            self.logger.error('Must either pass the lonImage in the call or set self.lonFilename.')
            raise Exception

        if(self.heightRImage == None and not self.heightRFilename == ''):
            self.heightRImage = IF.createImage()
            accessMode = 'write'
            dataType = 'FLOAT'
            width = self.width
            self.heightRImage.initImage(self.heightRFilename,accessMode,width,dataType)
        elif(self.heightRImage == None):
            self.logger.error('Must either pass the heightRImage in the call or set self.heightRFilename.')
            raise Exception

        if(self.heightSchImage == None and not self.heightSchFilename == ''):
            self.heightSchImage = IF.createImage()
            accessMode = 'write'
            dataType = 'FLOAT'
            width = self.width
            self.heightSchImage.initImage(self.heightSchFilename,accessMode,width,dataType)
        elif(self.heightSchImage == None):
            self.logger.error('Must either pass the heightSchImage in the call or set self.heightSchFilename.')
            raise Exception

        if(self.losImage == None and not self.losFilename == ''):
            self.losImage = IF.createImage()
            accessMode = 'write'
            dataType ='FLOAT'
            bands = 2
            scheme = 'BIL'
            width = self.width
            self.losImage.initImage(self.losFilename,accessMode,width,dataType,bands=bands,scheme=scheme)

        if (self.incImage == None and not self.incFilename == ''):
            self.incImage = IF.createImage()
            accessMode = 'write'
            dataType = 'FLOAT'
            bands = 1
            scheme = 'BIL'
            width = self.width
            self.incImage.initImage(self.incFilename,accessMode,width,dataType,bands=bands,scheme=scheme)

        #self.intImage.createImage()
        #the dem image could have different datatype so create a caster here
        #the short is the data type used in the fortran.
        self.demImage.setCaster('read','FLOAT')
        self.demImage.createImage()
        self.latImage.createImage()
        self.lonImage.createImage()
        self.heightRImage.createImage()
        self.heightSchImage.createImage()
        self.losImage.createImage()

        if self.incImage:
            self.incImage.createImage()

    def setState(self):
        topo.setNumberIterations_Py(int(self.numberIterations))
        topo.setDemWidth_Py(int(self.demWidth))
        topo.setDemLength_Py(int(self.demLength))
        topo.setReferenceOrbit_Py(self.referenceOrbit, self.dim1_referenceOrbit)
        topo.setFirstLatitude_Py(float(self.firstLatitude))
        topo.setFirstLongitude_Py(float(self.firstLongitude))
        topo.setDeltaLatitude_Py(float(self.deltaLatitude))
        topo.setDeltaLongitude_Py(float(self.deltaLongitude))
        topo.setISMocomp_Py(int(self.isMocomp))
        topo.setEllipsoidMajorSemiAxis_Py(float(self.ellipsoidMajorSemiAxis))
        topo.setEllipsoidEccentricitySquared_Py(float(self.ellipsoidEccentricitySquared))
        topo.setLength_Py(int(self.length))
        topo.setWidth_Py(int(self.width))
        topo.setRangePixelSpacing_Py(float(self.slantRangePixelSpacing))
        topo.setRangeFirstSample_Py(float(self.rangeFirstSample))
        topo.setSpacecraftHeight_Py(float(self.spacecraftHeight))
        topo.setPlanetLocalRadius_Py(float(self.planetLocalRadius))
        topo.setBodyFixedVelocity_Py(float(self.bodyFixedVelocity))
        topo.setNumberRangeLooks_Py(int(self.numberRangeLooks))
        topo.setNumberAzimuthLooks_Py(int(self.numberAzimuthLooks))
        topo.setPegLatitude_Py(float(self.pegLatitude))
        topo.setPegLongitude_Py(float(self.pegLongitude))
        topo.setPegHeading_Py(float(self.pegHeading))
        topo.setPRF_Py(float(self.prf))
        topo.setRadarWavelength_Py(float(self.radarWavelength))
        topo.setLatitudePointer_Py(int(self.latAccessor))
        topo.setLongitudePointer_Py(int(self.lonAccessor))
        topo.setHeightRPointer_Py(int(self.heightRAccessor))
        topo.setHeightSchPointer_Py(int(self.heightSchAccessor))
        topo.setIncPointer_Py(int(self.incAccessor))
        topo.setLosPointer_Py(int(self.losAccessor))
        topo.setLookSide_Py(int(self.lookSide))

        tstart = DTU.seconds_since_midnight(self.sensingStart) + (self.numberAzimuthLooks-1)/(2.0 * self.prf)
        topo.setSensingStart_Py(tstart)

        intpKey = self.interpolationMethods[self.demInterpolationMethod.upper()]
        topo.setMethod_Py(int(intpKey))
        return None


    def setNumberIterations(self,var):
        self.numberIterations = int(var)
        return None

    def setDemWidth(self,var):
        self.demWidth = int(var)
        return None

    def setDemLength(self,var):
        self.demLength = int(var)
        return None

    def setReferenceOrbit(self,var):
        self.referenceOrbit = var
        return None

    def setFirstLatitude(self,var):
        self.firstLatitude = float(var)
        return None

    def setFirstLongitude(self,var):
        self.firstLongitude = float(var)
        return None

    def setDeltaLatitude(self,var):
        self.deltaLatitude = float(var)
        return None

    def setDeltaLongitude(self,var):
        self.deltaLongitude = float(var)
        return None

    def setISMocomp(self,var):
        self.isMocomp = int(var)
        return None

    def setEllipsoidMajorSemiAxis(self,var):
        self.ellipsoidMajorSemiAxis = float(var)
        return None

    def setEllipsoidEccentricitySquared(self,var):
        self.ellipsoidEccentricitySquared = float(var)
        return None

    def setLength(self,var):
        self.length = int(var)
        return None

    def setWidth(self,var):
        self.width = int(var)
        return None

    def setRangePixelSpacing(self,var):
        self.slantRangePixelSpacing = float(var)
        return None

    def setRangeFirstSample(self,var):
        self.rangeFirstSample = float(var)
        return None

    def setSpacecraftHeight(self,var):
        self.spacecraftHeight = float(var)
        return None

    def setPlanetLocalRadius(self,var):
        self.planetLocalRadius = float(var)
        return None

    def setBodyFixedVelocity(self,var):
        self.bodyFixedVelocity = float(var)
        return None

    def setNumberRangeLooks(self,var):
        self.numberRangeLooks = int(var)
        return None

    def setNumberAzimuthLooks(self,var):
        self.numberAzimuthLooks = int(var)
        return None

    def setPegLatitude(self,var):
        self.pegLatitude = float(var)
        return None

    def setPegLongitude(self,var):
        self.pegLongitude = float(var)
        return None

    def setPegHeading(self,var):
        self.pegHeading = float(var)
        return None

    def setDopplerCentroidConstantTerm(self,var):
        self.dopplerCentroidConstantTerm = float(var)
        return None

    def setPolyDoppler(self,var):
        self.polyDoppler = var.copy()
        return None

    def setPRF(self,var):
        self.prf = float(var)
        return None

    def setRadarWavelength(self,var):
        self.radarWavelength = float(var)
        return None

    def setLosFilename(self,var):
        self.losFilename = var
        return None

    def setLatFilename(self,var):
        self.latFilename = var
        return None

    def setLonFilename(self,var):
        self.lonFilename = var
        return None

    def setHeightRFilename(self,var):
        self.heightRFilename = var
        return None

    def setHeightSchFilename(self,var):
        self.heightSchFilename = var
        return None

    def setIncidenceFilename(self,var):
        self.incFilename = var
        return None

    def setLookSide(self,var):
        self.lookSide = int(var)
        return None

    def getState(self):
        self.azimuthSpacing = topo.getAzimuthSpacing_Py()
        self.planetLocalRadius = topo.getPlanetLocalRadius_Py()
        self.sCoordinateFirstLine = topo.getSCoordinateFirstLine_Py()
        self.sCoordinateLastLine = topo.getSCoordinateLastLine_Py()
        self.minimumLatitude = topo.getMinimumLatitude_Py()
        self.minimumLongitude = topo.getMinimumLongitude_Py()
        self.maximumLatitude = topo.getMaximumLatitude_Py()
        self.maximumLongitude = topo.getMaximumLongitude_Py()
        self.squintshift = topo.getSquintShift_Py(self.dim1_squintshift)
        self.length = topo.getLength_Py()

        return None

    def getAzimuthSpacing(self):
        return self.azimuthSpacing

    def getPlanetLocalRadius(self):
        return self.planetLocalRadius

    def getSCoordinateFirstLine(self):
        return self.sCoordinateFirstLine

    def getSCoordinateLastLine(self):
        return self.sCoordinateLastLine

    def getMinimumLatitude(self):
        return self.minimumLatitude

    def getMinimumLongitude(self):
        return self.minimumLongitude

    def getMaximumLatitude(self):
        return self.maximumLatitude

    def getMaximumLongitude(self):
        return self.maximumLongitude

    def getSquintShift(self):
        return self.squintshift

    def allocateArrays(self):
        if (self.dim1_referenceOrbit == None):
            self.dim1_referenceOrbit = len(self.referenceOrbit)

        if (not self.dim1_referenceOrbit):
            print("Error. Trying to allocate zero size array")

            raise Exception

        topo.allocate_s_mocompArray_Py(self.dim1_referenceOrbit)

        if (self.dim1_squintshift == None):
            self.dim1_squintshift = len(self.squintshift)

        if (not self.dim1_squintshift):
            print("Error. Trying to allocate zero size array")

            raise Exception

        topo.allocate_squintshift_Py(self.dim1_squintshift)

        return None

    def deallocateArrays(self):
        topo.deallocate_s_mocompArray_Py()
        topo.deallocate_squintshift_Py()
        return None

    def addPeg(self):
        peg = self._inputPorts.getPort(name='peg').getObject()
        if (peg):
            try:
                self.planetLocalRadius = peg.getRadiusOfCurvature()
                self.pegLatitude = math.radians(peg.getLatitude())
                self.pegLongitude = math.radians(peg.getLongitude())
                self.pegHeading = math.radians(peg.getHeading())
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
                #self.rangeFirstSample = frame.getStartingRange() - Piyush
                instrument = frame.getInstrument()
                self.slantRangePixelSpacing = instrument.getRangePixelSize()
                self.prf = instrument.getPulseRepetitionFrequency()
                self.radarWavelength = instrument.getRadarWavelength()
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError

    def addReferenceSlc(self):     #Piyush
        formslc = self._inputPorts.getPort(name='referenceslc').getObject()

        if (formslc):
            try:
                self.rangeFirstSample = formslc.startingRange
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError

            self.dopplerCentroidCoeffs = formslc.dopplerCentroidCoefficients
            self.orbit = formslc.outOrbit
            self.sensingStart = formslc.slcSensingStart

    def addDEM(self):
        dem = self._inputPorts.getPort(name='dem').getObject()
        if (dem):
            try:
                self.demImage = dem
                self.demWidth = dem.getWidth()
                self.demLength = dem.getLength()
                self.firstLatitude = dem.getFirstLatitude()
                self.firstLongitude = dem.getFirstLongitude()
                self.deltaLatitude = dem.getDeltaLatitude()
                self.deltaLongitude = dem.getDeltaLongitude()
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError

    def addInterferogram(self):
        ifg = self._inputPorts.getPort(name='interferogram').getObject()
        if (ifg):
            try:
                self.intImage = ifg
                self.width = ifg.getWidth()
                self.length = ifg.getLength()
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError

    logging_name = "isce.stdproc.topo"

    family = 'topo'

    def __init__(self,family='',name=''):
        super(Topo, self).__init__(family if family else  self.__class__.family, name=name)
        self.demInterpolationMethod = demInterpolationMethod
        self.dim1_referenceOrbit = None
        self.demFilename = ''
        self.latFilename = ''
        self.lonFilename = ''
        self.heightRFilename = ''
        self.heightSchFilename = ''
        self.losFilename = ''
        self.incFilename = ''
        self.demImage = None
        self.latImage = None
        self.lonImage = None
        self.heightRImage = None
        self.heightSchImage = None
        self.losImage = None
        self.incImage = None
        self.demAccessor = None
        self.incAccessor = None
        self.losAccessor = None
        self.dim1_squintshift = None
        self.lookSide = -1     #Default set to right side
        self.polyDoppler = None
        self.polyDopplerAccessor = None



        ####For dumping and loading
        self._times = []
        self._fmt = '%Y-%m-%dT%H:%M:%S.%f'

        self.initOptionalAndMandatoryLists()
        return None

    def createPorts(self):
        self.inputPorts['peg'] = self.addPeg
        self.inputPorts['frame'] = self.addFrame
        self.inputPorts['planet'] = self.addPlanet
        self.inputPorts['dem'] = self.addDEM
        self.inputPorts['interferogram'] = self.addInterferogram
        slcPort = Port(name='referenceslc', method=self.addReferenceSlc)  #Piyush
        self.inputPorts.add(slcPort)     #Piyush
        return None


    def adaptToRender(self):
        import copy
        # make a copy of the stateVectors to restore it after dumping
        self._times = [copy.copy(self.sensingStart)]
        self.sensingStart = self.sensingStart.strftime(self._fmt)

    def restoreAfterRendering(self):
        self.sensingStart = self._times[0]

    def initProperties(self,catalog):
        keys = ['SENSING_START']

        for k in keys:
            kl = k.lower()
            if kl in catalog:
                v = catalog[kl]
                attrname = getattr(globals()[k],'attrname')
                val = datetime.datetime.strptime(v,self._fmt)
                setattr(self,attrname,val)
                catalog.pop(kl)
        super().initProperties(catalog)


    pass
