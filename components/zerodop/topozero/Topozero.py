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
from zerodop.topozero import topozero
from isceobj.Util import combinedlibmodule
from iscesys import DateTimeUtil as DTU
from isceobj.Planet import Ellipsoid
import datetime
from isceobj.Util import Poly2D
import numpy as np

class Topo(Component):

    interpolationMethods = { 'SINC' : 0,
                             'BILINEAR' : 1,
                             'BICUBIC' : 2,
                             'NEAREST' : 3,
                             'AKIMA' : 4,
                             'BIQUINTIC' : 5}

    orbitInterpolationMethods = { 'HERMITE' : 0,
                                  'SCH'     : 1,
                                  'LEGENDRE': 2}

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

        self.demAccessor = self.demImage.getImagePointer()
        self.latAccessor = self.latImage.getImagePointer()
        self.lonAccessor = self.lonImage.getImagePointer()
        self.heightAccessor = self.heightImage.getImagePointer()
        self.losAccessor = self.losImage.getImagePointer()

        if isinstance(self.slantRangeImage, Poly2D.Poly2D):
            self.slantRangeImage.createPoly2D()
            self.slantRangeAccessor = self.slantRangeImage.getPointer()
        else:
            self.slantRangeAccessor = self.slantRangeImage.getImagePointer()

        if self.incImage:
            self.incAccessor = self.incImage.getImagePointer()
        else:
            self.incAccessor = 0

        if self.maskImage:
            self.maskAccessor = self.maskImage.getImagePointer()
        else:
            self.maskAccessor = 0


        self.polyDoppler.createPoly2D()
        self.polyDopplerAccessor = self.polyDoppler.getPointer()

        self.setState()

        cOrbit = self.orbit.exportToC(reference=self.sensingStart)
        topozero.setOrbit_Py(cOrbit)
        topozero.topo_Py(self.demAccessor, self.polyDopplerAccessor, self.slantRangeAccessor)
        combinedlibmodule.freeCOrbit(cOrbit)

        self.getState()
        self.destroyImages()

        return None

    def setDefaults(self):
        if self.ellipsoidMajorSemiAxis is None:
            self.ellipsoidMajorSemiAxis = CN.EarthMajorSemiAxis

        if self.ellipsoidEccentricitySquared is None:
            self.ellipsoidEccentricitySquared = CN.EarthEccentricitySquared

        if self.numberIterations is None:
            self.numberIterations = 25

        if self.secondaryIterations is None:
            self.secondaryIterations = 10

        if self.threshold is None:
            self.threshold = 0.05
        
        if self.heightFilename == '':
            self.heightFilename = 'z.rdr'
            self.logger.warning('The real height file has been given the default name %s' % (self.heightFilename))
        if self.latFilename == '':
            self.latFilename = 'lat.rdr'
            self.logger.warning('The latitude file has been given the default name %s' % (self.latFilename))
        if self.lonFilename == '':
            self.lonFilename = 'lon.rdr'
            self.logger.warning('The longitude file has been given the default name %s' % (self.lonFilename))
        if self.losFilename == '':
            self.losFilename = 'los.rdr'
            self.logger.warning('The los file has been given the default name %s' % (self.losFilename))

        if self.pegHeading is None:
            ###Compute the peg value here and set it
            tbef = self.sensingStart + datetime.timedelta(seconds=(0.5*self.length / self.prf))
            self.pegHeading = np.radians(self.orbit.getENUHeading(tbef))

            self.logger.warning('Default Peg heading set to: ' + str(self.pegHeading))

        if self.polyDoppler is None:
            self.polyDoppler = Poly2D.Poly2D(name=self.name+'_dopplerPoly')
            self.polyDoppler.setWidth(self.width)
            self.polyDoppler.setLength(self.length)
            self.polyDoppler.setNormRange(1.0)
            self.polyDoppler.setNormAzimuth(1.0)
            self.polyDoppler.setMeanRange(0.0)
            self.polyDoppler.setMeanAzimuth(0.0)
            self.polyDoppler.initPoly(rangeOrder=0, azimuthOrder=0, coeffs=[[0.0]])
        else:
            if self.polyDoppler.getWidth() != self.width:
                raise Exception('Doppler Centroid object does not have the same width as input image')

            if self.polyDoppler.getLength() != self.length:
                raise Exception('Doppler Centroid object does not have the same length as input image')

        if self.demInterpolationMethod is None:
            self.demInterpolationMethod = 'BILINEAR'

        else:
            if self.demInterpolationMethod.upper() not in list(self.interpolationMethods.keys()):
                raise Exception ('Interpolation method must be one of ' + str(list(self.interpolationMethods.keys())))

        if self.orbitInterpolationMethod is None:
            self.orbitInterpolationMethod = 'HERMITE'
        else:
            if self.orbitInterpolationMethod.upper() not in list(self.orbitInterpolationMethods.keys()):
                raise Exception('Orbit interpolation method must be one of ' + str(list(self.demInterpolationMethods.keys())))

        ###Slant range settings
        if self.slantRangeFilename in ['',None]:
            if self.slantRangePixelSpacing is None:
                raise Exception('No slant range file provided. slantRangePixelSpacing cannot be None')    

            if self.rangeFirstSample is None:
                raise Exception('No slant range file provided. rangeFirstSample cannot be None')


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
        
        
        self.heightImage.addDescription('Pixel-by-pixel height in meters.')
        self.heightImage.finalizeImage()
        self.heightImage.renderHdr()
        
        descr = '''Two channel Line-Of-Sight geometry image (all angles in degrees). Represents vector drawn from target to platform. 
                Channel 1: Incidence angle measured from vertical at target (always +ve).
                Channel 2: Azimuth angle measured from North in Anti-clockwise direction.'''
        self.losImage.setImageType('bil')
        self.losImage.addDescription(descr)
        self.losImage.finalizeImage()
        self.losImage.renderHdr()
    
        #finalizing of the images handled here
        self.demImage.finalizeImage()

        if self.incImage:
            descr = '''Two channel angle file.
                    Channel 1: Angle between ray to target and the vertical at the sensor
                    Channel 2: Local incidence angle accounting for DEM slope at target'''
            self.incImage.addDescription(descr)
            self.incImage.finalizeImage()
            self.incImage.renderHdr()

        if self.maskImage:
            descr = 'Radar shadow-layover mask. 1 - Radar Shadow. 2 - Radar Layover. 3 - Both.'
            self.maskImage.addDescription(descr)
            self.maskImage.finalizeImage()
            self.maskImage.renderHdr()

        if self.slantRangeImage:
            try:
                self.slantRangeImage.finalizeImage()
            except:
                pass

        return

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

        if(self.heightImage == None and not self.heightFilename == ''):
            self.heightImage = IF.createImage()
            accessMode = 'write'
            dataType = 'DOUBLE'
            width = self.width
            self.heightImage.initImage(self.heightFilename,accessMode,width,dataType)
        elif(self.heightImage == None):
            self.logger.error('Must either pass the heightImage in the call or set self.heightFilename.')
            raise Exception

        ####User provided an input file name for slant range to work with
        if(self.slantRangeImage == None and not self.slantRangeFilename == ''):

            if self.rangeFirstSample:
                raise Exception('Cannot provide both slant range image and range first sample as input')

            if self.slantRangePixelSpacing:
                raise Exception('Cannot provide both slant range image and slant range pixel spacing as input')

            self.slantRangeImage = IF.createImage()
            self.slantRangeImage.load(self.slantRangeFilename + '.xml')
            self.slantRangeImage.setAccessMode = 'READ'

            if self.slantRangeImage.width != self.width:
                raise Exception('Slant Range Image width {0} does not match input width {1}'.format(self.slantRangeImage.width, self.width))

            if self.slantRangeImage.length != self.length:
                raise Exception('Slant Range Image length {0} does not match input length {1}'.format(self.slantRangeImage.length, self.length))

            self.slantRangeImage.createImage()
            ###Set these to zero since not used but bindings need it - PSA
            self.rangeFirstSample = 0.0
            self.slantRangePixelSpacing = 0.0

        ####User provided an image like object (maybe polynomial)
        elif self.slantRangeImage is not None:
            if self.slantRangeImage.width != self.width:
                raise Exception('Slant Range Image width {0} does not match input width {1}'.format(self.slantRangeImage.width, self.width))

            if self.slantRangeImage.length != self.length:
                raise Exception('Slant Range Image length {0} does not match input length {1}'.format(self.slantRangeImage.length, self.length))

        #####Standard operation
        else:
            r0 = self.rangeFirstSample
            dr = self.slantRangePixelSpacing*self.numberRangeLooks
            self.slantRangeImage = Poly2D.Poly2D()
            self.slantRangeImage.setWidth(self.width)
            self.slantRangeImage.setLength(self.length)
            self.slantRangeImage.setNormRange(1.0)
            self.slantRangeImage.setNormAzimuth(1.0)
            self.slantRangeImage.setMeanRange(0.0)
            self.slantRangeImage.setMeanAzimuth(0.0)
            self.slantRangeImage.initPoly(rangeOrder=1, azimuthOrder=0, coeffs=[[r0,dr]])



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
            bands = 2
            scheme = 'BIL'
            width = self.width
            self.incImage.initImage(self.incFilename, accessMode, width, dataType, bands=bands, scheme=scheme)

        if (self.maskImage == None and not self.maskFilename == ''):
            self.maskImage = IF.createImage()
            accessMode = 'write'
            dataType = 'BYTE'
            bands = 1
            scheme = 'BIL'
            width = self.width
            self.maskImage.initImage(self.maskFilename, accessMode, width, dataType, bands=bands, scheme=scheme)

        #the dem image could have different datatype so create a caster here
        #the short is the data type used in the fortran. 
        self.demImage.setCaster('read','FLOAT')
        self.demImage.createImage()
        self.latImage.createImage()
        self.lonImage.createImage()
        self.heightImage.createImage()
        self.losImage.createImage()

        if self.incImage:
            self.incImage.createImage()

        if self.maskImage:
            self.maskImage.createImage()

        return
    
    def setState(self):
        topozero.setNumberIterations_Py(int(self.numberIterations))
        topozero.setSecondaryIterations_Py(int(self.secondaryIterations))
        topozero.setThreshold_Py(float(self.threshold))
        topozero.setDemWidth_Py(int(self.demWidth))
        topozero.setDemLength_Py(int(self.demLength))
        topozero.setFirstLatitude_Py(float(self.firstLatitude))
        topozero.setFirstLongitude_Py(float(self.firstLongitude))
        topozero.setDeltaLatitude_Py(float(self.deltaLatitude))
        topozero.setDeltaLongitude_Py(float(self.deltaLongitude))
        topozero.setEllipsoidMajorSemiAxis_Py(float(self.ellipsoidMajorSemiAxis))
        topozero.setEllipsoidEccentricitySquared_Py(float(self.ellipsoidEccentricitySquared))
        topozero.setPegHeading_Py(float(self.pegHeading))
        topozero.setLength_Py(int(self.length))
        topozero.setWidth_Py(int(self.width))
        topozero.setRangePixelSpacing_Py(float(self.slantRangePixelSpacing))
        topozero.setRangeFirstSample_Py(float(self.rangeFirstSample))
        topozero.setNumberRangeLooks_Py(int(self.numberRangeLooks))
        topozero.setNumberAzimuthLooks_Py(int(self.numberAzimuthLooks))
        topozero.setPRF_Py(float(self.prf))
        topozero.setRadarWavelength_Py(float(self.radarWavelength))
        topozero.setLatitudePointer_Py(int(self.latAccessor))
        topozero.setLongitudePointer_Py(int(self.lonAccessor))
        topozero.setHeightPointer_Py(int(self.heightAccessor))
        topozero.setLosPointer_Py(int(self.losAccessor))
        topozero.setIncPointer_Py(int(self.incAccessor))
        topozero.setMaskPointer_Py(int(self.maskAccessor))
        topozero.setLookSide_Py(int(self.lookSide))
        topozero.setSensingStart_Py(DTU.seconds_since_midnight(self.sensingStart))
        
        intpKey = self.interpolationMethods[self.demInterpolationMethod.upper()]
        topozero.setMethod_Py(int(intpKey))

        orbitIntpKey = self.orbitInterpolationMethods[self.orbitInterpolationMethod.upper()]
        topozero.setOrbitMethod_Py(int(orbitIntpKey))
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

    def setOrbit(self,var):
        self.orbit = var
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

    def setNumberRangeLooks(self,var):
        self.numberRangeLooks = int(var)
        return None

    def setNumberAzimuthLooks(self,var):
        self.numberAzimuthLooks = int(var)
        return None

    def setPegHeading(self,var):
        self.pegHeading = float(var)
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

    def setHeightFilename(self,var):
        self.heightFilename = var
        return None
    
    def setIncidenceFilename(self,var):
        self.incFilename = var
        return None

    def setMaskFilename(self, var):
        self.maskFilename = var
        return None

    def setLookSide(self,var):
        self.lookSide = int(var)
        return None

    def setPolyDoppler(self, var):
        self.polyDoppler = var.copy()
        return None

    def getState(self):
        self.minimumLatitude = topozero.getMinimumLatitude_Py()
        self.minimumLongitude = topozero.getMinimumLongitude_Py()
        self.maximumLatitude = topozero.getMaximumLatitude_Py()
        self.maximumLongitude = topozero.getMaximumLongitude_Py()
        return None

    def getMinimumLatitude(self):
        return self.minimumLatitude

    def getMinimumLongitude(self):
        return self.minimumLongitude

    def getMaximumLatitude(self):
        return self.maximumLatitude

    def getMaximumLongitude(self):
        return self.maximumLongitude

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
                self.orbit = frame.getOrbit()
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError

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

    logging_name = "isce.zerodop.topozero"

    def __init__(self):
        super(Topo, self).__init__()
        self.numberIterations = None
        self.secondaryIterations = None
        self.threshold = None
        self.demWidth = None
        self.demLength = None
        self.orbit = None
        self.sensingStart = None
        self.firstLatitude = None
        self.firstLongitude = None
        self.deltaLatitude = None
        self.deltaLongitude = None
        self.ellipsoidMajorSemiAxis = None
        self.ellipsoidEccentricitySquared = None
        self.length = None
        self.width = None
        self.slantRangePixelSpacing = None
        self.rangeFirstSample = None
        self.numberRangeLooks = None
        self.numberAzimuthLooks = None
        self.pegHeading = None
        self.prf = None
        self.sensingStart = None
        self.radarWavelength = None
        self.demFilename = ''
        self.latFilename = ''
        self.lonFilename = ''
        self.heightFilename = ''
        self.losFilename = ''
        self.incFilename = ''
        self.maskFilename = ''
        self.slantRangeFilename = ''
        self.demImage = None
        self.latImage = None
        self.lonImage = None
        self.heightImage = None
        self.losImage = None
        self.incImage = None
        self.maskImage = None
        self.slantRangeImage = None
        self.demAccessor = None
        self.latAccessor = None
        self.lonAccessor = None
        self.heightAccessor = None
        self.losAccessor = None
        self.incAccessor = None
        self.maskAccessor = None
        self.slantRangeAccessor = None
        self.minimumLatitude = None
        self.minimumLongitude = None
        self.maximumLatitude = None
        self.maximumLongitude = None
        self.lookSide = None     #Default set to right side
        self.polyDoppler = None
        self.polyDopplerAccessor = None
        self.demInterpolationMethod = None
        self.orbitInterpolationMethod = None
        self.dictionaryOfVariables = { 
            'NUMBER_ITERATIONS' : ['numberIterations', 'int','optional'], 
            'DEM_WIDTH' : ['demWidth', 'int','mandatory'], 
            'DEM_LENGTH' : ['demLength', 'int','mandatory'], 
            'FIRST_LATITUDE' : ['firstLatitude', 'float','mandatory'], 
            'FIRST_LONGITUDE' : ['firstLongitude', 'float','mandatory'], 
            'DELTA_LATITUDE' : ['deltaLatitude', 'float','mandatory'], 
            'DELTA_LONGITUDE' : ['deltaLongitude', 'float','mandatory'], 
            'ELLIPSOID_MAJOR_SEMIAXIS' : ['ellipsoidMajorSemiAxis', 'float','optional'], 
            'ELLIPSOID_ECCENTRICITY_SQUARED' : ['ellipsoidEccentricitySquared', 'float','optional'], 
            'LENGTH' : ['length', 'int','mandatory'], 
            'WIDTH' : ['width', 'int','mandatory'], 
            'SLANT_RANGE_PIXEL_SPACING' : ['slantRangePixelSpacing', 'float','mandatory'], 
            'RANGE_FIRST_SAMPLE' : ['rangeFirstSample', 'float','mandatory'], 
            'NUMBER_RANGE_LOOKS' : ['numberRangeLooks', 'int','mandatory'], 
            'NUMBER_AZIMUTH_LOOKS' : ['numberAzimuthLooks', 'int','mandatory'], 
            'PEG_HEADING' : ['pegHeading', 'float','mandatory'], 
            'PRF' : ['prf', 'float','mandatory'], 
            'RADAR_WAVELENGTH' : ['radarWavelength', 'float','mandatory'], 
            'LAT_ACCESSOR' : ['latAccessor', 'int','optional'], 
            'LON_ACCESSOR' : ['lonAccessor', 'int','optional'], 
            'HEIGHT_R_ACCESSOR' : ['heightAccessor', 'int','optional'], 
            }
        self.dictionaryOfOutputVariables = { 
            'MINIMUM_LATITUDE' : 'minimumLatitude', 
            'MINIMUM_LONGITUDE' : 'minimumLongitude', 
            'MAXIMUM_LATITUDE' : 'maximumLatitude', 
            'MAXIMUM_LONGITUDE' : 'maximumLongitude', 
            }
        self.descriptionOfVariables = {}
        self.mandatoryVariables = []
        self.optionalVariables = []
        self.initOptionalAndMandatoryLists()
        return None
    
    def createPorts(self):
        self.inputPorts['frame'] = self.addFrame
        self.inputPorts['planet'] = self.addPlanet
        self.inputPorts['dem'] = self.addDEM
        self.inputPorts['interferogram'] = self.addInterferogram
        return None

    pass



