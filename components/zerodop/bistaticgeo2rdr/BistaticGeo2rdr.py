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
from zerodop.bistaticgeo2rdr import bistaticgeo2rdr
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from isceobj.Util import combinedlibmodule
from isceobj.Util.Poly1D import Poly1D
import os
import datetime
import sys

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

ACTIVE_RANGE_FIRST_SAMPLE = Component.Parameter('activeRangeFirstSample',
        public_name = 'ACTIVE_RANGE_FIRST_SAMPLE',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Range to first sample')

PASSIVE_RANGE_FIRST_SAMPLE = Component.Parameter('passiveRangeFirstSample',
        public_name = 'ACTIVE_RANGE_FIRST_SAMPLE',
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

RANGE_FILENAME = Component.Parameter('rangeFilename',
        public_name = 'RANGE_FILENAME',
        default=None,
        type=str,
        mandatory=True,
        doc = 'Filename of the output range in meters')

AZIMUTH_FILENAME = Component.Parameter('azimuthFilename',
        public_name = 'AZIMUTH_FILENAME',
        default = None,
        type=str,
        mandatory=True,
        doc = 'Filename of the output azimuth in seconds')

RANGE_OFFSET_FILENAME = Component.Parameter('rangeOffFilename',
        public_name = 'RANGE_OFFSET_FILENAME',
        default = None,
        type=str,
        mandatory = True,
        doc = 'Filename of the output range offsets for use with resamp')

AZIMUTH_OFFSET_FILENAME = Component.Parameter('azimuthOffFilename',
        public_name = 'AZIMUTH_OFFSET_FILENAME',
        default = None,
        type=str,
        mandatory=True,
        doc = 'Filename of the output azimuth offsets for use with resamp')

LOOK_SIDE = Component.Parameter('lookSide',
        public_name = 'LOOK_SIDE',
        default = None,
        type = int,
        mandatory = True,
        doc = 'Right (-1) / Left (1) . Look direction of the radar platform')

BISTATIC_DELAY_CORRECTION_FLAG = Component.Parameter('bistaticDelayCorrectionFlag',
        public_name = 'BISTATIC_DELAY_CORRECTION_FLAG',
        default = None,
        type = bool,
        mandatory = True,
        doc = 'Include bistatic delay correction term. E.g: ASAR / ALOS-1')

OUTPUT_PRECISION = Component.Parameter('outputPrecision',
        public_name = 'OUTPUT_PRECISION',
        default = 'single',
        type = bool,
        mandatory = True,
        doc = 'Set to double for double precision offsets / coordinates. Angles are always single precision.')

ORBIT_INTERPOLATION_METHOD = Component.Parameter('orbitInterpolationMethod',
        public_name="orbit interpolation method",
        default = None,
        type = str,
        mandatory = True,
        doc = 'Set to HERMITE/ SCH / LEGENDRE')

class BistaticGeo2rdr(Component):

    family = 'bistaticgeo2rdr'
    logging_name = 'isce.zerodop.bistaticgeo2rdr'


    parameter_list = (RANGE_FILENAME,
                      AZIMUTH_FILENAME,
                      RANGE_OFFSET_FILENAME,
                      AZIMUTH_OFFSET_FILENAME,
                      SLANT_RANGE_PIXEL_SPACING,
                      ELLIPSOID_ECCENTRICITY_SQUARED,
                      ELLIPSOID_MAJOR_SEMIAXIS,
                      ACTIVE_RANGE_FIRST_SAMPLE,
                      PASSIVE_RANGE_FIRST_SAMPLE,
                      SENSING_START,
                      NUMBER_RANGE_LOOKS,
                      NUMBER_AZIMUTH_LOOKS,
                      PRF,
                      RADAR_WAVELENGTH,
                      LOOK_SIDE,
                      BISTATIC_DELAY_CORRECTION_FLAG,
                      OUTPUT_PRECISION,
                      ORBIT_INTERPOLATION_METHOD)

    orbitMethods = { 'HERMITE' : 0,
                     'SCH'     : 1,
                     'LEGENDRE': 2 }

    #####Actual geocoding
    def bistaticgeo2rdr(self, latImage=None, lonImage=None, demImage=None):
        self.activateInputPorts()

        if latImage is not None:
            self.latImage = latImage

        if lonImage is not None:
            self.lonImage = lonImage

        if demImage is not None:
            self.demImage = demImage

        if self.activeOrbit is None:
            raise Exception('No active orbit provided for geocoding')

        if self.passiveOrbit is None:
            raise Exception('No passive orbit provided for geocoding')

        self.setDefaults()
        self.createImages()
        self.setState()

        #this inits the image in the c++ bindings
        self.demImage.setCaster('read','DOUBLE')
        self.demImage.createImage()
        demAccessor = self.demImage.getImagePointer()

        self.latImage.createImage()
        latAccessor = self.latImage.getImagePointer()

        self.lonImage.createImage()
        lonAccessor = self.lonImage.getImagePointer()

        
        ####Get output accessor
        rangeAcc = 0
        if self.rangeImage is not None:
            rangeAcc = self.rangeImage.getImagePointer()
        
        azimuthAcc = 0
        if self.azimuthImage is not None:
            azimuthAcc = self.azimuthImage.getImagePointer()

        rangeOffAcc = 0
        if self.rangeOffsetImage is not None:
            rangeOffAcc = self.rangeOffsetImage.getImagePointer()

        azimuthOffAcc = 0
        if self.azimuthOffsetImage is not None:
            azimuthOffAcc = self.azimuthOffsetImage.getImagePointer()


        cActiveOrbit = self.activeOrbit.exportToC()
        bistaticgeo2rdr.setActiveOrbit_Py(cActiveOrbit)

        cPassiveOrbit=self.passiveOrbit.exportToC()
        bistaticgeo2rdr.setPassiveOrbit_Py(cPassiveOrbit)

        #####Output cropped DEM for first band
        bistaticgeo2rdr.bistaticgeo2rdr_Py(latAccessor,
                           lonAccessor,
                           demAccessor,
                           azimuthAcc, rangeAcc,
                           azimuthOffAcc, rangeOffAcc)

        combinedlibmodule.freeCOrbit(cActiveOrbit)
        combinedlibmodule.freeCOrbit(cPassiveOrbit)


        self.destroyImages()
        return None

    def setDefaults(self):
        if self.polyDoppler is None:
            self.polyDoppler = Poly1D(name=self.name+'_geo2rdrPoly')
            self.polyDoppler.setMean(0.0)
            self.polyDoppler.initPoly(order=len(self.dopplerCentroidCoeffs)-1,
                coeffs = self.dopplerCentroidCoeffs)

        if all(v is None for v in [self.rangeImageName, self.azimuthImageName,
                    self.rangeOffsetImageName, self.azimuthOffsetImageName]):
            print('No outputs requested from geo2rdr. Check again.')
            sys.exit(0)

        if self.demWidth is None:
            self.demWidth = self.demImage.width

        if self.demLength is None:
            self.demLength = self.demImage.length

        if any(v != self.demWidth  for v in [self.demImage.width, self.latImage.width, self.lonImage.width]):
            print('Input lat, lon, z images should all have the same width')
            sys.exit(0)

        if any(v != self.demLength for v in [self.demImage.length, self.latImage.length, self.lonImage.length]):
            print('Input lat, lon, z images should all have the same length')
            sys.exit(0)

        if self.bistaticDelayCorrectionFlag is None:
            self.bistaticDelayCorrectionFlag = False
            print('Turning off bistatic delay correction term by default.')

        if self.orbitInterpolationMethod is None:
            self.orbitInterpolationMethod = 'HERMITE'

        pass


    def destroyImages(self):
        from isceobj.Util import combinedlibmodule as CL

        for outfile in [self.rangeImage, self.azimuthImage,
                self.rangeOffsetImage, self.azimuthOffsetImage]:

            if outfile is not None:
                outfile.finalizeImage()
                outfile.renderHdr()

        #####Clean out polynomial object
        CL.freeCPoly1D(self.polyDopplerAccessor)
        self.polyDopplerAccessor = None

        self.latImage.finalizeImage()
        self.lonImage.finalizeImage()
        self.demImage.finalizeImage()

    def createImages(self):
        if self.rangeImageName:
            self.rangeImage = createImage()
            self.rangeImage.setFilename(self.rangeImageName)
            self.rangeImage.setAccessMode('write')

            if self.outputPrecision.upper() == 'SINGLE':
                self.rangeImage.setDataType('FLOAT')
                self.rangeImage.setCaster('write', 'DOUBLE')
            elif self.outputPrecision.upper() == 'DOUBLE':
                self.rangeImage.setDataType('DOUBLE')
            else:
                raise Exception('Undefined output precision for range image in geo2rdr.')

            self.rangeImage.setWidth(self.demWidth)
            self.rangeImage.createImage()

        if self.rangeOffsetImageName:
            self.rangeOffsetImage = createImage()
            self.rangeOffsetImage.setFilename(self.rangeOffsetImageName)
            self.rangeOffsetImage.setAccessMode('write')
            
            if self.outputPrecision.upper() == 'SINGLE':
                self.rangeOffsetImage.setDataType('FLOAT')
                self.rangeOffsetImage.setCaster('write', 'DOUBLE')
            elif self.outputPrecision.upper() == 'DOUBLE':
                self.rangeOffsetImage.setDataType('DOUBLE')
            else:
                raise Exception('Undefined output precision for range offset image in geo2rdr.')


            self.rangeOffsetImage.setWidth(self.demWidth)
            self.rangeOffsetImage.createImage()

        if self.azimuthImageName:
            self.azimuthImage = createImage()
            self.azimuthImage.setFilename(self.azimuthImageName)
            self.azimuthImage.setAccessMode('write')

            if self.outputPrecision.upper() == 'SINGLE':
                self.azimuthImage.setDataType('FLOAT')
                self.azimuthImage.setCaster('write', 'DOUBLE')
            elif self.outputPrecision.upper() == 'DOUBLE':
                self.azimuthImage.setDataType('DOUBLE')
            else:
                raise Exception('Undefined output precision for azimuth image in geo2rdr.')

            self.azimuthImage.setWidth(self.demWidth)
            self.azimuthImage.createImage()

        if self.azimuthOffsetImageName:
            self.azimuthOffsetImage = createImage()
            self.azimuthOffsetImage.setFilename(self.azimuthOffsetImageName)
            self.azimuthOffsetImage.setAccessMode('write')

            if self.outputPrecision.upper() == 'SINGLE':
                self.azimuthOffsetImage.setDataType('FLOAT')
                self.azimuthOffsetImage.setCaster('write', 'DOUBLE')
            elif self.outputPrecision.upper() == 'DOUBLE':
                self.azimuthOffsetImage.setDataType('DOUBLE')
            else:
                raise Exception('Undefined output precision for azimuth offset image in geo2rdr.')

            self.azimuthOffsetImage.setWidth(self.demWidth)
            self.azimuthOffsetImage.createImage()


        self.polyDopplerAccessor = self.polyDoppler.exportToC()

    def setState(self):
        bistaticgeo2rdr.setEllipsoidMajorSemiAxis_Py(float(self.ellipsoidMajorSemiAxis))
        bistaticgeo2rdr.setEllipsoidEccentricitySquared_Py(float(self.ellipsoidEccentricitySquared))
        bistaticgeo2rdr.setRangePixelSpacing_Py(float(self.slantRangePixelSpacing))
        bistaticgeo2rdr.setActiveRangeFirstSample_Py(float(self.activeRangeFirstSample))
        bistaticgeo2rdr.setPassiveRangeFirstSample_Py(float(self.passiveRangeFirstSample))
        bistaticgeo2rdr.setDopplerAccessor_Py(self.polyDopplerAccessor)
        bistaticgeo2rdr.setPRF_Py(float(self.prf))
        bistaticgeo2rdr.setRadarWavelength_Py(float(self.radarWavelength))
        bistaticgeo2rdr.setSensingStart_Py(float(self.sensingStart))
        bistaticgeo2rdr.setLength_Py(int(self.length))
        bistaticgeo2rdr.setWidth_Py(int(self.width))
        bistaticgeo2rdr.setNumberRangeLooks_Py(int(self.numberRangeLooks))
        bistaticgeo2rdr.setNumberAzimuthLooks_Py(int(self.numberAzimuthLooks))
        bistaticgeo2rdr.setDemWidth_Py(int(self.demWidth))
        bistaticgeo2rdr.setDemLength_Py(int(self.demLength))
        bistaticgeo2rdr.setLookSide_Py(self.lookSide)
        bistaticgeo2rdr.setBistaticCorrectionFlag_Py(int(self.bistaticDelayCorrectionFlag))
        bistaticgeo2rdr.setOrbitMethod_Py( int( self.orbitMethods[self.orbitInterpolationMethod.upper()]))


    def setEllipsoidMajorSemiAxis(self,var):
        self.ellipsoidMajorSemiAxis = float(var)

    def setEllipsoidEccentricitySquared(self,var):
        self.ellipsoidEccentricitySquared = float(var)

    def setRangePixelSpacing(self,var):
        self.slantRangePixelSpacing = float(var)

    def setActiveRangeFirstSample(self,var):
        self.activeRangeFirstSample = float(var)

    def setPassiveRangeFirstSample(self,var):
        self.passiveRangeFirstSample = float(var)

    def setPRF(self,var):
        self.prf = float(var)

    def setRadarWavelength(self,var):
        self.radarWavelength = float(var)

    def setSensingStart(self,var):
        rtime = datetime.datetime.combine(var.date(), datetime.time(0,0,0))
        secs = (var - rtime).total_seconds()
        self.sensingStart = float(secs)

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

    def setPolyDoppler(self,var):
        self.polyDoppler = var

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

    def addDem(self):
        dem = self._inputPorts.getPort(name='dem').getObject()
        if (dem):
            try:
                self.demImage = dem
                self.demWidth = dem.getWidth()
                self.demLength = dem.getLength()
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError

    def addRadarImage(self):
        ifg = self._inputPorts.getPort(name='radarImage').getObject()
        if (ifg):
            try:
                self.inputImage = ifg
                self.width = ifg.getWidth()
                self.length = ifg.getLength()

            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError


    def __init__(self, name='') :
        super(BistaticGeo2rdr, self).__init__(self.__class__.family, name)

        # Dem information
        self.latImage = None
        self.lonImage = None
        self.demImage = None
        self.demWidth = None
        self.demLength = None

        ####Output images
        self.rangeImageName = None
        self.rangeImage = None

        self.azimuthImageName = None
        self.azimuthImage = None

        self.rangeOffsetImageName = None
        self.rangeOffsetImage = None

        self.azimuthOffsetImageName = None
        self.azimuthOffsetImage = None

        # Interferogram information
        self.length = None
        self.width = None

        #Doppler information
        self.polyDoppler = None
        self.polyDopplerAccessor = None
        self.dopplerCentroidCoeffs = None

        self.activeOrbit = None
        self.passiveOrbit = None

        self.bistaticDelayCorrectionFlag = None

        self.dictionaryOfOutputVariables = {}

        return None


    def createPorts(self):
        framePort = Port(name='frame',method=self.addFrame)
        planetPort = Port(name='planet', method=self.addPlanet)
        demPort = Port(name='dem',method=self.addDem)
        ifgPort = Port(name='radarImage',method=self.addRadarImage)

        self._inputPorts.add(framePort)
        self._inputPorts.add(planetPort)
        self._inputPorts.add(demPort)
        self._inputPorts.add(ifgPort)
        return None
