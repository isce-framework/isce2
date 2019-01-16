#!/usr/bin/env python3 

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





import sys
import os
import math
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from mroipac.formimage import formslc
from iscesys.Component.Component import Component, Port
from isceobj.Constants import SPEED_OF_LIGHT
import datetime

NUMBER_GOOD_BYTES = Component.Parameter(
    'numberGoodBytes',
    public_name='NUMBER_GOOD_BYTES',
    default=None,
    type=int,
    mandatory=True,
    doc='Number of bytes used in a range line in the raw image'
)
NUMBER_BYTES_PER_LINE = Component.Parameter(
    'numberBytesPerLine',
    public_name='NUMBER_BYTES_PER_LINE',
    default=None,
    type=int,
    mandatory=True,
    doc='Number of bytes per line in the raw image'
)
FIRST_LINE = Component.Parameter(
    'firstLine',
    public_name='FIRST_LINE',
    default=0,
    type=int,
    mandatory=False,
    doc='First line processed in the raw image'
)
NUMBER_VALID_PULSES = Component.Parameter(
    'numberValidPulses',
    public_name='NUMBER_VALID_PULSES',
    default=None,
    type=int,
    mandatory=True,
    doc='Number of lines to be stored from each azimuth patch'
)
FIRST_SAMPLE = Component.Parameter(
    'firstSample',
    public_name='FIRST_SAMPLE',
    default=None,
    type=int,
    mandatory=True,
    doc='First valid sample in the raw image range line.'
)
NUMBER_PATCHES = Component.Parameter(
    'numberPatches',
    public_name='NUMBER_PATCHES',
    default=None,
    type=int,
    mandatory=True,
    doc='Number of patches used.'
)
START_RANGE_BIN = Component.Parameter(
    'startRangeBin',
    public_name='START_RANGE_BIN',
    default=1,
    type=int,
    mandatory=False,
    doc=('Starting range bin to read from the raw data. '+
         'Must have positive value.'
    )
)
NUMBER_RANGE_BIN = Component.Parameter(
    'numberRangeBin',
    public_name='NUMBER_RANGE_BIN',
    default=None,
    type=int,
    mandatory=True,
    doc=('Number of range bins in the input raw image. '+
         'Used in the computation of the slcWidth. '
    )
)
NUMBER_AZIMUTH_LOOKS = Component.Parameter(
    'numberAzimuthLooks',
    public_name='NUMBER_AZIMUTH_LOOKS',
    default=None,
    type=int,
    mandatory=True,
    doc='Number of looks in the azimuth direction'
)
NEAR_RANGE_CHIRP_EXTENSION_FRAC = Component.Parameter(
        'nearRangeChirpExtFrac',
        default = 0.0,
        type=float,
        mandatory=False,
        doc='Chirp extension at near range')
FAR_RANGE_CHIRP_EXTENSION_FRAC = Component.Parameter(
        'farRangeChirpExtFrac',
        default = 0.0,
        type = float,
        mandatory = False,
        doc = 'Chirp extension at far range')
EARLY_AZIMUTH_CHIRP_EXTENSION_FRAC = Component.Parameter(
        'earlyAzimuthChirpExtFrac',
        default = 0.0,
        type = float,
        mandatory = False,
        doc = 'Azimuth chirp extension at the start of image')
LATE_AZIMUTH_CHIRP_EXTENSION_FRAC = Component.Parameter(
        'lateAzimuthChirpExtFrac',
        default = 0.0,
        type = float,
        mandatory = False,
        doc = 'Azimuth chirp extension at the end of image')
AZIMUTH_PATCH_SIZE = Component.Parameter(
    'azimuthPatchSize',
    public_name='AZIMUTH_PATCH_SIZE',
    default=None,
    type=int,
    mandatory=True,
    doc='Number of lines in an azimuth patch'
)
OVERLAP = Component.Parameter(
    'overlap',
    public_name='OVERLAP',
    default=0,
    type=int,
    mandatory=False,
    doc='Overlap between consecutive azimuth patches'
)
RAN_FFTOV = Component.Parameter(
    'ranfftov',
    public_name='RAN_FFTOV',
    default=65536,
    type=int,
    mandatory=False,
    doc='FFT size for offset video'
)
RAN_FFTIQ = Component.Parameter(
    'ranfftiq',
    public_name='RAN_FFTIQ',
    default=32768,
    type=int,
    mandatory=False,
    doc='FFT size for I/Q processing'
)
DEBUG_FLAG = Component.Parameter(
    'debugFlag',
    public_name='DEBUG_FLAG',
    default=False,
    type=bool,
    mandatory=False,
    doc='Debug output flag'
)
CALTONE_LOCATION = Component.Parameter(
    'caltoneLocation',
    public_name='CALTONE_LOCATION',
    default=0,
    type=int,
    mandatory=False,
    doc='Location of the calibration tone'
)
PLANET_LOCAL_RADIUS = Component.Parameter('planetLocalRadius',
        public_name = 'PLANET_LOCAL_RADIUS',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Local radius of the planet')
PLANET_GM = Component.Parameter('planetGM',
        public_name = 'PLANET_GM',
        default = 398600448073000.,
        type=float,
        mandatory=True,
        doc = 'Planet gravitational constant')
BODY_FIXED_VELOCITY = Component.Parameter(
    'bodyFixedVelocity',
    public_name='BODY_FIXED_VELOCITY',
    default=None,
    type=float,
    mandatory=True,
    doc='Platform velocity'
)
SPACECRAFT_HEIGHT = Component.Parameter(
    'spacecraftHeight',
    public_name='SPACECRAFT_HEIGHT',
    default=None,
    type=float,
    mandatory=True,
    doc='Spacecraft height'
)
PRF = Component.Parameter(
    'prf',
    public_name='PRF',
    default=None,
    type=float,
    mandatory=True,
    doc='Pulse repetition frequency'
)
INPHASE_VALUE = Component.Parameter(
    'inPhaseValue',
    public_name='INPHASE_VALUE',
    default=None,
    type=float,
    mandatory=True,
    doc=''
)
QUADRATURE_VALUE = Component.Parameter(
    'quadratureValue',
    public_name='QUADRATURE_VALUE',
    default=None,
    type=float,
    mandatory=True,
    doc=''
)
AZIMUTH_RESOLUTION = Component.Parameter(
    'azimuthResolution',
    public_name='AZIMUTH_RESOLUTION',
    default=None,
    type=float,
    mandatory=True,
    doc='Desired azimuth resolution for determining azimuth B/W'
)
RANGE_SAMPLING_RATE = Component.Parameter(
    'rangeSamplingRate',
    public_name='RANGE_SAMPLING_RATE',
    default=None,
    type=float,
    mandatory=True,
    doc='Sampling frequency of the range pixels'
)
CHIRP_SLOPE = Component.Parameter(
    'chirpSlope',
    public_name='CHIRP_SLOPE',
    default=None,
    type=float,
    mandatory=True,
    doc='Frequency slope of the transmitted chirp'
)
RANGE_PULSE_DURATION = Component.Parameter(
    'rangePulseDuration',
    public_name='RANGE_PULSE_DURATION',
    default=None,
    type=float,
    mandatory=True,
    doc='Range pulse duration'
)
RADAR_WAVELENGTH = Component.Parameter(
    'radarWavelength',
    public_name='RADAR_WAVELENGTH',
    default=None,
    type=float,
    mandatory=True,
    doc='Radar wavelength'
)
RANGE_FIRST_SAMPLE = Component.Parameter(
    'rangeFirstSample',
    public_name='RANGE_FIRST_SAMPLE',
    default=None,
    type=float,
    mandatory=True,
    doc='Range of the first sample in meters'
)
RANGE_SPECTRAL_WEIGHTING = Component.Parameter(
    'rangeSpectralWeighting',
    public_name='RANGE_SPECTRAL_WEIGHTING',
    default=1.0,
    type=float,
    mandatory=False,
    doc='Spectral weights for range spectrum.'
)
SPECTRAL_SHIFT_FRACTIONS = Component.Parameter(
    'spectralShiftFractions',
    public_name='SPECTRAL_SHIFT_FRACTION',
    default=[0., 0.],
    type=list,
    mandatory=False,
    doc='Spectral shift for range spectrum.'
)
IQ_FLIP = Component.Parameter(
    'IQFlip',
    public_name='IQ_FLIP',
    default=False,
    type=bool,
    mandatory=False,
    doc='If I/Q channels are flipped in the raw data file'
)
DESKEW_FLAG = Component.Parameter(
    'deskewFlag',
    public_name='DESKEW_FLAG',
    default=False,
    type=bool,
    mandatory=False,
    doc='If deskewing is desired'
)
SECONDARY_RANGE_MIGRATION_FLAG = Component.Parameter(
    'secondaryRangeMigrationFlag',
    public_name='SECONDARY_RANGE_MIGRATION_FLAG',
    default=False,
    type=bool,
    mandatory=False,
    doc='If secondary range migration is desired'
)
DOPPLER_CENTROID_COEFFICIENTS = Component.Parameter(
    'dopplerCentroidCoefficients',
    public_name='DOPPLER_CENTROID_COEFFICIENTS',
    default=[],
    type=list,
    mandatory=True,
    doc='Doppler centroid coefficients'
)
STARTING_RANGE = Component.Parameter(
    'startingRange',
    public_name='STARTING_RANGE',
    default=None,
    type=float,
    mandatory=False,
    private=True,
    intent='output',
    doc='Modified starting range for the SLC'
)
SLC_SENSING_START = Component.Parameter(
        'slcSensingStart',
        public_name='SLC_SENSING_START',
        default=None,
        mandatory=False,
        type=datetime.datetime,
        private=True,
        intent='output',
        doc='Modified sensing Start for the SLC'
)
SENSING_START = Component.Parameter(
        'sensingStart',
        public_name='SENSING_START',
        default=None,
        mandatory=True,
        type=datetime.datetime,
        doc='Sensing time of the first line of the RAW data')
ANTENNA_SCH_VELOCITY = Component.Parameter(
        'antennaSCHVelocity',
        public_name='ANTENNA_SCH_VELOCITY',
        default=[],
        type=list,
        mandatory=False,
        doc='Antenna SCH Velocity')

ANTENNA_SCH_ACCELERATION = Component.Parameter(
        'anntenaSCHAcceleration',
        public_name='ANTENNA_SCH_ACCELERATION',
        default = [],
        type = list,
        mandatory=False,
        doc='Antenna SCH Acceleration')
ANTENNA_LENGTH = Component.Parameter(
        'antennaLength',
        public_name= 'ANTENNA_LENGTH',
        default=None,
        type=float,
        mandatory=True,
        doc='Antenna length')
POINTING_DIRECTION = Component.Parameter(
        'pointingDirection',
        public_name='POINTING_DIRECTION',
        default=-1,
        type=int,
        mandatory=False,
        doc='Right: -1, Left: 1')

LINEAR_RESAMPLING_COEFFS = Component.Parameter(
        'linearResamplingCoefficients',
        public_name='LINEAR_RESAMPLING_COEFFS',
        default=[0.,0.,0.,0.],
        type=list,
        mandatory=False,
        doc='Linear resampling coefficients')
LINEAR_RESAMPLING_DELTAS = Component.Parameter(
        'linearResamplingDeltas',
        public_name='LINEAR_RESAMPLING_DELTAS',
        default = [0.,0.,0.,0.],
        type=list,
        mandatory=False,
        doc = 'Linear resampling spacings')

####Facilities
SLC_IMAGE = Component.Facility(
    'slcImage',
    public_name='slcImage',
    module='isceobj.Image',
    args=(),
    factory='createSlcImage',
    mandatory=True,
    doc='Single Look Complex Image object'  
)
RAW_IMAGE = Component.Facility(
    'rawImage',
    public_name='rawImage',
    module='isceobj.Image',
    args=(),
    factory='createRawIQImage',
    mandatory=True,
    doc='Raw Image object'  
)


class FormSLC(Component):

    family = 'formslc'
    logging_name = 'mroipac.formslc'

    parameter_list = (NUMBER_GOOD_BYTES,
                      NUMBER_BYTES_PER_LINE,
                      FIRST_LINE,
                      NUMBER_VALID_PULSES,
                      FIRST_SAMPLE,
                      NUMBER_PATCHES,
                      START_RANGE_BIN,
                      NUMBER_RANGE_BIN,
                      NUMBER_AZIMUTH_LOOKS,
                      NEAR_RANGE_CHIRP_EXTENSION_FRAC,
                      FAR_RANGE_CHIRP_EXTENSION_FRAC,
                      EARLY_AZIMUTH_CHIRP_EXTENSION_FRAC,
                      LATE_AZIMUTH_CHIRP_EXTENSION_FRAC,
                      AZIMUTH_PATCH_SIZE,
                      OVERLAP,
                      RAN_FFTOV,
                      RAN_FFTIQ,
                      DEBUG_FLAG,
                      CALTONE_LOCATION,
                      PLANET_LOCAL_RADIUS,
                      PLANET_GM,
                      BODY_FIXED_VELOCITY,
                      SPACECRAFT_HEIGHT,
                      PRF,
                      INPHASE_VALUE,
                      QUADRATURE_VALUE,
                      AZIMUTH_RESOLUTION,
                      RANGE_SAMPLING_RATE,
                      CHIRP_SLOPE,
                      RANGE_PULSE_DURATION,
                      RADAR_WAVELENGTH,
                      RANGE_FIRST_SAMPLE,
                      RANGE_SPECTRAL_WEIGHTING,
                      SPECTRAL_SHIFT_FRACTIONS,
                      IQ_FLIP,
                      DESKEW_FLAG,
                      SECONDARY_RANGE_MIGRATION_FLAG,
                      DOPPLER_CENTROID_COEFFICIENTS,
                      STARTING_RANGE,
                      ANTENNA_SCH_VELOCITY,
                      ANTENNA_SCH_ACCELERATION,
                      ANTENNA_LENGTH,
                      POINTING_DIRECTION,
                      LINEAR_RESAMPLING_COEFFS,
                      LINEAR_RESAMPLING_DELTAS,
                      SLC_SENSING_START,
                      SENSING_START,
                    )

    facility_list = (
                     SLC_IMAGE,
                     RAW_IMAGE,
                     )


    def formslc(self, rawImage=None):
        if rawImage is not None:
            self.rawImage = rawImage

        self.checkInitialization()
        self.createOutputImage()
        self.setState()
        slcImagePt = self.slcImage.getImagePointer()
        rawImagePt = self.rawImage.getImagePointer()
        formslc.formslc_Py(rawImagePt,slcImagePt)
        self.getState()
        self.rawImage.finalizeImage()
        self.slcImage.finalizeImage()
        self.slcImage.renderHdr()

    def getState(self):
        outStart = formslc.getSLCStartingRange_Py()

        if outStart != self.startingRange:
            raise Exception('Starting Range mismatch: {0} {1}'.format(outStart, self.startingRange))

        deltat = formslc.getSLCStartingLine_Py()
        self.slcSensingStart = self.sensingStart + datetime.timedelta(seconds = deltat / self.prf)
        return

    def setState(self):
        formslc.setNumberGoodBytes_Py(int(self.numberGoodBytes))
        formslc.setNumberBytesPerLine_Py(int(self.numberBytesPerLine))
        formslc.setDebugFlag_Py(int(self.debugFlag))
        formslc.setDeskewFlag_Py(int(self.deskewFlag))
        formslc.setSecondaryRangeMigrationFlag_Py(int(self.secondaryRangeMigrationFlag))
        formslc.setFirstLine_Py(int(self.firstLine))
        formslc.setNumberPatches_Py(int(self.numberPatches))
        formslc.setFirstSample_Py(int(self.firstSample))
        formslc.setAzimuthPatchSize_Py(int(self.azimuthPatchSize))
        formslc.setNumberValidPulses_Py(int(self.numberValidPulses))
        formslc.setCaltoneLocation_Py(float(self.caltoneLocation))
        formslc.setStartRangeBin_Py(int(self.startRangeBin))
        formslc.setNumberRangeBin_Py(int(self.numberRangeBin))
        formslc.setDopplerCentroidCoefficients_Py(self.dopplerCentroidCoefficients)
        formslc.setPlanetRadiusOfCurvature_Py(float(self.planetLocalRadius))
        formslc.setBodyFixedVelocity_Py(float(self.bodyFixedVelocity))
        formslc.setSpacecraftHeight_Py(float(self.spacecraftHeight))
        formslc.setPlanetGravitationalConstant_Py(float(self.planetGM))
        formslc.setPointingDirection_Py(int(self.pointingDirection))
        formslc.setAntennaSCHVelocity_Py(self.antennaSCHVelocity)
        formslc.setAntennaSCHAcceleration_Py(self.antennaSCHAcceleration)
        formslc.setRangeFirstSample_Py(float(self.rangeFirstSample))
        formslc.setPRF_Py(float(self.prf))
        formslc.setInPhaseValue_Py(float(self.inPhaseValue))
        formslc.setQuadratureValue_Py(float(self.quadratureValue))
        formslc.setIQFlip_Py(int(self.IQFlip))
        formslc.setAzimuthResolution_Py(float(self.azimuthResolution))
        formslc.setNumberAzimuthLooks_Py(int(self.numberAzimuthLooks))
        formslc.setRangeSamplingRate_Py(float(self.rangeSamplingRate))
        formslc.setChirpSlope_Py(float(self.chirpSlope))
        formslc.setRangePulseDuration_Py(float(self.rangePulseDuration))
        formslc.setRangeChirpExtensionPoints_Py(int(self.rangeChirpExtensionPoints))
        formslc.setRadarWavelength_Py(float(self.radarWavelength))
        formslc.setRangeSpectralWeighting_Py(float(self.rangeSpectralWeighting))
        formslc.setSpectralShiftFractions_Py(self.spectralShiftFractions)
        formslc.setLinearResamplingCoefficiets_Py(self.linearResamplingCoefficients)
        formslc.setLinearResamplingDeltas_Py(self.linearResamplingDeltas)

        return

    def createOutputImage(self):
        '''
        Compute SLC output width here.
        '''
        self.slcImage.setWidth(self.numberRangeBin)
        self.slcImage.setAccessMode('WRITE')
        self.slcImage.createImage()

        return

    def checkInitialization(self):
        '''
        Check that inputs are set correctly.
        '''
        pulseSamples = int( self.rangeSamplingRate * self.rangePulseDuration)

        nearRangeExt = int(pulseSamples * self.nearRangeChirpExtFrac)
        farRangeExt = int(pulseSamples * self.farRangeChirpExtFrac)


        spacing = SPEED_OF_LIGHT * 0.5 / self.rangeSamplingRate 
        slcWidth = int((self.numberGoodBytes - self.firstSample)/2) + nearRangeExt - farRangeExt- pulseSamples

        if (slcWidth <= 0):
            raise Exception('Range chirp extensions Error. Eventual SLC width is zero.')

        self.numberRangeBin = slcWidth


        slcStartingRange = self.rangeFirstSample - spacing * nearRangeExt 
        print('Estimated SLC Starting Range: ', slcStartingRange)


        ####Compute azimuth patch parameters
        chunksize=1024
        rawFileSize = self.rawImage.getLength() * self.rawImage.getWidth()
        linelength = int(self.rawImage.getXmax())
        width = self.rawImage.getWidth()

        synthApertureSamps = (
            self.radarWavelength* (slcStartingRange +
                slcWidth*SPEED_OF_LIGHT*0.5/self.rangeSamplingRate)*
                self.prf/(self.antennaLength*self.bodyFixedVelocity))
        nSAS = int((synthApertureSamps-1)/chunksize)+1
        chunkedSAS = chunksize*nSAS
        nxP = self.nxPower(nSAS)
        azP = chunksize*2*(2**nxP)      #Patchsize
        nV = azP-chunkedSAS             #Numbervalid
        if self.azimuthPatchSize:
            if self.azimuthPatchSize != 2**self.nxPower(self.azimuthPatchSize):
                self.azimuthPatchSize = 2**self.nxPower(self.azimuthPatchSize)
                self.logger.info(
                    "Patch size must equal power of 2. Resetting to %d" %
                    self.azimuthPatchSize
                    )

        if self.azimuthPatchSize and self.numberValidPulses:
            if (self.azimuthPatchSize < self.numberValidPulses or
                self.azimuthPatchSize < chunkedSAS+chunksize):
                self.azimuthPatchSize = azP
                self.numberValidPulses = nV
            elif self.numberValidPulses > self.azimuthPatchSize-chunkedSAS:
                msg = ("Number of valid pulses specified is too large "+
                       "for full linear convolution. ")
                msg += ("Should be less than %d" %
                        (self.azimuthPatchSize-chunkedSAS))
                self.logger.info(msg)
                self.logger.info(
                    "Continuing with specified value of %d" %
                    self.numberValidPulses
                    )

        elif self.azimuthPatchSize and not self.numberValidPulses:
            if self.azimuthPatchSize < chunkedSAS+chunksize:
                self.azimuthPatchSize = azP
                self.numberValidPulses = nV
            else:
                self.numberValidPulses = self.azimuthPatchSize-chunkedSAS
                if self.numberValidPulses > self.azimuthPatchSize-chunkedSAS:
                    msg = ("Number of valid pulses specified is too large "+
                           "for full linear convolution. ")
                    msg += ("Should be less than %d" %
                            (self.azimuthPatchSize-chunkedSAS))
                    self.logger.info(msg)
                    self.logger.info(
                        "Continuing with specified value of %d" %
                        self.numberValidPulses
                        )

        elif not self.azimuthPatchSize and self.numberValidPulses:
            self.azimuthPatchSize=2**self.nxPower(self.numberValidPulses+
                                                  synthApertureSamps)
            if self.azimuthPatchSize > self.maxAzPatchSize:
                msg = ("%d is a rather large patch size. " %
                         self.azimuthPatchSize)
                msg += ("Check that the number of valid pulses is in a "+
                        "reasonable range. Proceeding anyway...")
                self.logger.info(msg)

        elif not self.azimuthPatchSize and not self.numberValidPulses:
            self.azimuthPatchSize=azP
            self.numberValidPulses=nV


        ####Set azimuth extensions
        earlyExt = int(self.earlyAzimuthChirpExtFrac * synthApertureSamps)
        lateExt = int(self.lateAzimuthChirpExtFrac * synthApertureSamps)

        procStart = -earlyExt
        procEnd = self.rawImage.getLength() + lateExt

        overhead = self.azimuthPatchSize - self.numberValidPulses
        if not self.numberPatches:
            self.numberPatches = (1 + int(
                                    (procEnd - procStart - overhead)/self.numberValidPulses))

        self.firstLine = procStart

        if nearRangeExt < 0:
            self.startRangeBin = 1-nearRangeExt
            self.rangeChirpExtensionPoints = 0
        else:
            self.startRangeBin = 1
            self.rangeChirpExtensionPoints = nearRangeExt

        self.startingRange = slcStartingRange

        pass

    def setNumberGoodBytes(self,var):
        self.numberGoodBytes = int(var)
        return

    def setNumberBytesPerLine(self,var):
        self.numberBytesPerLine = int(var)
        return

    def setDebugFlag(self,var):
        self.debugFlag = str(var)
        return

    def setDeskewFlag(self,var):
        self.deskewFlag = str(var)
        return

    def setSecondaryRangeMigrationFlag(self,var):
        self.secondaryRangeMigrationFlag = str(var)
        return

    def setFirstLine(self,var):
        self.firstLine = int(var)
        return

    def setNumberPatches(self,var):
        self.numberPatches = int(var)
        return

    def setFirstSample(self,var):
        self.firstSample = int(var)
        return

    def setAzimuthPatchSize(self,var):
        self.azimuthPatchSize = int(var)
        return

    def setNumberValidPulses(self,var):
        self.numberValidPulses = int(var)
        return

    def setCaltoneLocation(self,var):
        self.caltoneLocation = float(var)
        return

    def setStartRangeBin(self,var):
        self.startRangeBin = int(var)
        return

    def setNumberRangeBin(self,var):
        self.numberRangeBin = int(var)
        return

    def setDopplerCentroidCoefficients(self,var):
        self.dopplerCentroidCoefficients = var
        return

    def setPlanetRadiusOfCurvature(self,var):
        self.planetRadiusOfCurvature = float(var)
        return

    def setBodyFixedVelocity(self,var):
        self.bodyFixedVelocity = float(var)
        return

    def setSpacecraftHeight(self,var):
        self.spacecraftHeight = float(var)
        return

    def setPlanetGravitationalConstant(self,var):
        self.planetGravitationalConstant = float(var)
        return

    def setPointingDirection(self,var):
        self.pointingDirection = int(var)
        return

    def setAntennaSCHVelocity(self,var):
        self.antennaSCHVelocity = var
        return

    def setAntennaSCHAcceleration(self,var):
        self.antennaSCHAcceleration = var
        return

    def setRangeFirstSample(self,var):
        self.rangeFirstSample = float(var)
        return

    def setPRF(self,var):
        self.PRF = float(var)
        return

    def setInPhaseValue(self,var):
        self.inPhaseValue = float(var)
        return

    def setQuadratureValue(self,var):
        self.quadratureValue = float(var)
        return

    def setIQFlip(self,var):
        self.IQFlip = str(var)
        return

    def setAzimuthResolution(self,var):
        self.azimuthResolution = float(var)
        return

    def setNumberAzimuthLooks(self,var):
        self.numberAzimuthLooks = int(var)
        return

    def setRangeSamplingRate(self,var):
        self.rangeSamplingRate = float(var)
        return

    def setChirpSlope(self,var):
        self.chirpSlope = float(var)
        return

    def setRangePulseDuration(self,var):
        self.rangePulseDuration = float(var)
        return

    def setRadarWavelength(self,var):
        self.radarWavelength = float(var)
        return

    def setRangeSpectralWeighting(self,var):
        self.rangeSpectralWeighting = float(var)
        return

    def setSpectralShiftFractions(self,var):
        self.spectralShiftFractions = var
        return

    def setLinearResamplingCoefficients(self,var):
        self.linearResamplingCoefficients = var
        return

    def setLinearResamplingDeltas(self,var):
        self.linearResamplingDeltas = var
        return

    @staticmethod
    def nxPower(num):
        power=0
        k=0
        while power < num:
            k+=1
            power=2**k
        return k



    def __init__(self,name=''):
        
        super(FormSLC, self).__init__(family=self.__class__.family, name=name) 
       
        self.rangeChirpExtensionPoints = None
        self.descriptionOfVariables = {}
        self.dictionaryOfVariables = {}
        self.mandatoryVariables = []
        self.optionalVariables = []

        return





#end class




if __name__ == "__main__":
    sys.exit(main())
