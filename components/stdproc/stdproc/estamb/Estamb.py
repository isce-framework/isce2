#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Piyush Agram
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import isce
from isceobj.Image.Image import Image
from iscesys.Component.Component import Component, Port
from stdproc.stdproc.estamb import estamb


NUMBER_GOOD_BYTES = Component.Parameter('numberGoodBytes',
                                        public_name='NUMBER_GOOD_BYTES',
                                        default=None,
                                        type=int,
                                        mandatory=True,
                                        doc='Number of bytes used in a range line in the raw image'
                                       )
NUMBER_BYTES_PER_LINE = Component.Parameter('numberBytesPerLine',
                                            public_name='NUMBER_BYTES_PER_LINE',
                                            default=None,
                                            type=int,
                                            mandatory=True,
                                            doc='Number of bytes per line in the raw image'
                                           )
FIRST_LINE = Component.Parameter('firstLine',
                                 public_name='FIRST_LINE',
                                 default=None,
                                 type=int,
                                 mandatory=True,
                                 doc='First line processed in the raw image'
                                )
NUMBER_VALID_PULSES = Component.Parameter('numberValidPulses',
                                          public_name='NUMBER_VALID_PULSES',
                                          default=None,
                                          type=int,
                                          mandatory=True,
                                          doc='Number of lines to be stored from each azimuth patch'
                                         )
FIRST_SAMPLE = Component.Parameter('firstSample',
                                   public_name='FIRST_SAMPLE',
                                   default=None,
                                   type=int,
                                   mandatory=True,
                                   doc='First valid sample in the range line'
                                  )

NUMBER_PATCHES = Component.Parameter('numberPatches',
                                     public_name='NUMBER_PATCHES',
                                     default=1,
                                     type=int,
                                     mandatory=False,
                                     doc='Number of patches used.'
                                    )
START_RANGE_BIN = Component.Parameter('startRangeBin',
                                      public_name='START_RANGE_BIN',
                                      default=1,
                                      type=int,
                                      mandatory=False,
                                      doc='Starting bin in the range direction. If negative, indicates near range extension.'
                                     )
NUMBER_RANGE_BIN = Component.Parameter('numberRangeBin',
                                       public_name='NUMBER_RANGE_BIN',
                                       default=None,
                                       type=int,
                                       mandatory=True,
                                       doc='Number of range bins to output. If greater than that of raw image, indicates near/far range extension.'
                                      )
AZIMUTH_PATCH_SIZE = Component.Parameter('azimuthPatchSize',
                                         public_name='AZIMUTH_PATCH_SIZE',
                                         default=None,
                                         type=int,
                                         mandatory=True,
                                         doc='Number of lines in an azimuth patch'
                                        )
OVERLAP = Component.Parameter('overlap',
                              public_name='OVERLAP',
                              default=0,
                              type=int,
                              mandatory=False,
                              doc='Overlap between consecutive azimuth patches'
                             )
RAN_FFTOV = Component.Parameter('ranfftov',
                                public_name='RAN_FFTOV',
                                default=65536,
                                type=int,
                                mandatory=False,
                                doc='FFT size for offset video'
                               )
RAN_FFTIQ = Component.Parameter('ranfftiq',
                                public_name='RAN_FFTIQ',
                                default=32768,
                                type=int,
                                mandatory=False,
                                doc='FFT size for I/Q processing'
                               )
CALTONE_LOCATION = Component.Parameter('caltoneLocation',
                                       public_name='CALTONE_LOCATION',
                                       default=0,
                                       type=int,
                                       mandatory=False,
                                       doc='Location of the calibration tone'
                                      )
PLANET_LOCAL_RADIUS = Component.Parameter('planetLocalRadius',
                                          public_name='PLANET_LOCAL_RADIUS',
                                          default=None,
                                          type=float,
                                          mandatory=True,
                                          doc='Local radius of the planet'
                                         )
BODY_FIXED_VELOCITY = Component.Parameter('bodyFixedVelocity',
                                          public_name='BODY_FIXED_VELOCITY',
                                          default=None,
                                          type=float,
                                          mandatory=True,
                                          doc='Platform velocity'
                                         )
SPACECRAFT_HEIGHT = Component.Parameter('spacecraftHeight',
                                        public_name='SPACECRAFT_HEIGHT',
                                        default=None,
                                        type=float,
                                        mandatory=True,doc='Spacecraft height'
                                       )
PRF = Component.Parameter('prf',
                          public_name='PRF',
                          default=None,
                          type=float,
                          mandatory=True,
                          doc='Pulse repetition frequency'
                         )
INPHASE_VALUE = Component.Parameter('inPhaseValue',
                                    public_name='INPHASE_VALUE',
                                    default=None,
                                    type=float,
                                    mandatory=True,
                                    doc=''
                                   )
QUADRATURE_VALUE = Component.Parameter('quadratureValue',
                                       public_name='QUADRATURE_VALUE',
                                       default=None,
                                       type=float,
                                       mandatory=True,
                                       doc=''
                                      )
AZIMUTH_RESOLUTION = Component.Parameter('azimuthResolution',
                                         public_name='AZIMUTH_RESOLUTION',
                                         default=None,
                                         type=float,
                                         mandatory=True,
                                         doc='Desired azimuth resolution for determining azimuth B/W'
                                        )
RANGE_SAMPLING_RATE = Component.Parameter('rangeSamplingRate',
                                          public_name='RANGE_SAMPLING_RATE',
                                          default=None,
                                          type=float,
                                          mandatory=True,
                                          doc='Sampling frequency of the range pixels'
                                         )
CHIRP_SLOPE = Component.Parameter('chirpSlope',
                                  public_name='CHIRP_SLOPE',
                                  default=None,
                                  type=float,
                                  mandatory=True,
                                  doc='Frequency slope of the transmitted chirp'
                                 )
RANGE_PULSE_DURATION = Component.Parameter('rangePulseDuration',
                                           public_name='RANGE_PULSE_DURATION',
                                           default=None,
                                           type=float,
                                           mandatory=True,
                                           doc='Range pulse duration'
                                          )
RADAR_WAVELENGTH = Component.Parameter('radarWavelength',
                                       public_name='RADAR_WAVELENGTH',
                                       default=None,
                                       type=float,
                                       mandatory=True,
                                       doc='Radar wavelength'
                                      )
RANGE_FIRST_SAMPLE = Component.Parameter('rangeFirstSample',
                                         public_name='RANGE_FIRST_SAMPLE',
                                         default=None,
                                         type=float,
                                         mandatory=True,
                                         doc='Range of the first sample in meters'
                                        )
IQ_FLIP = Component.Parameter('IQFlip',
                              public_name='IQ_FLIP',
                              default='n',
                              type=str,
                              mandatory=False,
                              doc='If I/Q channels are flipped in the raw data file'
                             )
POSITION = Component.Parameter('position',
    public_name='POSITION',
    default=[],
    container=list,
    type=float,
    mandatory=True,
    doc='Position vector'
)
TIME = Component.Parameter('time',
    public_name='TIME',
    default=[],
    container=list,
    type=float,
    mandatory=True,
    doc='Time vector'
)
DOPPLER_CENTROID_COEFFICIENTS = Component.Parameter(
    'dopplerCentroidCoefficients',
    public_name='DOPPLER_CENTROID_COEFFICIENTS',
    default=[],
    container=list,
    type=float,
    mandatory=True,
    doc='Doppler centroid coefficients'
)
ENTROPY = Component.Parameter('entropy',
    public_name='ENTROPY',
    default=[],
    container=list,
    type=float,
    mandatory=False,
    private=True,
    doc='ENTROPY'
)
MINIMUM_AMBIGUITY = Component.Parameter('minAmb',
                                    public_name='MINIMUM_AMBIGUITY',
                                    default = -3,
                                    mandatory = None,
                                    doc = 'Minimum doppler ambiguity for search window.'
                                    )
MAXIMUM_AMBIGUITY = Component.Parameter('maxAmb',
                                    public_name='MAXIMUM_AMBIGUITY',
                                    default = 3,
                                    mandatory = None,
                                    doc = 'Maximum doppler ambiguity for search window.'
                                    )
DOPPLER_AMBIGUITY = Component.Parameter('dopplerAmbiguity',
                                    public_name='DOPPLER_AMBIGUITY',
                                    default=None,
                                    mandatory=False,
                                    private=True,
                                    doc='Doppler ambiguity estimated by estamb'
                                    )

## This decorator takes a setter and only executes it if the argument is True
def set_if_true(func):
    """Decorate a setter to only set if the value is nonzero"""
    def new_func(self, var):
        if var:
            func(self, var)
    return new_func

#@pickled
class Estamb(Component):

    dont_pickle_me = ()

    parameter_list = (NUMBER_GOOD_BYTES,
                      NUMBER_BYTES_PER_LINE,
                      FIRST_LINE,
                      NUMBER_VALID_PULSES,
                      FIRST_SAMPLE,
                      NUMBER_PATCHES,
                      START_RANGE_BIN,
                      NUMBER_RANGE_BIN,
                      AZIMUTH_PATCH_SIZE,
                      OVERLAP,
                      RAN_FFTOV,
                      RAN_FFTIQ,
                      CALTONE_LOCATION,
                      PLANET_LOCAL_RADIUS,
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
                      IQ_FLIP,
                      POSITION,
                      TIME,
                      DOPPLER_CENTROID_COEFFICIENTS,
                      ENTROPY,
                      DOPPLER_AMBIGUITY,
                      MINIMUM_AMBIGUITY,
                      MAXIMUM_AMBIGUITY
                    )
    _vars = (
        Component.Variable('numberGoodBytes', int, True),
        Component.Variable('numberBytesPerLine', int, True),
        Component.Variable('firstLine', int, False),
        Component.Variable('numberValidPulses', int, True),
        Component.Variable('firstSample', int, True),
        Component.Variable('numberPatches', int, True),
        Component.Variable('startRangeBin', int, False),
        Component.Variable('numberRangeBin', int, True),
        Component.Variable('azimuthPatchSize', int, False),
        Component.Variable('overlap', int, False),
        Component.Variable('ranfftov', int, False),
        Component.Variable('ranfftiq', int, False),
        Component.Variable('caltoneLocation', float, True),
        Component.Variable('planetLocalRadius', float, True),
        Component.Variable('bodyFixedVelocity', float, True),
        Component.Variable('spacecraftHeight', float, True),
        Component.Variable('prf', float, True),
        Component.Variable('inPhaseValue', float, True),
        Component.Variable('quadratureValue', float, True),
        Component.Variable('azimuthResolution', float, True),
        Component.Variable('rangeSamplingRate', float, True),
        Component.Variable('chirpSlope', float, True),
        Component.Variable('rangePulseDuration', float, True),
        Component.Variable('radarWavelength', float, True),
        Component.Variable('rangeFirstSample', float, True),
        Component.Variable('IQFlip', str, True),
        Component.Variable('position', '', True),
        Component.Variable('time', float, True),
        Component.Variable(
            'dopplerCentroidCoefficients',
            float,
            True
            ),
        Component.Variable('minAmb', int, False),
        Component.Variable('maxAmb', int, False),
        )

    maxAzPatchSize = 32768

    def estamb(self):
        for item in self.inputPorts:
            item()

        self.computeRangeParams()

        try:
            self.rawAccessor = self.rawImage.getImagePointer()
        except AttributeError:
            self.logger.error("Error in accessing image pointers")
            raise AttributeError

        self.computePatchParams()
        self.allocateArrays()
        self.setDefaults()
        self.setState()
        estamb.estamb_Py(self.rawAccessor)

        self.getState()
        self.deallocateArrays()
        return self.entropy, self.dopplerAmbiguity

    @staticmethod
    def nxPower(num):
        power=0
        k=0
        while power < num:
            k+=1
            power=2**k
        return k

    def computeRangeParams(self):
        '''Ensure that the given range parameters are valid.'''
        from isceobj.Constants import SPEED_OF_LIGHT
        import isceobj

        self.rangeChirpExtensionPoints = 0

        if self.startRangeBin <= 0:
            raise ValueError('startRangeBin should be greater than or equal to 1')

        self.logger.info('Number of Range Bins: %d'%self.numberRangeBin)
        self.slcWidth = self.numberRangeBin + self.rangeChirpExtensionPoints + (self.startRangeBin - 1)
        delr = self.rangeSamplingRate

        #Will be set here and passed on to Fortran. - Piyush
        self.startingRange = self.rangeFirstSample + (self.startRangeBin - 1 - self.rangeChirpExtensionPoints) * SPEED_OF_LIGHT*0.5/self.rangeSamplingRate

    def computePatchParams(self):

        from isceobj.Constants import SPEED_OF_LIGHT
        chunksize=1024
        rawFileSize = self.rawImage.getLength() * self.rawImage.getWidth()
        linelength = int(self.rawImage.getXmax())

        synthApertureSamps = (
            self.radarWavelength* (self.startingRange + self.slcWidth*SPEED_OF_LIGHT*0.5/self.rangeSamplingRate)
                *self.prf/(self.antennaLength*self.bodyFixedVelocity))
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
                self.logger.info(
                    "Number of valid pulses specified is too large for full linear convolution. Should be less than %d" % self.azimuthPatchSize-chunkedSAS)
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
                    self.logger.info(
                        "Number of valid pulses specified is too large for full linear convolution. Should be less than %d" %
                        self.azimuthPatchSize-chunkedSAS
                        )
                    self.logger.info(
                        "Continuing with specified value of %d" %
                        self.numberValidPulses
                        )

        elif not self.azimuthPatchSize and self.numberValidPulses:
            self.azimuthPatchSize=2**self.nxPower(self.numberValidPulses+
                                                  synthApertureSamps)
            if self.azimuthPatchSize > self.maxAzPatchSize:
                self.logger.info(
                    "%d is a rather large patch size. Check that the number of valid pulses is in a reasonable range.  Proceeding anyway..." %
                    self.azimuthPatchSize
                    )

        elif not self.azimuthPatchSize and not self.numberValidPulses:
            self.azimuthPatchSize=azP
            self.numberValidPulses=nV

        overhead = self.azimuthPatchSize - self.numberValidPulses
        if not self.numberPatches:
            self.numberPatches = (
                1+int(
                    (rawFileSize/float(linelength)-overhead)/
                    self.numberValidPulses
                    )
                )


    def getState(self):
        self.entropy = estamb.getEntropy_Py(
            (self.maxAmb - self.minAmb+ 1)
            )
        self.dopplerAmbiguity = self.entropy.index(max(self.entropy)) + self.minAmb

    def setDefaults(self):
        if self.firstLine is None:
            self.firstLine = self.numberPatches * self.numberValidPulses

    def setState(self):
        estamb.setStdWriter_Py(int(self.stdWriter))
        estamb.setNumberGoodBytes_Py(int(self.numberGoodBytes))
        estamb.setNumberBytesPerLine_Py(int(self.numberBytesPerLine))
        estamb.setFirstLine_Py(int(self.firstLine))
        estamb.setNumberValidPulses_Py(int(self.numberValidPulses))
        estamb.setFirstSample_Py(int(self.firstSample))
        estamb.setNumberPatches_Py(int(self.numberPatches))
        estamb.setStartRangeBin_Py(int(self.startRangeBin))
        estamb.setNumberRangeBin_Py(int(self.numberRangeBin))
        estamb.setRangeChirpExtensionPoints_Py(
            int(self.rangeChirpExtensionPoints)
            )
        estamb.setAzimuthPatchSize_Py(int(self.azimuthPatchSize))
        estamb.setOverlap_Py(int(self.overlap))
        estamb.setRanfftov_Py(int(self.ranfftov))
        estamb.setRanfftiq_Py(int(self.ranfftiq))
        estamb.setDebugFlag_Py(int(self.debugFlag))
        estamb.setCaltoneLocation_Py(float(self.caltoneLocation))
        estamb.setPlanetLocalRadius_Py(float(self.planetLocalRadius))
        estamb.setBodyFixedVelocity_Py(float(self.bodyFixedVelocity))
        estamb.setSpacecraftHeight_Py(float(self.spacecraftHeight))
        estamb.setPRF_Py(float(self.prf))
        estamb.setInPhaseValue_Py(float(self.inPhaseValue))
        estamb.setQuadratureValue_Py(float(self.quadratureValue))
        estamb.setAzimuthResolution_Py(float(self.azimuthResolution))
        estamb.setRangeSamplingRate_Py(float(self.rangeSamplingRate))
        estamb.setChirpSlope_Py(float(self.chirpSlope))
        estamb.setRangePulseDuration_Py(float(self.rangePulseDuration))
        estamb.setRadarWavelength_Py(float(self.radarWavelength))
        estamb.setRangeFirstSample_Py(float(self.rangeFirstSample))
        estamb.setRangeSpectralWeighting_Py(float(self.rangeSpectralWeighting))
        estamb.setSpectralShiftFraction_Py(float(self.spectralShiftFraction))
        estamb.setIMRC1_Py(int(self.imrc1Accessor))
        estamb.setIMMocomp_Py(int(self.immocompAccessor))
        estamb.setIMRCAS1_Py(int(self.imrcas1Accessor))
        estamb.setIMRCRM1_Py(int(self.imrcrm1Accessor))
        estamb.setTransDat_Py(int(self.transAccessor))
        estamb.setIQFlip_Py(self.IQFlip)
        estamb.setDeskewFlag_Py(self.deskewFlag)
        estamb.setSecondaryRangeMigrationFlag_Py(
            self.secondaryRangeMigrationFlag
            )
        estamb.setPosition_Py(self.position,
                               self.dim1_position,
                               self.dim2_position)
        estamb.setVelocity_Py(self.velocity,
                               self.dim1_velocity,
                               self.dim2_velocity)
        estamb.setTime_Py(self.time,
                           self.dim1_time)
        estamb.setDopplerCentroidCoefficients_Py(
            self.dopplerCentroidCoefficients,
            self.dim1_dopplerCentroidCoefficients
            )
        estamb.setPegPoint_Py(self.pegLatitude,
                               self.pegLongitude,
                               self.pegHeading)
        estamb.setPlanet_Py(self.spin, self.gm)
        estamb.setEllipsoid_Py(self.a, self.e2)
        estamb.setSlcWidth_Py(self.slcWidth)
        estamb.setStartingRange_Py(self.startingRange)
        estamb.setLookSide_Py(self.lookSide)
        estamb.setShift_Py(self.shift) ##KK,ML 2013-07-15
        estamb.setMinAmb_Py(int(self.minAmb))
        estamb.setMaxAmb_Py(int(self.maxAmb))

    def setRawImage(self, raw):
        self.rawImage = raw

    def setNumberGoodBytes(self, var):
        self.numberGoodBytes = int(var)

    def setNumberBytesPerLine(self, var):
        self.numberBytesPerLine = int(var)

    def setFirstLine(self, var):
        self.firstLine = int(var)

    def setLookSide(self, var):
        self.lookSide = int(var)

    @set_if_true
    def setNumberValidPulses(self, var):
        self.numberValidPulses = int(var)

    def setFirstSample(self, var):
        self.firstSample = int(var)

    @set_if_true
    def setNumberPatches(self,var):
        self.numberPatches = int(var)

    def setStartRangeBin(self, var):
        self.startRangeBin = int(var)

    def setStartingRange(self, var):
        self.startingRange = float(var)

    def setNumberRangeBin(self, var):
        self.numberRangeBin = int(var)

    @set_if_true
    def setAzimuthPatchSize(self, var):
        self.azimuthPatchSize = int(var)

    def setOverlap(self, var):
        self.overlap = int(var)

    def setRanfftov(self, var):
        self.ranfftov = int(var)

    def setRanfftiq(self, var):
        self.ranfftiq = int(var)

    def setCaltoneLocation(self, var):
        self.caltoneLocation = float(var)

    def setPlanetLocalRadius(self, var):
        self.planetLocalRadius = float(var)

    def setBodyFixedVelocity(self, var):
        self.bodyFixedVelocity = float(var)

    def setSpacecraftHeight(self, var):
        self.spacecraftHeight = float(var)

    def setPRF(self, var):
        self.prf = float(var)

    def setInPhaseValue(self, var):
        self.inPhaseValue = float(var)

    def setQuadratureValue(self, var):
        self.quadratureValue = float(var)

    def setAzimuthResolution(self, var):
        self.azimuthResolution = float(var)

    def setRangeSamplingRate(self, var):
        self.rangeSamplingRate = float(var)

    def setChirpSlope(self, var):
        self.chirpSlope = float(var)

    def setRangePulseDuration(self, var):
        self.rangePulseDuration = float(var)

    def setRadarWavelength(self, var):
        self.radarWavelength = float(var)

    def setRangeFirstSample(self, var):
        self.rangeFirstSample = float(var)

    def setIQFlip(self, var):
        self.IQFlip = str(var)

    def setPosition(self, var):
        self.position = var

    def setVelocity(self, var):
        self.velocity = var

    def setTime(self, var):
        self.time = var

    def setSlcWidth(self, var):
        self.slcWidth = var

    def setDopplerCentroidCoefficients(self, var):
        self.dopplerCentroidCoefficients = var


    def _testArraySize(self,*args):
        """Test for array dimesions that are zero or smaller"""
        for dimension in args:
            if (dimension <= 0):
                self.logger.error("Error, trying to allocate zero size array")
                raise ValueError

    def allocateArrays(self):
        # Set array sizes from their arrays
        try:
            self.dim1_position = len(self.position)
            self.dim2_position = len(self.position[0])
            self.dim1_velocity = len(self.velocity)
            self.dim2_velocity = len(self.velocity[0])
            self.dim1_time = len(self.time)
            self.dim1_dopplerCentroidCoefficients = len(self.dopplerCentroidCoefficients)
        except TypeError:
            self.logger.error("Some input arrays were not set")
            raise TypeError

        # Test that the arrays have a size greater than zero
        self._testArraySize(self.dim1_position,self.dim2_position)
        self._testArraySize(self.dim1_velocity,self.dim2_velocity)
        self._testArraySize(self.dim1_time)
        self._testArraySize(self.dim1_dopplerCentroidCoefficients)

        # Allocate the arrays
        estamb.allocate_sch_Py(self.dim1_position, self.dim2_position)
        estamb.allocate_vsch_Py(self.dim1_velocity, self.dim2_velocity)
        estamb.allocate_time_Py(self.dim1_time)
        estamb.allocate_dopplerCoefficients_Py(self.dim1_dopplerCentroidCoefficients)
        estamb.allocate_entropy_Py(int(self.maxAmb - self.minAmb + 1))

    def deallocateArrays(self):
        estamb.deallocate_sch_Py()
        estamb.deallocate_vsch_Py()
        estamb.deallocate_time_Py()
        estamb.deallocate_dopplerCoefficients_Py()
        estamb.deallocate_entropy_Py()
        pass

    def addRawImage(self):
        image = self.inputPorts['rawImage']
        if image:
            if isinstance(image, Image):
                self.rawImage = image
                self.numberBytesPerLine = self.rawImage.getWidth()
                self.numberGoodBytes = self.rawImage.getNumberGoodBytes()
                self.firstSample = int(self.rawImage.getXmin()/2)
            else:
                self.logger.error(
                    "Object %s must be an instance of Image" % image
                    )
                raise TypeError

    def addOrbit(self):
        orbit = self.inputPorts['orbit']
        if orbit:
            try:
                time,position,velocity,offset = orbit._unpackOrbit()
                self.time = time
                self.position = position
                self.velocity = velocity
            except AttributeError:
                self.logger.error(
                    "Object %s requires an _unpackOrbit() method" %
                    orbit.__class__
                    )
                raise AttributeError

    def addFrame(self):
        frame = self.inputPorts['frame']
        if frame:
            try:
                self.rangeFirstSample = frame.getStartingRange()
                self.rangeLastSample = frame.getFarRange()
                instrument = frame.getInstrument()
                self.inPhaseValue = instrument.getInPhaseValue()
                self.quadratureValue = instrument.getQuadratureValue()
                self.rangeSamplingRate = instrument.getRangeSamplingRate()
                self.chirpSlope = instrument.getChirpSlope()
                self.rangePulseDuration = instrument.getPulseLength()
                self.radarWavelength = instrument.getRadarWavelength()
                self.prf = instrument.getPulseRepetitionFrequency()
                self.antennaLength = instrument.getPlatform().getAntennaLength()
                self.azimuthResolution = self.antennaLength/2.0
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError

    def addPlanet(self):
        planet = self.inputPorts['planet']
        if planet:
            try:
                self.spin = planet.spin
                self.gm = planet.GM
                ellipsoid = planet.ellipsoid
                self.a = ellipsoid.a
                self.e2 = ellipsoid.e2
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError

    def addPeg(self):
        peg = self.inputPorts['peg']
        if peg:
            try:
                self.pegLatitude = peg.getLatitude()
                self.pegLongitude = peg.getLongitude()
                self.pegHeading = peg.getHeading()
                self.planetLocalRadius = peg.getRadiusOfCurvature()
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError

    def addDoppler(self):
        doppler = self.inputPorts['doppler']
        if doppler:
            try:
                self.dopplerCentroidCoefficients = (
                    doppler.getDopplerCoefficients(inHz=False)
                    )
                self.dim1_dopplerCentroidCoefficients = len(
                    self.dopplerCentroidCoefficients
                    )
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError

    def _parameters(self):
        """Define the user configurable parameters for this application"""
        for item in self.__class__.parameter_list:
            try:
                setattr(self,
                        item.attrname,
                        self.parameter(item.attrname,
                                       public_name=item.public_name,
                                       default=item.default,
                                       units=None,
                                       doc=item.doc,
                                       type=item.type,
                                       mandatory=item.mandatory
                                       )
                        )
            except AttributeError:
                message = (
                    "Failed to set parameter %s type %s in %s" %
                    (str(item), item.__class__.__name__, repr(self))
                    )
                raise AttributeError(message)
            pass
        return None


    def _facilities(self):
        self.rawImage = self.facility('rawImage',public_name='rawImage',module='isceobj.Image',factory='createRawImage',
                                    mandatory=True,doc='Raw Image object')

    def createPorts(self):
        self.inputPorts['rawImage'] = self.addRawImage
        self.inputPorts['orbit'] = self.addOrbit
        self.inputPorts['frame'] = self.addFrame
        self.inputPorts['peg'] = self.addPeg
        self.inputPorts['planet'] = self.addPlanet
        self.inputPorts['doppler'] = self.addDoppler
        return None

    logging_name = 'isce.estamb'

    def __init__(self, name=None):
        super(Estamb, self).__init__('estamb', name)
        self.rawImage = None
        self.numberGoodBytes = None
        self.numberBytesPerLine = None
        self.numberRangeBin = None
        self.firstSample = None
        self.lookSide = -1    #By default right looking (to be consistent with old code)

        # These pertain to the image, but aren't explicitly set
        self.firstLine = None
        self.numberValidPulses = None
        self.startRangeBin = 1
        self.shift = -0.5 ##KK,ML 2013-07-15

        # Planet information
        # the code does not actually uses the ones set to -9999,
        ## but they are passed so they
        # need to be set
        self.a = -9999
        self.e2 = -9999
        self.spin = -9999
        self.gm = -9999

        # Peg Information
        self.pegLatitude = -9999#see comment above
        self.pegLongitude = -9999
        self.pegHeading = -9999


        self.planetLocalRadius = None
        self.bodyFixedVelocity = None
        self.spacecraftHeight = None
        # Instrument Information
        self.prf = None
        self.inPhaseValue = None
        self.quadratureValue = None
        self.azimuthResolution = 5
        self.rangeSamplingRate = None
        self.chirpSlope = None
        self.rangePulseDuration = None
        self.radarWavelength = None
        # Frame Information
        self.rangeFirstSample = None
        # Orbit Information
        self.position = []
        self.dim1_position = None
        self.dim2_position = None
        self.velocity = []
        self.dim1_velocity = None
        self.dim2_velocity = None
        self.time = []
        self.dim1_time = None
        # Doppler Information
        self.dopplerCentroidCoefficients = []
        self.dim1_dopplerCentroidCoefficients = None
        # These are options
        self.numberAzimuthLooks = None
        self.numberPatches = None
        self.caltoneLocation = 0
        self.rangeChirpExtensionPoints = 0
        self.azimuthPatchSize = None
        self.overlap = 0
        self.ranfftov = 65536
        self.ranfftiq = 32768
        self.debugFlag = 0
        self.rangeSpectralWeighting = 1
        self.spectralShiftFraction = 0
        self.imrc1Accessor = 0
        self.immocompAccessor = 0
        self.imrcas1Accessor = 0
        self.imrcrm1Accessor = 0
        self.transAccessor = 0
        self.rawAccessor = 0
        self.slcAccessor = 0
        self.slcWidth = 0
        self.IQFlip = 'n'
        self.deskewFlag = 'n'
        self.secondaryRangeMigrationFlag = 'n'
        self.minAmb = -3
        self.maxAmb = 3
        # These are output
        self.entropy = []
        self.dopplerAmbiguity = 0

        self.createPorts()

        self.dictionaryOfOutputVariables = {
            'ENTROPY' : 'entropy' ,
            'DOPPLER_AMBIGUITY' : 'dopplerAmbiguity'
            }

        ## Set dictionary of Variables (more to refactor..)
        for d in (
            item.to_dict() for item in self.__class__._vars
            ):
            self.dictionaryOfVariables.update(d)

        self.descriptionOfVariables = {}
        self.mandatoryVariables = []
        self.optionalVariables = []
        self.initOptionalAndMandatoryLists()
        return None
    pass


if __name__ == '__main__':
    '''Sample implementation. Estimates ambiguity on the reference.'''
    import isceobj
    import stdproc
    from iscesys.StdOEL.StdOELPy import create_writer

    def load_pickle(step='orbit2sch'):
        import cPickle

        insarObj = cPickle.load(open('PICKLE/{0}'.format(step), 'rb'))
        return insarObj

    def runEstamb(insar):
        import copy

        stdWriter = create_writer("log", "", True, filename="estamb.log")
        objRaw = insar.referenceRawImage.copy(access_mode='read')
        v,h = insar.vh()

        objFormSlc = stdproc.createestamb()
        objFormSlc.minAmb = -3
        objFormSlc.maxAmb = 3
#        objFormSlc.setAzimuthPatchSize(8192)
#        objFormSlc.setNumberValidPulses(6144)
        objFormSlc.setBodyFixedVelocity(v)
        objFormSlc.setSpacecraftHeight(h)
        objFormSlc.setFirstLine(5000)
        objFormSlc.setNumberPatches(1)
        objFormSlc.setNumberRangeBin(insar._referenceFrame.numberRangeBins)
        objFormSlc.setLookSide(insar._lookSide)
        doppler = copy.deepcopy(insar.referenceDoppler)
#        doppler.fractionalCentroid = 0.39
        doppler.linearTerm = 0.
        doppler.quadraticTerm = 0.
        doppler.cubicTerm = 0.

        print ("Focusing Reference image")
        objFormSlc.stdWriter = stdWriter
        entropy, Amb = objFormSlc(rawImage=objRaw,
                orbit=insar.referenceOrbit,
                frame=insar.referenceFrame,
                planet=insar.referenceFrame.instrument.platform.planet,
                doppler=doppler,
                peg=insar.peg)

        objRaw.finalizeImage()
        stdWriter.finalize()

        print ('Input Doppler: ', doppler.fractionalCentroid)
        print ('Doppler Ambiguity: ', Amb)


    ####The main driver
    iObj = load_pickle()
    runEstamb(iObj)
