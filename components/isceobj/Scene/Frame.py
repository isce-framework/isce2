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
# Author: Walter Szeliga
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import datetime

from isceobj.Attitude.Attitude import Attitude
from iscesys.Component.Component import Component
from isceobj.Image.Image import Image
from isceobj.Orbit.Orbit import Orbit
from isceobj.Radar.Radar import Radar
from isceobj.Util.decorators import type_check

SCHHEIGHT = Component.Parameter('_schHeight',
        public_name='SCHHEIGHT',
        default=None,
        type=float,
        mandatory=True,
        doc = 'SCH HEIGHT')

SCHVELOCITY = Component.Parameter('_schVelocity',
        public_name = 'SCHVELOCITY',
        default = None,
        type = float,
        mandatory=True,
        doc = 'SCH VELOCITY')

POLARIZATION = Component.Parameter('_polarization',
        public_name = 'POLARIZATION',
        default=None,
        type=str,
        mandatory=False,
        doc = 'Polarization.')

NUMBER_OF_SAMPLES = Component.Parameter('_numberOfSamples',
        public_name = 'NUMBER_OF_SAMPLES',
        default = None,
        type=int,
        mandatory=False,
        doc = 'Number of samples in a range line.')

NUMBER_OF_LINES = Component.Parameter('_numberOfLines',
        public_name = 'NUMBER_OF_LINES',
        default=None,
        type=int,
        mandatory=False,
        doc = 'Number of lines in the image')

STARTING_RANGE = Component.Parameter('_startingRange',
        public_name = 'STARTING_RANGE',
        default=None,
        type=float,
        mandatory=False,
        doc = 'Range to the first valid sample in the image')

SENSING_START = Component.Parameter('_sensingStart',
        public_name = 'SENSING_START',
        default = None,
        type = datetime.datetime,
        mandatory=False,
        doc = 'Date time object for UTC of first line')

SENSING_MID = Component.Parameter('_sensingMid',
        public_name = 'SENSING_MID',
        default = None,
        type = datetime.datetime,
        mandatory=False,
        doc = 'Date time object for UTC of middle of image')

SENSING_STOP = Component.Parameter('_sensingStop',
        public_name = 'SENSING_STOP',
        default = None,
        type = datetime.datetime,
        mandatory = False,
        doc = 'Date time object for UTC of last line of image')

TRACK_NUMBER = Component.Parameter('_trackNumber',
        public_name = 'TRACK_NUMBER',
        default=None,
        type = int,
        mandatory=False,
        doc = 'Track number for the acquisition')

FRAME_NUMBER = Component.Parameter('_frameNumber',
        public_name = 'FRAME_NUMBER',
        default=None,
        type = int,
        mandatory=False,
        doc = 'Frame number for the acquisition')

ORBIT_NUMBER = Component.Parameter('orbitNumber',
        public_name='ORBIT_NUMBER',
        default=None,
        type = int,
        mandatory = False,
        doc = 'Orbit number for the acquisition')

PASS_DIRECTION = Component.Parameter('_passDirection',
    public_name='PASS_DIRECTION',
    default=None,
    type = str,
    mandatory = False,
    doc = 'Ascending or Descending direction of orbit')

PROCESSING_FACILITY = Component.Parameter('_processingFacility',
    public_name='PROCESSING_FACILITY',
    default=None,
    type = str,
    mandatory = False,
    doc = 'Processing facility information')

PROCESSING_SYSTEM = Component.Parameter('_processingSystem',
    public_name='PROCESSING_SYSTEM',
    default=None,
    type = str,
    mandatory = False,
    doc = 'Processing system information')

PROCESSING_LEVEL = Component.Parameter('_processingLevel',
    public_name='PROCESSING_LEVEL',
    default=None,
    type = str,
    mandatory = False,
    doc = 'Processing level of the product')

PROCESSING_SYSTEM_VERSION = Component.Parameter('_processingSoftwareVersion',
    public_name='PROCESSING_SYSTEM_VERSION',
    default=None,
    type = str,
    mandatory = False,
    doc = 'Processing system software version')

AUX_FILE = Component.Parameter('_auxFile',
    public_name='AUX_FILE',
    default=None,
    type = str,
    mandatory = False,
    doc = 'Auxiliary file for the acquisition')

NUMBER_RANGE_BINS = Component.Parameter('_numberRangeBins',
    public_name = 'NUMBER_RANGE_BINS',
    default = None,
    type = int,
    mandatory=False,
    doc = 'Number of range bins')

SQUINT_ANGLE = Component.Parameter('_squintAngle',
    public_name = 'SQUINT_ANGLE',
    default = None,
    type = float,
    mandatory=False,
    doc = 'squint angle')

FAR_RANGE = Component.Parameter('_farRange',
    public_name = 'FAR_RANGE',
    default = None,
    type = float,
    mandatory=False,
    doc = 'Far range')

DOPPLER_VS_PIXEL = Component.Parameter('_dopplerVsPixel',
        public_name = 'DOPPLER_VS_PIXEL',
        default = None,
        type = float,
        mandatory = True,
        container = list,
        doc = 'Doppler polynomial coefficients vs pixel number')


class Frame(Component):
    """A class to represent a frame along a radar track"""

    family = 'frame'
    logging_name = 'isce.isceobj.scene.frame'

    parameter_list = (SCHHEIGHT,
                      SCHVELOCITY,
                      NUMBER_RANGE_BINS,
                      SQUINT_ANGLE,
                      POLARIZATION,
                      NUMBER_OF_SAMPLES,
                      NUMBER_OF_LINES,
                      STARTING_RANGE,
                      FAR_RANGE,
                      SENSING_START,
                      SENSING_MID,
                      SENSING_STOP,
                      TRACK_NUMBER,
                      FRAME_NUMBER,
                      ORBIT_NUMBER,
                      PASS_DIRECTION,
                      PROCESSING_FACILITY,
                      PROCESSING_SYSTEM,
                      PROCESSING_LEVEL,
                      PROCESSING_SYSTEM_VERSION,
                      AUX_FILE,
                      DOPPLER_VS_PIXEL)


    def _facilities(self):
        '''
        Defines all the user configurable facilities for this application.
        '''

        self._instrument = self.facility(
                '_instrument',
                public_name='INSTRUMENT',
                module='isceobj.Radar.Radar',
                factory='createRadar',
                args=(),
                mandatory=True,
                doc = "Radar information")

        self._orbit = self.facility(
                '_orbit',
                public_name='ORBIT',
                module = 'isceobj.Orbit.Orbit',
                factory = 'createOrbit',
                args=(),
                mandatory=True,
                doc = "Orbit information")

        self._attitude = self.facility(
                '_attitude',
                public_name='ATTITUDE',
                module='isceobj.Attitude.Attitude',
                factory='createAttitude',
                args=(),
                mandatory=True,
                doc = "Attitude Information")

        self._image = self.facility(
                '_image',
                public_name = 'IMAGE',
                module = 'isceobj.Image',
                factory = 'createRawImage',
                args=(),
                mandatory=True,
                doc = "Image Information")

    ## this init will be removed when super no longer overides the class's
    ## dictionaryOfVariables
    def __init__(self, name=''):
        super(Frame, self).__init__(family=self.__class__.family, name=name)
#        self._instrument.configure()
        self._ellipsoid = None
        self._times = []
        self._fmt = '%Y-%m-%dT%H:%M:%S.%f'
        return None

    #until a more general solutionis implemented do the apporpriate conversion
    #from datetime to str here
    def adaptToRender(self):
        import copy
        # make a copy of the stateVectors to restore it after dumping
        self._times = [copy.copy(self._sensingStart),copy.copy(self._sensingMid),copy.copy(self._sensingStop)]
        self._sensingStart = self._sensingStart.strftime(self._fmt)
        self._sensingMid = self._sensingMid.strftime(self._fmt)
        self._sensingStop = self._sensingStop.strftime(self._fmt)

    def restoreAfterRendering(self):
        self._sensingStart = self._times[0]
        self._sensingMid = self._times[1]
        self._sensingStop = self._times[2]

    def initProperties(self,catalog):
        keys = ['SENSING_START','SENSING_MID','SENSING_STOP']

        for k in keys:
            kl = k.lower()
            if kl in catalog:
                v = catalog[kl]
                attrname = getattr(globals()[k],'attrname')
                val = datetime.datetime.strptime(v,self._fmt)
                setattr(self,attrname,val)
                catalog.pop(kl)
        super().initProperties(catalog)

    @property
    def platform(self):
        return self.instrument.platform
    @property
    def planet(self):
        return self.platform.planet
    @property
    def ellipsoid(self):
        if not self._ellipsoid:
            self._ellipsoid = self.planet.ellipsoid
        return self.planet.ellipsoid
    @property
    def PRF(self):
        return self.instrument.PRF
    @property
    def radarWavelegth(self):
        return self.instrument.radarWavelength
    @property
    def rangeSamplingRate(self):
        return self.instrument.rangeSamplingRate
    @property
    def pulseLength(self):
        return self.instrument.pulseLength


    def setSchHeight(self, h):
        self._schHeight = h

    def getSchHeight(self):
        return self._schHeight

    def setNumberRangeBins(self, nrb):
        self._numberRangeBins = nrb

    def getNumberRangeBins(self):
        return self._numberRangeBins

    def setSchVelocity(self, v):
        self._schVelocity = v

    def getSchVelocity(self):
        return self._schVelocity

    def setSquintAngle(self, angle):
        self._squintAngle = angle

    def getSquintAngle(self):
        return self._squintAngle

    def setStartingRange(self, rng):
        self._startingRange = rng

    def getStartingRange(self):
        """The Starting Range, in km"""
        return self._startingRange

    def setFarRange(self, rng):
        self._farRange = rng

    def getFarRange(self):
        """The Far Range, in km"""
        return self._farRange


    @type_check(datetime.datetime)
    def setSensingStart(self, time):
        self._sensingStart = time
        pass

    def getSensingStart(self):
        """The UTC date and time of the first azimuth line"""
        return self._sensingStart

    @type_check(datetime.datetime)
    def setSensingMid(self, time):
        self._sensingMid = time
        pass

    def getSensingMid(self):
        """The UTC date and time of the azimuth line at the center of the
        scene"""
        return self._sensingMid

    @type_check(datetime.datetime)
    def setSensingStop(self, time):
        self._sensingStop = time
        pass

    def getSensingStop(self):
        """The UTC date and time of the last azimuth line"""
        return self._sensingStop

    @type_check(Radar)
    def setInstrument(self, instrument):
        self._instrument = instrument
        pass

    def getInstrument(self):
        return self._instrument

    def setOrbit(self, orbit):
        self._orbit = orbit

    def getOrbit(self):
        return self._orbit


    @type_check(Attitude)
    def setAttitude(self, attitude):
        self._attitude = attitude
        pass

    def getAttitude(self):
        return self._attitude

    @type_check(Image)
    def setImage(self, image):
        self._image = image
        pass

    def getImage(self):
        return self._image

    @property
    def image(self):
        return self._image
    @image.setter
    def image(self, image):
        return self.setImage(image)

    def getAuxFile(self):
        return self._auxFile

    def setAuxFile(self,aux):
        self._auxFile = aux

    def setPolarization(self, polarization):
        self._polarization = polarization

    def getPolarization(self):
        """The polarization of the scene"""
        return self._polarization

    def setNumberOfSamples(self, samples):
        self._numberOfSamples = samples

    def getNumberOfSamples(self):
        """The number of samples in range"""
        return self._numberOfSamples

    def setNumberOfLines(self, lines):
        self._numberOfLines = lines

    def getNumberOfLines(self):
        """The number of azimuth lines"""
        return self._numberOfLines

    def setTrackNumber(self, track):
        self._trackNumber = track

    def getTrackNumber(self):
        """The Track number of the scene"""
        return self._trackNumber

    def setOrbitNumber(self, orbit):
        self._orbitNumber = orbit

    def getOrbitNumber(self):
        """The orbit number of the scene"""
        return self._orbitNumber

    def setFrameNumber(self, frame):
        self._frameNumber = frame

    def getFrameNumber(self):
        """The frame number of the scene"""
        return self._frameNumber

    def setPassDirection(self, dir):
        self._passDirection = dir

    def getPassDirection(self):
        """The pass direction of the satellite, either ascending or descending
        """
        return self._passDirection

    def setProcessingFacility(self, facility):
        self._processingFacility = facility

    def getProcessingFacility(self):
        """The facility that processed the raw data"""
        return self._processingFacility

    def setProcessingSystem(self, system):
        self._processingSystem = system

    def getProcessingSystem(self):
        """The software used to process the raw data"""
        return self._processingSystem

    def setProcessingLevel(self, level):
        self._processingLevel = level

    def getProcessingLevel(self):
        """The level to which the raw data was processed"""
        return self._processingLevel

    def setProcessingSoftwareVersion(self, ver):
        self._processingSoftwareVersion = ver

    def getProcessingSoftwareVersion(self):
        """The software version of the processing software"""
        return self._processingSoftwareVersion

    def __str__(self):
        retstr = "Sensing Start Time: (%s)\n"
        retlst = (self._sensingStart, )
        retstr += "Sensing Mid Time: (%s)\n"
        retlst += (self._sensingMid, )
        retstr += "Sensing Stop Time: (%s)\n"
        retlst += (self._sensingStop, )
        retstr += "Orbit Number: (%s)\n"
        retlst += (self._orbitNumber, )
        retstr += "Frame Number: (%s)\n"
        retlst += (self._frameNumber, )
        retstr += "Track Number: (%s)\n"
        retlst += (self._trackNumber, )
        retstr += "Number of Lines: (%s)\n"
        retlst += (self._numberOfLines, )
        retstr += "Number of Samples: (%s)\n"
        retlst += (self._numberOfSamples, )
        retstr += "Starting Range: (%s)\n"
        retlst += (self._startingRange, )
        retstr += "Polarization: (%s)\n"
        retlst += (self._polarization, )
        retstr += "Processing Facility: (%s)\n"
        retlst += (self._processingFacility, )
        retstr += "Processing Software: (%s)\n"
        retlst += (self._processingSystem, )
        retstr += "Processing Software Version: (%s)\n"
        retlst += (self._processingSoftwareVersion, )

        return retstr % retlst


    frameNumber = property(getFrameNumber, setFrameNumber)
    instrument = property(getInstrument, setInstrument)
    numberOfLines = property(getNumberOfLines, setNumberOfLines)
    numberOfSamples = property(getNumberOfSamples, setNumberOfSamples)
    numberRangeBins = property(getNumberRangeBins, setNumberRangeBins)
    orbit = property(getOrbit, setOrbit)
    attitude = property(getAttitude, setAttitude)
    orbitNumber = property(getOrbitNumber, setOrbitNumber)
    passDirection = property(getPassDirection, setPassDirection)
    polarization = property(getPolarization, setPolarization)
    processingFacility = property(getProcessingFacility, setProcessingFacility)
    processingLevel = property(getProcessingLevel, setProcessingLevel)
    processingSoftwareVersion = property(getProcessingSoftwareVersion, setProcessingSoftwareVersion)
    processingSystem = property(getProcessingSystem, setProcessingSystem)
    sensingMid = property(getSensingMid, setSensingMid)
    sensingStart = property(getSensingStart, setSensingStart)
    sensingStop = property(getSensingStop, setSensingStop)
    squintAngle = property(getSquintAngle, setSquintAngle)
    startingRange = property(getStartingRange, setStartingRange)
    trackNumber = property(getTrackNumber, setTrackNumber)
    schHeight = property(getSchHeight, setSchHeight)
    schVelocity = property(getSchVelocity, setSchVelocity)
    auxFile = property(getAuxFile, setAuxFile)

    pass



## A mixin for objects with a Frame() that they need to look through-via
## read-only attributes.
class FrameMixin(object):
    """Mixin flattens frame's attributes"""


    @property
    def instrument(self):
        return self.frame.instrument

    @property
    def platform(self):
        return self.frame.platform

    @property
    def planet(self):
        return self.frame.planet

    @property
    def ellipsoid(self):
        return self.frame.ellipsoid

    @property
    def orbit(self):
        return self.frame.orbit

    @property
    def sensingStart(self):
        return self.frame.sensingStart

    @property
    def sensingMid(self):
        return self.frame.sensingMid

    @property
    def startingRange(self):
        return self.frame.startingRange

    @property
    def PRF(self):
        return self.frame.PRF

    @property
    def radarWavelength(self):
        return self.instrument.radarWavelength

    @property
    def squintAngle(self):
        return self.frame.squintAngle

    @squintAngle.setter
    def squintAngle(self, value):
        self.frame.squintAngle = value

    @property
    def rangeSamplingRate(self):
        return self.frame.rangeSamplingRate
    @property
    def pulseLength(self):
        return self.frame.pulseLength

    pass
