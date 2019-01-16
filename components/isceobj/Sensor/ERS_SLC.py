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



import os
import math
import array
import numpy
import string
import random
import logging
import datetime
import isceobj
from . import CEOS
from isceobj.Scene.Track import Track
from isceobj.Scene.Frame import Frame
from isceobj.Planet.Planet import Planet
from isceobj.Planet.AstronomicalHandbook import Const
from isceobj.Orbit.Orbit import StateVector
from iscesys.Component.Component import Component
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTU
#from Sensor.ReadOrbitPulseERS import ReadOrbitPulseERS
from isceobj.Sensor import xmlPrefix
from isceobj.Util.decorators import pickled, logged


LEADERFILE = Component.Parameter('_leaderFile',
    public_name='LEADERFILE',
    default = None,
    type=str,
    mandatory=True,
    doc="List of names of ERS SLC Leaderfile"
)

IMAGEFILE = Component.Parameter('_imageFile',
    public_name='IMAGEFILE',
    default = None,
    type=str,
    mandatory=True,
    doc="List of names of ERS SLC Imagefile"
)

ORBIT_TYPE = Component.Parameter('_orbitType',
    public_name='ORBIT_TYPE',
    default='',
    type=str,
    mandatory=True,
    doc="Options: ODR, PRC, PDS"
)

ORBIT_DIRECTORY = Component.Parameter('_orbitDir',
    public_name='ORBIT_DIRECTORY',
    default='',
    type=str,
    mandatory=False,
    doc="Path to the directory containing the orbit files."
)

ORBIT_FILE = Component.Parameter('_orbitFile',
    public_name='ORBIT_FILE',
    default='',
    type=str,
    mandatory=False,
    doc='Only used with PDS ORBIT_TYPE'
)

##
# Code to read CEOSFormat leader files for ERS-1/2 SAR data.  The tables used
# to create this parser are based on document number ER-IS-EPO-GS-5902.1 from
# the European Space Agency.

from .Sensor import Sensor
class ERS_SLC(Sensor):

    family = 'ers_slc'
    logging_name = 'isce.sensor.ers_slc'

    parameter_list = (IMAGEFILE,
                      LEADERFILE,
                      ORBIT_TYPE,
                      ORBIT_DIRECTORY,
                      ORBIT_FILE)      + Sensor.parameter_list

    @logged
    def __init__(self, name=''):
        super().__init__(family=self.__class__.family, name=name)

        self.frame = Frame()
        self.frame.configure()
        self.dopplerRangeTime = None

        # Constants are from
        # J. J. Mohr and S. N. Madsen. Geometric calibration of ERS satellite
        # SAR images. IEEE T. Geosci. Remote, 39(4):842-850, Apr. 2001.
        self.constants = {'polarization': 'VV',
                          'antennaLength': 10,
                          'lookDirection': 'RIGHT',
                          'chirpPulseBandwidth': 15.50829e6,
                          'rangeSamplingRate': 18.962468e6,
                          'delayTime':6.622e-6,
                          'iBias': 15.5,
                          'qBias': 15.5}
        return None

    def getFrame(self):
        return self.frame

    def parse(self):
        self.leaderFile = LeaderFile(file=self._leaderFile)
        self.leaderFile.parse()

        self.imageFile = ImageFile(self)
        self.imageFile.parse()

        self.populateMetadata()

    def populateMetadata(self):
        """
        Create the appropriate metadata objects from our CEOSFormat metadata
        """

        self._populatePlatform()
        self._populateInstrument()
        self._populateFrame()
        if (self._orbitType == 'ODR'):
            self._populateDelftOrbits()
        elif (self._orbitType == 'PRC'):
            self._populatePRCOrbits()
        elif (self._orbitType == 'PDS'):
            self._populatePDSOrbits()
        else:
            self._populateHeaderOrbit()

        self._populateDoppler()

    def _populatePlatform(self):
        """
        Populate the platform object with metadata
        """

        platform = self.frame.getInstrument().getPlatform()

        platform.setMission(self.leaderFile.sceneHeaderRecord.metadata[
            'Sensor platform mission identifier'])
        platform.setAntennaLength(self.constants['antennaLength'])
        platform.setPointingDirection(-1)
        platform.setPlanet(Planet(pname='Earth'))

    def _populateInstrument(self):
        """Populate the instrument object with metadata"""
        instrument = self.frame.getInstrument()
        prf = self.leaderFile.sceneHeaderRecord.metadata['Pulse Repetition Frequency']
        rangeSamplingRate = self.leaderFile.sceneHeaderRecord.metadata['Range sampling rate']*1.0e6
        rangePixelSize = Const.c/(2.0*rangeSamplingRate)

        instrument.setRadarWavelength(
            self.leaderFile.sceneHeaderRecord.metadata['Radar wavelength'])
        instrument.setIncidenceAngle(
            self.leaderFile.sceneHeaderRecord.metadata[
            'Incidence angle at scene centre'])
        instrument.setPulseRepetitionFrequency(prf)
        instrument.setRangeSamplingRate(rangeSamplingRate)
        instrument.setRangePixelSize(rangePixelSize)
        instrument.setPulseLength(self.leaderFile.sceneHeaderRecord.metadata[
            'Range pulse length']*1e-6)
        instrument.setChirpSlope(self.constants['chirpPulseBandwidth']/
            (self.leaderFile.sceneHeaderRecord.metadata['Range pulse length']*
             1e-6))
        instrument.setInPhaseValue(self.constants['iBias'])
        instrument.setQuadratureValue(self.constants['qBias'])

    def _populateFrame(self):
        """Populate the scene object with metadata"""
        rangeSamplingRate = self.constants['rangeSamplingRate']
        rangePixelSize = Const.c/(2.0*rangeSamplingRate)
        pulseInterval = 1.0/self.frame.getInstrument().getPulseRepetitionFrequency()
        frame = self._decodeSceneReferenceNumber(
            self.leaderFile.sceneHeaderRecord.metadata[
            'Scene reference number'])

        prf = self.frame.instrument.getPulseRepetitionFrequency()
        tau0 = self.leaderFile.sceneHeaderRecord.metadata['Zero-doppler range time of first range pixel']*1.0e-3
        startingRange = tau0*Const.c/2.0
        farRange = startingRange + self.imageFile.width*rangePixelSize


        first_line_utc = datetime.datetime.strptime(self.leaderFile.sceneHeaderRecord.metadata['Zero-doppler azimuth time of first azimuth pixel'], "%d-%b-%Y %H:%M:%S.%f")
        mid_line_utc = first_line_utc + datetime.timedelta(seconds = (self.imageFile.length-1) * 0.5 / prf)

        last_line_utc = first_line_utc + datetime.timedelta(seconds = (self.imageFile.length-1)/prf)

        self.logger.debug("Frame UTC start, mid, end times:  %s %s %s" % (first_line_utc,mid_line_utc,last_line_utc))

        self.frame.setFrameNumber(frame)
        self.frame.setOrbitNumber(self.leaderFile.sceneHeaderRecord.metadata['Orbit number'])
        self.frame.setStartingRange(startingRange)
        self.frame.setFarRange(farRange)
        self.frame.setProcessingFacility(self.leaderFile.sceneHeaderRecord.metadata['Processing facility identifier'])
        self.frame.setProcessingSystem(self.leaderFile.sceneHeaderRecord.metadata['Processing system identifier'])
        self.frame.setProcessingSoftwareVersion(self.leaderFile.sceneHeaderRecord.metadata['Processing version identifier'])
        self.frame.setPolarization(self.constants['polarization'])
        self.frame.setNumberOfLines(self.imageFile.length)
        self.frame.setNumberOfSamples(self.imageFile.width)
        self.frame.setSensingStart(first_line_utc)
        self.frame.setSensingMid(mid_line_utc)
        self.frame.setSensingStop(last_line_utc)

    def _populateHeaderOrbit(self):
        """Populate an orbit object with the header orbits"""
        self.logger.info("Using Header Orbits")
        orbit = self.frame.getOrbit()

        orbit.setOrbitSource('Header')
        orbit.setOrbitQuality('Unknown')
        t0 = datetime.datetime(year=self.leaderFile.platformPositionRecord.metadata['Year of data point'],
                               month=self.leaderFile.platformPositionRecord.metadata['Month of data point'],
                               day=self.leaderFile.platformPositionRecord.metadata['Day of data point'])
        t0 = t0 + datetime.timedelta(microseconds=self.leaderFile.platformPositionRecord.metadata['Seconds of day']*1e6)
        for i in range(self.leaderFile.platformPositionRecord.metadata['Number of data points']):
            vec = StateVector()
            deltaT = self.leaderFile.platformPositionRecord.metadata['Time interval between DATA points']
            t = t0 + datetime.timedelta(microseconds=i*deltaT*1e6)
            vec.setTime(t)
            dataPoints = self.leaderFile.platformPositionRecord.metadata['Positional Data Points'][i]
            vec.setPosition([dataPoints['Position vector X'], dataPoints['Position vector Y'], dataPoints['Position vector Z']])
            vec.setVelocity([dataPoints['Velocity vector X'], dataPoints['Velocity vector Y'], dataPoints['Velocity vector Z']])
            orbit.addStateVector(vec)

    def _populateDelftOrbits(self):
        """Populate an orbit object with the Delft orbits"""
        from isceobj.Orbit.ODR import ODR, Arclist
        self.logger.info("Using Delft Orbits")
        arclist = Arclist(os.path.join(self._orbitDir,'arclist'))
        arclist.parse()
        orbitFile = arclist.getOrbitFile(self.frame.getSensingStart())
        odr = ODR(file=os.path.join(self._orbitDir,orbitFile))


        startTimePreInterp = self.frame.getSensingStart() - datetime.timedelta(minutes=60)
        stopTimePreInterp = self.frame.getSensingStop() + datetime.timedelta(minutes=60)
        odr.parseHeader(startTimePreInterp,stopTimePreInterp)
        startTime = self.frame.getSensingStart() - datetime.timedelta(minutes=5)
        stopTime = self.frame.getSensingStop() + datetime.timedelta(minutes=5)
        self.logger.debug("Extracting orbits between %s and %s" % (startTime,stopTime))
        orbit = odr.trimOrbit(startTime,stopTime)
        self.frame.setOrbit(orbit)

    def _populatePRCOrbits(self):
        """Populate an orbit object the D-PAF PRC orbits"""
        from isceobj.Orbit.PRC import PRC, Arclist
        self.logger.info("Using PRC Orbits")
        arclist = Arclist(os.path.join(self._orbitDir,'arclist'))
        arclist.parse()
        orbitFile = arclist.getOrbitFile(self.frame.getSensingStart())
        self.logger.debug("Using file %s" % (orbitFile))
        prc = PRC(file=os.path.join(self._orbitDir,orbitFile))
        prc.parse()
        startTime = self.frame.getSensingStart() - datetime.timedelta(minutes=5)
        stopTime = self.frame.getSensingStop() + datetime.timedelta(minutes=5)
        self.logger.debug("Extracting orbits between %s and %s" % (startTime,stopTime))
        fullOrbit = prc.getOrbit()
        orbit = fullOrbit.trimOrbit(startTime,stopTime)
        self.frame.setOrbit(orbit)

    def _populatePDSOrbits(self):
        """
        Populate an orbit object using the ERS-2 PDS format
        """
        from isceobj.Orbit.PDS import PDS
        self.logger.info("Using PDS Orbits")
        pds = PDS(file=self._orbitFile)
        pds.parse()
        startTime = self.frame.getSensingStart() - datetime.timedelta(minutes=5)
        stopTime = self.frame.getSensingStop() + datetime.timedelta(minutes=5)
        self.logger.debug("Extracting orbits between %s and %s" % (startTime,stopTime))
        fullOrbit = pds.getOrbit()
        orbit = fullOrbit.trimOrbit(startTime,stopTime)
        self.frame.setOrbit(orbit)

    def _populateDoppler(self):
        '''
        Extract doppler from the CEOS file.
        '''

        prf = self.frame.instrument.getPulseRepetitionFrequency()

        #####ERS provides doppler as a function of slant range time in seconds
        d0 = self.leaderFile.sceneHeaderRecord.metadata['Cross track Doppler frequency centroid constant term']

        d1 = self.leaderFile.sceneHeaderRecord.metadata['Cross track Doppler frequency centroid linear term']

        d2 = self.leaderFile.sceneHeaderRecord.metadata['Cross track Doppler frequency centroid quadratic term']

        self.dopplerRangeTime = [d0, d1, d2]

        return

    def extractDoppler(self):

        width = self.frame.getNumberOfSamples()
        prf = self.frame.instrument.getPulseRepetitionFrequency()

        midtime = 0.5*width/self.frame.instrument.getRangeSamplingRate() 

        fd_mid = 0.0
        x = 1.0
        for ind, coeff in enumerate(self.dopplerRangeTime):
            fd_mid += coeff * x
            x *= midtime

        ####For insarApp
        quadratic = {}
        quadratic['a'] = fd_mid / prf
        quadratic['b'] = 0.0
        quadratic['c'] = 0.0


        ###For roiApp more accurate
        ####Convert stuff to pixel wise coefficients
        dr = self.frame.getInstrument().getRangePixelSize()
        norm = 0.5*Const.c/dr

        dcoeffs = []
        for ind, val in enumerate(self.dopplerRangeTime):
            dcoeffs.append( val / (norm**ind))


        self.frame._dopplerVsPixel = dcoeffs
        print('Doppler Fit: ', fit[::-1])

        return quadratic



    def extractImage(self):
        import array
        import math

        self.parse()
        try:
            out = open(self.output, 'wb')
        except:
            raise Exception('Cannot open output file: %s'%(self.output))

        self.imageFile.extractImage(output=out)
        out.close()

        rawImage = isceobj.createSlcImage()
        rawImage.setByteOrder('l')
        rawImage.setAccessMode('read')
        rawImage.setFilename(self.output)
        rawImage.setWidth(self.imageFile.width)
        rawImage.setXmin(0)
        rawImage.setXmax(self.imageFile.width)
        self.frame.setImage(rawImage)

        prf = self.frame.getInstrument().getPulseRepetitionFrequency()
        senStart = self.frame.getSensingStart()
        numPulses = int(math.ceil(DTU.timeDeltaToSeconds(self.frame.getSensingStop()-senStart)*prf))
        musec0 = (senStart.hour*3600 + senStart.minute*60 + senStart.second)*10**6 + senStart.microsecond
        maxMusec = (24*3600)*10**6#use it to check if we went across  a day. very rare
        day0 = (datetime.datetime(senStart.year,senStart.month,senStart.day) - datetime.datetime(senStart.year,1,1)).days + 1
        outputArray  = array.array('d',[0]*2*numPulses)
        self.frame.auxFile = self.output + '.aux'
        fp = open(self.frame.auxFile,'wb')
        j = -1
        for i1 in range(numPulses):
            j += 1
            musec = round((j/prf)*10**6) + musec0
            if musec >= maxMusec:
                day0 += 1
                musec0 = musec%maxMusec
                musec = musec0
                j = 0
            outputArray[2*i1] = day0
            outputArray[2*i1+1] = musec

        outputArray.tofile(fp)
        fp.close()


    def _decodeSceneReferenceNumber(self,referenceNumber):
        frameNumber = referenceNumber.split('=')
        if (len(frameNumber) > 2):
            frameNumber = frameNumber[2].strip()
        else:
            frameNumber = frameNumber[0]

        return frameNumber

class LeaderFile(object):

    def __init__(self,file=None):
        self.file = file
        self.leaderFDR = None
        self.sceneHeaderRecord = None
        self.mapProjectionRecord = None
        self.platformPositionRecord = None
        self.facilityRecord = None
        self.facilityPCSRecord = None
        self.logger = logging.getLogger('isce.sensor.ers_slc')

    def parse(self):
        """
            Parse the leader file to create a header object
        """
        try:
            fp = open(self.file,'rb')
        except IOError as strerr:
            self.logger.error("IOError: %s" % strerr)
            return

        # Leader record
        self.leaderFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'ers_slc/leader_file.xml'),dataFile=fp)
        self.leaderFDR.parse()
        fp.seek(self.leaderFDR.getEndOfRecordPosition())
        if (self.leaderFDR.metadata['Number of data set summary records'] > 0):
            # Scene Header
            self.sceneHeaderRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'ers_slc/scene_record.xml'),dataFile=fp)
            self.sceneHeaderRecord.parse()
            fp.seek(self.sceneHeaderRecord.getEndOfRecordPosition())

        if (self.leaderFDR.metadata['Number of map projection data records'] > 0):
            self.mapProjectionRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix, 'ers_slc/map_proj_record.xml'), dataFile=fp)
            self.mapProjectionRecord.parse()
            fp.seek(self.mapProjectionRecord.getEndOfRecordPosition())

        if (self.leaderFDR.metadata['Number of platform pos. data records'] > 0):
            # Platform Position
            self.platformPositionRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'ers_slc/platform_position_record.xml'),dataFile=fp)
            self.platformPositionRecord.parse()
            fp.seek(self.platformPositionRecord.getEndOfRecordPosition())

#        if (self.leaderFDR.metadata['Number of facility data records'] > 0):
            # Facility Record
#            self.facilityRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'ers_slc/facility_record.xml'), dataFile=fp)
#            self.facilityRecord.parse()
#            fp.seek(self.facilityRecord.getEndOfRecordPosition())
            # Facility PCS Record
#            self.facilityPCSRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'ers_slc/facility_related_pcs_record.xml'), dataFile=fp)
#            self.facilityPCSRecord.parse()
#            fp.seek(self.facilityPCSRecord.getEndOfRecordPosition())

        fp.close()

class VolumeDirectoryFile(object):

    def __init__(self,file=None):
        self.file = file
        self.metadata = {}
        self.logger = logging.getLogger('isce.sensor.ers_slc')

    def parse(self):
        try:
            fp = open(self.file,'r')
        except IOError as strerr:
            self.logger.error("IOError: %s" % strerr)
            return

        volumeFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'ers/volume_descriptor.xml'),dataFile=fp)
        volumeFDR.parse()
        fp.seek(volumeFDR.getEndOfRecordPosition())

        fp.close()

        import pprint
        pp = pprint.PrettyPrinter()
        pp.pprint(volumeFDR.metadata)

class ImageFile(object):

    def __init__(self,parent):
        self.parent = parent
        self.width = None
        self.length = None
        self.startTime = None
        self.imageFDR = None
        self.logger = logging.getLogger('isce.sensor.ers_slc')

        self.image_record = os.path.join(xmlPrefix,'ers_slc/image_record.xml')
        facility = self.parent.leaderFile.sceneHeaderRecord.metadata['Processing facility identifier']
        version = self.parent.leaderFile.sceneHeaderRecord.metadata['Processing system identifier']
        self.parent.logger.debug("Processing Facility: " + facility )

        self.parent.logger.debug("Processing System: " + version)
#        if(facility in ('CRDC_SARDPF','GTS - ERS')):
#            self.image_record = os.path.join(xmlPrefix,'ers/crdc-sardpf_image_record.xml')
#        elif((facility == 'D-PAF') and (version=='MSAR')):
#            self.image_record = os.path.join(xmlPrefix, 'ers/new-d-paf_image_record.xml')

    def parse(self):
        try:
            fp = open(self.parent._imageFile,'rb')
        except IOError as strerr:
            self.logger.error("IOError: %s" % strerr)
            return

        self.imageFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'ers_slc/image_file.xml'), dataFile=fp)
        self.imageFDR.parse()
        fp.seek(self.imageFDR.getEndOfRecordPosition())
#        self._calculateRawDimensions(fp)
        self.length = self.imageFDR.metadata['Number of SAR DATA records']
        self.width  =  self.imageFDR.metadata['Number of left border pixels per line'] +  \
                     self.imageFDR.metadata['Number of pixels per line per SAR channel']


        fp.close()

    def extractImage(self,output=None):
        """
            Extract the I and Q channels from the image file
        """
        if (not self.imageFDR):
            self.parse()
        try:
            fp = open(self.parent._imageFile,'rb')
        except IOError as strerr:
            self.logger.error("IOError %s" % strerr)
            return

        pixelCount = self.width
        lines = self.length

        fp.seek(self.imageFDR.getEndOfRecordPosition(),os.SEEK_SET)


        # Extract the I and Q channels
        imageData = CEOS.CEOSDB(xml=self.image_record,dataFile=fp)
        #jng  use this line as a template
        for line in range(lines):
            if ((line%1000) == 0):
                self.logger.debug("Extracting line %s" % line)

            imageData.parseFast()

            # Find missing lines
            lineCounter = imageData.metadata['Record Length']

            IQ = numpy.fromfile(fp,dtype='>i2',count=2*pixelCount)
            # Output the padded line
            IQ.astype(numpy.float32).tofile(output)

        imageData.finalizeParser()
        fp.close()

    #Parsers.CEOS.CEOSFormat.ceosTypes['text'] =
    #    {'typeCode': 63, 'subtypeCode': [18,18,18]}
    #Parsers.CEOS.CEOSFormat.ceosTypes['leaderFile'] =
    #    {'typeCode': 192, 'subtypeCode': [63,18,18]}
    #Parsers.CEOS.CEOSFormat.ceosTypes['dataSetSummary'] =
    #    {'typeCode': 10, 'subtypeCode': [10,31,20]}
    #Parsers.CEOS.CEOSFormat.ceosTypes['platformPositionData'] =
    #    {'typeCode': 30, 'subtypeCode': [10,31,20]}
    #Parsers.CEOS.CEOSFormat.ceosTypes['facilityData'] =
    #    {'typeCode': 200, 'subtypeCode': [10,31,50]}
    #Parsers.CEOS.CEOSFormat.ceosTypes['datafileDescriptor'] =
    #    {'typeCode': 192, 'subtypeCode':[63,18,18]}
    #Parsers.CEOS.CEOSFormat.ceosTypes['signalData'] =
    #    {'typeCode': 10, 'subtypeCode': [50,31,20]}
    #Parsers.CEOS.CEOSFormat.ceosTypes['nullFileDescriptor'] =
    #    {'typeCode': 192, 'subtypeCode': [192,63,18]}
