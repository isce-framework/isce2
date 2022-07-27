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


LEADERFILE = Component.Parameter('_leaderFileList',
    public_name='LEADERFILE',
    default = '',
    container=list,
    type=str,
    mandatory=True,
    doc="List of names of ALOS Leaderfile"
)

IMAGEFILE = Component.Parameter('_imageFileList',
    public_name='IMAGEFILE',
    default = '',
    container=list,
    type=str,
    mandatory=True,
    doc="List of names of ALOS Imagefile"
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
class ERS(Sensor):

    family = 'ers'
    logging_name = 'isce.sensor.ers'

    parameter_list = (IMAGEFILE,
                      LEADERFILE,
                      ORBIT_TYPE,
                      ORBIT_DIRECTORY,
                      ORBIT_FILE)     + Sensor.parameter_list

    @logged
    def __init__(self, name=''):
        super().__init__(family=self.__class__.family, name=name)
        self._leaderFile = None
        self._imageFile = None
        self.frameList = []

        self.frame = Frame()
        self.frame.configure()

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
        pri = self.imageFile.firstPri
        rangeSamplingRate = self.constants['rangeSamplingRate']
        #rangeSamplingRate = self.leaderFile.sceneHeaderRecord.metadata[
        #    'Range sampling rate']*1e6
        rangePixelSize = Const.c/(2.0*rangeSamplingRate)
        pulseInterval = 4.0/rangeSamplingRate*(pri+2.0)
        prf = 1.0/pulseInterval

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
        #rangeSamplingRate = self.leaderFile.sceneHeaderRecord.metadata[
        #    'Range sampling rate']*1e6
        rangePixelSize = Const.c/(2.0*rangeSamplingRate)
        pulseInterval = 1.0/self.frame.getInstrument().getPulseRepetitionFrequency()
        frame = self._decodeSceneReferenceNumber(
            self.leaderFile.sceneHeaderRecord.metadata[
            'Scene reference number'])
        startingRange = (9*pulseInterval + self.imageFile.minSwst*4/rangeSamplingRate-self.constants['delayTime'])*Const.c/2.0
        farRange = startingRange + self.imageFile.width*rangePixelSize
        # Use the Scene center time to get the date, then use the ICU on board time from the image for the rest
        centerLineTime = datetime.datetime.strptime(self.leaderFile.sceneHeaderRecord.metadata['Scene centre time'],"%Y%m%d%H%M%S%f")
        first_line_utc = datetime.datetime(year=centerLineTime.year, month=centerLineTime.month, day=centerLineTime.day)
        if(self.leaderFile.sceneHeaderRecord.metadata['Processing facility identifier'] in ('CRDC_SARDPF','GTS - ERS')):
            first_line_utc = first_line_utc + datetime.timedelta(milliseconds=self.imageFile.startTime)
        else:
            deltaSeconds = (self.imageFile.startTime - self.leaderFile.sceneHeaderRecord.metadata['Satellite encoded binary time code'])* 1/256.0
            # Sometimes, the ICU on board clock is corrupt, if the time suggested by the on board clock is more than
            # 5 days from the satellite clock time, assume its bogus and use the low-precision scene centre time
            if (math.fabs(deltaSeconds) > 5*86400):
                        self.logger.warning("ICU on board time appears to be corrupt, resorting to low precision clock")
                        first_line_utc = centerLineTime - datetime.timedelta(microseconds=pulseInterval*(self.imageFile.length/2.0)*1e6)
            else:
                satelliteClockTime = datetime.datetime.strptime(self.leaderFile.sceneHeaderRecord.metadata['Satellite clock time'],"%Y%m%d%H%M%S%f")
                first_line_utc = satelliteClockTime + datetime.timedelta(microseconds=int(deltaSeconds*1e6))
        mid_line_utc = first_line_utc + datetime.timedelta(microseconds=pulseInterval*(self.imageFile.length/2.0)*1e6)
        last_line_utc = first_line_utc + datetime.timedelta(microseconds=pulseInterval*self.imageFile.length*1e6)
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
        self.logger.info('Using ODR file: ' + orbitFile)

        odr = ODR(file=os.path.join(self._orbitDir,orbitFile))
        #jng it seem that for this tipe of orbit points are separated by 60 sec. In ODR at least 9 state vectors are needed to compute the velocities. add
        # extra time before and after to allow interpolation, but do not do it for all data points. too slow
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

    def extractImage(self):
        import array
        import math
        if(len(self._imageFileList) != len(self._leaderFileList)):
            self.logger.error("Number of leader files different from number of image files.")
            raise Exception

        self.frameList = []

        for i in range(len(self._imageFileList)):
            appendStr = "_" + str(i)
            #if only one file don't change the name
            if(len(self._imageFileList) == 1):
                appendStr = ''

            self.frame = Frame()
            self.frame.configure()

            self._leaderFile = self._leaderFileList[i]
            self._imageFile = self._imageFileList[i]
            self.leaderFile = LeaderFile(file=self._leaderFile)
            self.leaderFile.parse()

            self.imageFile = ImageFile(self)

            try:
                outputNow = self.output + appendStr
                out = open(outputNow,'wb')
            except IOError as strerr:
                self.logger.error("IOError: %s" % strerr)
                return

            self.imageFile.extractImage(output=out)
            out.close()

            rawImage = isceobj.createRawImage()
            rawImage.setByteOrder('l')
            rawImage.setAccessMode('read')
            rawImage.setFilename(outputNow)
            rawImage.setWidth(self.imageFile.width)
            rawImage.setXmin(0)
            rawImage.setXmax(self.imageFile.width)
            self.frame.setImage(rawImage)
            self.populateMetadata()
            self.frameList.append(self.frame)
            #jng Howard Z at this point adjusts the sampling starting time for imagery generated from CRDC_SARDPF facility.
            # for now create the orbit aux file based in starting time and prf
            prf = self.frame.getInstrument().getPulseRepetitionFrequency()
            senStart = self.frame.getSensingStart()
            numPulses = int(math.ceil(DTU.timeDeltaToSeconds(self.frame.getSensingStop()-senStart)*prf))
            # the aux files has two entries per line. day of the year and microseconds in the day
            musec0 = (senStart.hour*3600 + senStart.minute*60 + senStart.second)*10**6 + senStart.microsecond
            maxMusec = (24*3600)*10**6#use it to check if we went across  a day. very rare
            day0 = (datetime.datetime(senStart.year,senStart.month,senStart.day) - datetime.datetime(senStart.year,1,1)).days + 1
            outputArray  = array.array('d',[0]*2*numPulses)
            self.frame.auxFile = outputNow + '.aux'
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

        tk = Track()
        if(len(self._imageFileList) > 1):
            self.frame = tk.combineFrames(self.output,self.frameList)

            for i in range(len(self._imageFileList)):
                try:
                    os.remove(self.output + "_" + str(i))
                except OSError:
                    print("Error. Cannot remove temporary file",self.output + "_" + str(i))
                    raise OSError



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
        self.platformPositionRecord = None
        self.facilityRecord = None
        self.facilityPCSRecord = None
        self.logger = logging.getLogger('isce.sensor.ers')

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
        self.leaderFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'ers/leader_file.xml'),dataFile=fp)
        self.leaderFDR.parse()
        fp.seek(self.leaderFDR.getEndOfRecordPosition())
        if (self.leaderFDR.metadata['Number of data set summary records'] > 0):
            # Scene Header
            self.sceneHeaderRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'ers/scene_record.xml'),dataFile=fp)
            self.sceneHeaderRecord.parse()
            fp.seek(self.sceneHeaderRecord.getEndOfRecordPosition())
        if (self.leaderFDR.metadata['Number of platform pos. data records'] > 0):
            # Platform Position
            self.platformPositionRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'ers/platform_position_record.xml'),dataFile=fp)
            self.platformPositionRecord.parse()
            fp.seek(self.platformPositionRecord.getEndOfRecordPosition())
        if (self.leaderFDR.metadata['Number of facility data records'] > 0):
            # Facility Record
            self.facilityRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'ers/facility_record.xml'), dataFile=fp)
            self.facilityRecord.parse()
            fp.seek(self.facilityRecord.getEndOfRecordPosition())
            # Facility PCS Record
            self.facilityPCSRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'ers/facility_related_pcs_record.xml'), dataFile=fp)
            self.facilityPCSRecord.parse()
            fp.seek(self.facilityPCSRecord.getEndOfRecordPosition())

        fp.close()

class VolumeDirectoryFile(object):

    def __init__(self,file=None):
        self.file = file
        self.metadata = {}
        self.logger = logging.getLogger('isce.sensor.ers')

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
        self.minSwst = None
        self.maxSwst = None
        self.firstPri = None
        self.startTime = None
        self.imageFDR = None
        self.logger = logging.getLogger('isce.sensor.ers')

        self.image_record = os.path.join(xmlPrefix,'ers/image_record.xml')
        facility = self.parent.leaderFile.sceneHeaderRecord.metadata['Processing facility identifier']
        version = self.parent.leaderFile.sceneHeaderRecord.metadata['Processing system identifier']
        self.parent.logger.debug("Processing Facility: " + facility )

        self.parent.logger.debug("Processing System: " + version)
        if(facility in ('CRDC_SARDPF','GTS - ERS')):
            self.image_record = os.path.join(xmlPrefix,'ers/crdc-sardpf_image_record.xml')
        elif((facility == 'D-PAF') and (version=='MSAR')):
            self.image_record = os.path.join(xmlPrefix, 'ers/new-d-paf_image_record.xml')

    def parse(self):
        try:
            fp = open(self.parent._imageFile,'rb')
        except IOError as strerr:
            self.logger.error("IOError: %s" % strerr)
            return

        self.imageFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'ers/image_file.xml'), dataFile=fp)
        self.imageFDR.parse()
        fp.seek(self.imageFDR.getEndOfRecordPosition())
        self._calculateRawDimensions(fp)

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

        (maxSwst,minSwst) = self._calculateRawDimensions(fp)

        lines = self.imageFDR.metadata['Number of SAR DATA records']
        pixelCount = self.imageFDR.metadata['Number of left border pixels per line'] +  \
                     self.imageFDR.metadata['Number of pixels per line per SAR channel'] + \
                     self.imageFDR.metadata['Number of right border pixels per line']
        suffixSize = self.imageFDR.metadata['Number of bytes of suffix data per record']

        fp.seek(self.imageFDR.getEndOfRecordPosition(),os.SEEK_SET)
        lastSwst = 0
        lastLineCounter = 0
        lineGap = 0
        # Extract the I and Q channels
        imageData = CEOS.CEOSDB(xml=self.image_record,dataFile=fp)
        #jng  use this line as a template
        IQLine = array.array('B',[random.randint(15,16)*x for x in [1]*self.width])
        IQ = array.array('B',[x for x in [0]*self.width])
        IQFile = array.array('B',[x for x in [0]*2*pixelCount])
        for line in range(lines):
            if ((line%1000) == 0):
                self.logger.debug("Extracting line %s" % line)

            imageData.parseFast()

            # Find missing range values
            currentSwst = imageData.metadata['Sampling window start time']
            if ((currentSwst>500) and (currentSwst<1500) and (currentSwst-minSwst)%22 == 0):
                lastSwst = currentSwst
            leftPad = (lastSwst - minSwst)*8
            rightPad = self.width - leftPad - 2*pixelCount

            # Find missing lines
            lineCounter = imageData.metadata['Image format counter']

            if (lineCounter == 0):
                self.logger.warning("Zero line counter at line %s" % (line+1))
                lastLineCounter += 1
                continue

            # Initialize the line counter
            if (line == 0):
                lastLineCounter = lineCounter-1

            lineGap = lineCounter - lastLineCounter-1
            #self.logger.debug("Line Counter: %s Last Line Counter: %s Line Gap: %s line: %s" % (lineCounter,lastLineCounter,lineGap,line))
            skipLine = False
            if (lineGap > 0):
                if (lineGap > 30000):
                    self.logger.warn("Bad Line Counter on line %s, Gap length too large (%s)" % (line+1,lineGap))
                    fp.seek((2*pixelCount+suffixSize),os.SEEK_CUR)
                    lastLineCounter += 1
                    continue
                self.logger.debug("Gap of length %s at line %s" % (lineGap,(line+1)))
                #jng just put a predefine sequence af random values. randint very slow
                #IQ = array.array('B',[random.randint(15,16)*x for x in [1]*(leftPad+2*pixelCount+rightPad)])
                IQ = array.array('B',[IQLine[i] for i in range(self.width)])
                for i in range(lineGap):
                    IQ.tofile(output) # It may be better to fill missing lines with random 15's and 16's rather than copying the last good line
                    lastLineCounter += 1
            elif (lineGap == -1):
                skipLine = True
            elif (lineGap < 0):
                self.logger.warn("Unusual Line Gap %s at line %s" % (lineGap,(line+1)))
                raise IndexError

            #self.logger.debug("Extracting line %s" % (line+1))
            # Pad data with random integers around the I and Q bias of 15.5 on the left
            #jng just put a predefine sequence af random values. randint very slow
            #IQ = array.array('B',[random.randint(15,16)*x for x in [1]*leftPad])
            IQ = array.array('B',[IQLine[i] for i in range(leftPad)])
            # Read the I and Q values
            IQ.fromfile(fp,2*pixelCount)
            fp.seek(suffixSize,os.SEEK_CUR)
            # Now pad on the right
            #jng just put a predefine sequence af random values. randint very slow
            #IQ.extend([random.randint(15,16)*x for x in [1]*rightPad])
            IQ.extend([IQLine[i] for i in range(rightPad)])
            # Output the padded line
            if not skipLine:
                IQ.tofile(output)
                lastLineCounter += 1

        imageData.finalizeParser()
        fp.close()

    def _calculateRawDimensions(self,fp):
        """
            Run through the data file once, and calculate the valid sampling window start time range.
        """
        lines = self.imageFDR.metadata['Number of SAR DATA records']
        pixelCount = self.imageFDR.metadata['Number of left border pixels per line'] + self.imageFDR.metadata['Number of pixels per line per SAR channel'] + self.imageFDR.metadata['Number of right border pixels per line']
        suffixSize = self.imageFDR.metadata['Number of bytes of suffix data per record']
        self.length = lines
        expectedFileSize = self.imageFDR.metadata['Record Length'] + self.imageFDR.metadata['SAR DATA record length']*self.imageFDR.metadata['Number of SAR DATA records']

        fp.seek(0,os.SEEK_END)
        actualSize = fp.tell()
        if (expectedFileSize != actualSize):
            self.logger.info("File too short.  Expected %s bytes, found %s bytes" % (expectedFileSize,actualSize))
            lines = (actualSize - self.imageFDR.metadata['Record Length'])/(12+self.imageFDR.metadata['Number of bytes of prefix data per record']+self.imageFDR.metadata['Number of bytes of SAR data per record']+self.imageFDR.metadata['Number of bytes of suffix data per record'])
            expectedFileSize = self.imageFDR.metadata['Record Length'] + self.imageFDR.metadata['SAR DATA record length']*lines
            self.logger.info("%s (%s bytes total) lines of data estimated (%s expected)" % (lines,expectedFileSize,self.length))

        fp.seek(self.imageFDR.getEndOfRecordPosition(),os.SEEK_SET)

        mstime = []
        icu = []
        swst = []
        pri = []
        lastLineCounter = None
        lineGap = 0
        # Calculate the minimum and maximum Sampling Window Start Times
        imageData = CEOS.CEOSDB(xml=self.image_record,dataFile=fp)
        mstime = [0]*lines
        icu = [0]*lines
        pri = [0]*lines
        swst = [0]*lines
        i = 0
        for line in range(lines):
            imageData.parseFast()
            lineCounter = imageData.metadata['Image format counter']
            if (not lastLineCounter):
                lastLineCounter = lineCounter
            else:
                lineGap = lineCounter - lastLineCounter-1
                lastLineCounter = lineCounter
            if (lineGap != 0):
                self.length += lineGap
            mstime[i] = imageData.metadata['Record time in milliseconds']
            icu[i] = imageData.metadata['ICU on board time']
            swst[i]  = imageData.metadata['Sampling window start time']
            pri[i] = imageData.metadata['Pulse repetition interval']
            fp.seek(2*pixelCount,os.SEEK_CUR)
            fp.seek(suffixSize,os.SEEK_CUR)
            i += 1
        imageData.finalizeParser()
        if(self.parent.leaderFile.sceneHeaderRecord.metadata['Processing facility identifier'] in ('CRDC_SARDPF','GTS - ERS')):
            self.startTime = mstime[0]
        else:
            self.startTime = icu[0]
        self.firstPri= pri[0]
        s = swst[:]
        for val in swst:
            if ((val<500) or (val>1500) or ((val-swst[0])%22 != 0)):
                s.remove(val)

        self.minSwst = min(s)
        self.maxSwst = max(s)
        pad = (self.maxSwst - self.minSwst)*8
        self.width = 2*pixelCount + pad

        return self.maxSwst,self.minSwst


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
