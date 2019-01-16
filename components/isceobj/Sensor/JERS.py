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
# Author: Walter Szeliga
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




import os
import string
import datetime
#import CEOS
import isceobj.Sensor.CEOS as CEOS
from isceobj.Scene.Frame import Frame
from isceobj.Planet import Planet
from isceobj import Constants
from isceobj.Orbit.Orbit import StateVector
from iscesys.Component.Component import Component
from isceobj.Sensor import xmlPrefix

class JERS(Component):
    """
    Code to read CEOSFormat leader files for ERS-1/2 SAR data.
    The tables used to create this parser are based on document
    number ER-IS-EPO-GS-5902.1 from the European Space Agency.
    """

    #Parsers.CEOS.CEOSFormat.ceosTypes['text'] = {'typeCode': 63, 'subtypeCode': [18,18,18]}
    #Parsers.CEOS.CEOSFormat.ceosTypes['leaderFile'] = {'typeCode': 192, 'subtypeCode': [63,18,18]}
    #Parsers.CEOS.CEOSFormat.ceosTypes['dataSetSummary'] = {'typeCode': 10, 'subtypeCode': [10,31,20]}
    #Parsers.CEOS.CEOSFormat.ceosTypes['platformPositionData'] = {'typeCode': 30, 'subtypeCode': [10,31,20]}
    #Parsers.CEOS.CEOSFormat.ceosTypes['facilityData'] = {'typeCode': 200, 'subtypeCode': [10,31,50]}
    #Parsers.CEOS.CEOSFormat.ceosTypes['datafileDescriptor'] = {'typeCode': 192, 'subtypeCode':[63,18,18]}
    #Parsers.CEOS.CEOSFormat.ceosTypes['signalData'] = {'typeCode': 10, 'subtypeCode': [50,31,20]}
    #Parsers.CEOS.CEOSFormat.ceosTypes['nullFileDescriptor'] = {'typeCode': 192, 'subtypeCode': [192,63,18]}

    def __init__(self):
        Component.__init__(self)
        self._leaderFile = None
        self._imageFile = None
        self.output = None

        self.frame = Frame()
        self.frame.configure()

        self.constants = {'polarization': 'HH',
                          'antennaLength': 12}

        self.descriptionOfVariables = {}
        self.dictionaryOfVariables = {'LEADERFILE': ['self._leaderFile','str','mandatory'],
                                      'IMAGEFILE': ['self._imageFile','str','mandatory'],
                                      'OUTPUT': ['self.output','str','optional']}

    def getFrame(self):
        return self.frame

    def parse(self):
        self.leaderFile = LeaderFile(file=self._leaderFile)
        self.leaderFile.parse()

        self.imageFile = ImageFile(file=self._imageFile)
        self.imageFile.parse()

        self.populateMetadata()

    def populateMetadata(self):
        """
            Create the appropriate metadata objects from our CEOSFormat metadata
        """
        frame = self.leaderFile.sceneHeaderRecord.metadata['Scene reference number'].strip()
        frame = self._decodeSceneReferenceNumber(frame)
        rangePixelSize = Constants.SPEED_OF_LIGHT/(2*self.leaderFile.sceneHeaderRecord.metadata['Range sampling rate']*1e6)

        self.frame.getInstrument().getPlatform().setMission(self.leaderFile.sceneHeaderRecord.metadata['Sensor platform mission identifier'])
        self.frame.getInstrument().getPlatform().setPlanet(Planet(pname='Earth'))

        self.frame.getInstrument().setWavelength(self.leaderFile.sceneHeaderRecord.metadata['Radar wavelength'])
        self.frame.getInstrument().setIncidenceAngle(self.leaderFile.sceneHeaderRecord.metadata['Incidence angle at scene centre'])
        self.frame.getInstrument().setPulseRepetitionFrequency(self.leaderFile.sceneHeaderRecord.metadata['Pulse Repetition Frequency'])
        self.frame.getInstrument().setRangePixelSize(rangePixelSize)
        self.frame.getInstrument().setPulseLength(self.leaderFile.sceneHeaderRecord.metadata['Range pulse length']*1e-6)
        chirpPulseBandwidth = 15.50829e6 # Is this really not in the CEOSFormat Header?
        self.frame.getInstrument().setChirpSlope(chirpPulseBandwidth/(self.leaderFile.sceneHeaderRecord.metadata['Range pulse length']*1e-6))

        self.frame.setFrameNumber(frame)
        self.frame.setOrbitNumber(self.leaderFile.sceneHeaderRecord.metadata['Orbit number'])
        #self.frame.setStartingRange(self.leaderFile.facilityRecord.metadata['Slant range reference'])
        self.frame.setProcessingFacility(self.leaderFile.sceneHeaderRecord.metadata['Processing facility identifier'])
        self.frame.setProcessingSystem(self.leaderFile.sceneHeaderRecord.metadata['Processing system identifier'])
        self.frame.setProcessingSoftwareVersion(self.leaderFile.sceneHeaderRecord.metadata['Processing version identifier'])
        self.frame.setPolarization('HH')
        self.frame.setNumberOfLines(self.imageFile.imageFDR.metadata['Number of lines per data set'])
        self.frame.setNumberOfSamples(self.imageFile.imageFDR.metadata['Number of pixels per line per SAR channel'])


        self.frame.getOrbit().setOrbitSource('Header')
        t0 = datetime.datetime(year=self.leaderFile.platformPositionRecord.metadata['Year of data point'],
                               month=self.leaderFile.platformPositionRecord.metadata['Month of data point'],
                               day=self.leaderFile.platformPositionRecord.metadata['Day of data point'])
        t0 = t0 + datetime.timedelta(seconds=self.leaderFile.platformPositionRecord.metadata['Seconds of day'])
        for i in range(self.leaderFile.platformPositionRecord.metadata['Number of data points']):
            vec = StateVector()
            t = t0 + datetime.timedelta(seconds=(i*self.leaderFile.platformPositionRecord.metadata['Time interval between DATA points']))
            vec.setTime(t)
            dataPoints = self.leaderFile.platformPositionRecord.metadata['Positional Data Points'][i]
            vec.setPosition([dataPoints['Position vector X'], dataPoints['Position vector Y'], dataPoints['Position vector Z']])
            vec.setVelocity([dataPoints['Velocity vector X'], dataPoints['Velocity vector Y'], dataPoints['Velocity vector Z']])
            self.frame.getOrbit().addStateVector(vec)

    def extractImage(self):
        raise NotImplementedError()

    def _decodeSceneReferenceNumber(self,referenceNumber):
        return referenceNumber

class LeaderFile(object):

    def __init__(self,file=None):
        self.file = file
        self.leaderFDR = None
        self.sceneHeaderRecord = None
        self.platformPositionRecord = None
        self.facilityRecord = None

    def parse(self):
        """
            Parse the leader file to create a header object
        """
        try:
            fp = open(self.file,'r')
        except IOError as errs:
            errno,strerr = errs
            print("IOError: %s" % (strerr))
            return

        # Leader record
        self.leaderFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'jers/leader_file.xml'),dataFile=fp)
        self.leaderFDR.parse()
        fp.seek(self.leaderFDR.getEndOfRecordPosition())
        # Scene Header
        self.sceneHeaderRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'jers/scene_record.xml'),dataFile=fp)
        self.sceneHeaderRecord.parse()
        fp.seek(self.sceneHeaderRecord.getEndOfRecordPosition())
        # Platform Position
        self.platformPositionRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'jers/platform_position_record.xml'),dataFile=fp)
        self.platformPositionRecord.parse()
        fp.seek(self.platformPositionRecord.getEndOfRecordPosition())
        # Facility Record
        self.facilityRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'jers/facility_record.xml'), dataFile=fp)
        self.facilityRecord.parse()
        fp.seek(self.facilityRecord.getEndOfRecordPosition())

        fp.close()

class VolumeDirectoryFile(object):

    def __init__(self,file=None):
        self.file = file
        self.metadata = {}

    def parse(self):
        try:
            fp = open(self.file,'r')
        except IOError as errs:
            errno,strerr = errs
            print("IOError: %s" % strerr)
            return

        volumeFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'jers/volume_descriptor.xml'),dataFile=fp)
        volumeFDR.parse()
        fp.seek(volumeFDR.getEndOfRecordPosition())

        fp.close()

        import pprint
        pp = pprint.PrettyPrinter()
        pp.pprint(volumeFDR.metadata)

class ImageFile(object):

    def __init__(self,file=None):
        self.file = file
        self.imageFDR = None

    def parse(self):
        try:
            fp = open(self.file,'r')
        except IOError as errs:
            errno,strerr = errs
            print("IOError: %s" % strerr)
            return

        self.imageFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'jers/image_file.xml'), dataFile=fp)
        self.imageFDR.parse()
        fp.seek(self.imageFDR.getEndOfRecordPosition())

        fp.close()
