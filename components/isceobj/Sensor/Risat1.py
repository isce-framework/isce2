#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2014 California Institute of Technology. ALL RIGHTS RESERVED.
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




import os
import datetime
import isceobj.Sensor.CEOS as CEOS
import logging
from isceobj.Scene.Frame import Frame
from isceobj.Orbit.Orbit import StateVector,Orbit
from isceobj.Orbit.Inertial import ECI2ECR
from isceobj.Orbit.OrbitExtender import OrbitExtender
from isceobj.Planet.AstronomicalHandbook import Const
from isceobj.Planet.Planet import Planet
from iscesys.Component.Component import Component
from isceobj.Util.decorators import pickled, logged
from isceobj.Sensor import xmlPrefix
from isceobj.Util import Polynomial
from iscesys.DateTimeUtil import secondsSinceMidnight
import numpy as np
import struct
import pprint

LEADERFILE = Component.Parameter(
    '_leaderFile',
    public_name='LEADERFILE',
    default = '',
    type=str,
    mandatory=True,
    doc="Name of Risat1 Leaderfile"
)

IMAGEFILE = Component.Parameter(
    '_imageFile',
    public_name='IMAGEFILE',
    default = '',
    type=str,
    mandatory=True,
    doc="name of Risat1 Imagefile"
)

METAFILE = Component.Parameter(
    '_metaFile',
    public_name='METAFILE',
    default = '',
    type=str,
    mandatory=False,
    doc="Name of Risat1 metafile"
)

from .Sensor import Sensor

class Risat1(Sensor):
    """
        Code to read CEOSFormat leader files for Risat-1 SAR data.
    """
    family = "risat1"
    logging_name = 'isce.sensor.Risat1'
    parameter_list = (IMAGEFILE, LEADERFILE, METAFILE) + Sensor.parameter_list

    @logged
    def __init__(self, name=''):
        super().__init__(family=self.__class__.family, name=name)
        self.imageFile = None
        self.leaderFile = None

        #####Specific doppler functions for RISAT1
        self.doppler_coeff = None
        self.lineDirection = None
        self.pixelDirection = None

        self.frame = Frame()
        self.frame.configure()

        self.constants = {'polarization': 'HH',
                          'antennaLength': 15}


    def getFrame(self):
        return self.frame

    def parse(self):
        self.leaderFile = LeaderFile(self, file=self._leaderFile)
        self.leaderFile.parse()

        self.imageFile = ImageFile(self, file=self._imageFile)
        self.imageFile.parse()

        self.populateMetadata()

    def populateMetadata(self):
        """
            Create the appropriate metadata objects from our CEOSFormat metadata
        """
        frame = self._decodeSceneReferenceNumber(self.leaderFile.sceneHeaderRecord.metadata['Scene reference number'])
        try:
            rangePixelSize = Const.c/(2*self.leaderFile.sceneHeaderRecord.metadata['Range sampling rate'])
        except ZeroDivisionError:
            rangePixelSize = 0


        print('Average height: ', self.leaderFile.sceneHeaderRecord.metadata['Average terrain height in km'])

        ins = self.frame.getInstrument()
        platform = ins.getPlatform()
        platform.setMission(self.leaderFile.sceneHeaderRecord.metadata['Sensor platform mission identifier'])
        platform.setAntennaLength(self.constants['antennaLength'])
        platform.setPointingDirection(1)
        platform.setPlanet(Planet(pname='Earth'))

        ins.setRadarWavelength(self.leaderFile.sceneHeaderRecord.metadata['Radar wavelength'])
        ins.setIncidenceAngle(self.leaderFile.sceneHeaderRecord.metadata['Incidence angle at scene centre'])
        self.frame.getInstrument().setPulseRepetitionFrequency(self.leaderFile.sceneHeaderRecord.metadata['Pulse Repetition Frequency'])
        ins.setRangePixelSize(rangePixelSize)
        ins.setRangeSamplingRate(self.leaderFile.sceneHeaderRecord.metadata['Range sampling rate'])
        ins.setPulseLength(self.leaderFile.sceneHeaderRecord.metadata['Range pulse length'])

#        chirpPulseBandwidth = self.leaderFile.processingRecord.metadata['Pulse bandwidth code']*1e4
#        ins.setChirpSlope(chirpPulseBandwidth/self.leaderFile.sceneHeaderRecord.metadata['Range pulse length'])
        
        ins.setChirpSlope(7.5e12)
        ins.setInPhaseValue(127.0)
        ins.setQuadratureValue(127.0)

        self.lineDirection = self.leaderFile.sceneHeaderRecord.metadata['Time direction indicator along line direction'].strip()
        self.pixelDirection =  self.leaderFile.sceneHeaderRecord.metadata['Time direction indicator along pixel direction'].strip()

        ######RISAT-1 sensor orientation convention is opposite to ours
#        lookSide = self.leaderFile.processingRecord.metadata['Sensor orientation']
#        if lookSide == 'RIGHT':
#            platform.setPointingDirection(1)
#        elif lookSide == 'LEFT':
#            platform.setPointingDirection(-1)
#        else:
#            raise Exception('Unknown look side')

        self.frame.setFrameNumber(frame)
        self.frame.setOrbitNumber(self.leaderFile.sceneHeaderRecord.metadata['Orbit number'])
        self.frame.setProcessingFacility(self.leaderFile.sceneHeaderRecord.metadata['Processing facility identifier'])
        self.frame.setProcessingSystem(self.leaderFile.sceneHeaderRecord.metadata['Processing system identifier'])
        self.frame.setProcessingSoftwareVersion(self.leaderFile.sceneHeaderRecord.metadata['Processing version identifier'])
        self.frame.setPolarization(self.constants['polarization'])
        self.frame.setNumberOfLines(self.imageFile.imageFDR.metadata['Number of lines per data set'])
        self.frame.setNumberOfSamples(self.imageFile.imageFDR.metadata['Number of pixels per line per SAR channel'])

        ######

        self.frame.getOrbit().setOrbitSource('Header')
        self.frame.getOrbit().setOrbitQuality(self.leaderFile.platformPositionRecord.metadata['Orbital elements designator'])
        t0 = datetime.datetime(year=2000+self.leaderFile.platformPositionRecord.metadata['Year of data point'],
                               month=self.leaderFile.platformPositionRecord.metadata['Month of data point'],
                               day=self.leaderFile.platformPositionRecord.metadata['Day of data point'])
        t0 = t0 + datetime.timedelta(seconds=self.leaderFile.platformPositionRecord.metadata['Seconds of day'])

        #####Read in orbit in inertial coordinates
        orb = Orbit()
        deltaT = self.leaderFile.platformPositionRecord.metadata['Time interval between DATA points']
        numPts = self.leaderFile.platformPositionRecord.metadata['Number of data points']


        for i in range(numPts):
            vec = StateVector()
            t = t0 + datetime.timedelta(seconds=i*deltaT)
            vec.setTime(t)

            dataPoints = self.leaderFile.platformPositionRecord.metadata['Positional Data Points'][i]
            pos = [dataPoints['Position vector X'], dataPoints['Position vector Y'], dataPoints['Position vector Z']]
            vel = [dataPoints['Velocity vector X'], dataPoints['Velocity vector Y'], dataPoints['Velocity vector Z']]
            vec.setPosition(pos)
            vec.setVelocity(vel)
            orb.addStateVector(vec)

        #####Convert orbits from ECI to ECEF frame.
        t0 = orb._stateVectors[0]._time
        ang = self.leaderFile.platformPositionRecord.metadata['Greenwich mean hour angle']

        cOrb = ECI2ECR(orb, GAST=ang, epoch=t0)
        wgsorb = cOrb.convert()


        #####Extend the orbits by a few points
        planet = self.frame.instrument.platform.planet
        orbExt = OrbitExtender()
        orbExt.configure()
        orbExt._newPoints = 4
        newOrb = orbExt.extendOrbit(wgsorb)

        orb = self.frame.getOrbit()

        for sv in newOrb:
            orb.addStateVector(sv)




    def extractImage(self):
        import isceobj
        if (self.imageFile is None) or (self.leaderFile is None):
            self.parse()

        try:
            out = open(self.output, 'wb')
        except IOError as strerr:
            self.logger.error("IOError: %s" % strerr)

        self.imageFile.extractImage(output=out)
        out.close()

        self.frame.setSensingStart(self.imageFile.sensingStart )
        self.frame.setSensingStop(self.imageFile.sensingStop)
        sensingMid = self.imageFile.sensingStart + datetime.timedelta(seconds = 0.5* (self.imageFile.sensingStop - self.imageFile.sensingStart).total_seconds())
        self.frame.setSensingMid(sensingMid)

        dr = self.frame.instrument.rangePixelSize
        self.frame.setStartingRange(self.imageFile.nearRange)
        self.frame.setFarRange(self.imageFile.nearRange + (self.imageFile.width-1)*dr)
        self.doppler_coeff = self.imageFile.dopplerCoeff
        self.frame.getInstrument().setPulseRepetitionFrequency(self.imageFile.prf)
        self.frame.instrument.setPulseLength(self.imageFile.chirpLength)

        print('Pulse length: ', self.imageFile.chirpLength)
        print('Roll angle: ', self.imageFile.roll)

        if self.imageFile.roll > 0. :
            self.frame.instrument.platform.setPointingDirection(1)
        else:
            self.frame.instrument.platform.setPointingDirection(-1)

        rawImage = isceobj.createRawImage()
        rawImage.setByteOrder('l')
        rawImage.setAccessMode('read')
        rawImage.setFilename(self.output)
        rawImage.setWidth(self.imageFile.width*2)
        rawImage.setXmin(0)
        rawImage.setXmax(self.imageFile.width*2)
        rawImage.renderHdr()
        self.frame.setImage(rawImage)

        return


    def extractDoppler(self):
        '''
        Evaluate the doppler polynomial and return the average value for now.
        '''
        print('Doppler: ', self.doppler_coeff)
        quadratic = {}
        quadratic['a'] = self.doppler_coeff[1] / self.frame.getInstrument().getPulseRepetitionFrequency()
        quadratic['b'] = 0.
        quadratic['c'] = 0.
        return quadratic

    def _decodeSceneReferenceNumber(self,referenceNumber):
        return referenceNumber



class LeaderFile(object):

    def __init__(self, parent, file=None):
        self.parent = parent
        self.file = file
        self.leaderFDR = None
        self.sceneHeaderRecord = None
        self.platformPositionRecord = None

    def parse(self):
        """
            Parse the leader file to create a header object
        """
        try:
            fp = open(self.file,'rb')
        except IOError as errs:
            errno,strerr = errs
            print("IOError: %s" % strerr)
            return
        # Leader record
        self.leaderFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'risat/leader_file.xml'),dataFile=fp)
        self.leaderFDR.parse()
        fp.seek(self.leaderFDR.getEndOfRecordPosition())
#        pprint.pprint(self.leaderFDR.metadata)

        # Scene Header
        self.sceneHeaderRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'risat/scene_record.xml'),dataFile=fp)
        self.sceneHeaderRecord.parse()
        fp.seek(self.sceneHeaderRecord.getEndOfRecordPosition())
        pprint.pprint(self.sceneHeaderRecord.metadata)

        # Platform Position
        self.platformPositionRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'risat/platform_position_record.xml'),dataFile=fp)
        self.platformPositionRecord.parse()
        fp.seek(self.platformPositionRecord.getEndOfRecordPosition())

        fp.close()

class VolumeDirectoryFile(object):

    def __init__(self,file=None):
        self.file = file
        self.metadata = {}

    def parse(self):
        try:
            fp = open(self.file,'rb')
        except IOError as errs:
            errno,strerr = errs
            print("IOError: %s" % strerr)
            return

        volumeFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'risat/volume_descriptor.xml'),dataFile=fp)
        volumeFDR.parse()
        fp.seek(volumeFDR.getEndOfRecordPosition())

        fp.close()


class ImageFile(object):

    def __init__(self, parent, file=None):
        self.parent = parent
        self.file = file
        self.imageFDR = None
        self.sensingStart = None
        self.sensingStop = None
        self.nearRange = None
        self.farRange = None
        self.prf = None
        self.chirpLength = None
        self.roll = None
        self.dopplerCoeff = None
        self.image_record = os.path.join(xmlPrefix,'risat/image_record.xml')
        self.logger = logging.getLogger('isce.sensor.risat')

    def parse(self):
        try:
            fp = open(self.file,'rb')
        except IOError as errs:
            errno,strerr = errs
            print("IOError: %s" % strerr)
            return

        self.imageFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'risat/image_file.xml'), dataFile=fp)
        self.imageFDR.parse()
        fp.seek(self.imageFDR.getEndOfRecordPosition())
        self._calculateRawDimensions(fp)

        fp.close()

    def writeRawData(self, fp, line):
        '''
        Convert complex integer to complex64 format.
        '''
#        cJ = np.complex64(1j)
#        data = line[0::2] + cJ * line[1::2]
        (line+127.0).astype(np.uint8).tofile(fp)


    def extractImage(self, output=None):
        """
            Extract I and Q channels from the image file
        """
        if self.imageFDR is None:
            self.parse()

        try:
            fp = open(self.file, 'rb')
        except IOError as strerr:
            self.logger.error(" IOError: %s" % strerr)
            return



        fp.seek(self.imageFDR.getEndOfRecordPosition(),os.SEEK_SET)

#        pprint.pprint(self.imageFDR.metadata)

        prf = self.parent.leaderFile.sceneHeaderRecord.metadata['Pulse Repetition Frequency']
        # Extract the I and Q channels
        imageData = CEOS.CEOSDB(xml=self.image_record,dataFile=fp)
        self.length = self.length - 1

        for line in range(self.length):
            if ((line%1000) == 0):
                self.logger.debug("Extracting line %s" % line)

            imageData.parseFast()


            if line==0:
#                pprint.pprint(imageData.metadata)
                dataLen = imageData.metadata['Actual count of data pixels']
                yr = imageData.metadata['Sensor acquisition year']
                dys = imageData.metadata['Sensor acquisition day of year']
                msecs = imageData.metadata['Sensor acquisition milliseconds of day'] + imageData.metadata['Acquisition time bias in ms']
                self.sensingStart = datetime.datetime(yr,1,1) + datetime.timedelta(days=(dys-1)) + datetime.timedelta(seconds = msecs*1e-3)
                self.nearRange = imageData.metadata['Slant range to 1st pixel']
                self.prf = imageData.metadata['PRF']
                self.roll = imageData.metadata['Platform roll in micro degs'] * 1.0e-6

            if line==(self.length-1):
                yr = imageData.metadata['Sensor acquisition year']
                dys = imageData.metadata['Sensor acquisition day of year']
                msecs = imageData.metadata['Sensor acquisition milliseconds of day'] + imageData.metadata['Acquisition time bias in ms']
                self.sensingStop = datetime.datetime(yr,1,1) + datetime.timedelta(days=(dys-1)) + datetime.timedelta(seconds=msecs*1e-3)

            IQLine = np.fromfile(fp, dtype='>i1', count=2*dataLen)
            trailer = np.fromfile(fp, dtype='>i1', count=2)

            self.writeRawData(output, IQLine)


        ###Read the replica and write it to file
        imageData.parseFast()
        chirpLength = imageData.metadata['Actual count of data pixels']

        ###Rewind to skip missing aux
        fp.seek(-192, os.SEEK_CUR)
        IQLine = np.fromfile(fp, dtype='>i1', count=2*chirpLength)
        IQLine.astype(np.float32).tofile('replica.bin')
#        pprint.pprint(imageData.metadata) 
        
        self.chirpLength = imageData.metadata['Pulse length in ns'] * 1.0e-9
    

        self.width = dataLen



    def _calculateRawDimensions(self,fp):
        """
            Run through the data file once, and calculate the valid sampling window start time range.
        """
        self.length = self.imageFDR.metadata['Number of SAR DATA records']
        self.width = self.imageFDR.metadata['SAR DATA record length']

        print(self.length, self.width)
        return None
