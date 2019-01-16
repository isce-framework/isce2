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


DATATYPE = Component.Parameter(
        '_dataType',
        public_name='DATATYPE',
        default='short',
        type=str,
        mandatory=False,
        doc='short or float')

from .Sensor import Sensor

class Risat1_SLC(Sensor):
    """
        Code to read CEOSFormat leader files for Risat-1 SAR data.
    """
    family = "risat1"
    logging_name = 'isce.sensor.Risat1'
    parameter_list = (IMAGEFILE, LEADERFILE, METAFILE, DATATYPE) + Sensor.parameter_list

    @logged
    def __init__(self, name=''):
        super().__init__(family=self.__class__.family, name=name)
        self.imageFile = None
        self.leaderFile = None

        #####Specific doppler functions for RISAT1
        self.doppler_coeff = None
        self.azfmrate_coeff = None
        self.lineDirection = None
        self.pixelDirection = None

        self.frame = Frame()
        self.frame.configure()

        self.constants = {
                            'antennaLength': 6,
                         }

        self.TxPolMap = { 
                                1 : 'V',
                                2 : 'H',
                                3 : 'L',
                                4 : 'R',
                            }

        self.RxPolMap = {
                                1 : 'V',
                                2 : 'H',
                        }


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


        print('Average terrain height: ', 1000*self.leaderFile.sceneHeaderRecord.metadata['Average terrain height in km'])

        ins = self.frame.getInstrument()
        platform = ins.getPlatform()
        platform.setMission(self.leaderFile.sceneHeaderRecord.metadata['Sensor platform mission identifier'])
        platform.setAntennaLength(self.constants['antennaLength'])
        platform.setPlanet(Planet(pname='Earth'))

        ins.setRadarWavelength(self.leaderFile.sceneHeaderRecord.metadata['Radar wavelength'])
        ins.setIncidenceAngle(self.leaderFile.sceneHeaderRecord.metadata['Incidence angle at scene centre'])
        self.frame.getInstrument().setPulseRepetitionFrequency(self.leaderFile.sceneHeaderRecord.metadata['Pulse Repetition Frequency'])
        ins.setRangePixelSize(rangePixelSize)
        ins.setRangeSamplingRate(self.leaderFile.sceneHeaderRecord.metadata['Range sampling rate'])
        ins.setPulseLength(self.leaderFile.sceneHeaderRecord.metadata['Range pulse length'])

        chirpPulseBandwidth = self.leaderFile.processingRecord.metadata['Pulse bandwidth code']*1e4
        ins.setChirpSlope(chirpPulseBandwidth/self.leaderFile.sceneHeaderRecord.metadata['Range pulse length'])
        ins.setInPhaseValue(0.0)
        ins.setQuadratureValue(0.0)

        self.lineDirection = self.leaderFile.sceneHeaderRecord.metadata['Time direction indicator along line direction'].strip()
        self.pixelDirection =  self.leaderFile.sceneHeaderRecord.metadata['Time direction indicator along pixel direction'].strip()

        ######RISAT-1 sensor orientation convention is opposite to ours
        lookSide = self.leaderFile.processingRecord.metadata['Sensor orientation']
        if lookSide == 'RIGHT':
            platform.setPointingDirection(1)
        elif lookSide == 'LEFT':
            platform.setPointingDirection(-1)
        else:
            raise Exception('Unknown look side')

        print('Leader file look side: ', lookSide)

        self.frame.setFrameNumber(frame)
        self.frame.setOrbitNumber(self.leaderFile.sceneHeaderRecord.metadata['Orbit number'])
        self.frame.setProcessingFacility(self.leaderFile.sceneHeaderRecord.metadata['Processing facility identifier'])
        self.frame.setProcessingSystem(self.leaderFile.sceneHeaderRecord.metadata['Processing system identifier'])
        self.frame.setProcessingSoftwareVersion(self.leaderFile.sceneHeaderRecord.metadata['Processing version identifier'])
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

        #####Convert orbits from ECI to ECR frame
        t0 = orb._stateVectors[0]._time
        ang = self.leaderFile.platformPositionRecord.metadata['Greenwich mean hour angle']

        cOrb = ECI2ECR(orb, GAST=ang, epoch=t0)
        iOrb = cOrb.convert()


        #####Extend the orbits by a few points
        #####Expect large azimuth shifts - absolutely needed
        #####Since CEOS contains state vectors that barely covers scene extent
        planet = self.frame.instrument.platform.planet
        orbExt = OrbitExtender()
        orbExt.configure()
        orbExt._newPoints = 4
        newOrb = orbExt.extendOrbit(iOrb)

        orb = self.frame.getOrbit()

        for sv in newOrb:
            orb.addStateVector(sv)

        
        self.doppler_coeff = [self.leaderFile.sceneHeaderRecord.metadata['Cross track Doppler frequency centroid constant term'],
        self.leaderFile.sceneHeaderRecord.metadata['Cross track Doppler frequency centroid linear term'],
        self.leaderFile.sceneHeaderRecord.metadata['Cross track Doppler frequency centroid quadratic term']]


        self.azfmrate_coeff =  [self.leaderFile.sceneHeaderRecord.metadata['Cross track Doppler frequency rate constant term'],
        self.leaderFile.sceneHeaderRecord.metadata['Cross track Doppler frequency rate linear term'],
        self.leaderFile.sceneHeaderRecord.metadata['Cross track Doppler frequency rate quadratic term']]


    def extractImage(self):
        import isceobj
        if (self.imageFile is None) or (self.leaderFile is None):
            self.parse()

        try:
            out = open(self.output, 'wb')
        except IOError as strerr:
            self.logger.error("IOError: %s" % strerr)

        self.imageFile.extractImage(output=out, dtype=self._dataType)
        out.close()

        self.frame.setSensingStart(self.imageFile.sensingStart )
        self.frame.setSensingStop(self.imageFile.sensingStop)
        sensingMid = self.imageFile.sensingStart + datetime.timedelta(seconds = 0.5* (self.imageFile.sensingStop - self.imageFile.sensingStart).total_seconds())
        self.frame.setSensingMid(sensingMid)

        self.frame.setStartingRange(self.imageFile.nearRange)
        self.frame.setFarRange(self.imageFile.farRange)
#        self.doppler_coeff = self.imageFile.dopplerCoeff
        self.frame.getInstrument().setPulseRepetitionFrequency(self.imageFile.prf)



        pol = self.TxPolMap[int(self.imageFile.polarization[0])] +  self.TxPolMap[int(self.imageFile.polarization[1])]
        self.frame.setPolarization(pol)


        rawImage = isceobj.createSlcImage()
        rawImage.setByteOrder('l')
        rawImage.setAccessMode('read')
        rawImage.setFilename(self.output)
        rawImage.setWidth(self.imageFile.width)
        rawImage.setXmin(0)
        rawImage.setXmax(self.imageFile.width)
        rawImage.renderHdr()
        self.frame.setImage(rawImage)

        return


    def extractDoppler(self):
        '''
        Evaluate the doppler polynomial and return the average value for now.
        '''

        ####For insarApp

        quadratic = {}
        quadratic['a'] = self.doppler_coeff[0] / self.frame.getInstrument().getPulseRepetitionFrequency()
        quadratic['b'] = 0.
        quadratic['c'] = 0.


        ###For roiApp
        ###More accurate
        self.frame._dopplerVsPixel = self.doppler_coeff

        return quadratic

    def _decodeSceneReferenceNumber(self,referenceNumber):
        return referenceNumber



class LeaderFile(object):

    def __init__(self, parent, file=None):
        self.parent = parent
        self.file = file
        self.leaderFDR = None
        self.sceneHeaderRecord = None
        self.processingRecord = None
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
        self.leaderFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'risat_slc/leader_file.xml'),dataFile=fp)
        self.leaderFDR.parse()
        fp.seek(self.leaderFDR.getEndOfRecordPosition())

        # Scene Header
        self.sceneHeaderRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'risat_slc/scene_record.xml'),dataFile=fp)
        self.sceneHeaderRecord.parse()
        fp.seek(self.sceneHeaderRecord.getEndOfRecordPosition())

        #Data quality summary
        qualityRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'risat_slc/data_quality_summary_record.xml'), dataFile=fp)
        qualityRecord.parse()
        fp.seek(qualityRecord.getEndOfRecordPosition())

        #Data histogram records
        for ind in range(self.leaderFDR.metadata['Number of data histograms records']):
            histRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix, 'risat_slc/data_histogram_record.xml'), dataFile=fp)
            histRecord.parse()
            fp.seek(histRecord.getEndOfRecordPosition())

        self.processingRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix, 'risat_slc/detailed_processing_record.xml'), dataFile=fp)
        self.processingRecord.parse()
        fp.seek(self.processingRecord.getEndOfRecordPosition())


        # Platform Position
        self.platformPositionRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'risat_slc/platform_position_record.xml'),dataFile=fp)
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

        volumeFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'risat_slc/volume_descriptor.xml'),dataFile=fp)
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
        self.polarization = None
        self.dopplerCoeff = None
        self.image_record = os.path.join(xmlPrefix,'risat_slc/image_record.xml')
        self.logger = logging.getLogger('isce.sensor.risat')

    def parse(self):
        try:
            fp = open(self.file,'rb')
        except IOError as errs:
            errno,strerr = errs
            print("IOError: %s" % strerr)
            return

        self.imageFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'risat_slc/image_file.xml'), dataFile=fp)
        self.imageFDR.parse()
        fp.seek(self.imageFDR.getEndOfRecordPosition())
        self._calculateRawDimensions(fp)

        fp.close()

    def writeRawData(self, fp, line):
        '''
        Convert complex integer to complex64 format.
        '''
        cJ = np.complex64(1j)
        data = line[0::2] + cJ * line[1::2]
        data.astype(np.complex64).tofile(fp)


    def extractImage(self, output=None, dtype='short'):
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

        prf = self.parent.leaderFile.sceneHeaderRecord.metadata['Pulse Repetition Frequency']
        # Extract the I and Q channels
        imageData = CEOS.CEOSDB(xml=self.image_record,dataFile=fp)
        dataLen =  self.imageFDR.metadata['Number of pixels per line per SAR channel']


        for line in range(self.length):
            if ((line%1000) == 0):
                self.logger.debug("Extracting line %s" % line)

            imageData.parseFast()

            if line==0:
                yr = imageData.metadata['Sensor acquisition year']
                dys = imageData.metadata['Sensor acquisition day of year']
                msecs = imageData.metadata['Sensor acquisition milliseconds of day'] + imageData.metadata['Acquisition time bias in ms']
                self.sensingStart = datetime.datetime(yr,1,1) + datetime.timedelta(days=(dys-1)) + datetime.timedelta(seconds = msecs*1e-3)
                self.nearRange = imageData.metadata['Slant range to 1st pixel']
                self.farRange = imageData.metadata['Slant range to last pixel']
                self.dopplerCoeff = [ imageData.metadata['First pixel Doppler centroid'],
                                      imageData.metadata['Mid-pixel Doppler centroid'],
                                      imageData.metadata['Last pixel Doppler centroid'] ]
                self.prf = imageData.metadata['PRF']

                self.polarization = (imageData.metadata['Transmitted polarization'], imageData.metadata['Received polarization'])

            if line==(self.length-1):
                yr = imageData.metadata['Sensor acquisition year']
                dys = imageData.metadata['Sensor acquisition day of year']
                msecs = imageData.metadata['Sensor acquisition milliseconds of day'] + imageData.metadata['Acquisition time bias in ms']
                self.sensingStop = datetime.datetime(yr,1,1) + datetime.timedelta(days=(dys-1)) + datetime.timedelta(seconds=msecs*1e-3)


            if dtype=='short':
                IQLine = np.fromfile(fp, dtype='>i2', count=2*dataLen)
            else:
                IQLine = np.fromfile(fp, dtype='>i4', count=2*dataLen)
            self.writeRawData(output, IQLine)

        self.width = dataLen



    def _calculateRawDimensions(self,fp):
        """
            Run through the data file once, and calculate the valid sampling window start time range.
        """
        self.length = self.imageFDR.metadata['Number of SAR DATA records']
        self.width = self.imageFDR.metadata['SAR DATA record length']

        return None
