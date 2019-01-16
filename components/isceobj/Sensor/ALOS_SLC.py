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
import datetime
import isceobj.Sensor.CEOS as CEOS
import logging
from isceobj.Scene.Frame import Frame
from isceobj.Orbit.Orbit import StateVector
from isceobj.Planet.AstronomicalHandbook import Const
from isceobj.Planet.Planet import Planet
from iscesys.Component.Component import Component
from isceobj.Sensor import xmlPrefix
import numpy as np
from isceobj.Sensor.Polarimetry import Distortion


LEADERFILE = Component.Parameter(
    '_leaderFile',
    public_name='leaderfile',
    default=None,
    type=str,
    mandatory=False,
    doc='Radar wavelength in meters.'
)

IMAGEFILE = Component.Parameter(
    '_imageFile',
    public_name='imagefile',
    default=None,
    type=str,
    mandatory=False,
    doc='Radar wavelength in meters.'
)

WAVELENGTH = Component.Parameter(
    'wavelength',
    public_name='radar wavelength',
    default=None,
    type=float,
    mandatory=False,
    doc='Radar wavelength in meters.'
)

from .Sensor import Sensor


class ALOS_SLC(Sensor):
    """
        Code to read CEOSFormat leader files for ALOS SLC data.
    """

    parameter_list = (WAVELENGTH,
                      LEADERFILE,
                      IMAGEFILE) + Sensor.parameter_list
    family = 'alos_slc'
    logging_name = 'isce.sensor.ALOS_SLC'

    #Orbital Elements (Quality) Designator
    #ALOS-2/PALSAR-2 Level 1.1/1.5/2.1/3.1 CEOS SAR Product Format Description
    #PALSAR-2_xx_Format_CEOS_E_r.pdf
    orbitElementsDesignator = {'0': 'preliminary',
                               '1': 'decision',
                               '2': 'high precision'}

    def __init__(self, name=''):
        super().__init__(family=self.__class__.family, name=name)
        self.imageFile = None
        self.leaderFile = None

        # Specific doppler functions for ALOS
        self.doppler_coeff = None
        self.azfmrate_coeff = None
        self.lineDirection = None
        self.pixelDirection = None

        self.frame = Frame()
        self.frame.configure()

        self.constants = {'antennaLength': 15}

    def getFrame(self):
        return self.frame

    def parse(self):
        self.leaderFile = LeaderFile(self, file=self._leaderFile)
        self.leaderFile.parse()

        self.imageFile = ImageFile(self, file=self._imageFile)
        self.imageFile.parse()
        self.populateMetadata()
        self._populateExtras()

    def populateMetadata(self):
        """
            Create the appropriate metadata objects from our CEOSFormat metadata
        """
        frame = self._decodeSceneReferenceNumber(self.leaderFile.sceneHeaderRecord.metadata['Scene reference number'])

        fsamplookup = self.leaderFile.sceneHeaderRecord.metadata['Range sampling rate in MHz']*1.0e6

        rangePixelSize = Const.c/(2*fsamplookup)

        ins = self.frame.getInstrument()
        platform = ins.getPlatform()
        platform.setMission(self.leaderFile.sceneHeaderRecord.metadata['Sensor platform mission identifier'])
        platform.setAntennaLength(self.constants['antennaLength'])
        platform.setPointingDirection(1)
        platform.setPlanet(Planet(pname='Earth'))

        if self.wavelength:
            ins.setRadarWavelength(float(self.wavelength))
        else:
            ins.setRadarWavelength(self.leaderFile.sceneHeaderRecord.metadata['Radar wavelength'])

        ins.setIncidenceAngle(self.leaderFile.sceneHeaderRecord.metadata['Incidence angle at scene centre'])
        self.frame.getInstrument().setPulseRepetitionFrequency(self.leaderFile.sceneHeaderRecord.metadata['Pulse Repetition Frequency in mHz']*1.0e-3)
        ins.setRangePixelSize(rangePixelSize)
        ins.setRangeSamplingRate(fsamplookup)
        ins.setPulseLength(self.leaderFile.sceneHeaderRecord.metadata['Range pulse length in microsec']*1.0e-6)
        chirpSlope = self.leaderFile.sceneHeaderRecord.metadata['Nominal range pulse (chirp) amplitude coefficient linear term']
        chirpPulseBandwidth = abs(chirpSlope * self.leaderFile.sceneHeaderRecord.metadata['Range pulse length in microsec']*1.0e-6)
        ins.setChirpSlope(chirpSlope)
        ins.setInPhaseValue(7.5)
        ins.setQuadratureValue(7.5)

        self.lineDirection = self.leaderFile.sceneHeaderRecord.metadata['Time direction indicator along line direction'].strip()
        self.pixelDirection =  self.leaderFile.sceneHeaderRecord.metadata['Time direction indicator along pixel direction'].strip()

        ######ALOS includes this information in clock angle
        clockAngle = self.leaderFile.sceneHeaderRecord.metadata['Sensor clock angle']
        if clockAngle == 90.0:
            platform.setPointingDirection(-1)
        elif clockAngle == -90.0:
            platform.setPointingDirection(1)
        else:
            raise Exception('Unknown look side. Clock Angle = {0}'.format(clockAngle))

        self.frame.setFrameNumber(frame)
        self.frame.setOrbitNumber(self.leaderFile.sceneHeaderRecord.metadata['Orbit number'])
        self.frame.setProcessingFacility(self.leaderFile.sceneHeaderRecord.metadata['Processing facility identifier'])
        self.frame.setProcessingSystem(self.leaderFile.sceneHeaderRecord.metadata['Processing system identifier'])
        self.frame.setProcessingSoftwareVersion(self.leaderFile.sceneHeaderRecord.metadata['Processing version identifier'])
        self.frame.setNumberOfLines(self.imageFile.imageFDR.metadata['Number of lines per data set'])
        self.frame.setNumberOfSamples(self.imageFile.imageFDR.metadata['Number of pixels per line per SAR channel'])
        self.frame.instrument.setAzimuthPixelSize(self.leaderFile.dataQualitySummaryRecord.metadata['Azimuth resolution'])
        
        ######
        orb = self.frame.getOrbit()

        orb.setOrbitSource('Header')
        orb.setOrbitQuality(
            self.orbitElementsDesignator[
                self.leaderFile.platformPositionRecord.metadata['Orbital elements designator']
            ]
        )
        t0 = datetime.datetime(year=self.leaderFile.platformPositionRecord.metadata['Year of data point'],
                               month=self.leaderFile.platformPositionRecord.metadata['Month of data point'],
                               day=self.leaderFile.platformPositionRecord.metadata['Day of data point'])
        t0 = t0 + datetime.timedelta(seconds=self.leaderFile.platformPositionRecord.metadata['Seconds of day'])

        #####Read in orbit in inertial coordinates
        deltaT = self.leaderFile.platformPositionRecord.metadata['Time interval between data points']
        numPts = self.leaderFile.platformPositionRecord.metadata['Number of data points']


        orb = self.frame.getOrbit()
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


        self.doppler_coeff = [self.leaderFile.sceneHeaderRecord.metadata['Cross track Doppler frequency centroid constant term'],
        self.leaderFile.sceneHeaderRecord.metadata['Cross track Doppler frequency centroid linear term'],
        self.leaderFile.sceneHeaderRecord.metadata['Cross track Doppler frequency centroid quadratic term']]


        self.azfmrate_coeff =  [self.leaderFile.sceneHeaderRecord.metadata['Cross track Doppler frequency rate constant term'],
        self.leaderFile.sceneHeaderRecord.metadata['Cross track Doppler frequency rate linear term'],
        self.leaderFile.sceneHeaderRecord.metadata['Cross track Doppler frequency rate quadratic term']]

    def _populateExtras(self):
        dataset = self.leaderFile.radiometricRecord.metadata
        print("Record Number: %d" % (dataset["Record Number"]))
        print("First Record Subtype: %d" % (dataset["First Record Subtype"]))
        print("Record Type Code: %d" % (dataset["Record Type Code"]))
        print("Second Record Subtype: %d" % (dataset["Second Record Subtype"]))
        print("Third Record Subtype: %d" % (dataset["Third Record Subtype"]))
        print("Record Length: %d" % (dataset["Record Length"]))
        print("SAR channel indicator: %d" % (dataset["SAR channel indicator"]))
        print("Number of data sets: %d" % (dataset["Number of data sets"]))
        numPts = dataset['Number of data sets']
        for i in range(numPts):
            if i > 1:
                break
            print('Radiometric record field: %d' % (i+1))
            dataset = self.leaderFile.radiometricRecord.metadata[
                'Radiometric data sets'][i]
            DT11 = complex(dataset['Real part of DT 1,1'],
                           dataset['Imaginary part of DT 1,1'])
            DT12 = complex(dataset['Real part of DT 1,2'],
                           dataset['Imaginary part of DT 1,2'])
            DT21 = complex(dataset['Real part of DT 2,1'],
                           dataset['Imaginary part of DT 2,1'])
            DT22 = complex(dataset['Real part of DT 2,2'],
                           dataset['Imaginary part of DT 2,2'])
            DR11 = complex(dataset['Real part of DR 1,1'],
                           dataset['Imaginary part of DR 1,1'])
            DR12 = complex(dataset['Real part of DR 1,2'],
                           dataset['Imaginary part of DR 1,2'])
            DR21 = complex(dataset['Real part of DR 2,1'],
                           dataset['Imaginary part of DR 2,1'])
            DR22 = complex(dataset['Real part of DR 2,2'],
                           dataset['Imaginary part of DR 2,2'])
            print("Calibration factor [dB]: %f" %
                  (dataset["Calibration factor"]))
            print('Distortion matrix Trasmission [DT11, DT12, DT21, DT22]: '
                  '[%s, %s, %s, %s]' %
                  (str(DT11), str(DT12), str(DT21), str(DT22)))
            print('Distortion matrix Reception [DR11, DR12, DR21, DR22]: '
                  '[%s, %s, %s, %s]' %
                  (str(DR11), str(DR12), str(DR21), str(DR22)))
            self.transmit = Distortion(DT12, DT21, DT22)
            self.receive = Distortion(DR12, DR21, DR22)
            self.calibrationFactor = float(
                dataset['Calibration factor'])

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

        self.frame.setSensingStart(self.imageFile.sensingStart)
        self.frame.setSensingStop(self.imageFile.sensingStop)
        sensingMid = self.imageFile.sensingStart + datetime.timedelta(seconds = 0.5* (self.imageFile.sensingStop - self.imageFile.sensingStart).total_seconds())
        self.frame.setSensingMid(sensingMid)

        try:
            rngGate=  Const.c*0.5*self.leaderFile.sceneHeaderRecord.metadata['Range gate delay in microsec']*1e-6
        except:
            rngGate = None

        if (rngGate is None) or (rngGate == 0.0):
            rngGate = self.imageFile.nearRange

        self.frame.setStartingRange(rngGate)

        self.frame.getInstrument().setPulseRepetitionFrequency(self.imageFile.prf)

        pixelSize = self.frame.getInstrument().getRangePixelSize()
        farRange = self.imageFile.nearRange + (pixelSize-1) * self.imageFile.width
        self.frame.setFarRange(farRange)
        self.frame.setPolarization(self.imageFile.current_polarization)
        
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
        midwidth = self.frame.getNumberOfSamples() / 2.0
        dop = 0.0
        prod = 1.0
        for ind, kk in enumerate(self.doppler_coeff):
            dop += kk * prod
            prod *= midwidth

        print ('Average Doppler: {0}'.format(dop))

        ####For insarApp
        quadratic = {}
        quadratic['a'] = dop / self.frame.getInstrument().getPulseRepetitionFrequency()
        quadratic['b'] = 0.
        quadratic['c'] = 0.


        ####For roiApp
        ####More accurate
        ####CEOS already provides function vs pixel
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
        self.leaderFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos_slc/leader_file.xml'),dataFile=fp)
        self.leaderFDR.parse()
        fp.seek(self.leaderFDR.getEndOfRecordPosition())

        # Scene Header
        self.sceneHeaderRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos_slc/scene_record.xml'),dataFile=fp)
        self.sceneHeaderRecord.parse()
        fp.seek(self.sceneHeaderRecord.getEndOfRecordPosition())

        # Platform Position
        self.platformPositionRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos_slc/platform_position_record.xml'),dataFile=fp)
        self.platformPositionRecord.parse()
        fp.seek(self.platformPositionRecord.getEndOfRecordPosition())

        # Spacecraft Attitude
        # if (self.leaderFDR.metadata['Number of attitude data records'] == 1):
        # self.platformAttitudeRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/attitude_record.xml'),dataFile=fp)
        # self.platformAttitudeRecord.parse()
        # fp.seek(self.platformAttitudeRecord.getEndOfRecordPosition())

        # Radiometric Record
        fp.seek(8192, 1)
        self.radiometricRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix, 'alos_slc/radiometric_record.xml'), dataFile=fp)
        self.radiometricRecord.parse()
        fp.seek(self.radiometricRecord.getEndOfRecordPosition())

        # Data Quality Summary Record
        self.dataQualitySummaryRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix, 'alos_slc/data_quality_summary_record.xml'), dataFile=fp)
        self.dataQualitySummaryRecord.parse()
        fp.seek(self.dataQualitySummaryRecord.getEndOfRecordPosition())
                                  
        # 1 File descriptor         720
        # 2 Data set summary        4096
        # 3 Map projection data     1620
        # 4 Platform position data  4680
        # 5 Attitude data           8192
        # 6 Radiometric data        9860
        # 7 Data quality summary    1620
        # 8 Calibration data        13212
        # 9 Facility related        Variable
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

        volumeFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos_slc/volume_descriptor.xml'),dataFile=fp)
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
        self.prf = None
        self.image_record = os.path.join(xmlPrefix,'alos_slc/image_record.xml')
        self.logger = logging.getLogger('isce.sensor.alos')
        self.current_polarization = None
        
    def parse(self):
        try:
            fp = open(self.file,'rb')
        except IOError as errs:
            errno,strerr = errs
            print("IOError: %s" % strerr)
            return

        self.imageFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos_slc/image_file.xml'), dataFile=fp)
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
        data.tofile(fp)

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

        fp.seek(self.imageFDR.getEndOfRecordPosition(), os.SEEK_SET)

        # Extract the I and Q channels
        imageData = CEOS.CEOSDB(xml=self.image_record, dataFile=fp)
        dataLen = self.imageFDR.metadata['Number of pixels per line per SAR channel']
        prf = self.parent.leaderFile.sceneHeaderRecord.metadata['Pulse Repetition Frequency in mHz']*1.0e-3
        
        for line in range(self.length):
            if ((line % 1000) == 0):
                self.logger.debug("Extracting line %s" % line)

            a = fp.tell()
            imageData.parseFast()
            msecs = imageData.metadata['Sensor acquisition milliseconds of day']

            if line==0:
                yr = imageData.metadata['Sensor acquisition year']
                dys = imageData.metadata['Sensor acquisition day of year']
                msecs = imageData.metadata['Sensor acquisition milliseconds of day']
                self.sensingStart = datetime.datetime(yr,1,1) + datetime.timedelta(days=(dys-1)) + datetime.timedelta(seconds = msecs*1e-3)
                self.nearRange = imageData.metadata['Slant range to 1st data sample']

            if line == (self.length-1):
                yr = imageData.metadata['Sensor acquisition year']
                dys = imageData.metadata['Sensor acquisition day of year']
                msecs = imageData.metadata['Sensor acquisition milliseconds of day']

            IQLine = np.fromfile(fp, dtype='>f', count=2*dataLen)
            self.writeRawData(output, IQLine)

        self.width = dataLen
        self.prf = prf
        self.sensingStop = self.sensingStart + datetime.timedelta(seconds=(self.length-1)/self.prf)
        transmitted_polarization_bool = imageData.metadata['Transmitted polarization']
        received_polarization_bool = imageData.metadata['Received polarization']
        transmitted_polarization = 'V' if transmitted_polarization_bool else 'H'
        received_polarization = 'V' if received_polarization_bool else 'H'
        self.current_polarization = transmitted_polarization + received_polarization
        
    def _calculateRawDimensions(self, fp):
        """
            Run through the data file once, and calculate the valid sampling window start time range.
        """
        self.length = self.imageFDR.metadata['Number of SAR DATA records']
        self.width = self.imageFDR.metadata['SAR DATA record length']

        return None
