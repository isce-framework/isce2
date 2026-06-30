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
#
# Modified: Francisco Delgado 2025/08/19. for stripmap data. Key differences with ALOS-2
# 1. skip the 720-byte preamble
# 2. read float32 (big-endian) I/Q
# 3. used header=720 / trailer=80 per line
# 4. converted near range in cm to m
# 5. frame number properly read from metadata (not used in processing)
# Not tested with ScanSAR data (must check if time tag PRF is ok), so that whole 
# section is commented
# also modified __init__.py and components/isceobj/StripmapProc/Factories.py
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




import os
import datetime
import isceobj.Sensor.CEOS as CEOS
import logging
from isceobj.Scene.Frame import Frame
from isceobj.Orbit.Orbit import StateVector,Orbit
from isceobj.Planet.AstronomicalHandbook import Const
from isceobj.Planet.Planet import Planet
from iscesys.Component.Component import Component
from isceobj.Sensor import xmlPrefix
from isceobj.Util import Polynomial
from iscesys.DateTimeUtil import secondsSinceMidnight
import numpy as np
import struct
import pprint

#Sometimes the wavelength in the meta data is not correct.
#If the user sets this parameter, then the value in the
#meta data file is ignored.
WAVELENGTH = Component.Parameter(
    'wavelength',
    public_name='radar wavelength',
    default=None,
    type=float,
    mandatory=False,
    doc='Radar wavelength in meters.'
)

LEADERFILE = Component.Parameter(
    '_leaderFile',
    public_name='leaderfile',
    default=None,
    type=str,
    mandatory=True,
    doc='Name of the leaderfile.'
)

IMAGEFILE = Component.Parameter(
    '_imageFile',
    public_name='imagefile',
    default=None,
    type=str,
    mandatory=True,
    doc='Name of the imagefile.'
)

from .Sensor import Sensor

class ALOS4(Sensor):
    """
        Code to read CEOSFormat leader files for ALOS2 SLC data.
    """

    family = 'alos4'

    parameter_list = (WAVELENGTH,
                      LEADERFILE,
                      IMAGEFILE) + Sensor.parameter_list

    fsampConst = { 98: 9.824218687500000E+07,
                   49: 4.912109343750000E+07,
                   32: 3.274739562500000E+07,
                   16: 1.637369781250000E+07 } 

    #Orbital Elements (Quality) Designator
    #ALOS-4 PALSAR-3 Standard Product Format Description (CEOS format)
    #FTR-240031A_ALOS-4_PALSAR-3_StandardProduct_Format_CEOS_En.pdf
    orbitElementsDesignator = {'0':'preliminary',
                               '1':'decision',
                               '2':'high precision'}

    def __init__(self, name=''):
        super().__init__(family=self.__class__.family, name=name)
        self.leaderFile = None
        self.imageFile = None

        #####Specific doppler functions for ALOS2
        self.doppler_coeff = None
        self.azfmrate_coeff = None
        self.lineDirection = None
        self.pixelDirection = None

        self.frame = Frame()
        self.frame.configure()

        self.constants = {'polarization': 'HH',
                          'antennaLength': 10}


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
        frame = self._decodeSceneReferenceNumber(self.leaderFile.sceneHeaderRecord.metadata['Scene id'])
        frame = frame[8:12] #this is correct
        #print("\n".join(f"{k}: {v}" for k, v in self.leaderFile.sceneHeaderRecord.metadata.items()))
        fsamplookup = int(self.leaderFile.sceneHeaderRecord.metadata['Range sampling rate in MHz'])

        rangePixelSize = Const.c/(2*self.fsampConst[fsamplookup])

        ins = self.frame.getInstrument()
        #print(ins)
        platform = ins.getPlatform()
        platform.setMission(self.leaderFile.sceneHeaderRecord.metadata['Sensor platform mission identifier'])
        platform.setAntennaLength(self.constants['antennaLength'])
        platform.setPointingDirection(1)
        platform.setPlanet(Planet(pname='Earth'))

        if self.wavelength:
            ins.setRadarWavelength(float(self.wavelength))
#            print('ins.radarWavelength = ', ins.getRadarWavelength(),
#                  type(ins.getRadarWavelength()))
        else:
            ins.setRadarWavelength(self.leaderFile.sceneHeaderRecord.metadata['Radar wavelength'])

        ins.setIncidenceAngle(self.leaderFile.sceneHeaderRecord.metadata['Incidence angle at scene centre'])
        self.frame.getInstrument().setPulseRepetitionFrequency(self.leaderFile.sceneHeaderRecord.metadata['Pulse Repetition Frequency in mHz']*1.0e-3)
        ins.setRangePixelSize(rangePixelSize)
        ins.setRangeSamplingRate(self.fsampConst[fsamplookup])
        ins.setPulseLength(self.leaderFile.sceneHeaderRecord.metadata['Range pulse length in microsec']*1.0e-6)
        chirpSlope = self.leaderFile.sceneHeaderRecord.metadata['Nominal range pulse (chirp) amplitude coefficient linear term']
        chirpPulseBandwidth = abs(chirpSlope * self.leaderFile.sceneHeaderRecord.metadata['Range pulse length in microsec']*1.0e-6)
        ins.setChirpSlope(chirpSlope)
        ins.setInPhaseValue(7.5)
        ins.setQuadratureValue(7.5)

        self.lineDirection = self.leaderFile.sceneHeaderRecord.metadata['Time direction indicator along line direction'].strip()
        self.pixelDirection =  self.leaderFile.sceneHeaderRecord.metadata['Time direction indicator along pixel direction'].strip()

        ######ALOS2 includes this information in clock angle
        clockAngle = self.leaderFile.sceneHeaderRecord.metadata['Sensor clock angle']
        if clockAngle == 90.0:
            platform.setPointingDirection(-1)
        elif clockAngle == -90.0:
            platform.setPointingDirection(1)
        else:
            raise Exception('Unknown look side. Clock Angle = {0}'.format(clockAngle))

#        print(self.leaderFile.sceneHeaderRecord.metadata["Sensor ID and mode of operation for this channel"])
        self.frame.setFrameNumber(frame)
        self.frame.setOrbitNumber(self.leaderFile.sceneHeaderRecord.metadata['Orbit number'])
        self.frame.setProcessingFacility(self.leaderFile.sceneHeaderRecord.metadata['Processing facility identifier'])
        self.frame.setProcessingSystem(self.leaderFile.sceneHeaderRecord.metadata['Processing system identifier'])
        self.frame.setProcessingSoftwareVersion(self.leaderFile.sceneHeaderRecord.metadata['Processing version identifier'])
        self.frame.setPolarization(self.constants['polarization'])
        self.frame.setNumberOfLines(self.imageFile.imageFDR.metadata['Number of lines per data set'])
        self.frame.setNumberOfSamples(self.imageFile.imageFDR.metadata['Number of pixels per line per SAR channel'])

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

        #print(vars(self.leaderFile.leaderFDR))
        #print(vars(self.leaderFile.sceneHeaderRecord))
        #print(vars(self.leaderFile.platformPositionRecord))
        #print(vars(self.leaderFile.facilityRecord))

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

#        print('Terrain height: ', self.leaderFile.sceneHeaderRecord.metadata['Average terrain ellipsoid height'])


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

#        rangeGate = self.leaderFile.sceneHeaderRecord.metadata['Range gate delay in microsec']*1e-6
#        delt = datetime.timedelta(seconds=rangeGate)

        delt = datetime.timedelta(seconds=0.0)
        self.frame.setSensingStart(self.imageFile.sensingStart +delt )
        self.frame.setSensingStop(self.imageFile.sensingStop + delt)
        sensingMid = self.imageFile.sensingStart + datetime.timedelta(seconds = 0.5* (self.imageFile.sensingStop - self.imageFile.sensingStart).total_seconds()) + delt
        self.frame.setSensingMid(sensingMid)

        self.frame.setStartingRange(self.imageFile.nearRange)

        self.frame.getInstrument().setPulseRepetitionFrequency(self.imageFile.prf)

        pixelSize = self.frame.getInstrument().getRangePixelSize()
        farRange = self.imageFile.nearRange + (pixelSize-1) * self.imageFile.width
        self.frame.setFarRange(farRange)

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
        self.leaderFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/leader_file.xml'),dataFile=fp)
        self.leaderFDR.parse()
        fp.seek(self.leaderFDR.getEndOfRecordPosition())

        # Scene Header
        self.sceneHeaderRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/scene_record.xml'),dataFile=fp)
        self.sceneHeaderRecord.parse()
        fp.seek(self.sceneHeaderRecord.getEndOfRecordPosition())

        # Platform Position
        self.platformPositionRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/platform_position_record.xml'),dataFile=fp)
        self.platformPositionRecord.parse()
        fp.seek(self.platformPositionRecord.getEndOfRecordPosition())

        #####Skip attitude information
        fp.seek(16384,1) #same alos-2/alos-4

        #####Skip radiometric information
        fp.seek(65536,1)  #9860 alos-2

        ####Skip the data quality information
        fp.seek(1620,1) #same alos-2/4


        ####Skip facility 1-4
        fp.seek(325000 + 511000 + 3072 + 728000, 1)#same alos-2/4


        ####Read facility 5
        self.facilityRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/facility_record.xml'), dataFile=fp)
        self.facilityRecord.parse()
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

        volumeFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/volume_descriptor.xml'),dataFile=fp)
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
        self.image_record = os.path.join(xmlPrefix,'alos2_slc/image_record.xml')
        self.logger = logging.getLogger('isce.sensor.alos2')

    def parse(self):
        try:
            fp = open(self.file,'rb')
        except IOError as errs:
            errno,strerr = errs
            print("IOError: %s" % strerr)
            return

        self.imageFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/image_file.xml'), dataFile=fp)
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



        fp.seek(self.imageFDR.getEndOfRecordPosition(),os.SEEK_SET)

        # Extract the I and Q channels
        imageData = CEOS.CEOSDB(xml=self.image_record,dataFile=fp)
        dataLen =  self.imageFDR.metadata['Number of pixels per line per SAR channel']
        print('Number of pixels per line per SAR channel',dataLen)
        print('IMAGE DATA',imageData)

        delta = 0.0
        prf = self.parent.leaderFile.sceneHeaderRecord.metadata['Pulse Repetition Frequency in mHz']*1.0e-3
        print('LEADERFILE  PRF: ', prf)

        for line in range(self.length):
            if ((line%1000) == 0):
                self.logger.debug("Extracting line %s" % line)
#                pprint.pprint(imageData.metadata)

            pos0 = fp.tell()

            imageData.parseFast()

            usecs = imageData.metadata['Sensor acquisition micro-seconds of day']

            if line==0:
                yr = imageData.metadata['Sensor acquisition year']
                dys = imageData.metadata['Sensor acquisition day of year']
                msecs = imageData.metadata['Sensor acquisition milliseconds of day']
                self.sensingStart = datetime.datetime(yr,1,1) + datetime.timedelta(days=(dys-1)) + datetime.timedelta(seconds = usecs*1e-6)
                self.nearRange = imageData.metadata['Slant range to 1st data sample']  * 1.0e-2 #cm to m
                prf1 = imageData.metadata['PRF'] * 1.0e-3
                #print('Slant range to 1st data sample',self.nearRange)
	
            if line==(self.length-1):
                yr = imageData.metadata['Sensor acquisition year']
                dys = imageData.metadata['Sensor acquisition day of year']
                msecs = imageData.metadata['Sensor acquisition milliseconds of day']
#                self.sensingStop = datetime.datetime(yr,1,1) + datetime.timedelta(days=(dys-1)) + datetime.timedelta(seconds=usecs*1e-6)

            if line > 0:
                delta += (usecs - prevline)

            prevline = usecs

            delta += (usecs - prevline) #adds zero to make it compatible with moving the file pointer
            fp.seek(pos0, os.SEEK_SET)  #move the pointer back to pos0
            fp.seek(720, os.SEEK_CUR)   #skip the header by moving 720 bytes forward
            IQLine = np.fromfile(fp, dtype='>f4', count=2*dataLen) #big-endian 32-bit floats
            fp.seek(80, os.SEEK_CUR)    #skip the trailer by moving 80 bytes forward

            self.writeRawData(output, IQLine)

        self.width = dataLen
        prf2 =  (self.length-1) / (delta*1.0e-6)
        print('TIME TAG PRF: ', prf2)
        print('LINE TAG PRF: ', prf1)

#        print('Using Leaderfile PRF')
#        self.prf = prf

        #choose PRF according to operation mode. Cunren Liang, 2015
        operationMode = "{}".format(self.parent.leaderFile.sceneHeaderRecord.metadata['Sensor ID and mode of operation for this channel'])
        operationMode =operationMode[10:12]
        if operationMode == '11':  #11 is ScanSAR , 8-10 are Stripmap [10 m] 
            # Operation mode
            # '01': Spotlight mode
            # '02': Stripmap [3 m] mode (200 km observation)
            # '03': Stripmap [3 m] mode (100 km observation)
            # '04': Stripmap [3 m] mode (full polarimetry observation)
            # '05': Stripmap [6 m] mode (200 km observation)
            # '06': Stripmap [6 m] mode (100 km observation)
            # '07': Stripmap [6 m] mode (full polarimetry observation) 
            # '08': Stripmap [10 m] mode (200 km observation)
            # '09': Stripmap [10 m] mode (100 km observation)
            # '10': Stripmap [10 m] mode (full polarimetry observation) 
            # '11': ScanSAR mode
            # '14': Manual observation
            print('ScanSAR nominal mode, using PRF from the line header')
            self.prf = prf1
        else:
            self.prf = prf

        ##if operationMode == '11':
        ###    #adding burst information here. Cunren, 14-DEC-2015. Not sure it works for ALOS-4 ScanSAR, so commented, F.D., 2025/08/19
        ###    sceneCenterIncidenceAngle = self.parent.leaderFile.sceneHeaderRecord.metadata['Incidence angle at scene centre']
        ###    sarChannelId = imageData.metadata['SAR channel indicator']
        ###    scanId = imageData.metadata['Scan ID'] #Scan ID starts with 1
        ###    
        ###    #if (sceneCenterIncidenceAngle > 39.032 - 5.0 and sceneCenterIncidenceAngle < 39.032 + 5.0) and (sarChannelId == 2):
        ###    if 1:
        ###        #burst parameters, currently only for the second, dual polarization, ScanSAR nominal mode 
        ###        #that is the second WBD mode.
        ###        #p.25 and p.115 of ALOS-2/PALSAR-2 Level 1.1/1.5/2.1/3.1 CEOS SAR Product Format Description
        ###        #for the definations of wide swath mode
        ###        nbraw = [358,        470,        358,        355,        487]
        ###        ncraw = [2086.26,    2597.80,    1886.18,    1779.60,    2211.17]

        ###        self.parent.frame.nbraw = nbraw[scanId-1]
        ###        self.parent.frame.ncraw = ncraw[scanId-1]

        ###        #this is the prf fraction (total azimuth bandwith) used in extracting burst.
        ###        #here the total bandwith is 0.93 * prfs[3] for all subswaths, which is the following values:
        ###        #[0.7933, 0.6371, 0.8774, 0.9300, 0.7485] 
        ###        prfs=[2661.847, 3314.512, 2406.568, 2270.575, 2821.225]
        ###        self.parent.frame.prffrac = 0.93 * prfs[3]/prfs[scanId-1]







        self.sensingStop = self.sensingStart + datetime.timedelta(seconds = (self.length-1)/self.prf)

    def _calculateRawDimensions(self,fp):
        """
            Run through the data file once, and calculate the valid sampling window start time range.
        """
        self.length = self.imageFDR.metadata['Number of SAR DATA records']
        self.width = self.imageFDR.metadata['SAR DATA record length']
        print('SAR DATA records',self.length)
        print('SAR DATA record length',self.width)

        return None
