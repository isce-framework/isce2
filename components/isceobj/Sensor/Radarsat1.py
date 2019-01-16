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
from isceobj.Orbit.Orbit import StateVector,Orbit
from isceobj.Planet.AstronomicalHandbook import Const
from isceobj.Planet.Planet import Planet
from iscesys.Component.Component import Component
from isceobj.Util.decorators import pickled, logged
from isceobj.Sensor import xmlPrefix
from isceobj.Util import Poly2D
from iscesys.DateTimeUtil import secondsSinceMidnight
import numpy as np
import struct
import pprint

LEADERFILE = Component.Parameter(
    '_leaderFile',
    public_name='LEADERFILE',
    default='',
    type=str,
    mandatory=True,
    doc='RadarSAT1 Leader file'
)

IMAGEFILE = Component.Parameter(
    '_imageFile',
    public_name='IMAGEFILE',
    default='',
    type=str,
    mandatory=True,
    doc='RadarSAT1 image file'
)

PARFILE = Component.Parameter(
    '_parFile',
    public_name='PARFILE',
    default='',
    type=str,
    mandatory=False,
    doc='RadarSAT1 par file'
)

from .Sensor import Sensor

class Radarsat1(Sensor):
    """
    Code to read CEOSFormat leader files for Radarsat-1 SAR data.
    The tables used to create this parser are based on document number
    ER-IS-EPO-GS-5902.1 from the European Space Agency.
    """
    family = 'radarsat1'
    logging_name = 'isce.sensor.radarsat1'

    parameter_list = (LEADERFILE, IMAGEFILE, PARFILE) + Sensor.parameter_list

    auxLength = 50

    @logged
    def __init__(self, name=''):
        super().__init__(family=self.__class__.family, name=name)
        self.imageFile = None
        self.leaderFile = None

        #####Soecific doppler functions for RSAT1
        self.doppler_ref_range = None
        self.doppler_ref_azi = None
        self.doppler_predict = None
        self.doppler_DAR = None
        self.doppler_coeff = None


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

        if self._parFile:
           self.parseParFile()
        else:
            self.populateCEOSOrbit()

    def populateMetadata(self):
        """
            Create the appropriate metadata objects from our CEOSFormat metadata
        """
        frame = self._decodeSceneReferenceNumber(self.leaderFile.sceneHeaderRecord.metadata['Scene reference number'])
        try:
            rangePixelSize = Const.c/(2*self.leaderFile.sceneHeaderRecord.metadata['Range sampling rate']*1e6)
        except ZeroDivisionError:
            rangePixelSize = 0

        ins = self.frame.getInstrument()
        platform = ins.getPlatform()
        platform.setMission(self.leaderFile.sceneHeaderRecord.metadata['Sensor platform mission identifier'])
        platform.setAntennaLength(self.constants['antennaLength'])
        platform.setPointingDirection(-1)
        platform.setPlanet(Planet(pname='Earth'))

        ins.setRadarWavelength(self.leaderFile.sceneHeaderRecord.metadata['Radar wavelength'])
        ins.setIncidenceAngle(self.leaderFile.sceneHeaderRecord.metadata['Incidence angle at scene centre'])
        ##RSAT-1 does not have PRF for raw data in leader file.
#        self.frame.getInstrument().setPulseRepetitionFrequency(self.leaderFile.sceneHeaderRecord.metadata['Pulse Repetition Frequency'])
        ins.setRangePixelSize(rangePixelSize)
        ins.setPulseLength(self.leaderFile.sceneHeaderRecord.metadata['Range pulse length']*1e-6)
        chirpPulseBandwidth = 15.50829e6 # Is this really not in the CEOSFormat Header?
        ins.setChirpSlope(chirpPulseBandwidth/(self.leaderFile.sceneHeaderRecord.metadata['Range pulse length']*1e-6))
        ins.setInPhaseValue(7.5)
        ins.setQuadratureValue(7.5)


        self.frame.setFrameNumber(frame)
        self.frame.setOrbitNumber(self.leaderFile.sceneHeaderRecord.metadata['Orbit number'])
        self.frame.setProcessingFacility(self.leaderFile.sceneHeaderRecord.metadata['Processing facility identifier'])
        self.frame.setProcessingSystem(self.leaderFile.sceneHeaderRecord.metadata['Processing system identifier'])
        self.frame.setProcessingSoftwareVersion(self.leaderFile.sceneHeaderRecord.metadata['Processing version identifier'])
        self.frame.setPolarization(self.constants['polarization'])
        self.frame.setNumberOfLines(self.imageFile.imageFDR.metadata['Number of lines per data set'])
        self.frame.setNumberOfSamples(self.imageFile.imageFDR.metadata['Number of pixels per line per SAR channel'])


        self.frame.getOrbit().setOrbitSource('Header')
        self.frame.getOrbit().setOrbitQuality(self.leaderFile.platformPositionRecord.metadata['Orbital elements designator'])



    def populateCEOSOrbit(self):
        from isceobj.Orbit.Inertial import ECI2ECR
        
        t0 = datetime.datetime(year=self.leaderFile.platformPositionRecord.metadata['Year of data point'],
                               month=self.leaderFile.platformPositionRecord.metadata['Month of data point'],
                               day=self.leaderFile.platformPositionRecord.metadata['Day of data point'])
        t0 = t0 + datetime.timedelta(seconds=self.leaderFile.platformPositionRecord.metadata['Seconds of day'])

        #####Read in orbit in inertial coordinates
        orb = Orbit()
        for i in range(self.leaderFile.platformPositionRecord.metadata['Number of data points']):
            vec = StateVector()
            t = t0 + datetime.timedelta(seconds=(i*self.leaderFile.platformPositionRecord.metadata['Time interval between DATA points']))
            vec.setTime(t)
            dataPoints = self.leaderFile.platformPositionRecord.metadata['Positional Data Points'][i]
            vec.setPosition([dataPoints['Position vector X'], dataPoints['Position vector Y'], dataPoints['Position vector Z']])
            vec.setVelocity([dataPoints['Velocity vector X']/1000., dataPoints['Velocity vector Y']/1000., dataPoints['Velocity vector Z']/1000.])
            orb.addStateVector(vec)

        #####Convert orbits from ECI to ECEF frame.
        t0 = orb._stateVectors[0]._time
        ang = self.leaderFile.platformPositionRecord.metadata['Greenwich mean hour angle']

        cOrb = ECI2ECR(orb, GAST=ang, epoch=t0)
        wgsorb = cOrb.convert()


        orb = self.frame.getOrbit()

        for sv in wgsorb:
            orb.addStateVector(sv)
            print(sv)



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

        ####RSAT1 is weird. Contains all useful info in RAW data and not leader.
        ins = self.frame.getInstrument()
        ins.setPulseRepetitionFrequency(self.imageFile.prf)
        ins.setPulseLength(self.imageFile.pulseLength)
        ins.setRangeSamplingRate(self.imageFile.rangeSamplingRate)
        ins.setRangePixelSize( Const.c/ (2*self.imageFile.rangeSamplingRate))
        ins.setChirpSlope(self.imageFile.chirpSlope)

        ######
        self.frame.setSensingStart(self.imageFile.sensingStart)
        sensingStop = self.imageFile.sensingStart + datetime.timedelta(seconds = ((self.frame.getNumberOfLines()-1)/self.imageFile.prf))
        sensingMid = self.imageFile.sensingStart + datetime.timedelta(seconds = 0.5* (sensingStop - self.imageFile.sensingStart).total_seconds())
        self.frame.setSensingStop(sensingStop)
        self.frame.setSensingMid(sensingMid)
        self.frame.setNumberOfSamples(self.imageFile.width)
        self.frame.setStartingRange(self.imageFile.startingRange)
        farRange = self.imageFile.startingRange + ins.getRangePixelSize() * self.imageFile.width* 0.5
        self.frame.setFarRange(farRange)

        rawImage = isceobj.createRawImage()
        rawImage.setByteOrder('l')
        rawImage.setAccessMode('read')
        rawImage.setFilename(self.output)
        rawImage.setWidth(self.imageFile.width)
        rawImage.setXmin(0)
        rawImage.setXmax(self.imageFile.width)
        rawImage.renderHdr()
        self.frame.setImage(rawImage)


    def parseParFile(self):
        '''Parse the par file if any is available.'''
        if self._parFile not in (None, ''):
            par = ParFile(self._parFile)


            ####Update orbit
            svs = par['prep_block']['sensor']['ephemeris']['sv_block']['state_vector']
            datefmt='%Y%m%d%H%M%S%f'
            for entry in svs:
                sv = StateVector()
                sv.setPosition([float(entry['x']), float(entry['y']), float(entry['z'])])
                sv.setVelocity([float(entry['xv']), float(entry['yv']), float(entry['zv'])])
                sv.setTime(datetime.datetime.strptime(entry['Date'], datefmt))
                self.frame.orbit.addStateVector(sv)

            self.frame.orbit._stateVectors = sorted(self.frame.orbit._stateVectors, key=lambda x: x.getTime())

            doppinfo = par['prep_block']['sensor']['beam']['DopplerCentroidParameters']
            #######Selectively update some values.
            #######Currently used only for doppler centroids.

            self.doppler_ref_range = float(doppinfo['reference_range'])
            self.doppler_ref_azi = datetime.datetime.strptime(doppinfo['reference_date'], '%Y%m%d%H%M%S%f')
            self.doppler_predict = float(doppinfo['Predict_doppler'])
            self.doppler_DAR = float(doppinfo['DAR_doppler'])

            coeff = doppinfo['doppler_centroid_coefficients']
            rngOrder = int(coeff['number_of_coefficients_first_dimension'])-1
            azOrder = int(coeff['number_of_coefficients_second_dimension'])-1

            self.doppler_coeff = Poly2D.Poly2D()
            self.doppler_coeff.initPoly(rangeOrder = rngOrder, azimuthOrder=azOrder)
            self.doppler_coeff.setMeanRange(self.doppler_ref_range)
            self.doppler_coeff.setMeanAzimuth(secondsSinceMidnight(self.doppler_ref_azi))

            parms = []
            for ii in range(azOrder+1):
                row = []
                for jj in range(rngOrder+1):
                    key = 'a%d%d'%(ii,jj)
                    val = float(coeff[key])
                    row.append(val)

                parms.append(row)

            self.doppler_coeff.setCoeffs(parms)


    def extractDoppler(self):
        '''
        Evaluate the doppler polynomial and return the average value for now.
        '''

        rmin = self.frame.getStartingRange()
        rmax = self.frame.getFarRange()
        rmid = 0.5*(rmin + rmax)
        delr = Const.c/ (2*self.frame.instrument.rangeSamplingRate)

        azmid = secondsSinceMidnight(self.frame.getSensingMid())

        print(rmid, self.doppler_coeff.getMeanRange())
        print(azmid, self.doppler_coeff.getMeanAzimuth())

        if self.doppler_coeff is None:
            raise Exception('ASF PARFILE was not provided. Cannot determine default doppler.')

        dopav = self.doppler_coeff(azmid, rmid)
        prf = self.frame.getInstrument().getPulseRepetitionFrequency()
        quadratic = {}
        quadratic['a'] = dopav / prf
        quadratic['b'] = 0.
        quadratic['c'] = 0.


        ######Set up the doppler centroid computation just like CSK at mid azimuth
        order = self.doppler_coeff._rangeOrder 
        rng = np.linspace(rmin, rmax, num=(order+2))
        pix = (rng - rmin)/delr
        val =[self.doppler_coeff(azmid,x) for x in rng]

        print(rng,val)
        print(delr, pix)
        fit = np.polyfit(pix, val, order)
        self.frame._dopplerVsPixel = list(fit[::-1])
#        self.frame._dopplerVsPixel = [dopav,0.,0.,0.]
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
        self.leaderFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'radarsat/leader_file.xml'),dataFile=fp)
        self.leaderFDR.parse()
        fp.seek(self.leaderFDR.getEndOfRecordPosition())
        # Scene Header
        self.sceneHeaderRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'radarsat/scene_record.xml'),dataFile=fp)
        self.sceneHeaderRecord.parse()
        fp.seek(self.sceneHeaderRecord.getEndOfRecordPosition())
        # Platform Position
        self.platformPositionRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'radarsat/platform_position_record.xml'),dataFile=fp)
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

        volumeFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'radarsat/volume_descriptor.xml'),dataFile=fp)
        volumeFDR.parse()
        fp.seek(volumeFDR.getEndOfRecordPosition())

        fp.close()

        pprint.pprint(volumeFDR.metadata)

class ImageFile(object):

    maxLineGap = 126
    rsatSMO = 129.2683e6
    oneWayDelay = 2.02e-6
    beamToRangeGateEdge = { '1' :  [7, 428],
                            '2' :  [7, 428],
                            '3' :  [8, 176],
                            '4' :  [8, 176],
                            '5' :  [8, 176],
                            '6' :  [9, 176],
                            '7' :  [9, 176],
                            '16':  [8, 176],
                            '17':  [8, 176],
                            '18':  [8, 176],
                            '19':  [9, 176],
                            '20':  [9, 176]}

    def __init__(self, parent, file=None):
        self.parent = parent
        self.file = file
        self.prf = None
        self.pulseLength = None
        self.rangeSamplingRate = None
        self.chirpSlope = None
        self.imageFDR = None
        self.sensingStart = None
        self.image_record = os.path.join(xmlPrefix,'radarsat/image_record.xml')
        self.logger = logging.getLogger('isce.sensor.rsat1')

    def parse(self):
        try:
            fp = open(self.file,'rb')
        except IOError as errs:
            errno,strerr = errs
            print("IOError: %s" % strerr)
            return

        self.imageFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'radarsat/image_file.xml'), dataFile=fp)
        self.imageFDR.parse()
        fp.seek(self.imageFDR.getEndOfRecordPosition())
        self._calculateRawDimensions(fp)

        fp.close()

    def extractAUXinformation(self, fp):
        '''
        Read 50 bytes of data and interpret the aux information.
        Does not CEOS reader format as we want access to sub-byte data.
        Currently only extracts the ones we want.
        '''

#% The parameters encoded in the auxilary data bits are defined in RSI-D6
#% also known as RSCSA-IC0009 (X-band ICD)
#% -------------------------------------------------------------------------
#% PARAMETER NAME                          LOCATION         LENGTH   ID
#% -------------------------------------------------------------------------
#% aux_sync_marker         = aux_bits (:,   1:  32);     % 32 bit -  1
#% image_ref_id            = aux_bits (:,  33:  64);     % 32 bit -  2
#% payload_status          = aux_bits (:,  65:  80);     % 16 bit -  3
#% replica_AGC             = aux_bits (:,  81:  86);     %  6 bit -  4
#% CALN_atten_LPT_pow_set  = aux_bits (:,  89:  96);     %  8 bit -  6
#% pulse_waveform_number   = aux_bits (:,  97: 100);     %  4 bit -  7
#% temperature             = aux_bits (:, 113: 144);     % 32 bit -  9
#% beam_sequence           = aux_bits (:, 145: 160);     % 16 bit - 10
#% ephemeris               = aux_bits (:, 161: 176);     % 16 bit - 11
#% number_of_beams         = aux_bits (:, 177: 178);     %  2 bit - 12
#% ADC_rate                = aux_bits (:, 179: 180);     %  2 bit - 13
#% pulse_count_1           = aux_bits (:, 185: 192);     %  8 bit - 15
#% pulse_count_2           = aux_bits (:, 193: 200);     %  8 bit - 16
#% PRF_beam                = aux_bits (:, 201: 213);     % 13 bit - 17
#% beam_select             = aux_bits (:, 214: 215);     %  2 bit - 18
#% Rx_window_start_time    = aux_bits (:, 217: 228);     % 12 bit - 20
#% Rx_window_duration      = aux_bits (:, 233: 244);     % 12 bit - 22
#% altitude                = aux_bits (:, 249: 344);     % 96 bit - 24
#% time                    = aux_bits (:, 345: 392);     % 48 bit - 25
#% SC_T02_defaults         = aux_bits (:, 393: 393);     %  1 bit - 26
#% first_replica           = aux_bits (:, 394: 394);     %  1 bit - 27
#% Rx_AGC_setting          = aux_bits (:, 395: 400);     %  6 bit - 28
#% -------------------------------------------------------------------------
#%                                    Total  => 50 bytes (400 bits)
#% -------------------------------------------------------------------------
        aux_bytes = np.fromfile(fp, dtype='B', count=self.parent.auxLength)
        metadata = {}
        sec = (np.bitwise_and(aux_bytes[44], np.byte(31)) << 12) | (np.bitwise_and(aux_bytes[45], np.byte(255)) << 4) | (np.bitwise_and(aux_bytes[46], np.byte(240)) >> 4)
        millis = (np.bitwise_and(aux_bytes[46], np.byte(15)) << 6 ) | (np.bitwise_and(aux_bytes[47], np.byte(252)) >> 2)
        micros = (np.bitwise_and(aux_bytes[47], np.byte(3)) << 8) | np.bitwise_and(aux_bytes[48], np.byte(255))
        metadata['Record Time'] = sec + millis*1.0e-3 + micros*1.0e-6

        adc_code = np.bitwise_and(aux_bytes[22], np.byte(63)) >> 4
        prf_code = (np.bitwise_and(aux_bytes[25], np.byte(255)) << 5) | (np.bitwise_and(aux_bytes[26], np.byte(255)) >> 3)
        timeBase = (4.0 + 3.0*adc_code)/self.rsatSMO


        dwp_code = (np.bitwise_and(aux_bytes[27], np.byte(255)) <<4) | (np.bitwise_and(aux_bytes[28], np.byte(240)) >> 4)
        dwp = (5 + dwp_code)*6*timeBase

        beam = struct.unpack(">H", aux_bytes[18:20])[0]
        metadata['PRF'] = 1.0/((2+prf_code)*6.0*timeBase)
        metadata['fsamp'] = 1.0 / timeBase
        metadata['hasReplica'] = np.bitwise_and(aux_bytes[49] >> 6, 1)
        metadata['code'] = aux_bytes[25:27]
        metadata['beam'] = beam
#        metadata['Pulse length'] = int(0.000044559/timeBase)*2
        metadata['Pulse length'] = 42.0e-6
        metadata['Replica length'] = 2*int(0.000044559/timeBase)

        if metadata['fsamp'] > 30.0e6:
            metadata['Chirp slope'] = -7.214e11
        elif metadata['fsamp'] > 15.0e6:
            metadata['Chirp slope'] = -4.1619047619047619629e11
        else:
            metadata['Chirp slope'] = -2.7931e11


        rangeGateFactor, rightEdge = self.beamToRangeGateEdge[str(beam)]
        metadata['Starting Range'] = 0.5*Const.c* (dwp + rangeGateFactor/metadata['PRF'] - (2.0 * self.oneWayDelay))
        return metadata

    def writeRawData(self, fp, line):
        '''
        Radarsat stores the raw data in a format different from needed for ISCE or ROI_PAC.
        '''
        mask = line >= 8
        line[mask] -= 8
        line[np.logical_not(mask)] += 8
        line.tofile(fp)


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

        auxLength = self.parent.auxLength
        prf = self.parent.leaderFile.sceneHeaderRecord.metadata['Pulse Repetition Frequency']
        # Extract the I and Q channels
        imageData = CEOS.CEOSDB(xml=self.image_record,dataFile=fp)
        firstRecTime = None
        nominalPRF = None
        nominalPulseLength = None
        nominalRangeSamplingRate = None
        nominalChirpSlope = None
        nominalStartingRange = None
        startUTC = None
        lineCount = 0
        lineWidth = 0

        for line in range(self.length):
            if ((line%1000) == 0):
                self.logger.debug("Extracting line %s" % line)

            imageData.parseFast()
            auxData = self.extractAUXinformation(fp)

#            pprint.pprint(imageData.metadata)
            #####Check length of current record
            #######12 CEOS + 180 prefix + 50
            dataLen = imageData.recordLength - self.imageFDR.metadata['Number of bytes of prefix data per record'] - auxLength-12
            recTime = auxData['Record Time']
            if firstRecTime is None:
                firstRecTime = recTime
                nominalPRF = auxData['PRF']
                nominalPulseLength = auxData['Pulse length']
                nominalChirpSlope = auxData['Chirp slope']
                nominalRangeSamplingRate = auxData['fsamp']
                nominalStartingRange = auxData['Starting Range']
                replicaLength = auxData['Replica length']
                startUTC = datetime.datetime(imageData.metadata['Sensor acquisition year'],1,1) + datetime.timedelta(imageData.metadata['Sensor acquisition day of year']-1) + datetime.timedelta(seconds=auxData['Record Time'])
                prevRecTime = recTime

#                pprint.pprint(imageData.metadata)
                if (auxData['hasReplica']):
                    lineWidth = dataLen - replicaLength
                else:
                    lineWidth = dataLen


#            pprint.pprint(auxData)

            
            IQLine = np.fromfile(fp, dtype='B', count=dataLen)

            if (recTime > (prevRecTime + 0.001)):
                self.logger.debug('Time gap of %f sec at RecNum %d'%(recTime-prevRecTime,
                    imageData.metadata['Record Number']))

            if (auxData['hasReplica']):
#                self.logger.debug("Removing replica from Line %s" % line)
                IQLine[0:dataLen-replicaLength]= IQLine[replicaLength:dataLen]
                IQLine[dataLen-replicaLength:] = 16
                dataLen = dataLen-replicaLength

#            print('Line: ', line, dataLen, lineWidth, auxData['Replica length'])

            if dataLen >= lineWidth:
                IQout = IQLine
            else:
                IQout = 16 * np.ones(lineWidth, dtype='b')
                IQout[:dataLen] = IQLine


            lineGap = int(0.5+(recTime-prevRecTime)*nominalPRF)
            if ((lineGap == 1) or (line==0)):
                self.writeRawData(output, IQout[:lineWidth])
                lineCount += 1
                prevRecTime = recTime
            elif ((lineGap > 1) and (lineGap < self.maxLineGap)):
                for kk in range(lineGap):
                    self.writeRawData(output, IQout[:lineWidth])
                    lineCount += 1
                prevRecTime = recTime + (lineGap - 1)*8.0e-4
            elif (lineGap >= self.maxLineGap):
                raise Exception('Line Gap too big to be filled')

        self.prf = nominalPRF
        self.chirpSlope = nominalChirpSlope
        self.rangeSamplingRate = nominalRangeSamplingRate
        self.pulseLength = nominalPulseLength
        self.startingRange = nominalStartingRange
        self.sensingStart = startUTC
        self.length = lineCount
        self.width = lineWidth


    def _calculateRawDimensions(self,fp):
        """
            Run through the data file once, and calculate the valid sampling window start time range.
        """
        self.length = self.imageFDR.metadata['Number of SAR DATA records']
        self.width = self.imageFDR.metadata['SAR DATA record length']

        return None



class Node(object):
    def __init__(self, parent):
        self.parent = parent
        self.data = {}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, val):
        self.data[key] = val


class ParFile(object):
    '''
    Read ASF format parfile.
    '''

    def __init__(self, filename):
        self.filename = filename
        self.data = Node(None)
        self.parse()

    def __getitem__(self, key):
        return self.data[key]

    def parse(self):

        fid = open(self.filename, 'r')
        data = fid.readlines()
        fid.close()

        node = self.data

        for line in data:
            inline = line.strip()
            if inline == '':
                continue

            #####If start of section
            if inline.endswith('{'):
                sectionName = inline.split()[0]
                newNode = Node(node)

                if sectionName in node.data.keys():
#                    print(node[sectionName])
#                    print('Entering same named section: ', sectionName)
                    if not isinstance(node[sectionName], list):
                        node[sectionName] = [ node[sectionName] ]

                    node[sectionName].append(newNode)
                else:
                    node[sectionName] = newNode

                node = newNode


            ######If end of section
            elif inline.startswith('}'):
                node = node.parent

            #####Actually has some data
            else:
                try:
                    (key, val) = inline.split(':')
                except:
                    raise Exception('Could not parse line: ', inline)

                node[key.strip()] = val.strip()
