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




import re
import os
import copy
import numpy as np
import struct
import datetime
import logging
import isceobj
from isceobj import *
import ctypes
from isceobj.Planet.Planet import Planet
from isceobj.Orbit.Orbit import Orbit
from isceobj.Orbit.Orbit import StateVector
from isceobj.Planet.AstronomicalHandbook import Const
from isceobj.Scene.Frame import Frame
from isceobj.Scene.Track import Track
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTUtil
from iscesys.Component.Component import Component

ORBIT_DIRECTORY = Component.Parameter(
    'orbitDir',
    public_name='ORBIT_DIRECTORY',
    default=None,
    type=str,
    mandatory=False,
    intent='input',
    doc='Location of the orbit directory if an orbit file is not provided.'
)

ORBITFILE = Component.Parameter(
    'orbitFile',
    public_name='ORBITFILE',
    default=None,
    type=str,
    mandatory=True,
    intent='input',
    doc='Orbit file.'
)

INSTRUMENTFILE = Component.Parameter(
    'instrumentFile',
    public_name='INSTRUMENTFILE',
    default=None,
    type=str,
    mandatory=True,
    intent='input',
    doc='Instrument file.'
)

INSTRUMENT_DIRECTORY = Component.Parameter(
    'instrumentDir',
    public_name='INSTRUMENT_DIRECTORY',
    default=None,
    type=str,
    mandatory=False,
    intent='input',
    doc='Instrument directory if an instrument file is not provided.'
)

IMAGEFILE = Component.Parameter(
    '_imageFileName',
    public_name='IMAGEFILE',
    default='',
    type=str,
    mandatory=True,
    intent='input',
    doc='Input image file.'
)

from .Sensor import Sensor
class EnviSAT_SLC(Sensor):

    parameter_list = (ORBIT_DIRECTORY,
                      ORBITFILE,
                      INSTRUMENTFILE,
                      INSTRUMENT_DIRECTORY,
                      IMAGEFILE)           + Sensor.parameter_list

    """
        A Class for parsing EnviSAT instrument and imagery files
    """

    family = 'envisat'

    def __init__(self,family='',name=''):
        super(EnviSAT_SLC, self).__init__(family if family else  self.__class__.family, name=name)
        self._imageFile = None
        self._instrumentFileData = None
        self._imageryFileData = None
        self.dopplerRangeTime = None
        self.rangeRefTime = None
        self.logger = logging.getLogger("isce.sensor.EnviSAT_SLC")
        self.frame = None
        self.frameList = []


        self.constants = {'antennaLength': 10.0,
                          'iBias': 128,
                          'qBias': 128}

    def getFrame(self):
        return self.frame

    def parse(self):
        """
            Parse both imagery and instrument files and create
            objects representing the platform, instrument and scene
        """

        self.frame = Frame()
        self.frame.configure()

        self._imageFile = ImageryFile(fileName=self._imageFileName)
        self._imageryFileData = self._imageFile.parse()

        if self.instrumentFile in [None, '']:
            self.findInstrumentFile()

        instrumentFileParser = InstrumentFile(fileName=self.instrumentFile)
        self._instrumentFileData = instrumentFileParser.parse()

        self.populateMetadata()

    def populateMetadata(self):

        self._populatePlatform()
        self._populateInstrument()
        self._populateFrame()
        self._populateOrbit()
        self.dopplerRangeTime = self._imageryFileData['doppler']
        self.rangeRefTime = self._imageryFileData['dopplerOrigin'][0] * 1.0e-9
#        print('Doppler confidence: ', 100.0 * self._imageryFileData['dopplerConfidence'][0])

    def _populatePlatform(self):
        """Populate the platform object with metadata"""
        platform = self.frame.getInstrument().getPlatform()

        # Populate the Platform and Scene objects
        platform.setMission("Envisat")
        platform.setPointingDirection(-1)
        platform.setAntennaLength(self.constants['antennaLength'])
        platform.setPlanet(Planet(pname="Earth"))

    def _populateInstrument(self):
        """Populate the instrument object with metadata"""
        instrument = self.frame.getInstrument()

        rangeSampleSpacing = Const.c/(2*self._imageryFileData['rangeSamplingRate'])
        pri = self._imageryFileData['pri']


        ####These shouldnt matter for SLC data since data is already focused.
        txPulseLength = 512 / 19207680.000000
        chirpPulseBandwidth = 16.0e6
        chirpSlope = chirpPulseBandwidth/txPulseLength

        instrument.setRangePixelSize(rangeSampleSpacing)
        instrument.setPulseLength(txPulseLength)
        #instrument.setSwath(imageryFileData['SWATH'])
        instrument.setRadarFrequency(self._instrumentFileData['frequency'])
        instrument.setChirpSlope(chirpSlope)
        instrument.setRangeSamplingRate(self._imageryFileData['rangeSamplingRate'])
        instrument.setPulseRepetitionFrequency(1.0/pri)
        #instrument.setRangeBias(rangeBias)
        instrument.setInPhaseValue(self.constants['iBias'])
        instrument.setQuadratureValue(self.constants['qBias'])

    def _populateFrame(self):
        """Populate the scene object with metadata"""
        numberOfLines = self._imageryFileData['numLines']
        numberOfSamples = self._imageryFileData['numSamples']
        pri = self._imageryFileData['pri']
        startingRange = Const.c * float(self._imageryFileData['timeToFirstSample']) * 1.0e-9 / 2.0
        rangeSampleSpacing = Const.c/(2*self._imageryFileData['rangeSamplingRate'])
        farRange = startingRange + numberOfSamples*rangeSampleSpacing
        first_line_utc = datetime.datetime.strptime(self._imageryFileData['FIRST_LINE_TIME'], '%d-%b-%Y %H:%M:%S.%f')
        center_line_utc = datetime.datetime.strptime(self._imageryFileData['FIRST_LINE_TIME'], '%d-%b-%Y %H:%M:%S.%f')
        last_line_utc = datetime.datetime.strptime(self._imageryFileData['LAST_LINE_TIME'], '%d-%b-%Y %H:%M:%S.%f')
        centerTime = DTUtil.timeDeltaToSeconds(last_line_utc-first_line_utc)/2.0
        center_line_utc = center_line_utc + datetime.timedelta(microseconds=int(centerTime*1e6))

        self.frame.setStartingRange(startingRange)
        self.frame.setFarRange(farRange)
        self.frame.setProcessingFacility(self._imageryFileData['PROC_CENTER'])
        self.frame.setProcessingSystem(self._imageryFileData['SOFTWARE_VER'])
        self.frame.setTrackNumber(int(self._imageryFileData['REL_ORBIT']))
        self.frame.setOrbitNumber(int(self._imageryFileData['ABS_ORBIT']))
        self.frame.setPolarization(self._imageryFileData['MDS1_TX_RX_POLAR'])
        self.frame.setNumberOfSamples(numberOfSamples)
        self.frame.setNumberOfLines(numberOfLines)
        self.frame.setSensingStart(first_line_utc)
        self.frame.setSensingMid(center_line_utc)
        self.frame.setSensingStop(last_line_utc)

    def _populateOrbit(self):
        if self.orbitFile in [None, '']:
            self.findOrbitFile()

        dorParser = DOR(fileName=self.orbitFile)
        dorParser.parse()
        startTime = self.frame.getSensingStart() - datetime.timedelta(minutes=5)
        stopTime = self.frame.getSensingStop() + datetime.timedelta(minutes=5)
        self.frame.setOrbit(dorParser.orbit.trimOrbit(startTime,stopTime))

    def _populateImage(self,outname,width,length):

        #farRange = self.frame.getStartingRange() + width*self.frame.getInstrument().getRangeSamplingRate()
        # Update the NumberOfSamples and NumberOfLines in the Frame object
        self.frame.setNumberOfSamples(width)
        self.frame.setNumberOfLines(length)
        #self.frame.setFarRange(farRange)
        # Create a RawImage object
        rawImage = createSlcImage()
        rawImage.setFilename(outname)
        rawImage.setAccessMode('read')
        rawImage.setByteOrder('l')
        rawImage.setXmin(0)
        rawImage.setXmax(width)
        rawImage.setWidth(width)
        self.frame.setImage(rawImage)

    def extractImage(self):
        from datetime import datetime as dt
        import tempfile as tf

        self.parse()
        width = self._imageryFileData['numSamples']
        length = self._imageryFileData['numLines']
        self._imageFile.extractImage(self.output, width, length)
        self._populateImage(self.output, width, length)

        pass

    def findOrbitFile(self):

        datefmt = '%Y%m%d%H%M%S'
#        sensingStart = self.frame.getSensingStart()
        sensingStart = datetime.datetime.strptime(self._imageryFileData['FIRST_LINE_TIME'], '%d-%b-%Y %H:%M:%S.%f')
        outFile = None

        if self.orbitDir in [None,'']:
            raise Exception('No Envisat Orbit File or Orbit Directory specified')

        try:
            for fname in os.listdir(self.orbitDir):
                if not os.path.isfile(os.path.join(self.orbitDir,fname)):
                    continue

                if not fname.startswith('DOR'):
                    continue

                fields = fname.split('_')
                procdate = datetime.datetime.strptime(fields[-6][-8:] + fields[-5], datefmt)
                startdate = datetime.datetime.strptime(fields[-4] + fields[-3], datefmt)
                enddate = datetime.datetime.strptime(fields[-2] + fields[-1], datefmt)

                if (sensingStart > startdate) and (sensingStart < enddate):
                    outFile = os.path.join(self.orbitDir, fname)
                    break

        except:
            raise Exception('Error occured when trying to find orbit file in %s'%(self.orbitDir))

        if not outFile:
            raise Exception('Envisat orbit file could not be found in %s'%(self.orbitDir))

        self.orbitFile = outFile
        return

    def findInstrumentFile(self):

        datefmt = '%Y%m%d%H%M%S'

        sensingStart =   datetime.datetime.strptime(self._imageryFileData['FIRST_LINE_TIME'], '%d-%b-%Y %H:%M:%S.%f')
        print('sens: ', sensingStart)
        outFile = None

        if self.instrumentDir in [None,'']:
            raise Exception('No Envisat Instrument File or Instrument Directory specified')

        try:
            for fname in os.listdir(self.instrumentDir):
                if not os.path.isfile(os.path.join(self.instrumentDir,fname)):
                    continue

                if not fname.startswith('ASA_INS'):
                    continue

                fields = fname.split('_')
                procdate = datetime.datetime.strptime(fields[-6][-8:] + fields[-5], datefmt)
                startdate = datetime.datetime.strptime(fields[-4] + fields[-3], datefmt)
                enddate = datetime.datetime.strptime(fields[-2] + fields[-1], datefmt)

                if (sensingStart > startdate) and (sensingStart < enddate):
                    outFile = os.path.join(self.instrumentDir, fname)
                    break

        except:
            raise Exception('Error occured when trying to find instrument file in %s'%(self.instrumentDir))

        if not outFile:
            raise Exception('Envisat instrument file could not be found in %s'%(self.instrumentDir))

        self.instrumentFile = outFile
        return


    def extractDoppler(self):
        """
        Return the doppler centroid as defined in the ASAR file.
        """
        quadratic = {}

        r0 = self.frame.getStartingRange()
        dr = self.frame.instrument.getRangePixelSize()
        width = self.frame.getNumberOfSamples()

        midr = r0 + (width/2.0) * dr
        midtime = 2 * midr/ Const.c - self.rangeRefTime

        fd_mid = 0.0
        tpow = midtime
        for kk in self.dopplerRangeTime:
            fd_mid += kk * tpow
            tpow *= midtime


        ####For insarApp
        quadratic['a'] = fd_mid/self.frame.getInstrument().getPulseRepetitionFrequency()
        quadratic['b'] = 0.
        quadratic['c'] = 0.

        
        ####For roiApp
        ####More accurate
        from isceobj.Util import Poly1D
        
        coeffs = self.dopplerRangeTime
        dr = self.frame.getInstrument().getRangePixelSize()
        rref = 0.5 * Const.c * self.rangeRefTime 
        r0 = self.frame.getStartingRange()
        norm = 0.5*Const.c/dr

        dcoeffs = []
        for ind, val in enumerate(coeffs):
            dcoeffs.append( val / (norm**ind))


        poly = Poly1D.Poly1D()
        poly.initPoly(order=len(coeffs)-1)
        poly.setMean( (rref - r0)/dr - 1.0)
        poly.setCoeffs(dcoeffs)


        pix = np.linspace(0, self.frame.getNumberOfSamples(), num=len(coeffs)+1)
        evals = poly(pix)
        fit = np.polyfit(pix,evals, len(coeffs)-1)
        self.frame._dopplerVsPixel = list(fit[::-1])
        print('Doppler Fit: ', fit[::-1])


        return quadratic


class BaseEnvisatFile(object):
    """Class for parsing common Envisat metadata"""

    def __init__(self):
        self.fp = None
        self.mphLength = 1247
        self.sphLength = None
        self.mph = {}
        self.sph = {}

    def readMPH(self):
        """Unpack the Main Product Header (MPH)"""
        mphString = self.fp.read(self.mphLength)
        header = mphString.splitlines()
        for line in header:
            (key, sep, value) = line.decode('utf8').partition('=')
            if (key.isspace() == False):
                value = str.replace(value,'"','')
                value = str.strip(value)
                self.mph[key] = value

        # Grab the length of the SPH section
        self.sphLength = self._extractValue(value=self.mph['SPH_SIZE'], type=int)

    def readSPH(self):
        """Unpack the Specific Product Header (SPH)"""
        self.fp.seek(self.mphLength)
        sphString = self.fp.read(self.sphLength)
        header = sphString.splitlines()

        dsSeen = False
        dataSet = {}
        dataSets = []
        # the Specific Product Header is made of up key-value pairs.
        # At the end of the header, there are a number of data blocks that
        # represent the data sets that follow.  Since their key names are
        # not unique, we need to capture them in an array and then tack
        # this array on the dictionary later.  These data sets begin with
        # a key named "DS_NAME"
        for line in header:
            (key, sep, value) = line.decode('utf8').partition('=')
            if (key.isspace() == False):
                value = str.replace(value,'"','')
                value = str.strip(value)
                # Check to see if we are reading a Data Set record
                if ((key == 'DS_NAME') and (dsSeen == False)):
                    dsSeen = True

                if (dsSeen == False):
                    self.sph[key] = value
                else:
                    dataSet[key] = value

                if (key == 'DSR_SIZE'):
                    dataSets.append(copy.copy(dataSet))

        self.sph['dataSets'] = dataSets



    def _readAndUnpackData(self, length=None, format=None, type=None, numberOfFields=1):
        """
            Convenience method for reading and unpacking data.

            length is the length of the field in bytes [required]
            format is the format code to use in struct.unpack() [required]
            numberOfFields is the number of fields expected from the call to struct.unpack() [default = 1]
            type is the function through which the output of struct.unpack will be passed [default = None]
        """
        line = self.fp.read(length)
        data = struct.unpack(format, line)
        if (numberOfFields == 1):
            data = data[0]
        if (type):
            try:
                data = type(data)
            except ValueError:
                pass

        return data

    def _extractValue(self,value=None,type=None):
        """
            Some MPH and SPH fields have units appended to the value in the form of:
            124<bytes>.  This method strips off the units and returns a value of the
            correct type.
        """
        matches = re.search("([+-]?[\w\.]+)<[\w/]+>",value)
        answer = matches.group(1)
        if (answer == None):
            print("No Matches Found")
            return

        if (type != None):
            answer = type(answer)

        return answer

class InstrumentFile(BaseEnvisatFile):
    """Parse an Envisat Instrument Calibration file"""

    def __init__(self, fileName=None):
        BaseEnvisatFile.__init__(self)
        self.fileName = fileName

    def parse(self):

        instrumentDict = {}
        try:
            self.fp = open(self.fileName, 'rb')
        except IOError as errs:
            errno,strerr = errs
            print("IOError: {} {}".format(strerr,self.fileName))
            return

        self.readMPH()
        self.readSPH()
        gadsDict = self.readGADS()

        self.fp.close()

        instrumentDict.update(self.mph)
        instrumentDict.update(self.sph)
        instrumentDict.update(gadsDict)

        return instrumentDict

    def readGADS(self):
        """
            Read the Global Auxillary Data Set (GADS) for the Instrument Characterization Auxillary File
        """
        gadsDict = {}

        self.fp.seek((self.mphLength + self.sphLength))
        gadsDict['mjd'] = self._readAndUnpackData(12, ">3I", numberOfFields=3)
        gadsDict['dsrLength'] = self._readAndUnpackData(4, ">I")
        gadsDict['frequency'] = self._readAndUnpackData(4, ">f", type=float)
        gadsDict['sampleRate'] = self._readAndUnpackData(4, ">f",type=float)
        gadsDict['offsetFrequency'] = self._readAndUnpackData(4, ">f")

        # There are many, many other entries in this file.  Most of the remaining
        # entries are calibration pulses.  I'm going to cheat and skip ahead to read
        # the values for the number of PRIs between transmit and receive.  If you are
        # bored and want to code the remaining 130+ values, there is a table at:
        # http://envisat.esa.int/handbooks/asar/CNTR6-6-3.htm#eph.asar.asardf.asarrec.ASA_INS_AX_GADS

        self.fp.seek(69884, os.SEEK_CUR) # Seek to record 66
        gadsDict['rangeGateBias'] = self._readAndUnpackData(length=4, format=">f", type=float)
        self.fp.seek(91678, os.SEEK_CUR) # Seek to record 105
        self.fp.seek(28, os.SEEK_CUR) # Skip to the r_values
        r_values = [None]*7
        for i in range(7):
            r_values[i] = self._readAndUnpackData(length=2, format=">H", type=int)

        gadsDict['r_values'] = r_values

        return gadsDict

class ImageryFile(BaseEnvisatFile):
    """Parse an Envisat Imagery File"""

    def __init__(self, fileName=None):
        BaseEnvisatFile.__init__(self)
        self.fileName = fileName
        self.sqLength = 170
        self.procParamLength = None
        self.doppParamLength = 55
        self.chirpParamLength = 1483
        self.geoParamLength = None

    def parse(self):

        def getDictByKey(inlist, key):
            for kk in inlist:
                if kk['DS_NAME'] == key:
                    return kk

            return None

        import pprint
        imageryDict = {}
        try:
            self.fp = open(self.fileName, 'rb')
        except IOError as errs:
            errno,strerr = errs
            print("IOError: %s %s" % (strerr,self.fileName))
            return

        self.readMPH()
        self.readSPH()

        self.sqLength = self._extractValue(value = getDictByKey(self.sph['dataSets'], 
                                        'MDS1 SQ ADS')['DS_SIZE'], type=int)
        self.procParamLength = self._extractValue(value=getDictByKey(self.sph['dataSets'],
                                        'MAIN PROCESSING PARAMS ADS')['DS_SIZE'], type=int)
        self.doppParamLength = self._extractValue(value=getDictByKey(self.sph['dataSets'],
                                        'DOP CENTROID COEFFS ADS')['DS_SIZE'], type=int)
        self.chirpParamLength = self._extractValue(value=getDictByKey(self.sph['dataSets'],
                                        'CHIRP PARAMS ADS')['DS_SIZE'], type=int)
        self.geoParamLength = self._extractValue(value=getDictByKey(self.sph['dataSets'],
                                        'GEOLOCATION GRID ADS')['DS_SIZE'], type=int)

        ####Handling software version change in 6.02
        ver = float(self.mph['SOFTWARE_VER'].strip()[-4:])

        if ver < 6.02:
            print('Old ESA Software version: ', ver)
#            self.procParamLength = 2009
#            self.geoParamLength = 521*12
        else:
            print('New ESA Software version: ', ver)
#            self.procParamLength = 10069
#            self.geoParamLength = 521*13


        procDict = self.readProcParams()
        doppDict = self.readDopplerParams()
        geoDict = self.readGeoParams()
        self.fp.close()

        imageryDict.update(self.mph)
        imageryDict.update(self.sph)
        imageryDict.update(procDict)
        imageryDict.update(doppDict)
        imageryDict.update(geoDict)
        return imageryDict


    def getTotalHeaderLength(self):
        headerLength = self.mphLength + self.sphLength + self.sqLength +\
                self.procParamLength + self.doppParamLength + self.chirpParamLength +\
                self.geoParamLength

        return headerLength


    def extractImage(self, outname, width, length):
        try:
            self.fp = open(self.fileName, 'rb')
        except IOError as errs:
            errno,strerr = errs
            print("IOError: %s %s" % (strerr,self.fileName))
            return

        self.fp.seek(self.getTotalHeaderLength())

        fout = open(outname, 'wb')
        for kk in range(length):

            if ((kk+1) %1000 == 0):
                print('Extracted line: %d'%(kk+1))

            rec = self.fp.read(17)
#            num = struct.unpack(">L", rec[13:17])[0]
            line = np.fromfile(self.fp, dtype='>h', count=2*width)
            line.astype(np.float32).tofile(fout)

        fout.close()
        self.fp.close()

        return


    def readProcParams(self):
        """Unpack information from the processing parameters dataset"""
        headerLength = self.mphLength + self.sphLength + self.sqLength
        self.fp.seek(headerLength)
        record = self.fp.read(self.procParamLength)

        procDict = {}
        procDict['mdsFirstTime'] = struct.unpack(">3L",record[:12])
        procDict['mdsLastTime'] = struct.unpack(">3L", record[13:25])
        procDict['timeDiffSensing'] = struct.unpack(">f",record[37:41])[0]
        procDict['rangeSpacing'] = struct.unpack(">f",record[44:48])[0]
        procDict['azimuthSpacing'] = struct.unpack(">f",record[48:52])[0]
        procDict['pri'] = struct.unpack(">f", record[52:56])[0]
        procDict['numLines'] = int(struct.unpack(">L", record[56:60])[0])
        procDict['numSamples'] = int(struct.unpack(">L", record[60:64])[0])
        procDict['timeDiffZeroDoppler'] = struct.unpack(">f", record[73:77])[0]
        procDict['firstProcSample'] = int(struct.unpack(">L", record[975:979])[0])
        procDict['referenceRange'] = struct.unpack(">f", record[979:983])[0]
        procDict['rangeSamplingRate'] = struct.unpack(">f", record[983:987])[0]
        procDict['radarFrequency'] = struct.unpack(">f", record[987:991])[0]

        procDict['azimuthFMRate'] = struct.unpack(">3f",record[1289:1301])
        procDict['azimuthFMOrigin'] = struct.unpack(">f", record[1301:1305])[0]

        procDict['averageEllipiseHeight'] = struct.unpack(">f", record[1537:1541])[0]

        ####State vectors starting from 1761
        return procDict

    def readDopplerParams(self):
        """Unpack information from the doppler coefficients dataset"""
        headerLength = self.mphLength + self.sphLength + self.sqLength + self.procParamLength
        self.fp.seek(headerLength)
        record = self.fp.read(self.doppParamLength)

        doppDict = {}
        doppDict['dopTime'] = struct.unpack(">3L", record[:12])
        doppDict['dopplerOrigin'] = struct.unpack(">f", record[13:17])
        doppDict['doppler'] = struct.unpack(">5f",record[17:37])
        doppDict['dopplerConfidence'] = struct.unpack(">f", record[37:41])
        doppDict['dopplerDeltas'] = struct.unpack(">5h",record[42:52])
        return doppDict


    def readGeoParams(self):
        '''Unpack information regarding starting range.'''

        headerLength = self.mphLength + self.sphLength + self.sqLength +\
                self.procParamLength + self.doppParamLength + self.chirpParamLength

        self.fp.seek(headerLength + 25+44)
        record = self.fp.read(4)

        geoDict = {}
        geoDict['timeToFirstSample'] = struct.unpack(">f",record)[0]

        return geoDict


class DOR(BaseEnvisatFile):
    """A class for parsing Envisat DORIS orbit files"""

    def __init__(self,fileName=None):
        BaseEnvisatFile.__init__(self)
        self.fileName = fileName
        self.fp = None
        self.orbit = Orbit()
        self.orbit.setOrbitSource('DORIS')
        self.orbit.setReferenceFrame('ECR')

    def parse(self):

        orbitDict = {}
        try:
            self.fp = open(self.fileName, 'rb')
        except IOError as errs:
            errno,strerr = errs
            print("IOError: %s" % strerr)
            return

        self.readMPH()
        self.readSPH()
        self.readStateVectors()

        self.fp.close()

        if (self.sph['dataSets'][0]['DS_NAME'] == 'DORIS PRELIMINARY ORBIT'):
            self.orbit.setOrbitQuality('Preliminary')
        elif (self.sph['dataSets'][0]['DS_NAME'] == 'DORIS PRECISE ORBIT'):
            self.orbit.setOrbitQuality('Precise')

        orbitDict.update(self.mph)
        orbitDict.update(self.sph)

        return orbitDict

    def readStateVectors(self):
        headerLength = self.mphLength + self.sphLength
        self.fp.seek(headerLength)

        for line in self.fp.readlines():
            vals = line.decode('utf8').split()
            dateTime = self._parseDateTime(vals[0] + ' ' + vals[1])
            position = list(map(float,vals[4:7]))
            velocity = list(map(float,vals[7:10]))
            sv = StateVector()
            sv.setTime(dateTime)
            sv.setPosition(position)
            sv.setVelocity(velocity)
            self.orbit.addStateVector(sv)

    def _parseDateTime(self,dtString):
        dateTime = datetime.datetime.strptime(dtString,'%d-%b-%Y %H:%M:%S.%f')
        return dateTime

class ImageOutput():
    """An object to represent the output struct from asa_im_decode"""

    def __init__(self, samples, lines):
        self.samples = samples
        self.lines = lines
