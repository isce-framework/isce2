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
    '_imageryFileList',
    public_name='IMAGEFILE',
    default='',
    container=list,
    type=str,
    mandatory=True,
    intent='input',
    doc='Input image file.'
)

from .Sensor import Sensor
class EnviSAT(Sensor):


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
        super(EnviSAT, self).__init__(family if family else  self.__class__.family, name=name)
        self.imageryFile = None
        self._instrumentFileData = None
        self._imageryFileData = None
        self.logger = logging.getLogger("isce.sensor.EnviSAT")
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

        imageryFileParser = ImageryFile(fileName=self.imageryFile)
        self._imageryFileData = imageryFileParser.parse()
        first_line_utc = datetime.datetime.strptime(self._imageryFileData['SENSING_START'], '%d-%b-%Y %H:%M:%S.%f')

        if self.instrumentFile in [None, '']:
            self.findInstrumentFile(first_line_utc)

        instrumentFileParser = InstrumentFile(fileName=self.instrumentFile)
        self._instrumentFileData = instrumentFileParser.parse()

        self.populateMetadata()

    def populateMetadata(self):

        self._populatePlatform()
        self._populateInstrument()
        self._populateFrame()
        self._populateOrbit()

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

        rangeSampleSpacing = Const.c/(2*self._instrumentFileData['sampleRate'])
        txPulseLength = self._imageryFileData['TxPulseLengthCodeword']/self._instrumentFileData['sampleRate']
        pri = self._imageryFileData['priCodeword']/self._instrumentFileData['sampleRate']
        chirpPulseBandwidth = self._imageryFileData['chirpPulseBandwidthCodeword']*16.0e6/255.0
        chirpSlope = chirpPulseBandwidth/txPulseLength

        ####ChirpSlope from GADS
        index = self._imageryFileData['antennaBeamSetNumber']-1
        chirpSlope = 2.0*self._instrumentFileData['nom_chirp_{0}'.format(index)][6]

        if (chirpSlope * txPulseLength) > chirpPulseBandwidth:
            print('Warning: Chirp Bandwidth > Slope * Pulse length')
            print('Check parser again .....')

        instrument.setRangePixelSize(rangeSampleSpacing)
        instrument.setPulseLength(txPulseLength)
        #instrument.setSwath(imageryFileData['SWATH'])
        instrument.setRadarFrequency(self._instrumentFileData['frequency'])
        instrument.setChirpSlope(chirpSlope)
        instrument.setRangeSamplingRate(self._instrumentFileData['sampleRate'])
        instrument.setPulseRepetitionFrequency(1/pri)
        #instrument.setRangeBias(rangeBias)
        instrument.setInPhaseValue(self.constants['iBias'])
        instrument.setQuadratureValue(self.constants['qBias'])

    def _populateFrame(self):
        """Populate the scene object with metadata"""
        # Decode some code words, and calculate some parameters
        numberOfLines = None
        for dataSet in self._imageryFileData['dataSets']:
            if (dataSet['DS_NAME'] == 'ASAR_SOURCE_PACKETS'):
                numberOfLines = int(dataSet['NUM_DSR'])

        numberOfSamples = self._imageryFileData['numberOfSamples']
        pri = self._imageryFileData['priCodeword']/self._instrumentFileData['sampleRate']
        windowStartTime = self._imageryFileData['windowStartTimeCodeword']/self._instrumentFileData['sampleRate']
        rangeSampleSpacing = Const.c/(2*self._instrumentFileData['sampleRate'])
        index = self._imageryFileData['antennaBeamSetNumber']-1
        startingRange = (self._instrumentFileData['r_values'][index]*pri + windowStartTime) * Const.c/2.0
        farRange = startingRange + numberOfSamples*rangeSampleSpacing
        rangeBias = self._instrumentFileData['rangeGateBias']*Const.c/2
        # The %b in the next lines strptime read the abbreviated month of the year by locale and could
        # present a problem for people with a different locale set.
        first_line_utc = datetime.datetime.strptime(self._imageryFileData['SENSING_START'], '%d-%b-%Y %H:%M:%S.%f')
        center_line_utc = datetime.datetime.strptime(self._imageryFileData['SENSING_START'], '%d-%b-%Y %H:%M:%S.%f')
        last_line_utc = datetime.datetime.strptime(self._imageryFileData['SENSING_STOP'], '%d-%b-%Y %H:%M:%S.%f')
        centerTime = DTUtil.timeDeltaToSeconds(last_line_utc-first_line_utc)/2.0
        center_line_utc = center_line_utc + datetime.timedelta(microseconds=int(centerTime*1e6))

        self.frame.setStartingRange(startingRange-rangeBias)
        self.frame.setFarRange(farRange-rangeBias)
        self.frame.setProcessingFacility(self._imageryFileData['PROC_CENTER'])
        self.frame.setProcessingSystem(self._imageryFileData['SOFTWARE_VER'])
        self.frame.setTrackNumber(int(self._imageryFileData['REL_ORBIT']))
        self.frame.setOrbitNumber(int(self._imageryFileData['ABS_ORBIT']))
        self.frame.setPolarization(self._imageryFileData['TX_RX_POLAR'])
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

    def _populateImage(self,outStruct,outname):
        width = outStruct.samples
        length = outStruct.lines

        #farRange = self.frame.getStartingRange() + width*self.frame.getInstrument().getRangeSamplingRate()
        # Update the NumberOfSamples and NumberOfLines in the Frame object
        self.frame.setNumberOfSamples(width)
        self.frame.setNumberOfLines(length)
        #self.frame.setFarRange(farRange)
        # Create a RawImage object
        rawImage = createRawImage()
        rawImage.setFilename(outname)
        rawImage.setAccessMode('read')
        rawImage.setByteOrder('l')
        rawImage.setXmin(0)
        rawImage.setXmax(2*width)
        rawImage.setWidth(2*width)
        self.frame.setImage(rawImage)





    def extractImage(self):
        from datetime import datetime as dt
        import tempfile as tf
        lib = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + '/envisat.so')
        #check if input file is a string or a list (then do concatenation)
        #ussume that one orbit and one instrument is enough for all the frame in the list
#        if isinstance(self._imageryFileList,str):
#            self._imageryFileList = [self._imageryFileList]
        self.frameList = []
        for i in range(len(self._imageryFileList)):
            appendStr = '_' + str(i) #intermediate raw files suffix
            if len(self._imageryFileList) == 1:
                appendStr = '' # no suffix if only one file

            outputNow = self.output + appendStr
            auxImage = isceobj.createImage()
            widthAux = 2
            auxName = outputNow + '.aux'
            self.imageryFile = self._imageryFileList[i]
            self.frame = Frame()
            self.frame.configure()

            #add the auxFile as part of the frame and diring the stitching create also a combined aux file
            self.frame.auxFile = auxName
            self.parse()


            #Declare the types of the arguments to asa_im_decode
            lib.asa_im_decode.argtypes = [ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_ushort,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int)]

            #Set the daysToRemove variable for the call to asa_im_decode
            sensingYear = self.frame.getSensingStart().year
            daysToRemove = ctypes.c_int(
                               (dt(sensingYear-1,12,31) - dt(2000,1,1)).days
                           )

            #Create memory for pointers nsamps and nlines to be set by
            #asa_im_decode
            a = 1
            b = 2
            nsamps = ctypes.pointer(ctypes.c_int(a))
            nlines = ctypes.pointer(ctypes.c_int(b))

            #Variables for outType and windowStartTimeCodeword0 passed to
            #asa_im_decode
            c = 1
            d = 0

            lib.asa_im_decode(ctypes.c_char_p(bytes(self.imageryFile,'utf-8')),
                ctypes.c_char_p(bytes(self.instrumentFile,'utf-8')),
                ctypes.c_char_p(bytes(outputNow,'utf-8')),
                ctypes.c_char_p(bytes(auxName,'utf-8')),
                ctypes.c_int(c),
                ctypes.c_ushort(d),
                daysToRemove,
                nsamps,
                nlines)

            #Create the outStruct for the call to populateImage
            outStruct = ImageOutput(nsamps[0], nlines[0])

            self._populateImage(outStruct,outputNow)
            self.frameList.append(self.frame)
            pass

        ## refactor this with __init__.tkfunc
        tk = Track()
        if(len(self._imageryFileList) > 1):
            self.frame = tk.combineFrames(self.output,self.frameList)

            for i in range(len(self._imageryFileList)):
                try:
                    os.remove(self.output + "_" + str(i))
                except OSError:
                    print("Error. Cannot remove temporary file",self.output + "_" + str(i))
                    raise OSError
                pass
            pass
        pass

    def findOrbitFile(self):

        datefmt = '%Y%m%d%H%M%S'
        sensingStart = self.frame.getSensingStart()

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
        return outFile


    def findInstrumentFile(self, sensingStart):

        datefmt = '%Y%m%d%H%M%S'
        if sensingStart is None:
            raise Exception('Image data not read in yet')

        outFile = None

        if self.instrumentDir in [None,'']:
            raise Exception('No Envisat Instrument File or Instrument Directory specified')

        try:
            for fname in os.listdir(self.instrumentDir):
                if not os.path.isfile(os.path.join(self.instrumentDir, fname)):
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
        return outFile









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
        self.fp.seek(65788, os.SEEK_CUR)
        for ii in range(7):
            gadsDict['nom_chirp_{0}'.format(ii)] = self._readAndUnpackData(36,">9f", numberOfFields=9)

        self.fp.seek(4096 - 36*7, os.SEEK_CUR) # Seek to record 66
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

    def parse(self):

        imageryDict = {}
        try:
            self.fp = open(self.fileName, 'rb')
        except IOError as errs:
            errno,strerr = errs
            print("IOError: %s %s" % (strerr,self.fileName))
            return

        self.readMPH()
        self.readSPH()
        mdsrDict = self.readMDSR()

        self.fp.close()

        imageryDict.update(self.mph)
        imageryDict.update(self.sph)
        imageryDict.update(mdsrDict)

        return imageryDict


    def readMDSR(self):
        """Unpack information from the Measurement Data Set Record (MDSR)"""
        headerLength = self.mphLength + self.sphLength
        self.fp.seek(headerLength)

        mdsrDict = {}

        # Front End Processor (FEP) Header
        mdsrDict['dsrTime'] = self._readAndUnpackData(length=12, format=">3L", numberOfFields=3)
        mdsrDict['groundStationReferenceTime'] = self._readAndUnpackData(length=12, format=">3L", numberOfFields=3)
        mdsrDict['ispLength'] = self._readAndUnpackData(length=2, format=">H", type=int)
        mdsrDict['crcErrors'] = self._readAndUnpackData(length=2, format=">H", type=int)
        mdsrDict['rcErrors'] = self._readAndUnpackData(length=2, format=">H", type=int)
        self.fp.seek(2, os.SEEK_CUR)

        mdsrDict['numberOfSamples'] = ((mdsrDict['ispLength']+1-30)/64)*63 + ((mdsrDict['ispLength']+1-30) % 64) -1

        # ISP Packet Header
        mdsrDict['packetIdentification'] = self._readAndUnpackData(length=2, format=">H", type=int)
        mdsrDict['packetSequenceControl'] = self._readAndUnpackData(length=2, format=">H", type=int)
        mdsrDict['packetLength'] = self._readAndUnpackData(length=2, format=">H", type=int)

        # Read entire packet header
        packetDataHeader = self._readAndUnpackData(length=30, format=">15H", numberOfFields=15)

        mdsrDict['dataFieldHeaderLength'] = packetDataHeader[0]
        mdsrDict['modeId'] = packetDataHeader[1]
        mdsrDict['onBoardTimeMSW'] = packetDataHeader[2]
        mdsrDict['onBoardTimeLSW'] = packetDataHeader[3]
        mdsrDict['onBoardTimeLSB'] = ((packetDataHeader[4] >> 8) & 255)
        mdsrDict['modePacketCount'] = (packetDataHeader[5]*256 + ((packetDataHeader[6] >> 8) & 256) )
        mdsrDict['antennaBeamSetNumber'] = ((packetDataHeader[6] >> 2) & 63)
        mdsrDict['compressionRatio'] = (packetDataHeader[6] & 3)
        mdsrDict['echoFlag'] = ((packetDataHeader[7] >> 15) & 1)
        mdsrDict['noiseFlag'] = ((packetDataHeader[7] >> 14) & 1)
        mdsrDict['calFlag']  = ((packetDataHeader[7] >> 13) & 1)
        mdsrDict['calType'] = ((packetDataHeader[7] >> 12) & 1)
        mdsrDict['cyclePacketCount'] = (packetDataHeader[7] & 4095)
        mdsrDict['priCodeword'] = packetDataHeader[8]
        mdsrDict['windowStartTimeCodeword'] = packetDataHeader[9]
        mdsrDict['windowLengthCodeword'] = packetDataHeader[10]
        mdsrDict['upConverterLevel'] = ((packetDataHeader[11] >> 12) & 15)
        mdsrDict['downConverterLevel'] = ((packetDataHeader[11] >> 7) & 31)
        mdsrDict['TxPolarization'] = ((packetDataHeader[11] >> 6) & 1)
        mdsrDict['RxPolarization'] = ((packetDataHeader[11] >> 5) & 1)
        mdsrDict['calibrationRowNumber'] = (packetDataHeader[11] & 31)
        mdsrDict['TxPulseLengthCodeword'] = ((packetDataHeader[12] >> 6) & 1023)
        mdsrDict['beamAdjustmentCodeword'] = (packetDataHeader[12] & 63)
        mdsrDict['chirpPulseBandwidthCodeword'] = ((packetDataHeader[13] >> 8) & 255)
        mdsrDict['auxillaryTxMonitorLevel'] = (packetDataHeader[13] & 255)
        mdsrDict['resamplingFactor'] = packetDataHeader[14]

        return mdsrDict


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
