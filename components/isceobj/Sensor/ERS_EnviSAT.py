#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Han Bao
# Copyright 2017
# ALL Rights RESERVED
#
# Modified from EnviSat.py originally written by Walter Szeliga
# and from make_raw_ers.pl written by Marie-Pierre Doin

import os
import copy
import math
import struct
import array
import string
import random
import logging
import datetime
import isceobj
from isceobj import *
from isceobj.Sensor import xmlPrefix
from isceobj.Scene.Track import Track
from isceobj.Scene.Frame import Frame
from isceobj.Planet.Planet import Planet
from isceobj.Orbit.Orbit import StateVector
from iscesys.Component.Component import Component
from isceobj.Planet.AstronomicalHandbook import Const
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTU


IMAGEFILE = Component.Parameter( 
    '_imageFileList',
    public_name='IMAGEFILE',
    default = '',
    container=list,
    type=str,
    mandatory=True,
    intent='input',
    doc="Input image file."
)

ORBIT_TYPE = Component.Parameter(
    '_orbitType',
    public_name='ORBIT_TYPE',
    default='',
    type=str,
    mandatory=True,
    doc="Options: ODR, PRC, PDS"
)

ORBIT_DIRECTORY = Component.Parameter(
    '_orbitDir',
    public_name='ORBIT_DIRECTORY',
    default='',
    type=str,
    mandatory=False,
    doc="Path to the directory containing the orbit files."
)

ORBIT_FILE = Component.Parameter(
    '_orbitFile',
    public_name='ORBIT_FILE',
    default='',
    type=str,
    mandatory=False,
    doc='Only used with PDS ORBIT_TYPE'
)

# Code to process ERS-1/2 Envisat-format SAR data. The code is wrote with two
# reference documents: PX-SP-50-9105 and ER-IS-EPO-GS-5902.1 from the Europe-
# an Space Agency The author also referred a ROI_PAC script 'make_raw_ers.pl'
# which is written by Marie-Pierre Doin. (Han Bao, 04/10/2019)

from .Sensor import Sensor
class ERS_EnviSAT(Sensor):

    parameter_list = (IMAGEFILE,
                      ORBIT_TYPE,
                      ORBIT_DIRECTORY,
                      ORBIT_FILE)     + Sensor.parameter_list
    """
        A Class for parsing ERS_EnviSAT_format image files
        There is no LEADERFILE file, which was required for old ERS [disk image] raw
        data, i.e. [ERS.py] sensor. There is also no INSTRUMENT file. All required
        default headers/parameters are stored in the image data file itself
    """

    family = 'ers_envisat'
    
    def __init__(self,family='',name=''):
        super(ERS_EnviSAT, self).__init__(family if family else  self.__class__.family, name=name)
        self.imageFile = None
        self._imageFileData = None
        self.logger = logging.getLogger("isce.sensor.ERA_EnviSAT")

        self.frame = None
        self.frameList = []

        # Constants are from the paper below:
        # J. J. Mohr and Soren. N. Madsen. Geometric calibration of ERS satellite
        # SAR images. IEEE T. Geosci. Remote, 39(4):842-850, Apr. 2001.
        self.constants = {'polarization': 'VV',
                          'antennaLength': 10,
                          'lookDirection': 'RIGHT',
                          'chirpPulseBandwidth': 15.50829e6,
                          'rangeSamplingRate': 18.962468e6,
                          'delayTime':6.622e-6,
                          'iBias': 15.5,
                          'qBias': 15.5,
                          'chirp': 0.419137466e12,
                          'waveLength': 0.0565646,
                          'pulseLength': 37.10e-06,
                          'SEC_PER_PRI_COUNT': 210.943006e-9,
                          'numberOfSamples': 11232
                          }
        return None

    def getFrame(self):
        return self.frame

    def populateMetadata(self):
        """Create the appropriate metadata objects from ERS Envisat-type metadata"""
        print("Debug Flag: start populate Metadata")
        self._populatePlatform()
        self._populateInstrument()
        self._populateFrame()
        if (self._orbitType == 'ODR'):
            self._populateDelftOrbits()
        elif (self._orbitType == 'PRC'):
            self._populatePRCOrbits()
        else:
            self.logger.error("ERROR: no orbit type (ODR or PRC")

    def _populatePlatform(self):
        """Populate the platform object with metadata"""
        platform = self.frame.getInstrument().getPlatform()

        platform.setMission("ERS" + str(self._imageFileData['PRODUCT'][61]))
        platform.setAntennaLength(self.constants['antennaLength'])
        platform.setPointingDirection(-1)
        platform.setPlanet(Planet(pname='Earth'))

    def _populateInstrument(self):
        """Populate the instrument object with metadata"""
        instrument = self.frame.getInstrument()
        pri = (self._imageFileData['pri_counter']+2.0) * self.constants['SEC_PER_PRI_COUNT']
        print("Debug: pri = ",pri)
        rangeSamplingRate = self.constants['rangeSamplingRate']
        rangePixelSize = Const.c/(2.0*rangeSamplingRate)
        prf = 1.0/pri
        self._imageFileData['prf']=prf

        instrument.setRadarWavelength(self.constants['waveLength'])
        # instrument.setIncidenceAngle(self.leaderFile.sceneHeaderRecord.metadata['Incidence angle at scene centre']) # comment out +HB, need to check
        instrument.setPulseRepetitionFrequency(prf)
        instrument.setRangeSamplingRate(rangeSamplingRate)
        instrument.setRangePixelSize(rangePixelSize)
        instrument.setPulseLength(self.constants['pulseLength'])
        instrument.setChirpSlope(self.constants['chirp'])
        instrument.setInPhaseValue(self.constants['iBias'])
        instrument.setQuadratureValue(self.constants['qBias'])
        print("Debug Flag Populate Instrument Done")

    def _populateFrame(self):
        """Populate the scene object with metadata"""
        numberOfLines = self._imageFileData['length'] # updated the value after findSwst and extractIQ
        numberOfSamples = self._imageFileData['width']# updated value after findSwst and extractIQ
        frame = 0  # ERS in Envisat format header does not contain frame number!
        pulseInterval = (self._imageFileData['pri_counter']+2.0) * self.constants['SEC_PER_PRI_COUNT']
        rangeSamplingRate = self.constants['rangeSamplingRate']
        rangePixelSize = Const.c/(2.0*rangeSamplingRate)
        startingRange = (9*pulseInterval + self._imageFileData['minSwst']*4/rangeSamplingRate-self.constants['delayTime'])*Const.c/2.0
        farRange = startingRange + self._imageFileData['width']*rangePixelSize

        first_line_utc = datetime.datetime.strptime(self._imageFileData['SENSING_START'], '%d-%b-%Y %H:%M:%S.%f')
        mid_line_utc = datetime.datetime.strptime(self._imageFileData['SENSING_START'], '%d-%b-%Y %H:%M:%S.%f')
        last_line_utc = datetime.datetime.strptime(self._imageFileData['SENSING_STOP'], '%d-%b-%Y %H:%M:%S.%f')
        centerTime = DTU.timeDeltaToSeconds(last_line_utc-first_line_utc)/2.0
        mid_line_utc = mid_line_utc + datetime.timedelta(microseconds=int(centerTime*1e6))
    
        print("Debug Print: Frame UTC start, mid, end times:  %s %s %s" % (first_line_utc,mid_line_utc,last_line_utc))

        self.frame.setFrameNumber(frame)
        self.frame.setProcessingFacility(self._imageFileData['PROC_CENTER'])
        self.frame.setProcessingSystem(self._imageFileData['SOFTWARE_VER'])
        self.frame.setTrackNumber(int(self._imageFileData['REL_ORBIT']))
        self.frame.setOrbitNumber(int(self._imageFileData['ABS_ORBIT']))
        self.frame.setPolarization(self.constants['polarization'])
        self.frame.setNumberOfSamples(numberOfSamples)
        self.frame.setNumberOfLines(numberOfLines)
        self.frame.setStartingRange(startingRange)
        self.frame.setFarRange(farRange)
        self.frame.setSensingStart(first_line_utc)
        self.frame.setSensingMid(mid_line_utc)
        self.frame.setSensingStop(last_line_utc)

    def _populateDelftOrbits(self):
        """Populate an orbit object with the Delft orbits"""
        from isceobj.Orbit.ODR import ODR, Arclist
        self.logger.info("Using Delft Orbits")
        arclist = Arclist(os.path.join(self._orbitDir,'arclist'))
        arclist.parse()
        orbitFile = arclist.getOrbitFile(self.frame.getSensingStart())
        self.logger.info('Using ODR file: ' + orbitFile)

        odr = ODR(file=os.path.join(self._orbitDir,orbitFile))
        # It seem that for this tipe of orbit points are separated by 60 sec. In ODR at 
        # least 9 state vectors are needed to compute the velocities. add extra time before
        # and after to allow interpolation, but do not do it for all data points. too slow
        startTimePreInterp = self.frame.getSensingStart() - datetime.timedelta(minutes=60)
        stopTimePreInterp = self.frame.getSensingStop() + datetime.timedelta(minutes=60)
        odr.parseHeader(startTimePreInterp,stopTimePreInterp)
        startTime = self.frame.getSensingStart() - datetime.timedelta(minutes=5)
        stopTime = self.frame.getSensingStop() + datetime.timedelta(minutes=5)
        self.logger.debug("Extracting orbits between %s and %s" % (startTime,stopTime))
        orbit = odr.trimOrbit(startTime,stopTime)
        self.frame.setOrbit(orbit)
        print("Debug populate Delft Orbits Done")
        print(startTime,stopTime)

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

    def _populateImage(self,outname):
        width  = self._imageFileData['width']

        # Create a RawImage object
        rawImage = isceobj.createRawImage()
        rawImage.setFilename(outname)
        rawImage.setAccessMode('read')
        rawImage.setByteOrder('l')
        rawImage.setXmin(0)
        rawImage.setXmax(width)
        rawImage.setWidth(width)
        self.frame.setImage(rawImage)

    def extractImage(self):
        import array
        import math

        self.frameList = []

        for i in range(len(self._imageFileList)):
            appendStr = "_" + str(i) #intermediate raw files suffix
            if(len(self._imageFileList) == 1):
                appendStr = '' #if only one file don't change the name
            
            outputNow = self.output + appendStr 
            auxImage = isceobj.createImage()    # unused
            widthAux = 2                        # unused
            auxName = outputNow + '.aux'

            self.imageFile = self._imageFileList[i]
            self.frame = Frame()
            self.frame.configure()

            self.frame.auxFile = auxName #add the auxFile as part of the frame and diring the stitching create also a combined aux file # HB: added from Envisat.py
            imageFileParser = ImageFile(fileName=self.imageFile)
            self._imageFileData = imageFileParser.parse() # parse image and get swst values and new width
            
            try:
                outputNow = self.output + appendStr
                out = open(outputNow,'wb')
            except IOError as strerr:
                self.logger.error("IOError: %s" % strerr)
                return

            imageFileParser.extractIQ(output=out) # IMPORTANT for ERS Envisat-type data
            out.close()

            self.populateMetadata() # populate Platform, Instrument, Frame, and Orbit
            self._populateImage(outputNow)
            self.frameList.append(self.frame)

            ### Below: create a aux file
            # for now create the orbit aux file based in starting time and prf
            prf = self.frame.getInstrument().getPulseRepetitionFrequency()
            senStart = self.frame.getSensingStart()
            numPulses = int(math.ceil(DTU.timeDeltaToSeconds(self.frame.getSensingStop()-senStart)*prf))
            # the aux files has two entries per line. day of the year and microseconds in the day
            musec0 = (senStart.hour*3600 + senStart.minute*60 + senStart.second)*10**6 + senStart.microsecond
            maxMusec = (24*3600)*10**6 # use it to check if we went across a day. very rare
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

        ## refactor this with __init__.tkfunc
        tk = Track()
        if(len(self._imageFileList) > 1):
            self.frame = tk.combineFrames(self.output,self.frameList)

            for i in range(len(self._imageFileList)):
                try:
                    os.remove(self.output + "_" + str(i))
                except OSError:
                    print("Error. Cannot remove temporary file",self.output + "_" + str(i))
                    raise OSError


class BaseErsEnvisatFile(object):  # from Envisat.py
    """Class for parsing common Envisat metadata"""

    def __init__(self):
        self.fp = None
        self.width = 11498; # width is the total length of the Level 0 MDSR structure. See Table 4-5 document PX-SP-50-9105
        self.xmin = 266;    # xmin is the length before 'Measurement Data'. See Table 4-5 4-6 4-7 of document PX-SP-50-9105
        self.xmax = 11498;  # xmax is the same as width for now
        self.lineCounterOffset=54 
        self.SWSToffset=58; 
        self.PRIoffset=60;  
        self.annotationsLength = 32 # Level_0_ProcessorAnnotationLength(12 bytes) + FEP_AnnotationLength(20 bytes)
        self.mphLength = 1247   # total length of the Main Product Header
        self.sphLength = 1956   # total length of the Specific Product Header
        self.MPHpSPH   = 3203   # Length of MPH (1247 bytes) plus SPH (1956 bytes)
        self.mph = {}
        self.sph = {}
        self.mdsr ={}
        self.tmpData = {}

    def readMPH(self):
        """
        Unpack the Main Product Header (MPH). MPH identifies the product
        and its main characteristics. The Main Product Header is an ASCII
        structure containing information needed for all ENVISAT sensors. 
        It is of fixed length and format for all products.
        """
        mphString = self.fp.read(self.mphLength)
        header = mphString.splitlines()
        for line in header:
            (key, sep, value) = line.decode('utf8').partition('=')
            if (key.isspace() == False):
                value = str.replace(value,'"','')
                value = str.strip(value)
                self.mph[key] = value

    def readSPH(self):
        """
        Unpack the Specific Product Header (SPH). SPH is included with every product.
        It contains information specific to the product itself. This information may
        include Product Confidence Data (PCD) information applying to the whole
        product, and/or relevant processing parameters. At a minimum, each SPH includes
        an SPH descriptor, and at least one Data Set Descriptor (DSD).
        """
        self.fp.seek(self.mphLength)            # Skip the 1st section--MPH
        sphString = self.fp.read(self.sphLength)
        header = sphString.splitlines()

        dsSeen = False
        dataSet = {}
        dataSets = []
        # the Specific Product Header is made of up key-value pairs. At the end of the
        # header, there are a number of data blocks that represent the data sets (DS)
        # that follow.  Since their key names are not unique, we need to capture them
        # in an array and then tack this array on the dictionary later.  These data 
        # sets begin with a key named "DS_NAME" and ends with a key name "DSR_SIZE".
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

    def readMDSR(self):
        """Unpack information from the Measurement Data Set Record (MDSR)"""
        self.fp.seek(self.MPHpSPH) # skip the 1st (MPH) & 2nd (SPH) sections
        # MDS is the 3rd section, after the 1st MPH and the 2nd SPH

        # [1] Level 0 Processor Annotation (12 bytes)
        self.mdsr['dsrTime'] = self._readAndUnpackData(length=12, format=">3L", numberOfFields=3)
        
        # [2] Front End Processor (FEP) Header
        #     For ERS-Envisat_format data, (1)gsrt is set to FFFF..., (2)ispLength = [(length of source packet) - 1]
        #     (3)crcErrors, (4)rsErrors, and (5)spare should be blank
        self.fp.seek(12, os.SEEK_CUR) # skip the (1)gsrt which is set to zero for ERS_Envisat data
        self.mdsr['ispLength'] = self._readAndUnpackData(length=2, format=">H", type=int)
        self.fp.seek(2, os.SEEK_CUR)  # skip the (3)crcErrors which is set to zero for ERS_Envisat data
        self.fp.seek(2, os.SEEK_CUR)  # skip the (4)rsErrors which is set to zero for ERS_Envisat data
        self.fp.seek(2, os.SEEK_CUR)  # skip the (5)spare which is set to zero for ERS_Envisat data

        self.mdsr['numberOfSamples'] = 11232 # Hardcoded, from document PX-SP-50-9105
        # [3] Unlike Envisat data, there is no ISP Packet Header for ERS_Envisat data
        # [4] Unlike Envisat data, there is no Packet Data Field Header for ERS_Envisat data

        # [5] Find reference counter, pulseRepetitionInterval (pri) and pulseRepetitionFreq (prf)        
        self.fp.seek(self.width+self.SWSToffset-self.annotationsLength, os.SEEK_CUR) # Seek to swst_counter
        self.mdsr['swst_counter'] = self._readAndUnpackData(length=2, format=">H", type=int) # 2-byte integer
        self.mdsr['pri_counter']  = self._readAndUnpackData(length=2, format=">H", type=int) # 2-byte integer
        if(self.mdsr['swst_counter'] > 1100):
            print("Error reading swst counter read: $swst_counter\n")
            print("Affecting to swst counter the value 878 !\n")
            self.mdsr['swst_counter'] = 878
        print("swst counter read: %s" % self.mdsr['swst_counter'])
        self.mdsr['reference_counter'] = self.mdsr['swst_counter']


    def _findSWST(self):
        """
            Run through the data file once, and calculate the valid sampling window start time range.
        """
        for dataSet in self.sph['dataSets']:
            if (dataSet['DS_NAME'] == 'SAR_SOURCE_PACKETS'): 
                lines = int(dataSet['NUM_DSR'])    # Number of SAR DATA records, i.e. lines
        pixelCount = int(self.mdsr['numberOfSamples']/2)
        self.length = lines
        expectedFileSize = self.MPHpSPH + self.width * lines
        self.fp.seek(0,os.SEEK_END)
        actualSize = self.fp.tell()
        if (expectedFileSize != actualSize):
            raise Exception('Error! File too short.  Expected %s bytes, found %s bytes' % (expectedFileSize,actualSize)) # Checked

        print("Debug Flag: Start findSwst for each line")
        lastLineCounter = None
        lineGap = 0
        # Calculate the minimum and maximum Sampling Window Start Times (swst)
        swst = [0]*lines
        lineCounter = [0]*lines
        self.fp.seek(self.MPHpSPH) # skip the MPH and SPH header
        for line in range(lines):
            self.fp.seek(self.lineCounterOffset, os.SEEK_CUR) # seek to the Data Record Number (lineCounter)
            lineCounter[line] = self._readAndUnpackData(length=4, format=">I", type=int)
            if (line<10):
                print("Debug Print: lineCounter is : ", lineCounter[line])
            if (not lastLineCounter):
                lastLineCounter = lineCounter[line]
            else:
                lineGap = lineCounter[line] - lastLineCounter-1
                lastLineCounter = lineCounter[line]
            if (lineGap != 0):
                self.length += lineGap

            self.fp.seek(self.SWSToffset-self.lineCounterOffset-4, os.SEEK_CUR) # skip to swst in the current record/line
            swst[line] = self._readAndUnpackData(length=2, format=">H", type=int)
            self.fp.seek(self.xmin-self.SWSToffset-2,os.SEEK_CUR)
            self.fp.seek(2*pixelCount,os.SEEK_CUR)
            if ((line+1)%20000==0):
                print("Checking 'Line Number': %i ; and 'swst': %i " % (lineCounter[line], swst[line]))
        s = swst[:]
        for val in swst:
            if ((val<500) or (val>1500) or ((val-swst[0])%22 != 0)):
                s.remove(val)

        self.tmpData['swst']=swst
        self.tmpData['lineCounter']=lineCounter
        self.mdsr['minSwst'] = min(s)
        self.mdsr['maxSwst'] = max(s)
        pad = (self.mdsr['maxSwst'] - self.mdsr['minSwst'])*8
        self.width = 2*pixelCount + pad # update width to accommendate records with different swst, no more header for each line in output raw image
        self.mdsr['width'] = self.width # update the width
        self.mdsr['length'] = self.length

    def extractIQ(self,output=None):
        """
            Checking lines, taking care of delay shift of swst and Extract 
            the I and Q channels from the image file
        """
        for dataSet in self.sph['dataSets']:
            if (dataSet['DS_NAME'] == 'SAR_SOURCE_PACKETS'): 
                lines = int(dataSet['NUM_DSR'])     # Number of SAR DATA records
        pixelCount = int(self.mdsr['numberOfSamples']/2)

        self.fp.seek(self.MPHpSPH)
        lastSwst = 0
        lastLineCounter = 0
        lineGap = 0
        # Extract the I and Q channels
        IQLine = array.array('B',[random.randint(15,16)*x for x in [1]*self.width]) 
        IQ = array.array('B',[x for x in [0]*self.width])

        for line in range(lines):
            if ((line+1)%10000 == 0):
                print("Extracting line %s" % (line+1) )
            
            # Find missing range values
            currentSwst = self.tmpData['swst'][line]
            if ((currentSwst>500) and (currentSwst<1500) and (currentSwst-self.mdsr['minSwst'])%22 == 0):
                lastSwst = currentSwst
            leftPad = (lastSwst - self.mdsr['minSwst'])*8
            rightPad = self.width - leftPad - 2*pixelCount

            # Find missing lines
            lineCounter = self.tmpData['lineCounter'][line]
            if (lineCounter == 0):
                print("WARNING! Zero line counter at line %s" % (line+1))
                lastLineCounter += 1
                continue

            # Initialize the line counter
            if (line == 0):
                lastLineCounter = lineCounter-1

            lineGap = lineCounter - lastLineCounter-1

            skipLine = False
            if (lineGap > 0):
                if (lineGap > 30000):
                    print("Bad Line Counter on line %s, Gap length too large (%s)" % (line+1,lineGap))
                    self.fp.seek((2*pixelCount),os.SEEK_CUR)
                    lastLineCounter += 1
                    continue
                print("Gap of length %s at line %s" % (lineGap,(line+1)))

                IQ = array.array('B',[IQLine[i] for i in range(self.width)])
                for i in range(lineGap):
                    IQ.tofile(output) # It may be better to fill missing lines with random 15's and 16's rather than copying the last good line
                    lastLineCounter += 1
            elif (lineGap == -1):
                skipLine = True
            elif (lineGap < 0):
                print("WARNING! Unusual Line Gap %s at line %s" % (lineGap,(line+1)))
            
            # Pad data with random integers around the I and Q bias of 15.5 on the left
            IQ = array.array('B',[IQLine[i] for i in range(leftPad)])

            # Read the I and Q values
            self.fp.seek(self.xmin,os.SEEK_CUR) # skip the header of each line (266 bytes)
            IQ.fromfile(self.fp,2*pixelCount) # get all sample data

            # Now pad data with random integers around the I and Q bias of 15.5 on the right
            IQ.extend([IQLine[i] for i in range(rightPad)])

            # Output the padded line
            if not skipLine:
                IQ.tofile(output)
                lastLineCounter += 1
        self.fp.close()

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


# class ImageFile(object):
class ImageFile(BaseErsEnvisatFile):
    """Parse an ERS-Envisat_format Imagery File"""

    def __init__(self, fileName=None):
        BaseErsEnvisatFile.__init__(self)
        self.fileName = fileName
        self.logger = logging.getLogger("isce.sensor.ERA_EnviSAT")
    
    def parse(self):
        imageDict = {}
        try:
            self.fp = open(self.fileName, 'rb')
        except IOError as errs:
            errno,strerr = errs
            print("IOError: %s %s" % (strerr,self.fileName))
            return

        self.readMPH()
        self.readSPH()
        self.readMDSR()
        self._findSWST()

        imageDict.update(self.mph)
        imageDict.update(self.sph)
        imageDict.update(self.mdsr)
        imageDict.update(self.tmpData)

        return imageDict
