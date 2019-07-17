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
from . import CEOS
import isceobj
from isceobj.Scene.Frame import Frame
from isceobj.Orbit.Orbit import Orbit
from isceobj.Orbit.Orbit import StateVector as OrbitStateVector
from isceobj.Attitude.Attitude import Attitude
from isceobj.Attitude.Attitude import StateVector as AttitudeStateVector
from isceobj.Constants import SPEED_OF_LIGHT
from isceobj.Planet.Planet import Planet
from isceobj.Sensor import alos, VolumeDirectoryBase, tkfunc, Constants
from isceobj.Sensor.Polarimetry import Distortion
from iscesys import DateTimeUtil as DTUtil
from iscesys.Component.Component import Component
from stdproc.alosreformat.ALOS_fbs2fbd.ALOS_fbs2fbdPy import ALOS_fbs2fbdPy
from stdproc.alosreformat.ALOS_fbd2fbs.ALOS_fbd2fbsPy import ALOS_fbd2fbsPy
from isceobj.Util.decorators import pickled, logged
from isceobj.Sensor import xmlPrefix
from isceobj.Util.decorators import use_api

# temporary comments: replaced a static dictionary with a class --convention
# precluded namedtuple (constants).
# Need to discuss how the _populate methods work.
# 2 state vectors?

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

RESAMPLE_FLAG = Component.Parameter('_resampleFlag',
    public_name='RESAMPLE_FLAG',
    default='',
    type=str,
    mandatory=False,
    doc="""
        Indicate whether to resample: empty string indicates no resample;
        'single2dual' indicates resample single polarized data to dual
            polarized data sample rate;
        'dual2single' indicates resample dual polarized data to single
            polarize data sample rate.
        """
)

from .Sensor import Sensor

@pickled
class ALOS(Sensor):
    """Code to read CEOSFormat leader files for ALOS SAR data.
    The tables used to create this parser are based on document number
    ER-IS-EPO-GS-5902.1 from the European
    Space Agency.
    """

    family = 'alos'
    logging_name = 'isce.sensor.ALOS'

    parameter_list = (IMAGEFILE,
                      LEADERFILE,
                      RESAMPLE_FLAG) + Sensor.parameter_list

#    polarizationMap = ['H','V','H+V']

    ## This is manifestly better than a method complete the lazy instantation
    ## of an instance attribute
    transmit = Distortion(complex(2.427029e-3,1.293019e-2),
                          complex(-1.147240e-2,-6.228230e-3),
                          complex(9.572169e-1,3.829563e-1))
    receive = Distortion(complex(-6.263392e-3,7.082863e-3),
                         complex(-6.297074e-3,8.026685e-3),
                         complex(7.217117e-1,-2.367683e-2))

    constants = Constants(iBias=15.5,
                          qBias=15.5,
                          pointingDirection=-1,
                          antennaLength=8.9)


    HEADER_LINES = 720

    RESAMPLE_FLAG = {'':'do Nothing',
                     'single2dual' : 'resample from single to dual pole',
                     'dual2single' : 'resample from dual to single'}

    @logged
    def __init__(self, name=''):
        super().__init__(family=self.__class__.family, name=name)
        self._leaderFile = None
        self._imageFile = None
        self.frame =  None
        return None

    #2013-06-03 Kosal: the functions below overwrite the transmit property
    #initiated above
    '''
    @property
    def transmit(self):
        return self.__class__.transmit
    @transmit.setter
    def transmit(self, x):
        raise TypeError(
            "ALOS.transmit is a protected class attribute and cannot be set"
            )
    @property
    def receive(self):
        return self.__class__.receive
    @receive.setter
    def receive(self, x):
        raise TypeError(
            "ALOS.receive is a protected class attribute and cannot be set"
            )
    '''
    #Kosal

    def getFrame(self):
        return self.frame

    def setLeaderFile(self,ldr):
        self._leaderFile = ldr
        return

    def parse(self):
        self.leaderFile = LeaderFile(file=self._leaderFile)
        self.imageFile = ImageFile(self)
        try:
            self.leaderFile.parse()
            self.imageFile.parse()
        except IOError:
            return
        self.populateMetadata()

    def populateMetadata(self):
        """
        Create the appropriate metadata objects from our CEOSFormat metadata
        """
        self._populatePlatform()
        self._populateInstrument()
        self._populateFrame()
        # Header orbits
        self._populateOrbit()
        self._populateAttitude()
        self._populateDistortions()

        productLevel = float(self.leaderFile.sceneHeaderRecord.metadata[
            'Product level code'])
        if productLevel == 1.0:
            self.updateRawParameters()
        pass

    def _populatePlatform(self):
        platform = self.frame.getInstrument().getPlatform()

        platform.setMission(self.leaderFile.sceneHeaderRecord.metadata[
            'Sensor platform mission identifier'])
        platform.setPointingDirection(self.constants.pointing_direction)
        platform.setAntennaLength(self.constants.antenna_length)
        platform.setPlanet(Planet(pname='Earth'))

    def _populateInstrument(self):
        instrument = self.frame.getInstrument()
        rangePixelSize = None
        rangeSamplingRate = None
        chirpSlope = None
        bandwidth = None
        prf = None
        try:
            rangeSamplingRate = self.leaderFile.sceneHeaderRecord.metadata[
                'Range sampling rate'
                ]*1e6
            pulseLength = self.leaderFile.sceneHeaderRecord.metadata[
                'Range pulse length'
                ]*1e-6
            rangePixelSize = SPEED_OF_LIGHT/(2.0*rangeSamplingRate)
            prf = self.leaderFile.sceneHeaderRecord.metadata[
                'Pulse Repetition Frequency']/1000.

            ###Fix for quad pol data
            if prf > 3000:
                prf = prf / 2.0

            print('LEADER PRF: ', prf)
            beamNumber = self.leaderFile.sceneHeaderRecord.metadata[
                'Antenna beam number']
#            if self.imageFile.prf:
#                prf = self.imageFile.prf
#            else:
#                self.logger.info("Using nominal PRF")
            bandwidth = self.leaderFile.calibrationRecord.metadata[
                'Band width']*1e6
            #if (not bandwidth):
            #    bandwidth = self.leaderFile.sceneHeaderRecord.metadata[
            #    'Bandwidth per look in range']
            chirpSlope = -(bandwidth/pulseLength)
        except AttributeError:
            self.logger.info("Some of the instrument parameters were not set")

        self.logger.debug("PRF: %s" % prf)
        self.logger.debug("Bandwidth: %s" % bandwidth)
        self.logger.debug("Pulse Length: %s" % pulseLength)
        self.logger.debug("Chirp Slope: %s" % chirpSlope)
        self.logger.debug("Range Pixel Size: %s" % rangePixelSize)
        self.logger.debug("Range Sampling Rate: %s" % rangeSamplingRate)
        self.logger.debug("Beam Number: %s" % beamNumber)
        instrument.setRadarWavelength(
            self.leaderFile.sceneHeaderRecord.metadata['Radar wavelength']
            )
        instrument.setIncidenceAngle(
            self.leaderFile.sceneHeaderRecord.metadata[
                'Incidence angle at scene centre']
            )
        instrument.setPulseRepetitionFrequency(prf)
        instrument.setRangePixelSize(rangePixelSize)
        instrument.setRangeSamplingRate(rangeSamplingRate)
        instrument.setPulseLength(pulseLength)
        instrument.setChirpSlope(chirpSlope)
        instrument.setInPhaseValue(self.constants['iBias'])
        instrument.setQuadratureValue(self.constants['qBias'])
        instrument.setBeamNumber(beamNumber)
        return None

    def _populateFrame(self, polarization='HH', farRange=None):
        frame = self._decodeSceneReferenceNumber(
            self.leaderFile.sceneHeaderRecord.metadata['Scene reference number']
            )

        try:
            first_line_utc = self.imageFile.start_time
            last_line_utc = self.imageFile.stop_time
            centerTime = DTUtil.timeDeltaToSeconds(
                last_line_utc-first_line_utc
                )/2.0
            center_line_utc = first_line_utc + datetime.timedelta(
                microseconds=int(centerTime*1e6)
                )
            self.frame.setSensingStart(first_line_utc)
            self.frame.setSensingMid(center_line_utc)
            self.frame.setSensingStop(last_line_utc)
            rangePixelSize = self.frame.getInstrument().getRangePixelSize()
            farRange = (
                self.imageFile.startingRange +
                self.imageFile.width*rangePixelSize
                )
        except TypeError as strerr:
            self.logger.warn(strerr)

        self.frame.frameNumber = frame
        self.frame.setOrbitNumber(
            self.leaderFile.sceneHeaderRecord.metadata['Orbit number']
            )
        self.frame.setStartingRange(self.imageFile.startingRange)
        self.frame.setFarRange(farRange)
        self.frame.setProcessingFacility(
            self.leaderFile.sceneHeaderRecord.metadata[
            'Processing facility identifier'])
        self.frame.setProcessingSystem(
            self.leaderFile.sceneHeaderRecord.metadata[
            'Processing system identifier'])
        self.frame.setProcessingSoftwareVersion(
            self.leaderFile.sceneHeaderRecord.metadata[
            'Processing version identifier'])
        self.frame.setPolarization(polarization)
        self.frame.setNumberOfLines(self.imageFile.length)
        self.frame.setNumberOfSamples(self.imageFile.width)

    def _populateOrbit(self):
        orbit = self.frame.getOrbit()
        velocityScale = 1.0
        if (self.leaderFile.sceneHeaderRecord.metadata[
            'Processing facility identifier'] == 'ERSDAC'):
            # The ERSDAC header orbits are in mm/s
            velocityScale = 1000.0

        orbit.setReferenceFrame(
            self.leaderFile.platformPositionRecord.metadata[
            'Reference coordinate system'])
        orbit.setOrbitSource('Header')
        orbitQuality = self._decodeOrbitQuality(
            self.leaderFile.platformPositionRecord.metadata[
            'Orbital elements designator'])
        orbit.setOrbitQuality(orbitQuality)

        t0 = datetime.datetime(
            year=self.leaderFile.platformPositionRecord.metadata[
                 'Year of data point'],
            month=self.leaderFile.platformPositionRecord.metadata[
                 'Month of data point'],
            day=self.leaderFile.platformPositionRecord.metadata[
                 'Day of data point'])
        t0 = t0 + datetime.timedelta(seconds=
            self.leaderFile.platformPositionRecord.metadata['Seconds of day'])
        for i in range(
            self.leaderFile.platformPositionRecord.metadata[
            'Number of data points']):
            vec = OrbitStateVector()
            t = t0 + datetime.timedelta(seconds=
                i*self.leaderFile.platformPositionRecord.metadata[
                'Time interval between DATA points'])
            vec.setTime(t)
            dataPoints = self.leaderFile.platformPositionRecord.metadata[
                'Positional Data Points'][i]
            vec.setPosition([
                dataPoints['Position vector X'],
                dataPoints['Position vector Y'],
                dataPoints['Position vector Z']])
            vec.setVelocity([
                dataPoints['Velocity vector X']/velocityScale,
                dataPoints['Velocity vector Y']/velocityScale,
                dataPoints['Velocity vector Z']/velocityScale])
            orbit.addStateVector(vec)

    def _populateAttitude(self):
        if (self.leaderFile.leaderFDR.metadata[
            'Number of attitude data records'] != 1):
            return

        attitude = self.frame.getAttitude()
        attitude.setAttitudeSource("Header")

        year = int(self.leaderFile.sceneHeaderRecord.metadata[
            'Scene centre time'][0:4])
        t0 = datetime.datetime(year=year,month=1,day=1)

        for i in range(self.leaderFile.platformAttitudeRecord.metadata[
            'Number of attitude data points']):
            vec = AttitudeStateVector()

            dataPoints = self.leaderFile.platformAttitudeRecord.metadata[
                'Attitude Data Points'][i]
            t = t0 + datetime.timedelta(
                         days=(dataPoints['Day of the year']-1),
                         milliseconds=dataPoints['Millisecond of day'])
            vec.setTime(t)
            vec.setPitch(dataPoints['Pitch'])
            vec.setRoll(dataPoints['Roll'])
            vec.setYaw(dataPoints['Yaw'])
            attitude.addStateVector(vec)

    def _populateDistortions(self):
        return None


    def readOrbitPulse(self, leader, raw, width):
        '''
        No longer used. Can't rely on raw data headers. Should be done as part of extract Image.
        '''

        from isceobj.Sensor import readOrbitPulse as ROP
        print('TTTT')
        rawImage = isceobj.createRawImage()
        leaImage = isceobj.createStreamImage()
        auxImage = isceobj.createImage()
        rawImage.initImage(raw,'read',width)
        rawImage.renderVRT()
        rawImage.createImage()
        rawAccessor = rawImage.getImagePointer()
        leaImage.initImage(leader,'read')
        leaImage.createImage()
        leaAccessor = leaImage.getImagePointer()
        widthAux = 2
        auxName = raw + '.aux'
        self.frame.auxFile = auxName
        auxImage.initImage(auxName,'write',widthAux,type = 'DOUBLE')
        auxImage.createImage()
        auxAccessor = auxImage.getImagePointer()
        length = rawImage.getLength()
        ROP.setNumberBitesPerLine_Py(width)
        ROP.setNumberLines_Py(length)
        ROP.readOrbitPulse_Py(leaAccessor,rawAccessor,auxAccessor)
        rawImage.finalizeImage()
        leaImage.finalizeImage()
        auxImage.finalizeImage()
        return None

    def makeFakeAux(self, outputNow):
        '''
        Generate an aux file based on sensing start and prf.
        '''
        import math, array

        prf = self.frame.getInstrument().getPulseRepetitionFrequency()
        senStart = self.frame.getSensingStart()
        numPulses = self.frame.numberOfLines
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


    ## Can this even be done/
    ## should the pointer be an __Int__?
    def readOrbitPulseDevelopement(self, leader, raw, width):
        from isceobj.Sensor import readOrbitPulse as ROP
        with isceobj.contextRawImage(width=width, accessMode='read',) as rawImage:
            with isceobj.contextStreamImage(width=width,accessMode='read', ) as leaImage:
                with isceobj.contextImage(width=width, accessMode='write', ) as auxImage:

                    rawAccessor = rawImage.getImagePointer()
                    leaAccessor = leaImage.getImagePointer()
                    widthAux = 2
                    auxName = raw + '.aux'
                    self.frame.auxFile = auxName
                    auxImage.initImage(auxName, 'write', widthAux,
                        type = 'DOUBLE')
                    auxImage.createImage()
                    auxAccessor = auxImage.getImagePointer()
                    length = rawImage.getLength()
                    ROP.setNumberBitesPerLine_Py(width)
                    ROP.setNumberLines_Py(length)
                    ROP.readOrbitPulse_Py(leaAccessor,rawAccessor,auxAccessor)
                    pass #rawImage.finalizeImage()
                pass #leaImage.finalizeImage()
            pass #auxImage.finalizeImage()
        return None

    def extractImage(self):
        if(len(self._imageFileList) != len(self._leaderFileList)):
            self.logger.error(
                "Number of leader files different from number of image files.")
            raise RuntimeError
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
            self.imageFile = ImageFile(self)

            try:
                self.leaderFile.parse()
                self.imageFile.parse(calculateRawDimensions=False)
                outputNow = self.output + appendStr
                if not (self._resampleFlag == ''):
                    filein = self.output + '__tmp__'
                    self.imageFile.extractImage(filein)
                    self.populateMetadata()
                    objResample = None
                    if(self._resampleFlag == 'single2dual'):
                        objResample = ALOS_fbs2fbdPy()
                    else:
                        objResample = ALOS_fbd2fbsPy()
                    objResample.wireInputPort('frame',object = self.frame)
                    objResample.setInputFilename(filein)
                    objResample.setOutputFilename(outputNow)
                    objResample.run()
                    objResample.updateFrame(self.frame)
                    os.remove(filein)
                else:
                    self.imageFile.extractImage(outputNow)
                    self.populateMetadata()
                width = self.frame.getImage().getWidth()
#                self.readOrbitPulse(self._leaderFile,outputNow,width)
                self.makeFakeAux(outputNow)
                self.frameList.append(self.frame)
            except IOError:
                return
            pass
        ## refactor this with __init__.tkfunc
        return tkfunc(self)

    def _decodeSceneReferenceNumber(self, referenceNumber):
        return referenceNumber

    def _decodeOrbitQuality(self,quality):
        try:
            quality = int(quality)
        except ValueError:
            quality = None

        qualityString = ''
        if (quality == 0):
            qualityString = 'Preliminary'
        elif (quality == 1):
            qualityString = 'Decision'
        elif (quality == 2):
            qualityString = 'High Precision'
        else:
            qualityString = 'Unknown'

        return qualityString


    def updateRawParameters(self):
        '''
        Parse the data in python.
        '''
        with open(self._imageFile,'rb') as fp:
            width = self.imageFile.width
            numberOfLines = self.imageFile.length
            prefix = self.imageFile.prefix
            suffix = self.imageFile.suffix
            dataSize = self.imageFile.dataSize

            fp.seek(720, os.SEEK_SET) # Skip the header
            tags = []

            print('WIDTH: ', width)
            print('LENGTH: ', numberOfLines)
            print('PREFIX: ', prefix)
            print('SUFFIX: ', suffix)
            print('DATASIZE: ', dataSize)

            for i in range(numberOfLines):
                if not i%1000: self.logger.info("Line %s" % i)
                imageRecord = CEOS.CEOSDB(
                            xml = os.path.join(xmlPrefix,'alos/image_record.xml'),
                            dataFile=fp)
                imageRecord.parse()

                tags.append(float(imageRecord.metadata[
                            'Sensor acquisition milliseconds of day']))
                data = fp.read(dataSize)
                pass
        ###Do parameter fit
        import numpy as np


        tarr = np.array(tags) - tags[0]
        ref = np.arange(tarr.size) / self.frame.PRF
        print('PRF: ', self.frame.PRF)
        ####Check every 20 microsecs
        off = np.arange(50)*2.0e-5
        res = np.zeros(off.size)

        ###Check which offset produces the same millisec truncation
        ###Assumes PRF is correct
        for xx in range(off.size):
            ttrunc = np.floor((ref+off[xx])*1000)
            res[xx] = np.sum(tarr-ttrunc)

        res = np.abs(res)
        
#        import matplotlib.pyplot as plt
#        plt.plot(res)
#        plt.show()


        delta = datetime.timedelta(seconds=np.argmin(res)*2.0e-5)
        print('TIME OFFSET: ', delta)
        self.frame.sensingStart += delta
        self.frame.sensingMid += delta
        self.frame.sensingStop += delta
        return None




class LeaderFile(object):

    def __init__(self,file=None):
        self.file = file
        self.leaderFDR = None
        self.sceneHeaderRecord = None
        self.platformPositionRecord = None
        self.platformAttitudeRecord = None
        self.calibrationRecord = None
        return None

    def parse(self):
        """Parse the leader file to create a header object"""
        try:
            with open(self.file,'rb') as fp:
                # Leader record
                self.leaderFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,
                    'alos', 'leader_file.xml'),dataFile=fp)
                self.leaderFDR.parse()
                fp.seek(self.leaderFDR.getEndOfRecordPosition())
                # Scene Header, called the "Data Set Summary Record" by JAXA
                if (self.leaderFDR.metadata[
                    'Number of data set summary records'] == 1):
                    self.sceneHeaderRecord = CEOS.CEOSDB(
                        xml=os.path.join(xmlPrefix,'alos', 'scene_record.xml'),
                        dataFile=fp)
                    self.sceneHeaderRecord.parse()
                    fp.seek(self.sceneHeaderRecord.getEndOfRecordPosition())
                    pass
                # Platform Position
                if (self.leaderFDR.metadata[
                    'Number of platform pos. data records'] == 1):
                    self.platformPositionRecord = CEOS.CEOSDB(
                        xml=os.path.join(xmlPrefix,
                        'alos/platform_position_record.xml'),dataFile=fp)
                    self.platformPositionRecord.parse()
                    fp.seek(
                        self.platformPositionRecord.getEndOfRecordPosition())
                    pass
                # Spacecraft Attitude
                if (self.leaderFDR.metadata[
                    'Number of attitude data records'] == 1):
                    self.platformAttitudeRecord = CEOS.CEOSDB(
                    xml=os.path.join(xmlPrefix,'alos/attitude_record.xml'),
                    dataFile=fp)
                    self.platformAttitudeRecord.parse()
                    fp.seek(
                        self.platformAttitudeRecord.getEndOfRecordPosition())
                    pass
                # Spacecraft calibration
                if (self.leaderFDR.metadata[
                    'Number of calibration records'] == 1):
                    self.calibrationRecord = CEOS.CEOSDB(
                        xml=os.path.join(xmlPrefix,
                            'alos/calibration_record.xml'),dataFile=fp)
                    self.calibrationRecord.parse()
                    fp.seek(self.calibrationRecord.getEndOfRecordPosition())
                    pass
                pass
            pass
        except IOError as errs:
            strerr = errs.strerror
            print("IOError: %s" % strerr)

        return None

    pass

class VolumeDirectoryFile(VolumeDirectoryBase):
    volume_fdr_arg = os.path.join('alos', 'volume_descriptor.xml')
    pass

class ImageFile(object):

    def __init__(self, parent):
        self.parent = parent
        self.length = None
        self.width = None
        self.start_time = None
        self.stop_time = None
        self.startingRange = None
        self.imageFDR = None
        self.numberOfSarChannels = None
        self.prf = None
        self.prefix=None
        self.suffix=None
        self.dataSize = None
        return None

    def parse(self, calculateRawDimensions=True):
        try:
            with open(self.parent._imageFile, 'rb') as fp:
            # Image Header
                self.imageFDR = CEOS.CEOSDB(
                    xml=os.path.join(xmlPrefix,'alos','image_file.xml'),
                    dataFile=fp)
                self.imageFDR.parse()
                fp.seek(self.imageFDR.getEndOfRecordPosition(),os.SEEK_SET)

                self.numberOfSarChannels = self.imageFDR.metadata[
                    'Number of SAR channels in this file']
                if calculateRawDimensions: self._calculateRawDimensions(fp)
                pass
        except IOError as errs:
            errno, strerr = errs
            print("IOError: %s" % strerr)

        return None

    def extractImage(self,output=None):
        """For now, call a wrapped version of ALOS_pre_process"""
        productLevel = float(self.parent.leaderFile.sceneHeaderRecord.metadata[
            'Product level code'])
        self.parent.logger.info("Extracting Level %s data" % (productLevel))
        if productLevel == 1.5:
            raise NotImplementedError
        elif productLevel == 1.1:
            self.extractSLC(output)
        elif productLevel == 1.0:
            self.extractRaw(output)
        else:
            raise ValueError(productLevel)
        return None

    @use_api
    def extractRaw(self,output=None):
        #if (self.numberOfSarChannels == 1):
        #    print "Single Pol Data Found"
        #    self.extractSinglePolImage(output=output)
        #elif (self.numberOfSarChannels == 3):
        #    print "Dual Pol Data Found"
        #elif (self.numberOfSarChannels == 6):
        #    print "Quad Pol Data Found"
        if self.parent.leaderFile.sceneHeaderRecord.metadata[
            'Processing facility identifier'] == 'ERSDAC':
            prmDict = alos.alose_Py(self.parent._leaderFile,
                self.parent._imageFile, output)
        else:
            prmDict = alos.alos_Py(self.parent._leaderFile,
                self.parent._imageFile, output)
            pass
        
        # updated 07/24/2012
        self.width = prmDict['NUMBER_BYTES_PER_LINE'] - 2 * prmDict['FIRST_SAMPLE']
        self.length = self.imageFDR.metadata['Number of lines per data set']
        self.prefix = self.imageFDR.metadata[
                    'Number of bytes of prefix data per record']
        self.suffix = self.imageFDR.metadata[
                    'Number of bytes of suffix data per record']
        self.dataSize = self.imageFDR.metadata[
                    'Number of bytes of SAR data per record']
        self.start_time = self._parseClockTime(prmDict['SC_CLOCK_START'])
        self.stop_time = self._parseClockTime(prmDict['SC_CLOCK_STOP'])
        self.startingRange = prmDict['NEAR_RANGE']
        self.prf = prmDict['PRF']

        rawImage = isceobj.createRawImage()
        rawImage.setFilename(output)
        rawImage.setAccessMode('read')
        rawImage.setWidth(self.width)
        rawImage.setXmax(self.width)
        rawImage.setXmin(0)
        self.parent.getFrame().setImage(rawImage)
        rawImage.renderVRT()
        # updated 07/24/2012
        return None





    def extractSLC(self, output=None):
        """
        For now, just skip the header and dump the SLC;
        it should be complete and without missing lines
        """

        with open(self.parent._imageFile,'rb') as fp:
            with open(output,'wb') as out:

                self.width = int(self.imageFDR.metadata[
                    'Number of bytes of SAR data per record']/
                    self.imageFDR.metadata['Number of bytes per data group'])
                self.length = int(self.imageFDR.metadata[
                    'Number of lines per data set'])

                ## JEB: use arguments?
                slcImage = isceobj.createSlcImage()
                slcImage.setFilename(output)
                slcImage.setByteOrder('b')
                slcImage.setAccessMode('read')
                slcImage.setWidth(self.width)
                slcImage.setXmin(0)
                slcImage.setXmax(self.width)
                self.parent.getFrame().setImage(slcImage)

                numberOfLines = self.imageFDR.metadata[
                    'Number of lines per data set']
                prefix = self.imageFDR.metadata[
                    'Number of bytes of prefix data per record']
                suffix = self.imageFDR.metadata[
                    'Number of bytes of suffix data per record']
                dataSize = self.imageFDR.metadata[
                    'Number of bytes of SAR data per record']

                fp.seek(self.HEADER_LINES, os.SEEK_SET) # Skip the header

                for i in range(numberOfLines):
                    if not i%1000: self.parent.logger.info("Line %s" % i)
                    imageRecord = CEOS.CEOSDB(
                            xml = os.path.join(xmlPrefix,'alos/image_record.xml'),
                            dataFile=fp)
                    imageRecord.parse()

                    if i == 0:
                        self.start_time = self._getAcquisitionTime(imageRecord)
                        self.startingRange = self._getSlantRange(imageRecord)
                        self.prf = self._getPRF(imageRecord)
                    elif i == (numberOfLines-1):
                        self.stop_time = self._getAcquisitionTime(imageRecord)
#                    else:
                        # Skip the first 412 bytes of each line
#                        fp.seek(prefix, os.SEEK_CUR)
#                        pass

                    data = fp.read(dataSize)
                    out.write(data)
                    fp.seek(suffix, os.SEEK_CUR)
                    pass


                pass
            pass
        return None

    def _getSlantRange(self,imageRecord):
        slantRange = imageRecord.metadata['Slant range to 1st pixel']
        return slantRange

    def _getPRF(self,imageRecord):
        prf = imageRecord.metadata['PRF']/1000.0 # PRF is in mHz
        return prf

    def _getAcquisitionTime(self,imageRecord):
        acquisitionTime = datetime.datetime(
            year=imageRecord.metadata['Sensor acquisition year'],month=1,day=1)
        acquisitionTime = acquisitionTime + datetime.timedelta(
            days=(imageRecord.metadata['Sensor acquisition day of year']-1),
            milliseconds=imageRecord.metadata[
            'Sensor acquisition milliseconds of day'])
        return acquisitionTime

    ## Arguemnt doesn't make sense, since file is repopend
    def _calculateRawDimensions(self, fp=None):
        """"
        Run through the data file once, and calculate the valid sampling window
        start time range.
        """
        ## If you have a file, and you've parsed it: go for it
        if fp and self.imageFDR:
            lines = int(self.imageFDR.metadata['Number of lines per data set'])
            prefix = self.imageFDR.metadata[
                'Number of bytes of prefix data per record']
            suffix = self.imageFDR.metadata[
                'Number of bytes of suffix data per record']
            dataSize = self.imageFDR.metadata[
                'Number of bytes of SAR data per record']
            self.length = lines
            self.width = dataSize+suffix
            # Need to get the Range sampling rate as well to calculate the
            # number of pixels to shift each line when the starting range
            # changes

            fp.seek(self.imageFDR.getEndOfRecordPosition(),os.SEEK_SET)
            lastPRF = 0
            lastSlantRange = 0
            for line in range(lines):

                if not line%1000:
                    self.parent.logger.info("Parsing line %s" % line)

                imageRecord = CEOS.CEOSDB(
                    xml=os.path.join(xmlPrefix,'alos/image_record.xml'),
                    dataFile=fp)
                imageRecord.parse()

                acquisitionTime = self._getAcquisitionTime(imageRecord)
                prf = imageRecord.metadata['PRF']
                if lastPRF == 0:
                    lastPRF = prf
                elif lastPRF != prf:
                    self.parent.logger.info("PRF change detected")
                    lastPRF = prf

                txPolarization = imageRecord.metadata[
                    'Transmitted polarization']
                rxPolarization = imageRecord.metadata['Received polarization']
                slantRange = self._getSlantRange(imageRecord)
                if lastSlantRange == 0:
                    lastSlantRange = slantRange
                elif lastSlantRange != slantRange:
                    self.parent.logger.info("Slant range offset detected")
                    lastSlantRange = slantRange
                    pass
                if line==0:
                    self.start_time = acquisitionTime
                    self.startingRange = slantRange
                elif line == (lines-1):
                    self.stop_time = acquisitionTime
                    pass
                fp.seek(dataSize+suffix,os.SEEK_CUR)
                pass
            pass
        else:
            ## The parse method will call this one properly
            self.parse(True)
        return None

    def extractSinglePolImage(self, output=None):
        import array
        if not self.imageFDR:
            self.parse()
            pass
        try:
            with open(self.file,'r') as fp:
                with open(output,'wb') as out:
                    lines = self.imageFDR.metadata[
                        'Number of lines per data set']
                    pixelCount = (self.imageFDR.metadata[
                        'Number of left border pixels per line'] +
                        self.imageFDR.metadata[
                        'Number of pixels per line per SAR channel'] +
                        self.imageFDR.metadata[
                        'Number of right border pixels per line']
                    )
                    # Need to get the Range sampling rate as well to calculate
                    # the number of pixels to shift each line when the starting
                    # range changes

                    fp.seek(self.imageFDR.getEndOfRecordPosition(),os.SEEK_SET)
                    lastSlantRange = 0
                    for line in range(lines):
                        if not line%1000: print("Extracting line %s" % line)
                        imageRecord = CEOS.CEOSDB(
                            xml=os.path.join(xmlPrefix,'alos/image_record.xml'),
                            dataFile=fp)
                        imageRecord.parse()
                        prf = imageRecord.metadata['PRF']
                        txPolarization = imageRecord.metadata[
                            'Transmitted polarization']
                        rxPolarization = imageRecord.metadata[
                            'Received polarization']
                        slantRange = imageRecord.metadata[
                            'Slant range to 1st pixel']
                        if lastSlantRange == 0:
                            lastSlantRange = slantRange
                        elif lastSlantRange != slantRange:
                            print("Slant range offset detected")
                            lastSlantRange = slantRange
                            pass
                        acquisitionTime = datetime.datetime(
                            year=imageRecord.metadata[
                                 'Sensor acquisition year'],month=1,day=1)
                        acquisitionTime = acquisitionTime + datetime.timedelta(
                            days=imageRecord.metadata[
                                 'Sensor acquisition day of year'],
                                 milliseconds=imageRecord.metadata[
                                 'Sensor acquisition milliseconds of day'])
                        IQ = array.array('B')
                        IQ.fromfile(fp,2*pixelCount)
                        IQ.tofile(out)
                        pass
                    pass
                pass
        except IOError as errs:
            errno, strerr = errs
            print("IOError: %s" % strerr)

        return None

    @staticmethod
    def _parseClockTime(clockTime):
        from iscesys.DateTimeUtil import DateTimeUtil as DTU
        date, time = str(clockTime).split('.')
        year = int(date[0:4])
        doy = int(date[4:7])
        utc_seconds = ( clockTime - int(date) ) * DTU.day
        dt = datetime.datetime(year=year, month=1, day=1)
        dt = dt + datetime.timedelta(days=(doy - 1), seconds=utc_seconds)
        return dt

    pass
