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
# Author: Piyush Agram
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import datetime
import isceobj
from isceobj.Orbit.Orbit import StateVector
from isceobj.Planet.AstronomicalHandbook import Const
from isceobj.Planet.Planet import Planet
from isceobj.Scene.Frame import Frame
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTUtil
from isceobj.Sensor import tkfunc,createAuxFile
from iscesys.Component.Component import Component
from isceobj.Sensor.Sensor import Sensor


def read_rsc(inname):
    '''
    Read RSC file contents into a dict.
    '''
    rpacdict = {}
    try:
        infile = open(inname+'.rsc', mode='r')
    except:
        raise Exception('File : {0} cannot be opened for reading.'.format(inname+'.rsc'))

    line = infile.readline()
    while line:
        llist = line.strip().split()
        if len(llist)==2 :
            rpacdict[llist[0]] = llist[1]
        line = infile.readline()
    infile.close()

    return rpacdict

class ROI_PAC(Sensor):
    """
        A class to parse ROI_PAC raw metadata
    """
    logging_name = "isce.sensor.ROI_PAC"

    def __init__(self):
        super(ROI_PAC,self).__init__()
        self._rawFile = None
        self._hdrFile = None

        self.dopplerVsRangePixel = []
        self.constants = {}
        self.dictionaryOfVariables = {
            'RAWFILE': ['rawFile','str','mandatory'],
            'HDRFILE': ['hdrFile','str','mandatory']
            }

        return None

    def getRawFile(self):
        return self._rawFile

    def getHdrFile(self):
        return self._hdrFile

    def setRawFile(self, fname):
        self._rawFile = str(fname)
        pass

    def setHdrFile(self, fname):
        self._hdrFile = str(fname)
        pass

    def getFrame(self):
        return self.frame


    def parse(self):
        metaDict = read_rsc(self.rawFile)
        self.constants.update(metaDict)
        self.populateMetadata()


    def _populatePlatform(self):
        mdict = self.constants
        platform = self.frame.getInstrument().getPlatform()

        platform.setMission(mdict['PLATFORM'])
        platform.setPlanet(Planet(pname="Earth"))
        platform.setPointingDirection(int(mdict['ANTENNA_SIDE']))

        platform.setAntennaLength(float(mdict['ANTENNA_LENGTH']))

    def _populateInstrument(self, mdict=None):
        if mdict is None:
            mdict = self.constants

        instrument = self.frame.getInstrument()
        fs = float(mdict['RANGE_SAMPLING_FREQUENCY'])

        rangePixelSize = Const.c/(2*fs)

        instrument.setRadarWavelength(float(mdict['WAVELENGTH']))
        instrument.setPulseRepetitionFrequency(float(mdict['PRF']))
        instrument.setRangePixelSize(rangePixelSize)
        instrument.setPulseLength(float(mdict['PULSE_LENGTH']))
        instrument.setChirpSlope(float(mdict['CHIRP_SLOPE']))
        instrument.setRangeSamplingRate(fs)
        instrument.setInPhaseValue(float(mdict['I_BIAS']))
        instrument.setQuadratureValue(float(mdict['Q_BIAS']))

    def _populateFrame(self, mdict=None):
        if mdict is None:
            mdict = self.constants

        startRange = float(mdict['STARTING_RANGE'])
        try:
            rangeBias = float(mdict['RANGE_BIAS'])
        except KeyError:
            rangeBias = 0.0

        #####Adjust for range gate bias if any
        startRange = startRange - rangeBias

        ####Compute the UTC times
        acqDate = mdict['FIRST_LINE_YEAR'] + '-' + mdict['FIRST_LINE_MONTH_OF_YEAR'] + '-' + mdict['FIRST_LINE_DAY_OF_MONTH']
        date0 = datetime.datetime.strptime(acqDate,'%Y-%m-%d')

        sensingStart = date0 + datetime.timedelta(seconds=float(mdict['FIRST_LINE_UTC']))
        sensingMid = date0 + datetime.timedelta(seconds=float(mdict['CENTER_LINE_UTC']))
        sensingStop = date0 + datetime.timedelta(seconds=float(mdict['LAST_LINE_UTC']))

        self.frame.setStartingRange(startRange)
        self.frame.setPassDirection(mdict['ORBIT_DIRECTION'].upper())
        self.frame.setOrbitNumber(mdict['ORBIT_NUMBER'])

        try:
            self.frame.setProcessingFacility(mdict['PROCESSING_FACILITY'])
        except:
            self.frame.setProcessingFacility('Dummy')

        try:
            self.frame.setProcessingSoftwareVersion(mdict['PROCESSING_SYSTEM'])
        except:
            self.frame.setProcessingSoftwareVersion('Dummy')

        try:
            self.frame.setPolarization(mdict['POLARIZATION'])
        except:
            self.frame.setPolarization('HH')
        self.frame.setNumberOfLines(int(mdict['FILE_LENGTH']))

        ####Account for correct width and first byte here
        self.frame.setNumberOfSamples(int(mdict['WIDTH'])/2)


        self.frame.setSensingStart(sensingStart)
        self.frame.setSensingMid(sensingMid)
        self.frame.setSensingStop(sensingStop)

        rangePixelSize = self.frame.getInstrument().getRangePixelSize()
        farRange = startRange +  self.frame.getNumberOfSamples()*rangePixelSize
        self.frame.setFarRange(farRange)


    def _populateOrbit(self, mdict=None):
        orbit = self.frame.getOrbit()

        orbit.setReferenceFrame('ECR')
        orbit.setOrbitSource('Header')
        refDate = self.frame.getSensingStart().date()
        t0 = datetime.datetime(refDate.year, refDate.month, refDate.day)
        lines = [line.strip() for line in open(self.hdrFile,'r')]

        for line in lines:
            vec = StateVector()
            if len(line.split()) == 7:
                fields =[float(val) for val in line.split()]
                dt = t0 + datetime.timedelta(seconds=fields[0])
                vec.setTime(dt)
                vec.setPosition(fields[1:4])
                vec.setVelocity(fields[4:7])
                orbit.addStateVector(vec)

    def populateImage(self):
        mdict = self.constants
        rawImage = isceobj.createRawImage()
        rawImage.setByteOrder('l')
        rawImage.setFilename(self.rawFile)
        rawImage.setAccessMode('read')
        rawImage.setWidth(2*self.frame.getNumberOfSamples())
        rawImage.setXmax(2*self.frame.getNumberOfSamples())
        rawImage.setXmin(int(mdict['XMIN']))
        self.getFrame().setImage(rawImage)

    def _populateExtras(self):
        """
        Populate some extra fields.
        """
        mdict = self.constants
        prf=float(mdict['PRF'])
        self.frame._dopplerVsPixel = [float(mdict['DOPPLER_RANGE0'])*prf,
                                      float(mdict['DOPPLER_RANGE1'])*prf,
                                      float(mdict['DOPPLER_RANGE2'])*prf ]

    def extractImage(self):
        """Extract the raw image data"""
        self.parse()
        self._populateExtras()
        self.populateImage()
        createAuxFile(self.frame,self.rawFile + '.aux')

    def extractDoppler(self):
        """
        Return the doppler centroid as defined in the HDF5 file.
        """
        self._populateExtras()
        prf = float(self.constants['PRF'])
        quadratic = {}
        dopp = self.frame._dopplerVsPixel
        mid = 0.5*(int(self.constants['WIDTH']) - int(self.constants['XMIN']))
        fd_mid = dopp[0] + (dopp[1] + dopp[2]*mid)*mid

        quadratic['a'] = dopp[0]/prf
        quadratic['b'] = dopp[1]/prf
        quadratic['c'] = dopp[2]/prf                
        return quadratic

    rawFile = property(getRawFile, setRawFile)
    hdrFile = property(getHdrFile, setHdrFile)
