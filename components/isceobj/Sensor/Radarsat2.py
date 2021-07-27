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




from xml.etree.ElementTree import ElementTree
import datetime
import isceobj

from isceobj.Scene.Frame import Frame
from isceobj.Planet.Planet import Planet
from isceobj.Orbit.Orbit import StateVector, Orbit
from isceobj.Orbit.OrbitExtender import OrbitExtender
from isceobj.Planet.AstronomicalHandbook import Const
from iscesys.Component.Component import Component
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTUtil
import os
import numpy as np

sep = "\n"
tab = "    "
lookMap = { 'RIGHT' : -1,
            'LEFT' : 1}

TIFF = Component.Parameter(
    'tiff',
    public_name='TIFF',
    default='',
    type=str,
    mandatory=True,
    doc='RadarSAT2 tiff imagery file'
)

XML = Component.Parameter(
    'xml',
    public_name='XML',
    default='',
    type=str,
    mandatory=True,
    doc='RadarSAT2 xml metadata file'
)

ORBIT_DIRECTORY = Component.Parameter(
        'orbitDirectory',
        public_name = 'orbit directory',
        default=None,
        type=str,
        mandatory=False,
        doc='Directory with Radarsat2 precise orbits')

ORBIT_FILE = Component.Parameter(
        'orbitFile',
        public_name = 'orbit file',
        default = None,
        type = str,
        mandatory = False,
        doc = 'Precise orbit file to use')

from .Sensor import Sensor
class Radarsat2(Sensor):
    """
        A Class representing RADARSAT 2 data
    """

    family='radarsat2'
    parameter_list = (XML, TIFF, ORBIT_DIRECTORY, ORBIT_FILE ) + Sensor.parameter_list

    def __init__(self, family='', name=''):
        super().__init__(family if family else  self.__class__.family, name=name)
        self.product = _Product()
        self.frame = Frame()
        self.frame.configure()


    def getFrame(self):
        return self.frame

    def parse(self):
        try:
            fp = open(self.xml,'r')
        except IOError as strerr:
            print("IOError: %s" % strerr)
            return
        self._xml_root = ElementTree(file=fp).getroot()
        self.product.set_from_etnode(self._xml_root)
        self.populateMetadata()

        fp.close()

    def populateMetadata(self):
        """
            Create metadata objects from the metadata files
        """
        mission = self.product.sourceAttributes.satellite
        swath = self.product.sourceAttributes.radarParameters.beams
        frequency = self.product.sourceAttributes.radarParameters.radarCenterFrequency
        orig_prf = self.product.sourceAttributes.radarParameters.prf  # original PRF not necessarily effective PRF
        rangePixelSize = self.product.imageAttributes.rasterAttributes.sampledPixelSpacing
        rangeSamplingRate = Const.c/(2*rangePixelSize)
        pulseLength = self.product.sourceAttributes.radarParameters.pulseLengths[0]
        pulseBandwidth = self.product.sourceAttributes.radarParameters.pulseBandwidths[0]
        polarization = self.product.sourceAttributes.radarParameters.polarizations
        lookSide = lookMap[self.product.sourceAttributes.radarParameters.antennaPointing.upper()]
        facility = self.product.imageGenerationParameters.generalProcessingInformation._processingFacility
        version = self.product.imageGenerationParameters.generalProcessingInformation.softwareVersion
        lines = self.product.imageAttributes.rasterAttributes.numberOfLines
        samples = self.product.imageAttributes.rasterAttributes.numberOfSamplesPerLine
        startingRange = self.product.imageGenerationParameters.slantRangeToGroundRange.slantRangeTimeToFirstRangeSample * (Const.c/2)
        incidenceAngle = (self.product.imageGenerationParameters.sarProcessingInformation.incidenceAngleNearRange + self.product.imageGenerationParameters.sarProcessingInformation.incidenceAngleFarRange)/2
        # some RS2 scenes have oversampled SLC images because processed azimuth bandwidth larger than PRF EJF 2015/08/15
        azimuthPixelSize = self.product.imageAttributes.rasterAttributes.sampledLineSpacing  # ground spacing in meters
        totalProcessedAzimuthBandwidth = self.product.imageGenerationParameters.sarProcessingInformation.totalProcessedAzimuthBandwidth
        prf = orig_prf * np.ceil(totalProcessedAzimuthBandwidth / orig_prf) # effective PRF can be double original, suggested by Piyush
        print("effective PRF %f, original PRF %f" % (prf, orig_prf) )

        lineFlip =  (self.product.imageAttributes.rasterAttributes.lineTimeOrdering.upper() == 'DECREASING')

        if lineFlip:
            dataStopTime = self.product.imageGenerationParameters.sarProcessingInformation.zeroDopplerTimeFirstLine
            dataStartTime = self.product.imageGenerationParameters.sarProcessingInformation.zeroDopplerTimeLastLine
        else:
            dataStartTime = self.product.imageGenerationParameters.sarProcessingInformation.zeroDopplerTimeFirstLine
            dataStopTime = self.product.imageGenerationParameters.sarProcessingInformation.zeroDopplerTimeLastLine

        passDirection = self.product.sourceAttributes.orbitAndAttitude.orbitInformation.passDirection
        height = self.product.imageGenerationParameters.sarProcessingInformation._satelliteHeight

        ####Populate platform
        platform = self.frame.getInstrument().getPlatform()
        platform.setPlanet(Planet(pname="Earth"))
        platform.setMission(mission)
        platform.setPointingDirection(lookSide)
        platform.setAntennaLength(15.0)

        ####Populate instrument
        instrument = self.frame.getInstrument()
        instrument.setRadarFrequency(frequency)
        instrument.setPulseRepetitionFrequency(prf)
        instrument.setPulseLength(pulseLength)
        instrument.setChirpSlope(pulseBandwidth/pulseLength)
        instrument.setIncidenceAngle(incidenceAngle)
        #self.frame.getInstrument().setRangeBias(0)
        instrument.setRangePixelSize(rangePixelSize)
        instrument.setRangeSamplingRate(rangeSamplingRate)
        instrument.setBeamNumber(swath)
        instrument.setPulseLength(pulseLength)


        #Populate Frame
        #self.frame.setSatelliteHeight(height)
        self.frame.setSensingStart(dataStartTime)
        self.frame.setSensingStop(dataStopTime)
        diffTime = DTUtil.timeDeltaToSeconds(dataStopTime - dataStartTime)/2.0
        sensingMid = dataStartTime + datetime.timedelta(microseconds=int(diffTime*1e6))
        self.frame.setSensingMid(sensingMid)
        self.frame.setPassDirection(passDirection)
        self.frame.setPolarization(polarization)
        self.frame.setStartingRange(startingRange)
        self.frame.setFarRange(startingRange + (samples-1)*rangePixelSize)
        self.frame.setNumberOfLines(lines)
        self.frame.setNumberOfSamples(samples)
        self.frame.setProcessingFacility(facility)
        self.frame.setProcessingSoftwareVersion(version)
        self.frame.setPassDirection(passDirection)

        self.frame.getOrbit().setOrbitSource(self.product.sourceAttributes.orbitAndAttitude.orbitInformation.orbitDataFile)
        
        if (self.orbitFile is None) and (self.orbitDirectory is None):
            self.extractOrbit()

        elif (self.orbitDirectory is not None):
            self.orbitFile = findPreciseOrbit(self.orbitDirectory, self.frame.getOrbit().getOrbitSource(), self.frame.sensingStart.year)

        if self.orbitFile is not None:
            self.extractPreciseOrbit(self.orbitFile, self.frame.sensingStart, self.frame.sensingStop)
        

# save the Doppler centroid coefficients, converting units from product.xml file
# units in the file are quadratic coefficients in Hz, Hz/sec, and Hz/(sec^2)
# ISCE expects Hz, Hz/(range sample), Hz((range sample)^2
# note that RS2 Doppler values are estimated at time dc.dopplerCentroidReferenceTime,
# so the values might need to be adjusted for ISCE usage
# added EJF 2015/08/17
        dc = self.product.imageGenerationParameters.dopplerCentroid
        poly = dc.dopplerCentroidCoefficients
        # need to convert units
        poly[1] = poly[1]/rangeSamplingRate
        poly[2] = poly[2]/rangeSamplingRate**2
        self.doppler_coeff = poly

# similarly save Doppler azimuth fm rate values, converting units
# units in the file are quadratic coefficients in Hz, Hz/sec, and Hz/(sec^2)
# Guessing that ISCE expects Hz, Hz/(range sample), Hz((range sample)^2
# note that RS2 Doppler values are estimated at time dc.dopplerRateReferenceTime,
# so the values might need to be adjusted for ISCE usage
# added EJF 2015/08/17
        dr = self.product.imageGenerationParameters.dopplerRateValues
        fmpoly = dr.dopplerRateValuesCoefficients
        # need to convert units
        fmpoly[1] = fmpoly[1]/rangeSamplingRate
        fmpoly[2] = fmpoly[2]/rangeSamplingRate**2
        self.azfmrate_coeff = fmpoly

        # now calculate effective PRF from the azimuth line spacing after we have the orbit info EJF 2015/08/15
        # this does not work because azimuth spacing is on ground. Instead use bandwidth ratio calculated above  EJF
#        SCHvelocity = self.frame.getSchVelocity()
#        SCHvelocity = 7550.75  # hard code orbit velocity for now m/s
#        prf = SCHvelocity/azimuthPixelSize
#        instrument.setPulseRepetitionFrequency(prf)

    def extractOrbit(self):
        '''
        Extract the orbit state vectors from the XML file.
        '''

        # Initialize orbit objects
        # Read into temp orbit first.
        # Radarsat 2 needs orbit extensions.
        tempOrbit = Orbit()

        self.frame.getOrbit().setOrbitSource('Header: ' + self.frame.getOrbit().getOrbitSource())
        stateVectors = self.product.sourceAttributes.orbitAndAttitude.orbitInformation.stateVectors
        for i in range(len(stateVectors)):
            position = [stateVectors[i].xPosition, stateVectors[i].yPosition, stateVectors[i].zPosition]
            velocity = [stateVectors[i].xVelocity, stateVectors[i].yVelocity, stateVectors[i].zVelocity]
            vec = StateVector()
            vec.setTime(stateVectors[i].timeStamp)
            vec.setPosition(position)
            vec.setVelocity(velocity)
            tempOrbit.addStateVector(vec)

        planet = self.frame.instrument.platform.planet
        orbExt = OrbitExtender(planet=planet)
        orbExt.configure()
        newOrb = orbExt.extendOrbit(tempOrbit)

        for sv in newOrb:
            self.frame.getOrbit().addStateVector(sv)

        print('Successfully read state vectors from product XML')

    def extractPreciseOrbit(self, orbitfile, tstart, tend):
        '''
        Extract precise orbits for given time-period from orbit file.
        '''

        self.frame.getOrbit().setOrbitSource('File: ' + orbitfile)

        tmin = tstart - datetime.timedelta(seconds=30.)
        tmax = tstart + datetime.timedelta(seconds=30.)

        fid = open(orbitfile, 'r')
        for line in fid:
            if not line.startswith('; Position'):
                continue
            else:
                break

        for line in fid:
            if not line.startswith(';###END'):
                tstamp = convertRSTimeToDateTime(line)
                
                if (tstamp >= tmin) and (tstamp <= tmax):
                    sv = StateVector()
                    sv.configure()
                    sv.setTime( tstamp)
                    sv.setPosition( [float(x) for x in fid.readline().split()])
                    sv.setVelocity( [float(x) for x in fid.readline().split()])

                    self.frame.getOrbit().addStateVector(sv)
                else:
                    fid.readline()
                    fid.readline()

                dummy = fid.readline()
                if not dummy.startswith(';'):
                    raise Exception('Expected line to start with ";". Got {0}'.format(dummy))

        fid.close()
        print('Successfully read {0} state vectors from {1}'.format( len(self.frame.getOrbit()._stateVectors), orbitfile))

    def extractImage(self, verbose=True):
        '''
        Use gdal to extract the slc.
        '''

        try:
            from osgeo import gdal
        except ImportError:
            raise Exception('GDAL python bindings not found. Need this for RSAT2 / TandemX / Sentinel1A.')

        self.parse()

        width = self.frame.getNumberOfSamples()
        lgth  = self.frame.getNumberOfLines()
        lineFlip = (self.product.imageAttributes.rasterAttributes.lineTimeOrdering.upper() == 'DECREASING')
        pixFlip = (self.product.imageAttributes.rasterAttributes.pixelTimeOrdering.upper() == 'DECREASING')

        src = gdal.Open(self.tiff.strip(), gdal.GA_ReadOnly)
        cJ = np.complex64(1.0j)

        ####Images are small enough that we can do it all in one go - Piyush
        real = src.GetRasterBand(1).ReadAsArray(0,0,width,lgth)
        imag = src.GetRasterBand(2).ReadAsArray(0,0,width,lgth)

        if (real is None) or (imag is None):
            raise Exception('Input Radarsat2 SLC seems to not be a 2 band Int16 image.')

        data = real+cJ*imag

        real = None
        imag = None
        src = None

        if lineFlip:
            if verbose:
                print('Vertically Flipping data')
            data = np.flipud(data)

        if pixFlip:
            if verbose:
                print('Horizontally Flipping data')
            data = np.fliplr(data)

        data.tofile(self.output)

        ####
        slcImage = isceobj.createSlcImage()
        slcImage.setByteOrder('l')
        slcImage.setFilename(self.output)
        slcImage.setAccessMode('read')
        slcImage.setWidth(width)
        slcImage.setLength(lgth)
        slcImage.setXmin(0)
        slcImage.setXmax(width)
#        slcImage.renderHdr()
        self.frame.setImage(slcImage)


    def extractDoppler(self):
        '''
        self.parse()
        Extract doppler information as needed by mocomp
        '''
        ins = self.frame.getInstrument()
        dc = self.product.imageGenerationParameters.dopplerCentroid
        quadratic = {}

        r0 = self.frame.startingRange
        fs = ins.getRangeSamplingRate()
        tNear = 2*r0/Const.c

        tMid = tNear + 0.5*self.frame.getNumberOfSamples()/fs
        t0 = dc.dopplerCentroidReferenceTime
        poly = dc.dopplerCentroidCoefficients

        fd_mid = 0.0
        for kk in range(len(poly)):
            fd_mid += poly[kk] * (tMid - t0)**kk

        ####For insarApp
        quadratic['a'] = fd_mid / ins.getPulseRepetitionFrequency()
        quadratic['b'] = 0.
        quadratic['c'] = 0.


        ####For roiApp
        ####More accurate
        from isceobj.Util import Poly1D

        coeffs = poly
        dr = self.frame.getInstrument().getRangePixelSize()
        rref = 0.5 * Const.c * t0
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

class Radarsat2Namespace(object):
    def __init__(self):
        self.uri = "http://www.rsi.ca/rs2/prod/xml/schemas"

    def elementName(self,element):
        return "{%s}%s" % (self.uri,element)

    def convertToDateTime(self,string):
        dt = datetime.datetime.strptime(string,"%Y-%m-%dT%H:%M:%S.%fZ")
        return dt

class _Product(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.productId = None
        self.documentId = None
        self.sourceAttributes = _SourceAttributes()
        self.imageGenerationParameters = _ImageGenerationParameters()
        self.imageAttributes = _ImageAttributes()

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('productId'):
                self.productId = z.text
            elif z.tag == self.elementName('documentIdentifier'):
                self.documentId = z.text
            elif z.tag == self.elementName('sourceAttributes'):
                self.sourceAttributes.set_from_etnode(z)
            elif z.tag == self.elementName('imageGenerationParameters'):
                self.imageGenerationParameters.set_from_etnode(z)
            elif z.tag == self.elementName('imageAttributes'):
                self.imageAttributes.set_from_etnode(z)

    def __str__(self):
        retstr  = "Product:"+sep+tab
        retlst  = ()
        retstr += "productID=%s"+sep+tab
        retlst += (self.productId,)
        retstr += "documentIdentifier=%s"+sep
        retlst += (self.documentId,)
        retstr += "%s"+sep
        retlst += (str(self.sourceAttributes),)
        retstr += "%s"+sep
        retlst += (str(self.imageGenerationParameters),)
        retstr += ":Product"
        return retstr % retlst

class _SourceAttributes(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.satellite = None
        self.sensor = None
        self.inputDatasetId = None
        self.imageId = None
        self.inputDatasetFacilityId = None
        self.beamModeId = None
        self.beamModeMnemonic = None
        self.rawDataStartTime = None
        self.radarParameters = _RadarParameters()
        self.rawDataAttributes = _RawDataAttributes()
        self.orbitAndAttitude = _OrbitAndAttitude()

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('satellite'):
                self.satellite = z.text
            elif z.tag == self.elementName('sensor'):
                self.sensor = z.text
            elif z.tag == self.elementName('inputDatasetId'):
                self.inputDatasetId = z.text
            elif z.tag == self.elementName('imageID'):
                self.imageId = z.text
            elif z.tag == self.elementName('inputDatasetFacilityId'):
                self.inputDatasetFacilityId = z.text
            elif z.tag == self.elementName('beamModeID'):
                self.beamModeId = z.text
            elif z.tag == self.elementName('beamModeMnemonic'):
                self.beamModeMnemonic = z.text
            elif z.tag == self.elementName('rawDataStartTime'):
                self.rawDataStartTime = self.convertToDateTime(z.text)
            elif z.tag == self.elementName('radarParameters'):
                self.radarParameters.set_from_etnode(z)
            elif z.tag == self.elementName('rawDataAttributes'):
                self.rawDataAttributes.set_from_etnode(z)
            elif z.tag == self.elementName('orbitAndAttitude'):
                self.orbitAndAttitude.set_from_etnode(z)

    def __str__(self):
        retstr  = "SourceAttributes:"+sep+tab
        retlst  = ()
        retstr += "satellite=%s"+sep+tab
        retlst += (self.satellite,)
        retstr += "sensor=%s"+sep+tab
        retlst += (self.sensor,)
        retstr += "inputDatasetID=%s"+sep
        retlst += (self.inputDatasetId,)
        retstr += "%s"
        retlst += (str(self.radarParameters),)
        retstr += "%s"
        retlst += (str(self.rawDataAttributes),)
        retstr += "%s"
        retlst += (str(self.orbitAndAttitude),)
        retstr += ":SourceAttributes"
        return retstr % retlst

class _RadarParameters(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.acquisitionType = None
        self.beams = None
        self.polarizations = None
        self.pulses = None
        self.rank = None
        self.settableGains = []
        self.radarCenterFrequency = None
        self.prf = None
        self.pulseLengths = []
        self.pulseBandwidths = []
        self.antennaPointing = None
        self.adcSamplingRate = []
        self.yawSteeringFlag = None
        self.geodeticFlag = None
        self.rawBitsPerSample = None
        self.samplesPerEchoLine = None
        self.referenceNoiseLevels = [_ReferenceNoiseLevel()]*3

    def set_from_etnode(self,node):
        i = 0
        for z in node:
            if z.tag == self.elementName('acquisitionType'):
                self.acquisitionType = z.text
            elif z.tag == self.elementName('beams'):
                self.beams = z.text
            elif z.tag == self.elementName('polarizations'):
                self.polarizations = z.text
            elif z.tag == self.elementName('pulses'):
                self.pulses = z.text
            elif z.tag == self.elementName('rank'):
                self.rank = z.text
            elif z.tag == self.elementName('settableGain'):
                self.settableGains.append(z.text)
            elif z.tag == self.elementName('radarCenterFrequency'):
                self.radarCenterFrequency = float(z.text)
            elif z.tag == self.elementName('pulseRepetitionFrequency'):
                self.prf = float(z.text)
            elif z.tag == self.elementName('pulseLength'):
                self.pulseLengths.append(float(z.text))
            elif z.tag == self.elementName('pulseBandwidth'):
                self.pulseBandwidths.append(float(z.text))
            elif z.tag == self.elementName('antennaPointing'):
                self.antennaPointing = z.text
            elif z.tag == self.elementName('adcSamplingRate'):
                self.adcSamplingRate.append(float(z.text))
            elif z.tag == self.elementName('yawSteeringFlag'):
                self.yawSteeringFlag = z.text
            elif z.tag == self.elementName('rawBitsPerSample'):
                self.rawBitsPerSample = int(z.text)
            elif z.tag == self.elementName('samplesPerEchoLine'):
                self.samplesPerEchoLine = int(z.text)
            elif z.tag == self.elementName('referenceNoiseLevels'):
                self.referenceNoiseLevels[i].set_from_etnode(z)
                i += 1

    def __str__(self):
        retstr = "RadarParameters:"+sep+tab
        retlst = ()
        retstr += "acquisitionType=%s"+sep+tab
        retlst += (self.acquisitionType,)
        retstr += "beams=%s"+sep+tab
        retlst += (self.beams,)
        retstr += "polarizations=%s"+sep+tab
        retlst += (self.polarizations,)
        retstr += "pulses=%s"+sep+tab
        retlst += (self.pulses,)
        retstr += "rank=%s"+sep
        retlst += (self.rank,)
        retstr += ":RadarParameters"+sep
        return retstr % retlst

class _ReferenceNoiseLevel(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.pixelFirstNoiseValue = None
        self.stepSize = None
        self.numberOfNoiseLevelValues = None
        self.noiseLevelValues = []

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('pixelFirstNoiseValue'):
                self.pixelFirstNoiseValue = int(z.text)
            elif z.tag == self.elementName('stepSize'):
                self.stepSize = int(z.text)
            elif z.tag == self.elementName('numberOfNoiseLevelValues'):
                self.numberOfNoiseLevelValues = int(z.text)
            elif z.tag == self.elementName('noiseLevelValues'):
                self.noiseLevelValues = list(map(float,z.text.split()))

    def __str__(self):
        retstr  = "ReferenceNoiseLevel:"+sep+tab
        retlst  = ()
        retstr += "pixelFirstNoiseValue=%s"+sep+tab
        retlst += (self.pixelFirstNoiseValue,)
        retstr += "stepSize=%s"+sep+tab
        retlst += (self.stepSize,)
        retstr += "numberOfNoiseLevelValues=%s"+sep+tab
        retlst += (self.numberOfNoiseLevelValues,)
        retstr += "noiseLevelValues=%s"+sep+tab
        retlst += (self.noiseLevelValues,)
        retstr += sep+":ReferenceNoiseLevel"
        return retstr % retlst

class _RawDataAttributes(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.numberOfInputDataGaps = None
        self.gapSize = None
        self.numberOfMissingLines = None
        self.rawDataAnalysis = [_RawDataAnalysis]*4

    def set_from_etnode(self,node):
        pass

    def __str__(self):
        return ""

class _RawDataAnalysis(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)

    def set_from_etnode(self,node):
        pass

class _OrbitAndAttitude(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.orbitInformation = _OrbitInformation()
        self.attitudeInformation = _AttitudeInformation()

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('orbitInformation'):
                self.orbitInformation.set_from_etnode(z)
            elif z.tag == self.elementName('attitudeInformation'):
                self.attitudeInformation.set_from_etnode(z)

    def __str__(self):
        retstr = "OrbitAndAttitude:"+sep
        retlst = ()
        retstr += "%s"
        retlst += (str(self.orbitInformation),)
        retstr += "%s"
        retlst += (str(self.attitudeInformation),)
        retstr += ":OrbitAndAttitude"+sep
        return retstr % retlst

class _OrbitInformation(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.passDirection = None
        self.orbitDataSource = None
        self.orbitDataFile = None
        self.stateVectors = []

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('passDirection'):
                self.passDirection = z.text
            elif z.tag == self.elementName('orbitDataSource'):
                self.orbitDataSource = z.text
            elif z.tag == self.elementName('orbitDataFile'):
                self.orbitDataFile = z.text
            elif z.tag == self.elementName('stateVector'):
                sv = _StateVector()
                sv.set_from_etnode(z)
                self.stateVectors.append(sv)

    def __str__(self):
        retstr = "OrbitInformation:"+sep+tab
        retlst = ()
        retstr += "passDirection=%s"+sep+tab
        retlst += (self.passDirection,)
        retstr += "orbitDataSource=%s"+sep+tab
        retlst += (self.orbitDataSource,)
        retstr += "orbitDataFile=%s"+sep
        retlst += (self.orbitDataFile,)
        retstr += ":OrbitInformation"+sep
        return retstr % retlst


class _StateVector(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.timeStamp = None
        self.xPosition = None
        self.yPosition = None
        self.zPosition = None
        self.xVelocity = None
        self.yVelocity = None
        self.zVelocity = None

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('timeStamp'):
                self.timeStamp = self.convertToDateTime(z.text)
            elif z.tag == self.elementName('xPosition'):
                self.xPosition = float(z.text)
            elif z.tag == self.elementName('yPosition'):
                self.yPosition = float(z.text)
            elif z.tag == self.elementName('zPosition'):
                self.zPosition = float(z.text)
            elif z.tag == self.elementName('xVelocity'):
                self.xVelocity = float(z.text)
            elif z.tag == self.elementName('yVelocity'):
                self.yVelocity = float(z.text)
            elif z.tag == self.elementName('zVelocity'):
                self.zVelocity = float(z.text)

    def __str__(self):
        retstr = "StateVector:"+sep+tab
        retlst = ()
        retstr += "timeStamp=%s"+sep+tab
        retlst += (self.timeStamp,)
        retstr += "xPosition=%s"+sep+tab
        retlst += (self.xPosition,)
        retstr += "yPosition=%s"+sep+tab
        retlst += (self.yPosition,)
        retstr += "zPosition=%s"+sep+tab
        retlst += (self.zPosition,)
        retstr += "xVelocity=%s"+sep+tab
        retlst += (self.xVelocity,)
        retstr += "yVelocity=%s"+sep+tab
        retlst += (self.yVelocity,)
        retstr += "zVelocity=%s"+sep+tab
        retlst += (self.zVelocity,)
        retstr += sep+":StateVector"
        return retstr % retlst

class _AttitudeInformation(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.attitudeDataSource = None
        self.attitudeOffsetApplied = None
        self.attitudeAngles = []

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('attitudeDataSource'):
                self.attitudeDataSource = z.text
            elif z.tag == self.elementName('attitudeOffsetApplied'):
                self.attitudeOffsetApplied = z.text
            elif z.tag == self.elementName('attitudeAngles'):
                aa = _AttitudeAngles()
                aa.set_from_etnode(z)
                self.attitudeAngles.append(aa)

    def __str__(self):
        retstr = "AttitudeInformation:"+sep+tab
        retlst = ()
        retstr += "attitudeDataSource=%s"+sep+tab
        retlst += (self.attitudeDataSource,)
        retstr += "attitudeOffsetApplied=%s"+sep+tab
        retlst += (self.attitudeOffsetApplied,)
        retstr += "%s"+sep+tab
        retlst += (map(str,self.attitudeAngles),)
        retstr += ":AttitudeInformation"+sep
        return retstr % retlst

class _AttitudeAngles(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.timeStamp = None
        self.yaw = None
        self.roll = None
        self.pitch = None

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('timeStamp'):
                self.timeStamp = self.convertToDateTime(z.text)
            elif z.tag == self.elementName('yaw'):
                self.yaw = float(z.text)
            elif z.tag == self.elementName('roll'):
                self.roll = float(z.text)
            elif z.tag == self.elementName('pitch'):
                self.pitch = float(z.text)

    def __str__(self):
        retstr = "AttitudeAngles:"+sep+tab
        retlst = ()
        retstr += "timeStamp=%s"+sep+tab
        retlst += (self.timeStamp,)
        retstr += "yaw=%s"+sep+tab
        retlst += (self.yaw,)
        retstr += "roll=%s"+sep+tab
        retlst += (self.roll,)
        retstr += "pitch=%s"+sep+tab
        retlst += (self.pitch,)
        retstr += sep+":AttitudeAngles"
        return retstr % retlst

class _ImageGenerationParameters(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.generalProcessingInformation = _GeneralProcessingInformation()
        self.sarProcessingInformation = _SarProcessingInformation()
        self.dopplerCentroid = _DopplerCentroid()
        self.dopplerRateValues = _DopplerRateValues()
        self.chirp = []
        self.slantRangeToGroundRange = _SlantRangeToGroundRange()
        self.payloadCharacteristicsFile = []

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('generalProcessingInformation'):
                self.generalProcessingInformation.set_from_etnode(z)
            elif z.tag == self.elementName('sarProcessingInformation'):
                self.sarProcessingInformation.set_from_etnode(z)
            elif z.tag == self.elementName('dopplerCentroid'):
                self.dopplerCentroid.set_from_etnode(z)
            elif z.tag == self.elementName('dopplerRateValues'):
                self.dopplerRateValues.set_from_etnode(z)
            elif z.tag == self.elementName('slantRangeToGroundRange'):
                self.slantRangeToGroundRange.set_from_etnode(z)

    def __str__(self):
        retstr = "ImageGenerationParameters:"+sep
        retlst = ()
        retstr += "%s"
        retlst += (str(self.generalProcessingInformation),)
        retstr += "%s"
        retlst += (str(self.sarProcessingInformation),)
        retstr += "%s"
        retlst += (str(self.dopplerCentroid),)
        retstr += "%s"
        retlst += (str(self.dopplerRateValues),)
        retstr += ":ImageGenerationParameters"
        return retstr % retlst


class _GeneralProcessingInformation(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.productType = None
        self._processingFacility = None
        self.processingTime = None
        self.softwareVersion = None

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('productType'):
                self.productType = z.text
            elif z.tag == self.elementName('_processingFacility'):
                self._processingFacility = z.text
            elif z.tag == self.elementName('processingTime'):
                self.processingTime = self.convertToDateTime(z.text)
            elif z.tag == self.elementName('softwareVersion'):
                self.softwareVersion = z.text

    def __str__(self):
        retstr = "GeneralProcessingInformation:"+sep+tab
        retlst = ()
        retstr += "productType=%s"+sep+tab
        retlst += (self.productType,)
        retstr += "_processingFacility=%s"+sep+tab
        retlst += (self._processingFacility,)
        retstr += "processingTime=%s"+sep+tab
        retlst += (self.processingTime,)
        retstr += "softwareVersion=%s"+sep
        retlst += (self.softwareVersion,)
        retstr += ":GeneralProcessingInformation"+sep
        return retstr % retlst

class _SarProcessingInformation(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.lutApplied = None
        self.elevationPatternCorrection = None
        self.rangeSpreadingLossCorrection = None
        self.pulseDependantGainCorrection = None
        self.receiverSettableGain = None
        self.rawDataCorrection = None
        self.rangeReferenceFunctionSource = None
        self.interPolarizationMatricesCorrection = None
        self.zeroDopplerTimeFirstLine = None
        self.zeroDopplerTimeLastLine = None
        self.numberOfLinesProcessed = None
        self.samplingWindowStartTimeFirstRawLine = None
        self.samplingWindowStartTimeLastRawLine = None
        self.numberOfSwstChanges = None
        self.numberOfRangeLooks = None
        self.rangeLookBandwidth = None
        self.totalProcessedRangeBandwidth = None
        self.numberOfAzimuthLooks = None
        self.scalarLookWeights = None
        self.azimuthLookBandwidth = None
        self.totalProcessedAzimuthBandwidth = None
        self.azimuthWindow = _Window('Azimuth')
        self.rangeWindow = _Window('Range')
        self.incidenceAngleNearRange = None
        self.incidenceAngleFarRange = None
        self.slantRangeNearEdge = None
        self._satelliteHeight = None

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('lutApplied'):
                self.lutApplied = z.text
            elif z.tag == self.elementName('numberOfLinesProcessed'):
                self.numberOfLinesProcessed = int(z.text)
            elif z.tag == self.elementName('azimuthWindow'):
                self.azimuthWindow.set_from_etnode(z)
            elif z.tag == self.elementName('rangeWindow'):
                self.rangeWindow.set_from_etnode(z)
            elif z.tag == self.elementName('incidenceAngleNearRange'):
                self.incidenceAngleNearRange = float(z.text)
            elif z.tag == self.elementName('incidenceAngleFarRange'):
                self.incidenceAngleFarRange = float(z.text)
            elif z.tag == self.elementName('slantRangeNearEdge'):
                self.slantRangeNearEdge = float(z.text)
            elif z.tag == self.elementName('totalProcessedAzimuthBandwidth'):
                self.totalProcessedAzimuthBandwidth = float(z.text)
            elif z.tag == self.elementName('_satelliteHeight'):
                self._satelliteHeight = float(z.text)
            elif z.tag == self.elementName('zeroDopplerTimeFirstLine'):
                self.zeroDopplerTimeFirstLine = self.convertToDateTime(z.text)
            elif z.tag == self.elementName('zeroDopplerTimeLastLine'):
                self.zeroDopplerTimeLastLine = self.convertToDateTime(z.text)

    def __str__(self):
        retstr = "sarProcessingInformation:"+sep+tab
        retlst = ()
        retstr += "lutApplied=%s"+sep+tab
        retlst += (self.lutApplied,)
        retstr += "numberOfLineProcessed=%s"+sep
        retlst += (self.numberOfLinesProcessed,)
        retstr += "%s"+sep
        retlst += (str(self.azimuthWindow),)
        retstr += "%s"+sep
        retlst += (str(self.rangeWindow),)
        retstr += ":sarProcessingInformation"+sep
        return retstr % retlst

class _Window(Radarsat2Namespace):
    def __init__(self,type):
        Radarsat2Namespace.__init__(self)
        self.type = type
        self.windowName = None
        self.windowCoefficient = None

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('windowName'):
                self.windowName = z.text
            elif z.tag == self.elementName('windowCoefficient'):
                self.windowCoefficient = float(z.text)

    def __str__(self):
        retstr = "%sWindow:"+sep+tab
        retlst = (self.type,)
        retstr += "windowName=%s"+sep+tab
        retlst += (self.windowName,)
        retstr += "windowCoefficient=%s"+sep
        retlst += (self.windowCoefficient,)
        retstr += ":%sWindow"
        retlst += (self.type,)
        return retstr % retlst

class _DopplerCentroid(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.timeOfDopplerCentroidEstimate = None
        self.dopplerAmbiguity = None
        self.dopplerAmbiguityConfidence= None
        self.dopplerCentroidReferenceTime = None
        self.dopplerCentroidPolynomialPeriod = None
        self.dopplerCentroidCoefficients = []
        self.dopplerCentroidConfidence = None

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('timeOfDopplerCentroidEstimate'):
                self.timeOfDopplerCentroidEstimate = self.convertToDateTime(z.text)
            elif z.tag == self.elementName('dopplerAmbiguity'):
                self.dopplerAmbiguity = z.text
            elif z.tag == self.elementName('dopplerCentroidCoefficients'):
                self.dopplerCentroidCoefficients = list(map(float,z.text.split()))
            elif z.tag == self.elementName('dopplerCentroidReferenceTime'):
                self.dopplerCentroidReferenceTime = float(z.text)

    def __str__(self):
        retstr = "DopplerCentroid:"+sep+tab
        retlst = ()
        retstr += "dopplerAmbiguity=%s"+sep+tab
        retlst += (self.dopplerAmbiguity,)
        retstr += "dopplerCentroidCoefficients=%s"+sep
        retlst += (self.dopplerCentroidCoefficients,)
        retstr += ":DopplerCentroid"+sep
        return retstr % retlst

class _DopplerRateValues(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.dopplerRateReferenceTime = None
        self.dopplerRateValuesCoefficients = []

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('dopplerRateReferenceTime'):
                self.dopplerRateReferenceTime = float(z.text)
            elif z.tag == self.elementName('dopplerRateValuesCoefficients'):
                self.dopplerRateValuesCoefficients = list(map(float,z.text.split()))

    def __str__(self):
        retstr = "DopplerRateValues:"+sep+tab
        retlst = ()
        retstr += "dopplerRateReferenceTime=%s"+sep+tab
        retlst += (self.dopplerRateReferenceTime,)
        retstr += "dopplerRateValuesCoefficients=%s"+sep+tab
        retlst += (self.dopplerRateValuesCoefficients,)
        retstr += ":DopplerRateValues"
        return retstr % retlst

class _Chirp(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)

class _SlantRangeToGroundRange(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.zeroDopplerAzimuthTime = None
        self.slantRangeTimeToFirstRangeSample = None
        self.groundRangeOrigin = None
        self.groundToSlantRangeCoefficients = []

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('zeroDopplerAzimuthTime'):
                self.zeroDopplerAzimuthTime = self.convertToDateTime(z.text)
            elif z.tag == self.elementName('slantRangeTimeToFirstRangeSample'):
                self.slantRangeTimeToFirstRangeSample = float(z.text)

class _ImageAttributes(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.productFormat = None
        self.outputMediaInterleaving = None
        self.rasterAttributes = _RasterAttributes()
        self.geographicInformation = _GeographicInformation()
        self.fullResolutionImageData = None

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('productFormat'):
                self.productFormat = z.text
            elif z.tag == self.elementName('outputMediaInterleaving'):
                self.outputMediaInterleaving = z.text
            elif z.tag == self.elementName('rasterAttributes'):
                self.rasterAttributes.set_from_etnode(z)
            elif z.tag == self.elementName('geographicInformation'):
                self.geographicInformation.set_from_etnode(z)
            elif z.tag == self.elementName('fullResolutionImageData'):
                self.fullResolutionImageData = z.text

class _RasterAttributes(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.dataType = None
        self.bitsPerSample = []
        self.numberOfSamplesPerLine = None
        self.numberOfLines = None
        self.sampledPixelSpacing = None
        self.sampledLineSpacing = None
        self.lineTimeOrdering = None
        self.pixelTimeOrdering = None

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('dataType'):
                self.dataType = z.text
            elif z.tag == self.elementName('bitsPerSample'):
                self.bitsPerSample.append(z.text)  # TODO: Make this a dictionary with keys of 'imaginary' and 'real'
            elif z.tag == self.elementName('numberOfSamplesPerLine'):
                self.numberOfSamplesPerLine = int(z.text)
            elif z.tag == self.elementName('numberOfLines'):
                self.numberOfLines = int(z.text)
            elif z.tag == self.elementName('sampledPixelSpacing'):
                self.sampledPixelSpacing = float(z.text)
            elif z.tag == self.elementName('sampledLineSpacing'):
                self.sampledLineSpacing = float(z.text)
            elif z.tag == self.elementName('lineTimeOrdering'):
                self.lineTimeOrdering = z.text
            elif z.tag == self.elementName('pixelTimeOrdering'):
                self.pixelTimeOrdering = z.text

class _GeographicInformation(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.geolocationGrid = _GeolocationGrid()
        self.rationalFunctions = _RationalFunctions()
        self.referenceEllipsoidParameters = _ReferenceEllipsoidParameters()

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('geolocationGrid'):
                self.geolocationGrid.set_from_etnode(z)

class _GeolocationGrid(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.imageTiePoint = []

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('imageTiePoint'):
                tp = _ImageTiePoint()
                tp.set_from_etnode(z)
                self.imageTiePoint.append(tp)


class _ImageTiePoint(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.imageCoordinates = _ImageCoordinates()
        self.geodeticCoordinates = _GeodeticCoordinates()

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('imageCoordinate'):
                self.imageCoordinates.set_from_etnode(z)
            elif z.tag == self.elementName('geodeticCoordinate'):
                self.geodeticCoordinates.set_from_etnode(z)

    def __str__(self):
        retstr = "ImageTiePoint:"+sep+tab
        retlst = ()
        retstr += "%s"
        retlst += (str(self.imageCoordinates),)
        retstr += "%s"
        retlst += (str(self.geodeticCoordinates),)
        retstr += ":ImageTiePoint"
        return retstr % retlst

class _ImageCoordinates(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.line = None
        self.pixel = None

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('line'):
                self.line = float(z.text)
            elif z.tag == self.elementName('pixel'):
                self.pixel = float(z.text)

    def __str__(self):
        retstr = "ImageCoordinate:"+sep+tab
        retlst = ()
        retstr += "line=%s"+sep+tab
        retlst += (self.line,)
        retstr += "pixel=%s"+sep+tab
        retlst += (self.pixel,)
        retstr += ":ImageCoordinate"
        return retstr % retlst

class _GeodeticCoordinates(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.latitude = None
        self.longitude = None
        self.height = None

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('latitude'):
                self.latitude = float(z.text)
            elif z.tag == self.elementName('longitude'):
                self.longitude = float(z.text)
            elif z.tag == self.elementName('height'):
                self.height = float(z.text)

    def __str__(self):
        retstr = "GeodeticCoordinate:"+sep+tab
        retlst = ()
        retstr += "latitude=%s"+sep+tab
        retlst += (self.latitude,)
        retstr += "longitude=%s"+sep+tab
        retlst += (self.longitude,)
        retstr += "height=%s"+sep+tab
        retlst += (self.height,)
        retstr += ":GeodeticCoordinate"
        return retstr % retlst

class _ReferenceEllipsoidParameters(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)
        self.ellipsoidName = None
        self.semiMajorAxis = None
        self.semiMinorAxis = None
        self.geodeticTerrainHeight = None

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == self.elementName('ellipsoidName'):
                self.ellipsoidName = z.text
            elif z.tag == self.elementName('semiMajorAxis'):
                self.semiMajorAxis = float(z.text)
            elif z.tag == self.elementName('semiMinorAxis'):
                self.semiMinorAxis = float(z.text)
            elif z.tag == self.elementName('geodeticTerrainHeight'):
                self.geodeticTerrainHeight = float(z.text)

    def __str__(self):
        return ""

class _RationalFunctions(Radarsat2Namespace):
    def __init__(self):
        Radarsat2Namespace.__init__(self)

    def set_from_etnode(self,node):
        pass

    def __str__(self):
        return ""


def findPreciseOrbit(dirname, fname, year):
    '''
    Find precise orbit file in given folder.
    '''

    import glob

    ###First try root folder itself
    res = glob.glob( os.path.join(dirname, fname.lower()))
    if len(res) == 0:

        res = glob.glob( os.path.join(dirname, "{0}".format(year), fname.lower()))
        if len(res) == 0:
            raise Exception('Orbit Dirname provided but no suitable orbit file found in {0}'.format(dirname))

    
    if len(res) > 1:
        print('More than one matching result found. Using first result.')

    return res[0]

def convertRSTimeToDateTime(instr):
    '''
    Convert RS2 orbit time string to datetime.
    '''

    parts = instr.strip().split('-')
    tparts = parts[-1].split(':')
    secs = float(tparts[2])
    intsecs = int(secs)
    musecs = int((secs - intsecs)*1e6)

    timestamp = datetime.datetime(int(parts[0]),1,1, int(tparts[0]), int(tparts[1]), intsecs, musecs) + datetime.timedelta(days = int(parts[1])-1)

    return timestamp
