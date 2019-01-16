#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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
import logging
try:
    import h5py
except ImportError:
    raise ImportError(
            "Python module h5py is required to process COSMO-SkyMed data"
            )

import isceobj
from isceobj.Scene.Frame import Frame
from isceobj.Orbit.Orbit import StateVector
from isceobj.Planet.Planet import Planet
from isceobj.Planet.AstronomicalHandbook import Const
from isceobj.Sensor import cosar
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTU
from isceobj.Sensor import tkfunc,createAuxFile
from iscesys.Component.Component import Component

HDF5 = Component.Parameter(
    'hdf5',
    public_name='HDF5',
    default=None,
    type=str,
    mandatory=True,
    intent='input',
    doc='CSK slc hdf5 input file'
)

from .Sensor import Sensor
class COSMO_SkyMed_SLC(Sensor):
    """
    A class representing a Level1Product meta data.
    Level1Product(hdf5=h5filename) will parse the hdf5
    file and produce an object with attributes for metadata.
    """
    parameter_list = (HDF5,) + Sensor.parameter_list
    logging_name = 'isce.Sensor.COSMO_SkyMed_SLC'
    family = 'cosmo_skymed_slc'

    def __init__(self,family='',name=''):
        super(COSMO_SkyMed_SLC,self).__init__(family if family else  self.__class__.family, name=name)
        self.frame = Frame()
        self.frame.configure()
        # Some extra processing parameters unique to CSK SLC (currently)
        self.dopplerRangeTime = []
        self.dopplerAzimuthTime = []
        self.azimuthRefTime = None
        self.rangeRefTime = None
        self.rangeFirstTime = None
        self.rangeLastTime = None


        self.lookMap = {'RIGHT': -1,
                        'LEFT': 1}
        return

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self,d):
        self.__dict__.update(d)
        self.logger = logging.getLogger('isce.Sensor.COSMO_SkyMed_SLC')
        return


    def getFrame(self):
        return self.frame

    def parse(self):
        try:
            fp = h5py.File(self.hdf5,'r')
        except Exception as strerr:
            self.logger.error("IOError: %s" % strerr)
            return None

        self.populateMetadata(fp)
        fp.close()

    def populateMetadata(self, file):
        """
            Populate our Metadata objects
        """

        self._populatePlatform(file)
        self._populateInstrument(file)
        self._populateFrame(file)
        self._populateOrbit(file)
        self._populateExtras(file)
        

    def _populatePlatform(self, file):
        platform = self.frame.getInstrument().getPlatform()

        platform.setMission(file.attrs['Satellite ID'])
        platform.setPointingDirection(self.lookMap[file.attrs['Look Side'].decode('utf-8')])
        platform.setPlanet(Planet(pname="Earth"))

        ####This is an approximation for spotlight mode
        ####In spotlight mode, antenna length changes with azimuth position
        platform.setAntennaLength(file.attrs['Antenna Length'])
        try:
            if file.attrs['Multi-Beam ID'].startswith('ES'):
                platform.setAntennaLength(16000.0/file['S01/SBI'].attrs['Line Time Interval'])
        except:
            pass

    def _populateInstrument(self, file):
        instrument = self.frame.getInstrument()

#        rangePixelSize = Const.c/(2*file['S01'].attrs['Sampling Rate'])
        rangePixelSize = file['S01/SBI'].attrs['Column Spacing']
        instrument.setRadarWavelength(file.attrs['Radar Wavelength'])
#        instrument.setPulseRepetitionFrequency(file['S01'].attrs['PRF'])
        instrument.setPulseRepetitionFrequency(1.0/file['S01/SBI'].attrs['Line Time Interval'])
        instrument.setRangePixelSize(rangePixelSize)
        instrument.setPulseLength(file['S01'].attrs['Range Chirp Length'])
        instrument.setChirpSlope(file['S01'].attrs['Range Chirp Rate'])
#        instrument.setRangeSamplingRate(file['S01'].attrs['Sampling Rate'])
        instrument.setRangeSamplingRate(1.0/file['S01/SBI'].attrs['Column Time Interval'])

        incangle = 0.5*(file['S01/SBI'].attrs['Far Incidence Angle'] +
                 file['S01/SBI'].attrs['Near Incidence Angle'])
        instrument.setIncidenceAngle(incangle)


    def _populateFrame(self, file):

        rft = file['S01/SBI'].attrs['Zero Doppler Range First Time']
        slantRange = rft*Const.c/2.0
        self.frame.setStartingRange(slantRange)

        referenceUTC = self._parseNanoSecondTimeStamp(file.attrs['Reference UTC'])
        relStart = file['S01/SBI'].attrs['Zero Doppler Azimuth First Time']
        relEnd = file['S01/SBI'].attrs['Zero Doppler Azimuth Last Time']
        relMid = 0.5*(relStart + relEnd)

        sensingStart = self._combineDateTime(referenceUTC, relStart)
        sensingStop = self._combineDateTime(referenceUTC, relEnd)
        sensingMid = self._combineDateTime(referenceUTC, relMid)


        self.frame.setPassDirection(file.attrs['Orbit Direction'])
        self.frame.setOrbitNumber(file.attrs['Orbit Number'])
        self.frame.setProcessingFacility(file.attrs['Processing Centre'])
        self.frame.setProcessingSoftwareVersion(file.attrs['L0 Software Version'])
        self.frame.setPolarization(file['S01'].attrs['Polarisation'])
        self.frame.setNumberOfLines(file['S01/SBI'].shape[0])
        self.frame.setNumberOfSamples(file['S01/SBI'].shape[1])
        self.frame.setSensingStart(sensingStart)
        self.frame.setSensingMid(sensingMid)
        self.frame.setSensingStop(sensingStop)

        rangePixelSize = self.frame.getInstrument().getRangePixelSize()
        farRange = slantRange +  (self.frame.getNumberOfSamples()-1)*rangePixelSize
        self.frame.setFarRange(farRange)

    def _populateOrbit(self,file):
        orbit = self.frame.getOrbit()

        orbit.setReferenceFrame('ECR')
        orbit.setOrbitSource('Header')
        t0 = datetime.datetime.strptime(file.attrs['Reference UTC'].decode('utf-8'),'%Y-%m-%d %H:%M:%S.%f000')
        t = file.attrs['State Vectors Times']
        position = file.attrs['ECEF Satellite Position']
        velocity = file.attrs['ECEF Satellite Velocity']

        for i in range(len(position)):
            vec = StateVector()
            dt = t0 + datetime.timedelta(seconds=t[i])
            vec.setTime(dt)
            vec.setPosition([position[i,0],position[i,1],position[i,2]])
            vec.setVelocity([velocity[i,0],velocity[i,1],velocity[i,2]])
            orbit.addStateVector(vec)


    def _populateExtras(self, file):
        """
        Populate some of the extra fields unique to processing TSX data.
        In the future, other sensors may need this information as well,
        and a re-organization may be necessary.
        """
        from isceobj.Doppler.Doppler import Doppler

        self.dopplerRangeTime = file.attrs['Centroid vs Range Time Polynomial']
        self.dopplerAzimuthTime = file.attrs['Centroid vs Azimuth Time Polynomial']
        self.rangeRefTime = file.attrs['Range Polynomial Reference Time']
        self.azimuthRefTime = file.attrs['Azimuth Polynomial Reference Time']
        self.rangeFirstTime = file['S01/SBI'].attrs['Zero Doppler Range First Time']
        self.rangeLastTime = file['S01/SBI'].attrs['Zero Doppler Range Last Time']
        
        # get Doppler rate information, vs. azimuth first EJF 2015/00/05
        # guessing that same scale applies as for Doppler centroid 
        self.dopplerRateCoeffs = file.attrs['Doppler Rate vs Azimuth Time Polynomial']
        
    def extractImage(self):
        import os
        from ctypes import cdll, c_char_p
        extract_csk = cdll.LoadLibrary(os.path.dirname(__file__)+'/csk.so')
        inFile_c = c_char_p(bytes(self.hdf5, 'utf-8'))
        outFile_c = c_char_p(bytes(self.output, 'utf-8'))

        extract_csk.extract_csk_slc(inFile_c, outFile_c)

        self.parse()
        slcImage = isceobj.createSlcImage()
        slcImage.setFilename(self.output)
        slcImage.setXmin(0)
        slcImage.setXmax(self.frame.getNumberOfSamples())
        slcImage.setWidth(self.frame.getNumberOfSamples())
        slcImage.setAccessMode('r')
        self.frame.setImage(slcImage)


    def _parseNanoSecondTimeStamp(self,timestamp):
        """
            Parse a date-time string with nanosecond precision and return a datetime object
        """
        dateTime,nanoSeconds = timestamp.decode('utf-8').split('.')
        microsec = float(nanoSeconds)*1e-3
        dt = datetime.datetime.strptime(dateTime,'%Y-%m-%d %H:%M:%S')
        dt = dt + datetime.timedelta(microseconds=microsec)
        return dt

    def _combineDateTime(self,dobj, secsstr):
        '''Takes the date from dobj and time from secs to spit out a date time object.
        '''
        sec = float(secsstr)
        dt = datetime.timedelta(seconds = sec)
        return datetime.datetime.combine(dobj.date(), datetime.time(0,0)) + dt


    def extractDoppler(self):
        """
        Return the doppler centroid as defined in the HDF5 file.
        """
        import numpy as np

        quadratic = {}
        midtime = (self.rangeLastTime + self.rangeFirstTime)*0.5 - self.rangeRefTime

        fd_mid = 0.0
        x = 1.0
        for ind, coeff in enumerate(self.dopplerRangeTime):
            fd_mid += coeff*x
            x *= midtime


        ####insarApp style
        quadratic['a'] = fd_mid/self.frame.getInstrument().getPulseRepetitionFrequency()
        quadratic['b'] = 0.
        quadratic['c'] = 0.


        ####For roiApp more accurate
        ####Convert stuff to pixel wise coefficients
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

#EMG - 20160420 This section was introduced in the populateMetadata method by EJF in r2022
#Its pupose seems to be to set self.doppler_coeff and self.azfmrate_coeff, which don't seem
#to be used anywhere in ISCE. Need to take time to understand the need for this and consult
#with EJF.
#
## save the Doppler centroid coefficients, converting units from .h5 file
## units in the file are quadratic coefficients in Hz, Hz/sec, and Hz/(sec^2)
## ISCE expects Hz, Hz/(range sample), Hz/(range sample)^2
## note that RS2 Doppler values are estimated at time dc.dopplerCentroidReferenceTime,
## so the values might need to be adjusted for ISCE usage
## adapted from RS2 version EJF 2015/09/05
#        poly = self.frame._dopplerVsPixel
#        rangeSamplingRate = self.frame.getInstrument().getPulseRepetitionFrequency()
#        # need to convert units 
#        poly[1] = poly[1]/rangeSamplingRate
#        poly[2] = poly[2]/rangeSamplingRate**2
#        self.doppler_coeff = poly
#        
## similarly save Doppler azimuth fm rate values, converting units
## units in the file are quadratic coefficients in Hz, Hz/sec, and Hz/(sec^2)
## units are already converted below
## Guessing that ISCE expects Hz, Hz/(azimuth line), Hz/(azimuth line)^2
## note that RS2 Doppler values are estimated at time dc.dopplerRateReferenceTime,
## so the values might need to be adjusted for ISCE usage
## modified from RS2 version EJF 2015/09/05
## CSK Doppler azimuth FM rate not yet implemented in reading section, set to zero for now
#
#        fmpoly = self.dopplerRateCoeffs
#        # don't need to convert units 
##        fmpoly[1] = fmpoly[1]/rangeSamplingRate
##        fmpoly[2] = fmpoly[2]/rangeSamplingRate**2
#        self.azfmrate_coeff = fmpoly
#EMG - 20160420

        return quadratic
