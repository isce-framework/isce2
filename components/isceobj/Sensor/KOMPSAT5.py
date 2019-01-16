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


class KOMPSAT5(Component):
    """
    A class representing a Level1Product meta data.
    Level1Product(hdf5=h5filename) will parse the hdf5
    file and produce an object with attributes for metadata.
    """

    logging_name = 'isce.Sensor.KOMPSAT5'

    def __init__(self):
        super(KOMPSAT5,self).__init__()
        self.hdf5 = None
        self.output = None
        self.frame = Frame()
        self.frame.configure()
        # Some extra processing parameters unique to CSK SLC (currently)
        self.dopplerCoeffs = []
        self.rangeFirstTime = None
        self.rangeLastTime = None
        self.rangeRefTime = None
        self.refUTC = None
        
        self.descriptionOfVariables = {}
        self.dictionaryOfVariables = {'HDF5': ['self.hdf5','str','mandatory'],
                                      'OUTPUT': ['self.output','str','optional']}
        
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

        scale = file['S01'].attrs['PRF'] * file['S01/SBI'].attrs['Line Time Interval']
        self.dopplerCoeffs = file.attrs['Centroid vs Range Time Polynomial']*scale
        self.rangeRefTime = file.attrs['Range Polynomial Reference Time']
        self.rangeFirstTime = file['S01/SBI'].attrs['Zero Doppler Range First Time']
        self.rangeLastTime = file['S01/SBI'].attrs['Zero Doppler Range Last Time']

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
        return dobj + dt
       

    def extractDoppler(self):
        """
        Return the doppler centroid as defined in the HDF5 file.
        """
        quadratic = {}
        midtime = (self.rangeLastTime + self.rangeFirstTime)*0.5 - self.rangeRefTime
        fd_mid = self.dopplerCoeffs[0] + self.dopplerCoeffs[1]*midtime + self.dopplerCoeffs[2]*midtime*midtime

        quadratic['a'] = fd_mid/self.frame.getInstrument().getPulseRepetitionFrequency()
        quadratic['b'] = 0.
        quadratic['c'] = 0.
        return quadratic
