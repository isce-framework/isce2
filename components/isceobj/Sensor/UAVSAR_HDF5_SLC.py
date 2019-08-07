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
# Author: Heresh Fattahi
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
from isceobj.Constants import SPEED_OF_LIGHT

HDF5 = Component.Parameter(
    'hdf5',
    public_name='HDF5',
    default=None,
    type=str,
    mandatory=True,
    intent='input',
    doc='UAVSAR slc input file in HDF5 format'
)

FREQUENCY = Component.Parameter(
    'frequency',
    public_name='FREQUENCY',
    default='frequencyA',
    type=str,
    mandatory=True,
    intent='input',
    doc='frequency band of the UAVSAR slc file to be processed (frequencyA or frequencyB)'
)

POLARIZATION = Component.Parameter(
    'polarization',
    public_name='POLARIZATION',
    default='HH',
    type=str,
    mandatory=True,
    intent='input',
    doc='polarization channel of the UAVSAR slc file to be processed'
)

from .Sensor import Sensor
class UAVSAR_HDF5_SLC(Sensor):
    """
    A class representing a Level1Product meta data.
    Level1Product(hdf5=h5filename) will parse the hdf5
    file and produce an object with attributes for metadata.
    """
    parameter_list = (HDF5,
                      FREQUENCY,
                      POLARIZATION) + Sensor.parameter_list

    logging_name = 'isce.Sensor.UAVSAR_HDF5_SLC'
    family = 'uavsar_hdf5_slc'

    def __init__(self,family='',name=''):# , frequency='frequencyA', polarization='HH'):
        super(UAVSAR_HDF5_SLC,self).__init__(family if family else  self.__class__.family, name=name)
        self.frame = Frame()
        self.frame.configure()
        # Some extra processing parameters unique to UAVSAR HDF5 SLC (currently)
        self.dopplerRangeTime = []
        self.dopplerAzimuthTime = []
        self.azimuthRefTime = None
        self.rangeRefTime = None
        self.rangeFirstTime = None
        self.rangeLastTime = None
        #self.frequency = frequency
        #self.polarization = polarization

        self.lookMap = {'right': -1,
                        'left': 1}
        return

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self,d):
        self.__dict__.update(d)
        self.logger = logging.getLogger('isce.Sensor.UAVSAR_HDF5_SLC')
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
        

    def _populatePlatform(self, file):
        platform = self.frame.getInstrument().getPlatform()

        platform.setMission(file['/science/LSAR/identification'].get('missionId')[()].decode('utf-8'))
        platform.setPointingDirection(self.lookMap[file['/science/LSAR/identification'].get('lookDirection')[()].decode('utf-8')])
        platform.setPlanet(Planet(pname="Earth"))

        # We are not using this value anywhere. Let's fix it for now.
        platform.setAntennaLength(12.0)

    def _populateInstrument(self, file):
        instrument = self.frame.getInstrument()

        rangePixelSize = file['/science/LSAR/SLC/swaths/' + self.frequency + '/slantRangeSpacing'][()]
        wvl = SPEED_OF_LIGHT/file['/science/LSAR/SLC/swaths/' + self.frequency + '/processedCenterFrequency'][()]
        instrument.setRadarWavelength(wvl)
        instrument.setPulseRepetitionFrequency(1.0/file['/science/LSAR/SLC/swaths/zeroDopplerTimeSpacing'][()])
        rangePixelSize = file['/science/LSAR/SLC/swaths/' + self.frequency + '/slantRangeSpacing'][()]
        instrument.setRangePixelSize(rangePixelSize)

        # Chrip slope and length only are used in the split spectrum workflow to compute the bandwidth.
        # Therefore fixing it to 1.0 won't breack anything
        Chirp_slope = 1.0
        rangeBandwidth = file['/science/LSAR/SLC/swaths/' + self.frequency + '/processedRangeBandwidth'][()]
        Chirp_length = rangeBandwidth/Chirp_slope
        instrument.setPulseLength(Chirp_length)
        instrument.setChirpSlope(Chirp_slope)
        rangeSamplingFrequency = SPEED_OF_LIGHT/2./rangePixelSize
        instrument.setRangeSamplingRate(rangeSamplingFrequency)

        incangle = 0.0
        instrument.setIncidenceAngle(incangle)


    def _populateFrame(self, file):

        slantRange = file['/science/LSAR/SLC/swaths/' + self.frequency + '/slantRange'][0]
        self.frame.setStartingRange(slantRange)

        referenceUTC = file['/science/LSAR/SLC/swaths/zeroDopplerTime'].attrs['units'].decode('utf-8')
        referenceUTC = referenceUTC.replace('seconds since ','')
        format_str = '%Y-%m-%d %H:%M:%S'
        if '.' in referenceUTC:
            format_str += '.%f'
        referenceUTC = datetime.datetime.strptime(referenceUTC, format_str)

        relStart = file['/science/LSAR/SLC/swaths/zeroDopplerTime'][0]
        relEnd = file['/science/LSAR/SLC/swaths/zeroDopplerTime'][-1]
        relMid = 0.5*(relStart + relEnd)

        sensingStart = self._combineDateTime(referenceUTC, relStart)
        sensingStop = self._combineDateTime(referenceUTC, relEnd)
        sensingMid = self._combineDateTime(referenceUTC, relMid)


        self.frame.setPassDirection(file['/science/LSAR/identification'].get('orbitPassDirection')[()].decode('utf-8'))  
        self.frame.setOrbitNumber(file['/science/LSAR/identification'].get('trackNumber')[()])
        self.frame.setProcessingFacility('JPL')
        self.frame.setProcessingSoftwareVersion(file['/science/LSAR/SLC/metadata/processingInformation/algorithms'].get('ISCEVersion')[()].decode('utf-8'))
        self.frame.setPolarization(self.polarization)
        self.frame.setNumberOfLines(file['/science/LSAR/SLC/swaths/' + self.frequency + '/' + self.polarization].shape[0])
        self.frame.setNumberOfSamples(file['/science/LSAR/SLC/swaths/' + self.frequency + '/' + self.polarization].shape[1])
        self.frame.setSensingStart(sensingStart)
        self.frame.setSensingMid(sensingMid)
        self.frame.setSensingStop(sensingStop)

        rangePixelSize = self.frame.instrument.rangePixelSize
        farRange = slantRange +  (self.frame.getNumberOfSamples()-1)*rangePixelSize
        self.frame.setFarRange(farRange)

    def _populateOrbit(self,file):
        orbit = self.frame.getOrbit()

        orbit.setReferenceFrame('ECR')
        orbit.setOrbitSource('Header')

        referenceUTC = file['/science/LSAR/SLC/swaths/zeroDopplerTime'].attrs['units'].decode('utf-8')
        referenceUTC = referenceUTC.replace('seconds since ','')
        format_str = '%Y-%m-%d %H:%M:%S'
        if '.' in referenceUTC:
            format_str += '.%f'
        t0 = datetime.datetime.strptime(referenceUTC, format_str)
        t = file['/science/LSAR/SLC/metadata/orbit/time']
        position = file['/science/LSAR/SLC/metadata/orbit/position']
        velocity = file['/science/LSAR/SLC/metadata/orbit/velocity']

        for i in range(len(position)):
            vec = StateVector()
            dt = t0 + datetime.timedelta(seconds=t[i])
            vec.setTime(dt)
            vec.setPosition([position[i,0],position[i,1],position[i,2]])
            vec.setVelocity([velocity[i,0],velocity[i,1],velocity[i,2]])
            orbit.addStateVector(vec)


    def extractImage(self):

        import numpy as np
        import h5py

        self.parse()
        
        fid = h5py.File(self.hdf5, 'r')
        ds = fid['/science/LSAR/SLC/swaths/' + self.frequency + '/' + self.polarization]
        nLines = ds.shape[0]

        # force casting to complex64
        with ds.astype(np.complex64):
            with open(self.output, 'wb') as fout:
                for ii in range(nLines):
                    ds[ii, :].tofile(fout)

        fid.close()
  
        slcImage = isceobj.createSlcImage()
        slcImage.setFilename(self.output)
        slcImage.setXmin(0)
        slcImage.setXmax(self.frame.getNumberOfSamples())
        slcImage.setWidth(self.frame.getNumberOfSamples())
        slcImage.setAccessMode('r')
        slcImage.renderHdr()
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

        import h5py
        from scipy.interpolate import UnivariateSpline
        import numpy as np

        h5 = h5py.File(self.hdf5,'r')

        # extract the 2D LUT of Doppler and choose only one range line as the data duplicates for other range lines
        dop = h5['/science/LSAR/SLC/metadata/processingInformation/parameters/' + self.frequency + '/dopplerCentroid'][0,:]
        rng = h5['/science/LSAR/SLC/metadata/processingInformation/parameters/slantRange']

        # extract the slant range of the image grid
        imgRng = h5['/science/LSAR/SLC/swaths/' + self.frequency + '/slantRange']

        # use only part of the slant range that closely covers image ranges and ignore the rest
        ind0 = np.argmin(np.abs(rng-imgRng[0])) - 1
        ind0 = np.max([0,ind0])
        ind1 = np.argmin(np.abs(rng-imgRng[-1])) + 1
        ind1 = np.min([ind1, rng.shape[0]])

        dop = dop[ind0:ind1]
        rng = rng[ind0:ind1]

        f = UnivariateSpline(rng, dop)
        imgDop = f(imgRng)

        dr = imgRng[1]-imgRng[0]
        pix = (imgRng - imgRng[0])/dr
        fit = np.polyfit(pix, imgDop, 41)

        self.frame._dopplerVsPixel = list(fit[::-1])

        ####insarApp style (doesn't get used for stripmapApp). A fixed Doppler at the middle of the scene
        quadratic = {}
        quadratic['a'] = imgDop[int(imgDop.shape[0]/2)]/self.frame.getInstrument().getPulseRepetitionFrequency()
        quadratic['b'] = 0.
        quadratic['c'] = 0.

        return quadratic
