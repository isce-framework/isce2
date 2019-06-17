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
            "Python module h5py is required to process ICEYE data"
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
    doc='ICEYE slc hdf5 input file'
)

APPLY_SLANT_RANGE_PHASE = Component.Parameter(
        'applySlantRangePhase',
        public_name='APPLY_SLANT_RANGE_PHASE',
        default=False,
        type=bool,
        mandatory=True,
        intent='input',
        doc='Recenter spectra by applying range spectra shift'
)

from .Sensor import Sensor
class ICEYE_SLC(Sensor):
    """
    A class representing a Level1Product meta data.
    Level1Product(hdf5=h5filename) will parse the hdf5
    file and produce an object with attributes for metadata.
    """
    parameter_list = (HDF5, APPLY_SLANT_RANGE_PHASE) + Sensor.parameter_list
    logging_name = 'isce.Sensor.ICEYE_SLC'
    family = 'iceye_slc'

    def __init__(self,family='',name=''):
        super(ICEYE_SLC,self).__init__(family if family else  self.__class__.family, name=name)
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
        self.logger = logging.getLogger('isce.Sensor.ICEYE_SLC')
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

        platform.setMission(file['satellite_name'][()])
        platform.setPointingDirection(self.lookMap[file['look_side'][()].upper()])
        platform.setPlanet(Planet(pname="Earth"))

        ####This is an approximation for spotlight mode
        ####In spotlight mode, antenna length changes with azimuth position
        platform.setAntennaLength(2 * file['azimuth_ground_spacing'][()])

        assert( file['range_looks'][()] == 1)
        assert( file['azimuth_looks'][()] == 1)

    def _populateInstrument(self, file):
        instrument = self.frame.getInstrument()

        rangePixelSize = file['slant_range_spacing'][()]
        instrument.setRadarWavelength(Const.c / file['carrier_frequency'][()])
        instrument.setPulseRepetitionFrequency(file['processing_prf'][()])
        instrument.setRangePixelSize(rangePixelSize)
        instrument.setPulseLength(file['chirp_duration'][()])
        instrument.setChirpSlope(file['chirp_bandwidth'][()]/ file['chirp_duration'][()])
        instrument.setRangeSamplingRate(file['range_sampling_rate'][()])

        incangle = file['local_incidence_angle']
        instrument.setIncidenceAngle(incangle[incangle.size//2])


    def _populateFrame(self, file):

        rft = file['first_pixel_time'][()]
        slantRange = rft*Const.c/2.0
        self.frame.setStartingRange(slantRange)


        sensingStart = datetime.datetime.strptime(file['zerodoppler_start_utc'][()].decode('utf-8'),'%Y-%m-%dT%H:%M:%S.%f')
        sensingStop = datetime.datetime.strptime(file['zerodoppler_end_utc'][()].decode('utf-8'),'%Y-%m-%dT%H:%M:%S.%f')
        sensingMid = sensingStart + 0.5 * (sensingStop - sensingStart)

        self.frame.setPassDirection(file['orbit_direction'][()])
        self.frame.setOrbitNumber(file['orbit_absolute_number'][()])
        self.frame.setProcessingFacility('ICEYE')
        self.frame.setProcessingSoftwareVersion(str(file['processor_version'][()]))
        self.frame.setPolarization(file['polarization'][()])
        self.frame.setNumberOfLines(file['number_of_azimuth_samples'][()])
        self.frame.setNumberOfSamples(file['number_of_range_samples'][()])
        self.frame.setSensingStart(sensingStart)
        self.frame.setSensingMid(sensingMid)
        self.frame.setSensingStop(sensingStop)

        rangePixelSize = self.frame.getInstrument().getRangePixelSize()
        farRange = slantRange +  (self.frame.getNumberOfSamples()-1)*rangePixelSize
        self.frame.setFarRange(farRange)

    def _populateOrbit(self,file):
        import numpy as np
        orbit = self.frame.getOrbit()

        orbit.setReferenceFrame('ECR')
        orbit.setOrbitSource('Header')
        t = file['state_vector_time_utc'][:]
        position = np.zeros((t.size,3))
        position[:,0] = file['posX'][:]
        position[:,1] = file['posY'][:]
        position[:,2] = file['posZ'][:]

        velocity = np.zeros((t.size,3))
        velocity[:,0] = file['velX'][:]
        velocity[:,1] = file['velY'][:]
        velocity[:,2] = file['velZ'][:]

        for ii in range(t.size):
            vec = StateVector()
            vec.setTime(datetime.datetime.strptime(t[ii][0].decode('utf-8'), '%Y-%m-%dT%H:%M:%S.%f'))
            vec.setPosition([position[ii,0],position[ii,1],position[ii,2]])
            vec.setVelocity([velocity[ii,0],velocity[ii,1],velocity[ii,2]])
            orbit.addStateVector(vec)


    def _populateExtras(self, file):
        """
        Populate some of the extra fields unique to processing TSX data.
        In the future, other sensors may need this information as well,
        and a re-organization may be necessary.
        """
        import numpy as np
        self.dcpoly = np.mean(file['dc_estimate_coeffs'][:], axis=0)
        
    def extractImage(self):
        import numpy as np
        import h5py

        self.parse()

        fid = h5py.File(self.hdf5, 'r')

        si = fid['s_i']
        sq = fid['s_q']

        nLines = si.shape[0]
        spectralShift = 2 * self.frame.getInstrument().getRangePixelSize() / self.frame.getInstrument().getRadarWavelength() 
        spectralShift -= np.floor(spectralShift)
        phsShift = np.exp(-1j * 2 * np.pi * spectralShift * np.arange(si.shape[1])) 
        with open(self.output, 'wb') as fout:
            for ii in range(nLines):
                line = (si[ii,:] + 1j*sq[ii,:])
                if self.applySlantRangePhase:
                    line *= phsShift
                line.astype(np.complex64).tofile(fout)
        
        fid.close()

        slcImage = isceobj.createSlcImage()
        slcImage.setFilename(self.output)
        slcImage.setXmin(0)
        slcImage.setXmax(self.frame.getNumberOfSamples())
        slcImage.setWidth(self.frame.getNumberOfSamples())
        slcImage.setAccessMode('r')
        self.frame.setImage(slcImage)

    def extractDoppler(self):
        """
        Return the doppler centroid as defined in the HDF5 file.
        """
        import numpy as np

        quadratic = {}
    
        rangePixelSize = self.frame.getInstrument().getRangePixelSize()
        rt0 = self.frame.getStartingRange() / (2 * Const.c)
        rt1 = rt0 +((self.frame.getNumberOfSamples()-1)*rangePixelSize) / (2 * Const.c) 


        ####insarApp style
        quadratic['a'] = np.polyval( self.dcpoly, 0.5 * (rt0 + rt1)) / self.frame.PRF
        quadratic['b'] = 0.
        quadratic['c'] = 0.


        ####For roiApp more accurate
        ####Convert stuff to pixel wise coefficients
        x = np.linspace(rt0, rt1, num=len(self.dcpoly)+1)
        pix = np.linspace(0, self.frame.getNumberOfSamples(), num=len(self.dcpoly)+1)
        evals = np.polyval(self.dcpoly, x)
        fit = np.polyfit(pix, evals, len(self.dcpoly)-1)
        self.frame._dopplerVsPixel = list(fit[::-1])
        print('Doppler Fit: ', self.frame._dopplerVsPixel)

        return quadratic
