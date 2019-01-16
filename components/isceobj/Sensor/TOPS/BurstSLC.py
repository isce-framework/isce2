#!/usr/bin/env python3

import isce
import datetime
import isceobj
import numpy as np
from iscesys.Component.Component import Component
from isceobj.Image.Image import Image
from isceobj.Orbit.Orbit import Orbit
from isceobj.Util.decorators import type_check
from iscesys.Traits import datetimeType


####List of parameters
NUMBER_OF_SAMPLES = Component.Parameter('numberOfSamples',
        public_name='number of samples',
        default=None,
        type=int,
        mandatory=True,
        doc='Width of the burst slc')

NUMBER_OF_LINES = Component.Parameter('numberOfLines',
        public_name='number of lines',
        default=None,
        type=int,
        mandatory=True,
        doc='Length of the burst slc')

STARTING_RANGE = Component.Parameter('startingRange',
        public_name='starting range',
        default=None,
        type=float,
        mandatory=True,
        doc='Slant range to first pixel in m')

SENSING_START = Component.Parameter('sensingStart',
        public_name='sensing start',
        default=None,
        type=datetimeType,
        mandatory=True,
        doc='UTC time corresponding to first line of burst SLC')

SENSING_STOP = Component.Parameter('sensingStop',
        public_name='sensing stop',
        default=None,
        type=datetimeType,
        mandatory=True,
        doc='UTC time corresponding to last line of burst SLC')

BURST_START_UTC = Component.Parameter('burstStartUTC',
        public_name = 'burst start utc',
        default=None,
        type=datetimeType,
        mandatory=True,
        doc='Actual sensing time corresponding to start of the burst')

BURST_STOP_UTC = Component.Parameter('burstStopUTC',
        public_name = 'burst stop utc',
        default = None,
        type=datetimeType,
        mandatory=True,
        doc='Actual sensing time corresponding to end of the burst')

TRACK_NUMBER = Component.Parameter('trackNumber',
        public_name = 'track number',
        default = None,
        type=int,
        mandatory = True,
        doc = 'Track number for bookkeeping')

FRAME_NUMBER = Component.Parameter('frameNumber',
        public_name = 'frame number',
        default = None,
        type =int,
        mandatory=True,
        doc = 'Frame number for bookkeeping')

ORBIT_NUMBER = Component.Parameter('orbitNumber',
        public_name = 'orbit number',
        default = None,
        type = int,
        mandatory = True,
        doc = 'Orbit number for bookkeeping')

SWATH_NUMBER = Component.Parameter('swathNumber',
        public_name = 'swath number',
        default = None,
        type = int,
        mandatory = True,
        doc = 'Swath number for bookkeeping')

BURST_NUMBER = Component.Parameter('burstNumber',
        public_name = 'burst number',
        default = None,
        type = int,
        mandatory = True,
        doc = 'Burst number for bookkeeping')

PASS_DIRECTION = Component.Parameter('passDirection',
        public_name='pass direction',
        default = None,
        type=str,
        mandatory=True,
        doc = 'Ascending or descending')

AZIMUTH_STEERING_RATE = Component.Parameter('azimuthSteeringRate',
        public_name = 'azimuth steering rate',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Azimuth steering rate in radians per sec')

RANGE_PIXEL_SIZE = Component.Parameter('rangePixelSize',
        public_name = 'range pixel size',
        default = None,
        type=float,
        mandatory = True,
        doc = 'Slant range pixel size in m')

RANGE_SAMPLING_RATE = Component.Parameter('rangeSamplingRate',
        public_name = 'range sampling rate',
        default = None,
        type = float,
         mandatory = True,
         doc = 'Range sampling rate in Hz')

AZIMUTH_TIME_INTERVAL = Component.Parameter('azimuthTimeInterval',
        public_name = 'azimuth time interval',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Azimuth time interval between lines in seconds')

RADAR_WAVELENGTH = Component.Parameter('radarWavelength',
        public_name = 'radarWavelength',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Radar wavelength in m')

POLARIZATION = Component.Parameter('polarization',
        public_name = 'polarization',
        default = None,
        type = str,
        mandatory = True,
        doc = 'Polarization')

TERRAIN_HEIGHT = Component.Parameter('terrainHeight',
        public_name = 'terrain height',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Average terrain height used for focusing')

PRF = Component.Parameter('prf',
        public_name = 'pulse repetition frequency',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Pulse repetition frequency in Hz')

FIRST_VALID_LINE = Component.Parameter('firstValidLine',
        public_name = 'first valid line',
        default = None,
        type = int,
        mandatory = True,
        doc = 'First valid line in the burst SLC')

NUMBER_VALID_LINES = Component.Parameter('numValidLines',
        public_name = 'number of valid lines',
        default = None,
        type = int,
        mandatory = True,
        doc = 'Number of valid lines in the burst SLC')

FIRST_VALID_SAMPLE = Component.Parameter('firstValidSample',
        public_name = 'first valid sample',
        default = None,
        type = int,
        mandatory = True,
        doc = 'First valid sample in the burst SLC')

NUMBER_VALID_SAMPLES = Component.Parameter('numValidSamples',
        public_name = 'number of valid samples',
        default = None,
        type = int,
        mandatory = True,
        doc = 'Number of valid samples in the burst SLC')

#add these for doing bandpass filtering, Cunren Liang, 27-FEB-2018
RANGE_WINDOW_TYPE = Component.Parameter('rangeWindowType',
        public_name='range window type',
        default = None,
        type=str,
        mandatory=True,
        doc = 'Range weight window type')

RANGE_WINDOW_COEEFICIENT = Component.Parameter('rangeWindowCoefficient',
        public_name = 'range window coefficient',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Range weight window coefficient')

RANGE_PROCESSING_BANDWIDTH = Component.Parameter('rangeProcessingBandwidth',
        public_name = 'range processing bandwidth',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Range processing bandwidth in Hz')

AZIMUTH_WINDOW_TYPE = Component.Parameter('azimuthWindowType',
        public_name='azimuth window type',
        default = None,
        type=str,
        mandatory=True,
        doc = 'Azimuth weight window type')

AZIMUTH_WINDOW_COEEFICIENT = Component.Parameter('azimuthWindowCoefficient',
        public_name = 'azimuth window coefficient',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Azimuth weight window coefficient')

AZIMUTH_PROCESSING_BANDWIDTH = Component.Parameter('azimuthProcessingBandwidth',
        public_name = 'azimuth processing bandwidth',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Azimuth processing bandwidth in Hz')


####List of facilities
ORBIT = Component.Facility('orbit',
    public_name='orbit',
    module='isceobj.Orbit.Orbit',
    factory='createOrbit',
    args=(),
    doc = 'Orbit information')

IMAGE = Component.Facility('image',
        public_name='image',
        module='isceobj.Image',
        factory='createSlcImage',
        args = (),
        doc = 'Image on disk')

DOPPLER = Component.Facility('doppler',
        public_name='doppler',
        module = 'isceobj.Util.PolyFactory',
        factory = 'createPoly',
        args=('1d',),
        doc = 'Doppler polynomial')

AZIMUTH_FM_RATE = Component.Facility('azimuthFMRate',
        public_name = 'azimuthFMRate',
        module = 'isceobj.Util.PolyFactory',
        factory = 'createPoly',
        args = ('1d'),
        doc = 'Azimuth FM rate polynomial')

class BurstSLC(Component):
    """A class to represent a burst SLC along a radar track"""
    
    family = 'burstslc'
    logging_name = 'isce.burstslc'

    parameter_list = (NUMBER_OF_LINES,
                      NUMBER_OF_SAMPLES,
                      STARTING_RANGE,
                      SENSING_START,
                      SENSING_STOP,
                      BURST_START_UTC,
                      BURST_STOP_UTC,
                      TRACK_NUMBER,
                      FRAME_NUMBER,
                      ORBIT_NUMBER,
                      SWATH_NUMBER,
                      BURST_NUMBER,
                      RANGE_PIXEL_SIZE,
                      AZIMUTH_TIME_INTERVAL,
                      PASS_DIRECTION,
                      AZIMUTH_STEERING_RATE,
                      RADAR_WAVELENGTH,
                      PRF,
                      POLARIZATION,
                      TERRAIN_HEIGHT,
                      FIRST_VALID_LINE,
                      NUMBER_VALID_LINES,
                      FIRST_VALID_SAMPLE,
                      NUMBER_VALID_SAMPLES,
                      RANGE_WINDOW_TYPE,
                      RANGE_WINDOW_COEEFICIENT,
                      RANGE_PROCESSING_BANDWIDTH,
                      AZIMUTH_WINDOW_TYPE,
                      AZIMUTH_WINDOW_COEEFICIENT,
                      AZIMUTH_PROCESSING_BANDWIDTH,
                    )


    facility_list = (ORBIT,
                     IMAGE,
                     DOPPLER,
                     AZIMUTH_FM_RATE,)



    def __init__(self,name=''):
        super(BurstSLC, self).__init__(family=self.__class__.family, name=name)
        return None


    @property
    def lastValidLine(self):
        return self.firstValidLine + self.numValidLines

    @property
    def lastValidSample(self):
        return self.firstValidSample + self.numValidSamples

    @property
    def sensingMid(self):
        return self.sensingStart + 0.5 * (self.sensingStop - self.sensingStart)

    @property
    def burstMidUTC(self):
        return self.burstStartUTC + 0.5 * (self.burstStopUTC - self.burstStartUTC)

    @property
    def farRange(self):
        return self.startingRange + (self.numberOfSamples-1) * self.rangePixelSize

    @property
    def midRange(self):
        return 0.5 * (self.startingRange + self.farRange)

    def getBbox(self ,hgtrange=[-500,9000]):
        '''
        Bounding box estimate.
        '''

        ts = [self.sensingStart, self.sensingStop]
        rngs = [self.startingRange, self.farRange]
       
        pos = []
        for ht in hgtrange:
            for tim in ts:
                for rng in rngs:
                    llh = self.orbit.rdr2geo(tim, rng, height=ht)
                    pos.append(llh)

        pos = np.array(pos)

        bbox = [np.min(pos[:,0]), np.max(pos[:,0]), np.min(pos[:,1]), np.max(pos[:,1])]
        return bbox

    def clone(self):
        import copy
        res = copy.deepcopy(self)
        res.image._accessor = None
        res.image._factory = None

        return res

