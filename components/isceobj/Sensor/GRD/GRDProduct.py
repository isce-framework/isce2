#!/usr/bin/env python3

import isce
import datetime
import isceobj
import numpy as np
from isceobj.Attitude.Attitude import Attitude
from iscesys.Component.Component import Component
from isceobj.Image.Image import Image
from isceobj.Orbit.Orbit import Orbit
from isceobj.Util.decorators import type_check
from iscesys.Traits.Datetime import datetimeType

NUMBER_OF_SAMPLES = Component.Parameter('numberOfSamples',
        public_name='number of samples',
        default=None,
        type=int,
        mandatory=True,
        doc='Width of grd image')

NUMBER_OF_LINES = Component.Parameter('numberOfLines',
        public_name='number of lines',
        default=None,
        type=int,
        mandatory=True,
        doc='Length of grd image')

STARTING_GROUND_RANGE = Component.Parameter('startingGroundRange',
        public_name='starting ground range',
        default=None,
        type=float,
        mandatory=True,
        doc='Ground range to first pixel in m')


STARTING_SLANT_RANGE = Component.Parameter('startingSlantRange',
        public_name='starting slant range',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Slant range to first pixel in m')

ENDING_SLANT_RANGE = Component.Parameter('endingSlantRange',
        public_name='ending slant range',
        default=None,
        type=float,
        mandatory=True,
        doc = 'Slant range to last pixel in m')

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

SOFTWARE_VERSION = Component.Parameter('softwareVersion',
        public_name='software version',
        default = None,
        type=str,
        mandatory=True,
        doc = 'Software version use to generate GRD product')

TRACK_NUMBER = Component.Parameter('trackNumber',
        public_name='track number',
        default=None,
        type=int,
        mandatory=False,
        doc='Track number of the acquisition')

FRAME_NUMBER = Component.Parameter('frameNumber',
        public_name='frame number',
        default=None,
        type=int,
        mandatory=False,
        doc='Frame number of the acquisition')

ORBIT_NUMBER = Component.Parameter('orbitNumber',
        public_name='orbit number',
        default=None,
        type=int,
        mandatory=False,
        doc='orbit number of the acquisition')

PASS_DIRECTION = Component.Parameter('passDirection',
        public_name='pass direction',
        default=None,
        type=str,
        mandatory=False,
        doc='Ascending or descending pass')

LOOK_SIDE = Component.Parameter('lookSide',
        public_name='look side',
        default=None,
        type=str,
        mandatory=False,
        doc='Right or left')
        
AZIMUTH_TIME_INTERVAL = Component.Parameter('azimuthTimeInterval',
        public_name='azimuth time interval',
        default = None,
        type = float,
        mandatory = False,
        doc = 'Time interval between consecutive lines (single look)')

GROUND_RANGE_PIXEL_SIZE = Component.Parameter('groundRangePixelSize',
        public_name='ground range pixel size',
        default = None,
        type = float,
        mandatory = False,
        doc = 'Ground range spacing in m between consecutive pixels')

AZIMUTH_PIXEL_SIZE = Component.Parameter('azimuthPixelSize',
        public_name='azimuth pixel size',
        default = None,
        type = float,
        mandatory = False,
        doc = 'Azimuth spacing in m between consecutive lines')

RADAR_WAVELENGTH = Component.Parameter('radarWavelength',
        public_name='radar wavelength',
        default = None,
        type = float,
        mandatory = False,
        doc = 'Radar wavelength in m')

POLARIZATION = Component.Parameter('polarization',
        public_name='polarization',
        default = None,
        type=str,
        mandatory = False,
        doc = 'One out of HH/HV/VV/VH/RH/RV')

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
        factory='createImage',
        args = (),
        doc = 'Image on disk for beta0')

SLANT_RANGE_IMAGE = Component.Facility('slantRangeImage',
        public_name='slantRangeImage',
        module='isceobj.Image',
        factory='createImage',
        args=(),
        doc='Image on disl for slant range in m')

class GRDProduct(Component):
    """A class to represent a burst SLC along a radar track"""
    
    family = 'grdproduct'
    logging_name = 'isce.grdProduct'

    parameter_list = (NUMBER_OF_LINES,
                      NUMBER_OF_SAMPLES,
                      STARTING_GROUND_RANGE,
                      STARTING_SLANT_RANGE,
                      ENDING_SLANT_RANGE,
                      SENSING_START,
                      SENSING_STOP,
                      TRACK_NUMBER,
                      FRAME_NUMBER,
                      ORBIT_NUMBER,
                      PASS_DIRECTION,
                      LOOK_SIDE,
                      AZIMUTH_TIME_INTERVAL,
                      GROUND_RANGE_PIXEL_SIZE,
                      AZIMUTH_PIXEL_SIZE,
                      RADAR_WAVELENGTH,
                      POLARIZATION)


    facility_list = (ORBIT,
                     IMAGE,
                     SLANT_RANGE_IMAGE)



    def __init__(self,name=''):
        super(GRDProduct, self).__init__(family=self.__class__.family, name=name)

    @property
    def sensingMid(self):
        self.sensingStart + 0.5 * (self.sensingStop - self.sensingStart)

    @property
    def nearSlantRange(self):
        '''
        Return slant range to the first pixel.
        '''
        return self.startingSlantRange



    @property
    def nearGroundRange(self):
        '''
        Return ground range to the first pixel.
        '''

        return self.startingGroundRange


    @property
    def farSlantRange(self):
        '''
        Return slant range to the last pixel.

        def bisection(f, a, b, rhs, TOL=1.0e-3):
            c = (a+b)/2.0
            while ((b-a)/2.0 > TOL):
                if (f(c) == rng):
                    return c
                elif ((f(a)-rhs) * (f(c)-rhs)) < 0:
                    b = c
                else:
                    a = c

                c = (a+b)/2.0

            return c

        if self.mapDirection == "SR2GR":
            rng = self.farGroundRange()
            r0 = self.nearSlantRange()
            rmax = r0 + (self.numberOfSamples-1) * self.groundRangePixelSize

            return max( bisection(self.mapPolynomials[0].poly, r0, rmax, rng),
                        bisection(self.mapPolynomials[-1].poly, r0, rmax, rng))

        elif self.mapDirection == "GR2SR":
            rng = self.startingGroundRange + (self.numberOfSamples-1)*self.groundRangePixelSpacing
            return max( self.mapPolynomials[0].poly(rng),
                        self.mapPolynomials[-1].poly(rng))
        else:
            raise Exception('Unknown map direction: {0}'.format(self.mapDirection))
        '''
        return self.endingSlantRange

    @property
    def farGroundRange(self):
        '''
        Return ground range to the last pixel.
        '''

        return (self.startingGroundRange + (self.numberOfSamples-1) * self.groundRangePixelSize)


    @property
    def side(self):
        if self.lookSide.upper() == 'RIGHT':
            return -1
        elif self.lookSide.upper() == 'LEFT':
            return 1
        else:
            raise Exception('Look side not set')

    def getBbox(self, hgtrange=[-500., 9000.]):
        '''
        Bounding box estimate.
        '''
        
        r0 = self.nearSlantRange
        r1 = self.farSlantRange

        ts = [self.sensingStart, self.sensingStop]
        rngs = [r0, r1]

        pos = []
        for ht in hgtrange:
            for tim in ts:
                for rng in rngs:
                    llh = self.orbit.rdr2geo(tim, rng, height=ht, side = self.side)
                    pos.append(llh)
        
        pos = np.array(pos)
        bbox = [np.min(pos[:,0]), np.max(pos[:,0]), 
                np.min(pos[:,1]), np.max(pos[:,1])]
        return bbox

