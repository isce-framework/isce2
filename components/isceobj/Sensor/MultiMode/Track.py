#!/usr/bin/env python3

#Author: Cunren Liang, 2015-

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
PASS_DIRECTION = Component.Parameter('passDirection',
        public_name='pass direction',
        default = None,
        type=str,
        mandatory=True,
        doc = 'satellite flying direction, ascending/descending')

POINTING_DIRECTION = Component.Parameter('pointingDirection',
        public_name='pointing direction',
        default=None,
        type = str,
        mandatory = True,
        doc = 'antenna point direction: right/left')

OPERATION_MODE = Component.Parameter('operationMode',
        public_name='operation mode',
        default=None,
        type = str,
        mandatory = True,
        doc = 'JAXA ALOS-2 operation mode code')

RADAR_WAVELENGTH = Component.Parameter('radarWavelength',
        public_name = 'radarWavelength',
        default = None,
        type = float,
        mandatory = True,
        doc = 'radar wavelength in m')

NUMBER_OF_SAMPLES = Component.Parameter('numberOfSamples',
        public_name='number of samples',
        default=None,
        type=int,
        mandatory=True,
        doc='width of the burst slc')

NUMBER_OF_LINES = Component.Parameter('numberOfLines',
        public_name='number of lines',
        default=None,
        type=int,
        mandatory=True,
        doc='length of the burst slc')

STARTING_RANGE = Component.Parameter('startingRange',
        public_name='starting range',
        default=None,
        type=float,
        mandatory=True,
        doc='slant range to first pixel in m')

RANGE_SAMPLING_RATE = Component.Parameter('rangeSamplingRate',
        public_name = 'range sampling rate',
        default = None,
        type = float,
         mandatory = True,
         doc = 'range sampling rate in Hz')

RANGE_PIXEL_SIZE = Component.Parameter('rangePixelSize',
        public_name = 'range pixel size',
        default = None,
        type=float,
        mandatory = True,
        doc = 'slant range pixel size in m')

SENSING_START = Component.Parameter('sensingStart',
        public_name='sensing start',
        default=None,
        type=datetimeType,
        mandatory=True,
        doc='UTC time corresponding to first line of swath SLC')

PRF = Component.Parameter('prf',
        public_name = 'pulse repetition frequency',
        default = None,
        type = float,
        mandatory = True,
        doc = 'pulse repetition frequency in Hz')

AZIMUTH_PIXEL_SIZE = Component.Parameter('azimuthPixelSize',
        public_name = 'azimuth pixel size',
        default = None,
        type=float,
        mandatory = True,
        doc = 'azimuth pixel size on ground in m')

AZIMUTH_LINE_INTERVAL = Component.Parameter('azimuthLineInterval',
        public_name = 'azimuth line interval',
        default = None,
        type=float,
        mandatory = True,
        doc = 'azimuth line interval in s')

#########################################################################################################
#for dense offset
DOPPLER_VS_PIXEL = Component.Parameter('dopplerVsPixel',
        public_name = 'doppler vs pixel',
        default = None,
        type = float,
        mandatory = True,
        container = list,
        doc = 'Doppler (Hz) polynomial coefficients vs range pixel number')
#########################################################################################################


#ProductManager cannot handle two or more layers of createTraitSeq
#use list instead for bookkeeping
#can creat a class Frames(Component) and wrap trait sequence, but it
#leads to more complicated output xml file, which is not good for viewing
FRAMES = Component.Parameter('frames',
        public_name = 'frames',
        default = [],
        #type = float,
        mandatory = True,
        container = list,
        doc = 'sequence of frames')


####List of facilities
ORBIT = Component.Facility('orbit',
    public_name='orbit',
    module='isceobj.Orbit.Orbit',
    factory='createOrbit',
    args=(),
    doc = 'orbit state vectors')

# FRAMES = Component.Facility('frames',
#         public_name='frames',
#         module = 'iscesys.Component',
#         factory = 'createTraitSeq',
#         args=('frame',),
#         mandatory = False,
#         doc = 'trait sequence of frames')

class Track(Component):
    """A class to represent a track"""
    
    family = 'track'
    logging_name = 'isce.track'
##############################################################################
    parameter_list = (PASS_DIRECTION,
                      POINTING_DIRECTION,
                      OPERATION_MODE,
                      RADAR_WAVELENGTH,
                      NUMBER_OF_SAMPLES,
                      NUMBER_OF_LINES,
                      STARTING_RANGE,
                      RANGE_SAMPLING_RATE,
                      RANGE_PIXEL_SIZE,
                      SENSING_START,
                      PRF,
                      AZIMUTH_PIXEL_SIZE,
                      AZIMUTH_LINE_INTERVAL,
                      DOPPLER_VS_PIXEL,
                      FRAMES
                    )

    facility_list = (ORBIT,
                    )

    def __init__(self,name=''):
        super(Track, self).__init__(family=self.__class__.family, name=name)
        return None


    def clone(self):
        import copy
        res = copy.deepcopy(self)
        res.image._accessor = None
        res.image._factory = None

        return res
