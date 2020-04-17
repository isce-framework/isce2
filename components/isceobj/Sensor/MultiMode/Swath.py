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
SWATH_NUMBER = Component.Parameter('swathNumber',
        public_name = 'swath number',
        default = None,
        type = int,
        mandatory = True,
        doc = 'swath number for bookkeeping')

POLARIZATION = Component.Parameter('polarization',
        public_name = 'polarization',
        default = None,
        type = str,
        mandatory = True,
        doc = 'polarization')

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

RANGE_BANDWIDTH = Component.Parameter('rangeBandwidth',
        public_name = 'range bandwidth',
        default = None,
        type=float,
        mandatory = True,
        doc = 'range bandwidth in Hz')

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

DOPPLER_VS_PIXEL = Component.Parameter('dopplerVsPixel',
        public_name = 'doppler vs pixel',
        default = None,
        type = float,
        mandatory = True,
        container = list,
        doc = 'Doppler (Hz) polynomial coefficients vs range pixel number')

AZIMUTH_FMRATE_VS_PIXEL = Component.Parameter('azimuthFmrateVsPixel',
        public_name = 'azimuth fm rate vs pixel',
        default = [],
        type = float,
        mandatory = True,
        container = list,
        doc = 'azimuth FM rate (Hz/s) polynomial coefficients vs range pixel number')

#for ScanSAR full-aperture product
BURST_LENGTH = Component.Parameter('burstLength',
        public_name = 'Burst Length',
        default = None,
        type = float,
#        type = int,
        mandatory = False,
        doc = 'number of pulses in a raw burst')

BURST_CYCLE_LENGTH = Component.Parameter('burstCycleLength',
        public_name = 'Burst cycle length',
        default = None,
        type = float,
        mandatory = False,
        doc = 'number of pulses in a raw burst cycle')

BURST_START_TIME = Component.Parameter('burstStartTime',
        public_name='Burst start time',
        default=None,
        type=datetimeType,
        mandatory=False,
        doc='start time of a raw burst')

#for ScanSAR burst-by-burst processing
PRF_FRACTION = Component.Parameter('prfFraction',
        public_name = 'prf fraction',
        default = None,
        type = float,
        mandatory = False,
        doc = 'fraction of PRF for extracting bursts for bookkeeping')

NUMBER_OF_BURSTS = Component.Parameter('numberOfBursts',
        public_name='number of bursts',
        default=None,
        type=int,
        mandatory=False,
        doc='number of bursts in a swath')

FIRST_BURST_RAW_START_TIME = Component.Parameter('firstBurstRawStartTime',
        public_name='start time of first raw burst',
        default=None,
        type=datetimeType,
        mandatory=False,
        doc='start time of first raw burst')

FIRST_BURST_SLC_START_TIME = Component.Parameter('firstBurstSlcStartTime',
        public_name='start time of first burst slc',
        default=None,
        type=datetimeType,
        mandatory=False,
        doc='start time of first burst slc')

BURST_SLC_FIRST_LINE_OFFSETS = Component.Parameter('burstSlcFirstLineOffsets',
                                public_name = 'burst SLC first line offsets',
                                default = None,
                                type = int,
                                mandatory = False,
                                container = list,
                                doc = 'burst SLC first line offsets')

BURST_SLC_NUMBER_OF_SAMPLES = Component.Parameter('burstSlcNumberOfSamples',
        public_name='burst slc number of samples',
        default=None,
        type=int,
        mandatory=False,
        doc='burst slc width of the burst slc')

BURST_SLC_NUMBER_OF_LINES = Component.Parameter('burstSlcNumberOfLines',
        public_name='burst slc number of lines',
        default=None,
        type=int,
        mandatory=False,
        doc='burst slc length of the burst slc')


class Swath(Component):
    """A class to represent a swath SLC"""
    
    family = 'swath'
    logging_name = 'isce.swath'

    parameter_list = (SWATH_NUMBER,
                      POLARIZATION,
                      NUMBER_OF_SAMPLES,
                      NUMBER_OF_LINES,
                      STARTING_RANGE,
                      RANGE_SAMPLING_RATE,
                      RANGE_PIXEL_SIZE,
                      RANGE_BANDWIDTH,
                      SENSING_START,
                      PRF,
                      AZIMUTH_PIXEL_SIZE,
                      AZIMUTH_LINE_INTERVAL,
                      DOPPLER_VS_PIXEL,
                      AZIMUTH_FMRATE_VS_PIXEL,
                      BURST_LENGTH,
                      BURST_CYCLE_LENGTH,
                      BURST_START_TIME,
                      PRF_FRACTION,
                      NUMBER_OF_BURSTS,
                      FIRST_BURST_RAW_START_TIME,
                      FIRST_BURST_SLC_START_TIME,
                      BURST_SLC_FIRST_LINE_OFFSETS,
                      BURST_SLC_NUMBER_OF_SAMPLES,
                      BURST_SLC_NUMBER_OF_LINES
                    )


    def __init__(self,name=''):
        super(Swath, self).__init__(family=self.__class__.family, name=name)
        return None


    def clone(self):
        import copy
        res = copy.deepcopy(self)
        res.image._accessor = None
        res.image._factory = None

        return res
