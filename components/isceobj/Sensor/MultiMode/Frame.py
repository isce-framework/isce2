#!/usr/bin/env python3

#Author: Cunren Liang, 2015-

import isce
import datetime
import isceobj
import numpy as np
from iscesys.Component.Component import Component
from iscesys.Traits import datetimeType


####List of parameters
FRAME_NUMBER = Component.Parameter('frameNumber',
        public_name = 'frame number',
        default = None,
        type = str,
        mandatory = True,
        doc = 'frame number in unpacked file names (not in zip file name!)')

PROCESSING_FACILITY = Component.Parameter('processingFacility',
    public_name='processing facility',
    default=None,
    type = str,
    mandatory = False,
    doc = 'processing facility information')

PROCESSING_SYSTEM = Component.Parameter('processingSystem',
    public_name='processing system',
    default=None,
    type = str,
    mandatory = False,
    doc = 'processing system information')

PROCESSING_SYSTEM_VERSION = Component.Parameter('processingSoftwareVersion',
    public_name='processing software version',
    default=None,
    type = str,
    mandatory = False,
    doc = 'processing system software version')

ORBIT_QUALITY = Component.Parameter('orbitQuality',
    public_name='orbit quality',
    default=None,
    type = str,
    mandatory = False,
    doc = 'orbit quality. 0: preliminary, 1: decision, 2: high precision')

#note that following parameters consider range/azimuth number of looks in interferogram formation
#except: rangeSamplingRate, prf

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

####List of facilities
SWATHS = Component.Facility('swaths',
        public_name='swaths',
        module = 'iscesys.Component',
        factory = 'createTraitSeq',
        args=('swath',),
        mandatory = False,
        doc = 'trait sequence of swath SLCs')

class Frame(Component):
    """A class to represent a frame"""
    
    family = 'frame'
    logging_name = 'isce.frame'




    parameter_list = (FRAME_NUMBER,
                      PROCESSING_FACILITY,
                      PROCESSING_SYSTEM,
                      PROCESSING_SYSTEM_VERSION,
                      ORBIT_QUALITY,
                      NUMBER_OF_SAMPLES,
                      NUMBER_OF_LINES,
                      STARTING_RANGE,
                      RANGE_SAMPLING_RATE,
                      RANGE_PIXEL_SIZE,
                      SENSING_START,
                      PRF,
                      AZIMUTH_PIXEL_SIZE,
                      AZIMUTH_LINE_INTERVAL
                      )


    facility_list = (SWATHS,)


    def __init__(self,name=''):
        super(Frame, self).__init__(family=self.__class__.family, name=name)
        return None


    def clone(self):
        import copy
        res = copy.deepcopy(self)
        res.image._accessor = None
        res.image._factory = None

        return res
