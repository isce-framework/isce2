#!/usr/bin/env python3

#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
# 


import isce
import isceobj
import iscesys
from iscesys.Component.Application import Application


DATA_DIR = Application.Parameter('dataDir',
                                public_name='data directory',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc="directory of data, where data of each date are in an individual directory")

FRAMES = Application.Parameter('frames',
                                public_name = 'frames',
                                default = None,
                                type=str,
                                container=list,
                                mandatory=False,
                                doc = 'frames to process')

POLARIZATION = Application.Parameter('polarization',
                                public_name='polarization',
                                default='HH',
                                type=str,
                                mandatory=False,
                                doc="polarization to process")

STARTING_SWATH = Application.Parameter('startingSwath',
                                public_name='starting swath',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="starting swath to process")

ENDING_SWATH = Application.Parameter('endingSwath',
                                public_name='ending swath',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="ending swath to process")

DEM = Application.Parameter('dem',
                                public_name='dem for coregistration',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='dem for coregistration file')

DEM_GEO = Application.Parameter('demGeo',
                                public_name='dem for geocoding',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='dem for geocoding file')

WBD = Application.Parameter('wbd',
                                public_name='water body',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='water body file')

DATE_REFERENCE_STACK = Application.Parameter('dateReferenceStack',
                                public_name='reference date of the stack',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc="reference date of the stack")

GRID_FRAME = Application.Parameter('gridFrame',
                                public_name='grid frame',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc="resample all frames/swaths to the grid size of this frame")

GRID_SWATH = Application.Parameter('gridSwath',
                                public_name='grid swath',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="resample all frames/swaths to the grid size of this swath")

NUMBER_OF_SUBSEQUENT_DATES = Application.Parameter('numberOfSubsequentDates',
                                public_name='number of subsequent dates',
                                default=4,
                                type=int,
                                mandatory=False,
                                doc="number of subsequent dates used to form pairs")

PAIR_TIME_SPAN_MINIMUM = Application.Parameter('pairTimeSpanMinimum',
                                public_name = 'pair time span minimum in years',
                                default = None,
                                type=float,
                                mandatory=False,
                                doc = 'pair time span minimum in years')

PAIR_TIME_SPAN_MAXIMUM = Application.Parameter('pairTimeSpanMaximum',
                                public_name = 'pair time span maximum in years',
                                default = None,
                                type=float,
                                mandatory=False,
                                doc = 'pair time span maximum in years')

DATES_INCLUDED = Application.Parameter('datesIncluded',
                                public_name = 'dates to be included',
                                default = None,
                                type=str,
                                container=list,
                                mandatory=False,
                                doc = 'dates to be included')

#MUST BE FIRST DATE - SECOND DATE!!!
PAIRS_INCLUDED = Application.Parameter('pairsIncluded',
                                public_name = 'pairs to be included',
                                default = None,
                                type=str,
                                container=list,
                                mandatory=False,
                                doc = 'pairs to be included')

DATES_EXCLUDED = Application.Parameter('datesExcluded',
                                public_name = 'dates to be excluded',
                                default = None,
                                type=str,
                                container=list,
                                mandatory=False,
                                doc = 'dates to be excluded')

#MUST BE FIRST DATE - SECOND DATE!!!
PAIRS_EXCLUDED = Application.Parameter('pairsExcluded',
                                public_name = 'pairs to be excluded',
                                default = None,
                                type=str,
                                container=list,
                                mandatory=False,
                                doc = 'pairs to be excluded')

DATE_REFERENCE_STACK_ION = Application.Parameter('dateReferenceStackIon',
                                public_name='reference date of the stack for estimating ionosphere',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc="reference date of the stack in estimating ionosphere")

NUMBER_OF_SUBSEQUENT_DATES_ION = Application.Parameter('numberOfSubsequentDatesIon',
                                public_name='number of subsequent dates for estimating ionosphere',
                                default=4,
                                type=int,
                                mandatory=False,
                                doc="number of subsequent dates used to form pairs for estimating ionosphere")

PAIR_TIME_SPAN_MINIMUM_ION = Application.Parameter('pairTimeSpanMinimumIon',
                                public_name = 'pair time span minimum in years for estimating ionosphere',
                                default = None,
                                type=float,
                                mandatory=False,
                                doc = 'pair time span minimum in years for estimating ionosphere')

PAIR_TIME_SPAN_MAXIMUM_ION = Application.Parameter('pairTimeSpanMaximumIon',
                                public_name = 'pair time span maximum in years for estimating ionosphere',
                                default = None,
                                type=float,
                                mandatory=False,
                                doc = 'pair time span maximum in years for estimating ionosphere')

DATES_INCLUDED_ION = Application.Parameter('datesIncludedIon',
                                public_name = 'dates to be included for estimating ionosphere',
                                default = None,
                                type=str,
                                container=list,
                                mandatory=False,
                                doc = 'dates to be included for estimating ionosphere')

#MUST BE FIRST DATE - SECOND DATE!!!
PAIRS_INCLUDED_ION = Application.Parameter('pairsIncludedIon',
                                public_name = 'pairs to be included for estimating ionosphere',
                                default = None,
                                type=str,
                                container=list,
                                mandatory=False,
                                doc = 'pairs to be included for estimating ionosphere')

DATES_EXCLUDED_ION = Application.Parameter('datesExcludedIon',
                                public_name = 'dates to be excluded for estimating ionosphere',
                                default = None,
                                type=str,
                                container=list,
                                mandatory=False,
                                doc = 'dates to be excluded for estimating ionosphere')

#MUST BE FIRST DATE - SECOND DATE!!!
PAIRS_EXCLUDED_ION = Application.Parameter('pairsExcludedIon',
                                public_name = 'pairs to be excluded for estimating ionosphere',
                                default = None,
                                type=str,
                                container=list,
                                mandatory=False,
                                doc = 'pairs to be excluded for estimating ionosphere')

DATES_REPROCESS = Application.Parameter('datesReprocess',
                                public_name = 'reprocess already processed dates',
                                default=False,
                                type=bool,
                                mandatory=False,
                                doc = 'reprocess already processed dates')

PAIRS_REPROCESS = Application.Parameter('pairsReprocess',
                                public_name = 'reprocess already processed pairs',
                                default=False,
                                type=bool,
                                mandatory=False,
                                doc = 'reprocess already processed pairs')

PAIRS_REPROCESS_ION = Application.Parameter('pairsReprocessIon',
                                public_name = 'reprocess already processed pairs for estimating ionosphere',
                                default=False,
                                type=bool,
                                mandatory=False,
                                doc = 'reprocess already processed pairs for estimating ionosphere')

DATES_PROCESSING_DIR = Application.Parameter('datesProcessingDir',
                                public_name='dates processing directory',
                                default='dates',
                                type=str,
                                mandatory=False,
                                doc="directory for processing all dates")

DATES_RESAMPLED_DIR = Application.Parameter('datesResampledDir',
                                public_name='dates resampled directory',
                                default='dates_resampled',
                                type=str,
                                mandatory=False,
                                doc="directory for all dates resampled")

PAIRS_PROCESSING_DIR = Application.Parameter('pairsProcessingDir',
                                public_name='pairs processing directory',
                                default='pairs',
                                type=str,
                                mandatory=False,
                                doc="directory for processing all pairs")

BASELINE_DIR = Application.Parameter('baselineDir',
                                public_name='baseline directory',
                                default='baseline',
                                type=str,
                                mandatory=False,
                                doc="directory for baselines")

DATES_DIR_ION = Application.Parameter('datesDirIon',
                                public_name='dates directory for ionosphere',
                                default='dates_ion',
                                type=str,
                                mandatory=False,
                                doc="dates directory for ionosphere")

PAIRS_PROCESSING_DIR_ION = Application.Parameter('pairsProcessingDirIon',
                                public_name='pairs processing directory for estimating ionosphere',
                                default='pairs_ion',
                                type=str,
                                mandatory=False,
                                doc="directory for processing all pairs for estimating ionosphere")

#import insar processing parameters from alos2App.py
#from alos2App import REFERENCE_DIR
#from alos2App import SECONDARY_DIR
#from alos2App import REFERENCE_FRAMES
#from alos2App import SECONDARY_FRAMES
#from alos2App import REFERENCE_POLARIZATION
#from alos2App import SECONDARY_POLARIZATION
#from alos2App import STARTING_SWATH
#from alos2App import ENDING_SWATH
#from alos2App import DEM
#from alos2App import DEM_GEO
#from alos2App import WBD
from alos2App import USE_VIRTUAL_FILE
from alos2App import USE_GPU
#from alos2App import BURST_SYNCHRONIZATION_THRESHOLD
#from alos2App import CROP_SLC
from alos2App import USE_WBD_FOR_NUMBER_OFFSETS
from alos2App import NUMBER_RANGE_OFFSETS
from alos2App import NUMBER_AZIMUTH_OFFSETS
from alos2App import NUMBER_RANGE_LOOKS1
from alos2App import NUMBER_AZIMUTH_LOOKS1
from alos2App import NUMBER_RANGE_LOOKS2
from alos2App import NUMBER_AZIMUTH_LOOKS2
from alos2App import NUMBER_RANGE_LOOKS_SIM
from alos2App import NUMBER_AZIMUTH_LOOKS_SIM
from alos2App import SWATH_OFFSET_MATCHING
from alos2App import FRAME_OFFSET_MATCHING
from alos2App import FILTER_STRENGTH
from alos2App import FILTER_WINSIZE
from alos2App import FILTER_STEPSIZE 
from alos2App import REMOVE_MAGNITUDE_BEFORE_FILTERING
from alos2App import WATERBODY_MASK_STARTING_STEP
#from alos2App import GEOCODE_LIST
from alos2App import GEOCODE_BOUNDING_BOX
from alos2App import GEOCODE_INTERP_METHOD
                        #ionospheric correction parameters
from alos2App import DO_ION
from alos2App import APPLY_ION
from alos2App import NUMBER_RANGE_LOOKS_ION
from alos2App import NUMBER_AZIMUTH_LOOKS_ION
from alos2App import MASKED_AREAS_ION
from alos2App import SWATH_PHASE_DIFF_SNAP_ION
from alos2App import SWATH_PHASE_DIFF_LOWER_ION
from alos2App import SWATH_PHASE_DIFF_UPPER_ION
from alos2App import FIT_ION
from alos2App import FILT_ION
from alos2App import FIT_ADAPTIVE_ION
from alos2App import FILT_SECONDARY_ION
from alos2App import FILTERING_WINSIZE_MAX_ION
from alos2App import FILTERING_WINSIZE_MIN_ION
from alos2App import FILTERING_WINSIZE_SECONDARY_ION
from alos2App import FILTER_STD_ION
from alos2App import FILTER_SUBBAND_INT
from alos2App import FILTER_STRENGTH_SUBBAND_INT
from alos2App import FILTER_WINSIZE_SUBBAND_INT
from alos2App import FILTER_STEPSIZE_SUBBAND_INT
from alos2App import REMOVE_MAGNITUDE_BEFORE_FILTERING_SUBBAND_INT


## Common interface for all insar applications.
class Stack(Application):
    family = 'stackinsar'
    parameter_list = (DATA_DIR,
                        FRAMES,
                        POLARIZATION,
                        STARTING_SWATH,
                        ENDING_SWATH,
                        DEM,
                        DEM_GEO,
                        WBD,
                        DATE_REFERENCE_STACK,
                        GRID_FRAME,
                        GRID_SWATH,
                        NUMBER_OF_SUBSEQUENT_DATES,
                        PAIR_TIME_SPAN_MINIMUM,
                        PAIR_TIME_SPAN_MAXIMUM,
                        DATES_INCLUDED,
                        PAIRS_INCLUDED,
                        DATES_EXCLUDED,
                        PAIRS_EXCLUDED,
                        DATE_REFERENCE_STACK_ION,
                        NUMBER_OF_SUBSEQUENT_DATES_ION,
                        PAIR_TIME_SPAN_MINIMUM_ION,
                        PAIR_TIME_SPAN_MAXIMUM_ION,
                        DATES_INCLUDED_ION,
                        PAIRS_INCLUDED_ION,
                        DATES_EXCLUDED_ION,
                        PAIRS_EXCLUDED_ION,
                        DATES_REPROCESS,
                        PAIRS_REPROCESS,
                        PAIRS_REPROCESS_ION,
                        DATES_PROCESSING_DIR,
                        DATES_RESAMPLED_DIR,
                        PAIRS_PROCESSING_DIR,
                        BASELINE_DIR,
                        DATES_DIR_ION,
                        PAIRS_PROCESSING_DIR_ION,
                        #insar processing parameters, same as those in alos2App.py
                        USE_VIRTUAL_FILE,
                        USE_GPU,
                        USE_WBD_FOR_NUMBER_OFFSETS,
                        NUMBER_RANGE_OFFSETS,
                        NUMBER_AZIMUTH_OFFSETS,
                        NUMBER_RANGE_LOOKS1,
                        NUMBER_AZIMUTH_LOOKS1,
                        NUMBER_RANGE_LOOKS2,
                        NUMBER_AZIMUTH_LOOKS2,
                        NUMBER_RANGE_LOOKS_SIM,
                        NUMBER_AZIMUTH_LOOKS_SIM,
                        SWATH_OFFSET_MATCHING,
                        FRAME_OFFSET_MATCHING,
                        FILTER_STRENGTH,
                        FILTER_WINSIZE,
                        FILTER_STEPSIZE, 
                        REMOVE_MAGNITUDE_BEFORE_FILTERING,
                        WATERBODY_MASK_STARTING_STEP,
                        GEOCODE_BOUNDING_BOX,
                        GEOCODE_INTERP_METHOD,
                        #ionospheric correction parameters
                        DO_ION,
                        APPLY_ION,
                        NUMBER_RANGE_LOOKS_ION,
                        NUMBER_AZIMUTH_LOOKS_ION,
                        MASKED_AREAS_ION,
                        SWATH_PHASE_DIFF_SNAP_ION,
                        SWATH_PHASE_DIFF_LOWER_ION,
                        SWATH_PHASE_DIFF_UPPER_ION,
                        FIT_ION,
                        FILT_ION,
                        FIT_ADAPTIVE_ION,
                        FILT_SECONDARY_ION,
                        FILTERING_WINSIZE_MAX_ION,
                        FILTERING_WINSIZE_MIN_ION,
                        FILTERING_WINSIZE_SECONDARY_ION,
                        FILTER_STD_ION,
                        FILTER_SUBBAND_INT,
                        FILTER_STRENGTH_SUBBAND_INT,
                        FILTER_WINSIZE_SUBBAND_INT,
                        FILTER_STEPSIZE_SUBBAND_INT,
                        REMOVE_MAGNITUDE_BEFORE_FILTERING_SUBBAND_INT)

    facility_list = ()

    def __init__(self, family='', name='',cmdline=None):
        import isceobj

        super().__init__(
            family=family if family else  self.__class__.family, name=name,
            cmdline=cmdline)


        return None


