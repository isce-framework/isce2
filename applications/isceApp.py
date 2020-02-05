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
# Authors: Kosal Khun, Marco Lavalle
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




# adapted from applications/insarApp.py
# Description: This module generates an application running different steps
# of SAR, InSAR, PolInSAR and TomoSAR processing.


import time
import datetime
import os
import sys
import math
from isce import logging

import isce
import isceobj
import iscesys
from iscesys.Component.Application import Application
from iscesys.Component.Configurable import SELF
from iscesys.Compatibility import Compatibility
from iscesys.StdOEL.StdOELPy import create_writer
#from isceobj import IsceProc

from isceobj.Scene.Frame import FrameMixin
import isceobj.IsceProc as IsceProc
from isceobj.Location.Peg import Peg
from isceobj import Unwrap
from isceobj.Sensor import SENSORS
from contrib.demUtils.Correct_geoid_i2_srtm import Correct_geoid_i2_srtm
from pprint import pprint

POLS = ['hh', 'hv', 'vh', 'vv'] ##accepted polarizations

SENSOR_NAME = Application.Parameter(
    'sensorName',
    public_name='sensor name',
    default=None,
    type=str,
    mandatory=True,
    doc="Sensor name"
)

PEG_LAT = Application.Parameter(
    'pegLat',
    public_name='peg latitude (deg)',
    default=None,
    type=float,
    mandatory=False,
    doc='Peg Latitude in degrees'
)

PEG_LON = Application.Parameter(
    'pegLon',
    public_name='peg longitude (deg)',
    default=None,
    type=float,
    mandatory=False,
    doc='Peg Longitude in degrees'
)

PEG_HDG = Application.Parameter(
    'pegHdg',
    public_name='peg heading (deg)',
    default=None,
    type=float,
    mandatory=False,
    doc='Peg Heading in degrees'
)

PEG_RAD = Application.Parameter(
    'pegRad',
    public_name='peg radius (m)',
    default=None,
    type=float,
    mandatory=False,
    doc='Peg Radius of Curvature in meters'
)

DOPPLER_METHOD = Application.Parameter(
    'dopplerMethod',
    public_name='doppler method',
    default='useDOPIQ',
    type=str,
    mandatory=False,
    doc= (
          "Doppler calculation method.Choices: 'useDOPIQ', 'useCalcDop', \n" +
          "'useDoppler'.")
)

USE_DOP = Application.Parameter(
    'use_dop',
    public_name='use_dop',
    default="average",
    type=float,
    mandatory=False,
    doc=(
         "Choose whether to use scene_sid or average Doppler for\n"+
         "processing, where sid is the scene id to use."
    )
)

USE_HIGH_RESOLUTION_DEM_ONLY = Application.Parameter(
    'useHighResolutionDemOnly',
    public_name='useHighResolutionDemOnly',
    default=False,
    type=bool,
    mandatory=False,
    doc=(
         """If True and a dem is not specified in input, it will only
           download the SRTM highest resolution dem if it is available
           and fill the missing portion with null values (typically -32767)."""
        )
)

DEM_FILENAME = Application.Parameter(
    'demFilename',
    public_name='demFilename',
    default='',
    type=str,
    mandatory=False,
    doc="Filename of the DEM init file"
)

GEO_POSTING = Application.Parameter(
    'geoPosting',
    public_name='geoPosting',
    default=None,
    type=float,
    mandatory=False,
    doc=(
        "Output posting for geocoded images in degrees (latitude = longitude)"
    )
)

POSTING = Application.Parameter(
    'posting',
    public_name='posting',
    default=15,
    type=int,
    mandatory=False,
    doc="posting for interferogram"
)

PATCH_SIZE = Application.Parameter(
    'patchSize',
    public_name='azimuth patch size',
    default=None,
    type=int,
    mandatory=False,
    doc= "Size of overlap/save patch size for formslc"
)

GOOD_LINES = Application.Parameter(
    'goodLines',
    public_name='patch valid pulses',
    default=None,
    type=int,
    mandatory=False,
    doc= "Size of overlap/save save region for formslc"
)

NUM_PATCHES = Application.Parameter(
    'numPatches',
    public_name='number of patches',
    default=None,
    type=int,
    mandatory=False,
    doc="How many patches to process of all available patches"
)

AZ_SHIFT = Application.Parameter(
    'azShiftPixels',
    public_name='azimuth shift',
    default=None,
    type=int,
    mandatory=False,
    doc='Number of pixels to shift in azimuth'
)

SLC_RGLOOKS = Application.Parameter(
    'slcRgLooks',
    public_name='slc rangelooks',
    default=1,
    type=int,
    mandatory=False,
    doc="Multilooking factor in range direction for SLCs"
)

SLC_AZLOOKS = Application.Parameter(
    'slcAzLooks',
    public_name='slc azimuthlooks',
    default=1,
    type=int,
    mandatory=False,
    doc="Multilooking factor in azimuth direction for SLCs"
)

SLC_FILTERMETHOD = Application.Parameter(
    'slcFilterMethod',
    public_name='slc filtermethod',
    default='Gaussian',
    type=str,
    mandatory=False,
    doc="Filter method for SLCs: Gaussian, Goldstein, adaptative"
)

SLC_FILTERHEIGHT = Application.Parameter(
    'slcFilterHeight',
    public_name='slc filterheight',
    default=1,
    type=int,
    mandatory=False,
    doc="Window height for SLC filtering"
)

SLC_FILTERWIDTH = Application.Parameter(
    'slcFilterWidth',
    public_name='slc filterwidth',
    default=1,
    type=int,
    mandatory=False,
    doc="Window width for SLC filtering"
)

OFFSET_METHOD = Application.Parameter(
    'offsetMethod',
    public_name='slc offset method',
    default='offsetprf',
    type=str,
    mandatory=False,
    doc=("SLC offset estimation method name. "+
         "Use value=ampcor to run ampcor")
)

COREG_STRATEGY = Application.Parameter(
    'coregStrategy',
    public_name='coregistration strategy',
    default='single reference',
    type=str,
    mandatory=False,
    doc="How to coregister the stack: single reference or cascade"
)

REF_SCENE = Application.Parameter(
    'refScene',
    public_name='reference scene',
    default=None,
    type=str,
    mandatory=False,
    doc="Scene used as reference if coregistration strategy = single reference"
)

REF_POL = Application.Parameter(
    'refPol',
    public_name='reference polarization',
    default='hh',
    type=str,
    mandatory=False,
    doc=("Polarization used as reference if coregistration strategy = "+
         "single reference. Default: HH"
        )
)

OFFSET_SEARCH_WINDOW_SIZE = Application.Parameter(
    'offsetSearchWindowSize',
    public_name='offset search window size',
    default=None,
    type=int,
    mandatory=False,
    doc=("Search window size used in offsetprf "+
         "and rgoffset.")
)

GROSS_AZ = Application.Parameter(
    'grossAz',
    public_name='gross azimuth offset',
    default=None,
    type=int,
    mandatory=False,
    doc=("Override the value of the gross azimuth offset for offset " +
         "estimation prior to interferogram formation"
        )
)

GROSS_RG = Application.Parameter(
    'grossRg',
    public_name='gross range offset',
    default=None,
    type=int,
    mandatory=False,
    doc=(
         "Override the value of the gross range offset for offset" +
         "estimation prior to interferogram formation"
        )
)

CULLING_SEQUENCE = Application.Parameter(
    'culling_sequence',
    public_name='Culling Sequence',
    default= (10,5,3),
    container=tuple,
    type=int,
    doc="TBD"
)

NUM_FIT_COEFF = Application.Parameter(
    'numFitCoeff',
    public_name='Number of fit coefficients',
    default=6,
    type=int,
    doc="Number of fit coefficients for offoutliers."
)

RESAMP_RGLOOKS = Application.Parameter(
    'resampRgLooks',
    public_name='resamp range looks',
    default=None,
    type=int,
    mandatory=False,
    doc='Number of range looks to use in resamp'
)

RESAMP_AZLOOKS = Application.Parameter(
    'resampAzLooks',
    public_name='resamp azimuth looks',
    default=None,
    type=int,
    mandatory=False,
    doc='Number of azimuth looks to use in resamp'
)

FR_FILTER = Application.Parameter(
    'FR_filter',
    public_name='FR filter',
    default=None,
    type=str,
    mandatory=False,
    doc='Filter method for FR, if spatial filtering is desired'
)

FR_FILTERSIZE_X = Application.Parameter(
    'FR_filtersize_x',
    public_name='FR filtersize X',
    default=None,
    type=int,
    mandatory=False,
    doc='Filter width for FR'
)

FR_FILTERSIZE_Y = Application.Parameter(
    'FR_filtersize_y',
    public_name='FR filtersize Y',
    default=None,
    type=int,
    mandatory=False,
    doc='Filter height for FR'
)

FILTER_STRENGTH = Application.Parameter(
    'filterStrength',
    public_name='filter strength',
    default = None,
    type=float,
    mandatory=False,
    doc='Goldstein Werner Filter strength'
)

CORRELATION_METHOD = Application.Parameter(
    'correlation_method',
    public_name='correlation_method',
    default='cchz_wave',
    type=str,
    mandatory=False,
    doc=(
         """Select coherence estimation method:
                  cchz=cchz_wave
                  phase_gradient=phase gradient"""
        )
)

UNWRAPPER_NAME = Application.Parameter(
    'unwrapper_name',
    public_name='unwrapper name',
    default='',
    type=str,
    mandatory=False,
    doc="Unwrapping method to use. To be used in combination with UNWRAP."
)

GEOCODE_LIST = Application.Parameter(
    'geocode_list',
    public_name='geocode list',
    default = None,
    container=list,
    type=str,
    doc = "List of products to geocode."
)

GEOCODE_BOX = Application.Parameter(
    'geocode_bbox',
    public_name='geocode bounding box',
    default = None,
    container = list,
    type=float,
    doc='Bounding box for geocoding - South, North, West, East in degrees'
)

PICKLE_DUMPER_DIR = Application.Parameter(
    'pickleDumpDir',
    public_name='pickle dump directory',
    default='PICKLE',
    type=str,
    mandatory=False,
    doc= "If steps is used, the directory in which to store pickle objects."
)

PICKLE_LOAD_DIR = Application.Parameter(
    'pickleLoadDir',
    public_name='pickle load directory',
    default='PICKLE',
    type=str,
    mandatory=False,
    doc="If steps is used, the directory from which to retrieve pickle objects"
)

OUTPUT_DIR = Application.Parameter(
    'outputDir',
    public_name='output directory',
    default='.',
    type=str,
    mandatory=False,
    doc="Output directory, where log files and output files will be dumped."
)

SELECTED_SCENES = Application.Parameter(
    'selectedScenes',
    public_name='selectScenes',
    default=[],
    mandatory=False,
    container=list,
    type=str,
    doc="Comma-separated list of scene ids to process. If not given, process all scenes."
)

SELECTED_PAIRS = Application.Parameter(
    'selectedPairs',
    public_name='selectPairs',
    default=[],
    mandatory=False,
    container=list,
    type=str,
    doc=("Comma-separated list of pairs to process. Pairs are in the form sid1-sid2. "+
         "If not given, process all possible pairs."
    )
)

SELECTED_POLS = Application.Parameter(
    'selectedPols',
    public_name='selectPols',
    default=[],
    mandatory=False,
    container=list,
    type=str,
    doc=("Comma-separated list of polarizations to process. "+
         "If not given, process all polarizations."
    )
)

DO_PREPROCESS = Application.Parameter(
    'do_preprocess',
    public_name='do preprocess',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if preprocessor is desired."
)

DO_VERIFY_DEM = Application.Parameter(
    'do_verifyDEM',
    public_name='do verifyDEM',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if verify DEM is desired. If DEM not given, download DEM."
)

DO_PULSETIMING = Application.Parameter(
    'do_pulsetiming',
    public_name='do pulsetiming',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if running pulsetiming is desired."
)

DO_ESTIMATE_HEIGHTS = Application.Parameter(
    'do_estimateheights',
    public_name='do estimateheights',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if estimating heights is desired."
)

DO_SET_MOCOMPPATH = Application.Parameter(
    'do_mocomppath',
    public_name='do mocomppath',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if setting mocomppath is desired."
)

DO_ORBIT2SCH = Application.Parameter(
    'do_orbit2sch',
    public_name='do orbit2sch',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if converting orbit to SCH is desired."
)

DO_UPDATE_PREPROCINFO = Application.Parameter(
    'do_updatepreprocinfo',
    public_name='do updatepreprocinfo',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if updating info is desired."
    )

DO_FORM_SLC = Application.Parameter(
    'do_formslc',
    public_name='do formslc',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if form_slc is desired."
)

DO_MULTILOOK_SLC = Application.Parameter(
    'do_multilookslc',
    public_name='do multilookslc',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if slc multilooking is desired."
)

DO_FILTER_SLC = Application.Parameter(
    'do_filterslc',
    public_name='do filterslc',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if slc filtering is desired."
)

DO_GEOCODE_SLC = Application.Parameter(
    'do_geocodeslc',
    public_name='do geocodeslc',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if slc geocoding is desired."
)

DO_OFFSETPRF = Application.Parameter(
    'do_offsetprf',
    public_name='do offsetprf',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if running offsetprf is desired."
)

DO_OUTLIERS1 = Application.Parameter(
    'do_outliers1',
    public_name='do outliers1',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if running outliers is desired."
)

DO_PREPARE_RESAMPS = Application.Parameter(
    'do_prepareresamps',
    public_name='do prepareresamps',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if preparing resamps is desired."
)

DO_RESAMP = Application.Parameter(
    'do_resamp',
    public_name='do resamp',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if outputting of resampled slc is desired."
)

DO_RESAMP_IMAGE = Application.Parameter(
    'do_resamp_image',
    public_name='do resamp image',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if outputting of offset images is desired."
)

DO_POL_CORRECTION = Application.Parameter(
    'do_pol_correction',
    public_name='do polarimetric correction',
    default=False,
    type=bool,
    mandatory=False,
    doc='True if polarimetric correction is desired.'
)

DO_POL_PREPROCESS = Application.Parameter(
    'do_preprocess',
    public_name='do preprocess',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if preprocessor is desired."
)

DO_POL_FR = Application.Parameter(
    'do_pol_fr',
    public_name='do calculate FR',
    default=False,
    type=bool,
    mandatory=False,
    doc='True if calculating Faraday Rotation is desired.'
)

DO_POL_TEC = Application.Parameter(
    'do_pol_tec',
    public_name='do FR to TEC',
    default=False,
    type=bool,
    mandatory=False,
    doc='True if converting FR to TEC is desired.'
)

DO_POL_PHASE = Application.Parameter(
    'do_pol_phase',
    public_name='do TEC to phase',
    default=False,
    type=bool,
    mandatory=False,
    doc='True if converting TEC to phase is desired.'
)

DO_CROSSMUL = Application.Parameter(
    'do_crossmul',
    public_name='do crossmul',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if crossmultiplication is desired."
)

DO_MOCOMP_BASELINE = Application.Parameter(
    'do_mocompbaseline',
    public_name='do mocomp baseline',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if estimating mocomp baseline is desired."
)

DO_SET_TOPOINT1 = Application.Parameter(
    'do_settopoint1',
    public_name='do set topoint1',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if setting toppoint1 is desired."
)

DO_TOPO = Application.Parameter(
    'do_topo',
    public_name='do topo',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if estimating topography is desired."
)

DO_SHADE_CPX2RG = Application.Parameter(
    'do_shadecpx2rg',
    public_name='do shadecpx2rg',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if shadecpx2rg is desired."
)

DO_RG_OFFSET = Application.Parameter(
    'do_rgoffset',
    public_name='do rgoffset',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if rgoffset is desired."
)

DO_RG_OUTLIERS2 = Application.Parameter(
    'do_rg_outliers2',
    public_name='do rg outliers2',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if rg outliers2 is desired."
)

DO_RESAMP_ONLY = Application.Parameter(
    'do_resamp_only',
    public_name='do resamp only',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if resample only is desired."
)

DO_SET_TOPOINT2 = Application.Parameter(
    'do_settopoint2',
    public_name='do set topoint2',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if setting topoint2 is desired."
)

DO_CORRECT = Application.Parameter(
    'do_correct',
    public_name='do correct',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if correcting image is desired."
)

DO_COHERENCE = Application.Parameter(
    'do_coherence',
    public_name='do coherence',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if coherence estimation is desired."
)

DO_FILTER_INF = Application.Parameter(
    'do_filterinf',
    public_name='do filter interferogram',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if interferogram filtering is desired."
)

DO_UNWRAP = Application.Parameter(
    'do_unwrap',
    public_name='do unwrap',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if unwrapping is desired. To be used in combination with UNWRAPPER_NAME."
)

DO_GEOCODE_INF = Application.Parameter(
    'do_geocodeinf',
    public_name='do geocode interferogram',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if interferogram filtering is desired."
)

DO_GEOCODE = Application.Parameter(
    'do_geocode',
    public_name='do geocode',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if interferogram filtering is desired."
)

RENDERER = Application.Parameter(
    'renderer',
    public_name='renderer',
    default='pickle',
    type=str,
    mandatory=True,
    doc=(
    "Format in which the data is serialized when using steps. Options are xml (default) or pickle."
    )
)

_ISCE = Application.Facility(
    '_isce',
    public_name='isceproc',
    module='isceobj.IsceProc',
    factory='createIsceProc',
    args = ('isceAppContext', isceobj.createCatalog('isceProc')),
    mandatory=False,
    doc="IsceProc object"
)

STACK = Application.Facility(
    'stack',
    public_name='Stack',
    module='isceobj.Stack',
    factory='createStack',
    mandatory=True,
    doc="Stack component with a list of scenes."
)

DEM = Application.Facility(
    'dem',
    public_name='Dem',
    module='isceobj.Image',
    factory='createDemImage',
    mandatory=False,
    doc=(
         "Dem Image configurable component.  Do not include this in the "+
         "input file and an SRTM Dem will be downloaded for you."
    )
)

DEM_STITCHER = Application.Facility(
    'demStitcher',
    public_name='demStitcher',
    module='iscesys.DataManager',
    factory='createManager',
    args=('dem1', 'iscestitcher'),
    mandatory=False,
    doc="Object that based on the frame bounding boxes creates a DEM"
)

RUN_FORM_SLC = Application.Facility(
    'runFormSLC',
    public_name='Form SLC',
    module='isceobj.IsceProc',
    factory='createFormSLC',
    args=(SELF(), DO_FORM_SLC, SENSOR_NAME),
    mandatory=False,
    doc="SLC formation module"
)

RUN_UPDATE_PREPROC_INFO = Application.Facility(
    'runUpdatePreprocInfo',
    public_name='preproc info updater',
    module='isceobj.IsceProc',
    factory='createUpdatePreprocInfo',
    args=(SELF(), DO_UPDATE_PREPROCINFO, SENSOR_NAME),
    mandatory=False,
    doc="update preproc info module"
)

RUN_OFFSETPRF = Application.Facility(
    'runOffsetprf',
    public_name='slc offsetter',
    module='isceobj.IsceProc',
    factory='createOffsetprf',
    args=(SELF(), DO_OFFSETPRF, OFFSET_METHOD),
    mandatory=False,
    doc="Offset a pair of SLC images."
)

RUN_ESTIMATE_HEIGHTS = Application.Facility(
    'runEstimateHeights',
    public_name='Estimate Heights',
    module='isceobj.IsceProc',
    factory='createEstimateHeights',
    args=(SELF(), DO_ESTIMATE_HEIGHTS, SENSOR_NAME),
    mandatory=False,
    doc="mocomp height estimation module"
)

RUN_SET_MOCOMP_PATH = Application.Facility(
    'runSetmocomppath',
    public_name='set mocomp path',
    module='isceobj.IsceProc',
    factory='createSetmocomppath',
    args=(SELF(), DO_SET_MOCOMPPATH, SENSOR_NAME),
    mandatory=False,
    doc="mocomp set mocomp path module"
)

RUN_RG_OFFSET = Application.Facility(
    'runRgoffset',
    public_name='rg offsetter',
    module='isceobj.IsceProc',
    factory='createRgoffset',
    args=(SELF(), DO_RG_OFFSET, OFFSET_METHOD),
    mandatory=False,
    doc="mocomp dem offsetter module"
)

RUN_UNWRAPPER = Application.Facility(
    'runUnwrapper',
    public_name='Run unwrapper',
    module='isceobj.IsceProc',
    factory='createUnwrapper',
    args=(SELF(), DO_UNWRAP, UNWRAPPER_NAME,),
    mandatory=False,
    doc="Unwrapping module"
)

class IsceApp(Application, FrameMixin):
    """
    This class represents the application that reads the input xml file and runs the various processing steps accordingly.
    """

    family = "isce" #ML 2014-03-25

    ## Define Class parameters in this list
    parameter_list = (SENSOR_NAME,
                      PEG_LAT,
                      PEG_LON,
                      PEG_HDG,
                      PEG_RAD,
                      DOPPLER_METHOD,
                      USE_DOP,
                      USE_HIGH_RESOLUTION_DEM_ONLY,
                      DEM_FILENAME,
                      GEO_POSTING,
                      POSTING,
                      PATCH_SIZE,
                      GOOD_LINES,
                      NUM_PATCHES,
                      AZ_SHIFT,
                      SLC_RGLOOKS,
                      SLC_AZLOOKS,
                      SLC_FILTERMETHOD,
                      SLC_FILTERHEIGHT,
                      SLC_FILTERWIDTH,
                      OFFSET_METHOD,
                      COREG_STRATEGY,
                      REF_SCENE,
                      REF_POL,
                      OFFSET_SEARCH_WINDOW_SIZE,
                      GROSS_AZ,
                      GROSS_RG,
                      CULLING_SEQUENCE,
                      NUM_FIT_COEFF,
                      RESAMP_RGLOOKS,
                      RESAMP_AZLOOKS,
                      FR_FILTER,
                      FR_FILTERSIZE_X,
                      FR_FILTERSIZE_Y,
                      CORRELATION_METHOD,
                      FILTER_STRENGTH,
                      UNWRAPPER_NAME,
                      GEOCODE_LIST,
                      GEOCODE_BOX,
                      PICKLE_DUMPER_DIR,
                      PICKLE_LOAD_DIR,
                      OUTPUT_DIR,
                      SELECTED_SCENES,
                      SELECTED_PAIRS,
                      SELECTED_POLS,
                      DO_PREPROCESS,
                      DO_VERIFY_DEM,
                      DO_PULSETIMING,
                      DO_ESTIMATE_HEIGHTS,
                      DO_SET_MOCOMPPATH,
                      DO_ORBIT2SCH,
                      DO_UPDATE_PREPROCINFO,
                      DO_FORM_SLC,
                      DO_MULTILOOK_SLC,
                      DO_FILTER_SLC,
                      DO_GEOCODE_SLC,
                      DO_OFFSETPRF,
                      DO_OUTLIERS1,
                      DO_PREPARE_RESAMPS,
                      DO_RESAMP,
                      DO_RESAMP_IMAGE,
                      DO_POL_CORRECTION,
                      DO_POL_FR,
                      DO_POL_TEC,
                      DO_POL_PHASE,
                      DO_CROSSMUL, #2013-11-26
                      DO_MOCOMP_BASELINE,
                      DO_SET_TOPOINT1,
                      DO_TOPO,
                      DO_SHADE_CPX2RG,
                      DO_RG_OFFSET,
                      DO_RG_OUTLIERS2,
                      DO_RESAMP_ONLY,
                      DO_SET_TOPOINT2,
                      DO_CORRECT,
                      DO_COHERENCE,
                      DO_FILTER_INF,
                      DO_UNWRAP,
                      DO_GEOCODE_INF,
                      DO_GEOCODE,
                      RENDERER)

    facility_list = (STACK,
                     DEM,
                     DEM_STITCHER,
                     RUN_UPDATE_PREPROC_INFO,
                     RUN_ESTIMATE_HEIGHTS,
                     RUN_SET_MOCOMP_PATH,
                     RUN_RG_OFFSET,
                     RUN_FORM_SLC,
                     RUN_OFFSETPRF,
                     RUN_UNWRAPPER,
                     _ISCE)

    _pickleObj = "_isce"

    def Usage(self):
        print("Usage: isceApp.py <input-file.xml> [options]")
        print("Options:")
        print("None\t\tRun isceApp.py from start to end without pickling")
        print("--help\t\tDisplay configurable parameters and facilities that can be specified in <input-file.xml>")
        print("--help --steps\tDisplay list of available steps according to <input-file.xml>")
        print("--steps\t\tRun isceApp.py from start to end and pickle at each step")


    def __init__(self, family='',name='',cmdline=None):
        """
        Initialize the application: read the xml file and prepare the application.
        """
        super().__init__(family=family if family else  self.__class__.family, name=name,cmdline=cmdline)

        #store the processing start time
        now = datetime.datetime.now()

        self._stdWriter = create_writer("log", "", True, filename="isce.log")
        self._add_methods()
        from isceobj.IsceProc import IsceProc
        self._insarProcFact = IsceProc
        self.pairsToCoreg = [] ##pairs to coregister
        self.intromsg = '' ##intro message
        self.peg = None


    ## You need this to use the FrameMixin
    @property
    def frame(self):
        return self.isce.frame


    def _init(self):
        message =  (
            ("ISCE VERSION = %s, RELEASE_SVN_REVISION = %s,"+
             "RELEASE_DATE = %s, CURRENT_SVN_REVISION = %s") %
            (isce.__version__,
             isce.release_svn_revision,
             isce.release_date,
             isce.svn_revision)
            )
        self.intromsg = message

        print(message)
        if ( self.pegLat is not None
             and self.pegLon is not None
             and self.pegHdg is not None
             and self.pegRad is not None ):
             self.peg = Peg(latitude=self.pegLat,
                            longitude=self.pegLon,
                            heading=self.pegHdg,
                            radiusOfCurvature=self.pegRad)
        #for attribute in ["sensorName", "correlation_method", "use_dop", "geoPosting", "posting", "resampRgLooks", "resampAzLooks", "offsetMethod", "peg"]:
        #    print("%s = %s" % (attribute, getattr(self, attribute)))

    def _configure(self):

        self.isce.procDoc._addItem("ISCE_VERSION",
            "Release: %s, svn-%s, %s. Current svn-%s" %
            (isce.release_version, isce.release_svn_revision,
             isce.release_date, isce.svn_revision
            ),
            ["isceProc"]
            )
        #This method includes logger support
        self.verifyOutput()
        #This is a temporary fix to get the user interface back to the dem
        #facility interface while changes are being made in the DemImage class
        #to include within it the capabilities urrently in extractInfo and
        #createDem.
        if self.demFilename:
            import sys
            print(
            "The demFilename property is no longer supported as an " +
            "input parameter."
                )
            print(
                "The original method using a configurable facility for the " +
                "Dem is now restored."
                )
            print(
                "The automatic download feature is still supported in the " +
                " same way as before:"
                )
            print(
                "If you want automatic download of a Dem, then simply omit "+
                "any configuration\ninformation in your input file regarding "+
                "the Dem."
                )
            print()
            print(
                "Please replace the following information in your input file:"
                )
            print()
            print(
                "<property name='demFilename'><value>%s</value></property>" %
                self.demFilename
                )
            print()
            print("with the following information and try again:")
            print()
            print(
                "<component name=\'Dem\'><catalog>%s</catalog></component>" %
                self.demFilename
                )
            print()
        else:
            try:
                self.dem.checkInitialization()
                self.demFilename = "demFilename"
                self._isce.demImage = self.dem
            except Exception as err:
                pass
                #self.dem was not properly initialized
                #and self.demFilename is undefined.
                #There is a check on self.demFilename
                #below to download if necessary
            else:
                dem_snwe = self.dem.getsnwe()
                if self.geocode_bbox:
                    ####Adjust bbox according to dem
                    if self.geocode_bbox[0] < dem_snwe[0]:
                        logger.warn('Geocoding southern extent changed to match DEM')
                        self.geocode_bbox[0] = dem_snwe[0]

                    if self.geocode_bbox[1] > dem_snwe[1]:
                        logger.warn('Geocoding northern extent changed to match DEM')
                        self.geocode_bbox[1] = dem_snwe[1]

                    if self.geocode_bbox[2] < dem_snwe[2]:
                        logger.warn('Geocoding western extent changed to match DEM')
                        self.geocode_bbox[2] = dem_snwe[2]

                    if self.geocode_bbox[3] > dem_snwe[3]:
                        logger.warn('Geocoding eastern extent changed to match DEM')
                        self.geocode_bbox[3] = dem_snwe[3]

        #Ensure consistency in geocode_list maintained by isceApp and
        #IsceProc. If it is configured in both places, the one in isceApp
        #will be used. It is complicated to try to merge the two lists
        #because IsceProc permits the user to change the name of the files
        #and the linkage between filename and filetype is lost by the time
        #geocode_list is fully configured.  In order to safely change file
        #names and also specify the geocode_list, then isceApp should not
        #be given a geocode_list from the user.
        if(self.geocode_list is None):
            #if not provided by the user use the list from IsceProc
            self.geocode_list = self._isce.geocode_list
        else:
            #if geocode_list defined here, then give it to IsceProc
            #for consistency between isceApp and IsceProc and warn the user

            #check if the two geocode_lists differ in content
            g_count = 0
            for g in self.geocode_list:
                if g not in self._isce.geocode_list:
                    g_count += 1
            #warn if there are any differences in content
            if g_count > 0:
                print()
                logger.warn((
                    "Some filenames in isceApp.geocode_list configuration "+
                    "are different from those in IsceProc. Using names given"+
                    " to isceApp."))
                print("isceApp.geocode_list = {}".format(self.geocode_list))
                print(("IsceProc.geocode_list = {}".format(
                        self._isce.geocode_list)))

            self._isce.geocode_list = self.geocode_list

        return None


    @property
    def isce(self):
        return self._isce
    @isce.setter
    def isce(self, value):
        self._isce = value

    @property
    def procDoc(self):
        return self._isce.procDoc
    @procDoc.setter
    def procDoc(self):
        raise AttributeError(
            "Can not assign to .isce.procDoc-- but you hit all its other stuff"
            )

    def _finalize(self):
        pass


    def help(self):
        print(self.__doc__)
        lsensors = list(SENSORS.keys())
        lsensors.sort()
        print("The currently supported sensors are: ", lsensors)


    def help_steps(self):
        print(self.__doc__)
        print("A description of the individual steps can be found in the README file")
        print("and also in the ISCE.pdf document")


    def formatAttributes(self):
        self.sensorName = self.sensorName.upper()
        if not self.dopplerMethod.startswith('use'):
            self.dopplerMethod = 'use' + self.dopplerMethod
        self.selectedPols = list(map(str.lower, self.selectedPols))


    def prepareStack(self):
        """
        Populate stack with user data and prepare to run.
        """
        ##Get all scenes as given in xml file
        sceneids = []
        allscenes = []
        scenekeys = []
        for i in range(100):
            try:
                scene = getattr(self.stack, 'scene'+str(i))
            except AttributeError:
                pass
            else:
                if scene:
                    sceneids.append(scene['id'])
                    allscenes.append(scene)
                    scenekeys.extend(scene.keys())
        unique_scenekeys = set(scenekeys)
        sels = []
        for scene in self.selectedScenes:
            pairs = scene.split('-')
            if len(pairs) == 1:
                sid = pairs[0]
                try:
                    idx = sceneids.index(sid)
                except ValueError:
                    sys.exit("Scene id '%s' is not in list of scenes." % sid)
                else:
                    sels.append(sid)
            elif len(pairs) == 2:
                sid1 = pairs[0].strip()
                sid2 = pairs[1].strip()
                try:
                    idx1 = sceneids.index(sid1)
                    idx2 = sceneids.index(sid2)
                except ValueError as e:
                    print(e)
                    print(sceneids)
                    sys.exit(1)
                else:
                    first = min(idx1, idx2)
                    last = max(idx1, idx2)
                    for i in range(first, last+1):
                        sels.append(sceneids[i])
            else:
                sys.exit("Unknow value '%s' in selected scenes." % scene)

        # make sure that we have unique selected scenes ordered by their scene number
        self.selectedScenes = [ s for s in sceneids if s in sels ]
        if not self.selectedScenes: ##no scenes selected: process all scenes
            self.selectedScenes = sceneids
        for sceneid in self.selectedScenes:
            idx = sceneids.index(sceneid)
            scene = allscenes[idx]
            self.stack.addscene(scene)
            outdir = self.getoutputdir(sceneid)
            if not os.path.exists(outdir):
                os.mkdir(outdir)

        sels = []
        if not self.selectedPols: ##empty pols
            self.selectedPols = list(POLS) ##select all pols
        for pol in self.selectedPols:
            if pol in POLS:
                if pol in unique_scenekeys: ##the selected pols might not be in the givenkeys
                    sels.append(pol)
            else:
                sys.exit("Polarization '%s' is not in accepted list." % pol)
        if not sels:
            sys.exit("Make sure that all scenes have at least one accepted polarization: %s" % ', '.join(POLS))

        # make sure that we have unique selected pols in the same order as in POLS
        self.selectedPols = [ p for p in POLS if p in sels ]

        selPairs = []
        for pair in self.selectedPairs:
            try:
                scene1, scene2 = map(str.strip, pair.split('/')) # assume that it's a pair scene1/scene2
                if scene1 in self.selectedScenes and scene2 in self.selectedScenes:
                    selPairs.append( (scene1, scene2) )
            except ValueError: # not p1/p2
                try:
                    sid1, sid2 = map(str.strip, pair.split('-')) # assume that it's a range first-last
                    idx1 = sceneids.index(sid1)
                    idx2 = sceneids.index(sid2)
                    first = min(idx1, idx2)
                    last = max(idx1, idx2)
                    for i in range(first, last):
                        for j in range(i + 1, last + 1): #KK 2013-12-17
                            selPairs.append( (sceneids[i], sceneids[j]) )
                except ValueError: # unknown format
                    sys.exit("Unknow format in <selectPairs>: %s" % pair)

        #keep unique values.
        #pairs like (scene1, scene2) and (scene2, scene1) are considered different here
        #they will be processed as different pairs for now;
        #we might need to check that and remove one of the pairs (to be done)
        self.selectedPairs = list(set(selPairs))

        if not self.selectedPairs: ##empty value
            self.selectedPairs = []
            nbscenes = len(self.selectedScenes)
            for i in range(nbscenes):
                for j in range(i+1, nbscenes):
                    self.selectedPairs.append((self.selectedScenes[i], self.selectedScenes[j]))

        if self.refPol not in self.selectedPols:
            self.refPol = self.selectedPols[0] # get first selected polarization
        if self.refScene not in self.selectedScenes:
            self.refScene = self.selectedScenes[0] # get first selected scene

        if self.do_offsetprf or not self.do_offsetprf:
            #list of scenes that compose selected pairs
            scenesInPairs = []
            for pair in self.selectedPairs:
                #add scene1 and scene2 to list
                scenesInPairs.extend(pair)
            #keep unique values
            scenesInPairs = list(set(scenesInPairs))
            #order scenes by their scene number
            orderedScenesInPairs = [ s for s in self.selectedScenes if s in scenesInPairs ]

            if self.coregStrategy == 'single reference':
                for scene in orderedScenesInPairs:
                    self.pairsToCoreg.append( (self.refScene, scene) )
                if (self.refScene, self.refScene) in self.pairsToCoreg:
                    self.pairsToCoreg.remove( (self.refScene, self.refScene) )
            elif self.coregStrategy == 'cascade':
                for i in range(len(orderedScenesInPairs)-1):
                    self.pairsToCoreg.append((orderedScenesInPairs[i], orderedScenesInPairs[i+1]))
            else:
                sys.exit("Unknown coregistration strategy in runOffsetprf", self.coregStrategy)

            # creating output directories according to selectedPairs and pairsToCoreg
            #copy pairsToCoreg
            outputPairs = list(self.pairsToCoreg)
            for (p1, p2) in self.selectedPairs:
                #(p2, p1) might be already in pairsToCoreg but we consider them as different pairs
                if (p1, p2) not in self.pairsToCoreg:
                    outputPairs.append((p1, p2))
            for (p1, p2) in outputPairs:
                outdir = self.getoutputdir(p1, p2)
                if not os.path.exists(outdir):
                    os.mkdir(outdir)

        self._isce.selectedPols = self.selectedPols
        self._isce.selectedScenes = self.selectedScenes
        self._isce.selectedPairs = self.selectedPairs
        self._isce.coregStrategy = self.coregStrategy
        self._isce.refScene = self.refScene
        self._isce.refPol = self.refPol
        self._isce.pairsToCoreg = self.pairsToCoreg
        self._isce.srcFiles = self.stack.getscenes()

    def getoutputdir(self, sid1, sid2=''):
        """
        Return output directory for scene sid1.
        If sid2 is given, return output directory for pair sid1__sid2.
        """
        if sid2:
            outdir = '%s__%s' % (sid1, sid2)
        else:
            outdir = sid1
        return os.path.join(self.outputDir, outdir)


    def verifyOutput(self):
        """
        Check that output directory exists and instantiate logger.
        """
        global logger
        if not os.path.isdir(self.outputDir):
            sys.exit("Could not find the output directory: %s" % self.outputDir)
        os.chdir(self.outputDir) ##change working directory to given output directory

        logger = logging.getLogger('isce.isceProc')
        logger.info(self.intromsg)
        self._isce.dataDirectory = self.outputDir
        self._isce.processingDirectory = self.outputDir


    ## Method return True iff it changes the demFilename.
    from isceobj.Util.decorators import use_api
    @use_api
    def verifyDEM(self):
        #if an image has been specified, then no need to create one
        if not self.dem.filename:
            #the following lines should be included in the check on demFilename
            frames = self._isce.getAllFromPol(self._isce.refPol, self._isce.frames)
            info = self.extractInfo(frames)
            self.createDem(info)
        else:
            self._isce.demImage = self.dem
            #ensure that the dem vrt file exists by creating (or recreating) it
            self._isce.demImage.renderVRT()

        #at this point a dem image has been set into self._isce, whether it
        #was sitched together or read in input
        demImage =  self._isce.demImage
        #if the demImage is already in wgs84 (because was provided in input) then skip and proceed
        if demImage.reference.upper() != 'WGS84':
            wgs84demFilename = self._isce.demImage.filename+'.wgs84'
            wgs84demxmlFilename = wgs84demFilename+'.xml'
            #if the dem reference is EGM96 and the WGS84 corrected
            #dem files are not found, then create the WGS84 files
            #using the demStitcher's correct method
            if( demImage.reference.upper() == 'EGM96' and
                not (os.path.isfile(wgs84demFilename) and
                     os.path.isfile(wgs84demxmlFilename))
            ):
                self._isce.demImage = self.demStitcher.correct(demImage)
            #make sure to load the wgs84 if present
            elif(os.path.isfile(wgs84demFilename) and
                     os.path.isfile(wgs84demxmlFilename)):
                from isceobj import createDemImage
                self._isce.demImage  = createDemImage()
                self._isce.demImage.load(wgs84demxmlFilename)
                if(self._isce.demImage.reference.upper() != 'WGS84'):
                    print('The dem',wgs84demFilename,'is not wgs84')
                    raise Exception
            #ensure that the wgs84 dem vrt file exists
            self._isce.demImage.renderVRT()

        #get water mask
        #self.runCreateWbdMask(info)

        return None

    def renderProcDoc(self):
        self._isce.procDoc.renderXml()


    ## Run Offoutliers() repeatedly with arguments from "iterator" keyword
    def iterate_runOffoutliers(self, iterator=None):
        """
        runs runOffoutliers multiple times with values (integers) from iterator.
        iterator defaults to Stack._default_culling_sequence
        """
        if iterator is None:
            iterator = self.culling_sequence
        map(self.runOffoutliers, iterator)


    def set_topoint1(self):
        self._isce.topoIntImages = dict(self._isce.resampIntImages)


    def set_topoint2(self):
        self._isce.topoIntImages = dict(self._isce.resampOnlyImages)


    def startup(self):
        self.help()
        self.formatAttributes()
        self.prepareStack()
        self.timeStart = time.time()


    def endup(self):
        self.renderProcDoc()

    ## Add instance attribute RunWrapper functions, which emulate methods.
    def _add_methods(self):
        self.runPreprocessor = IsceProc.createPreprocessor(self)
        self.extractInfo = IsceProc.createExtractInfo(self)
        self.createDem = IsceProc.createCreateDem(self)
        self.runPulseTiming = IsceProc.createPulseTiming(self)
        self.runOrbit2sch = IsceProc.createOrbit2sch(self)
        self.updatePreprocInfo = IsceProc.createUpdatePreprocInfo(self)
        self.runOffoutliers = IsceProc.createOffoutliers(self)
        self.prepareResamps = IsceProc.createPrepareResamps(self)
        self.runResamp = IsceProc.createResamp(self)
        self.runResamp_image = IsceProc.createResamp_image(self)
        self.runISSI = IsceProc.createISSI(self)
        self.runCrossmul = IsceProc.createCrossmul(self) #2013-11-26
        self.runMocompbaseline = IsceProc.createMocompbaseline(self)
        self.runTopo = IsceProc.createTopo(self)
        self.runCorrect = IsceProc.createCorrect(self)
        self.runShadecpx2rg = IsceProc.createShadecpx2rg(self)
        self.runResamp_only = IsceProc.createResamp_only(self)
        self.runCoherence = IsceProc.createCoherence(self)
        self.runFilter = IsceProc.createFilter(self)
        self.runGrass = IsceProc.createGrass(self)
        self.runGeocode = IsceProc.createGeocode(self)


    def _steps(self):
        self.step('startup', func=self.startup,
            doc="Print a helpful message and set the startTime of processing",
            dostep=True)

        # Run a preprocessor for the sets of frames
        self.step('preprocess', func=self.runPreprocessor,
            doc="Preprocess scenes to raw images", dostep=self.do_preprocess)

        # Verify whether the DEM was initialized properly. If not, download a DEM
        self.step('verifyDEM', func=self.verifyDEM, dostep=self.do_verifyDEM)

        # Run pulsetiming for each set of frames
        self.step('pulsetiming', func=self.runPulseTiming, dostep=self.do_pulsetiming)

        # Estimate heights
        self.step('estimateHeights', func=self.runEstimateHeights, dostep=self.do_estimateheights)

        # Run setmocomppath
        self.step('mocompath', func=self.runSetmocomppath, args=(self.peg,),
            dostep=self.do_mocomppath)

        #init and run orbit2sch
        self.step('orbit2sch', func=self.runOrbit2sch, dostep=self.do_orbit2sch)

        #update quantities in objPreProc obtained from previous steps
        self.step('updatepreprocinfo', func=self.updatePreprocInfo,
                  args=(self.use_dop,), dostep=self.do_updatepreprocinfo)

        #form the single look complex image
        self.step('formslc', func=self.runFormSLC, dostep=self.do_formslc)

        #Get the list of polarimetric operations to be performed
        polopList = []
        if self.do_pol_correction:
            polopList.append('polcal')
        if self.do_pol_fr:
            polopList.append('fr')
        if self.do_pol_tec:
            polopList.append('tec')
        if self.do_pol_phase:
            polopList.append('phase')
        self.do_pol_correction = True if polopList else False

        # run polarimetric correction if polopList is not empty
        self.step('pol_correction', func=self.runISSI, args=(polopList,), dostep=self.do_pol_correction)

        self.step('offsetprf', func=self.runOffsetprf, dostep=self.do_offsetprf)

        # cull offoutliers
        self.step('outliers1', func=self.iterate_runOffoutliers,
            dostep=self.do_outliers1)

        # determine rg and az looks
        self.step('prepareresamps', func=self.prepareResamps,
            args=(self.resampRgLooks, self.resampAzLooks),
            dostep=self.do_prepareresamps)

        # output resampled slc (skip int and amp files)
        self.step('resamp', func=self.runResamp, dostep=self.do_resamp)

        # output images of offsets
        self.step('resamp_image', func=self.runResamp_image,
            dostep=self.do_resamp_image)

        # run crossmultiplication (output int and amp)
        self.step('crossmul', func=self.runCrossmul, dostep=self.do_crossmul)

        # mocompbaseline
        self.step('mocompbaseline', func=self.runMocompbaseline,
            dostep=self.do_mocompbaseline)

        # assign resampIntImage to topoIntImage
        self.step('settopoint1', func=self.set_topoint1,
            dostep=self.do_settopoint1)

        self.step('topo', func=self.runTopo, dostep=self.do_topo)

        self.step('shadecpx2rg', func=self.runShadecpx2rg,
            dostep=self.do_shadecpx2rg)

        # compute offsets and cull offoutliers
        self.step('rgoffset', func=self.runRgoffset, dostep=True)

        self.step('rg_outliers2', func=self.iterate_runOffoutliers,
            dostep=self.do_rg_outliers2)

        self.step('resamp_only', func=self.runResamp_only, dostep=self.do_resamp_only)

        # assign resampOnlyImage to topoIntImage
        self.step('settopoint2', func=self.set_topoint2, dostep=self.do_settopoint2)

        self.step('correct', func=self.runCorrect, dostep=self.do_correct)

        # coherence
        self.step('coherence', func=self.runCoherence,
            args=(self.correlation_method,), dostep=self.do_coherence)

        # filter
        self.step('filterinf', func=self.runFilter,
            args=(self.filterStrength,), dostep=self.do_filterinf)

        # unwrap
        self.step('unwrap', func=self.runUnwrapper, dostep=self.do_unwrap)

        # geocode
        self.step('geocodeinf', func=self.runGeocode,
            args=(self.geocode_list, self.do_unwrap, self.geocode_bbox),
            dostep=self.do_geocode)

#        self.step('endup', func=self.endup, dostep=True)


    def main(self):
        """
        Run the given processing steps.
        """
        self.startup()

        if self.do_preprocess:
            # Run a preprocessor for the sets of frames
            self.runPreprocessor()

        if self.do_verifyDEM:
            # Verify whether user defined  a dem component.  If not, then download
            # SRTM DEM.
            self.verifyDEM()

        if self.do_pulsetiming:
            # Run pulsetiming for each set of frames
            self.runPulseTiming()

        if self.do_estimateheights:
            self.runEstimateHeights()

        if self.do_mocomppath:
            # Run setmocomppath
            self.runSetmocomppath(peg=self.peg)

        if self.do_orbit2sch:
            # init and run orbit2sch
            self.runOrbit2sch()

        if self.do_updatepreprocinfo:
            # update quantities in objPreProc obtained from previous steps
            self.updatePreprocInfo(use_dop=self.use_dop)

        if self.do_formslc:
            self.runFormSLC()

        polopList = []
        if self.do_pol_correction:
            polopList.append('polcal')
        if self.do_pol_fr:
            polopList.append('fr')
        if self.do_pol_tec:
            polopList.append('tec')
        if self.do_pol_phase:
            polopList.append('phase')
        if polopList:
            self.runISSI(polopList)

        if self.do_offsetprf:
            self.runOffsetprf()

        if self.do_outliers1:
            # Cull offoutliers
            self.iterate_runOffoutliers()

        if self.do_prepareresamps:
            self.prepareResamps(self.resampRgLooks, self.resampAzLooks)

        if self.do_resamp:
            self.runResamp()

        if self.do_resamp_image:
            self.runResamp_image()

        if self.do_crossmul: #2013-11-26
            self.runCrossmul()

        if self.do_mocompbaseline:
            # mocompbaseline
            self.runMocompbaseline()

        if self.do_settopoint1:
            # assign resampIntImage to topoIntImage
            self.set_topoint1()

        if self.do_topo:
            # topocorrect
            self.runTopo()

        if self.do_shadecpx2rg:
            self.runShadecpx2rg()

        self.runRgoffset()

        if self.do_rg_outliers2:
            # Cull offoutliers
            self.iterate_runOffoutliers()

        if self.do_resamp_only:
            self.runResamp_only()

        if self.do_settopoint2:
            self.set_topoint2()

        if self.do_correct:
            self.runCorrect()

        if self.do_coherence:
            # Coherence ?
            self.runCoherence(method=self.correlation_method)

        if self.do_filterinf:
            # Filter ?
            self.runFilter(self.filterStrength) #KK 2013-12-12 filterStrength as argument

        if self.do_unwrap:
            # Unwrap ?
            self.runUnwrapper() #KK 2013-12-12 instead of self.verifyUnwrap()

        if self.do_geocode:
            # Geocode
            self.runGeocode(self.geocode_list, self.do_unwrap, self.geocode_bbox)

        self.endup()



if __name__ == "__main__":
    if not isce.stanford_license:
        print("This workflow requires the Stanford licensed code elemnts.")
        print("Unable to find the license information in the isce.stanford_license file.")
        print("Please either obtain a stanford license and follow the instructions to")
        print("install the stanford code elements or else choose a different workflow.")
        raise SystemExit(0)
    else:
        #create the isce object
        isceapp = IsceApp(name='isceApp')
        #configure the isceapp object
        isceapp.configure()
        #invoke the Application base class run method, which returns status
        status = isceapp.run()
        #inform Python of the status of the run to return to the shell
        raise SystemExit(status)
