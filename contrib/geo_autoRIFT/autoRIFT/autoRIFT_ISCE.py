#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2019 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Yang Lei
#
# Note: this is based on the MATLAB code, "auto-RIFT", written by Alex Gardner,
#       and has been translated to Python and further optimized.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import isce
from iscesys.Component.Component import Component
import pdb
import subprocess
import re
import string
import sys

WALLIS_FILTER_WIDTH = Component.Parameter('WallisFilterWidth',
        public_name='WALLIS_FILTER_WIDTH',
        default = 21,
        type = int,
        mandatory = False,
        doc = 'Width of the Wallis filter to be used for the pre-processing')

CHIP_SIZE_MIN_X = Component.Parameter('ChipSizeMinX',
        public_name='CHIP_SIZE_MIN',
        default = 32,
        type = int,
        mandatory = False,
        doc = 'Minimum size (in X direction) of the reference data window to be used for correlation')

CHIP_SIZE_MAX_X = Component.Parameter('ChipSizeMaxX',
        public_name='CHIP_SIZE_MAX',
        default = 64,
        type = int,
        mandatory = False,
        doc = 'Maximum size (in X direction) of the reference data window to be used for correlation')

CHIP_SIZE_0X = Component.Parameter('ChipSize0X',
        public_name='CHIP_SIZE_0X',
        default = 32,
        type = int,
        mandatory = False,
        doc = 'Minimum acceptable size (in X direction) of the reference data window to be used for correlation without resampling the grid; if a chip size greater than this value is provided, need to resize the sampling grid')

GRID_SPACING_X = Component.Parameter('GridSpacingX',
        public_name='GRID_SPACING_X',
        default = 32,
        type = int,
        mandatory = False,
        doc = 'Spacing (in X direction) of the sampling grid')

SCALE_CHIP_SIZE_Y = Component.Parameter('ScaleChipSizeY',
        public_name='SCALE_CHIP_SIZE_Y',
        default = 1,
        type = float,
        mandatory = False,
        doc = 'Scaling factor to get the Y-directed chip size in reference to the X-directed sizes')

SEARCH_LIMIT_X = Component.Parameter('SearchLimitX',
        public_name='SEARCH_LIMIT_X',
        default = 25,
        type = int,
        mandatory = False,
        doc = 'Limit (in X direction) of the search data window to be used for correlation')

SEARCH_LIMIT_Y = Component.Parameter('SearchLimitY',
        public_name='SEARCH_LIMIT_Y',
        default = 25,
        type = int,
        mandatory = False,
        doc = 'Limit (in Y direction) of the search data window to be used for correlation')

SKIP_SAMPLE_X = Component.Parameter('SkipSampleX',
        public_name = 'SKIP_SAMPLE_X',
        default = 32,
        type = int,
        mandatory = False,
        doc = 'Number of samples to skip between windows in X (range) direction.')

SKIP_SAMPLE_Y = Component.Parameter('SkipSampleY',
        public_name = 'SKIP_SAMPLE_Y',
        default = 32,
        type = int,
        mandatory=False,
        doc = 'Number of lines to skip between windows in Y ( "-" azimuth) direction.')

FILL_FILT_WIDTH = Component.Parameter('fillFiltWidth',
        public_name = 'FILL_FILT_WIDTH',
        default = 3,
        type = int,
        mandatory=False,
        doc = 'light interpolation Fill Filter width')

MIN_SEARCH = Component.Parameter('minSearch',
        public_name = 'MIN_SEARCH',
        default = 6,
        type = int,
        mandatory=False,
        doc = 'minimum search limit')

SPARSE_SEARCH_SAMPLE_RATE = Component.Parameter('sparseSearchSampleRate',
        public_name = 'SPARSE_SEARCH_SAMPLE_RATE',
        default = 4,
        type = int,
        mandatory=False,
        doc = 'sparse search sample rate')

FRAC_VALID = Component.Parameter('FracValid',
        public_name = 'FRAC_VALID',
        default = 8/25,
        type = float,
        mandatory=False,
        doc = 'Fraction of valid displacements')

FRAC_SEARCH = Component.Parameter('FracSearch',
        public_name = 'FRAC_SEARCH',
        default = 0.20,
        type = float,
        mandatory=False,
        doc = 'Fraction of search')

FILT_WIDTH = Component.Parameter('FiltWidth',
        public_name = 'FILT_WIDTH',
        default = 5,
        type = int,
        mandatory=False,
        doc = 'Disparity Filter width')

ITER = Component.Parameter('Iter',
        public_name = 'ITER',
        default = 3,
        type = int,
        mandatory=False,
        doc = 'Number of iterations')

MAD_SCALAR = Component.Parameter('MadScalar',
        public_name = 'MAD_SCALAR',
        default = 4,
        type = int,
        mandatory=False,
        doc = 'Mad Scalar')

COLFILT_CHUNK_SIZE = Component.Parameter('colfiltChunkSize',
        public_name = 'COLFILT_CHUNK_SIZE',
        default = 4,
        type = int,
        mandatory=False,
        doc = 'column filter chunk size')

BUFF_DISTANCE_C = Component.Parameter('BuffDistanceC',
        public_name = 'BUFF_DISTANCE_C',
        default = 8,
        type = int,
        mandatory=False,
        doc = 'buffer coarse corr mask by this many pixels for use as fine search mask')

COARSE_COR_CUTOFF = Component.Parameter('CoarseCorCutoff',
        public_name = 'COARSE_COR_CUTOFF',
        default = 0.01,
        type = float,
        mandatory=False,
        doc = 'coarse correlation search cutoff')

OVER_SAMPLE_RATIO = Component.Parameter('OverSampleRatio',
        public_name = 'OVER_SAMPLE_RATIO',
        default = 16,
        type = int,
        mandatory=False,
        doc = 'factor for pyramid up sampling for sub-pixel level offset refinement')

DATA_TYPE = Component.Parameter('DataType',
         public_name = 'DATA_TYPE',
         default = 0,
         type = int,
         mandatory=False,
         doc = 'Input data type: 0 -> uint8, 1 -> float32')

MULTI_THREAD = Component.Parameter('MultiThread',
         public_name = 'MULTI_THREAD',
         default = 0,
         type = int,
         mandatory=False,
         doc = 'Number of Threads; default specified by 0 uses single core and surpasses the multithreading routine')


try:
    # Try Autorift within ISCE first
    from .autoRIFT import autoRIFT
except ImportError:
    # Try global Autorift
    from autoRIFT import autoRIFT
except:
    raise Exception('Autorift does not appear to be installed.')



class autoRIFT_ISCE(autoRIFT, Component):
    '''
    Class for mapping regular geographic grid on radar imagery.
    '''
    
    parameter_list = (WALLIS_FILTER_WIDTH,
                      CHIP_SIZE_MIN_X,
                      CHIP_SIZE_MAX_X,
                      CHIP_SIZE_0X,
                      GRID_SPACING_X,
                      SCALE_CHIP_SIZE_Y,
                      SEARCH_LIMIT_X,
                      SEARCH_LIMIT_Y,
                      SKIP_SAMPLE_X,
                      SKIP_SAMPLE_Y,
                      FILL_FILT_WIDTH,
                      MIN_SEARCH,
                      SPARSE_SEARCH_SAMPLE_RATE,
                      FRAC_VALID,
                      FRAC_SEARCH,
                      FILT_WIDTH,
                      ITER,
                      MAD_SCALAR,
                      COLFILT_CHUNK_SIZE,
                      BUFF_DISTANCE_C,
                      COARSE_COR_CUTOFF,
                      OVER_SAMPLE_RATIO,
                      DATA_TYPE,
                      MULTI_THREAD)
    
    
    

    def __init__(self):
        
        super(autoRIFT_ISCE, self).__init__()

        

