#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2016 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Joshua Cohen
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import time
import sys
from isce import logging

import isce
import isceobj
from isceobj import TopsProc
from isce.applications.topsApp import TopsInSAR
from iscesys.Component.Application import Application
from isceobj.Util.decorators import use_api

logger = logging.getLogger('isce.insar')

WINDOW_SIZE_WIDTH = Application.Parameter(
    'winwidth',
    public_name='Ampcor window width',
    default=32,
    type=int,
    mandatory=False,
    doc='Ampcor main window size width. Used in runDenseOffsets.'
                                         )

WINDOW_SIZE_HEIGHT = Application.Parameter(
    'winhgt',
    public_name='Ampcor window height',
    default=32,
    type=int,
    mandatory=False,
    doc='Ampcor main window size height. Used in runDenseOffsets.'
                                            )

SEARCH_WINDOW_WIDTH = Application.Parameter(
    'srcwidth',
    public_name='Ampcor search window width',
    default=20,
    type=int,
    mandatory=False,
    doc='Ampcor search window size width. Used in runDenseOffsets.'
                                            )

SEARCH_WINDOW_HEIGHT = Application.Parameter(
    'srchgt',
    public_name='Ampcor search window height',
    default=20,
    type=int,
    mandatory=False,
    doc='Ampcor search window size height. Used in runDenseOffsets.'
                                            )

SKIP_SAMPLE_ACROSS = Application.Parameter(
    'skipwidth',
    public_name='Ampcor skip width',
    default=16,
    type=int,
    mandatory=False,
    doc='Ampcor skip across width. Used in runDenseOffsets.'
                                            )

SKIP_SAMPLE_DOWN = Application.Parameter(
    'skiphgt',
    public_name='Ampcor skip height',
    default=16,
    type=int,
    mandatory=False,
    doc='Ampcor skip down height. Used in runDenseOffsets.'
                                            )

OFFSET_MARGIN = Application.Parameter(
    'margin',
    public_name='Ampcor margin',
    default=50,
    type=int,
    mandatory=False,
    doc='Ampcor margin offset. Used in runDenseOffsets.'
                                        )

OVERSAMPLING_FACTOR = Application.Parameter(
    'oversample',
    public_name='Ampcor oversampling factor',
    default=32,
    type=int,
    mandatory=False,
    doc='Ampcor oversampling factor. Used in runDenseOffsets.'
                                            )

ACROSS_GROSS_OFFSET = Application.Parameter(
    'rgshift',
    public_name='Range shift',
    default=0,
    type=int,
    mandatory=False,
    doc='Ampcor gross offset across. Used in runDenseOffsets.'
                                            )

DOWN_GROSS_OFFSET = Application.Parameter(
    'azshift',
    public_name='Azimuth shift',
    default=0,
    type=int,
    mandatory=False,
    doc='Ampcor gross offset down. Used in runDenseOffsets.'
                                            )

OFFSET_SCALING_FACTOR = Application.Parameter(
    'scale_factor',
    public_name='Offset scaling factor',
    default=1.0,
    type=float,
    mandatory=False,
    doc='Offset field unit scaling factor (1.0 default is pixel)'
                                                )

OFFSET_WIDTH = Application.Parameter(
    'offset_width',
    public_name='Offset image nCols',
    default=None,
    type=int,
    mandatory=False,
    doc='Number of columns in the final offset field (calculated in DenseAmpcor).'
                                        )

OFFSET_LENGTH = Application.Parameter(
    'offset_length',
    public_name='Offset image nRows',
    default=None,
    type=int,
    mandatory=False,
    doc='Number of rows in the final offset field (calculated in DenseAmpcor).'
                                        )

OFFSET_TOP = Application.Parameter(
    'offset_top',
    public_name='Top offset location',
    default=None,
    type=int,
    mandatory=False,
    doc='Ampcor-calculated top offset location. Overridden by workflow.'
                                    )

OFFSET_LEFT = Application.Parameter(
    'offset_left',
    public_name='Left offset location',
    default=None,
    type=int,
    mandatory=False,
    doc='Ampcor-calculated left offset location. Overridden by workflow.'
                                    )

SNR_THRESHOLD = Application.Parameter(
    'snr_thresh',
    public_name='SNR Threshold factor',
    default=None,
    type=float,
    mandatory=False,
    doc='SNR Threshold factor used in filtering offset field objects.'
                                        )

FILTER_NULL = Application.Parameter(
    'filt_null',
    public_name='Filter NULL factor',
    default=-10000.,
    type=float,
    mandatory=False,
    doc='NULL factor to use in filtering offset fields to avoid numpy type issues.'
                                    )

FILTER_WIN_SIZE = Application.Parameter(
    'filt_size',
    public_name='Filter window size',
    default=5,
    type=int,
    mandatory=False,
    doc='Window size for median_filter.'
                                        )

OFFSET_OUTPUT_FILE = Application.Parameter(
    'offsetfile',
    public_name='Offset filename',
    default='dense_offsets',
    type=None,
    mandatory=False,
    doc='Filename for gross dense offsets BIL. Used in runDenseOffsets.'
                                            )

FILT_OFFSET_OUTPUT_FILE = Application.Parameter(
    'filt_offsetfile',
    public_name='Filtered offset filename',
    default='filt_dense_offsets',
    type=None,
    mandatory=False,
    doc='Filename for filtered dense offsets BIL.'
                                                )

OFFSET_MODE = Application.Parameter(
    'off_mode',
    public_name='Is offset mode',
    default=True,
    type=bool,
    mandatory=False,
    doc='Application-specific parameter to indicate whether running topsApp or topsOffsetApp.'
                                    )

OFFSET_GEOCODE_LIST = Application.Parameter(
    'off_geocode_list',
    public_name='offset geocode list',
    default=None,
    container=list,
    type=str,
    mandatory=False,
    doc='List of offset-specific files to geocode.'
                                            )

#Basically extends the TopsInSAR class
class TopsOffset(TopsInSAR):

    # Pull TopsInSAR's parameter/facility lists
    parameter_list = TopsInSAR.parameter_list + ( \
                     WINDOW_SIZE_WIDTH,
                     WINDOW_SIZE_HEIGHT,
                     SEARCH_WINDOW_WIDTH,
                     SEARCH_WINDOW_HEIGHT,
                     SKIP_SAMPLE_ACROSS,
                     SKIP_SAMPLE_DOWN,
                     OFFSET_MARGIN,
                     OVERSAMPLING_FACTOR,
                     ACROSS_GROSS_OFFSET,
                     DOWN_GROSS_OFFSET,
                     OFFSET_SCALING_FACTOR,
                     OFFSET_WIDTH,
                     OFFSET_LENGTH,
                     OFFSET_TOP,
                     OFFSET_LEFT,
                     SNR_THRESHOLD,
                     FILTER_NULL,
                     FILTER_WIN_SIZE,
                     OFFSET_OUTPUT_FILE,
                     FILT_OFFSET_OUTPUT_FILE,
                     OFFSET_MODE,
                     OFFSET_GEOCODE_LIST)
    facility_list = TopsInSAR.facility_list

    family = 'topsinsar'
    _pickleObj = '_insar'

    def __init__(self, family='', name='',cmdline=None):
        super().__init__(family=family if family else self.__class__.family, name=name,
                            cmdline=cmdline)
        self._add_methods()

    @use_api
    def main(self):

        timeStart = time.time()

        #self._steps()

        self.runMergeSLCs()
        self.runDenseOffsets()
        self.runCropOffsetGeo()
        self.runOffsetFilter()
        self.runOffsetGeocode()

        timeEnd = time.time()
        print('Total Time: %i seconds' % (timeEnd-timeStart))
        return None

    def _add_methods(self):
        self.verifyDEM = TopsProc.createVerifyDEM(self) ### NOTE: Not independently called, needed for
        self.runGeocode = TopsProc.createGeocode(self)  ###         runGeocode.py
        self.runMergeSLCs = TopsProc.createMergeSLCs(self)
        self.runDenseOffsets = TopsProc.createDenseOffsets(self)
        self.runCropOffsetGeo = TopsProc.createCropOffsetGeo(self)
        self.runOffsetFilter = TopsProc.createOffsetFilter(self)
        self.runOffsetGeocode = TopsProc.createOffsetGeocode(self)
        return None

    def _steps(self):

        self.step('startup', func=self.startup,
                        doc=('Print a helpful message and'+
                             'set the startTime of processing')
                    )

        self.step('mergeSLCs', func=self.runMergeSLCs)

        self.step('denseOffsets', func=self.runDenseOffsets)

        self.step('cropOffsetGeo', func=self.runCropOffsetGeo)

        self.step('offsetFilter', func=self.runOffsetFilter)

        self.step('offsetGeocode', func=self.runOffsetGeocode)

        return None


if __name__ == "__main__":
    topsOffset = TopsOffset(name="topsOffsetApp")
    topsOffset.configure()
    topsOffset.run()
