#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Walter Szeliga
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import datetime
from iscesys.Component.Component import Component
from isceobj.Scene.Frame import Frame

FMRATE_VS_PIXEL = Component.Parameter('_fmrateVsPixel',
        public_name = 'FMRATE_VS_PIXEL',
        default = [],
        type = float,
        mandatory = True,
        container = list,
        doc = 'Doppler polynomial coefficients vs pixel number')

BURST_LENGTH = Component.Parameter('_burstLength',
        public_name = 'Burst Length',
        default = None,
        type = int,
        mandatory = True,
        doc = 'Number of pulses in a burst')

BURST_CYCLE_LENGTH = Component.Parameter('_burstCycleLength',
        public_name = 'Burst cycle length',
        default = None,
        type = float,
        mandatory = True,
        doc = 'Number of pulses in a full cycle')

BURST_START_LINES = Component.Parameter('burstStartLines',
        public_name = 'Burst start lines',
        default = [],
        type = float,
        container=list,
        mandatory = True,
        doc = 'Start lines of bursts in SLC')

class FullApertureSwathSLCProduct(Frame):
    """A class to represent a frame along a radar track"""

    family = 'frame'
    logging_name = 'isce.isceobj.scansar.fullapertureswathslcproduct'

    parameter_list = Frame.parameter_list + (BURST_LENGTH,
                      BURST_CYCLE_LENGTH,
                      BURST_START_LINES,
                      FMRATE_VS_PIXEL)


    def __init__(self, name=''):
        super(FullApertureSwathSLCProduct, self).__init__(name=name)
        return None


    @property
    def nbraw(self):
        return self._burstLength

    @nbraw.setter
    def nbraw(self, x):
        self._burstLength = x

    @property
    def ncraw(self):
        return self._burstCycleLength

    @ncraw.setter
    def ncraw(self, x):
        self._burstCycleLength = x

    @property
    def burstLength(self):
        return self._burstLength

    @burstLength.setter
    def burstLength(self, x):
        self._burstLength = x

    @property
    def burstCycleLength(self):
        return self._burstCycleLength

    @burstCycleLength.setter
    def burstCycleLength(self, x):
        self._burstCycleLength = x

    @property
    def fmrateVsPixel(self):
        return self._fmrateVsPixel

    @fmrateVsPixel.setter
    def fmrateVsPixel(self,x):
        self._fmrateVsPixel = x
