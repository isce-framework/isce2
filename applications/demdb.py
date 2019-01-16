#!/usr/bin/env python3


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2018 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Eric Gurrola
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import sys
import isce
from isceobj.InsarProc.createDem import createDem
from iscesys.DataManager import createManager

class INSAR:
    def __init__(self, snwe):
        # flag to control continuation of processing in the case that
        # a dem is not available or cannot be downloaded. Obviously,
        # this should be False for this application
        self.proceedIfZeroDem = False

class SELF:
    def __init__(me, snwe, hiresonly=False):
        me.geocode_bbox = snwe
        me.insar = INSAR(snwe)
        me.demStitcher = createManager('dem1', 'iscestitcher')
        # True indicates, to only download from high res server.
        # False indicates, download high res dem if available,
        # otherwise download from the low res server.
        me.useHighResolutionDemOnly = hiresonly

class INFO:
    def __init__(self, snwe):
        self.extremes = snwe
    def getExtremes(self, x):
        return self.extremes

if __name__=="__main__":
    if len(sys.argv) < 5:
        print("Usage: demdb.py s n w e [h]")
        print("where s, n, w, e are latitude, longitude bounds in degrees")
        print("The optional 'h' flag indicates to only download a high res dem,"+
              "if available.\n"
              "If 'h' is not on the command line, then a low res dem will be "+
              "downloaded,\nif the hi res is not available.")

        sys.exit(0)

    snwe = list(map(float,sys.argv[1:5]))
    print("snwe = ", snwe)
    if 'h' in sys.argv:
        print("set hiresonly to True")
        hiresonly = True
    else:
        hiresonly = False

    self = SELF(snwe, hiresonly)
    info = INFO(snwe)
    createDem(self,info)
