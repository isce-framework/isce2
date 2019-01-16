#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2012 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





from __future__ import print_function
import sys
import os
import math
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from contrib.demUtils.Correct_geoid_i2_srtm import Correct_geoid_i2_srtm

def main():
    
    from iscesys.StdOEL.StdOELPy import StdOEL as ST
    stdWriter = ST()
    stdWriter.createWriters()
    stdWriter.configWriter("log","",True,"insar.log")
    stdWriter.init()
    obj = Correct_geoid_i2_srtm()
    obj.setInputFilename(sys.argv[1])
    #if outputFilenmae not specified the input one is overwritten
    obj.setOutputFilename(sys.argv[1] + '.id')
   
    obj.setStdWriter(stdWriter)
    obj.setWidth(int(sys.argv[2]))
    obj.setStartLatitude(float(sys.argv[3]))
    obj.setStartLongitude(float(sys.argv[4]))
    obj.setDeltaLatitude(float(sys.argv[5]))
    obj.setDeltaLongitude(float(sys.argv[6]))
    # -1 EGM96 -> WGS84, 1 WGS84 -> EGM96
    obj.setConversionType(int(sys.argv[7]))
    obj.correct_geoid_i2_srtm()
    
if __name__ == "__main__":
    sys.exit(main())
