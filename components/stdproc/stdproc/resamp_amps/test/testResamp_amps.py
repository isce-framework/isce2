#!/usr/bin/env python3

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
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





from __future__ import print_function
import sys
import os
import math
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from isceobj.Image.AmpImage import AmpImage
from iscesys.Component.InitFromXmlFile import InitFromXmlFile
from iscesys.Component.InitFromDictionary import InitFromDictionary
from stdproc.stdproc.resamp_amps.Resamp_amps import Resamp_amps

def main():
    filename = sys.argv[1] #rgoffset.out
    fin = open(filename)
    allLines = fin.readlines()
    locationAc = []
    locationAcOffset = []
    locationDn = []
    locationDnOffset = []
    snr = []
    for line in allLines:
        lineS = line.split()
        locationAc.append(float(lineS[0]))
        locationAcOffset.append(float(lineS[1]))
        locationDn.append(float(lineS[2]))
        locationDnOffset.append(float(lineS[3]))
        snr.append(float(lineS[4]))
    dict = {}
    dict['LOCATION_ACROSS1'] = locationAc
    dict['LOCATION_ACROSS_OFFSET1'] = locationAcOffset
    dict['LOCATION_DOWN1'] = locationDn
    dict['LOCATION_DOWN_OFFSET1'] = locationDnOffset
    dict['SNR1'] = snr
    initDict = InitFromDictionary(dict)
    
    initfileResamp_amps = 'Resamp_amps.xml'
    initResamp_amps = InitFromXmlFile(initfileResamp_amps)
    
    initfileAmpIn = 'AmpImageIn.xml'
    initAmpIn = InitFromXmlFile(initfileAmpIn)
    
    objAmpIn = AmpImage()
    # only sets the parameter
    objAmpIn.initComponent(initAmpIn)
    # it actually creates the C++ object
    objAmpIn.createImage()
    
    initfileAmpOut = 'AmpImageOut.xml'
    initAmpOut = InitFromXmlFile(initfileAmpOut)
    
    objAmpOut = AmpImage()
    # only sets the parameter
    objAmpOut.initComponent(initAmpOut)
    # it actually creates the C++ object
    objAmpOut.createImage()
    obj = Resamp_amps()
    obj.initComponent(initResamp_amps)
    obj.initComponent(initDict)
    obj.resamp_amps(objAmpIn,objAmpOut)

    ulr = obj.getULRangeOffset()
    ula = obj.getULAzimuthOffset()
    urr = obj.getURRangeOffset()
    ura = obj.getURAzimuthOffset()
    lrr = obj.getLRRangeOffset()
    lra = obj.getLRAzimuthOffset()
    llr = obj.getLLRangeOffset()
    lla = obj.getLLAzimuthOffset()
    cr = obj.getCenterRangeOffset()
    ca = obj.getCenterAzimuthOffset()
    print(ulr,ula,urr,ura,lrr,lra,llr,lla,cr,ca)

    objAmpIn.finalizeImage()
    objAmpOut.finalizeImage()
if __name__ == "__main__":
    sys.exit(main())
