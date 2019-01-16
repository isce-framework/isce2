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
from isceobj.Image.AmpImageBase import AmpImage
from iscesys.Component.InitFromXmlFile import InitFromXmlFile
from iscesys.Component.InitFromDictionary import InitFromDictionary
from stdproc.stdproc.resamp_slc.Resamp_slc import Resamp_slc

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
    objAmpIn = AmpImage()
    # only sets the parameter
    # it actually creates the C++ object
    objAmpIn.initImage('alos.int','read',2053)
    objAmpIn.createImage()
    

    objAmpOut = AmpImage()
    objAmpOut.initImage('resampImageOnly.int','write',2053)
    objAmpOut.createImage()
    # only sets the parameter
    # it actually creates the C++ object
    objAmpOut.createImage()
    obj = Resamp_slc()
    obj.setLocationAcross1(locationAc) 
    obj.setLocationAcrossOffset1(locationAcOffset) 
    obj.setLocationDown1(locationDn) 
    obj.setLocationDownOffset1(locationDnOffset) 
    obj.setSNR1(snr)
    obj.setNumberLines(2816) 
    obj.setNumberFitCoefficients(6)
    obj.setNumberRangeBin(2053)
    obj.setDopplerCentroidCoefficients([-0.224691,0,0,0])
    obj.radarWavelength = 0.0562356424
    obj.setSlantRangePixelSpacing(0)
    obj.resamp_slc(objAmpIn,objAmpOut)

    azCarrier = obj.getAzimuthCarrier()
    raCarrier = obj.getRangeCarrier()
    #for i in range(len(azCarrier)):
    #    print(azCarrier[i],raCarrier[i])
    objAmpIn.finalizeImage()
    objAmpOut.finalizeImage()
    print('goodbye')
if __name__ == "__main__":
    sys.exit(main())
