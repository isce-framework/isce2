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
from isceobj.Util.Offoutliers import Offoutliers

def main():

#############################
#be careful that the lists are passed by reference and changed during the computation. If need the old one do a  deep copy
############################
    filename = sys.argv[1]
    fin = open(filename)
    allLines = fin.readlines()
    locationAc = []
    locationAcOffset = []
    locationDn = []
    locationDnOffset = []
    snr = []
    distance = 10
    for line in allLines:
        lineS = line.split()
        locationAc.append(float(lineS[0]))
        locationAcOffset.append(float(lineS[1]))
        locationDn.append(float(lineS[2]))
        locationDnOffset.append(float(lineS[3]))
        snr.append(float(lineS[4]))
    obj = Offoutliers()
    obj.setLocationAcross(locationAc)
    obj.setLocationAcrossOffset(locationAcOffset)
    obj.setLocationDown(locationDn)
    obj.setLocationDownOffset(locationDnOffset)
    obj.setSNR(snr)
    sign = [1]*len(snr)
    obj.setSign(sign)
    obj.setDistance(distance)
    obj.offoutliers()
    indxA = obj.getIndexArray()
    '''
    for el in indxA:
        print(el,locationAc[el],locationAcOffset[el],locationDn[el],locationDnOffset[el],snr[el])
    '''
if __name__ == "__main__":
    sys.exit(main())
