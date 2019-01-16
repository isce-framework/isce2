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
from iscesys.Component.InitFromXmlFile import InitFromXmlFile
from isceobj.Image.DemImage import DemImage
from isceobj.Image.IntImage import IntImage
from stdproc.rectify.geocode.Geocode import Geocode

def main():
    referenceOrbit = sys.argv[1] #look for reference_orbit.txt
    fin1 = open(referenceOrbit)
    allLines = fin1.readlines()
    s_mocomp = []
    for line in allLines:
        lineS = line.split()
        s_mocomp.append(float(lineS[2]))
    fin1.close()
    initfileDem = 'DemImage.xml'
    initDem = InitFromXmlFile(initfileDem)
    objDem = DemImage()
    # only sets the parameter
    objDem.initComponent(initDem)
    # it actually creates the C++ object
    objDem.createImage()
    
    initfileTopo = 'TopoImage.xml'
    initTopo = InitFromXmlFile(initfileTopo)
    objTopo = IntImage()
    # only sets the parameter
    objTopo.initComponent(initTopo)
    # it actually creates the C++ object
    objTopo.createImage()
    initFile = 'Geocode.xml' 
    fileInit = InitFromXmlFile(initFile)

    obj = Geocode()
    obj.initComponent(fileInit)
    obj.setReferenceOrbit(s_mocomp)
    obj.geocode(objDem,objTopo)
    geoWidth= obj.getGeoWidth()
    geoLength  = obj.getGeoLength()
    latitudeSpacing = obj.getLatitudeSpacing()
    longitudeSpacing = obj.getLongitudeSpacing()
    minimumGeoLatitude = obj.getMinimumGeoLatitude()
    minimumGeoLongitude = obj.getMinimumGeoLongitude()
    maximumGeoLatitude = obj.getMaximumGeoLatitude()
    maximumGeoLongitude = obj.getMaxmumGeoLongitude()
    print(geoWidth,\
    geoLength,\
    latitudeSpacing,\
    longitudeSpacing,\
    minimumGeoLatitude,\
    minimumGeoLongitude,\
    maximumGeoLatitude,\
    maximumGeoLongitude)
    
    objDem.finalizeImage()
    objTopo.finalizeImage()
if __name__ == "__main__":
    sys.exit(main())
