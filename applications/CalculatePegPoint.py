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
# Author: Walter Szeliga
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





import math
from isce import logging
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from isceobj.Location.Peg import Peg
from iscesys.Component.FactoryInit import FactoryInit

class CalculatePegPoint(FactoryInit):

    def calculatePegPoint(self):
        self.logger.info("Parsing Raw Data")
        self.sensorObj.parse()
        frame = self.sensorObj.getFrame()
        # First, get the orbit nadir location at mid-swath and the end of the scene
        orbit = self.sensorObj.getFrame().getOrbit()
        midxyz = orbit.interpolateOrbit(frame.getSensingMid())
        endxyz = orbit.interpolateOrbit(frame.getSensingStop())
        # Next, calculate the satellite heading from the mid-point to the end of the scene
        ellipsoid = frame.getInstrument().getPlatform().getPlanet().get_elp()
        midllh = ellipsoid.xyz_to_llh(midxyz.getPosition())
        endllh = ellipsoid.xyz_to_llh(endxyz.getPosition())
        heading = ellipsoid.geo_hdg(midllh,endllh)
        # Then create a peg point from this data
        peg = Peg(latitude=midllh[0],longitude=midllh[1],heading=heading,ellipsoid=ellipsoid)
        self.logger.info("Peg Point:\n%s" % peg)

    def __init__(self,arglist):
        FactoryInit.__init__(self)
        self.initFactory(arglist)
        self.sensorObj = self.getComponent('Sensor')
        self.logger = logging.getLogger('isce.calculatePegPoint')

if __name__ == "__main__":
    import sys
    if (len(sys.argv) < 2):
        print("Usage:%s <xml-parameter file>" % sys.argv[0])
        sys.exit(1)
    runObj = CalculatePegPoint(sys.argv[1:])
    runObj.calculatePegPoint()
