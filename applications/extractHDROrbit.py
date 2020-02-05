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




import datetime
from isce import logging
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from iscesys.Component.FactoryInit import FactoryInit

class extractHDROrbit(FactoryInit):

    def main(self):
        # Parse the image metadata and extract the image
        self.logger.info('Parsing image metadata')
        self.sensorObj.parse()
        frame = self.sensorObj.getFrame()
        for sv in frame.getOrbit():
            epoch = self.datetimeToEpoch(sv.getTime())
            (x,y,z) = sv.getPosition()
            (vx,vy,vz) = sv.getVelocity()
            print(epoch,x,y,z,vx,vy,vz)

    def datetimeToEpoch(self,dt):
        epoch = dt.hour*60*60 + dt.minute*60 + dt.second
        return epoch

    def __init__(self,arglist):
        FactoryInit.__init__(self)
        self.initFactory(arglist)
        self.sensorObj = self.getComponent('Sensor')
        self.logger = logging.getLogger('isce.extractHDROrbits')

if __name__ == "__main__":
    import sys
    if (len(sys.argv) < 2):
        print("Usage:%s <xml-parameter file>" % sys.argv[0])
        sys.exit(1)
    runObj = extractHDROrbit(sys.argv[1:])
    runObj.main()
