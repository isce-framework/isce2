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




from isce import logging
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from iscesys.Component.FactoryInit import FactoryInit
from mroipac.baseline.Baseline import Baseline

class calculateBaselineApp(FactoryInit):

    def main(self):
        masterFrame = self.populateFrame(self.masterObj)
        slaveFrame = self.populateFrame(self.slaveObj)

        # Calculate the baseline information
        baseline = Baseline()
        baseline.wireInputPort(name='masterFrame',object=masterFrame)
        baseline.wireInputPort(name='slaveFrame',object=slaveFrame)
        baseline.wireInputPort(name='masterOrbit',object=masterFrame.getOrbit())
        baseline.wireInputPort(name='slaveOrbit',object=slaveFrame.getOrbit())
        baseline.wireInputPort(name='ellipsoid',object=masterFrame.getInstrument().getPlatform().getPlanet().get_elp())
        baseline.baseline()
        print(baseline)

    def populateFrame(self,sensorObj):
        # Parse the image metadata and extract the image
        self.logger.info('Parsing image metadata')
        sensorObj.parse()
        frame = sensorObj.getFrame()

        # Calculate the height, height_dt, and velocity
        self.logger.info("Calculating Spacecraft Velocity")
        frame.calculateHeightDt()
        frame.calculateVelocity()

        return frame

    def __init__(self,arglist):
        FactoryInit.__init__(self)
        self.initFactory(arglist)
        self.masterObj = self.getComponent('Master')
        self.slaveObj = self.getComponent('Slave')
        self.logger = logging.getLogger('isce.calculateBaseline')

if __name__ == "__main__":
    import sys
    if (len(sys.argv) < 2):
        print("Usage:%s <xml-parameter file>" % sys.argv[0])
        sys.exit(1)
    runObj = calculateBaselineApp(sys.argv[1:])
    runObj.main()
