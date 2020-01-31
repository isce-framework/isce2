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
import isceobj
from iscesys.Component.FactoryInit import FactoryInit

class ToGeneric(object):
# Convert from a satellite-specific format, to a generic HDF5-based format.

    def __init__(self,rawObj=None):
        self.rawObj = rawObj
        self.logger = logging.getLogger('isce.toGeneric')

    def convert(self):
        from isceobj.Sensor.Generic import Generic
        doppler = isceobj.Doppler.useDOPIQ()
        hhRaw = self.make_raw(self.rawObj,doppler)
        hhRaw.getFrame().getImage().createImage()

        writer = Generic()
        writer.frame = hhRaw.getFrame()
        writer.write('test.h5',compression='gzip')

    def make_raw(self,sensor,doppler):
        """
        Extract the unfocused SAR image and associated data

        @param sensor (\a isceobj.Sensor) the sensor object
        @param doppler (\a isceobj.Doppler) the doppler object
        @return (\a make_raw) a make_raw instance
        """
        from make_raw import make_raw
        import stdproc
        import isceobj

        # Extract raw image
        self.logger.info("Creating Raw Image")
        mr = make_raw()
        mr.wireInputPort(name='sensor',object=sensor)
        mr.wireInputPort(name='doppler',object=doppler)
        mr.make_raw()

        return mr

def main():
    import sys
    import isceobj

    fi = FactoryInit()
    fi.fileInit = sys.argv[1]
    fi.defaultInitModule = 'InitFromXmlFile'
    fi.initComponentFromFile()

    master = fi.getComponent('Master')

    toGeneric = ToGeneric(rawObj=master)
    toGeneric.convert()

if __name__ == "__main__":
    main()
