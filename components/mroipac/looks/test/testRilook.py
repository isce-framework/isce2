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
from iscesys.Component.InitFromDictionary import InitFromDictionary
from mroipac.looks.Rilooks import Rilooks

def main():
    obj = Rilooks()
    infile = "/Users/giangi/TEST_DIR/int_930110_950523/flat_PRC_930110-950523.int"
    outfile = 'testRi'
    rlook = 4;
    alook = 4;
    width = 5700
    height = 2593
    #with all arguments
    #dict = {'INPUT_ENDIANNESS':enIn,'OUTPUT_ENDIANNESS':enOut,'LENGTH':length,'WIDTH':width,'INPUT_IMAGE':infile,'OUTPUT_IMAGE':outfile,'RANGE_LOOK':rlook,'AZIMUTH_LOOK':alook}
    #with only mandatory arguments
    dict = {'WIDTH':width,'INPUT_IMAGE':infile,'OUTPUT_IMAGE':outfile,'RANGE_LOOK':rlook,'AZIMUTH_LOOK':alook}
    initDict = InitFromDictionary(dict)
    obj.initComponent(initDict)
    obj.rilooks()

if __name__ == "__main__":
    sys.exit(main())
