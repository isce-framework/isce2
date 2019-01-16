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
from iscesys.Component.InitFromXmlFile import InitFromXmlFile
from isceobj.Image.SlcImage import SlcImage
Compatibility.checkPythonVersion()
from isceobj.Util.Cpxmag2rg import Cpxmag2rg

def main():
    obj = Cpxmag2rg()
    initfileSlc1 = 'SlcImage1.xml'
    initSlc1 = InitFromXmlFile(initfileSlc1)
    objSlc1 = SlcImage()
    # only sets the parameter
    objSlc1.initComponent(initSlc1)
    # it actually creates the C++ object
    objSlc1.createImage()
    
    
    initfileSlc2 = 'SlcImage2.xml'
    initSlc2 = InitFromXmlFile(initfileSlc2)
    objSlc2 = SlcImage()
    # only sets the parameter
    objSlc2.initComponent(initSlc2)
    # it actually creates the C++ object
    objSlc2.createImage()
    outname = 'testRGOut'
    obj.setOutputImageName(outname)
    obj.cpxmag2rg(objSlc1,objSlc2)
    objSlc1.finalizeImage()
    objSlc2.finalizeImage()
if __name__ == "__main__":
    sys.exit(main())
