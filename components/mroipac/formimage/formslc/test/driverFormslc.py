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
from iscesys.Component.FactoryInit import FactoryInit
from mroipac.formimage.FormSLC import FormSLC
from iscesys.Compatibility import Compatibility
import getopt
Compatibility.checkPythonVersion()

class DriverFormSLC(FactoryInit):
    
            
        
    
    def main(self):
        #get the initialized objects i.e. the raw and slc image and the FormSLC 
        objSlc = self.getComponent('SlcImage')
        objSlc.createImage()
        objRaw = self.getComponent('RawImage')
        objRaw.createImage()
        objFormSlc = self.getComponent('FormSlc')        
        ####
        objFormSlc.formSLCImage(objRaw,objSlc)
        objSlc.finalizeImage()
        objRaw.finalizeImage()

    def __init__(self,argv):
        FactoryInit.__init__(self)
        #call the init factory passing the init file DriverFormSLC.xml as a argument when calling the script
        self.initFactory(argv[1:])

if __name__ == "__main__":
    runObj = DriverFormSLC(sys.argv)
    runObj.main()
