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
from isceobj.ImageFilter import Filter as FL
from isceobj.Image.Image import Image
import logging


class Filter:
#Use kwargs so possible subclasses can add parameters to the init function.  

    def init(self,imgIn,nameOut,**kwargs):
        """Abstract method"""
        raise NotImplementedError

    def finalize(self):
        """Call to the bindings finalize. Subclass can extend it but needs to call the baseclass one"""
        FL.finalize(self._filter)
        

    def extract(self):
        """Perform the data extraction"""
        FL.extract(self._filter)


#This is specific to the extract band filter. Put in the base class all the methods
#we need for the provided filters. New filters will implement their own if needed
#in the subclass
    
    def selectBand(self,band):
        """Select a specified band from the Image"""
        FL.selectBand(self._filter,band)
    
    def setStartLine(self,line):
        """Set the line where extraction should start"""
        FL.setStartLine(self._filter,line)
    
    def setEndLine(self,line):
        """Set the line where extraction should end"""
        FL.setEndLine(self._filter,line)
    
    def __init__(self):
        #get the filter C++ object pointer
        self._filter = None
        self._imgOut = None 



if __name__ == "__main__":
    sys.exit(main())
