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
from isceobj.ImageFilter.ComplexExtractor import ComplexExtractor
from isceobj.ImageFilter.BandExtractor import BandExtractor



def createFilter(typeExtractor,fromWhat):
    """Extractor factory"""
    instanceType = ''
    #What is passed here -> how it is passed to the FilterFactory.cpp -> What is instantiated in FilterFactory.cpp
    #(MagnitudeExtractor,'cartesian') -> (MagnitudeExtractor,0) -> MagnitudeExtractor
    #(MagnitudeExtractor,'polar') -> (ComponentExtractor,0) -> ComponentExtractor, 0
    #(PhaseExtractor,'cartesian') -> (PhaseExtractor,0) -> PhaseExtractor
    #(PhaseExtractor,'polar') -> (ComponentExtractor,1) -> ComponentExtractor, 1
    #(RealExtractor,'cartesian') -> (ComplexExtractor,0) -> ComponentExtractor 0
    #(ImagExtractor,'cartesian') -> (ComplexExtractor,1) -> ComponentExtractor 1
    #(RealExtractor,'polar') -> (RealExtractor,0) -> RealExtractor 
    #(ImagExtractor,'polar') -> (ImagExtractor,1) -> ImagExtractor 
    #(BandExtractor,band) -> (BandExtractor,band) -> BandExtractor band
    if typeExtractor.lower() == 'magnitudeextractor' and fromWhat.lower()  == 'cartesian':
        return ComplexExtractor('MagnitudeExtractor',0)
    elif typeExtractor.lower() == 'magnitudeextractor' and fromWhat.lower()  == 'polar':
        return ComplexExtractor('ComponentExtractor',0)
    elif typeExtractor.lower() == 'phaseextractor' and fromWhat.lower()  == 'cartesian':
        return ComplexExtractor('PhaseExtractor',0)
    elif typeExtractor.lower() == 'phaseextractor' and fromWhat.lower()  == 'polar':
        return ComplexExtractor('ComponentExtractor',1)
    elif typeExtractor.lower() == 'realextractor' and fromWhat.lower()  == 'cartesian':
        return ComplexExtractor('ComponentExtractor',0)
    elif typeExtractor.lower() == 'imagextractor' and fromWhat.lower()  == 'cartesian':
        return ComplexExtractor('ComponentExtractor',1)
    elif typeExtractor.lower() == 'realextractor' and fromWhat.lower()  == 'polar':
        return ComplexExtractor('RealExtractor',0)
    elif typeExtractor.lower() == 'imagextractor' and fromWhat.lower()  == 'polar':
        return ComplexExtractor('ImagExtractor',0)
    elif typeExtractor.lower() == 'bandextractor':
        #in this case fromWhat it's actually the band to extract
        return BandExtractor(typeExtractor,fromWhat)
    

    




if __name__ == "__main__":
    sys.exit(main())
