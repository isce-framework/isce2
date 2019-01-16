#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2012 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Authors: Giangi Sacco, Eric Belz
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



"""Docstring"""

Version = "$Revision: 876$"
# $Source$
from iscesys.Compatibility import Compatibility
from isceobj.Planet.Planet import Planet
from isceobj.Planet.AstronomicalHandbook import c as SPEED_OF_LIGHT

EARTH = Planet(pname='Earth')
EarthGM = EARTH.GM
EarthSpinRate = EARTH.spin
EarthMajorSemiAxis = EARTH.ellipsoid.a
EarthEccentricitySquared = EARTH.ellipsoid.e2

def nu2lambda(nu):
    return SPEED_OF_LIGHT/nu

def lambda2nu(lambda_):
    return SPEED_OF_LIGHT/lambda_
