#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2014 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Eric Belz
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




## \namespace rdf.units.addendum Non metric and user units.
"""his modules instantiates units that do not fit the:

<prefix><metric>

format. Units are collected in to tuples of like dimension, however, that
is utterly unessesary, as the mere act of instaniation memoizes them
in the GLOSSARY

Users could add units here, or perhaps read them from an input file
"""
import operator
import math
from iscesys.Parsers.rdf.units.physical_quantity import *

dBPower('dB', 1)

## Supported _Length conversions
LENGTHS = (Length('in', 0.0254),
           Length('ft', 0.3048),
           Length('mi', 1.609344e3),
           Length('m/pixel', 1))

MASSES = (Mass('g', 0.001), )


## Supported _Area conversions
AREAS = (Area('mm*mm', 1e-6),
         Area('cm*cm', 1e-4),
         Area('km*km', 1e6),
         Area('in*in', 6.4516e-4),
         Area('ft*ft', 9.290304e-2),
         Area('mi*mi', 2.58995511e6))

## Supported _Time conversions
TIMES = (Time('min', 60),
         Time('hour', 3600),
         Time('day', 86400),
         Time('sec', 1),
         Time('microsec', 1e-6))


## Supported _Velocity conversions
VELOCITES = (Velocity('km/hr', operator.truediv(5, 18)),
             Velocity('ft/s', 0.3048),
             Velocity('mi/h', 0.44704))

POWERS = ()

## Supported dB Power
DBPOWERS = (dBPower('dBm', adder=-30),)

## Supported Frequency conversions
FREQUENCIES = (Frequency('rpm', operator.truediv(1,60)),
               Frequency('hz', 1),
               Frequency('Mhz', 1e6))

BYTES = (Byte('bytes', 1),)
PIXELS = (Pixel('pixels', 1),)

## Supported Angle conversions
ANGLES = (Angle('deg', operator.truediv(math.pi,180)),
          Angle('"', operator.truediv(math.pi, 180*3600)),
          Angle("'", operator.truediv(math.pi, 180*60)),
          Angle("arcsec", operator.truediv(math.pi, 180*3600)))

## Supported Temperature Conversions
TEMPERATURES = (Temperature('degK', 1.0, 273),
                Temperature('degF', operator.truediv(5, 9), -32.0))
#                Temperature('eV', 1.602176565e-19/1.3806488e-23))
                



