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
# Author: Eric Belz
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



"""A place to store trig functions using degrees-- so if you don't have numpy
you can use math-- but just have numpy

"""
## \namespace geo.trig Trig functions in degrees


import numpy as np

## cosine in degress (math could be <a href="http://numpy.scipy.org/">numpy</a>
cosd = lambda x: np.cos(np.radians(x))
## sine in degrees
sind = lambda x: np.sin(np.radians(x))
## tangent, in degrees
tand = lambda x: np.tan(np.radians(x))
## arc tan in degrees (2 arg)
arctand2 = lambda y, x: np.degrees(np.arctan2(y, x))
## arc tan in degrees (1 arg)
arctand = lambda x: np.degrees(np.arctan(x)) 


