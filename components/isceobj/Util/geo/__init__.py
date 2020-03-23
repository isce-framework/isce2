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



"""geo is for doing coordinates on Earth. Here are the modules:


euclid           Scalar, Vector, Tensor objects in E3 -eucliden 3-space.
charts           rotations in E3, aka: charts on SO(3).
affine           rigid affine transformations in E3.
coordinates      Coordinates on Earth
ellipsoid        oblate ellipsoid of revolution (e.g, WGS84) with all the
                 bells and whistles.


  Note: sub-package use __all__, so they are:
  >>>from geo import *
  safe.

  See mainpage.txt for a complete dump of geo's philosophy-- otherwise,
  use the docstrings.
"""

## \namespace geo  Vector- and Affine-spaces, on Earth
__all__ = ['euclid', 'coordinates', 'ellipsoid', 'charts', 'affine', 'motion']
