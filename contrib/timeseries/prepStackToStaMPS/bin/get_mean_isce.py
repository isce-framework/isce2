#!/usr/bin/env python3 


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2016 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: David Bekaert
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




import sys
import osgeo
from osgeo import gdal
import os

# take priority for the vrt as the vrt has a no-data string in it.
file_in =sys.argv[1]
filename, file_extension = os.path.splitext(file_in)
if file_extension != '.vrt':
    vrt_str = ''
    if os.path.isfile(file_in + '.vrt'):
        vrt_str = '.vrt'
file_in = os.path.abspath(sys.argv[1] + vrt_str)
dataset_avg = gdal.Open(file_in,gdal.GA_ReadOnly)
stats =   dataset_avg.GetRasterBand(1).GetStatistics(0,1)
mean= stats[2]
dataset_avg = None

print(mean)




