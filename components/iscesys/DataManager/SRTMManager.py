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
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



from .TileManager import TileManager
from iscesys.Component.Component import Component
import numpy as np
EXTRA = Component.Parameter('_extra',
    public_name = 'extra',default = '',
    type = str,
    mandatory = False,
    doc = 'String to append to default name such as .SRTMGL1(3) for dem')
DATA_EXT = Component.Parameter('_dataExt',
    public_name = 'dataExt',default = '',
    type = str,
    mandatory = False,
    doc = 'Extension of the data such as .hgt')
ARCHIVE_EXT  = Component.Parameter('_archiveExt',
    public_name = 'archiveExt',default = '.zip',
    type = str,
    mandatory = False,
    doc = 'Extension of the compressed data')
##Base class to handle product such as dem or water mask
class SRTMManager(TileManager):
    family = 'srtmmanager'
    parameter_list = (
                       EXTRA,
                       DATA_EXT,
                       ARCHIVE_EXT
                       ) + TileManager.parameter_list
    
    def __init__(self,family = '', name = ''):
        super(SRTMManager, self).__init__(family if family else  self.__class__.family, name=name)

  
    def convertCoordinateToString(self,lat,lon):

        if(lon < 0):
            ew = 'W'
        else:
            ew = 'E'
        lonAbs = int(np.fabs(lon))
        if(lonAbs >= 100):
            ew += str(lonAbs)
        elif(lonAbs < 10):
            ew +=  '00' + str(lonAbs)
        else:
            ew +=  '0' + str(lonAbs)

        if(int(lat) >= 0):
            ns = 'N'
        else:
            ns = 'S'
        latAbs = int(np.fabs(lat))
        if(latAbs >= 10):
            ns += str(latAbs)
        else:
            ns += '0' +str(latAbs)

        return ns,ew
    
    
    def createFilename(self,lat,lon):
        ns,ew = self.convertCoordinateToString(lat,lon)
        #when using local the files no need to be unzipped
        if self._useLocal:
            return ns + ew + self._dataExt
        else:
            return ns + ew + self._extra +  self._dataExt +  self._archiveExt
    
    @property
    def extra(self):
        return self._extra
    
    @extra.setter
    def extra(self,val):
        self._extra = val
    
    @property
    def dataExt(self):
        return self._dataExt
    @dataExt.setter
    def dataExt(self,val):
        self._dataExt = val
    
    @property
    def archiveExt(self):
        return self._archiveExt
    
    @archiveExt.setter
    def archiveExt(self,val):
        self._archiveExt = val
   
