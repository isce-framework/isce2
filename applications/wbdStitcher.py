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




import isce
import logging
import logging.config
from iscesys.Component.Application import Application
from iscesys.Component.Component import Component
from contrib.demUtils.SWBDStitcher import SWBDStitcher

import os
STITCHER = Application.Facility(
    '_stitcher',
    public_name='wbd stitcher',
    module='contrib.demUtils',
    factory='createSWBDStitcher',
    args=('awbdstitcher',),
    mandatory=True,
    doc="Water body stitcher"
                              )
class Stitcher(Application):    
    def main(self):
        # prevent from deliting local files
        if(self._stitcher._useLocalDirectory):
            self._stitcher._keepAfterFailed = True
            self._stitcher._keepWbds = True
        # is a metadata file is created set the right type
        if(self._stitcher._meta == 'xml'):
            self._stitcher.setCreateXmlMetadata(True)
        
        # check for the action to be performed
        if(self._stitcher._action == 'stitch'):
            if(self._stitcher._bbox):
                lat = self._stitcher._bbox[0:2]
                lon = self._stitcher._bbox[2:4]
                if (self._stitcher._outputFile is None):
                    self._stitcher._outputFile = self._stitcher.defaultName(self._stitcher._bbox)
    
                if not(self._stitcher.stitchWbd(lat,lon,self._stitcher._outputFile,self._stitcher._downloadDir, \
                        keep=self._stitcher._keepWbds)):
                    print('Could not create a stitched water body mask. Some tiles are missing')
                
            else:
                print('Error. The "bbox" attribute must be specified when the action is "stitch"')
                raise ValueError
        elif(self._stitcher._action == 'download'):
            if(self._stitcher._bbox):
                lat = self._stitcher._bbox[0:2]
                lon = self._stitcher._bbox[2:4]
                self._stitcher.getWbdsInBox(lat,lon,self._stitcher._downloadDir)
          
        else:
            print('Unrecognized action ',self._stitcher._action)
            return
    
        if(self._stitcher._report):
            for k,v in list(self._stitcher._downloadReport.items()):
                print(k,'=',v)
                
    def Usage(self):
        print("\nUsage: wbdStitcher.py input.xml\n")
    
    facility_list = (STITCHER,)
    
    @property
    def stitcher(self):
        return self._stitcher
    @stitcher.setter
    def stitcher(self,stitcher):
        self._stitcher = stitcher
    
    family = 'wbdstitcher' 
    
    def __init__(self,family = '', name = ''):
        super(Stitcher, self).__init__(family if family else  self.__class__.family, name=name)
       

if __name__ == "__main__":
    import sys
    ds = Stitcher('wbdstitcher')
    ds.configure()
    ds.run()
