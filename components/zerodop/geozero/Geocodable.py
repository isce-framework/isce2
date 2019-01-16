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
import isce 
import sys 
import math
from iscesys.Component.Component import Component, Port
import os 


class Geocodable(Component):
    
    def __init__(self):
        super(Geocodable, self).__init__()
        self._image = None
        self._method = ''
        self._interp_map = {
                            'amp' : 'sinc',
                            'cpx' : 'sinc',
                            'cor' : 'nearest',
                            'unw' : 'nearest',
                            'rmg' : 'nearest'
                           }
    #there should be no need for a setter since this is a creator class
    @property
    def image(self):
        return self._image
    @property
    def method(self):
        return self._method
    def create(self,filename):
        from iscesys.Parsers.FileParserFactory import createFileParser
        from isceobj import createImage
        parser = createFileParser('xml')
        prop, fac, misc = parser.parse(filename + '.xml')

        self._image  = createImage()
        self._image.init(prop,fac,misc)
        self._image.accessMode = 'read'
        #try few ways. If the image type is not part of the map use sinc for complex and nearest for float 
        if self._image.imageType in self._interp_map:
            self._method = self._interp_map[self._image.imageType]
        elif self.image.dataType == 'CFLOAT':
            self._method = 'sinc'
        elif self.image.dataType == 'FLOAT':
            self._method = 'nearest'
        else:
            self._image = None
            self._method = None
        #allow to get image and method from the instance or as return value
        return self._image,self._method

def main(argv):
    ge = Geocodable()
    ge.create(argv[0])
    

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


    
