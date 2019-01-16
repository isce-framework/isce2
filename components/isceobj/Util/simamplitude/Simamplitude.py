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
from iscesys.Component.Component import Component
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from isceobj.Util import simamplitude
from isceobj.Util.decorators import dov, pickled, logged
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
import isceobj

@pickled
class Simamplitude(Component):

    logging_name = 'isce.stdproc.simamplitude'
    
    dictionaryOfVariables = { 
        'WIDTH' : ['width', int, False],
        'LENGTH' : ['length', int,  False], 
        'SHADE_SCALE' : ['shadeScale', float, False] 
        }

    @dov
    @logged
    def __init__(self):
        super(Simamplitude, self).__init__()
        self.topoImage = None
        self.simampImage = None
        self.width = None
        self.length = None
        self.shadeScale = None
        return None

    def simamplitude(self,
                    topoImage,
                    simampImage,
                    shade=None,
                    width=None,
                    length=None):
        if shade  is not None: self.shadeScale = shade
        if width  is not None: self.width = width
        if length is not None: self.length = length
        self.topoImage = isceobj.createImage()
        IU.copyAttributes(topoImage, self.topoImage)
        self.topoImage.setCaster('read', 'FLOAT')
        self.topoImage.createImage()

        self.simampImage = simampImage
        topoAccessor = self.topoImage.getImagePointer()
        simampAccessor = self.simampImage.getImagePointer()
        self.setDefaults()
        self.setState()
        simamplitude.simamplitude_Py(topoAccessor, simampAccessor)
        return

    def setDefaults(self):
        if self.width is None: self.width = self.topoImage.getWidth() 
        if self.length is None:
            self.length = self.topoImage.getLength()
        if  self.shadeScale is None:
            self.shadeScale = 1
            self.logger.warning(
            'The shade scale factor has been set to the default value %s'%
            (self.shadeScale)
            )
            pass
        return

    def setState(self):
        simamplitude.setStdWriter_Py(int(self.stdWriter))
        simamplitude.setImageWidth_Py(int(self.width))
        simamplitude.setImageLength_Py(int(self.length))
        simamplitude.setShadeScale_Py(float(self.shadeScale))
        return

    def setImageWidth(self, var):
        self.width = int(var)
        return

    def setImageLength(self, var):
        self.length = int(var)
        return

    def setShadeScale(self, var):
        self.shadeScale = float(var)
        return
    pass
