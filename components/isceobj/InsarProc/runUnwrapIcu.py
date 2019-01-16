#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Piyush Agram
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




import sys
import isce
from mroipac.icu.Icu import Icu
from iscesys.Component.Component import Component
from isceobj.Constants import SPEED_OF_LIGHT
import isceobj

# giangi: taken Piyush code grass.py and adapted

def runUnwrap(self):
    '''Specific connector from an insarApp object to a Snaphu object.'''

    wrapName = self.insar.topophaseFlatFilename
    unwrapName = self.insar.unwrappedIntFilename

    #Setup images
    ampImage = self.insar.resampAmpImage.copy(access_mode='read')
    width = ampImage.getWidth()

    #intImage
    intImage = isceobj.createIntImage()
    intImage.initImage(wrapName, 'read', width)
    intImage.createImage()

    #unwImage
    unwImage = isceobj.Image.createUnwImage()
    unwImage.setFilename(unwrapName)
    unwImage.setWidth(width)
    unwImage.imageType = 'unw'
    unwImage.bands = 2
    unwImage.scheme = 'BIL'
    unwImage.dataType = 'FLOAT'
    unwImage.setAccessMode('write')
    unwImage.createImage()

    icuObj = Icu(name='insarapp_icu')
    icuObj.configure()
    icuObj.icu(intImage=intImage, ampImage=ampImage, unwImage = unwImage)
    #At least one can query for the name used
    self.insar.connectedComponentsFilename =  icuObj.conncompFilename
    ampImage.finalizeImage()
    intImage.finalizeImage()
    unwImage.finalizeImage()
    unwImage.renderHdr()

