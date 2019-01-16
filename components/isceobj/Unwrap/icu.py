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


class icu(Component):
    '''Specific connector from an insarApp object to a Snaphu object.'''
    def __init__(self, obj):

        basename = obj.insar.topophaseFlatFilename
        wrapName = basename
        unwrapName = basename.replace('.flat', '.unw')

        #Setup images
        self.ampImage = obj.insar.resampAmpImage.copy(access_mode='read')
        self.width = self.ampImage.getWidth()

        #intImage
        intImage = isceobj.createIntImage()
        intImage.initImage(wrapName, 'read', self.width)
        intImage.createImage()
        self.intImage = intImage

        #unwImage
        unwImage = isceobj.Image.createImage()
        unwImage.setFilename(unwrapName)
        unwImage.setWidth(self.width)
        unwImage.imageType = 'unw'
        unwImage.bands = 2
        unwImage.scheme = 'BIL'
        unwImage.dataType = 'FLOAT'
        unwImage.setAccessMode('write')
        unwImage.createImage()
        self.unwImage = unwImage


    def unwrap(self):
        icuObj = Icu()
        icuObj.filteringFlag = False      ##insarApp.py already filters it
        icuObj.initCorrThreshold = 0.1
        icuObj.icu(intImage=self.intImage, ampImage=self.ampImage, unwImage = self.unwImage)

        self.ampImage.finalizeImage()
        self.intImage.finalizeImage()
        self.unwImage.finalizeImage()
        self.unwImage.renderHdr()

