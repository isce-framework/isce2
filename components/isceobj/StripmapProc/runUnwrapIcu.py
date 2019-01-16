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




# Heresh Fattahi, 2017
#   Generalized for full and sub-band interferograms


import sys
import isce
from mroipac.icu.Icu import Icu
from iscesys.Component.Component import Component
from isceobj.Constants import SPEED_OF_LIGHT
import isceobj
import os

# giangi: taken Piyush code grass.py and adapted

def runUnwrap(self , igramSpectrum = "full"):
    '''Specific connector from an insarApp object to a Snaphu object.'''

    if igramSpectrum == "full":
        ifgDirname = self.insar.ifgDirname

    elif igramSpectrum == "low":
        if not self.doDispersive:
            print('Estimating dispersive phase not requested ... skipping sub-band interferogram unwrapping')
            return
        ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.lowBandSlcDirname)

    elif igramSpectrum == "high":
        if not self.doDispersive:
            print('Estimating dispersive phase not requested ... skipping sub-band interferogram unwrapping')
            return
        ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.highBandSlcDirname)

    wrapName = os.path.join(ifgDirname , 'filt_' + self.insar.ifgFilename )

    if '.flat' in wrapName:
        unwrapName = wrapName.replace('.flat', '.unw')
    elif '.int' in wrapName:
        unwrapName = wrapName.replace('.int', '.unw')
    else:
        unwrapName = wrapName + '.unw'

    img1 = isceobj.createImage()
    img1.load(wrapName + '.xml')
    width = img1.getWidth()

    # Get amp image name
    originalWrapName = os.path.join(ifgDirname , self.insar.ifgFilename)
    if '.flat' in originalWrapName:
        resampAmpImage = originalWrapName.replace('.flat', '.amp')
    elif '.int' in originalWrapName:
        resampAmpImage = originalWrapName.replace('.int', '.amp')
    else:
        resampAmpImage = originalWrapName + '.amp'

    ampImage = isceobj.createAmpImage()
    ampImage.setWidth(width)
    ampImage.setFilename(resampAmpImage)
    ampImage.setAccessMode('read')
    ampImage.createImage()
    #width = ampImage.getWidth()

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
    icuObj.filteringFlag = False
    #icuObj.useAmplitudeFlag = False
    icuObj.singlePatch = True
    icuObj.initCorrThreshold = 0.1

    icuObj.icu(intImage=intImage, ampImage=ampImage, unwImage = unwImage)
    #At least one can query for the name used
    self.insar.connectedComponentsFilename =  icuObj.conncompFilename
    ampImage.finalizeImage()
    intImage.finalizeImage()
    unwImage.finalizeImage()
    unwImage.renderHdr()

