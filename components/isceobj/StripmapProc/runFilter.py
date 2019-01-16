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
# Author: Brett George
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




# Heresh Fattahi: adopted for stripmapApp and generalized for full-band and sub-band interferograms 


import logging
import isceobj

from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from mroipac.filter.Filter import Filter
from mroipac.icu.Icu import Icu
import os

logger = logging.getLogger('isce.insar.runFilter')

def runFilter(self, filterStrength, igramSpectrum = "full"):
    logger.info("Applying power-spectral filter")

    if igramSpectrum == "full":
        logger.info("Filtering the full-band interferogram")
        ifgDirname = self.insar.ifgDirname

    elif igramSpectrum == "low":
        if not self.doDispersive:
            print('Estimating dispersive phase not requested ... skipping sub-band interferograms')
            return
        logger.info("Filtering the low-band interferogram")
        ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.lowBandSlcDirname)

    elif igramSpectrum == "high":
        if not self.doDispersive:
            print('Estimating dispersive phase not requested ... skipping sub-band interferograms')
            return
        logger.info("Filtering the high-band interferogram")
        ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.highBandSlcDirname)

    topoflatIntFilename = os.path.join(ifgDirname , self.insar.ifgFilename)

    img1 = isceobj.createImage()
    img1.load(topoflatIntFilename + '.xml')
    widthInt = img1.getWidth()

    intImage = isceobj.createIntImage()
    intImage.setFilename(topoflatIntFilename)
    intImage.setWidth(widthInt)
    intImage.setAccessMode('read')
    intImage.createImage()

    # Create the filtered interferogram
    filtIntFilename = os.path.join(ifgDirname , 'filt_' + self.insar.ifgFilename)
    filtImage = isceobj.createIntImage()
    filtImage.setFilename(filtIntFilename)
    filtImage.setWidth(widthInt)
    filtImage.setAccessMode('write')
    filtImage.createImage()
    
    objFilter = Filter()
    objFilter.wireInputPort(name='interferogram',object=intImage)
    objFilter.wireOutputPort(name='filtered interferogram',object=filtImage)
    if filterStrength is not None:
        self.insar.filterStrength = filterStrength

    objFilter.goldsteinWerner(alpha=self.insar.filterStrength)

    intImage.finalizeImage()
    filtImage.finalizeImage()
    del filtImage
    
    #Create phase sigma correlation file here
    filtImage = isceobj.createIntImage()
    filtImage.setFilename(filtIntFilename)
    filtImage.setWidth(widthInt)
    filtImage.setAccessMode('read')
    filtImage.createImage()


    phsigImage = isceobj.createImage()
    phsigImage.dataType='FLOAT'
    phsigImage.bands = 1
    phsigImage.setWidth(widthInt)
    phsigImage.setFilename(os.path.join(ifgDirname , self.insar.coherenceFilename))
    phsigImage.setAccessMode('write')
    phsigImage.setImageType('cor')#the type in this case is not for mdx.py displaying but for geocoding method
    phsigImage.createImage()

    resampAmpImage = os.path.join(ifgDirname , self.insar.ifgFilename)

    if '.flat' in resampAmpImage:
        resampAmpImage = resampAmpImage.replace('.flat', '.amp')
    elif '.int' in resampAmpImage:
        resampAmpImage = resampAmpImage.replace('.int', '.amp')
    else:
        resampAmpImage += '.amp'

    ampImage = isceobj.createAmpImage()
    ampImage.setWidth(widthInt)
    ampImage.setFilename(resampAmpImage)
    #IU.copyAttributes(self.insar.resampAmpImage, ampImage)
    #IU.copyAttributes(resampAmpImage, ampImage)
    ampImage.setAccessMode('read')
    ampImage.createImage()


    icuObj = Icu(name='stripmapapp_filter_icu')
    icuObj.configure()
    icuObj.unwrappingFlag = False

    icuObj.icu(intImage = filtImage, ampImage=ampImage, phsigImage=phsigImage)

    filtImage.finalizeImage()
    phsigImage.finalizeImage()
    phsigImage.renderHdr()
    ampImage.finalizeImage()



    # Set the filtered image to be the one geocoded
    # self.insar.topophaseFlatFilename = filtIntFilename
