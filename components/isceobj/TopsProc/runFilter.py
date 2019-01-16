#
# Author: Piyush Agram
# Copyright 2016
#

import logging
import isceobj

from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from mroipac.filter.Filter import Filter
from mroipac.icu.Icu import Icu
import os

logger = logging.getLogger('isce.topsinsar.runFilter')

def runFilter(self):

    if not self.doInSAR:
        return

    logger.info("Applying power-spectral filter")

    mergedir = self._insar.mergedDirname
    filterStrength = self.filterStrength

    # Initialize the flattened interferogram
    inFilename = os.path.join(mergedir, self._insar.mergedIfgname)
    intImage = isceobj.createIntImage()
    intImage.load(inFilename + '.xml')
    intImage.setAccessMode('read')
    intImage.createImage()
    widthInt = intImage.getWidth()

    # Create the filtered interferogram
    filtIntFilename = os.path.join(mergedir, self._insar.filtFilename)
    filtImage = isceobj.createIntImage()
    filtImage.setFilename(filtIntFilename)
    filtImage.setWidth(widthInt)
    filtImage.setAccessMode('write')
    filtImage.createImage()

    objFilter = Filter()
    objFilter.wireInputPort(name='interferogram',object=intImage)
    objFilter.wireOutputPort(name='filtered interferogram',object=filtImage)

    objFilter.goldsteinWerner(alpha=filterStrength)

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
    phsigImage.setFilename(os.path.join(mergedir, self._insar.coherenceFilename))
    phsigImage.setAccessMode('write')
    phsigImage.setImageType('cor')#the type in this case is not for mdx.py displaying but for geocoding method
    phsigImage.createImage()


    icuObj = Icu(name='topsapp_filter_icu')
    icuObj.configure()
    icuObj.unwrappingFlag = False
    icuObj.useAmplitudeFlag = False

    icuObj.icu(intImage = filtImage, phsigImage=phsigImage)

    filtImage.finalizeImage()
    phsigImage.finalizeImage()
    phsigImage.renderHdr()

