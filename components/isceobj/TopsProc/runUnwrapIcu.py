#
# Author: Piyush Agram
# Copyright 2016
#

import sys
import isce
from mroipac.icu.Icu import Icu
from iscesys.Component.Component import Component
from isceobj.Constants import SPEED_OF_LIGHT
import isceobj
import os
from isceobj.Util.decorators import use_api

# giangi: taken Piyush code grass.py and adapted
@use_api
def runUnwrap(self):
    '''Specific connector from an insarApp object to a Snaphu object.'''

    wrapName = os.path.join( self._insar.mergedDirname, self._insar.filtFilename)
    unwrapName = os.path.join( self._insar.mergedDirname, self._insar.unwrappedIntFilename)

    print(wrapName, unwrapName)
    #intImage
    intImage = isceobj.createImage()
    intImage.load(wrapName + '.xml')
    intImage.setAccessMode('READ')
    intImage.createImage()

    #unwImage
    unwImage = isceobj.Image.createUnwImage()
    unwImage.setFilename(unwrapName)
    unwImage.setWidth(intImage.getWidth())
    unwImage.imageType = 'unw'
    unwImage.bands = 2
    unwImage.scheme = 'BIL'
    unwImage.dataType = 'FLOAT'
    unwImage.setAccessMode('write')
    unwImage.createImage()

    icuObj = Icu(name='topsapp_icu')
    icuObj.configure()
    icuObj.useAmplitudeFlag = False
    icuObj.icu(intImage=intImage, unwImage = unwImage)

    #At least one can query for the name used
    self._insar.connectedComponentsFilename =  icuObj.conncompFilename
    intImage.finalizeImage()
    unwImage.finalizeImage()
    unwImage.renderHdr()

