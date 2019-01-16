#
# Author: Piyush Agram
# Copyright 2016
#

import sys
import isceobj
from iscesys.Component.Component import Component
from mroipac.grass.grass import Grass
import os

# giangi: taken Piyush code grass.py and adapted

def runUnwrap(self):

    wrapName = os.path.join( self._insar.mergedDirname, self._insar.filtFilename)
    unwrapName = os.path.join( self._insar.mergedDirname, self._insar.unwrappedIntFilename)
    corName = os.path.join(self._insar.mergedDirname, self._insar.coherenceFilename)

    intImage = isceobj.createImage()
    intImage.load(wrapName + '.xml')
    intImage.setAccessMode('READ')


    cohImage = isceobj.createImage()
    cohImage.load(corName + '.xml')
    cohImage.setAccessMode('READ')


    unwImage = isceobj.createImage()
    unwImage.bands = 2
    unwImage.scheme = 'BIL'
    unwImage.dataType = 'FLOAT'
    unwImage.setFilename(unwrapName)
    unwImage.setWidth(intImage.getWidth())
    unwImage.setAccessMode('WRITE')
    

    grs=Grass(name='topsapp_grass')
    grs.configure()
    grs.wireInputPort(name='interferogram',
                    object=intImage)
    grs.wireInputPort(name='correlation',
                    object=cohImage)
    grs.wireInputPort(name='unwrapped interferogram',
                    object=unwImage)
    grs.unwrap()
    
    unwImage.renderHdr()

    return None
