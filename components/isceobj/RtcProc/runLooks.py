#
# Author: Piyush Agram
# Copyright 2016
#

import logging
import isceobj
import mroipac
import os
import numpy as np
from isceobj.Util.decorators import use_api

logger = logging.getLogger('isce.grdsar.looks')



def takeLooks(inimg, alks, rlks):
    '''
    Take looks.
    '''

    from mroipac.looks.Looks import Looks

    img = isceobj.createImage()
    img.load(inimg + '.xml')
    img.setAccessMode('READ')

    spl = os.path.splitext(inimg)
    ext = '.{0}alks_{1}rlks'.format(alks, rlks)
    outfile = spl[0] + ext + spl[1]


    lkObj = Looks()
    lkObj.setDownLooks(alks)
    lkObj.setAcrossLooks(rlks)
    lkObj.setInputImage(img)
    lkObj.setOutputFilename(outfile)
    lkObj.looks()

    return outfile


def runLooks(self):
    '''
    Make sure that a DEM is available for processing the given data.
    '''
   
    refPol = self._grd.polarizations[0]
    reference = self._grd.loadProduct( os.path.join(self._grd.outputFolder, 'beta_{0}.xml'.format(refPol)))


    azlooks, rglooks = self._grd.getLooks( self.posting, reference.groundRangePixelSize, reference.azimuthPixelSize, self.numberAzimuthLooks, self.numberRangeLooks)


    if (azlooks == 1) and (rglooks == 1):
        return

    slantRange = False
    for pol in self._grd.polarizations:
        inname = os.path.join( self._grd.outputFolder, 'beta_{0}.img'.format(pol) )
        takeLooks(inname, azlooks, rglooks)

        if not slantRange:
            inname = reference.slantRangeImage.filename
            takeLooks(inname, azlooks, rglooks)
            slantRange = True

    return
