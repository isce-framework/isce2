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



def runNormalize(self):
    '''
    Make sure that a DEM is available for processing the given data.
    '''
   
    refPol = self._grd.polarizations[0]
    master = self._grd.loadProduct( os.path.join(self._grd.outputFolder, 'beta_{0}.xml'.format(refPol)))


    azlooks, rglooks = self._grd.getLooks( self.posting, master.groundRangePixelSize, master.azimuthPixelSize, self.numberAzimuthLooks, self.numberRangeLooks)


    if (azlooks == 1) and (rglooks == 1):
        return

    slantRange = False
    for pol in self._grd.polarizations:
        inname = os.path.join( self._grd.outputFolder, 'beta_{0}.img'.format(pol) )
        takeLooks(inname, azlooks, rglooks)

        if not slantRange:
            inname = master.slantRangeImage.filename
            takeLooks(inname, azlooks, rglooks)
            slantRange = True

    return
