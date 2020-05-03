#!/usr/bin/env python3
# Author: Piyush Agram
# Copyright 2016
#

import logging
import isceobj
import mroipac
from .runTopo import filenameWithLooks
import os
import numpy as np
from isceobj.Util.decorators import use_api
from applications import imageMath

logger = logging.getLogger('isce.grdsar.looks')



def runNormalize(self):
    '''
    Make sure that a DEM is available for processing the given data.
    '''
    refPol = self._grd.polarizations[0]
    master = self._grd.loadProduct( os.path.join(self._grd.outputFolder, 'beta_{0}.xml'.format(refPol)))


    azlooks, rglooks = self._grd.getLooks( self.posting, master.groundRangePixelSize, master.azimuthPixelSize, self.numberAzimuthLooks, self.numberRangeLooks)


    for pol in self._grd.polarizations:
        if (azlooks == 1) and (rglooks == 1):
            inname = os.path.join( self._grd.outputFolder, 'beta_{0}.img'.format(pol))
        else:
            inname = os.path.join( self._grd.outputFolder, filenameWithLooks('beta_{0}.img'.format(pol), azlooks, rglooks))

        incname = os.path.join(self._grd.geometryFolder, self._grd.incFileName)
        outname = os.path.join(self._grd.outputFolder, filenameWithLooks('gamma_{0}.img'.format(pol), azlooks, rglooks))
        maskname = os.path.join(self._grd.geometryFolder, self._grd.slMaskFileName)

        cmd = "imageMath.py --e='a*cos(b_0*PI/180.)/cos(b_1*PI/180.) * (c==0)' --a={beta} --b={inc} --c={mask} -o {out} -t FLOAT -s BIL"

        cmdrun = cmd.format(inc = incname,
                            beta = inname,
                            out = outname,
                            mask = maskname)
        status = os.system(cmdrun)
            
    return
