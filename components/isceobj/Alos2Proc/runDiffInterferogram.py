#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging
import numpy as np

import isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import runCmd

logger = logging.getLogger('isce.alos2insar.runDiffInterferogram')

def runDiffInterferogram(self):
    '''Extract images.
    '''
    if hasattr(self, 'doInSAR'):
        if not self.doInSAR:
            return

    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    referenceTrack = self._insar.loadTrack(reference=True)

    insarDir = 'insar'
    os.makedirs(insarDir, exist_ok=True)
    os.chdir(insarDir)


    rangePixelSize = self._insar.numberRangeLooks1 * referenceTrack.rangePixelSize
    radarWavelength = referenceTrack.radarWavelength

    cmd = "imageMath.py -e='a*exp(-1.0*J*b*4.0*{}*{}/{}) * (b!=0)' --a={} --b={} -o {} -t cfloat".format(np.pi, rangePixelSize, radarWavelength, self._insar.interferogram, self._insar.rectRangeOffset, self._insar.differentialInterferogram)
    runCmd(cmd)


    os.chdir('../')

    catalog.printToLog(logger, "runDiffInterferogram")
    self._insar.procDoc.addAllFromCatalog(catalog)


