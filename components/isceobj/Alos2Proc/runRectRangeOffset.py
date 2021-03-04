#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging

import isceobj
from contrib.alos2proc_f.alos2proc_f import rect_with_looks
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml

logger = logging.getLogger('isce.alos2insar.runRectRangeOffset')

def runRectRangeOffset(self):
    '''rectify range offset
    '''
    if hasattr(self, 'doInSAR'):
        if not self.doInSAR:
            return

    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    referenceTrack = self._insar.loadTrack(reference=True)
    secondaryTrack = self._insar.loadTrack(reference=False)

    insarDir = 'insar'
    os.makedirs(insarDir, exist_ok=True)
    os.chdir(insarDir)


    #rectify
    rgoff = isceobj.createImage()
    rgoff.load(self._insar.rangeOffset+'.xml')

    if self._insar.radarDemAffineTransform == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]:
        if not os.path.isfile(self._insar.rectRangeOffset):
            os.symlink(self._insar.rangeOffset, self._insar.rectRangeOffset)
            create_xml(self._insar.rectRangeOffset, rgoff.width, rgoff.length, 'float')
    else:
        rect_with_looks(self._insar.rangeOffset,
                        self._insar.rectRangeOffset,
                        rgoff.width, rgoff.length,
                        rgoff.width, rgoff.length,
                        self._insar.radarDemAffineTransform[0], self._insar.radarDemAffineTransform[1],
                        self._insar.radarDemAffineTransform[2], self._insar.radarDemAffineTransform[3],
                        self._insar.radarDemAffineTransform[4], self._insar.radarDemAffineTransform[5],
                        self._insar.numberRangeLooksSim*self._insar.numberRangeLooks1, self._insar.numberAzimuthLooksSim*self._insar.numberAzimuthLooks1,
                        self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1,
                        'REAL',
                        'Bilinear')
        create_xml(self._insar.rectRangeOffset, rgoff.width, rgoff.length, 'float')

    os.chdir('../')

    catalog.printToLog(logger, "runRectRangeOffset")
    self._insar.procDoc.addAllFromCatalog(catalog)


