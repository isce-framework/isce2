#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging
import statistics
import numpy as np
from scipy.ndimage.filters import median_filter

import isceobj

logger = logging.getLogger('isce.alos2insar.runFiltOffset')

def runFiltOffset(self):
    '''filt offset fied
    '''
    if not self.doDenseOffset:
        return
    if not ((self._insar.modeCombination == 0) or (self._insar.modeCombination == 1)):
        return

    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    denseOffsetDir = 'dense_offset'
    os.makedirs(denseOffsetDir, exist_ok=True)
    os.chdir(denseOffsetDir)

    #referenceTrack = self._insar.loadProduct(self._insar.referenceTrackParameter)
    #secondaryTrack = self._insar.loadProduct(self._insar.secondaryTrackParameter)

#########################################################################################

    if not self.doOffsetFiltering:
        print('offset field filtering is not requested.')
        os.chdir('../')
        catalog.printToLog(logger, "runFiltOffset")
        self._insar.procDoc.addAllFromCatalog(catalog)
        return

    windowSize = self.offsetFilterWindowsize
    nullValue = 0
    snrThreshold = self.offsetFilterSnrThreshold

    if windowSize < 3:
        raise Exception('dense offset field filter window size must >= 3')
    if windowSize % 2 != 1:
        windowSize += 1
        print('dense offset field filter window size is not odd, changed to: {}'.format(windowSize))

    print('\noffset filter parameters:')
    print('**************************************')
    print('filter window size: {}'.format(windowSize))
    print('filter null value: {}'.format(nullValue))
    print('filter snr threshold: {}\n'.format(snrThreshold))


    img = isceobj.createImage()
    img.load(self._insar.denseOffset+'.xml')
    width = img.width
    length = img.length

    offset = np.fromfile(self._insar.denseOffset, dtype=np.float32).reshape(length*2, width)
    snr = np.fromfile(self._insar.denseOffsetSnr, dtype=np.float32).reshape(length, width)
    offsetFilt = np.zeros((length*2, width), dtype=np.float32)

    edge = int((windowSize-1)/2+0.5)
    for k in range(2):
        print('filtering band {} of {}'.format(k+1, 2))
        band = offset[k:length*2:2, :]
        bandFilt = offsetFilt[k:length*2:2, :]
        for i in range(0+edge, length-edge):
            for j in range(0+edge, width-edge):
                bandSub = band[i-edge:i+edge+1, j-edge:j+edge+1]
                snrSub = snr[i-edge:i+edge+1, j-edge:j+edge+1]
                #bandSubUsed is 1-d numpy array
                bandSubUsed = bandSub[np.nonzero(np.logical_and(snrSub>snrThreshold, bandSub!=nullValue))]
                if bandSubUsed.size == 0:
                    bandFilt[i, j] = nullValue
                else:
                    bandFilt[i, j] = statistics.median(bandSubUsed)

    offsetFilt.astype(np.float32).tofile(self._insar.denseOffsetFilt)
    outImg = isceobj.createImage()
    outImg.setDataType('FLOAT')
    outImg.setFilename(self._insar.denseOffsetFilt)
    outImg.setBands(2)
    outImg.scheme = 'BIL'
    outImg.setWidth(width)
    outImg.setLength(length)
    outImg.addDescription('two-band pixel offset file. 1st band: range offset, 2nd band: azimuth offset')
    outImg.setAccessMode('read')
    outImg.renderHdr()

#########################################################################################

    os.chdir('../')
    catalog.printToLog(logger, "runFiltOffset")
    self._insar.procDoc.addAllFromCatalog(catalog)


