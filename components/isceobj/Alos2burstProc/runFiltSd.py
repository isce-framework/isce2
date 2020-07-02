#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import shutil
import logging
import numpy as np

import isceobj
from mroipac.filter.Filter import Filter
from mroipac.icu.Icu import Icu
from isceobj.Alos2Proc.Alos2ProcPublic import runCmd
from isceobj.Alos2Proc.Alos2ProcPublic import renameFile
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
from contrib.alos2filter.alos2filter import psfilt1
from isceobj.Alos2Proc.Alos2ProcPublic import cal_coherence

logger = logging.getLogger('isce.alos2burstinsar.runFiltSd')

def runFiltSd(self):
    '''filter interferogram
    '''
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    #referenceTrack = self._insar.loadTrack(reference=True)
    #secondaryTrack = self._insar.loadTrack(reference=False)

    sdDir = 'sd'
    os.makedirs(sdDir, exist_ok=True)
    os.chdir(sdDir)

    sd = isceobj.createImage()
    sd.load(self._insar.multilookInterferogramSd[0]+'.xml')
    width = sd.width
    length = sd.length

    ############################################################
    # STEP 1. filter interferogram
    ############################################################
    for sdInterferogram, sdInterferogramFilt, sdCoherence in zip(self._insar.multilookInterferogramSd, self._insar.filteredInterferogramSd, self._insar.multilookCoherenceSd):
        print('filter interferogram: {}'.format(sdInterferogram))
        #remove mangnitude
        data = np.fromfile(sdInterferogram, dtype=np.complex64).reshape(length, width)
        index = np.nonzero(data!=0)
        data[index] /= np.absolute(data[index])
        data.astype(np.complex64).tofile('tmp.int')

        #filter
        windowSize = self.filterWinsizeSd
        stepSize = self.filterStepsizeSd
        psfilt1('tmp.int', sdInterferogramFilt, width, self.filterStrengthSd, windowSize, stepSize)
        create_xml(sdInterferogramFilt, width, length, 'int')
        os.remove('tmp.int')

        #restore magnitude
        data = np.fromfile(sdInterferogram, dtype=np.complex64).reshape(length, width)
        dataFilt = np.fromfile(sdInterferogramFilt, dtype=np.complex64).reshape(length, width)
        index = np.nonzero(dataFilt!=0)
        dataFilt[index] = dataFilt[index] / np.absolute(dataFilt[index]) * np.absolute(data[index])
        dataFilt.astype(np.complex64).tofile(sdInterferogramFilt)

        # #create a coherence using an interferogram with most sparse fringes
        # if sdInterferogramFilt == self._insar.filteredInterferogramSd[0]:
        #     print('create coherence using: {}'.format(sdInterferogramFilt))
        #     cor = cal_coherence(dataFilt, win=3, edge=2)
        #     cor.astype(np.float32).tofile(self._insar.multilookCoherenceSd)
        #     create_xml(self._insar.multilookCoherenceSd, width, length, 'float')

        cor = cal_coherence(dataFilt, win=3, edge=2)
        cor.astype(np.float32).tofile(sdCoherence)
        create_xml(sdCoherence, width, length, 'float')


    ############################################################
    # STEP 3. mask filtered interferogram using water body
    ############################################################
    if self.waterBodyMaskStartingStepSd=='filt':
        print('mask filtered interferogram using: {}'.format(self._insar.multilookWbdOutSd))
        wbd = np.fromfile(self._insar.multilookWbdOutSd, dtype=np.int8).reshape(length, width)
        cor=np.memmap(self._insar.multilookCoherenceSd, dtype='float32', mode='r+', shape=(length, width))
        cor[np.nonzero(wbd==-1)]=0
        for sdInterferogramFilt in self._insar.filteredInterferogramSd:
            filt=np.memmap(sdInterferogramFilt, dtype='complex64', mode='r+', shape=(length, width))
            filt[np.nonzero(wbd==-1)]=0

    os.chdir('../')

    catalog.printToLog(logger, "runFiltSd")
    self._insar.procDoc.addAllFromCatalog(catalog)

