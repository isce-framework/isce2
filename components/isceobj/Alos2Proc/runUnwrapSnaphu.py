#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import shutil
import logging
import datetime
import numpy as np

import isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import snaphuUnwrap
from isceobj.Alos2Proc.Alos2ProcPublic import snaphuUnwrapOriginal
from isceobj.Alos2Proc.Alos2ProcPublic import runCmd

logger = logging.getLogger('isce.alos2insar.runUnwrapSnaphu')

def runUnwrapSnaphu(self):
    '''unwrap filtered interferogram
    '''
    if hasattr(self, 'doInSAR'):
        if not self.doInSAR:
            return

    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    referenceTrack = self._insar.loadTrack(reference=True)
    #secondaryTrack = self._insar.loadTrack(reference=False)

    unwrapSnaphu(self, referenceTrack)

    catalog.printToLog(logger, "runUnwrapSnaphu")
    self._insar.procDoc.addAllFromCatalog(catalog)


def unwrapSnaphu(self, referenceTrack):
    insarDir = 'insar'
    os.makedirs(insarDir, exist_ok=True)
    os.chdir(insarDir)


    ############################################################
    # STEP 1. unwrap interferogram
    ############################################################
    if shutil.which('snaphu') != None:
        print('\noriginal snaphu program found')
        print('unwrap {} using original snaphu, rather than that in ISCE'.format(self._insar.filteredInterferogram))
        snaphuUnwrapOriginal(self._insar.filteredInterferogram, 
            self._insar.multilookPhsig, 
            self._insar.multilookAmplitude, 
            self._insar.unwrappedInterferogram, 
            costMode = 's', 
            initMethod = 'mcf')
    else:
        tmid = referenceTrack.sensingStart + datetime.timedelta(seconds=(self._insar.numberAzimuthLooks1-1.0)/2.0*referenceTrack.azimuthLineInterval+
               referenceTrack.numberOfLines/2.0*self._insar.numberAzimuthLooks1*referenceTrack.azimuthLineInterval)
        snaphuUnwrap(referenceTrack, tmid, 
            self._insar.filteredInterferogram, 
            self._insar.multilookPhsig, 
            self._insar.unwrappedInterferogram, 
            self._insar.numberRangeLooks1*self._insar.numberRangeLooks2, 
            self._insar.numberAzimuthLooks1*self._insar.numberAzimuthLooks2, 
            costMode = 'SMOOTH',initMethod = 'MCF', defomax = 2, initOnly = True)


    ############################################################
    # STEP 2. mask using connected components
    ############################################################
    cmd = "imageMath.py -e='a_0*(b>0);a_1*(b>0)' --a={} --b={} -s BIL -t float -o={}".format(self._insar.unwrappedInterferogram, self._insar.unwrappedInterferogram+'.conncomp', self._insar.unwrappedMaskedInterferogram)
    runCmd(cmd)


    ############################################################
    # STEP 3. mask using water body
    ############################################################

    if self.waterBodyMaskStartingStep=='unwrap':
        wbdImage = isceobj.createImage()
        wbdImage.load(self._insar.multilookWbdOut+'.xml')
        width = wbdImage.width
        length = wbdImage.length
        #if not os.path.exists(self._insar.multilookWbdOut):
        #    catalog.addItem('warning message', 'requested masking interferogram with water body, but water body does not exist', 'runUnwrapSnaphu')
        #else:
        wbd = np.fromfile(self._insar.multilookWbdOut, dtype=np.int8).reshape(length, width)
        unw=np.memmap(self._insar.unwrappedInterferogram, dtype='float32', mode='r+', shape=(length*2, width))
        (unw[0:length*2:2, :])[np.nonzero(wbd==-1)]=0
        (unw[1:length*2:2, :])[np.nonzero(wbd==-1)]=0
        del unw
        unw=np.memmap(self._insar.unwrappedMaskedInterferogram, dtype='float32', mode='r+', shape=(length*2, width))
        (unw[0:length*2:2, :])[np.nonzero(wbd==-1)]=0
        (unw[1:length*2:2, :])[np.nonzero(wbd==-1)]=0
        del unw, wbd

    os.chdir('../')



