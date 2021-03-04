#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging
import shutil
import numpy as np

import isceobj
from mroipac.filter.Filter import Filter
from contrib.alos2filter.alos2filter import psfilt1
from mroipac.icu.Icu import Icu
from isceobj.Alos2Proc.Alos2ProcPublic import runCmd
from isceobj.Alos2Proc.Alos2ProcPublic import renameFile
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml

logger = logging.getLogger('isce.alos2insar.runFilt')

def runFilt(self):
    '''filter interferogram
    '''
    if hasattr(self, 'doInSAR'):
        if not self.doInSAR:
            return

    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    #referenceTrack = self._insar.loadTrack(reference=True)
    #secondaryTrack = self._insar.loadTrack(reference=False)

    filt(self)
    
    catalog.printToLog(logger, "runFilt")
    self._insar.procDoc.addAllFromCatalog(catalog)


def filt(self):

    insarDir = 'insar'
    os.makedirs(insarDir, exist_ok=True)
    os.chdir(insarDir)


    ############################################################
    # STEP 1. filter interferogram
    ############################################################
    print('\nfilter interferogram: {}'.format(self._insar.multilookDifferentialInterferogram))

    toBeFiltered = self._insar.multilookDifferentialInterferogram
    if self.removeMagnitudeBeforeFiltering:
        toBeFiltered = 'tmp.int'
        cmd = "imageMath.py -e='a/(abs(a)+(a==0))' --a={} -o {} -t cfloat -s BSQ".format(self._insar.multilookDifferentialInterferogram, toBeFiltered)
        runCmd(cmd)

    #if shutil.which('psfilt1') != None:
    if True:
        intImage = isceobj.createIntImage()
        intImage.load(toBeFiltered + '.xml')
        width = intImage.width
        length = intImage.length
        # cmd = "psfilt1 {int} {filtint} {width} {filterstrength} 64 16".format(
        #        int = toBeFiltered,
        #        filtint = self._insar.filteredInterferogram,
        #        width = width,
        #        filterstrength = self.filterStrength
        #        )
        # runCmd(cmd)
        windowSize = self.filterWinsize
        stepSize = self.filterStepsize
        psfilt1(toBeFiltered, self._insar.filteredInterferogram, width, self.filterStrength, windowSize, stepSize)
        create_xml(self._insar.filteredInterferogram, width, length, 'int')
    else:
        #original
        intImage = isceobj.createIntImage()
        intImage.load(toBeFiltered + '.xml')
        intImage.setAccessMode('read')
        intImage.createImage()
        width = intImage.width
        length = intImage.length

        #filtered
        filtImage = isceobj.createIntImage()
        filtImage.setFilename(self._insar.filteredInterferogram)
        filtImage.setWidth(width)
        filtImage.setAccessMode('write')
        filtImage.createImage()

        #looks like the ps filtering program keep the original interferogram magnitude, which is bad for phase unwrapping?
        filters = Filter()
        filters.wireInputPort(name='interferogram',object=intImage)
        filters.wireOutputPort(name='filtered interferogram',object=filtImage)
        filters.goldsteinWerner(alpha=self.filterStrength)
        intImage.finalizeImage()
        filtImage.finalizeImage()
        del intImage, filtImage, filters

    if self.removeMagnitudeBeforeFiltering:
        os.remove(toBeFiltered)
        os.remove(toBeFiltered + '.vrt')
        os.remove(toBeFiltered + '.xml')

    #restore original magnitude
    tmpFile = 'tmp.int'
    renameFile(self._insar.filteredInterferogram, tmpFile)
    cmd = "imageMath.py -e='a*abs(b)' --a={} --b={} -o {} -t cfloat -s BSQ".format(tmpFile, self._insar.multilookDifferentialInterferogram, self._insar.filteredInterferogram)
    runCmd(cmd)
    os.remove(tmpFile)
    os.remove(tmpFile + '.vrt')
    os.remove(tmpFile + '.xml')


    ############################################################
    # STEP 2. create phase sigma using filtered interferogram
    ############################################################
    print('\ncreate phase sigma using: {}'.format(self._insar.filteredInterferogram))

    #recreate filtered image
    filtImage = isceobj.createIntImage()
    filtImage.load(self._insar.filteredInterferogram + '.xml')
    filtImage.setAccessMode('read')
    filtImage.createImage()

    #amplitude image
    ampImage = isceobj.createAmpImage()
    ampImage.load(self._insar.multilookAmplitude + '.xml')
    ampImage.setAccessMode('read')
    ampImage.createImage()

    #phase sigma correlation image
    phsigImage = isceobj.createImage()
    phsigImage.setFilename(self._insar.multilookPhsig)
    phsigImage.setWidth(width)
    phsigImage.dataType='FLOAT'
    phsigImage.bands = 1
    phsigImage.setImageType('cor')
    phsigImage.setAccessMode('write')
    phsigImage.createImage()

    icu = Icu(name='insarapp_filter_icu')
    icu.configure()
    icu.unwrappingFlag = False
    icu.icu(intImage = filtImage, ampImage=ampImage, phsigImage=phsigImage)

    phsigImage.renderHdr()

    filtImage.finalizeImage()
    ampImage.finalizeImage()
    phsigImage.finalizeImage()

    del filtImage
    del ampImage
    del phsigImage
    del icu


    ############################################################
    # STEP 3. mask filtered interferogram using water body
    ############################################################
    print('\nmask filtered interferogram using: {}'.format(self._insar.multilookWbdOut))

    if self.waterBodyMaskStartingStep=='filt':
        #if not os.path.exists(self._insar.multilookWbdOut):
        #    catalog.addItem('warning message', 'requested masking interferogram with water body, but water body does not exist', 'runFilt')
        #else:
        wbd = np.fromfile(self._insar.multilookWbdOut, dtype=np.int8).reshape(length, width)
        phsig=np.memmap(self._insar.multilookPhsig, dtype='float32', mode='r+', shape=(length, width))
        phsig[np.nonzero(wbd==-1)]=0
        del phsig
        filt=np.memmap(self._insar.filteredInterferogram, dtype='complex64', mode='r+', shape=(length, width))
        filt[np.nonzero(wbd==-1)]=0
        del filt
        del wbd


    os.chdir('../')
