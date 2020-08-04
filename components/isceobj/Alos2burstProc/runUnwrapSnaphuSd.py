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
from contrib.alos2proc.alos2proc import look
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml

logger = logging.getLogger('isce.alos2burstinsar.runUnwrapSnaphuSd')

def runUnwrapSnaphuSd(self):
    '''unwrap filtered interferogram
    '''
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    referenceTrack = self._insar.loadTrack(reference=True)
    #secondaryTrack = self._insar.loadTrack(reference=False)

    sdDir = 'sd'
    os.makedirs(sdDir, exist_ok=True)
    os.chdir(sdDir)


    ############################################################
    # STEP 1. unwrap interferogram
    ############################################################
    nsd = len(self._insar.filteredInterferogramSd)
    img = isceobj.createImage()
    img.load(self._insar.filteredInterferogramSd[0]+'.xml')
    width = img.width
    length = img.length

    if shutil.which('snaphu') != None:
        print('\noriginal snaphu program found, use it for unwrapping interferograms')
        useOriginalSnaphu = True
        #create an amplitude for use
        # amplitude = os.path.join('../insar', self._insar.amplitude)
        # amplitudeMultilook = 'tmp.amp'
        # img = isceobj.createImage()
        # img.load(amplitude+'.xml')
        # look(amplitude, amplitudeMultilook, img.width, self._insar.numberRangeLooksSd, self._insar.numberAzimuthLooksSd, 4, 1, 1)
    else:
        useOriginalSnaphu = False

    for sdCoherence, sdInterferogramFilt, sdInterferogramUnwrap in zip(self._insar.multilookCoherenceSd, self._insar.filteredInterferogramSd, self._insar.unwrappedInterferogramSd):
        if useOriginalSnaphu:
            amplitudeMultilook = 'tmp.amp'
            cmd = "imageMath.py -e='sqrt(abs(a));sqrt(abs(a))' --a={} -o {} -t float -s BSQ".format(sdInterferogramFilt, amplitudeMultilook)
            runCmd(cmd)
            snaphuUnwrapOriginal(sdInterferogramFilt, 
                sdCoherence, 
                amplitudeMultilook, 
                sdInterferogramUnwrap, 
                costMode = 's', 
                initMethod = 'mcf')
            os.remove(amplitudeMultilook)
            os.remove(amplitudeMultilook+'.vrt')
            os.remove(amplitudeMultilook+'.xml')
        else:
            tmid = referenceTrack.sensingStart + datetime.timedelta(seconds=(self._insar.numberAzimuthLooks1-1.0)/2.0*referenceTrack.azimuthLineInterval+
                   referenceTrack.numberOfLines/2.0*self._insar.numberAzimuthLooks1*referenceTrack.azimuthLineInterval)
            snaphuUnwrap(referenceTrack, tmid, 
                sdInterferogramFilt, 
                sdCoherence, 
                sdInterferogramUnwrap, 
                self._insar.numberRangeLooks1*self._insar.numberRangeLooksSd, 
                self._insar.numberAzimuthLooks1*self._insar.numberAzimuthLooksSd, 
                costMode = 'SMOOTH',initMethod = 'MCF', defomax = 2, initOnly = True)

    #if useOriginalSnaphu:
    #    os.remove(amplitudeMultilook)


    ############################################################
    # STEP 2. mask using connected components
    ############################################################
    for sdInterferogramUnwrap, sdInterferogramUnwrapMasked in zip(self._insar.unwrappedInterferogramSd, self._insar.unwrappedMaskedInterferogramSd):
        cmd = "imageMath.py -e='a_0*(b>0);a_1*(b>0)' --a={} --b={} -s BIL -t float -o={}".format(sdInterferogramUnwrap, sdInterferogramUnwrap+'.conncomp', sdInterferogramUnwrapMasked)
        runCmd(cmd)


    ############################################################
    # STEP 3. mask using water body
    ############################################################
    if self.waterBodyMaskStartingStepSd=='unwrap':
        wbd = np.fromfile(self._insar.multilookWbdOutSd, dtype=np.int8).reshape(length, width)

        for sdInterferogramUnwrap, sdInterferogramUnwrapMasked in zip(self._insar.unwrappedInterferogramSd, self._insar.unwrappedMaskedInterferogramSd):
            unw=np.memmap(sdInterferogramUnwrap, dtype='float32', mode='r+', shape=(length*2, width))
            (unw[0:length*2:2, :])[np.nonzero(wbd==-1)] = 0
            (unw[1:length*2:2, :])[np.nonzero(wbd==-1)] = 0
            unw=np.memmap(sdInterferogramUnwrapMasked, dtype='float32', mode='r+', shape=(length*2, width))
            (unw[0:length*2:2, :])[np.nonzero(wbd==-1)] = 0
            (unw[1:length*2:2, :])[np.nonzero(wbd==-1)] = 0


    ############################################################
    # STEP 4. convert to azimuth deformation
    ############################################################
    #burst cycle in s
    burstCycleLength = referenceTrack.frames[0].swaths[0].burstCycleLength / referenceTrack.frames[0].swaths[0].prf

    #compute azimuth fmrate
    #stack all azimuth fmrates
    index = np.array([], dtype=np.float64)
    ka = np.array([], dtype=np.float64)
    for frame in referenceTrack.frames:
        for swath in frame.swaths:
            startingRangeMultilook = referenceTrack.frames[0].swaths[0].startingRange + \
                                    (self._insar.numberRangeLooks1*self._insar.numberRangeLooksSd-1.0)/2.0*referenceTrack.frames[0].swaths[0].rangePixelSize
            rangePixelSizeMultilook = self._insar.numberRangeLooks1 * self._insar.numberRangeLooksSd * referenceTrack.frames[0].swaths[0].rangePixelSize
            index0 = (swath.startingRange + np.arange(swath.numberOfSamples) * swath.rangePixelSize - startingRangeMultilook) / rangePixelSizeMultilook
            ka0 = np.polyval(swath.azimuthFmrateVsPixel[::-1], np.arange(swath.numberOfSamples))
            index = np.concatenate((index, index0))
            ka = np.concatenate((ka, ka0))
    p = np.polyfit(index, ka, 3)
    #new ka
    ka = np.polyval(p, np.arange(width))

    #compute radar beam footprint velocity at middle track
    tmid = referenceTrack.sensingStart + datetime.timedelta(seconds=(self._insar.numberAzimuthLooks1-1.0)/2.0*referenceTrack.azimuthLineInterval+
           referenceTrack.numberOfLines/2.0*self._insar.numberAzimuthLooks1*referenceTrack.azimuthLineInterval)
    svmid = referenceTrack.orbit.interpolateOrbit(tmid, method='hermite')
    #earth radius in meters
    r = 6371 * 1000.0
    #radar footprint velocity
    veln = np.linalg.norm(svmid.getVelocity()) * r / np.linalg.norm(svmid.getPosition())
    print('radar beam footprint velocity at middle track: %8.2f m/s'%veln)

    #phase to defo factor
    factor = -1.0* veln / (2.0 * np.pi * ka * burstCycleLength)

    #process unwrapped without mask
    sdunw_out = np.zeros((length*2, width))
    flag = np.zeros((length, width))
    wgt = np.zeros((length, width))
    for i in range(nsd):
        sdunw = np.fromfile(self._insar.unwrappedInterferogramSd[i], dtype=np.float32).reshape(length*2, width)
        sdunw[1:length*2:2, :] *= factor[None, :] / (i+1.0)
        sdunw.astype(np.float32).tofile(self._insar.azimuthDeformationSd[i])
        create_xml(self._insar.azimuthDeformationSd[i], width, length, 'rmg')
        flag += (sdunw[1:length*2:2, :]!=0)
        #since the interferogram is filtered, we only use this light weight
        wgt0 = (i+1)**2
        wgt += wgt0 * (sdunw[1:length*2:2, :]!=0)
        sdunw_out[0:length*2:2, :] += (sdunw[0:length*2:2, :])**2
        sdunw_out[1:length*2:2, :] += wgt0 * sdunw[1:length*2:2, :]
    #output weighting average
    index = np.nonzero(flag!=0)
    (sdunw_out[0:length*2:2, :])[index] = np.sqrt((sdunw_out[0:length*2:2, :])[index] / flag[index])
    (sdunw_out[1:length*2:2, :])[index] = (sdunw_out[1:length*2:2, :])[index] / wgt[index]
    if not self.unionSd:
        (sdunw_out[0:length*2:2, :])[np.nonzero(flag<nsd)] = 0
        (sdunw_out[1:length*2:2, :])[np.nonzero(flag<nsd)] = 0
    sdunw_out.astype(np.float32).tofile(self._insar.azimuthDeformationSd[-1])
    create_xml(self._insar.azimuthDeformationSd[-1], width, length, 'rmg')

    #process unwrapped with mask
    sdunw_out = np.zeros((length*2, width))
    flag = np.zeros((length, width))
    wgt = np.zeros((length, width))
    for i in range(nsd):
        sdunw = np.fromfile(self._insar.unwrappedMaskedInterferogramSd[i], dtype=np.float32).reshape(length*2, width)
        sdunw[1:length*2:2, :] *= factor[None, :] / (i+1.0)
        sdunw.astype(np.float32).tofile(self._insar.maskedAzimuthDeformationSd[i])
        create_xml(self._insar.maskedAzimuthDeformationSd[i], width, length, 'rmg')
        flag += (sdunw[1:length*2:2, :]!=0)
        #since the interferogram is filtered, we only use this light weight
        wgt0 = (i+1)**2
        wgt += wgt0 * (sdunw[1:length*2:2, :]!=0)
        sdunw_out[0:length*2:2, :] += (sdunw[0:length*2:2, :])**2
        sdunw_out[1:length*2:2, :] += wgt0 * sdunw[1:length*2:2, :]
    #output weighting average
    index = np.nonzero(flag!=0)
    (sdunw_out[0:length*2:2, :])[index] = np.sqrt((sdunw_out[0:length*2:2, :])[index] / flag[index])
    (sdunw_out[1:length*2:2, :])[index] = (sdunw_out[1:length*2:2, :])[index] / wgt[index]
    if not self.unionSd:
        (sdunw_out[0:length*2:2, :])[np.nonzero(flag<nsd)] = 0
        (sdunw_out[1:length*2:2, :])[np.nonzero(flag<nsd)] = 0
    sdunw_out.astype(np.float32).tofile(self._insar.maskedAzimuthDeformationSd[-1])
    create_xml(self._insar.maskedAzimuthDeformationSd[-1], width, length, 'rmg')

    os.chdir('../')

    catalog.printToLog(logger, "runUnwrapSnaphuSd")
    self._insar.procDoc.addAllFromCatalog(catalog)

