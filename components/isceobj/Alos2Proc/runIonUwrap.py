#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging
import datetime
import numpy as np

import isceobj

logger = logging.getLogger('isce.alos2insar.runIonUwrap')

def runIonUwrap(self):
    '''unwrap subband interferograms
    '''
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    if not self.doIon:
        catalog.printToLog(logger, "runIonUwrap")
        self._insar.procDoc.addAllFromCatalog(catalog)
        return

    referenceTrack = self._insar.loadTrack(reference=True)
    secondaryTrack = self._insar.loadTrack(reference=False)
    wbdFile = os.path.abspath(self._insar.wbd)

    from isceobj.Alos2Proc.runIonSubband import defineIonDir
    ionDir = defineIonDir()
    subbandPrefix = ['lower', 'upper']

    ionCalDir = os.path.join(ionDir['ion'], ionDir['ionCal'])
    os.makedirs(ionCalDir, exist_ok=True)
    os.chdir(ionCalDir)


    ############################################################
    # STEP 1. take looks
    ############################################################
    from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
    from contrib.alos2proc.alos2proc import look
    from isceobj.Alos2Proc.Alos2ProcPublic import waterBodyRadar

    ml2 = '_{}rlks_{}alks'.format(self._insar.numberRangeLooks1*self._insar.numberRangeLooksIon, 
                              self._insar.numberAzimuthLooks1*self._insar.numberAzimuthLooksIon)

    for k in range(2):
        fullbandDir = os.path.join('../../', ionDir['insar'])
        subbandDir = os.path.join('../', ionDir['subband'][k], ionDir['insar'])
        prefix = subbandPrefix[k]

        amp = isceobj.createImage()
        amp.load(os.path.join(subbandDir, self._insar.amplitude)+'.xml')
        width = amp.width
        length = amp.length
        width2 = int(width / self._insar.numberRangeLooksIon)
        length2 = int(length / self._insar.numberAzimuthLooksIon)

        #take looks
        look(os.path.join(subbandDir, self._insar.differentialInterferogram), prefix+ml2+'.int', width, self._insar.numberRangeLooksIon, self._insar.numberAzimuthLooksIon, 4, 0, 1)
        create_xml(prefix+ml2+'.int', width2, length2, 'int')
        look(os.path.join(subbandDir, self._insar.amplitude), prefix+ml2+'.amp', width, self._insar.numberRangeLooksIon, self._insar.numberAzimuthLooksIon, 4, 1, 1)
        create_xml(prefix+ml2+'.amp', width2, length2, 'amp')

        # #water body
        # if k == 0:
        #     wbdOutFile = os.path.join(fullbandDir, self._insar.wbdOut)
        #     if os.path.isfile(wbdOutFile):
        #         look(wbdOutFile, 'wbd'+ml2+'.wbd', width, self._insar.numberRangeLooksIon, self._insar.numberAzimuthLooksIon, 0, 0, 1)
        #         create_xml('wbd'+ml2+'.wbd', width2, length2, 'byte')

        #water body
        if k == 0:
            look(os.path.join(fullbandDir, self._insar.latitude), 'lat'+ml2+'.lat', width, self._insar.numberRangeLooksIon, self._insar.numberAzimuthLooksIon, 3, 0, 1)
            look(os.path.join(fullbandDir, self._insar.longitude), 'lon'+ml2+'.lon', width, self._insar.numberRangeLooksIon, self._insar.numberAzimuthLooksIon, 3, 0, 1)
            create_xml('lat'+ml2+'.lat', width2, length2, 'double')
            create_xml('lon'+ml2+'.lon', width2, length2, 'double')
            waterBodyRadar('lat'+ml2+'.lat', 'lon'+ml2+'.lon', wbdFile, 'wbd'+ml2+'.wbd')


    ############################################################
    # STEP 2. compute coherence
    ############################################################
    from isceobj.Alos2Proc.Alos2ProcPublic import cal_coherence

    lowerbandInterferogramFile = subbandPrefix[0]+ml2+'.int'
    upperbandInterferogramFile = subbandPrefix[1]+ml2+'.int'
    lowerbandAmplitudeFile = subbandPrefix[0]+ml2+'.amp'
    upperbandAmplitudeFile = subbandPrefix[1]+ml2+'.amp'
    lowerbandCoherenceFile = subbandPrefix[0]+ml2+'.cor'
    upperbandCoherenceFile = subbandPrefix[1]+ml2+'.cor'
    coherenceFile = 'diff'+ml2+'.cor'

    lowerint = np.fromfile(lowerbandInterferogramFile, dtype=np.complex64).reshape(length2, width2)
    upperint = np.fromfile(upperbandInterferogramFile, dtype=np.complex64).reshape(length2, width2)
    loweramp = np.fromfile(lowerbandAmplitudeFile, dtype=np.float32).reshape(length2, width2*2)
    upperamp = np.fromfile(upperbandAmplitudeFile, dtype=np.float32).reshape(length2, width2*2)

    #compute coherence only using interferogram
    #here I use differential interferogram of lower and upper band interferograms
    #so that coherence is not affected by fringes
    cord = cal_coherence(lowerint*np.conjugate(upperint), win=3, edge=4)
    cor = np.zeros((length2*2, width2), dtype=np.float32)
    cor[0:length2*2:2, :] = np.sqrt( (np.absolute(lowerint)+np.absolute(upperint))/2.0 )
    cor[1:length2*2:2, :] = cord
    cor.astype(np.float32).tofile(coherenceFile)
    create_xml(coherenceFile, width2, length2, 'cor')

    #create lower and upper band coherence files
    #lower
    amp1 = loweramp[:, 0:width2*2:2]
    amp2 = loweramp[:, 1:width2*2:2]
    cor[1:length2*2:2, :] = np.absolute(lowerint)/(amp1+(amp1==0))/(amp2+(amp2==0))*(amp1!=0)*(amp2!=0)
    cor.astype(np.float32).tofile(lowerbandCoherenceFile)
    create_xml(lowerbandCoherenceFile, width2, length2, 'cor')

    #upper
    amp1 = upperamp[:, 0:width2*2:2]
    amp2 = upperamp[:, 1:width2*2:2]
    cor[1:length2*2:2, :] = np.absolute(upperint)/(amp1+(amp1==0))/(amp2+(amp2==0))*(amp1!=0)*(amp2!=0)
    cor.astype(np.float32).tofile(upperbandCoherenceFile)
    create_xml(upperbandCoherenceFile, width2, length2, 'cor')


    ############################################################
    # STEP 3. filtering subband interferograms
    ############################################################
    from contrib.alos2filter.alos2filter import psfilt1
    from isceobj.Alos2Proc.Alos2ProcPublic import runCmd
    from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
    from mroipac.icu.Icu import Icu

    if self.filterSubbandInt:
        for k in range(2):
            toBeFiltered = 'tmp.int'
            if self.removeMagnitudeBeforeFilteringSubbandInt:
                cmd = "imageMath.py -e='a/(abs(a)+(a==0))' --a={} -o {} -t cfloat -s BSQ".format(subbandPrefix[k]+ml2+'.int', toBeFiltered)
            else:
                #scale the inteferogram, otherwise its magnitude is too large for filtering
                cmd = "imageMath.py -e='a/100000.0' --a={} -o {} -t cfloat -s BSQ".format(subbandPrefix[k]+ml2+'.int', toBeFiltered)
            runCmd(cmd)

            intImage = isceobj.createIntImage()
            intImage.load(toBeFiltered + '.xml')
            width = intImage.width
            length = intImage.length

            windowSize = self.filterWinsizeSubbandInt
            stepSize = self.filterStepsizeSubbandInt
            psfilt1(toBeFiltered, 'filt_'+subbandPrefix[k]+ml2+'.int', width, self.filterStrengthSubbandInt, windowSize, stepSize)
            create_xml('filt_'+subbandPrefix[k]+ml2+'.int', width, length, 'int')

            os.remove(toBeFiltered)
            os.remove(toBeFiltered + '.vrt')
            os.remove(toBeFiltered + '.xml')

            #create phase sigma for phase unwrapping
            #recreate filtered image
            filtImage = isceobj.createIntImage()
            filtImage.load('filt_'+subbandPrefix[k]+ml2+'.int' + '.xml')
            filtImage.setAccessMode('read')
            filtImage.createImage()

            #amplitude image
            ampImage = isceobj.createAmpImage()
            ampImage.load(subbandPrefix[k]+ml2+'.amp' + '.xml')
            ampImage.setAccessMode('read')
            ampImage.createImage()

            #phase sigma correlation image
            phsigImage = isceobj.createImage()
            phsigImage.setFilename(subbandPrefix[k]+ml2+'.phsig')
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


    ############################################################
    # STEP 4. phase unwrapping
    ############################################################
    from isceobj.Alos2Proc.Alos2ProcPublic import snaphuUnwrap

    for k in range(2):
        tmid = referenceTrack.sensingStart + datetime.timedelta(seconds=(self._insar.numberAzimuthLooks1-1.0)/2.0*referenceTrack.azimuthLineInterval+
               referenceTrack.numberOfLines/2.0*self._insar.numberAzimuthLooks1*referenceTrack.azimuthLineInterval)

        if self.filterSubbandInt:
            toBeUnwrapped = 'filt_'+subbandPrefix[k]+ml2+'.int'
            coherenceFile = subbandPrefix[k]+ml2+'.phsig'
        else:
            toBeUnwrapped = subbandPrefix[k]+ml2+'.int'
            coherenceFile = 'diff'+ml2+'.cor'

        snaphuUnwrap(referenceTrack, tmid, 
            toBeUnwrapped, 
            coherenceFile, 
            subbandPrefix[k]+ml2+'.unw', 
            self._insar.numberRangeLooks1*self._insar.numberRangeLooksIon, 
            self._insar.numberAzimuthLooks1*self._insar.numberAzimuthLooksIon, 
            costMode = 'SMOOTH',initMethod = 'MCF', defomax = 2, initOnly = True)


    os.chdir('../../')
    catalog.printToLog(logger, "runIonUwrap")
    self._insar.procDoc.addAllFromCatalog(catalog)

