#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging
import numpy as np

import isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import runCmd

logger = logging.getLogger('isce.alos2insar.runCoherence')

def runCoherence(self):
    '''Extract images.
    '''
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    #referenceTrack = self._insar.loadTrack(reference=True)
    #secondaryTrack = self._insar.loadTrack(reference=False)

    insarDir = 'insar'
    os.makedirs(insarDir, exist_ok=True)
    os.chdir(insarDir)

    numberRangeLooks = self._insar.numberRangeLooks1 * self._insar.numberRangeLooks2
    numberAzimuthLooks = self._insar.numberAzimuthLooks1 * self._insar.numberAzimuthLooks2

    #here we choose not to scale interferogram and amplitude
    #scaleAmplitudeInterferogram

    #if (numberRangeLooks >= 5) and (numberAzimuthLooks >= 5):
    if (numberRangeLooks * numberAzimuthLooks >= 9):
        cmd = "imageMath.py -e='sqrt(b_0*b_1);abs(a)/(b_0+(b_0==0))/(b_1+(b_1==0))*(b_0!=0)*(b_1!=0)' --a={} --b={} -o {} -t float -s BIL".format(
            self._insar.multilookDifferentialInterferogram,
            self._insar.multilookAmplitude,
            self._insar.multilookCoherence)
        runCmd(cmd)
    else:
        #estimate coherence using a moving window
        coherence(self._insar.multilookAmplitude, self._insar.multilookDifferentialInterferogram, self._insar.multilookCoherence, 
            method="cchz_wave", windowSize=5)
    os.chdir('../')

    catalog.printToLog(logger, "runCoherence")
    self._insar.procDoc.addAllFromCatalog(catalog)


from isceobj.Util.decorators import use_api
@use_api
def coherence(amplitudeFile, interferogramFile, coherenceFile, method="cchz_wave", windowSize=5):
    ''' 
    compute coherence using a window
    '''
    import operator
    from mroipac.correlation.correlation import Correlation

    CORRELATION_METHOD = {
        'phase_gradient' : operator.methodcaller('calculateEffectiveCorrelation'),
        'cchz_wave' : operator.methodcaller('calculateCorrelation')
        }

    ampImage = isceobj.createAmpImage()
    ampImage.load(amplitudeFile + '.xml')
    ampImage.setAccessMode('read')
    ampImage.createImage()

    intImage = isceobj.createIntImage()
    intImage.load(interferogramFile + '.xml')
    intImage.setAccessMode('read')
    intImage.createImage()

    #there is no coherence image in the isceobj/Image
    cohImage = isceobj.createOffsetImage()
    cohImage.setFilename(coherenceFile)
    cohImage.setWidth(ampImage.width)
    cohImage.setAccessMode('write')
    cohImage.createImage()

    cor = Correlation()
    cor.configure()
    cor.wireInputPort(name='amplitude', object=ampImage)
    cor.wireInputPort(name='interferogram', object=intImage)
    cor.wireOutputPort(name='correlation', object=cohImage)
    
    cor.windowSize = windowSize

    cohImage.finalizeImage()
    intImage.finalizeImage()
    ampImage.finalizeImage()

    try:
        CORRELATION_METHOD[method](cor)
    except KeyError:
        print("Unrecognized correlation method")
        sys.exit(1)
        pass
    return None


def scaleAmplitudeInterferogram(amplitudeFile, interferogramFile, ratio=100000.0):
    '''
    scale amplitude and interferogram, and balace the two channels of amplitude image
    according to equation (2) in
    Howard A. Zebker and Katherine Chen, Accurate Estimation of Correlation in InSAR Observations
    IEEE GEOSCIENCE AND REMOTE SENSING LETTERS, VOL. 2, NO. 2, APRIL 2005.
    the operation of the program does not affect coherence estimation
    '''
    ampObj = isceobj.createImage()
    ampObj.load(amplitudeFile+'.xml')
    width = ampObj.width
    length = ampObj.length

    inf = np.fromfile(interferogramFile, dtype=np.complex64).reshape(length, width)
    amp = np.fromfile(amplitudeFile, dtype=np.complex64).reshape(length, width)

    flag = (inf!=0)*(amp.real!=0)*(amp.imag!=0)
    nvalid = np.sum(flag, dtype=np.float64)

    mpwr1 =  np.sqrt(np.sum(amp.real * amp.real * flag, dtype=np.float64) / nvalid)
    mpwr2 =  np.sqrt(np.sum(amp.imag * amp.imag * flag, dtype=np.float64) / nvalid)

    amp.real = amp.real / ratio
    amp.imag = amp.imag / ratio * mpwr1 / mpwr2
    inf = inf / ratio / ratio * mpwr1 / mpwr2

    amp.astype(np.complex64).tofile(inps.amp)
    inf.astype(np.complex64).tofile(inps.inf)
