#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging
import numpy as np

import isceobj
import stdproc
from iscesys.StdOEL.StdOELPy import create_writer
from isceobj.Alos2Proc.Alos2ProcPublic import readOffset
from isceobj.Alos2Proc.Alos2ProcPublic import runCmd

logger = logging.getLogger('isce.alos2insar.runFormInterferogram')

def runFormInterferogram(self):
    '''form interferograms.
    '''
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    referenceTrack = self._insar.loadTrack(reference=True)
    secondaryTrack = self._insar.loadTrack(reference=False)

    for i, frameNumber in enumerate(self._insar.referenceFrames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        os.chdir(frameDir)
        for j, swathNumber in enumerate(range(self._insar.startingSwath, self._insar.endingSwath + 1)):
            swathDir = 's{}'.format(swathNumber)
            os.chdir(swathDir)

            print('forming interferogram frame {}, swath {}'.format(frameNumber, swathNumber))

            referenceSwath = referenceTrack.frames[i].swaths[j]
            secondarySwath = secondaryTrack.frames[i].swaths[j]


            #############################################
            #1. form interferogram
            #############################################
            refinedOffsets = readOffset('cull.off')
            intWidth = int(referenceSwath.numberOfSamples / self._insar.numberRangeLooks1)
            intLength = int(referenceSwath.numberOfLines / self._insar.numberAzimuthLooks1)
            dopplerVsPixel = [i/secondarySwath.prf for i in secondarySwath.dopplerVsPixel]

            #reference slc
            mSLC = isceobj.createSlcImage()
            mSLC.load(self._insar.referenceSlc+'.xml')
            mSLC.setAccessMode('read')
            mSLC.createImage()

            #secondary slc
            sSLC = isceobj.createSlcImage()
            sSLC.load(self._insar.secondarySlc+'.xml')
            sSLC.setAccessMode('read')
            sSLC.createImage()

            #interferogram
            interf = isceobj.createIntImage()
            interf.setFilename(self._insar.interferogram)
            interf.setWidth(intWidth)
            interf.setAccessMode('write')
            interf.createImage()

            #amplitdue
            amplitude = isceobj.createAmpImage()
            amplitude.setFilename(self._insar.amplitude)
            amplitude.setWidth(intWidth)
            amplitude.setAccessMode('write')
            amplitude.createImage()

            #create a writer for resamp
            stdWriter = create_writer("log", "", True, filename="resamp.log")
            stdWriter.setFileTag("resamp", "log")
            stdWriter.setFileTag("resamp", "err")
            stdWriter.setFileTag("resamp", "out")


            #set up resampling program now
            #The setting has been compared with resamp_roi's setting in ROI_pac item by item.
            #The two kinds of setting are exactly the same. The number of setting items are
            #exactly the same
            objResamp = stdproc.createResamp()
            objResamp.wireInputPort(name='offsets', object=refinedOffsets)
            objResamp.stdWriter = stdWriter
            objResamp.setNumberFitCoefficients(6)
            objResamp.setNumberRangeBin1(referenceSwath.numberOfSamples)
            objResamp.setNumberRangeBin2(secondarySwath.numberOfSamples)    
            objResamp.setStartLine(1)
            objResamp.setNumberLines(referenceSwath.numberOfLines)
            objResamp.setFirstLineOffset(1)
            objResamp.setDopplerCentroidCoefficients(dopplerVsPixel)
            objResamp.setRadarWavelength(secondaryTrack.radarWavelength)
            objResamp.setSlantRangePixelSpacing(secondarySwath.rangePixelSize)
            objResamp.setNumberRangeLooks(self._insar.numberRangeLooks1)
            objResamp.setNumberAzimuthLooks(self._insar.numberAzimuthLooks1)
            objResamp.setFlattenWithOffsetFitFlag(0)
            objResamp.resamp(mSLC, sSLC, interf, amplitude) 
            
            #finialize images
            mSLC.finalizeImage()
            sSLC.finalizeImage()
            interf.finalizeImage()
            amplitude.finalizeImage()
            stdWriter.finalize()


            #############################################
            #2. trim amplitude
            #############################################
            # tmpAmplitude = 'tmp.amp'
            # cmd = "imageMath.py -e='a_0*(a_1>0);a_1*(a_0>0)' --a={} -o={} -s BIP -t float".format(
            #     self._insar.amplitude, 
            #     tmpAmplitude
            #     )
            # runCmd(cmd)
            # os.remove(self._insar.amplitude)
            # os.remove(tmpAmplitude+'.xml')
            # os.remove(tmpAmplitude+'.vrt')
            # os.rename(tmpAmplitude, self._insar.amplitude)

            #using memmap instead, which should be faster, since we only have a few pixels to change
            amp=np.memmap(self._insar.amplitude, dtype='complex64', mode='r+', shape=(intLength, intWidth))
            index = np.nonzero( (np.real(amp)==0) + (np.imag(amp)==0) )
            amp[index]=0
            del amp

            os.chdir('../')
        os.chdir('../')

    catalog.printToLog(logger, "runFormInterferogram")
    self._insar.procDoc.addAllFromCatalog(catalog)
