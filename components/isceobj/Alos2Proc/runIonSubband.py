#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging

import isceobj
from isceobj.Constants import SPEED_OF_LIGHT

logger = logging.getLogger('isce.alos2insar.runIonSubband')

def runIonSubband(self):
    '''create subband interferograms
    '''
    if hasattr(self, 'doInSAR'):
        if not self.doInSAR:
            return

    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    if not self.doIon:
        catalog.printToLog(logger, "runIonSubband")
        self._insar.procDoc.addAllFromCatalog(catalog)
        return

    referenceTrack = self._insar.loadTrack(reference=True)
    secondaryTrack = self._insar.loadTrack(reference=False)

    #using 1/3, 1/3, 1/3 band split
    radarWavelength = referenceTrack.radarWavelength
    rangeBandwidth = referenceTrack.frames[0].swaths[0].rangeBandwidth
    rangeSamplingRate = referenceTrack.frames[0].swaths[0].rangeSamplingRate
    radarWavelengthLower = SPEED_OF_LIGHT/(SPEED_OF_LIGHT / radarWavelength - rangeBandwidth / 3.0)
    radarWavelengthUpper = SPEED_OF_LIGHT/(SPEED_OF_LIGHT / radarWavelength + rangeBandwidth / 3.0)
    subbandRadarWavelength = [radarWavelengthLower, radarWavelengthUpper]
    subbandBandWidth = [rangeBandwidth / 3.0 / rangeSamplingRate, rangeBandwidth / 3.0 / rangeSamplingRate]
    subbandFrequencyCenter = [-rangeBandwidth / 3.0 / rangeSamplingRate, rangeBandwidth / 3.0 / rangeSamplingRate]

    subbandPrefix = ['lower', 'upper']

    '''
    ionDir = {
        ionDir['swathMosaic'] : 'mosaic',
        ionDir['insar'] : 'insar',
        ionDir['ion'] : 'ion',
        ionDir['subband'] : ['lower', 'upper'],
        ionDir['ionCal'] : 'ion_cal'
        }
    '''
    #define upper level directory names
    ionDir = defineIonDir()


    self._insar.subbandRadarWavelength = subbandRadarWavelength


    ############################################################
    # STEP 1. create directories
    ############################################################
    #create and enter 'ion' directory
    #after finishing each step, we are in this directory
    os.makedirs(ionDir['ion'], exist_ok=True)
    os.chdir(ionDir['ion'])

    #create insar processing directories
    for k in range(2):
        subbandDir = ionDir['subband'][k]
        for i, frameNumber in enumerate(self._insar.referenceFrames):
            frameDir = 'f{}_{}'.format(i+1, frameNumber)
            for j, swathNumber in enumerate(range(self._insar.startingSwath, self._insar.endingSwath + 1)):
                swathDir = 's{}'.format(swathNumber)
                fullDir = os.path.join(subbandDir, frameDir, swathDir)
                os.makedirs(fullDir, exist_ok=True)

    #create ionospheric phase directory
    os.makedirs(ionDir['ionCal'], exist_ok=True)


    ############################################################
    # STEP 2. create subband interferograms
    ############################################################
    import numpy as np
    import stdproc
    from iscesys.StdOEL.StdOELPy import create_writer
    from isceobj.Alos2Proc.Alos2ProcPublic import readOffset
    from contrib.alos2proc.alos2proc import rg_filter

    for i, frameNumber in enumerate(self._insar.referenceFrames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        for j, swathNumber in enumerate(range(self._insar.startingSwath, self._insar.endingSwath + 1)):
            swathDir = 's{}'.format(swathNumber)

            #skip this time consuming process, if interferogram already exists
            if os.path.isfile(os.path.join(ionDir['subband'][0], frameDir, swathDir, self._insar.interferogram)) and \
               os.path.isfile(os.path.join(ionDir['subband'][0], frameDir, swathDir, self._insar.interferogram+'.vrt')) and \
               os.path.isfile(os.path.join(ionDir['subband'][0], frameDir, swathDir, self._insar.interferogram+'.xml')) and \
               os.path.isfile(os.path.join(ionDir['subband'][0], frameDir, swathDir, self._insar.amplitude)) and \
               os.path.isfile(os.path.join(ionDir['subband'][0], frameDir, swathDir, self._insar.amplitude+'.vrt')) and \
               os.path.isfile(os.path.join(ionDir['subband'][0], frameDir, swathDir, self._insar.amplitude+'.xml')) and \
               os.path.isfile(os.path.join(ionDir['subband'][1], frameDir, swathDir, self._insar.interferogram)) and \
               os.path.isfile(os.path.join(ionDir['subband'][1], frameDir, swathDir, self._insar.interferogram+'.vrt')) and \
               os.path.isfile(os.path.join(ionDir['subband'][1], frameDir, swathDir, self._insar.interferogram+'.xml')) and \
               os.path.isfile(os.path.join(ionDir['subband'][1], frameDir, swathDir, self._insar.amplitude)) and \
               os.path.isfile(os.path.join(ionDir['subband'][1], frameDir, swathDir, self._insar.amplitude+'.vrt')) and \
               os.path.isfile(os.path.join(ionDir['subband'][1], frameDir, swathDir, self._insar.amplitude+'.xml')):
                print('interferogram already exists at swath {}, frame {}'.format(swathNumber, frameNumber))
                continue

            #filter reference and secondary images
            for slcx in [self._insar.referenceSlc, self._insar.secondarySlc]:
                slc = os.path.join('../', frameDir, swathDir, slcx)
                slcLower = os.path.join(ionDir['subband'][0], frameDir, swathDir, slcx)
                slcUpper = os.path.join(ionDir['subband'][1], frameDir, swathDir, slcx)
                rg_filter(slc, 2, 
                    [slcLower, slcUpper], 
                    subbandBandWidth, 
                    subbandFrequencyCenter, 
                    257, 2048, 0.1, 0, 0.0)
            #resample
            for k in range(2):
                os.chdir(os.path.join(ionDir['subband'][k], frameDir, swathDir))
                #recreate xml file to remove the file path
                #can also use fixImageXml.py?
                for x in [self._insar.referenceSlc, self._insar.secondarySlc]:
                    img = isceobj.createSlcImage()
                    img.load(x + '.xml')
                    img.setFilename(x)
                    img.extraFilename = x + '.vrt'
                    img.setAccessMode('READ')
                    img.renderHdr()

                #############################################
                #1. form interferogram
                #############################################
                referenceSwath = referenceTrack.frames[i].swaths[j]
                secondarySwath = secondaryTrack.frames[i].swaths[j]

                refinedOffsets = readOffset(os.path.join('../../../../', frameDir, swathDir, 'cull.off'))
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
                objResamp.setRadarWavelength(subbandRadarWavelength[k])
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
                #using memmap instead, which should be faster, since we only have a few pixels to change
                amp=np.memmap(self._insar.amplitude, dtype='complex64', mode='r+', shape=(intLength, intWidth))
                index = np.nonzero( (np.real(amp)==0) + (np.imag(amp)==0) )
                amp[index]=0

                #Deletion flushes memory changes to disk before removing the object:
                del amp

                #############################################
                #3. delete subband slcs
                #############################################
                os.remove(self._insar.referenceSlc)
                os.remove(self._insar.referenceSlc + '.vrt')
                os.remove(self._insar.referenceSlc + '.xml')
                os.remove(self._insar.secondarySlc)
                os.remove(self._insar.secondarySlc + '.vrt')
                os.remove(self._insar.secondarySlc + '.xml')

                os.chdir('../../../')


    ############################################################
    # STEP 3. mosaic swaths
    ############################################################
    from isceobj.Alos2Proc.runSwathMosaic import swathMosaic
    from isceobj.Alos2Proc.Alos2ProcPublic import create_xml

    for k in range(2):
        os.chdir(ionDir['subband'][k])
        for i, frameNumber in enumerate(self._insar.referenceFrames):
            frameDir = 'f{}_{}'.format(i+1, frameNumber)
            os.chdir(frameDir)

            mosaicDir = ionDir['swathMosaic']
            os.makedirs(mosaicDir, exist_ok=True)
            os.chdir(mosaicDir)

            if not (
                   ((self._insar.modeCombination == 21) or \
                    (self._insar.modeCombination == 22) or \
                    (self._insar.modeCombination == 31) or \
                    (self._insar.modeCombination == 32)) 
                   and
                   (self._insar.endingSwath-self._insar.startingSwath+1 > 1)
                   ):
                import shutil
                swathDir = 's{}'.format(referenceTrack.frames[i].swaths[0].swathNumber)
                
                # if not os.path.isfile(self._insar.interferogram):
                #     os.symlink(os.path.join('../', swathDir, self._insar.interferogram), self._insar.interferogram)
                # shutil.copy2(os.path.join('../', swathDir, self._insar.interferogram+'.vrt'), self._insar.interferogram+'.vrt')
                # shutil.copy2(os.path.join('../', swathDir, self._insar.interferogram+'.xml'), self._insar.interferogram+'.xml')
                # if not os.path.isfile(self._insar.amplitude):
                #     os.symlink(os.path.join('../', swathDir, self._insar.amplitude), self._insar.amplitude)
                # shutil.copy2(os.path.join('../', swathDir, self._insar.amplitude+'.vrt'), self._insar.amplitude+'.vrt')
                # shutil.copy2(os.path.join('../', swathDir, self._insar.amplitude+'.xml'), self._insar.amplitude+'.xml')

                os.rename(os.path.join('../', swathDir, self._insar.interferogram), self._insar.interferogram)
                os.rename(os.path.join('../', swathDir, self._insar.interferogram+'.vrt'), self._insar.interferogram+'.vrt')
                os.rename(os.path.join('../', swathDir, self._insar.interferogram+'.xml'), self._insar.interferogram+'.xml')
                os.rename(os.path.join('../', swathDir, self._insar.amplitude), self._insar.amplitude)
                os.rename(os.path.join('../', swathDir, self._insar.amplitude+'.vrt'), self._insar.amplitude+'.vrt')
                os.rename(os.path.join('../', swathDir, self._insar.amplitude+'.xml'), self._insar.amplitude+'.xml')

                #no need to update frame parameters here
                os.chdir('../')
                #no need to save parameter file here
                os.chdir('../')

                continue

            #choose offsets
            numberOfFrames = len(referenceTrack.frames)
            numberOfSwaths = len(referenceTrack.frames[i].swaths)
            if self.swathOffsetMatching:
                #no need to do this as the API support 2-d list
                #rangeOffsets = (np.array(self._insar.swathRangeOffsetMatchingReference)).reshape(numberOfFrames, numberOfSwaths)
                #azimuthOffsets = (np.array(self._insar.swathAzimuthOffsetMatchingReference)).reshape(numberOfFrames, numberOfSwaths)
                rangeOffsets = self._insar.swathRangeOffsetMatchingReference
                azimuthOffsets = self._insar.swathAzimuthOffsetMatchingReference

            else:
                #rangeOffsets = (np.array(self._insar.swathRangeOffsetGeometricalReference)).reshape(numberOfFrames, numberOfSwaths)
                #azimuthOffsets = (np.array(self._insar.swathAzimuthOffsetGeometricalReference)).reshape(numberOfFrames, numberOfSwaths)
                rangeOffsets = self._insar.swathRangeOffsetGeometricalReference
                azimuthOffsets = self._insar.swathAzimuthOffsetGeometricalReference

            rangeOffsets = rangeOffsets[i]
            azimuthOffsets = azimuthOffsets[i]

            #list of input files
            inputInterferograms = []
            inputAmplitudes = []
            #phaseDiff = [None]
            swathPhaseDiffIon = [self.swathPhaseDiffLowerIon, self.swathPhaseDiffUpperIon]
            phaseDiff = swathPhaseDiffIon[k]
            if swathPhaseDiffIon[k] is None:
                phaseDiff = None
            else:
                phaseDiff = swathPhaseDiffIon[k][i]
                phaseDiff.insert(0, None)

            for j, swathNumber in enumerate(range(self._insar.startingSwath, self._insar.endingSwath + 1)):
                swathDir = 's{}'.format(swathNumber)
                inputInterferograms.append(os.path.join('../', swathDir, self._insar.interferogram))
                inputAmplitudes.append(os.path.join('../', swathDir, self._insar.amplitude))

                if False:
                    #compute phase needed to be compensated using startingRange
                    if j >= 1:
                        #phaseDiffSwath1 = -4.0 * np.pi * (referenceTrack.frames[i].swaths[j-1].startingRange - secondaryTrack.frames[i].swaths[j-1].startingRange)/subbandRadarWavelength[k]
                        #phaseDiffSwath2 = -4.0 * np.pi * (referenceTrack.frames[i].swaths[j].startingRange - secondaryTrack.frames[i].swaths[j].startingRange)/subbandRadarWavelength[k]
                        phaseDiffSwath1 = +4.0 * np.pi * referenceTrack.frames[i].swaths[j-1].startingRange * (1.0/radarWavelength - 1.0/subbandRadarWavelength[k]) \
                                          -4.0 * np.pi * secondaryTrack.frames[i].swaths[j-1].startingRange * (1.0/radarWavelength - 1.0/subbandRadarWavelength[k])
                        phaseDiffSwath2 = +4.0 * np.pi * referenceTrack.frames[i].swaths[j].startingRange * (1.0/radarWavelength - 1.0/subbandRadarWavelength[k]) \
                                          -4.0 * np.pi * secondaryTrack.frames[i].swaths[j].startingRange * (1.0/radarWavelength - 1.0/subbandRadarWavelength[k])
                        if referenceTrack.frames[i].swaths[j-1].startingRange - secondaryTrack.frames[i].swaths[j-1].startingRange == \
                           referenceTrack.frames[i].swaths[j].startingRange - secondaryTrack.frames[i].swaths[j].startingRange:
                            #phaseDiff.append(phaseDiffSwath2 - phaseDiffSwath1)
                            #if reference and secondary versions are all before or after version 2.025 (starting range error < 0.5 m), 
                            #it should be OK to do the above.
                            #see results in neom where it meets the above requirement, but there is still phase diff
                            #to be less risky, we do not input values here
                            phaseDiff.append(None)
                        else:
                            phaseDiff.append(None)

            #note that frame parameters are updated after mosaicking, here no need to update parameters
            #mosaic amplitudes
            swathMosaic(referenceTrack.frames[i], inputAmplitudes, self._insar.amplitude, 
                rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, resamplingMethod=0)
            #mosaic interferograms
            #These are for ALOS-2, may need to change for ALOS-4!
            phaseDiffFixed = [0.0, 0.4754024578084084, 0.9509913179406437, 1.4261648478671614, 2.179664007520499, 2.6766909968024932, 3.130810857]

            if False:
                if (referenceTrack.frames[i].processingSoftwareVersion == '2.025' and secondaryTrack.frames[i].processingSoftwareVersion == '2.023') or \
                   (referenceTrack.frames[i].processingSoftwareVersion == '2.023' and secondaryTrack.frames[i].processingSoftwareVersion == '2.025'):
                    
                    #               changed value                number of samples to estimate new value            new values estimate area
                    ###########################################################################################################################
                    #  2.6766909968024932-->2.6581660335779866                    1808694                               d169-f2850, north CA
                    #  2.179664007520499 -->2.204125866652153                      131120                               d169-f2850, north CA
                    
                    phaseDiffFixed = [0.0, 0.4754024578084084, 0.9509913179406437, 1.4261648478671614, 2.204125866652153, 2.6581660335779866, 3.130810857]

            snapThreshold = 0.2

            #the above preparetions only applies to 'self._insar.modeCombination == 21'
            #looks like it also works for 31 (scansarNominalModes-stripmapModes)
            if self._insar.modeCombination != 21:
                phaseDiff = None
                phaseDiffFixed = None
                snapThreshold = None

            #whether snap for each swath
            if self.swathPhaseDiffSnapIon == None:
                snapSwath = [[True for jjj in range(numberOfSwaths-1)] for iii in range(numberOfFrames)]
            else:
                snapSwath = self.swathPhaseDiffSnapIon
                if len(snapSwath) != numberOfFrames:
                    raise Exception('please specify each frame for parameter: swath phase difference snap to fixed values')
                for iii in range(numberOfFrames):
                    if len(snapSwath[iii]) != (numberOfSwaths-1):
                       raise Exception('please specify correct number of swaths for parameter: swath phase difference snap to fixed values')

            (phaseDiffEst, phaseDiffUsed, phaseDiffSource, numberOfValidSamples) = swathMosaic(referenceTrack.frames[i], inputInterferograms, self._insar.interferogram, 
                rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, updateFrame=False, 
                phaseCompensation=True, phaseDiff=phaseDiff, phaseDiffFixed=phaseDiffFixed, snapThreshold=snapThreshold, snapSwath=snapSwath[i], pcRangeLooks=1, pcAzimuthLooks=4, 
                filt=False, resamplingMethod=1)

            #the first item is meaningless for all the following list, so only record the following items
            if phaseDiff == None:
                phaseDiff = [None for iii in range(self._insar.startingSwath, self._insar.endingSwath + 1)]
            catalog.addItem('frame {} {} band swath phase diff input'.format(frameNumber, ionDir['subband'][k]), phaseDiff[1:], 'runIonSubband')
            catalog.addItem('frame {} {} band swath phase diff estimated'.format(frameNumber, ionDir['subband'][k]), phaseDiffEst[1:], 'runIonSubband')
            catalog.addItem('frame {} {} band swath phase diff used'.format(frameNumber, ionDir['subband'][k]), phaseDiffUsed[1:], 'runIonSubband')
            catalog.addItem('frame {} {} band swath phase diff used source'.format(frameNumber, ionDir['subband'][k]), phaseDiffSource[1:], 'runIonSubband')
            catalog.addItem('frame {} {} band swath phase diff samples used'.format(frameNumber, ionDir['subband'][k]), numberOfValidSamples[1:], 'runIonSubband')
            #check if there is value around 3.130810857, which may not be stable
            phaseDiffUnstableExist = False
            for xxx in phaseDiffUsed:
                if abs(abs(xxx) - 3.130810857) < 0.2:
                    phaseDiffUnstableExist = True
            catalog.addItem('frame {} {} band swath phase diff unstable exists'.format(frameNumber, ionDir['subband'][k]), phaseDiffUnstableExist, 'runIonSubband')

            create_xml(self._insar.amplitude, referenceTrack.frames[i].numberOfSamples, referenceTrack.frames[i].numberOfLines, 'amp')
            create_xml(self._insar.interferogram, referenceTrack.frames[i].numberOfSamples, referenceTrack.frames[i].numberOfLines, 'int')

            #update secondary frame parameters here, here no need to update parameters
            os.chdir('../')
            #save parameter file, here no need to save parameter file
            os.chdir('../')
        os.chdir('../')


    ############################################################
    # STEP 4. mosaic frames
    ############################################################
    from isceobj.Alos2Proc.runFrameMosaic import frameMosaic
    from isceobj.Alos2Proc.Alos2ProcPublic import create_xml

    for k in range(2):
        os.chdir(ionDir['subband'][k])

        mosaicDir = ionDir['insar']
        os.makedirs(mosaicDir, exist_ok=True)
        os.chdir(mosaicDir)

        numberOfFrames = len(referenceTrack.frames)
        if numberOfFrames == 1:
            import shutil
            frameDir = os.path.join('f1_{}/mosaic'.format(self._insar.referenceFrames[0]))
            # if not os.path.isfile(self._insar.interferogram):
            #     os.symlink(os.path.join('../', frameDir, self._insar.interferogram), self._insar.interferogram)
            # #shutil.copy2() can overwrite
            # shutil.copy2(os.path.join('../', frameDir, self._insar.interferogram+'.vrt'), self._insar.interferogram+'.vrt')
            # shutil.copy2(os.path.join('../', frameDir, self._insar.interferogram+'.xml'), self._insar.interferogram+'.xml')
            # if not os.path.isfile(self._insar.amplitude):
            #     os.symlink(os.path.join('../', frameDir, self._insar.amplitude), self._insar.amplitude)
            # shutil.copy2(os.path.join('../', frameDir, self._insar.amplitude+'.vrt'), self._insar.amplitude+'.vrt')
            # shutil.copy2(os.path.join('../', frameDir, self._insar.amplitude+'.xml'), self._insar.amplitude+'.xml')

            os.rename(os.path.join('../', frameDir, self._insar.interferogram), self._insar.interferogram)
            os.rename(os.path.join('../', frameDir, self._insar.interferogram+'.vrt'), self._insar.interferogram+'.vrt')
            os.rename(os.path.join('../', frameDir, self._insar.interferogram+'.xml'), self._insar.interferogram+'.xml')
            os.rename(os.path.join('../', frameDir, self._insar.amplitude), self._insar.amplitude)
            os.rename(os.path.join('../', frameDir, self._insar.amplitude+'.vrt'), self._insar.amplitude+'.vrt')
            os.rename(os.path.join('../', frameDir, self._insar.amplitude+'.xml'), self._insar.amplitude+'.xml')

            #update track parameters, no need to update track parameters here

        else:
            #choose offsets
            if self.frameOffsetMatching:
                rangeOffsets = self._insar.frameRangeOffsetMatchingReference
                azimuthOffsets = self._insar.frameAzimuthOffsetMatchingReference
            else:
                rangeOffsets = self._insar.frameRangeOffsetGeometricalReference
                azimuthOffsets = self._insar.frameAzimuthOffsetGeometricalReference

            #list of input files
            inputInterferograms = []
            inputAmplitudes = []
            for i, frameNumber in enumerate(self._insar.referenceFrames):
                frameDir = 'f{}_{}'.format(i+1, frameNumber)
                inputInterferograms.append(os.path.join('../', frameDir, 'mosaic', self._insar.interferogram))
                inputAmplitudes.append(os.path.join('../', frameDir, 'mosaic', self._insar.amplitude))

            #note that track parameters are updated after mosaicking
            #mosaic amplitudes
            frameMosaic(referenceTrack, inputAmplitudes, self._insar.amplitude, 
                rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, 
                updateTrack=False, phaseCompensation=False, resamplingMethod=0)
            #mosaic interferograms
            (phaseDiffEst, phaseDiffUsed, phaseDiffSource, numberOfValidSamples) = frameMosaic(referenceTrack, inputInterferograms, self._insar.interferogram, 
                rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, 
                updateTrack=False, phaseCompensation=True, resamplingMethod=1)

            create_xml(self._insar.amplitude, referenceTrack.numberOfSamples, referenceTrack.numberOfLines, 'amp')
            create_xml(self._insar.interferogram, referenceTrack.numberOfSamples, referenceTrack.numberOfLines, 'int')

            catalog.addItem('{} band frame phase diff estimated'.format(ionDir['subband'][k]), phaseDiffEst[1:], 'runIonSubband')
            catalog.addItem('{} band frame phase diff used'.format(ionDir['subband'][k]), phaseDiffUsed[1:], 'runIonSubband')
            catalog.addItem('{} band frame phase diff used source'.format(ionDir['subband'][k]), phaseDiffSource[1:], 'runIonSubband')
            catalog.addItem('{} band frame phase diff samples used'.format(ionDir['subband'][k]), numberOfValidSamples[1:], 'runIonSubband')

            #update secondary parameters here, no need to update secondary parameters here

        os.chdir('../')
        #save parameter file, no need to save parameter file here
        os.chdir('../')


    ############################################################
    # STEP 5. clear frame processing files
    ############################################################
    import shutil
    from isceobj.Alos2Proc.Alos2ProcPublic import runCmd

    for k in range(2):
        os.chdir(ionDir['subband'][k])
        for i, frameNumber in enumerate(self._insar.referenceFrames):
            frameDir = 'f{}_{}'.format(i+1, frameNumber)
            #keep subswath interferograms
            #shutil.rmtree(frameDir)
            #cmd = 'rm -rf {}'.format(frameDir)
            #runCmd(cmd)
        os.chdir('../')


    ############################################################
    # STEP 6. create differential interferograms
    ############################################################
    import numpy as np
    from isceobj.Alos2Proc.Alos2ProcPublic import runCmd

    for k in range(2):
        os.chdir(ionDir['subband'][k])

        insarDir = ionDir['insar']
        os.makedirs(insarDir, exist_ok=True)
        os.chdir(insarDir)

        rangePixelSize = self._insar.numberRangeLooks1 * referenceTrack.rangePixelSize
        radarWavelength = subbandRadarWavelength[k]
        rectRangeOffset = os.path.join('../../../', insarDir, self._insar.rectRangeOffset)

        cmd = "imageMath.py -e='a*exp(-1.0*J*b*4.0*{}*{}/{}) * (b!=0)' --a={} --b={} -o {} -t cfloat".format(np.pi, rangePixelSize, radarWavelength, self._insar.interferogram, rectRangeOffset, self._insar.differentialInterferogram)
        runCmd(cmd)

        os.chdir('../../')


    os.chdir('../')
    catalog.printToLog(logger, "runIonSubband")
    self._insar.procDoc.addAllFromCatalog(catalog)


def defineIonDir():
    '''
    define directory names for ionospheric correction
    '''

    ionDir = {
        #swath mosaicking directory
        'swathMosaic' : 'mosaic',
        #final insar processing directory
        'insar' : 'insar',
        #ionospheric correction directory
        'ion' : 'ion',
        #subband directory
        'subband' : ['lower', 'upper'],
        #final ionospheric phase calculation directory
        'ionCal' : 'ion_cal'
        }

    return ionDir


def defineIonFilenames():
    pass







