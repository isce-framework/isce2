#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import shutil
import logging

import isceobj
from isceobj.Constants import SPEED_OF_LIGHT

logger = logging.getLogger('isce.alos2burstinsar.runIonSubband')

def runIonSubband(self):
    '''create subband interferograms
    '''
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
    os.makedirs(ionDir['ionCal'])


    ############################################################
    # STEP 2. create subband interferograms
    ############################################################
    import shutil
    import numpy as np
    from contrib.alos2proc.alos2proc import rg_filter
    from isceobj.Alos2Proc.Alos2ProcPublic import resampleBursts
    from isceobj.Alos2Proc.Alos2ProcPublic import mosaicBurstAmplitude
    from isceobj.Alos2Proc.Alos2ProcPublic import mosaicBurstInterferogram
    from isceobj.Alos2Proc.Alos2ProcPublic import create_xml

    for i, frameNumber in enumerate(self._insar.referenceFrames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        for j, swathNumber in enumerate(range(self._insar.startingSwath, self._insar.endingSwath + 1)):
            swathDir = 's{}'.format(swathNumber)
            #filter reference and secondary images
            for burstPrefix, swath in zip([self._insar.referenceBurstPrefix, self._insar.secondaryBurstPrefix], 
                                   [referenceTrack.frames[i].swaths[j], secondaryTrack.frames[i].swaths[j]]):
                slcDir = os.path.join('../', frameDir, swathDir, burstPrefix)
                slcLowerDir = os.path.join(ionDir['subband'][0], frameDir, swathDir, burstPrefix)
                slcUpperDir = os.path.join(ionDir['subband'][1], frameDir, swathDir, burstPrefix)
                os.makedirs(slcLowerDir, exist_ok=True)
                os.makedirs(slcUpperDir, exist_ok=True)
                for k in range(swath.numberOfBursts):
                    print('processing burst: %02d'%(k+1))
                    slc = os.path.join(slcDir, burstPrefix+'_%02d.slc'%(k+1))
                    slcLower = os.path.join(slcLowerDir, burstPrefix+'_%02d.slc'%(k+1))
                    slcUpper = os.path.join(slcUpperDir, burstPrefix+'_%02d.slc'%(k+1))
                    rg_filter(slc, 2, 
                        [slcLower, slcUpper], 
                        subbandBandWidth, 
                        subbandFrequencyCenter, 
                        257, 2048, 0.1, 0, 0.0)
            #resample
            for l in range(2):
                os.chdir(os.path.join(ionDir['subband'][l], frameDir, swathDir))
                #recreate xml file to remove the file path
                #can also use fixImageXml.py?
                for burstPrefix, swath in zip([self._insar.referenceBurstPrefix, self._insar.secondaryBurstPrefix], 
                                       [referenceTrack.frames[i].swaths[j], secondaryTrack.frames[i].swaths[j]]):
                    os.chdir(burstPrefix)
                    for k in range(swath.numberOfBursts):
                        slc = burstPrefix+'_%02d.slc'%(k+1)
                        img = isceobj.createSlcImage()
                        img.load(slc + '.xml')
                        img.setFilename(slc)
                        img.extraFilename = slc + '.vrt'
                        img.setAccessMode('READ')
                        img.renderHdr()
                    os.chdir('../')

                #############################################
                #1. form interferogram
                #############################################
                referenceSwath = referenceTrack.frames[i].swaths[j]
                secondarySwath = secondaryTrack.frames[i].swaths[j]

                #set up resampling parameters
                width = referenceSwath.numberOfSamples
                length = referenceSwath.numberOfLines
                polyCoeff = self._insar.rangeResidualOffsetCc[i][j]
                rgIndex = (np.arange(width)-polyCoeff[-1][0])/polyCoeff[-1][1]
                azIndex = (np.arange(length)-polyCoeff[-1][2])/polyCoeff[-1][3]
                rangeOffset =  polyCoeff[0][0] + polyCoeff[0][1]*rgIndex[None,:] + polyCoeff[0][2]*rgIndex[None,:]**2 + \
                              (polyCoeff[1][0] + polyCoeff[1][1]*rgIndex[None,:]) * azIndex[:, None] + \
                               polyCoeff[2][0] * azIndex[:, None]**2
                azimuthOffset = self._insar.azimuthResidualOffsetCc[i][j]

                secondaryBurstResampledDir = self._insar.secondaryBurstPrefix + '_2_coreg_cc'
                interferogramDir = 'burst_interf_2_coreg_cc'
                interferogramPrefix = self._insar.referenceBurstPrefix + '-' + self._insar.secondaryBurstPrefix
                resampleBursts(referenceSwath, secondarySwath, 
                    self._insar.referenceBurstPrefix, self._insar.secondaryBurstPrefix, secondaryBurstResampledDir, interferogramDir,
                    self._insar.referenceBurstPrefix, self._insar.secondaryBurstPrefix, self._insar.secondaryBurstPrefix, interferogramPrefix, 
                    os.path.join('../../../../{}/{}'.format(frameDir, swathDir), self._insar.rangeOffset), 
                    os.path.join('../../../../{}/{}'.format(frameDir, swathDir), self._insar.azimuthOffset), 
                    rangeOffsetResidual=rangeOffset, azimuthOffsetResidual=azimuthOffset)

                os.chdir(self._insar.referenceBurstPrefix)
                mosaicBurstAmplitude(referenceSwath, self._insar.referenceBurstPrefix, self._insar.referenceMagnitude, numberOfLooksThreshold=4)
                os.chdir('../')

                os.chdir(secondaryBurstResampledDir)
                mosaicBurstAmplitude(referenceSwath, self._insar.secondaryBurstPrefix, self._insar.secondaryMagnitude, numberOfLooksThreshold=4)
                os.chdir('../')

                os.chdir(interferogramDir)
                mosaicBurstInterferogram(referenceSwath, interferogramPrefix, self._insar.interferogram, numberOfLooksThreshold=4)
                os.chdir('../')


                amp = np.zeros((referenceSwath.numberOfLines, 2*referenceSwath.numberOfSamples), dtype=np.float32)
                amp[0:, 1:referenceSwath.numberOfSamples*2:2] = np.fromfile(os.path.join(secondaryBurstResampledDir, self._insar.secondaryMagnitude), \
                    dtype=np.float32).reshape(referenceSwath.numberOfLines, referenceSwath.numberOfSamples)
                amp[0:, 0:referenceSwath.numberOfSamples*2:2] = np.fromfile(os.path.join(self._insar.referenceBurstPrefix, self._insar.referenceMagnitude), \
                    dtype=np.float32).reshape(referenceSwath.numberOfLines, referenceSwath.numberOfSamples)
                amp.astype(np.float32).tofile(self._insar.amplitude)
                create_xml(self._insar.amplitude, referenceSwath.numberOfSamples, referenceSwath.numberOfLines, 'amp')

                os.rename(os.path.join(interferogramDir, self._insar.interferogram), self._insar.interferogram)
                os.rename(os.path.join(interferogramDir, self._insar.interferogram+'.vrt'), self._insar.interferogram+'.vrt')
                os.rename(os.path.join(interferogramDir, self._insar.interferogram+'.xml'), self._insar.interferogram+'.xml')

                #############################################
                #2. delete subband slcs
                #############################################
                shutil.rmtree(self._insar.referenceBurstPrefix)
                shutil.rmtree(self._insar.secondaryBurstPrefix)
                shutil.rmtree(secondaryBurstResampledDir)
                shutil.rmtree(interferogramDir)

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

            mosaicDir = 'mosaic'
            os.makedirs(mosaicDir, exist_ok=True)
            os.chdir(mosaicDir)

            if self._insar.endingSwath-self._insar.startingSwath+1 == 1:
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

                os.chdir('../')
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

                # #compute phase needed to be compensated using startingRange
                # if j >= 1:
                #     #phaseDiffSwath1 = -4.0 * np.pi * (referenceTrack.frames[i].swaths[j-1].startingRange - secondaryTrack.frames[i].swaths[j-1].startingRange)/subbandRadarWavelength[k]
                #     #phaseDiffSwath2 = -4.0 * np.pi * (referenceTrack.frames[i].swaths[j].startingRange - secondaryTrack.frames[i].swaths[j].startingRange)/subbandRadarWavelength[k]
                #     phaseDiffSwath1 = +4.0 * np.pi * referenceTrack.frames[i].swaths[j-1].startingRange * (1.0/radarWavelength - 1.0/subbandRadarWavelength[k]) \
                #                       -4.0 * np.pi * secondaryTrack.frames[i].swaths[j-1].startingRange * (1.0/radarWavelength - 1.0/subbandRadarWavelength[k])
                #     phaseDiffSwath2 = +4.0 * np.pi * referenceTrack.frames[i].swaths[j].startingRange * (1.0/radarWavelength - 1.0/subbandRadarWavelength[k]) \
                #                       -4.0 * np.pi * secondaryTrack.frames[i].swaths[j].startingRange * (1.0/radarWavelength - 1.0/subbandRadarWavelength[k])
                #     if referenceTrack.frames[i].swaths[j-1].startingRange - secondaryTrack.frames[i].swaths[j-1].startingRange == \
                #        referenceTrack.frames[i].swaths[j].startingRange - secondaryTrack.frames[i].swaths[j].startingRange:
                #         #phaseDiff.append(phaseDiffSwath2 - phaseDiffSwath1)
                #         #if reference and secondary versions are all before or after version 2.025 (starting range error < 0.5 m), 
                #         #it should be OK to do the above.
                #         #see results in neom where it meets the above requirement, but there is still phase diff
                #         #to be less risky, we do not input values here
                #         phaseDiff.append(None)
                #     else:
                #         phaseDiff.append(None)

            #note that frame parameters are updated after mosaicking
            #mosaic amplitudes
            swathMosaic(referenceTrack.frames[i], inputAmplitudes, self._insar.amplitude, 
                rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, resamplingMethod=0)
            #mosaic interferograms
            #These are for ALOS-2, may need to change for ALOS-4!
            phaseDiffFixed = [0.0, 0.4754024578084084, 0.9509913179406437, 1.4261648478671614, 2.179664007520499, 2.6766909968024932, 3.130810857]

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
                phaseCompensation=True, phaseDiff=phaseDiff, phaseDiffFixed=phaseDiffFixed, snapThreshold=snapThreshold, snapSwath=snapSwath[i], pcRangeLooks=1, pcAzimuthLooks=3, 
                filt=False, resamplingMethod=1)

            #the first item is meaningless for all the following list, so only record the following items
            if phaseDiff == None:
                phaseDiff = [None for iii in range(self._insar.startingSwath, self._insar.endingSwath + 1)]
            catalog.addItem('frame {} {} band subswath phase diff input'.format(frameNumber, ionDir['subband'][k]), phaseDiff[1:], 'runIonSubband')
            catalog.addItem('frame {} {} band subswath phase diff estimated'.format(frameNumber, ionDir['subband'][k]), phaseDiffEst[1:], 'runIonSubband')
            catalog.addItem('frame {} {} band subswath phase diff used'.format(frameNumber, ionDir['subband'][k]), phaseDiffUsed[1:], 'runIonSubband')
            catalog.addItem('frame {} {} band subswath phase diff used source'.format(frameNumber, ionDir['subband'][k]), phaseDiffSource[1:], 'runIonSubband')
            catalog.addItem('frame {} {} band subswath phase diff samples used'.format(frameNumber, ionDir['subband'][k]), numberOfValidSamples[1:], 'runIonSubband')
            #check if there is value around 3.130810857, which may not be stable
            phaseDiffUnstableExist = False
            for xxx in phaseDiffUsed:
                if abs(abs(xxx) - 3.130810857) < 0.2:
                    phaseDiffUnstableExist = True
            catalog.addItem('frame {} {} band subswath phase diff unstable exists'.format(frameNumber, ionDir['subband'][k]), phaseDiffUnstableExist, 'runIonSubband')

            create_xml(self._insar.amplitude, referenceTrack.frames[i].numberOfSamples, referenceTrack.frames[i].numberOfLines, 'amp')
            create_xml(self._insar.interferogram, referenceTrack.frames[i].numberOfSamples, referenceTrack.frames[i].numberOfLines, 'int')

            os.chdir('../')
            os.chdir('../')
        os.chdir('../')


    ############################################################
    # STEP 4. mosaic frames
    ############################################################
    from isceobj.Alos2Proc.runFrameMosaic import frameMosaic
    from isceobj.Alos2Proc.Alos2ProcPublic import create_xml

    for k in range(2):
        os.chdir(ionDir['subband'][k])

        mosaicDir = 'insar'
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

        os.chdir('../')
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
            shutil.rmtree(frameDir)
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







