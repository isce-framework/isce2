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

    masterTrack = self._insar.loadTrack(master=True)
    slaveTrack = self._insar.loadTrack(master=False)

    #using 1/3, 1/3, 1/3 band split
    radarWavelength = masterTrack.radarWavelength
    rangeBandwidth = masterTrack.frames[0].swaths[0].rangeBandwidth
    rangeSamplingRate = masterTrack.frames[0].swaths[0].rangeSamplingRate
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
        for i, frameNumber in enumerate(self._insar.masterFrames):
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

    for i, frameNumber in enumerate(self._insar.masterFrames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        for j, swathNumber in enumerate(range(self._insar.startingSwath, self._insar.endingSwath + 1)):
            swathDir = 's{}'.format(swathNumber)
            #filter master and slave images
            for burstPrefix, swath in zip([self._insar.masterBurstPrefix, self._insar.slaveBurstPrefix], 
                                   [masterTrack.frames[i].swaths[j], slaveTrack.frames[i].swaths[j]]):
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
                for burstPrefix, swath in zip([self._insar.masterBurstPrefix, self._insar.slaveBurstPrefix], 
                                       [masterTrack.frames[i].swaths[j], slaveTrack.frames[i].swaths[j]]):
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
                masterSwath = masterTrack.frames[i].swaths[j]
                slaveSwath = slaveTrack.frames[i].swaths[j]

                #set up resampling parameters
                width = masterSwath.numberOfSamples
                length = masterSwath.numberOfLines
                polyCoeff = self._insar.rangeResidualOffsetCc[i][j]
                rgIndex = (np.arange(width)-polyCoeff[-1][0])/polyCoeff[-1][1]
                azIndex = (np.arange(length)-polyCoeff[-1][2])/polyCoeff[-1][3]
                rangeOffset =  polyCoeff[0][0] + polyCoeff[0][1]*rgIndex[None,:] + polyCoeff[0][2]*rgIndex[None,:]**2 + \
                              (polyCoeff[1][0] + polyCoeff[1][1]*rgIndex[None,:]) * azIndex[:, None] + \
                               polyCoeff[2][0] * azIndex[:, None]**2
                azimuthOffset = self._insar.azimuthResidualOffsetCc[i][j]

                slaveBurstResampledDir = self._insar.slaveBurstPrefix + '_2_coreg_cc'
                interferogramDir = 'burst_interf_2_coreg_cc'
                interferogramPrefix = self._insar.masterBurstPrefix + '-' + self._insar.slaveBurstPrefix
                resampleBursts(masterSwath, slaveSwath, 
                    self._insar.masterBurstPrefix, self._insar.slaveBurstPrefix, slaveBurstResampledDir, interferogramDir,
                    self._insar.masterBurstPrefix, self._insar.slaveBurstPrefix, self._insar.slaveBurstPrefix, interferogramPrefix, 
                    os.path.join('../../../../{}/{}'.format(frameDir, swathDir), self._insar.rangeOffset), 
                    os.path.join('../../../../{}/{}'.format(frameDir, swathDir), self._insar.azimuthOffset), 
                    rangeOffsetResidual=rangeOffset, azimuthOffsetResidual=azimuthOffset)

                os.chdir(self._insar.masterBurstPrefix)
                mosaicBurstAmplitude(masterSwath, self._insar.masterBurstPrefix, self._insar.masterMagnitude, numberOfLooksThreshold=4)
                os.chdir('../')

                os.chdir(slaveBurstResampledDir)
                mosaicBurstAmplitude(masterSwath, self._insar.slaveBurstPrefix, self._insar.slaveMagnitude, numberOfLooksThreshold=4)
                os.chdir('../')

                os.chdir(interferogramDir)
                mosaicBurstInterferogram(masterSwath, interferogramPrefix, self._insar.interferogram, numberOfLooksThreshold=4)
                os.chdir('../')


                amp = np.zeros((masterSwath.numberOfLines, 2*masterSwath.numberOfSamples), dtype=np.float32)
                amp[0:, 1:masterSwath.numberOfSamples*2:2] = np.fromfile(os.path.join(slaveBurstResampledDir, self._insar.slaveMagnitude), \
                    dtype=np.float32).reshape(masterSwath.numberOfLines, masterSwath.numberOfSamples)
                amp[0:, 0:masterSwath.numberOfSamples*2:2] = np.fromfile(os.path.join(self._insar.masterBurstPrefix, self._insar.masterMagnitude), \
                    dtype=np.float32).reshape(masterSwath.numberOfLines, masterSwath.numberOfSamples)
                amp.astype(np.float32).tofile(self._insar.amplitude)
                create_xml(self._insar.amplitude, masterSwath.numberOfSamples, masterSwath.numberOfLines, 'amp')

                os.rename(os.path.join(interferogramDir, self._insar.interferogram), self._insar.interferogram)
                os.rename(os.path.join(interferogramDir, self._insar.interferogram+'.vrt'), self._insar.interferogram+'.vrt')
                os.rename(os.path.join(interferogramDir, self._insar.interferogram+'.xml'), self._insar.interferogram+'.xml')

                #############################################
                #2. delete subband slcs
                #############################################
                shutil.rmtree(self._insar.masterBurstPrefix)
                shutil.rmtree(self._insar.slaveBurstPrefix)
                shutil.rmtree(slaveBurstResampledDir)
                shutil.rmtree(interferogramDir)

                os.chdir('../../../')


    ############################################################
    # STEP 3. mosaic swaths
    ############################################################
    from isceobj.Alos2Proc.runSwathMosaic import swathMosaic
    from isceobj.Alos2Proc.Alos2ProcPublic import create_xml

    for k in range(2):
        os.chdir(ionDir['subband'][k])
        for i, frameNumber in enumerate(self._insar.masterFrames):
            frameDir = 'f{}_{}'.format(i+1, frameNumber)
            os.chdir(frameDir)

            mosaicDir = 'mosaic'
            os.makedirs(mosaicDir, exist_ok=True)
            os.chdir(mosaicDir)

            if self._insar.endingSwath-self._insar.startingSwath+1 == 1:
                import shutil
                swathDir = 's{}'.format(masterTrack.frames[i].swaths[0].swathNumber)
                
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
            numberOfFrames = len(masterTrack.frames)
            numberOfSwaths = len(masterTrack.frames[i].swaths)
            if self.swathOffsetMatching:
                #no need to do this as the API support 2-d list
                #rangeOffsets = (np.array(self._insar.swathRangeOffsetMatchingMaster)).reshape(numberOfFrames, numberOfSwaths)
                #azimuthOffsets = (np.array(self._insar.swathAzimuthOffsetMatchingMaster)).reshape(numberOfFrames, numberOfSwaths)
                rangeOffsets = self._insar.swathRangeOffsetMatchingMaster
                azimuthOffsets = self._insar.swathAzimuthOffsetMatchingMaster

            else:
                #rangeOffsets = (np.array(self._insar.swathRangeOffsetGeometricalMaster)).reshape(numberOfFrames, numberOfSwaths)
                #azimuthOffsets = (np.array(self._insar.swathAzimuthOffsetGeometricalMaster)).reshape(numberOfFrames, numberOfSwaths)
                rangeOffsets = self._insar.swathRangeOffsetGeometricalMaster
                azimuthOffsets = self._insar.swathAzimuthOffsetGeometricalMaster

            rangeOffsets = rangeOffsets[i]
            azimuthOffsets = azimuthOffsets[i]

            #list of input files
            inputInterferograms = []
            inputAmplitudes = []
            phaseDiff = [None]
            for j, swathNumber in enumerate(range(self._insar.startingSwath, self._insar.endingSwath + 1)):
                swathDir = 's{}'.format(swathNumber)
                inputInterferograms.append(os.path.join('../', swathDir, self._insar.interferogram))
                inputAmplitudes.append(os.path.join('../', swathDir, self._insar.amplitude))

                #compute phase needed to be compensated using startingRange
                if j >= 1:
                    #phaseDiffSwath1 = -4.0 * np.pi * (masterTrack.frames[i].swaths[j-1].startingRange - slaveTrack.frames[i].swaths[j-1].startingRange)/subbandRadarWavelength[k]
                    #phaseDiffSwath2 = -4.0 * np.pi * (masterTrack.frames[i].swaths[j].startingRange - slaveTrack.frames[i].swaths[j].startingRange)/subbandRadarWavelength[k]
                    phaseDiffSwath1 = +4.0 * np.pi * masterTrack.frames[i].swaths[j-1].startingRange * (1.0/radarWavelength - 1.0/subbandRadarWavelength[k]) \
                                      -4.0 * np.pi * slaveTrack.frames[i].swaths[j-1].startingRange * (1.0/radarWavelength - 1.0/subbandRadarWavelength[k])
                    phaseDiffSwath2 = +4.0 * np.pi * masterTrack.frames[i].swaths[j].startingRange * (1.0/radarWavelength - 1.0/subbandRadarWavelength[k]) \
                                      -4.0 * np.pi * slaveTrack.frames[i].swaths[j].startingRange * (1.0/radarWavelength - 1.0/subbandRadarWavelength[k])
                    if masterTrack.frames[i].swaths[j-1].startingRange - slaveTrack.frames[i].swaths[j-1].startingRange == \
                       masterTrack.frames[i].swaths[j].startingRange - slaveTrack.frames[i].swaths[j].startingRange:
                        #phaseDiff.append(phaseDiffSwath2 - phaseDiffSwath1)
                        #if master and slave versions are all before or after version 2.025 (starting range error < 0.5 m), 
                        #it should be OK to do the above.
                        #see results in neom where it meets the above requirement, but there is still phase diff
                        #to be less risky, we do not input values here
                        phaseDiff.append(None)
                    else:
                        phaseDiff.append(None)

            #note that frame parameters are updated after mosaicking
            #mosaic amplitudes
            swathMosaic(masterTrack.frames[i], inputAmplitudes, self._insar.amplitude, 
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

            (phaseDiffEst, phaseDiffUsed, phaseDiffSource) = swathMosaic(masterTrack.frames[i], inputInterferograms, self._insar.interferogram, 
                rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, updateFrame=False, 
                phaseCompensation=True, phaseDiff=phaseDiff, phaseDiffFixed=phaseDiffFixed, snapThreshold=snapThreshold, pcRangeLooks=1, pcAzimuthLooks=3, 
                filt=False, resamplingMethod=1)

            #the first item is meaningless for all the following list, so only record the following items
            if phaseDiff == None:
                phaseDiff = [None for iii in range(self._insar.startingSwath, self._insar.endingSwath + 1)]
            catalog.addItem('{} subswath phase difference input'.format(ionDir['subband'][k]), phaseDiff[1:], 'runIonSubband')
            catalog.addItem('{} subswath phase difference estimated'.format(ionDir['subband'][k]), phaseDiffEst[1:], 'runIonSubband')
            catalog.addItem('{} subswath phase difference used'.format(ionDir['subband'][k]), phaseDiffUsed[1:], 'runIonSubband')
            catalog.addItem('{} subswath phase difference used source'.format(ionDir['subband'][k]), phaseDiffSource[1:], 'runIonSubband')
            #check if there is value around 3.130810857, which may not be stable
            phaseDiffUnstableExist = False
            for xxx in phaseDiffUsed:
                if abs(abs(xxx) - 3.130810857) < 0.2:
                    phaseDiffUnstableExist = True
            catalog.addItem('{} subswath phase difference unstable exists'.format(ionDir['subband'][k]), phaseDiffUnstableExist, 'runIonSubband')

            create_xml(self._insar.amplitude, masterTrack.frames[i].numberOfSamples, masterTrack.frames[i].numberOfLines, 'amp')
            create_xml(self._insar.interferogram, masterTrack.frames[i].numberOfSamples, masterTrack.frames[i].numberOfLines, 'int')

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

        numberOfFrames = len(masterTrack.frames)
        if numberOfFrames == 1:
            import shutil
            frameDir = os.path.join('f1_{}/mosaic'.format(self._insar.masterFrames[0]))
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
                rangeOffsets = self._insar.frameRangeOffsetMatchingMaster
                azimuthOffsets = self._insar.frameAzimuthOffsetMatchingMaster
            else:
                rangeOffsets = self._insar.frameRangeOffsetGeometricalMaster
                azimuthOffsets = self._insar.frameAzimuthOffsetGeometricalMaster

            #list of input files
            inputInterferograms = []
            inputAmplitudes = []
            for i, frameNumber in enumerate(self._insar.masterFrames):
                frameDir = 'f{}_{}'.format(i+1, frameNumber)
                inputInterferograms.append(os.path.join('../', frameDir, 'mosaic', self._insar.interferogram))
                inputAmplitudes.append(os.path.join('../', frameDir, 'mosaic', self._insar.amplitude))

            #note that track parameters are updated after mosaicking
            #mosaic amplitudes
            frameMosaic(masterTrack, inputAmplitudes, self._insar.amplitude, 
                rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, 
                updateTrack=False, phaseCompensation=False, resamplingMethod=0)
            #mosaic interferograms
            frameMosaic(masterTrack, inputInterferograms, self._insar.interferogram, 
                rangeOffsets, azimuthOffsets, self._insar.numberRangeLooks1, self._insar.numberAzimuthLooks1, 
                updateTrack=False, phaseCompensation=True, resamplingMethod=1)

            create_xml(self._insar.amplitude, masterTrack.numberOfSamples, masterTrack.numberOfLines, 'amp')
            create_xml(self._insar.interferogram, masterTrack.numberOfSamples, masterTrack.numberOfLines, 'int')

        os.chdir('../')
        os.chdir('../')


    ############################################################
    # STEP 5. clear frame processing files
    ############################################################
    import shutil
    from isceobj.Alos2Proc.Alos2ProcPublic import runCmd

    for k in range(2):
        os.chdir(ionDir['subband'][k])
        for i, frameNumber in enumerate(self._insar.masterFrames):
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

        rangePixelSize = self._insar.numberRangeLooks1 * masterTrack.rangePixelSize
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







