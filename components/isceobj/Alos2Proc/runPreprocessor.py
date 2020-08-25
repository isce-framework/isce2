#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import glob
import logging
import datetime
import numpy as np

import isceobj
import isceobj.Sensor.MultiMode as MultiMode
from isceobj.Planet.Planet import Planet
from isceobj.Alos2Proc.Alos2ProcPublic import runCmd
from isceobj.Alos2Proc.Alos2ProcPublic import getBboxRdr
from isceobj.Alos2Proc.Alos2ProcPublic import getBboxGeo

logger = logging.getLogger('isce.alos2insar.runPreprocessor')

def runPreprocessor(self):
    '''Extract images.
    '''
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)


    #find files
    #actually no need to use absolute path any longer, since we are able to find file from vrt now. 27-JAN-2020, CRL.
    #denseoffset may still need absolute path when making links
    self.referenceDir = os.path.abspath(self.referenceDir)
    self.secondaryDir = os.path.abspath(self.secondaryDir)

    ledFilesReference = sorted(glob.glob(os.path.join(self.referenceDir, 'LED-ALOS2*-*-*')))
    imgFilesReference = sorted(glob.glob(os.path.join(self.referenceDir, 'IMG-{}-ALOS2*-*-*'.format(self.referencePolarization.upper()))))

    ledFilesSecondary = sorted(glob.glob(os.path.join(self.secondaryDir, 'LED-ALOS2*-*-*')))
    imgFilesSecondary = sorted(glob.glob(os.path.join(self.secondaryDir, 'IMG-{}-ALOS2*-*-*'.format(self.secondaryPolarization.upper()))))

    firstFrameReference = ledFilesReference[0].split('-')[-3][-4:]
    firstFrameSecondary = ledFilesSecondary[0].split('-')[-3][-4:]
    firstFrameImagesReference = sorted(glob.glob(os.path.join(self.referenceDir, 'IMG-{}-ALOS2*{}-*-*'.format(self.referencePolarization.upper(), firstFrameReference))))
    firstFrameImagesSecondary = sorted(glob.glob(os.path.join(self.secondaryDir, 'IMG-{}-ALOS2*{}-*-*'.format(self.secondaryPolarization.upper(), firstFrameSecondary))))


    #determin operation mode
    referenceMode = os.path.basename(ledFilesReference[0]).split('-')[-1][0:3]
    secondaryMode = os.path.basename(ledFilesSecondary[0]).split('-')[-1][0:3]
    spotlightModes = ['SBS']
    stripmapModes = ['UBS', 'UBD', 'HBS', 'HBD', 'HBQ', 'FBS', 'FBD', 'FBQ']
    scansarNominalModes = ['WBS', 'WBD', 'WWS', 'WWD']
    scansarWideModes = ['VBS', 'VBD']
    scansarModes = ['WBS', 'WBD', 'WWS', 'WWD', 'VBS', 'VBD']

    #usable combinations
    if (referenceMode in spotlightModes) and (secondaryMode in spotlightModes):
        self._insar.modeCombination = 0
    elif (referenceMode in stripmapModes) and (secondaryMode in stripmapModes):
        self._insar.modeCombination = 1
    elif (referenceMode in scansarNominalModes) and (secondaryMode in scansarNominalModes):
        self._insar.modeCombination = 21
    elif (referenceMode in scansarWideModes) and (secondaryMode in scansarWideModes):
        self._insar.modeCombination = 22
    elif (referenceMode in scansarNominalModes) and (secondaryMode in stripmapModes):
        self._insar.modeCombination = 31
    elif (referenceMode in scansarWideModes) and (secondaryMode in stripmapModes):
        self._insar.modeCombination = 32
    else:
        print('\n\nthis mode combination is not possible')
        print('note that for ScanSAR-stripmap, ScanSAR must be reference\n\n')
        raise Exception('mode combination not supported')

# pixel size from real data processing. azimuth pixel size may change a bit as
# the antenna points to a different swath and therefore uses a different PRF.

#   MODE  RANGE PIXEL SIZE (LOOKS)       AZIMUTH PIXEL SIZE (LOOKS)
# -------------------------------------------------------------------
#   SPT    [SBS]
#          1.4304222392897463 (2)         0.9351804642158579 (4)
#   SM1    [UBS,UBD]
#          1.4304222392897463 (2)         1.8291988125114438 (2)
#   SM2    [HBS,HBD,HBQ]
#          2.8608444785794984 (2)         3.0672373839847196 (2)
#   SM3    [FBS,FBD,FBQ]
#          4.291266717869248  (2)         3.2462615913656667 (4)

#   WD1    [WBS,WBD] [WWS,WWD]
#          8.582533435738496  (1)         2.6053935830031887 (14)
#          8.582533435738496  (1)         2.092362043327227  (14)
#          8.582533435738496  (1)         2.8817632034495717 (14)
#          8.582533435738496  (1)         3.054362492601842  (14)
#          8.582533435738496  (1)         2.4582084463356977 (14)

#   WD2    [VBS,VBD]
#          8.582533435738496  (1)         2.9215796012950728 (14)
#          8.582533435738496  (1)         3.088859074497863  (14)
#          8.582533435738496  (1)         2.8792293071133073 (14)
#          8.582533435738496  (1)         3.0592146044234854 (14)
#          8.582533435738496  (1)         2.8818767752199137 (14)
#          8.582533435738496  (1)         3.047038521027477  (14)
#          8.582533435738496  (1)         2.898816222039108  (14)

    #determine default number of looks:
    self._insar.numberRangeLooks1 = self.numberRangeLooks1
    self._insar.numberAzimuthLooks1 = self.numberAzimuthLooks1
    self._insar.numberRangeLooks2 = self.numberRangeLooks2
    self._insar.numberAzimuthLooks2 = self.numberAzimuthLooks2
    #the following two will be automatically determined by runRdrDemOffset.py
    self._insar.numberRangeLooksSim = self.numberRangeLooksSim
    self._insar.numberAzimuthLooksSim = self.numberAzimuthLooksSim
    self._insar.numberRangeLooksIon = self.numberRangeLooksIon
    self._insar.numberAzimuthLooksIon = self.numberAzimuthLooksIon

    if self._insar.numberRangeLooks1 == None:
        if referenceMode in ['SBS']:
            self._insar.numberRangeLooks1 = 2
        elif referenceMode in ['UBS', 'UBD']:
            self._insar.numberRangeLooks1 = 2
        elif referenceMode in ['HBS', 'HBD', 'HBQ']:
            self._insar.numberRangeLooks1 = 2
        elif referenceMode in ['FBS', 'FBD', 'FBQ']:
            self._insar.numberRangeLooks1 = 2
        elif referenceMode in ['WBS', 'WBD']:
            self._insar.numberRangeLooks1 = 1
        elif referenceMode in ['WWS', 'WWD']:
            self._insar.numberRangeLooks1 = 2
        elif referenceMode in ['VBS', 'VBD']:
            self._insar.numberRangeLooks1 = 1
        else:
            raise Exception('unknow acquisition mode')

    if self._insar.numberAzimuthLooks1 == None:
        if referenceMode in ['SBS']:
            self._insar.numberAzimuthLooks1 = 4
        elif referenceMode in ['UBS', 'UBD']:
            self._insar.numberAzimuthLooks1 = 2
        elif referenceMode in ['HBS', 'HBD', 'HBQ']:
            self._insar.numberAzimuthLooks1 = 2
        elif referenceMode in ['FBS', 'FBD', 'FBQ']:
            self._insar.numberAzimuthLooks1 = 4
        elif referenceMode in ['WBS', 'WBD']:
            self._insar.numberAzimuthLooks1 = 14
        elif referenceMode in ['WWS', 'WWD']:
            self._insar.numberAzimuthLooks1 = 14
        elif referenceMode in ['VBS', 'VBD']:
            self._insar.numberAzimuthLooks1 = 14
        else:
            raise Exception('unknow acquisition mode')

    if self._insar.numberRangeLooks2 == None:
        if referenceMode in spotlightModes:
            self._insar.numberRangeLooks2 = 4
        elif referenceMode in stripmapModes:
            self._insar.numberRangeLooks2 = 4
        elif referenceMode in scansarModes:
            self._insar.numberRangeLooks2 = 5
        else:
            raise Exception('unknow acquisition mode')

    if self._insar.numberAzimuthLooks2 == None:
        if referenceMode in spotlightModes:
            self._insar.numberAzimuthLooks2 = 4
        elif referenceMode in stripmapModes:
            self._insar.numberAzimuthLooks2 = 4
        elif referenceMode in scansarModes:
            self._insar.numberAzimuthLooks2 = 2
        else:
            raise Exception('unknow acquisition mode')

    if self._insar.numberRangeLooksIon == None:
        if referenceMode in spotlightModes:
            self._insar.numberRangeLooksIon = 16
        elif referenceMode in stripmapModes:
            self._insar.numberRangeLooksIon = 16
        elif referenceMode in scansarModes:
            self._insar.numberRangeLooksIon = 40
        else:
            raise Exception('unknow acquisition mode')

    if self._insar.numberAzimuthLooksIon == None:
        if referenceMode in spotlightModes:
            self._insar.numberAzimuthLooksIon = 16
        elif referenceMode in stripmapModes:
            self._insar.numberAzimuthLooksIon = 16
        elif referenceMode in scansarModes:
            self._insar.numberAzimuthLooksIon = 16
        else:
            raise Exception('unknow acquisition mode')


    #define processing file names
    self._insar.referenceDate = os.path.basename(ledFilesReference[0]).split('-')[2]
    self._insar.secondaryDate = os.path.basename(ledFilesSecondary[0]).split('-')[2]
    self._insar.setFilename(referenceDate=self._insar.referenceDate, secondaryDate=self._insar.secondaryDate, nrlks1=self._insar.numberRangeLooks1, nalks1=self._insar.numberAzimuthLooks1, nrlks2=self._insar.numberRangeLooks2, nalks2=self._insar.numberAzimuthLooks2)


    #find frame numbers
    if (self._insar.modeCombination == 31) or (self._insar.modeCombination == 32):
        if (self.referenceFrames == None) or (self.secondaryFrames == None):
            raise Exception('for ScanSAR-stripmap inteferometry, you must set reference and secondary frame numbers')
    #if not set, find frames automatically
    if self.referenceFrames == None:
        self.referenceFrames = []
        for led in ledFilesReference:
            frameNumber = os.path.basename(led).split('-')[1][-4:]
            if frameNumber not in self.referenceFrames:
                self.referenceFrames.append(frameNumber)
    if self.secondaryFrames == None:
        self.secondaryFrames = []
        for led in ledFilesSecondary:
            frameNumber = os.path.basename(led).split('-')[1][-4:]
            if frameNumber not in self.secondaryFrames:
                self.secondaryFrames.append(frameNumber)
    #sort frames
    self.referenceFrames = sorted(self.referenceFrames)
    self.secondaryFrames = sorted(self.secondaryFrames)
    #check number of frames
    if len(self.referenceFrames) != len(self.secondaryFrames):
        raise Exception('number of frames in reference dir is not equal to number of frames \
            in secondary dir. please set frame number manually')


    #find swath numbers (if not ScanSAR-ScanSAR, compute valid swaths)
    if (self._insar.modeCombination == 0) or (self._insar.modeCombination == 1):
        self.startingSwath = 1
        self.endingSwath = 1

    if self._insar.modeCombination == 21:
        if self.startingSwath == None:
            self.startingSwath = 1
        if self.endingSwath == None:
            self.endingSwath = 5

    if self._insar.modeCombination == 22:
        if self.startingSwath == None:
            self.startingSwath = 1
        if self.endingSwath == None:
            self.endingSwath = 7

    #determine starting and ending swaths for ScanSAR-stripmap, user's settings are overwritten
    #use first frame to check overlap
    if (self._insar.modeCombination == 31) or (self._insar.modeCombination == 32):
        if self._insar.modeCombination == 31:
            numberOfSwaths = 5
        else:
            numberOfSwaths = 7
        overlapSubswaths = []
        for i in range(numberOfSwaths):
            overlapRatio = check_overlap(ledFilesReference[0], firstFrameImagesReference[i], ledFilesSecondary[0], firstFrameImagesSecondary[0])
            if overlapRatio > 1.0 / 4.0:
                overlapSubswaths.append(i+1)
        if overlapSubswaths == []:
            raise Exception('There is no overlap area between the ScanSAR-stripmap pair')
        self.startingSwath = int(overlapSubswaths[0])
        self.endingSwath = int(overlapSubswaths[-1])

    #save the valid frames and swaths for future processing
    self._insar.referenceFrames = self.referenceFrames
    self._insar.secondaryFrames = self.secondaryFrames
    self._insar.startingSwath = self.startingSwath
    self._insar.endingSwath = self.endingSwath


    ##################################################
    #1. create directories and read data
    ##################################################
    self.reference.configure()
    self.secondary.configure()
    self.reference.track.configure()
    self.secondary.track.configure()
    for i, (referenceFrame, secondaryFrame) in enumerate(zip(self._insar.referenceFrames, self._insar.secondaryFrames)):
        #frame number starts with 1
        frameDir = 'f{}_{}'.format(i+1, referenceFrame)
        os.makedirs(frameDir, exist_ok=True)
        os.chdir(frameDir)

        #attach a frame to reference and secondary
        frameObjReference = MultiMode.createFrame()
        frameObjSecondary = MultiMode.createFrame()
        frameObjReference.configure()
        frameObjSecondary.configure()
        self.reference.track.frames.append(frameObjReference)
        self.secondary.track.frames.append(frameObjSecondary)

        #swath number starts with 1
        for j in range(self._insar.startingSwath, self._insar.endingSwath+1):
            print('processing frame {} swath {}'.format(referenceFrame, j))

            swathDir = 's{}'.format(j)
            os.makedirs(swathDir, exist_ok=True)
            os.chdir(swathDir)

            #attach a swath to reference and secondary
            swathObjReference = MultiMode.createSwath()
            swathObjSecondary = MultiMode.createSwath()
            swathObjReference.configure()
            swathObjSecondary.configure()
            self.reference.track.frames[-1].swaths.append(swathObjReference)
            self.secondary.track.frames[-1].swaths.append(swathObjSecondary)

            #setup reference
            self.reference.leaderFile = sorted(glob.glob(os.path.join(self.referenceDir, 'LED-ALOS2*{}-*-*'.format(referenceFrame))))[0]
            if referenceMode in scansarModes:
                self.reference.imageFile = sorted(glob.glob(os.path.join(self.referenceDir, 'IMG-{}-ALOS2*{}-*-*-F{}'.format(self.referencePolarization.upper(), referenceFrame, j))))[0]
            else:
                self.reference.imageFile = sorted(glob.glob(os.path.join(self.referenceDir, 'IMG-{}-ALOS2*{}-*-*'.format(self.referencePolarization.upper(), referenceFrame))))[0]
            self.reference.outputFile = self._insar.referenceSlc
            self.reference.useVirtualFile = self.useVirtualFile
            #read reference
            (imageFDR, imageData)=self.reference.readImage()
            (leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord)=self.reference.readLeader()
            self.reference.setSwath(leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData)
            self.reference.setFrame(leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData)
            self.reference.setTrack(leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData)

            #setup secondary
            self.secondary.leaderFile = sorted(glob.glob(os.path.join(self.secondaryDir, 'LED-ALOS2*{}-*-*'.format(secondaryFrame))))[0]
            if secondaryMode in scansarModes:
                self.secondary.imageFile = sorted(glob.glob(os.path.join(self.secondaryDir, 'IMG-{}-ALOS2*{}-*-*-F{}'.format(self.secondaryPolarization.upper(), secondaryFrame, j))))[0]
            else:
                self.secondary.imageFile = sorted(glob.glob(os.path.join(self.secondaryDir, 'IMG-{}-ALOS2*{}-*-*'.format(self.secondaryPolarization.upper(), secondaryFrame))))[0]
            self.secondary.outputFile = self._insar.secondarySlc
            self.secondary.useVirtualFile = self.useVirtualFile
            #read secondary
            (imageFDR, imageData)=self.secondary.readImage()
            (leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord)=self.secondary.readLeader()
            self.secondary.setSwath(leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData)
            self.secondary.setFrame(leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData)
            self.secondary.setTrack(leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData)

            os.chdir('../')
        self._insar.saveProduct(self.reference.track.frames[-1], self._insar.referenceFrameParameter)
        self._insar.saveProduct(self.secondary.track.frames[-1], self._insar.secondaryFrameParameter)
        os.chdir('../')
    self._insar.saveProduct(self.reference.track, self._insar.referenceTrackParameter)
    self._insar.saveProduct(self.secondary.track, self._insar.secondaryTrackParameter)


    ##################################################
    #2. compute burst synchronization
    ##################################################
    #burst synchronization may slowly change along a track as a result of the changing relative speed of the two flights
    #in one frame, real unsynchronized time is the same for all swaths
    unsynTime = 0
    #real synchronized time/percentage depends on the swath burst length (synTime = burstlength - abs(unsynTime))
    #synTime = 0
    synPercentage = 0

    numberOfFrames = len(self._insar.referenceFrames)
    numberOfSwaths = self._insar.endingSwath - self._insar.startingSwath + 1
    
    for i, frameNumber in enumerate(self._insar.referenceFrames):
        for j, swathNumber in enumerate(range(self._insar.startingSwath, self._insar.endingSwath + 1)):
            referenceSwath = self.reference.track.frames[i].swaths[j]
            secondarySwath = self.secondary.track.frames[i].swaths[j]
            #using Piyush's code for computing range and azimuth offsets
            midRange = referenceSwath.startingRange + referenceSwath.rangePixelSize * referenceSwath.numberOfSamples * 0.5
            midSensingStart = referenceSwath.sensingStart + datetime.timedelta(seconds = referenceSwath.numberOfLines * 0.5 / referenceSwath.prf)
            llh = self.reference.track.orbit.rdr2geo(midSensingStart, midRange)
            slvaz, slvrng = self.secondary.track.orbit.geo2rdr(llh)
            ###Translate to offsets
            #note that secondary range pixel size and prf might be different from reference, here we assume there is a virtual secondary with same
            #range pixel size and prf
            rgoff = ((slvrng - secondarySwath.startingRange) / referenceSwath.rangePixelSize) - referenceSwath.numberOfSamples * 0.5
            azoff = ((slvaz - secondarySwath.sensingStart).total_seconds() * referenceSwath.prf) - referenceSwath.numberOfLines * 0.5

            #compute burst synchronization
            #burst parameters for ScanSAR wide mode not estimed yet
            if self._insar.modeCombination == 21:
                scburstStartLine = (referenceSwath.burstStartTime - referenceSwath.sensingStart).total_seconds() * referenceSwath.prf + azoff
                #secondary burst start times corresponding to reference burst start times (100% synchronization)
                scburstStartLines = np.arange(scburstStartLine - 100000*referenceSwath.burstCycleLength, \
                                              scburstStartLine + 100000*referenceSwath.burstCycleLength, \
                                              referenceSwath.burstCycleLength)
                dscburstStartLines = -((secondarySwath.burstStartTime - secondarySwath.sensingStart).total_seconds() * secondarySwath.prf - scburstStartLines)
                #find the difference with minimum absolute value
                unsynLines = dscburstStartLines[np.argmin(np.absolute(dscburstStartLines))]
                if np.absolute(unsynLines) >= secondarySwath.burstLength:
                    synLines = 0
                    if unsynLines > 0:
                        unsynLines = secondarySwath.burstLength
                    else:
                        unsynLines = -secondarySwath.burstLength
                else:
                    synLines = secondarySwath.burstLength - np.absolute(unsynLines)

                unsynTime += unsynLines / referenceSwath.prf
                synPercentage += synLines / referenceSwath.burstLength * 100.0

                catalog.addItem('burst synchronization of frame {} swath {}'.format(frameNumber, swathNumber), '%.1f%%'%(synLines / referenceSwath.burstLength * 100.0), 'runPreprocessor')

            ############################################################################################
            #illustration of the sign of the number of unsynchronized lines (unsynLines)     
            #The convention is the same as ampcor offset, that is,
            #              secondaryLineNumber = referenceLineNumber + unsynLines
            #
            # |-----------------------|     ------------
            # |                       |        ^
            # |                       |        |
            # |                       |        |   unsynLines < 0
            # |                       |        |
            # |                       |       \ /
            # |                       |    |-----------------------|
            # |                       |    |                       |
            # |                       |    |                       |
            # |-----------------------|    |                       |
            #        Reference Burst          |                       |
            #                              |                       |
            #                              |                       |
            #                              |                       |
            #                              |                       |
            #                              |-----------------------|
            #                                     Secondary Burst
            #
            #
            ############################################################################################
 
            ##burst parameters for ScanSAR wide mode not estimed yet
            elif self._insar.modeCombination == 31:
                #scansar is reference
                scburstStartLine = (referenceSwath.burstStartTime - referenceSwath.sensingStart).total_seconds() * referenceSwath.prf + azoff
                #secondary burst start times corresponding to reference burst start times (100% synchronization)
                for k in range(-100000, 100000):
                    saz_burstx = scburstStartLine + referenceSwath.burstCycleLength * k
                    st_burstx = secondarySwath.sensingStart + datetime.timedelta(seconds=saz_burstx / referenceSwath.prf)
                    if saz_burstx >= 0.0 and saz_burstx <= secondarySwath.numberOfLines -1:
                        secondarySwath.burstStartTime = st_burstx
                        secondarySwath.burstLength = referenceSwath.burstLength
                        secondarySwath.burstCycleLength = referenceSwath.burstCycleLength
                        secondarySwath.swathNumber = referenceSwath.swathNumber
                        break
                #unsynLines = 0
                #synLines = referenceSwath.burstLength
                #unsynTime += unsynLines / referenceSwath.prf
                #synPercentage += synLines / referenceSwath.burstLength * 100.0
                catalog.addItem('burst synchronization of frame {} swath {}'.format(frameNumber, swathNumber), '%.1f%%'%(100.0), 'runPreprocessor')
            else:
                pass

        #overwrite original frame parameter file
        if self._insar.modeCombination == 31:
            frameDir = 'f{}_{}'.format(i+1, frameNumber)
            self._insar.saveProduct(self.secondary.track.frames[i], os.path.join(frameDir, self._insar.secondaryFrameParameter))

    #getting average
    if self._insar.modeCombination == 21:
        unsynTime /= numberOfFrames*numberOfSwaths
        synPercentage /= numberOfFrames*numberOfSwaths
    elif self._insar.modeCombination == 31:
        unsynTime = 0.
        synPercentage = 100.
    else:
        pass

    #record results
    if (self._insar.modeCombination == 21) or (self._insar.modeCombination == 31):
        self._insar.burstUnsynchronizedTime = unsynTime
        self._insar.burstSynchronization = synPercentage
        catalog.addItem('burst synchronization averaged', '%.1f%%'%(synPercentage), 'runPreprocessor')


    ##################################################
    #3. compute baseline
    ##################################################
    #only compute baseline at four corners and center of the reference track
    bboxRdr = getBboxRdr(self.reference.track)

    rangeMin = bboxRdr[0]
    rangeMax = bboxRdr[1]
    azimuthTimeMin = bboxRdr[2]
    azimuthTimeMax = bboxRdr[3]

    azimuthTimeMid = azimuthTimeMin+datetime.timedelta(seconds=(azimuthTimeMax-azimuthTimeMin).total_seconds()/2.0)
    rangeMid = (rangeMin + rangeMax) / 2.0

    points = [[azimuthTimeMin, rangeMin],
              [azimuthTimeMin, rangeMax],
              [azimuthTimeMax, rangeMin],
              [azimuthTimeMax, rangeMax],
              [azimuthTimeMid, rangeMid]]

    Bpar = []
    Bperp = []
    #modify Piyush's code for computing baslines
    refElp = Planet(pname='Earth').ellipsoid
    for x in points:
        referenceSV = self.reference.track.orbit.interpolate(x[0], method='hermite')
        target = self.reference.track.orbit.rdr2geo(x[0], x[1])

        slvTime, slvrng = self.secondary.track.orbit.geo2rdr(target)
        secondarySV = self.secondary.track.orbit.interpolateOrbit(slvTime, method='hermite')

        targxyz = np.array(refElp.LLH(target[0], target[1], target[2]).ecef().tolist())
        mxyz = np.array(referenceSV.getPosition())
        mvel = np.array(referenceSV.getVelocity())
        sxyz = np.array(secondarySV.getPosition())

        #to fix abrupt change near zero in baseline grid. JUN-05-2020
        mvelunit = mvel / np.linalg.norm(mvel)
        sxyz = sxyz - np.dot ( sxyz-mxyz, mvelunit) * mvelunit

        aa = np.linalg.norm(sxyz-mxyz)
        costheta = (x[1]*x[1] + aa*aa - slvrng*slvrng)/(2.*x[1]*aa)

        Bpar.append(aa*costheta)

        perp = aa * np.sqrt(1 - costheta*costheta)
        direction = np.sign(np.dot( np.cross(targxyz-mxyz, sxyz-mxyz), mvel))
        Bperp.append(direction*perp)    

    catalog.addItem('parallel baseline at upperleft of reference track', Bpar[0], 'runPreprocessor')
    catalog.addItem('parallel baseline at upperright of reference track', Bpar[1], 'runPreprocessor')
    catalog.addItem('parallel baseline at lowerleft of reference track', Bpar[2], 'runPreprocessor')
    catalog.addItem('parallel baseline at lowerright of reference track', Bpar[3], 'runPreprocessor')
    catalog.addItem('parallel baseline at center of reference track', Bpar[4], 'runPreprocessor')

    catalog.addItem('perpendicular baseline at upperleft of reference track', Bperp[0], 'runPreprocessor')
    catalog.addItem('perpendicular baseline at upperright of reference track', Bperp[1], 'runPreprocessor')
    catalog.addItem('perpendicular baseline at lowerleft of reference track', Bperp[2], 'runPreprocessor')
    catalog.addItem('perpendicular baseline at lowerright of reference track', Bperp[3], 'runPreprocessor')
    catalog.addItem('perpendicular baseline at center of reference track', Bperp[4], 'runPreprocessor')


    ##################################################
    #4. compute bounding box
    ##################################################
    referenceBbox = getBboxGeo(self.reference.track)
    secondaryBbox = getBboxGeo(self.secondary.track)

    catalog.addItem('reference bounding box', referenceBbox, 'runPreprocessor')
    catalog.addItem('secondary bounding box', secondaryBbox, 'runPreprocessor')


    catalog.printToLog(logger, "runPreprocessor")
    self._insar.procDoc.addAllFromCatalog(catalog)



def check_overlap(ldr_m, img_m, ldr_s, img_s):
    from isceobj.Constants import SPEED_OF_LIGHT

    rangeSamplingRateReference, widthReference, nearRangeReference = read_param_for_checking_overlap(ldr_m, img_m)
    rangeSamplingRateSecondary, widthSecondary, nearRangeSecondary = read_param_for_checking_overlap(ldr_s, img_s)

    farRangeReference = nearRangeReference + (widthReference-1) * 0.5 * SPEED_OF_LIGHT / rangeSamplingRateReference
    farRangeSecondary = nearRangeSecondary + (widthSecondary-1) * 0.5 * SPEED_OF_LIGHT / rangeSamplingRateSecondary

    #This should be good enough, although precise image offsets are not used.
    if farRangeReference <= nearRangeSecondary:
        overlapRatio = 0.0
    elif farRangeSecondary <= nearRangeReference:
        overlapRatio = 0.0
    else:
        #                     0                  1               2               3
        ranges = np.array([nearRangeReference, farRangeReference, nearRangeSecondary, farRangeSecondary])
        rangesIndex = np.argsort(ranges)
        overlapRatio = ranges[rangesIndex[2]]-ranges[rangesIndex[1]] / (farRangeReference-nearRangeReference)

    return overlapRatio


def read_param_for_checking_overlap(leader_file, image_file):
    from isceobj.Sensor import xmlPrefix
    import isceobj.Sensor.CEOS as CEOS

    #read from leader file
    fsampConst = { 104: 1.047915957140240E+08,
                   52: 5.239579785701190E+07,
                   34: 3.493053190467460E+07,
                   17: 1.746526595233730E+07 }

    fp = open(leader_file,'rb')
    leaderFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/leader_file.xml'),dataFile=fp)
    leaderFDR.parse()
    fp.seek(leaderFDR.getEndOfRecordPosition())
    sceneHeaderRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/scene_record.xml'),dataFile=fp)
    sceneHeaderRecord.parse()
    fp.seek(sceneHeaderRecord.getEndOfRecordPosition())

    fsamplookup = int(sceneHeaderRecord.metadata['Range sampling rate in MHz'])
    rangeSamplingRate = fsampConst[fsamplookup]
    fp.close()
    #print('{}'.format(rangeSamplingRate))

    #read from image file
    fp = open(image_file, 'rb')
    imageFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/image_file.xml'), dataFile=fp)
    imageFDR.parse()
    fp.seek(imageFDR.getEndOfRecordPosition())
    imageData = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/image_record.xml'), dataFile=fp)
    imageData.parseFast()

    width = imageFDR.metadata['Number of pixels per line per SAR channel']
    near_range = imageData.metadata['Slant range to 1st data sample']
    fp.close()
    #print('{}'.format(width))
    #print('{}'.format(near_range))

    return (rangeSamplingRate, width, near_range)


