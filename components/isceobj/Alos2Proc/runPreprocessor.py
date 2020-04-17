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
    self.masterDir = os.path.abspath(self.masterDir)
    self.slaveDir = os.path.abspath(self.slaveDir)

    ledFilesMaster = sorted(glob.glob(os.path.join(self.masterDir, 'LED-ALOS2*-*-*')))
    imgFilesMaster = sorted(glob.glob(os.path.join(self.masterDir, 'IMG-{}-ALOS2*-*-*'.format(self.masterPolarization.upper()))))

    ledFilesSlave = sorted(glob.glob(os.path.join(self.slaveDir, 'LED-ALOS2*-*-*')))
    imgFilesSlave = sorted(glob.glob(os.path.join(self.slaveDir, 'IMG-{}-ALOS2*-*-*'.format(self.slavePolarization.upper()))))

    firstFrameMaster = ledFilesMaster[0].split('-')[-3][-4:]
    firstFrameSlave = ledFilesSlave[0].split('-')[-3][-4:]
    firstFrameImagesMaster = sorted(glob.glob(os.path.join(self.masterDir, 'IMG-{}-ALOS2*{}-*-*'.format(self.masterPolarization.upper(), firstFrameMaster))))
    firstFrameImagesSlave = sorted(glob.glob(os.path.join(self.slaveDir, 'IMG-{}-ALOS2*{}-*-*'.format(self.slavePolarization.upper(), firstFrameSlave))))


    #determin operation mode
    masterMode = os.path.basename(ledFilesMaster[0]).split('-')[-1][0:3]
    slaveMode = os.path.basename(ledFilesSlave[0]).split('-')[-1][0:3]
    spotlightModes = ['SBS']
    stripmapModes = ['UBS', 'UBD', 'HBS', 'HBD', 'HBQ', 'FBS', 'FBD', 'FBQ']
    scansarNominalModes = ['WBS', 'WBD', 'WWS', 'WWD']
    scansarWideModes = ['VBS', 'VBD']
    scansarModes = ['WBS', 'WBD', 'WWS', 'WWD', 'VBS', 'VBD']

    #usable combinations
    if (masterMode in spotlightModes) and (slaveMode in spotlightModes):
        self._insar.modeCombination = 0
    elif (masterMode in stripmapModes) and (slaveMode in stripmapModes):
        self._insar.modeCombination = 1
    elif (masterMode in scansarNominalModes) and (slaveMode in scansarNominalModes):
        self._insar.modeCombination = 21
    elif (masterMode in scansarWideModes) and (slaveMode in scansarWideModes):
        self._insar.modeCombination = 22
    elif (masterMode in scansarNominalModes) and (slaveMode in stripmapModes):
        self._insar.modeCombination = 31
    elif (masterMode in scansarWideModes) and (slaveMode in stripmapModes):
        self._insar.modeCombination = 32
    else:
        print('\n\nthis mode combination is not possible')
        print('note that for ScanSAR-stripmap, ScanSAR must be master\n\n')
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
        if masterMode in ['SBS']:
            self._insar.numberRangeLooks1 = 2
        elif masterMode in ['UBS', 'UBD']:
            self._insar.numberRangeLooks1 = 2
        elif masterMode in ['HBS', 'HBD', 'HBQ']:
            self._insar.numberRangeLooks1 = 2
        elif masterMode in ['FBS', 'FBD', 'FBQ']:
            self._insar.numberRangeLooks1 = 2
        elif masterMode in ['WBS', 'WBD']:
            self._insar.numberRangeLooks1 = 1
        elif masterMode in ['WWS', 'WWD']:
            self._insar.numberRangeLooks1 = 2
        elif masterMode in ['VBS', 'VBD']:
            self._insar.numberRangeLooks1 = 1
        else:
            raise Exception('unknow acquisition mode')

    if self._insar.numberAzimuthLooks1 == None:
        if masterMode in ['SBS']:
            self._insar.numberAzimuthLooks1 = 4
        elif masterMode in ['UBS', 'UBD']:
            self._insar.numberAzimuthLooks1 = 2
        elif masterMode in ['HBS', 'HBD', 'HBQ']:
            self._insar.numberAzimuthLooks1 = 2
        elif masterMode in ['FBS', 'FBD', 'FBQ']:
            self._insar.numberAzimuthLooks1 = 4
        elif masterMode in ['WBS', 'WBD']:
            self._insar.numberAzimuthLooks1 = 14
        elif masterMode in ['WWS', 'WWD']:
            self._insar.numberAzimuthLooks1 = 14
        elif masterMode in ['VBS', 'VBD']:
            self._insar.numberAzimuthLooks1 = 14
        else:
            raise Exception('unknow acquisition mode')

    if self._insar.numberRangeLooks2 == None:
        if masterMode in spotlightModes:
            self._insar.numberRangeLooks2 = 4
        elif masterMode in stripmapModes:
            self._insar.numberRangeLooks2 = 4
        elif masterMode in scansarModes:
            self._insar.numberRangeLooks2 = 5
        else:
            raise Exception('unknow acquisition mode')

    if self._insar.numberAzimuthLooks2 == None:
        if masterMode in spotlightModes:
            self._insar.numberAzimuthLooks2 = 4
        elif masterMode in stripmapModes:
            self._insar.numberAzimuthLooks2 = 4
        elif masterMode in scansarModes:
            self._insar.numberAzimuthLooks2 = 2
        else:
            raise Exception('unknow acquisition mode')

    if self._insar.numberRangeLooksIon == None:
        if masterMode in spotlightModes:
            self._insar.numberRangeLooksIon = 16
        elif masterMode in stripmapModes:
            self._insar.numberRangeLooksIon = 16
        elif masterMode in scansarModes:
            self._insar.numberRangeLooksIon = 40
        else:
            raise Exception('unknow acquisition mode')

    if self._insar.numberAzimuthLooksIon == None:
        if masterMode in spotlightModes:
            self._insar.numberAzimuthLooksIon = 16
        elif masterMode in stripmapModes:
            self._insar.numberAzimuthLooksIon = 16
        elif masterMode in scansarModes:
            self._insar.numberAzimuthLooksIon = 16
        else:
            raise Exception('unknow acquisition mode')


    #define processing file names
    self._insar.masterDate = os.path.basename(ledFilesMaster[0]).split('-')[2]
    self._insar.slaveDate = os.path.basename(ledFilesSlave[0]).split('-')[2]
    self._insar.setFilename(masterDate=self._insar.masterDate, slaveDate=self._insar.slaveDate, nrlks1=self._insar.numberRangeLooks1, nalks1=self._insar.numberAzimuthLooks1, nrlks2=self._insar.numberRangeLooks2, nalks2=self._insar.numberAzimuthLooks2)


    #find frame numbers
    if (self._insar.modeCombination == 31) or (self._insar.modeCombination == 32):
        if (self.masterFrames == None) or (self.slaveFrames == None):
            raise Exception('for ScanSAR-stripmap inteferometry, you must set master and slave frame numbers')
    #if not set, find frames automatically
    if self.masterFrames == None:
        self.masterFrames = []
        for led in ledFilesMaster:
            frameNumber = os.path.basename(led).split('-')[1][-4:]
            if frameNumber not in self.masterFrames:
                self.masterFrames.append(frameNumber)
    if self.slaveFrames == None:
        self.slaveFrames = []
        for led in ledFilesSlave:
            frameNumber = os.path.basename(led).split('-')[1][-4:]
            if frameNumber not in self.slaveFrames:
                self.slaveFrames.append(frameNumber)
    #sort frames
    self.masterFrames = sorted(self.masterFrames)
    self.slaveFrames = sorted(self.slaveFrames)
    #check number of frames
    if len(self.masterFrames) != len(self.slaveFrames):
        raise Exception('number of frames in master dir is not equal to number of frames \
            in slave dir. please set frame number manually')


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
            overlapRatio = check_overlap(ledFilesMaster[0], firstFrameImagesMaster[i], ledFilesSlave[0], firstFrameImagesSlave[0])
            if overlapRatio > 1.0 / 4.0:
                overlapSubswaths.append(i+1)
        if overlapSubswaths == []:
            raise Exception('There is no overlap area between the ScanSAR-stripmap pair')
        self.startingSwath = int(overlapSubswaths[0])
        self.endingSwath = int(overlapSubswaths[-1])

    #save the valid frames and swaths for future processing
    self._insar.masterFrames = self.masterFrames
    self._insar.slaveFrames = self.slaveFrames
    self._insar.startingSwath = self.startingSwath
    self._insar.endingSwath = self.endingSwath


    ##################################################
    #1. create directories and read data
    ##################################################
    self.master.configure()
    self.slave.configure()
    self.master.track.configure()
    self.slave.track.configure()
    for i, (masterFrame, slaveFrame) in enumerate(zip(self._insar.masterFrames, self._insar.slaveFrames)):
        #frame number starts with 1
        frameDir = 'f{}_{}'.format(i+1, masterFrame)
        os.makedirs(frameDir, exist_ok=True)
        os.chdir(frameDir)

        #attach a frame to master and slave
        frameObjMaster = MultiMode.createFrame()
        frameObjSlave = MultiMode.createFrame()
        frameObjMaster.configure()
        frameObjSlave.configure()
        self.master.track.frames.append(frameObjMaster)
        self.slave.track.frames.append(frameObjSlave)

        #swath number starts with 1
        for j in range(self._insar.startingSwath, self._insar.endingSwath+1):
            print('processing frame {} swath {}'.format(masterFrame, j))

            swathDir = 's{}'.format(j)
            os.makedirs(swathDir, exist_ok=True)
            os.chdir(swathDir)

            #attach a swath to master and slave
            swathObjMaster = MultiMode.createSwath()
            swathObjSlave = MultiMode.createSwath()
            swathObjMaster.configure()
            swathObjSlave.configure()
            self.master.track.frames[-1].swaths.append(swathObjMaster)
            self.slave.track.frames[-1].swaths.append(swathObjSlave)

            #setup master
            self.master.leaderFile = sorted(glob.glob(os.path.join(self.masterDir, 'LED-ALOS2*{}-*-*'.format(masterFrame))))[0]
            if masterMode in scansarModes:
                self.master.imageFile = sorted(glob.glob(os.path.join(self.masterDir, 'IMG-{}-ALOS2*{}-*-*-F{}'.format(self.masterPolarization.upper(), masterFrame, j))))[0]
            else:
                self.master.imageFile = sorted(glob.glob(os.path.join(self.masterDir, 'IMG-{}-ALOS2*{}-*-*'.format(self.masterPolarization.upper(), masterFrame))))[0]
            self.master.outputFile = self._insar.masterSlc
            self.master.useVirtualFile = self.useVirtualFile
            #read master
            (imageFDR, imageData)=self.master.readImage()
            (leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord)=self.master.readLeader()
            self.master.setSwath(leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData)
            self.master.setFrame(leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData)
            self.master.setTrack(leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData)

            #setup slave
            self.slave.leaderFile = sorted(glob.glob(os.path.join(self.slaveDir, 'LED-ALOS2*{}-*-*'.format(slaveFrame))))[0]
            if slaveMode in scansarModes:
                self.slave.imageFile = sorted(glob.glob(os.path.join(self.slaveDir, 'IMG-{}-ALOS2*{}-*-*-F{}'.format(self.slavePolarization.upper(), slaveFrame, j))))[0]
            else:
                self.slave.imageFile = sorted(glob.glob(os.path.join(self.slaveDir, 'IMG-{}-ALOS2*{}-*-*'.format(self.slavePolarization.upper(), slaveFrame))))[0]
            self.slave.outputFile = self._insar.slaveSlc
            self.slave.useVirtualFile = self.useVirtualFile
            #read slave
            (imageFDR, imageData)=self.slave.readImage()
            (leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord)=self.slave.readLeader()
            self.slave.setSwath(leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData)
            self.slave.setFrame(leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData)
            self.slave.setTrack(leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData)

            os.chdir('../')
        self._insar.saveProduct(self.master.track.frames[-1], self._insar.masterFrameParameter)
        self._insar.saveProduct(self.slave.track.frames[-1], self._insar.slaveFrameParameter)
        os.chdir('../')
    self._insar.saveProduct(self.master.track, self._insar.masterTrackParameter)
    self._insar.saveProduct(self.slave.track, self._insar.slaveTrackParameter)


    ##################################################
    #2. compute burst synchronization
    ##################################################
    #burst synchronization may slowly change along a track as a result of the changing relative speed of the two flights
    #in one frame, real unsynchronized time is the same for all swaths
    unsynTime = 0
    #real synchronized time/percentage depends on the swath burst length (synTime = burstlength - abs(unsynTime))
    #synTime = 0
    synPercentage = 0

    numberOfFrames = len(self._insar.masterFrames)
    numberOfSwaths = self._insar.endingSwath - self._insar.startingSwath + 1
    
    for i, frameNumber in enumerate(self._insar.masterFrames):
        for j, swathNumber in enumerate(range(self._insar.startingSwath, self._insar.endingSwath + 1)):
            masterSwath = self.master.track.frames[i].swaths[j]
            slaveSwath = self.slave.track.frames[i].swaths[j]
            #using Piyush's code for computing range and azimuth offsets
            midRange = masterSwath.startingRange + masterSwath.rangePixelSize * masterSwath.numberOfSamples * 0.5
            midSensingStart = masterSwath.sensingStart + datetime.timedelta(seconds = masterSwath.numberOfLines * 0.5 / masterSwath.prf)
            llh = self.master.track.orbit.rdr2geo(midSensingStart, midRange)
            slvaz, slvrng = self.slave.track.orbit.geo2rdr(llh)
            ###Translate to offsets
            #note that slave range pixel size and prf might be different from master, here we assume there is a virtual slave with same
            #range pixel size and prf
            rgoff = ((slvrng - slaveSwath.startingRange) / masterSwath.rangePixelSize) - masterSwath.numberOfSamples * 0.5
            azoff = ((slvaz - slaveSwath.sensingStart).total_seconds() * masterSwath.prf) - masterSwath.numberOfLines * 0.5

            #compute burst synchronization
            #burst parameters for ScanSAR wide mode not estimed yet
            if self._insar.modeCombination == 21:
                scburstStartLine = (masterSwath.burstStartTime - masterSwath.sensingStart).total_seconds() * masterSwath.prf + azoff
                #slave burst start times corresponding to master burst start times (100% synchronization)
                scburstStartLines = np.arange(scburstStartLine - 100000*masterSwath.burstCycleLength, \
                                              scburstStartLine + 100000*masterSwath.burstCycleLength, \
                                              masterSwath.burstCycleLength)
                dscburstStartLines = -((slaveSwath.burstStartTime - slaveSwath.sensingStart).total_seconds() * slaveSwath.prf - scburstStartLines)
                #find the difference with minimum absolute value
                unsynLines = dscburstStartLines[np.argmin(np.absolute(dscburstStartLines))]
                if np.absolute(unsynLines) >= slaveSwath.burstLength:
                    synLines = 0
                    if unsynLines > 0:
                        unsynLines = slaveSwath.burstLength
                    else:
                        unsynLines = -slaveSwath.burstLength
                else:
                    synLines = slaveSwath.burstLength - np.absolute(unsynLines)

                unsynTime += unsynLines / masterSwath.prf
                synPercentage += synLines / masterSwath.burstLength * 100.0

                catalog.addItem('burst synchronization of frame {} swath {}'.format(frameNumber, swathNumber), '%.1f%%'%(synLines / masterSwath.burstLength * 100.0), 'runPreprocessor')

            ############################################################################################
            #illustration of the sign of the number of unsynchronized lines (unsynLines)     
            #The convention is the same as ampcor offset, that is,
            #              slaveLineNumber = masterLineNumber + unsynLines
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
            #        Master Burst          |                       |
            #                              |                       |
            #                              |                       |
            #                              |                       |
            #                              |                       |
            #                              |-----------------------|
            #                                     Slave Burst
            #
            #
            ############################################################################################
 
            ##burst parameters for ScanSAR wide mode not estimed yet
            elif self._insar.modeCombination == 31:
                #scansar is master
                scburstStartLine = (masterSwath.burstStartTime - masterSwath.sensingStart).total_seconds() * masterSwath.prf + azoff
                #slave burst start times corresponding to master burst start times (100% synchronization)
                for k in range(-100000, 100000):
                    saz_burstx = scburstStartLine + masterSwath.burstCycleLength * k
                    st_burstx = slaveSwath.sensingStart + datetime.timedelta(seconds=saz_burstx / masterSwath.prf)
                    if saz_burstx >= 0.0 and saz_burstx <= slaveSwath.numberOfLines -1:
                        slaveSwath.burstStartTime = st_burstx
                        slaveSwath.burstLength = masterSwath.burstLength
                        slaveSwath.burstCycleLength = masterSwath.burstCycleLength
                        slaveSwath.swathNumber = masterSwath.swathNumber
                        break
                #unsynLines = 0
                #synLines = masterSwath.burstLength
                #unsynTime += unsynLines / masterSwath.prf
                #synPercentage += synLines / masterSwath.burstLength * 100.0
                catalog.addItem('burst synchronization of frame {} swath {}'.format(frameNumber, swathNumber), '%.1f%%'%(100.0), 'runPreprocessor')
            else:
                pass

        #overwrite original frame parameter file
        if self._insar.modeCombination == 31:
            frameDir = 'f{}_{}'.format(i+1, frameNumber)
            self._insar.saveProduct(self.slave.track.frames[i], os.path.join(frameDir, self._insar.slaveFrameParameter))

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
    #only compute baseline at four corners and center of the master track
    bboxRdr = getBboxRdr(self.master.track)

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
        masterSV = self.master.track.orbit.interpolate(x[0], method='hermite')
        target = self.master.track.orbit.rdr2geo(x[0], x[1])

        slvTime, slvrng = self.slave.track.orbit.geo2rdr(target)
        slaveSV = self.slave.track.orbit.interpolateOrbit(slvTime, method='hermite')

        targxyz = np.array(refElp.LLH(target[0], target[1], target[2]).ecef().tolist())
        mxyz = np.array(masterSV.getPosition())
        mvel = np.array(masterSV.getVelocity())
        sxyz = np.array(slaveSV.getPosition())

        aa = np.linalg.norm(sxyz-mxyz)
        costheta = (x[1]*x[1] + aa*aa - slvrng*slvrng)/(2.*x[1]*aa)

        Bpar.append(aa*costheta)

        perp = aa * np.sqrt(1 - costheta*costheta)
        direction = np.sign(np.dot( np.cross(targxyz-mxyz, sxyz-mxyz), mvel))
        Bperp.append(direction*perp)    

    catalog.addItem('parallel baseline at upperleft of master track', Bpar[0], 'runPreprocessor')
    catalog.addItem('parallel baseline at upperright of master track', Bpar[1], 'runPreprocessor')
    catalog.addItem('parallel baseline at lowerleft of master track', Bpar[2], 'runPreprocessor')
    catalog.addItem('parallel baseline at lowerright of master track', Bpar[3], 'runPreprocessor')
    catalog.addItem('parallel baseline at center of master track', Bpar[4], 'runPreprocessor')

    catalog.addItem('perpendicular baseline at upperleft of master track', Bperp[0], 'runPreprocessor')
    catalog.addItem('perpendicular baseline at upperright of master track', Bperp[1], 'runPreprocessor')
    catalog.addItem('perpendicular baseline at lowerleft of master track', Bperp[2], 'runPreprocessor')
    catalog.addItem('perpendicular baseline at lowerright of master track', Bperp[3], 'runPreprocessor')
    catalog.addItem('perpendicular baseline at center of master track', Bperp[4], 'runPreprocessor')


    ##################################################
    #4. compute bounding box
    ##################################################
    masterBbox = getBboxGeo(self.master.track)
    slaveBbox = getBboxGeo(self.slave.track)

    catalog.addItem('master bounding box', masterBbox, 'runPreprocessor')
    catalog.addItem('slave bounding box', slaveBbox, 'runPreprocessor')


    catalog.printToLog(logger, "runPreprocessor")
    self._insar.procDoc.addAllFromCatalog(catalog)



def check_overlap(ldr_m, img_m, ldr_s, img_s):
    from isceobj.Constants import SPEED_OF_LIGHT

    rangeSamplingRateMaster, widthMaster, nearRangeMaster = read_param_for_checking_overlap(ldr_m, img_m)
    rangeSamplingRateSlave, widthSlave, nearRangeSlave = read_param_for_checking_overlap(ldr_s, img_s)

    farRangeMaster = nearRangeMaster + (widthMaster-1) * 0.5 * SPEED_OF_LIGHT / rangeSamplingRateMaster
    farRangeSlave = nearRangeSlave + (widthSlave-1) * 0.5 * SPEED_OF_LIGHT / rangeSamplingRateSlave

    #This should be good enough, although precise image offsets are not used.
    if farRangeMaster <= nearRangeSlave:
        overlapRatio = 0.0
    elif farRangeSlave <= nearRangeMaster:
        overlapRatio = 0.0
    else:
        #                     0                  1               2               3
        ranges = np.array([nearRangeMaster, farRangeMaster, nearRangeSlave, farRangeSlave])
        rangesIndex = np.argsort(ranges)
        overlapRatio = ranges[rangesIndex[2]]-ranges[rangesIndex[1]] / (farRangeMaster-nearRangeMaster)

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


