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

logger = logging.getLogger('isce.alos2burstinsar.runPreprocessor')

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


    if self._insar.modeCombination != 21:
        print('\n\nburst processing only support {}\n\n'.format(scansarNominalModes))
        raise Exception('mode combination not supported')


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
    self._insar.numberRangeLooksSd = self.numberRangeLooksSd
    self._insar.numberAzimuthLooksSd = self.numberAzimuthLooksSd

    #force number of looks 1 to 1
    self.numberRangeLooks1 = 1
    self.numberAzimuthLooks1 = 1
    self._insar.numberRangeLooks1 = 1
    self._insar.numberAzimuthLooks1 = 1
    if self._insar.numberRangeLooks2 == None:
        self._insar.numberRangeLooks2 = 7
    if self._insar.numberAzimuthLooks2 == None:
        self._insar.numberAzimuthLooks2 = 2
    if self._insar.numberRangeLooksIon == None:
        self._insar.numberRangeLooksIon = 42
    if self._insar.numberAzimuthLooksIon == None:
        self._insar.numberAzimuthLooksIon = 12
    if self._insar.numberRangeLooksSd == None:
        self._insar.numberRangeLooksSd = 14
    if self._insar.numberAzimuthLooksSd == None:
        self._insar.numberAzimuthLooksSd = 4

    #define processing file names
    self._insar.referenceDate = os.path.basename(ledFilesReference[0]).split('-')[2]
    self._insar.secondaryDate = os.path.basename(ledFilesSecondary[0]).split('-')[2]
    self._insar.setFilename(referenceDate=self._insar.referenceDate, secondaryDate=self._insar.secondaryDate, 
        nrlks1=self._insar.numberRangeLooks1, nalks1=self._insar.numberAzimuthLooks1, 
        nrlks2=self._insar.numberRangeLooks2, nalks2=self._insar.numberAzimuthLooks2)
    self._insar.setFilenameSd(referenceDate=self._insar.referenceDate, secondaryDate=self._insar.secondaryDate, 
        nrlks1=self._insar.numberRangeLooks1, nalks1=self._insar.numberAzimuthLooks1, 
        nrlks_sd=self._insar.numberRangeLooksSd, nalks_sd=self._insar.numberAzimuthLooksSd, nsd=3)

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


