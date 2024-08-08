#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging

import isceobj

logger = logging.getLogger('isce.alos2insar.runFrameOffset')

def runFrameOffset(self):
    '''estimate frame offsets.
    '''
    if hasattr(self, 'doInSAR'):
        if not self.doInSAR:
            return

    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    referenceTrack = self._insar.loadTrack(reference=True)
    secondaryTrack = self._insar.loadTrack(reference=False)

    mosaicDir = 'insar'
    os.makedirs(mosaicDir, exist_ok=True)
    os.chdir(mosaicDir)

    if len(referenceTrack.frames) > 1:
        if (self._insar.modeCombination == 21) or \
           (self._insar.modeCombination == 22) or \
           (self._insar.modeCombination == 31) or \
           (self._insar.modeCombination == 32):
            matchingMode=0
        else:
            matchingMode=1

        #compute swath offset
        offsetReference = frameOffset(referenceTrack, self._insar.referenceSlc, self._insar.referenceFrameOffset, 
                                   crossCorrelation=self.frameOffsetMatching, matchingMode=matchingMode)
        #only use geometrical offset for secondary
        offsetSecondary = frameOffset(secondaryTrack, self._insar.secondarySlc, self._insar.secondaryFrameOffset, 
                                  crossCorrelation=False, matchingMode=matchingMode)

        self._insar.frameRangeOffsetGeometricalReference = offsetReference[0]
        self._insar.frameAzimuthOffsetGeometricalReference = offsetReference[1]
        self._insar.frameRangeOffsetGeometricalSecondary = offsetSecondary[0]
        self._insar.frameAzimuthOffsetGeometricalSecondary = offsetSecondary[1]
        if self.frameOffsetMatching:
            self._insar.frameRangeOffsetMatchingReference = offsetReference[2]
            self._insar.frameAzimuthOffsetMatchingReference = offsetReference[3]
            #self._insar.frameRangeOffsetMatchingSecondary = offsetSecondary[2]
            #self._insar.frameAzimuthOffsetMatchingSecondary = offsetSecondary[3]


    os.chdir('../')

    catalog.printToLog(logger, "runFrameOffset")
    self._insar.procDoc.addAllFromCatalog(catalog)


def frameOffset(track, image, outputfile, crossCorrelation=True, matchingMode=0):
    '''
    compute frame offset
    track:                track object
    image:                image for doing matching
    outputfile:           output txt file for saving frame offset
    crossCorrelation:     whether do matching
    matchingMode:         how to match images. 0: ScanSAR full-aperture image, 1: regular image
    '''

    rangeOffsetGeometrical = []
    azimuthOffsetGeometrical = []
    rangeOffsetMatching = []
    azimuthOffsetMatching = []

    for j in range(len(track.frames)):
        frameNumber = track.frames[j].frameNumber
        swathNumber = track.frames[j].swaths[0].swathNumber
        swathDir = 'f{}_{}/s{}'.format(j+1, frameNumber, swathNumber)

        print('estimate offset frame {}'.format(frameNumber))

        if j == 0:
            rangeOffsetGeometrical.append(0.0)
            azimuthOffsetGeometrical.append(0.0)
            rangeOffsetMatching.append(0.0)
            azimuthOffsetMatching.append(0.0)
            swathDirLast = swathDir
            continue

        image1 = os.path.join('../', swathDirLast, image)
        image2 = os.path.join('../', swathDir, image)
        #swath1 = frame.swaths[j-1]
        #swath2 = frame.swaths[j]
        swath1 = track.frames[j-1].swaths[0]
        swath2 = track.frames[j].swaths[0]


        #offset from geometry
        offsetGeometrical = computeFrameOffset(swath1, swath2)
        rangeOffsetGeometrical.append(offsetGeometrical[0])
        azimuthOffsetGeometrical.append(offsetGeometrical[1])

        #offset from cross-correlation
        if crossCorrelation:
            offsetMatching = estimateFrameOffset(swath1, swath2, image1, image2, matchingMode=matchingMode)
            if offsetMatching != None:
                rangeOffsetMatching.append(offsetMatching[0])
                azimuthOffsetMatching.append(offsetMatching[1])
            else:
                print('******************************************************************')
                print('WARNING: bad matching offset, we are forced to use')
                print('         geometrical offset for frame mosaicking')
                print('******************************************************************')
                rangeOffsetMatching.append(offsetGeometrical[0])
                azimuthOffsetMatching.append(offsetGeometrical[1])

        swathDirLast = swathDir


    if crossCorrelation:
        offsetComp = "\n\ncomparision of offsets:\n\n"
        offsetComp += "offset type       i     geometrical           match      difference\n"
        offsetComp += "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
        for i, (offset1, offset2) in enumerate(zip(rangeOffsetGeometrical, rangeOffsetMatching)):
            offsetComp += "range offset     {:2d}   {:13.3f}   {:13.3f}   {:13.3f}\n".format(i, offset1, offset2, offset1 - offset2)
        for i, (offset1, offset2) in enumerate(zip(azimuthOffsetGeometrical, azimuthOffsetMatching)):
            offsetComp += "azimuth offset   {:2d}   {:13.3f}   {:13.3f}   {:13.3f}\n".format(i, offset1, offset2, offset1 - offset2)

        #write and report offsets
        with open(outputfile, 'w') as f:
            f.write(offsetComp)
        print("{}".format(offsetComp))


    if crossCorrelation:
        return (rangeOffsetGeometrical, azimuthOffsetGeometrical, rangeOffsetMatching, azimuthOffsetMatching)
    else:
        return (rangeOffsetGeometrical, azimuthOffsetGeometrical)


def computeFrameOffset(swath1, swath2):

    rangeOffset = -(swath2.startingRange - swath1.startingRange) / swath1.rangePixelSize
    azimuthOffset = -((swath2.sensingStart - swath1.sensingStart).total_seconds()) / swath1.azimuthLineInterval

    return (rangeOffset, azimuthOffset)


def estimateFrameOffset(swath1, swath2, image1, image2, matchingMode=0):
    '''
    estimate offset of two adjacent frames using matching
    matchingMode:  0: ScanSAR full-aperture image
                   1: regular image
    '''
    import isceobj
    from isceobj.Alos2Proc.Alos2ProcPublic import cullOffsets
    from isceobj.Alos2Proc.Alos2ProcPublic import cullOffsetsRoipac
    from isceobj.Alos2Proc.Alos2ProcPublic import meanOffset
    from mroipac.ampcor.Ampcor import Ampcor

    ##########################################
    #2. match using ampcor
    ##########################################
    ampcor = Ampcor(name='insarapp_slcs_ampcor')
    ampcor.configure()

    #mSLC = isceobj.createSlcImage()
    mSLC = isceobj.createImage()
    mSLC.load(image1+'.xml')
    mSLC.setFilename(image1)
    #mSLC.extraFilename = image1 + '.vrt'
    mSLC.setAccessMode('read')
    mSLC.createImage()

    #sSLC = isceobj.createSlcImage()
    sSLC = isceobj.createImage()
    sSLC.load(image2+'.xml')
    sSLC.setFilename(image2)
    #sSLC.extraFilename = image2 + '.vrt'
    sSLC.setAccessMode('read')
    sSLC.createImage()

    if mSLC.dataType.upper() == 'CFLOAT':
        ampcor.setImageDataType1('complex')
        ampcor.setImageDataType2('complex')
    elif mSLC.dataType.upper() == 'FLOAT':
        ampcor.setImageDataType1('real')
        ampcor.setImageDataType2('real')
    else:
        raise Exception('file type not supported yet.')

    ampcor.setReferenceSlcImage(mSLC)
    ampcor.setSecondarySlcImage(sSLC)

    #MATCH REGION
    #compute an offset at image center to use
    rgoff = -(swath2.startingRange - swath1.startingRange) / swath1.rangePixelSize
    azoff = -((swath2.sensingStart - swath1.sensingStart).total_seconds()) / swath1.azimuthLineInterval
    rgoff = int(rgoff)
    azoff = int(azoff)
    #it seems that we cannot use 0, haven't look into the problem
    if rgoff == 0:
        rgoff = 1
    if azoff == 0:
        azoff = 1
    firstSample = 1
    if rgoff < 0:
        firstSample = int(35 - rgoff)
    firstLine = 1
    if azoff < 0:
        firstLine = int(35 - azoff)
    ampcor.setAcrossGrossOffset(rgoff)
    ampcor.setDownGrossOffset(azoff)
    ampcor.setFirstSampleAcross(firstSample)
    ampcor.setLastSampleAcross(mSLC.width)
    ampcor.setNumberLocationAcross(30)
    ampcor.setFirstSampleDown(firstLine)
    ampcor.setLastSampleDown(mSLC.length)
    ampcor.setNumberLocationDown(10)

    #MATCH PARAMETERS
    #full-aperture mode
    if matchingMode==0:
        ampcor.setWindowSizeWidth(64)
        ampcor.setWindowSizeHeight(512)
        #note this is the half width/length of search area, number of resulting correlation samples: 32*2+1
        ampcor.setSearchWindowSizeWidth(32)
        ampcor.setSearchWindowSizeHeight(32)
        #triggering full-aperture mode matching
        ampcor.setWinsizeFilt(8)
        ampcor.setOversamplingFactorFilt(64)
    #regular mode
    else:
        ampcor.setWindowSizeWidth(64)
        ampcor.setWindowSizeHeight(64)
        ampcor.setSearchWindowSizeWidth(32)
        ampcor.setSearchWindowSizeHeight(32)

    #REST OF THE STUFF
    ampcor.setAcrossLooks(1)
    ampcor.setDownLooks(1)
    ampcor.setOversamplingFactor(64)
    ampcor.setZoomWindowSize(16)
    #1. The following not set
    #Matching Scale for Sample/Line Directions                       (-)    = 1. 1.
    #should add the following in Ampcor.py?
    #if not set, in this case, Ampcor.py'value is also 1. 1.
    #ampcor.setScaleFactorX(1.)
    #ampcor.setScaleFactorY(1.)

    #MATCH THRESHOLDS AND DEBUG DATA
    #2. The following not set
    #in roi_pac the value is set to 0 1
    #in isce the value is set to 0.001 1000.0
    #SNR and Covariance Thresholds                                   (-)    =  {s1} {s2}
    #should add the following in Ampcor?
    #THIS SHOULD BE THE ONLY THING THAT IS DIFFERENT FROM THAT OF ROI_PAC
    #ampcor.setThresholdSNR(0)
    #ampcor.setThresholdCov(1)
    ampcor.setDebugFlag(False)
    ampcor.setDisplayFlag(False)

    #in summary, only two things not set which are indicated by 'The following not set' above.

    #run ampcor
    ampcor.ampcor()
    offsets = ampcor.getOffsetField()
    #ampcorOffsetFile = 'ampcor.off'
    #writeOffset(offsets, ampcorOffsetFile)

    #finalize image, and re-create it
    #otherwise the file pointer is still at the end of the image
    mSLC.finalizeImage()
    sSLC.finalizeImage()


    #############################################
    #3. cull offsets
    #############################################
    #refinedOffsets = cullOffsets(offsets)
    refinedOffsets = cullOffsetsRoipac(offsets, numThreshold=50)

    if refinedOffsets != None:
        rangeOffset, azimuthOffset = meanOffset(refinedOffsets)
        return (rangeOffset, azimuthOffset)
    else:
        return None
