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
from isceobj.Alos2Proc.Alos2ProcPublic import multilook


logger = logging.getLogger('isce.alos2insar.runSwathOffset')

def runSwathOffset(self):
    '''estimate swath offsets.
    '''
    if hasattr(self, 'doInSAR'):
        if not self.doInSAR:
            return

    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    referenceTrack = self._insar.loadTrack(reference=True)
    secondaryTrack = self._insar.loadTrack(reference=False)

    for i, frameNumber in enumerate(self._insar.referenceFrames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        os.chdir(frameDir)

        mosaicDir = 'mosaic'
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

            os.chdir('../../')

            continue

        #compute swath offset
        offsetReference = swathOffset(referenceTrack.frames[i], self._insar.referenceSlc, self._insar.referenceSwathOffset, 
                                   crossCorrelation=self.swathOffsetMatching, numberOfAzimuthLooks=10)
        #only use geometrical offset for secondary
        offsetSecondary = swathOffset(secondaryTrack.frames[i], self._insar.secondarySlc, self._insar.secondarySwathOffset, 
                                  crossCorrelation=False, numberOfAzimuthLooks=10)

        #initialization
        if i == 0:
            self._insar.swathRangeOffsetGeometricalReference = []
            self._insar.swathAzimuthOffsetGeometricalReference = []
            self._insar.swathRangeOffsetGeometricalSecondary = []
            self._insar.swathAzimuthOffsetGeometricalSecondary = []
            if self.swathOffsetMatching:
                self._insar.swathRangeOffsetMatchingReference = []
                self._insar.swathAzimuthOffsetMatchingReference = []
                #self._insar.swathRangeOffsetMatchingSecondary = []
                #self._insar.swathAzimuthOffsetMatchingSecondary = []

        #append list directly, as the API support 2-d list
        self._insar.swathRangeOffsetGeometricalReference.append(offsetReference[0])
        self._insar.swathAzimuthOffsetGeometricalReference.append(offsetReference[1])
        self._insar.swathRangeOffsetGeometricalSecondary.append(offsetSecondary[0])
        self._insar.swathAzimuthOffsetGeometricalSecondary.append(offsetSecondary[1])
        if self.swathOffsetMatching:
            self._insar.swathRangeOffsetMatchingReference.append(offsetReference[2])
            self._insar.swathAzimuthOffsetMatchingReference.append(offsetReference[3])
            #self._insar.swathRangeOffsetMatchingSecondary.append(offsetSecondary[2])
            #self._insar.swathAzimuthOffsetMatchingSecondary.append(offsetSecondary[3])

        os.chdir('../../')

    catalog.printToLog(logger, "runSwathOffset")
    self._insar.procDoc.addAllFromCatalog(catalog)


def swathOffset(frame, image, outputfile, crossCorrelation=True, numberOfAzimuthLooks=10):
    '''
    compute swath offset
    frame:                frame object
    image:                image for doing matching
    outputfile:           output txt file for saving swath offset
    crossCorrelation:     whether do matching
    numberOfAzimuthLooks: number of looks to take in azimuth before matching
    '''

    rangeOffsetGeometrical = []
    azimuthOffsetGeometrical = []
    rangeOffsetMatching = []
    azimuthOffsetMatching = []

    for j in range(len(frame.swaths)):
        frameNumber = frame.frameNumber
        swathNumber = frame.swaths[j].swathNumber
        swathDir = 's{}'.format(swathNumber)

        print('estimate offset frame {}, swath {}'.format(frameNumber, swathNumber))

        if j == 0:
            rangeOffsetGeometrical.append(0.0)
            azimuthOffsetGeometrical.append(0.0)
            rangeOffsetMatching.append(0.0)
            azimuthOffsetMatching.append(0.0)
            swathDirLast = swathDir
            continue

        image1 = os.path.join('../', swathDirLast, image)
        image2 = os.path.join('../', swathDir, image)
        swath0 = frame.swaths[0]
        swath1 = frame.swaths[j-1]
        swath2 = frame.swaths[j]

        rangeScale1   = swath0.rangePixelSize / swath1.rangePixelSize
        azimuthScale1 = swath0.azimuthLineInterval / swath1.azimuthLineInterval
        rangeScale2   = swath0.rangePixelSize / swath2.rangePixelSize
        azimuthScale2 = swath0.azimuthLineInterval / swath2.azimuthLineInterval

        #offset from geometry
        offsetGeometrical = computeSwathOffset(swath1, swath2, rangeScale1, azimuthScale1)
        rangeOffsetGeometrical.append(offsetGeometrical[0])
        azimuthOffsetGeometrical.append(offsetGeometrical[1])

        #offset from cross-correlation
        if crossCorrelation:
            offsetMatching = estimateSwathOffset(swath1, swath2, image1, image2, rangeScale1, 
                                                 azimuthScale1, rangeScale2, azimuthScale2, numberOfAzimuthLooks)
            if offsetMatching != None:
                rangeOffsetMatching.append(offsetMatching[0])
                azimuthOffsetMatching.append(offsetMatching[1])
            else:
                print('******************************************************************')
                print('WARNING: bad matching offset, we are forced to use')
                print('         geometrical offset for swath mosaicking')
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


def computeSwathOffset(swath1, swath2, rangeScale1=1, azimuthScale1=1):

    rangeOffset = -(swath2.startingRange - swath1.startingRange) / swath1.rangePixelSize
    azimuthOffset = -((swath2.sensingStart - swath1.sensingStart).total_seconds()) / swath1.azimuthLineInterval

    rangeOffset /= rangeScale1
    azimuthOffset /= azimuthScale1

    return (rangeOffset, azimuthOffset)


def estimateSwathOffset(swath1, swath2, image1, image2, rangeScale1=1, azimuthScale1=1, rangeScale2=1, azimuthScale2=1, numberOfAzimuthLooks=10):
    '''
    estimate offset of two adjacent swaths using matching
    '''
    from osgeo import gdal
    import isceobj
    from contrib.alos2proc_f.alos2proc_f import rect_with_looks
    from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
    from isceobj.Alos2Proc.Alos2ProcPublic import cullOffsets
    from isceobj.Alos2Proc.Alos2ProcPublic import meanOffset
    from mroipac.ampcor.Ampcor import Ampcor

    
    #processing image 1
    rangeOff1 = int((swath2.startingRange - swath1.startingRange) / swath1.rangePixelSize)
    if rangeOff1 < 0:
        rangeOff1 = 0
    numberOfSamples1 = swath1.numberOfSamples - rangeOff1

    numberOfSamplesRect1 = int(numberOfSamples1/rangeScale1)
    numberOfLinesRect1 = int(swath1.numberOfLines/azimuthScale1)

    numberOfSamplesLook1 = int(numberOfSamplesRect1/1)
    numberOfLinesLook1 = int(numberOfLinesRect1/numberOfAzimuthLooks)

    #get magnitude image whether complex or not
    #ReadAsArray: https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html
    ds = gdal.Open(image1 + '.vrt', gdal.GA_ReadOnly)
    data = ds.ReadAsArray(rangeOff1, 0, numberOfSamples1, swath1.numberOfLines)
    ds = None
    (np.absolute(data)).astype(np.float32).tofile('image1.float')

    #rectify
    if rangeScale1 == 1 and azimuthScale1 == 1:
        os.rename('image1.float', 'image1_rect.float')
    else:
        rect_with_looks('image1.float',
                        'image1_rect.float',
                        numberOfSamples1, swath1.numberOfLines,
                        numberOfSamplesRect1, numberOfLinesRect1,
                        rangeScale1, 0.0,
                        0.0,azimuthScale1,
                        0.0,0.0,
                        1,1,
                        1,1,
                        'REAL',
                        'Bilinear')
        os.remove('image1.float')

    #take looks
    if numberOfAzimuthLooks == 1:
        os.rename('image1_rect.float', 'image1_look.float')
    else:
        data1 = np.fromfile('image1_rect.float', dtype=np.float32).reshape(numberOfLinesRect1, numberOfSamplesRect1)
        data1 = np.sqrt(multilook(data1**2, numberOfAzimuthLooks, 1))
        data1.astype(np.float32).tofile('image1_look.float')
        os.remove('image1_rect.float')
    create_xml('image1_look.float', numberOfSamplesLook1, numberOfLinesLook1, 'float')


    #processing image 2
    rangeOff2 = 0
    numberOfSamples2 = int((swath1.startingRange + swath1.rangePixelSize * (swath1.numberOfSamples - 1) - swath2.startingRange) / swath2.rangePixelSize) + 1
    if numberOfSamples2 > swath2.numberOfSamples:
        numberOfSamples2 = swath2.numberOfSamples

    numberOfSamplesRect2 = int(numberOfSamples2/rangeScale2)
    numberOfLinesRect2 = int(swath2.numberOfLines/azimuthScale2)

    numberOfSamplesLook2 = int(numberOfSamplesRect2/1)
    numberOfLinesLook2 = int(numberOfLinesRect2/numberOfAzimuthLooks)

    #get magnitude image whether complex or not
    ds = gdal.Open(image2 + '.vrt', gdal.GA_ReadOnly)
    data = ds.ReadAsArray(rangeOff2, 0, numberOfSamples2, swath2.numberOfLines)
    ds = None
    (np.absolute(data)).astype(np.float32).tofile('image2.float')

    #rectify
    if rangeScale2 == 1 and azimuthScale2 == 1:
        os.rename('image2.float', 'image2_rect.float')
    else:
        rect_with_looks('image2.float',
                        'image2_rect.float',
                        numberOfSamples2, swath2.numberOfLines,
                        numberOfSamplesRect2, numberOfLinesRect2,
                        rangeScale2, 0.0,
                        0.0,azimuthScale2,
                        0.0,0.0,
                        1,1,
                        1,1,
                        'REAL',
                        'Bilinear')
        os.remove('image2.float')

    #take looks
    if numberOfAzimuthLooks == 1:
        os.rename('image2_rect.float', 'image2_look.float')
    else:
        data2 = np.fromfile('image2_rect.float', dtype=np.float32).reshape(numberOfLinesRect2, numberOfSamplesRect2)
        data2 = np.sqrt(multilook(data2**2, numberOfAzimuthLooks, 1))
        data2.astype(np.float32).tofile('image2_look.float')
        os.remove('image2_rect.float')
    create_xml('image2_look.float', numberOfSamplesLook2, numberOfLinesLook2, 'float')


    #matching
    ampcor = Ampcor(name='insarapp_slcs_ampcor')
    ampcor.configure()

    mMag = isceobj.createImage()
    mMag.load('image1_look.float.xml')
    mMag.setAccessMode('read')
    mMag.createImage()

    sMag = isceobj.createImage()
    sMag.load('image2_look.float.xml')
    sMag.setAccessMode('read')
    sMag.createImage()

    ampcor.setImageDataType1('real')
    ampcor.setImageDataType2('real')

    ampcor.setReferenceSlcImage(mMag)
    ampcor.setSecondarySlcImage(sMag)

    #MATCH REGION
    rgoff = 0
    azoff = int((swath1.sensingStart - swath2.sensingStart).total_seconds() / swath1.azimuthLineInterval / azimuthScale1 / numberOfAzimuthLooks)
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
    ampcor.setLastSampleAcross(numberOfSamplesLook1)
    ampcor.setNumberLocationAcross(20)
    ampcor.setFirstSampleDown(firstLine)
    ampcor.setLastSampleDown(numberOfLinesLook1)
    ampcor.setNumberLocationDown(100)

    #MATCH PARAMETERS
    ampcor.setWindowSizeWidth(32)
    ampcor.setWindowSizeHeight(32)
    #note this is the half width/length of search area, so number of resulting correlation samples: 8*2+1
    ampcor.setSearchWindowSizeWidth(8)
    ampcor.setSearchWindowSizeHeight(8)

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
    refinedOffsets = cullOffsets(offsets)

    #finalize image, and re-create it
    #otherwise the file pointer is still at the end of the image
    mMag.finalizeImage()
    sMag.finalizeImage()

    os.remove('image1_look.float')
    os.remove('image1_look.float.vrt')
    os.remove('image1_look.float.xml')
    os.remove('image2_look.float')
    os.remove('image2_look.float.vrt')
    os.remove('image2_look.float.xml')

    if refinedOffsets != None:
        rangeOffset, azimuthOffset = meanOffset(refinedOffsets)
        rangeOffset   -= rangeOff1/rangeScale1
        azimuthOffset *= numberOfAzimuthLooks
        return (rangeOffset, azimuthOffset)
    else:
        return None




