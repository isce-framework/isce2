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
import mroipac
from mroipac.ampcor.Ampcor import Ampcor
from isceobj.Alos2Proc.Alos2ProcPublic import topo
from isceobj.Alos2Proc.Alos2ProcPublic import geo2rdr
from isceobj.Alos2Proc.Alos2ProcPublic import waterBodyRadar
from isceobj.Alos2Proc.Alos2ProcPublic import reformatGeometricalOffset
from isceobj.Alos2Proc.Alos2ProcPublic import writeOffset
from isceobj.Alos2Proc.Alos2ProcPublic import cullOffsets
from isceobj.Alos2Proc.Alos2ProcPublic import computeOffsetFromOrbit

logger = logging.getLogger('isce.alos2insar.runSlcOffset')

def runSlcOffset(self):
    '''estimate SLC offsets
    '''
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    referenceTrack = self._insar.loadTrack(reference=True)
    secondaryTrack = self._insar.loadTrack(reference=False)

    demFile = os.path.abspath(self._insar.dem)
    wbdFile = os.path.abspath(self._insar.wbd)

    for i, frameNumber in enumerate(self._insar.referenceFrames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        os.chdir(frameDir)
        for j, swathNumber in enumerate(range(self._insar.startingSwath, self._insar.endingSwath + 1)):
            swathDir = 's{}'.format(swathNumber)
            os.chdir(swathDir)

            print('estimating offset frame {}, swath {}'.format(frameNumber, swathNumber))

            referenceSwath = referenceTrack.frames[i].swaths[j]
            secondarySwath = secondaryTrack.frames[i].swaths[j]

            ##########################################
            #1. set number of matching points
            ##########################################
            #set initinial numbers
            if (self._insar.modeCombination == 21) or (self._insar.modeCombination == 22):
                numberOfOffsetsRange = 10
                numberOfOffsetsAzimuth = 40
            else:
                numberOfOffsetsRange = 20
                numberOfOffsetsAzimuth = 20

            #change the initial numbers using water body
            if self.useWbdForNumberOffsets and (self._insar.wbd != None):
                numberRangeLooks=100
                numberAzimuthLooks=100
                #compute land ratio using topo module
                topo(referenceSwath, referenceTrack, demFile, 'lat.rdr', 'lon.rdr', 'hgt.rdr', losFile='los.rdr', 
                    incFile=None, mskFile=None, 
                    numberRangeLooks=numberRangeLooks, numberAzimuthLooks=numberAzimuthLooks, multilookTimeOffset=False)
                waterBodyRadar('lat.rdr', 'lon.rdr', wbdFile, 'wbd.rdr')

                wbdImg = isceobj.createImage()
                wbdImg.load('wbd.rdr.xml')
                width = wbdImg.width
                length = wbdImg.length

                wbd = np.fromfile('wbd.rdr', dtype=np.byte).reshape(length, width)
                landRatio = np.sum(wbd==0) / (length*width)

                if (landRatio <= 0.00125):
                    print('\n\nWARNING: land too small for estimating slc offsets at frame {}, swath {}'.format(frameNumber, swathNumber))
                    print('proceed to use geometric offsets for forming interferogram')
                    print('but please consider not using this swath\n\n')
                    catalog.addItem('warning message', 'land too small for estimating slc offsets at frame {}, swath {}, use geometric offsets'.format(frameNumber, swathNumber), 'runSlcOffset')
                    
                    #compute geomtricla offsets
                    geo2rdr(secondarySwath, secondaryTrack, 'lat.rdr', 'lon.rdr', 'hgt.rdr', 'rg.rdr', 'az.rdr', numberRangeLooks=numberRangeLooks, numberAzimuthLooks=numberAzimuthLooks, multilookTimeOffset=False)
                    reformatGeometricalOffset('rg.rdr', 'az.rdr', 'cull.off', rangeStep=numberRangeLooks, azimuthStep=numberAzimuthLooks, maximumNumberOfOffsets=2000)

                    os.remove('lat.rdr')
                    os.remove('lat.rdr.vrt')
                    os.remove('lat.rdr.xml')
                    os.remove('lon.rdr')
                    os.remove('lon.rdr.vrt')
                    os.remove('lon.rdr.xml')
                    os.remove('hgt.rdr')
                    os.remove('hgt.rdr.vrt')
                    os.remove('hgt.rdr.xml')
                    os.remove('los.rdr')
                    os.remove('los.rdr.vrt')
                    os.remove('los.rdr.xml')
                    os.remove('wbd.rdr')
                    os.remove('wbd.rdr.vrt')
                    os.remove('wbd.rdr.xml')

                    os.remove('rg.rdr')
                    os.remove('rg.rdr.vrt')
                    os.remove('rg.rdr.xml')
                    os.remove('az.rdr')
                    os.remove('az.rdr.vrt')
                    os.remove('az.rdr.xml')

                    os.chdir('../')
                    continue


                os.remove('lat.rdr')
                os.remove('lat.rdr.vrt')
                os.remove('lat.rdr.xml')
                os.remove('lon.rdr')
                os.remove('lon.rdr.vrt')
                os.remove('lon.rdr.xml')
                os.remove('hgt.rdr')
                os.remove('hgt.rdr.vrt')
                os.remove('hgt.rdr.xml')
                os.remove('los.rdr')
                os.remove('los.rdr.vrt')
                os.remove('los.rdr.xml')
                os.remove('wbd.rdr')
                os.remove('wbd.rdr.vrt')
                os.remove('wbd.rdr.xml')

                #put the results on a grid with a specified interval
                interval = 0.2
                axisRatio = int(np.sqrt(landRatio)/interval)*interval + interval
                if axisRatio > 1:
                    axisRatio = 1

                numberOfOffsetsRange = int(numberOfOffsetsRange/axisRatio)
                numberOfOffsetsAzimuth = int(numberOfOffsetsAzimuth/axisRatio)
            else:
                catalog.addItem('warning message', 'no water mask used to determine number of matching points. frame {} swath {}'.format(frameNumber, swathNumber), 'runSlcOffset')

            #user's settings
            if self.numberRangeOffsets != None:
                numberOfOffsetsRange = self.numberRangeOffsets[i][j]
            if self.numberAzimuthOffsets != None:
                numberOfOffsetsAzimuth = self.numberAzimuthOffsets[i][j]

            catalog.addItem('number of offsets range frame {} swath {}'.format(frameNumber, swathNumber), numberOfOffsetsRange, 'runSlcOffset')
            catalog.addItem('number of offsets azimuth frame {} swath {}'.format(frameNumber, swathNumber), numberOfOffsetsAzimuth, 'runSlcOffset')

            ##########################################
            #2. match using ampcor
            ##########################################
            ampcor = Ampcor(name='insarapp_slcs_ampcor')
            ampcor.configure()

            mSLC = isceobj.createSlcImage()
            mSLC.load(self._insar.referenceSlc+'.xml')
            mSLC.setAccessMode('read')
            mSLC.createImage()

            sSLC = isceobj.createSlcImage()
            sSLC.load(self._insar.secondarySlc+'.xml')
            sSLC.setAccessMode('read')
            sSLC.createImage()

            ampcor.setImageDataType1('complex')
            ampcor.setImageDataType2('complex')

            ampcor.setReferenceSlcImage(mSLC)
            ampcor.setSecondarySlcImage(sSLC)

            #MATCH REGION
            #compute an offset at image center to use
            rgoff, azoff = computeOffsetFromOrbit(referenceSwath, referenceTrack, secondarySwath, secondaryTrack, 
                referenceSwath.numberOfSamples * 0.5, 
                referenceSwath.numberOfLines * 0.5)
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
            ampcor.setNumberLocationAcross(numberOfOffsetsRange)
            ampcor.setFirstSampleDown(firstLine)
            ampcor.setLastSampleDown(mSLC.length)
            ampcor.setNumberLocationDown(numberOfOffsetsAzimuth)

            #MATCH PARAMETERS
            #full-aperture mode
            if (self._insar.modeCombination == 21) or \
               (self._insar.modeCombination == 22) or \
               (self._insar.modeCombination == 31) or \
               (self._insar.modeCombination == 32):
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
            ampcorOffsetFile = 'ampcor.off'
            writeOffset(offsets, ampcorOffsetFile)

            #finalize image, and re-create it
            #otherwise the file pointer is still at the end of the image
            mSLC.finalizeImage()
            sSLC.finalizeImage()

            ##########################################
            #3. cull offsets
            ##########################################
            refinedOffsets = cullOffsets(offsets)
            if refinedOffsets == None:
                print('******************************************************************')
                print('WARNING: There are not enough offsets left, so we are forced to')
                print('         use offset without culling. frame {}, swath {}'.format(frameNumber, swathNumber))
                print('******************************************************************')
                catalog.addItem('warning message', 'not enough offsets left, use offset without culling. frame {} swath {}'.format(frameNumber, swathNumber), 'runSlcOffset')
                refinedOffsets = offsets

            cullOffsetFile = 'cull.off'
            writeOffset(refinedOffsets, cullOffsetFile)

            os.chdir('../')
        os.chdir('../')

    catalog.printToLog(logger, "runSlcOffset")
    self._insar.procDoc.addAllFromCatalog(catalog)
