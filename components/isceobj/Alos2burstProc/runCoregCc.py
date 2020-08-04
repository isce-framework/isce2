#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import copy
import shutil
import logging
import numpy as np

import isceobj
from mroipac.ampcor.Ampcor import Ampcor
from isceobj.Alos2Proc.Alos2ProcPublic import cullOffsetsRoipac
from isceobj.Alos2Proc.Alos2ProcPublic import meanOffset
from isceobj.Alos2Proc.Alos2ProcPublic import resampleBursts
from isceobj.Alos2Proc.Alos2ProcPublic import mosaicBurstAmplitude
from isceobj.Alos2Proc.Alos2ProcPublic import mosaicBurstInterferogram
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml

logger = logging.getLogger('isce.alos2burstinsar.runCoregCc')

def runCoregCc(self):
    '''coregister bursts by cross correlation
    '''
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    referenceTrack = self._insar.loadTrack(reference=True)
    secondaryTrack = self._insar.loadTrack(reference=False)

    #demFile = os.path.abspath(self._insar.dem)
    #wbdFile = os.path.abspath(self._insar.wbd)
###############################################################################
    self._insar.rangeResidualOffsetCc = [[] for i in range(len(referenceTrack.frames))]
    self._insar.azimuthResidualOffsetCc = [[] for i in range(len(referenceTrack.frames))]
    for i, frameNumber in enumerate(self._insar.referenceFrames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        os.chdir(frameDir)
        for j, swathNumber in enumerate(range(self._insar.startingSwath, self._insar.endingSwath + 1)):
            swathDir = 's{}'.format(swathNumber)
            os.chdir(swathDir)

            print('processing frame {}, swath {}'.format(frameNumber, swathNumber))

            referenceSwath = referenceTrack.frames[i].swaths[j]
            secondarySwath = secondaryTrack.frames[i].swaths[j]

            ##################################################
            # estimate cross-correlation offsets
            ##################################################
            #compute number of offsets to use
            wbdImg = isceobj.createImage()
            wbdImg.load(self._insar.wbdOut+'.xml')
            width = wbdImg.width
            length = wbdImg.length

            #initial number of offsets to use
            numberOfOffsets = 800

            #compute land ratio to further determine the number of offsets to use
            if self.useWbdForNumberOffsets:
                wbd=np.memmap(self._insar.wbdOut, dtype='byte', mode='r', shape=(length, width))
                landRatio = np.sum(wbd==0) / length / width
                del wbd
                if (landRatio <= 0.00125):
                    print('\n\nWARNING: land area too small for estimating offsets between reference and secondary magnitudes at frame {}, swath {}'.format(frameNumber, swathNumber))
                    print('set offsets to zero\n\n')
                    self._insar.rangeResidualOffsetCc[i].append(0.0)
                    self._insar.azimuthResidualOffsetCc[i].append(0.0)
                    catalog.addItem('warning message', 'land area too small for estimating offsets between reference and secondary magnitudes at frame {}, swath {}'.format(frameNumber, swathNumber), 'runCoregCc')
                    continue
                #total number of offsets to use
                numberOfOffsets /= landRatio

            #allocate number of offsets in range/azimuth according to image width/length
            #number of offsets to use in range/azimuth
            numberOfOffsetsRange = int(np.sqrt(numberOfOffsets * width / length))
            numberOfOffsetsAzimuth = int(length / width * np.sqrt(numberOfOffsets * width / length))

            #this should be better?
            numberOfOffsetsRange = int(np.sqrt(numberOfOffsets))
            numberOfOffsetsAzimuth = int(np.sqrt(numberOfOffsets))

            if numberOfOffsetsRange > int(width/2):
                numberOfOffsetsRange = int(width/2)
            if numberOfOffsetsAzimuth > int(length/2):
                numberOfOffsetsAzimuth = int(length/2)

            if numberOfOffsetsRange < 10:
                numberOfOffsetsRange = 10
            if numberOfOffsetsAzimuth < 10:
                numberOfOffsetsAzimuth = 10

            #user's settings
            if self.numberRangeOffsets != None:
                numberOfOffsetsRange = self.numberRangeOffsets[i][j]
            if self.numberAzimuthOffsets != None:
                numberOfOffsetsAzimuth = self.numberAzimuthOffsets[i][j]

            catalog.addItem('number of range offsets at frame {}, swath {}'.format(frameNumber, swathNumber), '{}'.format(numberOfOffsetsRange), 'runCoregCc')
            catalog.addItem('number of azimuth offsets at frame {}, swath {}'.format(frameNumber, swathNumber), '{}'.format(numberOfOffsetsAzimuth), 'runCoregCc')

            #need to cp to current directory to make it (gdal) work
            if not os.path.isfile(self._insar.referenceMagnitude):
                os.symlink(os.path.join(self._insar.referenceBurstPrefix, self._insar.referenceMagnitude), self._insar.referenceMagnitude)
            #shutil.copy2() can overwrite
            shutil.copy2(os.path.join(self._insar.referenceBurstPrefix, self._insar.referenceMagnitude+'.vrt'), self._insar.referenceMagnitude+'.vrt')
            shutil.copy2(os.path.join(self._insar.referenceBurstPrefix, self._insar.referenceMagnitude+'.xml'), self._insar.referenceMagnitude+'.xml')

            if not os.path.isfile(self._insar.secondaryMagnitude):
                os.symlink(os.path.join(self._insar.secondaryBurstPrefix + '_1_coreg_geom', self._insar.secondaryMagnitude), self._insar.secondaryMagnitude)
            #shutil.copy2() can overwrite
            shutil.copy2(os.path.join(self._insar.secondaryBurstPrefix + '_1_coreg_geom', self._insar.secondaryMagnitude+'.vrt'), self._insar.secondaryMagnitude+'.vrt')
            shutil.copy2(os.path.join(self._insar.secondaryBurstPrefix + '_1_coreg_geom', self._insar.secondaryMagnitude+'.xml'), self._insar.secondaryMagnitude+'.xml')

            #matching
            ampcor = Ampcor(name='insarapp_slcs_ampcor')
            ampcor.configure()

            mMag = isceobj.createImage()
            mMag.load(self._insar.referenceMagnitude+'.xml')
            mMag.setAccessMode('read')
            mMag.createImage()

            sMag = isceobj.createImage()
            sMag.load(self._insar.secondaryMagnitude+'.xml')
            sMag.setAccessMode('read')
            sMag.createImage()

            ampcor.setImageDataType1('real')
            ampcor.setImageDataType2('real')

            ampcor.setReferenceSlcImage(mMag)
            ampcor.setSecondarySlcImage(sMag)

            #MATCH REGION
            rgoff = 0
            azoff = 0
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
            ampcor.setLastSampleAcross(mMag.width)
            ampcor.setNumberLocationAcross(numberOfOffsetsRange)
            ampcor.setFirstSampleDown(firstLine)
            ampcor.setLastSampleDown(mMag.length)
            ampcor.setNumberLocationDown(numberOfOffsetsAzimuth)

            #MATCH PARAMETERS
            ampcor.setWindowSizeWidth(64)
            ampcor.setWindowSizeHeight(64)
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
            refinedOffsets = cullOffsetsRoipac(offsets, numThreshold=50)

            #finalize image, and re-create it
            #otherwise the file pointer is still at the end of the image
            mMag.finalizeImage()
            sMag.finalizeImage()

            #clear up
            os.remove(self._insar.referenceMagnitude)
            os.remove(self._insar.referenceMagnitude+'.vrt')
            os.remove(self._insar.referenceMagnitude+'.xml')
            os.remove(self._insar.secondaryMagnitude)
            os.remove(self._insar.secondaryMagnitude+'.vrt')
            os.remove(self._insar.secondaryMagnitude+'.xml')

            #compute average offsets to use in resampling
            if refinedOffsets == None:
                rangeOffset = 0
                azimuthOffset = 0
                self._insar.rangeResidualOffsetCc[i].append(rangeOffset)
                self._insar.azimuthResidualOffsetCc[i].append(azimuthOffset)
                print('\n\nWARNING: too few offsets left in matching reference and secondary magnitudes at frame {}, swath {}'.format(frameNumber, swathNumber))
                print('set offsets to zero\n\n')
                catalog.addItem('warning message', 'too few offsets left in matching reference and secondary magnitudes at frame {}, swath {}'.format(frameNumber, swathNumber), 'runCoregCc')
            else:
                rangeOffset, azimuthOffset = meanOffset(refinedOffsets)
                #for range offset, need to compute from a polynomial
                #see components/isceobj/Location/Offset.py and components/isceobj/Util/Library/python/Poly2D.py for definations
                (azimuthPoly, rangePoly) = refinedOffsets.getFitPolynomials(rangeOrder=2,azimuthOrder=2)
                #make a deep copy, otherwise it also changes original coefficient list of rangePoly, which affects following rangePoly(*, *) computation
                polyCoeff = copy.deepcopy(rangePoly.getCoeffs())
                rgIndex = (np.arange(width)-rangePoly.getMeanRange())/rangePoly.getNormRange()
                azIndex = (np.arange(length)-rangePoly.getMeanAzimuth())/rangePoly.getNormAzimuth()
                rangeOffset =  polyCoeff[0][0] + polyCoeff[0][1]*rgIndex[None,:] + polyCoeff[0][2]*rgIndex[None,:]**2 + \
                              (polyCoeff[1][0] + polyCoeff[1][1]*rgIndex[None,:]) * azIndex[:, None] + \
                               polyCoeff[2][0] * azIndex[:, None]**2
                polyCoeff.append([rangePoly.getMeanRange(), rangePoly.getNormRange(), rangePoly.getMeanAzimuth(), rangePoly.getNormAzimuth()])
                self._insar.rangeResidualOffsetCc[i].append(polyCoeff)
                self._insar.azimuthResidualOffsetCc[i].append(azimuthOffset)

                catalog.addItem('range residual offset at {} {} at frame {}, swath {}'.format(0, 0, frameNumber, swathNumber), 
                    '{}'.format(rangePoly(0, 0)), 'runCoregCc')
                catalog.addItem('range residual offset at {} {} at frame {}, swath {}'.format(0, width-1, frameNumber, swathNumber), 
                    '{}'.format(rangePoly(0, width-1)), 'runCoregCc')
                catalog.addItem('range residual offset at {} {} at frame {}, swath {}'.format(length-1, 0, frameNumber, swathNumber), 
                    '{}'.format(rangePoly(length-1, 0)), 'runCoregCc')
                catalog.addItem('range residual offset at {} {} at frame {}, swath {}'.format(length-1,width-1, frameNumber, swathNumber), 
                    '{}'.format(rangePoly(length-1,width-1)), 'runCoregCc')
                catalog.addItem('azimuth residual offset at frame {}, swath {}'.format(frameNumber, swathNumber), 
                    '{}'.format(azimuthOffset), 'runCoregCc')

                DEBUG=False
                if DEBUG:
                    print('+++++++++++++++++++++++++++++')
                    print(rangeOffset[0,0], rangePoly(0, 0))
                    print(rangeOffset[0,width-1], rangePoly(0, width-1))
                    print(rangeOffset[length-1,0], rangePoly(length-1, 0))
                    print(rangeOffset[length-1,width-1], rangePoly(length-1,width-1))
                    print(rangeOffset[int((length-1)/2),int((width-1)/2)], rangePoly(int((length-1)/2),int((width-1)/2)))
                    print('+++++++++++++++++++++++++++++')
                

            ##################################################
            # resample bursts
            ##################################################
            secondaryBurstResampledDir = self._insar.secondaryBurstPrefix + '_2_coreg_cc'
            #interferogramDir = self._insar.referenceBurstPrefix + '-' + self._insar.secondaryBurstPrefix + '_coreg_geom'
            interferogramDir = 'burst_interf_2_coreg_cc'
            interferogramPrefix = self._insar.referenceBurstPrefix + '-' + self._insar.secondaryBurstPrefix
            resampleBursts(referenceSwath, secondarySwath, 
                self._insar.referenceBurstPrefix, self._insar.secondaryBurstPrefix, secondaryBurstResampledDir, interferogramDir,
                self._insar.referenceBurstPrefix, self._insar.secondaryBurstPrefix, self._insar.secondaryBurstPrefix, interferogramPrefix, 
                self._insar.rangeOffset, self._insar.azimuthOffset, rangeOffsetResidual=rangeOffset, azimuthOffsetResidual=azimuthOffset)


            ##################################################
            # mosaic burst amplitudes and interferograms
            ##################################################
            os.chdir(secondaryBurstResampledDir)
            mosaicBurstAmplitude(referenceSwath, self._insar.secondaryBurstPrefix, self._insar.secondaryMagnitude, numberOfLooksThreshold=4)
            os.chdir('../')

            os.chdir(interferogramDir)
            mosaicBurstInterferogram(referenceSwath, interferogramPrefix, self._insar.interferogram, numberOfLooksThreshold=4)
            os.chdir('../')


            ##################################################
            # final amplitude and interferogram
            ##################################################
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

            os.chdir('../')
        os.chdir('../')

###############################################################################
    catalog.printToLog(logger, "runCoregCc")
    self._insar.procDoc.addAllFromCatalog(catalog)



