#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging
import numpy as np

import isceobj
from isceobj.Alos2Proc.runRdr2Geo import topoCPU
from isceobj.Alos2Proc.runRdr2Geo import topoGPU
from isceobj.Alos2Proc.runGeo2Rdr import geo2RdrCPU
from isceobj.Alos2Proc.runGeo2Rdr import geo2RdrGPU
from contrib.alos2proc.alos2proc import resamp
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
from isceobj.Alos2Proc.Alos2ProcPublic import renameFile
from isceobj.Alos2Proc.Alos2ProcPublic import waterBodyRadar
from mroipac.ampcor.Ampcor import Ampcor
from isceobj.Alos2Proc.Alos2ProcPublic import meanOffset
from isceobj.Alos2Proc.Alos2ProcPublic import cullOffsetsRoipac

logger = logging.getLogger('isce.alos2insar.runSlcMatch')

def runSlcMatch(self):
    '''match a pair of SLCs
    '''
    if not self.doDenseOffset:
        return
    if not ((self._insar.modeCombination == 0) or (self._insar.modeCombination == 1)):
        return

    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    demFile = os.path.abspath(self._insar.dem)
    wbdFile = os.path.abspath(self._insar.wbd)

    denseOffsetDir = 'dense_offset'
    os.makedirs(denseOffsetDir, exist_ok=True)
    os.chdir(denseOffsetDir)

    referenceTrack = self._insar.loadProduct(self._insar.referenceTrackParameter)
    secondaryTrack = self._insar.loadProduct(self._insar.secondaryTrackParameter)

#########################################################################################


    ##################################################
    # compute geometric offsets
    ##################################################
    if self.useGPU and self._insar.hasGPU():
        topoGPU(referenceTrack, 1, 1, demFile, 
                       'lat.rdr', 'lon.rdr', 'hgt.rdr', 'los.rdr')
        geo2RdrGPU(secondaryTrack, 1, 1, 
            'lat.rdr', 'lon.rdr', 'hgt.rdr', 'rg.off', 'az.off')
    else:
        topoCPU(referenceTrack, 1, 1, demFile, 
                       'lat.rdr', 'lon.rdr', 'hgt.rdr', 'los.rdr')
        geo2RdrCPU(secondaryTrack, 1, 1, 
            'lat.rdr', 'lon.rdr', 'hgt.rdr', 'rg.off', 'az.off')


    ##################################################
    # resample SLC
    ##################################################
    #SecondarySlcResampled = os.path.splitext(self._insar.secondarySlc)[0]+'_resamp'+os.path.splitext(self._insar.secondarySlc)[1]
    SecondarySlcResampled = self._insar.secondarySlcCoregistered
    rangeOffsets2Frac = 0.0
    azimuthOffsets2Frac = 0.0
    resamp(self._insar.secondarySlc,
           SecondarySlcResampled,
           'rg.off',
           'az.off',
           referenceTrack.numberOfSamples, referenceTrack.numberOfLines,
           secondaryTrack.prf,
           secondaryTrack.dopplerVsPixel,
           [rangeOffsets2Frac, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [azimuthOffsets2Frac, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    create_xml(SecondarySlcResampled, referenceTrack.numberOfSamples, referenceTrack.numberOfLines, 'slc')


    if self.estimateResidualOffset:

        numberOfOffsets = 800
        rangeStep = 50

        length = referenceTrack.numberOfLines
        width = referenceTrack.numberOfSamples
        waterBodyRadar('lat.rdr', 'lon.rdr', wbdFile, 'wbd.rdr')
        wbd=np.memmap('wbd.rdr', dtype=np.int8, mode='r', shape=(length, width))
        azimuthStep = int(length/width*rangeStep+0.5)
        landRatio = np.sum(wbd[0:length:azimuthStep,0:width:rangeStep]!=-1)/(int(length/azimuthStep)*int(width/rangeStep))
        del wbd

        if (landRatio <= 0.00125):
            print('\n\nWARNING: land area too small for estimating residual slc offsets')
            print('do not estimate residual offsets\n\n')
            catalog.addItem('warning message', 'land area too small for estimating residual slc offsets', 'runSlcMatch')
        else:
            numberOfOffsets /= landRatio
            #we use equal number of offsets in range and azimuth here
            numberOfOffsetsRange = int(np.sqrt(numberOfOffsets)+0.5)
            numberOfOffsetsAzimuth = int(np.sqrt(numberOfOffsets)+0.5)
            if numberOfOffsetsRange > int(width/2):
                numberOfOffsetsRange = int(width/2)
            if numberOfOffsetsAzimuth > int(length/2):
                numberOfOffsetsAzimuth = int(length/2)
            if numberOfOffsetsRange < 10:
                numberOfOffsetsRange = 10
            if numberOfOffsetsAzimuth < 10:
                numberOfOffsetsAzimuth = 10


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
            sSLC.load(SecondarySlcResampled+'.xml')
            sSLC.setAccessMode('read')
            sSLC.createImage()

            ampcor.setImageDataType1('complex')
            ampcor.setImageDataType2('complex')

            ampcor.setReferenceSlcImage(mSLC)
            ampcor.setSecondarySlcImage(sSLC)

            #MATCH REGION
            #compute an offset at image center to use
            rgoff = 0.0
            azoff = 0.0
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
                ampcor.setSearchWindowSizeWidth(16)
                ampcor.setSearchWindowSizeHeight(16)

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
            mSLC.finalizeImage()
            sSLC.finalizeImage()


            #3. cull offsets
            refinedOffsets = cullOffsetsRoipac(offsets, numThreshold=50)
            if refinedOffsets == None:
                print('\n\nWARNING: too few offsets left for slc residual offset estimation')
                print('do not estimate residual offsets\n\n')
                catalog.addItem('warning message', 'too few offsets left for slc residual offset estimation', 'runSlcMatch')
            else:
                rangeOffset, azimuthOffset = meanOffset(refinedOffsets)
                os.remove(SecondarySlcResampled)
                os.remove(SecondarySlcResampled+'.vrt')
                os.remove(SecondarySlcResampled+'.xml')
                
                rangeOffsets2Frac = rangeOffset
                azimuthOffsets2Frac = azimuthOffset
                resamp(self._insar.secondarySlc,
                       SecondarySlcResampled,
                       'rg.off',
                       'az.off',
                       referenceTrack.numberOfSamples, referenceTrack.numberOfLines,
                       secondaryTrack.prf,
                       secondaryTrack.dopplerVsPixel,
                       [rangeOffsets2Frac, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [azimuthOffsets2Frac, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                create_xml(SecondarySlcResampled, referenceTrack.numberOfSamples, referenceTrack.numberOfLines, 'slc')

                catalog.addItem('number of offsets range', numberOfOffsetsRange, 'runSlcMatch')
                catalog.addItem('number of offsets azimuth', numberOfOffsetsAzimuth, 'runSlcMatch')
                catalog.addItem('range residual offset after geometric coregistration', rangeOffset, 'runSlcMatch')
                catalog.addItem('azimuth residual offset after geometric coregistration', azimuthOffset, 'runSlcMatch')




    if self.deleteGeometryFiles:
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
        # if os.path.isfile('wbd.rdr'):
        #     os.remove('wbd.rdr')
        #     os.remove('wbd.rdr.vrt')
        #     os.remove('wbd.rdr.xml')

#########################################################################################

    os.chdir('../')
    catalog.printToLog(logger, "runSlcMatch")
    self._insar.procDoc.addAllFromCatalog(catalog)


