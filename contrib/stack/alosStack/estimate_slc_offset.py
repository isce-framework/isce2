#!/usr/bin/env python3

#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import glob
import datetime
import numpy as np

import isce, isceobj
import mroipac
from mroipac.ampcor.Ampcor import Ampcor
from isceobj.Alos2Proc.Alos2ProcPublic import topo
from isceobj.Alos2Proc.Alos2ProcPublic import geo2rdr
from isceobj.Alos2Proc.Alos2ProcPublic import waterBodyRadar
from isceobj.Alos2Proc.Alos2ProcPublic import reformatGeometricalOffset
from isceobj.Alos2Proc.Alos2ProcPublic import writeOffset
from isceobj.Alos2Proc.Alos2ProcPublic import cullOffsets
from isceobj.Alos2Proc.Alos2ProcPublic import computeOffsetFromOrbit

from StackPulic import loadTrack
from StackPulic import stackDateStatistics
from StackPulic import acquisitionModesAlos2


def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='estimate offset between a pair of SLCs for a number of dates')
    parser.add_argument('-idir', dest='idir', type=str, required=True,
            help = 'input directory where data of each date (YYMMDD) is located. only folders are recognized')
    parser.add_argument('-ref_date', dest='ref_date', type=str, required=True,
            help = 'reference date. format: YYMMDD')
    parser.add_argument('-sec_date', dest='sec_date', type=str, nargs='+', default=[],
            help = 'a number of secondary dates seperated by blanks. format: YYMMDD YYMMDD YYMMDD. If provided, only estimate offsets of these dates')
    parser.add_argument('-wbd', dest='wbd', type=str, default=None,
            help = 'water body used to determine number of offsets in range and azimuth')
    parser.add_argument('-dem', dest='dem', type=str, default=None,
            help = 'if water body is provided, dem file must also be provided')
    parser.add_argument('-use_wbd_offset', dest='use_wbd_offset', action='store_true', default=False,
            help='use water body to dertermine number of matching offsets')
    parser.add_argument('-num_rg_offset', dest='num_rg_offset', type=int, nargs='+', action='append', default=[],
            help = 'number of offsets in range. format (e.g. 2 frames, 3 swaths): -num_rg_offset 11 12 13 -num_rg_offset 14 15 16')
    parser.add_argument('-num_az_offset', dest='num_az_offset', type=int, nargs='+', action='append', default=[],
            help = 'number of offsets in azimuth. format (e.g. 2 frames, 3 swaths): -num_az_offset 11 12 13 -num_az_offset 14 15 16')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()


    #get user parameters from input
    idir = inps.idir
    dateReference = inps.ref_date
    dateSecondary = inps.sec_date
    wbd = inps.wbd
    dem = inps.dem
    useWbdForNumberOffsets = inps.use_wbd_offset
    numberOfOffsetsRangeInput = inps.num_rg_offset
    numberOfOffsetsAzimuthInput = inps.num_az_offset


    if wbd is not None:
        wbdFile = os.path.abspath(wbd)
    else:
        wbdFile = None
    if dem is not None:
        demFile = os.path.abspath(dem)
    else:
        demFile = None
    #######################################################


    spotlightModes, stripmapModes, scansarNominalModes, scansarWideModes, scansarModes = acquisitionModesAlos2()


    warningMessage = ''


    #get date statistics
    dateDirs,   dates,   frames,   swaths,   dateIndexReference = stackDateStatistics(idir, dateReference)
    ndate = len(dates)
    nframe = len(frames)
    nswath = len(swaths)


    #load reference track
    referenceTrack = loadTrack(dateDirs[dateIndexReference], dates[dateIndexReference])

    dateSecondaryFirst = None
    for idate in range(ndate):
        if idate == dateIndexReference:
            continue
        if dateSecondary != []:
            if dates[idate] not in dateSecondary:
                continue
        dateSecondaryFirst = dates[idate]
        break
    if dateSecondaryFirst is None:
        raise Exception('no secondary date is to be processed\n')

    #set number of matching points
    numberOfOffsetsRangeUsed = [[None for j in range(nswath)] for i in range(nframe)]
    numberOfOffsetsAzimuthUsed = [[None for j in range(nswath)] for i in range(nframe)]
    for i, frameNumber in enumerate(frames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        for j, swathNumber in enumerate(range(swaths[0], swaths[-1] + 1)):
            swathDir = 's{}'.format(swathNumber)

            print('determine number of range/azimuth offsets frame {}, swath {}'.format(frameNumber, swathNumber))
            referenceSwath = referenceTrack.frames[i].swaths[j]

            #1. set initinial numbers
            #in case there are long time span pairs that have bad coherence
            ratio = np.sqrt(1.5)
            if referenceTrack.operationMode in scansarModes:
                numberOfOffsetsRange = int(10*ratio+0.5)
                numberOfOffsetsAzimuth = int(40*ratio+0.5)
            else:
                numberOfOffsetsRange = int(20*ratio+0.5)
                numberOfOffsetsAzimuth = int(20*ratio+0.5)

            #2. change the initial numbers using water body
            if useWbdForNumberOffsets and (wbdFile is not None) and (demFile is not None):
                numberRangeLooks=100
                numberAzimuthLooks=100

                #compute land ratio using topo module
                # latFile = 'lat_f{}_{}_s{}.rdr'.format(i+1, frameNumber, swathNumber)
                # lonFile = 'lon_f{}_{}_s{}.rdr'.format(i+1, frameNumber, swathNumber)
                # hgtFile = 'hgt_f{}_{}_s{}.rdr'.format(i+1, frameNumber, swathNumber)
                # losFile = 'los_f{}_{}_s{}.rdr'.format(i+1, frameNumber, swathNumber)
                # wbdRadarFile = 'wbd_f{}_{}_s{}.rdr'.format(i+1, frameNumber, swathNumber)

                latFile = os.path.join(idir, dateSecondaryFirst, frameDir, swathDir, 'lat.rdr')
                lonFile = os.path.join(idir, dateSecondaryFirst, frameDir, swathDir, 'lon.rdr')
                hgtFile = os.path.join(idir, dateSecondaryFirst, frameDir, swathDir, 'hgt.rdr')
                losFile = os.path.join(idir, dateSecondaryFirst, frameDir, swathDir, 'los.rdr')
                wbdRadarFile = os.path.join(idir, dateSecondaryFirst, frameDir, swathDir, 'wbd.rdr')

                topo(referenceSwath, referenceTrack, demFile, latFile, lonFile, hgtFile, losFile=losFile, 
                    incFile=None, mskFile=None, 
                    numberRangeLooks=numberRangeLooks, numberAzimuthLooks=numberAzimuthLooks, multilookTimeOffset=False)
                waterBodyRadar(latFile, lonFile, wbdFile, wbdRadarFile)

                wbdImg = isceobj.createImage()
                wbdImg.load(wbdRadarFile+'.xml')
                width = wbdImg.width
                length = wbdImg.length

                wbd = np.fromfile(wbdRadarFile, dtype=np.byte).reshape(length, width)
                landRatio = np.sum(wbd==0) / (length*width)

                if (landRatio <= 0.00125):
                    print('\n\nWARNING: land too small for estimating slc offsets at frame {}, swath {}'.format(frameNumber, swathNumber))
                    print('proceed to use geometric offsets for forming interferogram')
                    print('but please consider not using this swath\n\n')
                    warningMessage += 'land too small for estimating slc offsets at frame {}, swath {}, use geometric offsets\n'.format(frameNumber, swathNumber)

                    numberOfOffsetsRange = 0
                    numberOfOffsetsAzimuth = 0
                else:
                    #put the results on a grid with a specified interval
                    interval = 0.2
                    axisRatio = int(np.sqrt(landRatio)/interval)*interval + interval
                    if axisRatio > 1:
                        axisRatio = 1

                    numberOfOffsetsRange = int(numberOfOffsetsRange/axisRatio)
                    numberOfOffsetsAzimuth = int(numberOfOffsetsAzimuth/axisRatio)
            else:
                warningMessage += 'no water mask used to determine number of matching points. frame {} swath {}\n'.format(frameNumber, swathNumber)

            #3. user's settings
            if numberOfOffsetsRangeInput != []:
                numberOfOffsetsRange = numberOfOffsetsRangeInput[i][j]
            if numberOfOffsetsAzimuthInput != []:
                numberOfOffsetsAzimuth = numberOfOffsetsAzimuthInput[i][j]

            #4. save final results
            numberOfOffsetsRangeUsed[i][j] = numberOfOffsetsRange
            numberOfOffsetsAzimuthUsed[i][j] = numberOfOffsetsAzimuth


    #estimate offsets
    for idate in range(ndate):
        if idate == dateIndexReference:
            continue
        if dateSecondary != []:
            if dates[idate] not in dateSecondary:
                continue

        secondaryTrack = loadTrack(dateDirs[idate], dates[idate])

        for i, frameNumber in enumerate(frames):
            frameDir = 'f{}_{}'.format(i+1, frameNumber)
            for j, swathNumber in enumerate(range(swaths[0], swaths[-1] + 1)):
                swathDir = 's{}'.format(swathNumber)

                print('estimating offset frame {}, swath {}'.format(frameNumber, swathNumber))
                referenceDir = os.path.join(dateDirs[dateIndexReference], frameDir, swathDir)
                secondaryDir = os.path.join(dateDirs[idate], frameDir, swathDir)
                referenceSwath = referenceTrack.frames[i].swaths[j]
                secondarySwath = secondaryTrack.frames[i].swaths[j]

                #compute geometrical offsets
                if (wbdFile is not None) and (demFile is not None) and (numberOfOffsetsRangeUsed[i][j] == 0) and (numberOfOffsetsAzimuthUsed[i][j] == 0):
                    #compute geomtricla offsets
                    # latFile = 'lat_f{}_{}_s{}.rdr'.format(i+1, frameNumber, swathNumber)
                    # lonFile = 'lon_f{}_{}_s{}.rdr'.format(i+1, frameNumber, swathNumber)
                    # hgtFile = 'hgt_f{}_{}_s{}.rdr'.format(i+1, frameNumber, swathNumber)
                    # losFile = 'los_f{}_{}_s{}.rdr'.format(i+1, frameNumber, swathNumber)
                    # rgOffsetFile = 'rg_offset_f{}_{}_s{}.rdr'.format(i+1, frameNumber, swathNumber)
                    # azOffsetFile = 'az_offset_f{}_{}_s{}.rdr'.format(i+1, frameNumber, swathNumber)
                    # wbdRadarFile = 'wbd_f{}_{}_s{}.rdr'.format(i+1, frameNumber, swathNumber)

                    latFile = os.path.join(idir, dateSecondaryFirst, frameDir, swathDir, 'lat.rdr')
                    lonFile = os.path.join(idir, dateSecondaryFirst, frameDir, swathDir, 'lon.rdr')
                    hgtFile = os.path.join(idir, dateSecondaryFirst, frameDir, swathDir, 'hgt.rdr')
                    losFile = os.path.join(idir, dateSecondaryFirst, frameDir, swathDir, 'los.rdr')
                    #put them in current date directory
                    rgOffsetFile = os.path.join(idir, dates[idate], frameDir, swathDir, 'rg_offset.rdr')
                    azOffsetFile = os.path.join(idir, dates[idate], frameDir, swathDir, 'az_offset.rdr')
                    wbdRadarFile = os.path.join(idir, dateSecondaryFirst, frameDir, swathDir, 'wbd.rdr')

                    geo2rdr(secondarySwath, secondaryTrack, latFile, lonFile, hgtFile, rgOffsetFile, azOffsetFile, numberRangeLooks=numberRangeLooks, numberAzimuthLooks=numberAzimuthLooks, multilookTimeOffset=False)
                    reformatGeometricalOffset(rgOffsetFile, azOffsetFile, os.path.join(secondaryDir, 'cull.off'), rangeStep=numberRangeLooks, azimuthStep=numberAzimuthLooks, maximumNumberOfOffsets=2000)

                    os.remove(rgOffsetFile)
                    os.remove(rgOffsetFile+'.vrt')
                    os.remove(rgOffsetFile+'.xml')
                    os.remove(azOffsetFile)
                    os.remove(azOffsetFile+'.vrt')
                    os.remove(azOffsetFile+'.xml')
                #estimate offsets using ampcor
                else:
                    ampcor = Ampcor(name='insarapp_slcs_ampcor')
                    ampcor.configure()

                    mSLC = isceobj.createSlcImage()
                    mSLC.load(os.path.join(referenceDir, dates[dateIndexReference]+'.slc.xml'))
                    mSLC.filename = os.path.join(referenceDir, dates[dateIndexReference]+'.slc')
                    mSLC.extraFilename = os.path.join(referenceDir, dates[dateIndexReference]+'.slc.vrt')
                    mSLC.setAccessMode('read')
                    mSLC.createImage()

                    sSLC = isceobj.createSlcImage()
                    sSLC.load(os.path.join(secondaryDir, dates[idate]+'.slc.xml'))
                    sSLC.filename = os.path.join(secondaryDir, dates[idate]+'.slc')
                    sSLC.extraFilename = os.path.join(secondaryDir, dates[idate]+'.slc.vrt')
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
                    ampcor.setNumberLocationAcross(numberOfOffsetsRangeUsed[i][j])
                    ampcor.setFirstSampleDown(firstLine)
                    ampcor.setLastSampleDown(mSLC.length)
                    ampcor.setNumberLocationDown(numberOfOffsetsAzimuthUsed[i][j])

                    #MATCH PARAMETERS
                    #full-aperture mode
                    if referenceTrack.operationMode in scansarModes:
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
                    ampcorOffsetFile = os.path.join(secondaryDir, 'ampcor.off')
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
                        warningMessage += 'not enough offsets left, use offset without culling. frame {} swath {}'.format(frameNumber, swathNumber)
                        refinedOffsets = offsets

                    cullOffsetFile = os.path.join(secondaryDir, 'cull.off')
                    writeOffset(refinedOffsets, cullOffsetFile)

                #os.chdir('../')
            #os.chdir('../')


    #delete geometry files
    for i, frameNumber in enumerate(frames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        for j, swathNumber in enumerate(range(swaths[0], swaths[-1] + 1)):
            swathDir = 's{}'.format(swathNumber)

            if (wbdFile is not None) and (demFile is not None):
                # latFile = 'lat_f{}_{}_s{}.rdr'.format(i+1, frameNumber, swathNumber)
                # lonFile = 'lon_f{}_{}_s{}.rdr'.format(i+1, frameNumber, swathNumber)
                # hgtFile = 'hgt_f{}_{}_s{}.rdr'.format(i+1, frameNumber, swathNumber)
                # losFile = 'los_f{}_{}_s{}.rdr'.format(i+1, frameNumber, swathNumber)
                # wbdRadarFile = 'wbd_f{}_{}_s{}.rdr'.format(i+1, frameNumber, swathNumber)

                latFile = os.path.join(idir, dateSecondaryFirst, frameDir, swathDir, 'lat.rdr')
                lonFile = os.path.join(idir, dateSecondaryFirst, frameDir, swathDir, 'lon.rdr')
                hgtFile = os.path.join(idir, dateSecondaryFirst, frameDir, swathDir, 'hgt.rdr')
                losFile = os.path.join(idir, dateSecondaryFirst, frameDir, swathDir, 'los.rdr')
                wbdRadarFile = os.path.join(idir, dateSecondaryFirst, frameDir, swathDir, 'wbd.rdr')

                os.remove(latFile)
                os.remove(latFile+'.vrt')
                os.remove(latFile+'.xml')

                os.remove(lonFile)
                os.remove(lonFile+'.vrt')
                os.remove(lonFile+'.xml')

                os.remove(hgtFile)
                os.remove(hgtFile+'.vrt')
                os.remove(hgtFile+'.xml')

                os.remove(losFile)
                os.remove(losFile+'.vrt')
                os.remove(losFile+'.xml')

                os.remove(wbdRadarFile)
                os.remove(wbdRadarFile+'.vrt')
                os.remove(wbdRadarFile+'.xml')


    numberOfOffsetsUsedTxt  = '\nnumber of offsets in cross correlation:\n'
    numberOfOffsetsUsedTxt += '  frame      swath      range      azimuth\n'
    numberOfOffsetsUsedTxt += '============================================\n'
    for i, frameNumber in enumerate(frames):
        frameDir = 'f{}_{}'.format(i+1, frameNumber)
        for j, swathNumber in enumerate(range(swaths[0], swaths[-1] + 1)):
            swathDir = 's{}'.format(swathNumber)
            numberOfOffsetsUsedTxt += '   {}        {}          {}          {}\n'.format(frameNumber, swathNumber, numberOfOffsetsRangeUsed[i][j], numberOfOffsetsAzimuthUsed[i][j])
    print(numberOfOffsetsUsedTxt)

    if warningMessage != '':
        print('\n'+warningMessage+'\n')
