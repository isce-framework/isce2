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
import isceobj.Sensor.MultiMode as MultiMode

from StackPulic import saveProduct
from StackPulic import acquisitionModesAlos2


def getAlos2StackDirs(dataDir):
    '''
    1. this function takes the data directory containing a list of folders, in each of
    which data of a date is located, and then returns a list of date directory sorted
    by acquisition date.

    2. under dataDir, only folders are recognized
    '''
    import os
    import glob

    def sorter(item):
        #return date
        return item.split('-')[-2]

    #get only folders in dataDir
    dateDirs = sorted(glob.glob(os.path.join(dataDir, '*')))
    dateDirs = [x for x in dateDirs if os.path.isdir(x)]
    ndate = len(dateDirs)

    #get first LED files in dateDirs
    dateFirstleaderFiles = [sorted(glob.glob(os.path.join(x, 'LED-ALOS2*-*-*')))[0] for x in dateDirs]
    #sort first LED files using date in LED file name
    dateFirstleaderFiles = sorted(dateFirstleaderFiles, key=sorter)
    #keep only directory from the path
    dateDirs = [os.path.dirname(x) for x in dateFirstleaderFiles]

    return dateDirs


def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='read a number of dates of data')
    parser.add_argument('-idir', dest='idir', type=str, required=True,
            help = 'input directory where data of each date is located. only folders are recognized')
    parser.add_argument('-odir', dest='odir', type=str, required=True,
            help = 'output directory where data of each date is output')
    parser.add_argument('-ref_date', dest='ref_date', type=str, required=True,
            help = 'reference date. format: YYMMDD')
    parser.add_argument('-sec_date', dest='sec_date', type=str, nargs='+', default=[],
            help = 'a number of secondary dates seperated by blanks, can also include reference date. format: YYMMDD YYMMDD YYMMDD. If provided, only read data of these dates')
    parser.add_argument('-pol', dest='pol', type=str, default='HH',
            help = 'polarization to process, default: HH')
    parser.add_argument('-frames', dest='frames', type=str, nargs='+', default=None,
            help = 'frames to process, must specify frame numbers of reference if frames are different among dates. e.g. -frames 2800 2850')
    parser.add_argument('-starting_swath', dest='starting_swath', type=int, default=None,
            help = 'starting swath to process.')
    parser.add_argument('-ending_swath', dest='ending_swath', type=int, default=None,
            help = 'starting swath to process')
    parser.add_argument('-virtual', dest='virtual', action='store_true', default=False,
            help='use virtual file')


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
    odir = inps.odir
    dateReference = inps.ref_date
    dateSecondary = inps.sec_date
    pol = inps.pol
    framesInput = inps.frames
    startingSwath = inps.starting_swath
    endingSwath = inps.ending_swath
    useVirtualFile = inps.virtual
    #######################################################


    #date directories sorted by acquistion date retrieved from filenames under each directory
    dateDirs = getAlos2StackDirs(os.path.abspath(idir))
    ndate = len(dateDirs)

    if framesInput is not None:
        framesInput = sorted(framesInput)
    else:
        framesInput = None


    #1. find index of reference date:
    dates = []
    dateIndexReference = None
    for i in range(ndate):
        ledFiles = sorted(glob.glob(os.path.join(dateDirs[i], 'LED-ALOS2*-*-*')))
        date = os.path.basename(ledFiles[0]).split('-')[-2]
        dates.append(date)
        if date == dateReference:
            dateIndexReference = i
    if dateIndexReference is None:
        raise Exception('cannot get reference date {} from the data list, pleasae check your input'.format(dateReference))
    

    #2. check if data are in the same mode
    spotlightModes, stripmapModes, scansarNominalModes, scansarWideModes, scansarModes = acquisitionModesAlos2()

    #first frame of reference date
    ledFilesReference = sorted(glob.glob(os.path.join(dateDirs[dateIndexReference], 'LED-ALOS2*-*-*')))
    modeReference = os.path.basename(ledFilesReference[0]).split('-')[-1][0:3]

    if modeReference in spotlightModes:
        modeGroupReference = spotlightModes
    if modeReference in stripmapModes:
        modeGroupReference = stripmapModes
    if modeReference in scansarNominalModes:
        modeGroupReference = scansarNominalModes
    if modeReference in scansarWideModes:
        modeGroupReference = scansarWideModes

    #check aquistion mode of all frames of each date
    for i in range(ndate):
        ledFiles = sorted(glob.glob(os.path.join(dateDirs[i], 'LED-ALOS2*-*-*')))
        nframe = len(ledFiles)
        for j in range(nframe):
            mode = os.path.basename(ledFiles[j]).split('-')[-1][0:3]
            if mode not in modeGroupReference:
                raise Exception('all data must be in the same acquistion mode: spotlight, stripmap, or ScanSAR mode')


    #3. find frame numbers and save it in a 2-d list
    frames = []
    #if not set, find frames automatically
    if framesInput is None:
        for i in range(ndate):
            frames0 = []
            ledFiles = sorted(glob.glob(os.path.join(dateDirs[i], 'LED-ALOS2*-*-*')))
            for led in ledFiles:
                frames0.append(  os.path.basename(led).split('-')[-3][-4:]  )
            frames.append(sorted(frames0))
    else:
        for i in range(ndate):
            frames.append(framesInput)

    framesReference = frames[dateIndexReference]

    #check if there is equal number of frames
    nframe = len(frames[dateIndexReference])
    for i in range(ndate):
        if nframe != len(frames[i]):
            raise Exception('there are not equal number of frames to process, please check your directory of each date')


    #4. set starting and ending swaths
    if modeReference in spotlightModes:
        if startingSwath is None:
            startingSwath = 1
        if endingSwath is None:
            endingSwath = 1
    if modeReference in stripmapModes:
        if startingSwath is None:
            startingSwath = 1
        if endingSwath is None:
            endingSwath = 1
    if modeReference in scansarNominalModes:
        if startingSwath is None:
            startingSwath = 1
        if endingSwath is None:
            endingSwath = 5
    if modeReference in scansarWideModes:
        if startingSwath is None:
            startingSwath = 1
        if endingSwath is None:
            endingSwath = 7

    #print result
    print('\nlist of dates:')
    print(' index      date            frames')
    print('=======================================================')
    for i in range(ndate):
        if dates[i] == dateReference:
            print('  %03d       %s'%(i, dates[i])+'      {}'.format(frames[i])+'    reference')
        else:
            print('  %03d       %s'%(i, dates[i])+'      {}'.format(frames[i]))
    print('\n')


    ##################################################
    #1. create directories and read data
    ##################################################
    if not os.path.isdir(odir):
        print('output directory {} does not exist, create'.format(odir))
        os.makedirs(odir, exist_ok=True)

    os.chdir(odir)
    for i in range(ndate):
        ledFiles = sorted(glob.glob(os.path.join(dateDirs[i], 'LED-ALOS2*-*-*')))
        date = os.path.basename(ledFiles[0]).split('-')[-2]
        dateDir = date

        if dateSecondary != []:
            if date not in dateSecondary:
                continue

        if os.path.isdir(dateDir):
            print('{} already exists, do not create'.format(dateDir))
            continue
        else:
            os.makedirs(dateDir, exist_ok=True)
            os.chdir(dateDir)

        sensor = MultiMode.createSensor(sensor='ALOS2', name=None)
        sensor.configure()
        sensor.track.configure()

        for j in range(nframe):
            #frame number starts with 1
            frameDir = 'f{}_{}'.format(j+1, framesReference[j])
            os.makedirs(frameDir, exist_ok=True)
            os.chdir(frameDir)

            #attach a frame to reference and secondary
            frameObj = MultiMode.createFrame()
            frameObj.configure()
            sensor.track.frames.append(frameObj)

            #swath number starts with 1
            for k in range(startingSwath, endingSwath+1):
                print('processing date {} frame {} swath {}'.format(date, framesReference[j], k))

                swathDir = 's{}'.format(k)
                os.makedirs(swathDir, exist_ok=True)
                os.chdir(swathDir)

                #attach a swath to sensor
                swathObj = MultiMode.createSwath()
                swathObj.configure()
                sensor.track.frames[-1].swaths.append(swathObj)

                #setup sensor
                #sensor.leaderFile = sorted(glob.glob(os.path.join(dateDirs[i], 'LED-ALOS2*{}-*-*'.format(framesReference[j]))))[0]
                sensor.leaderFile = sorted(glob.glob(os.path.join(dateDirs[i], 'LED-ALOS2*{}-*-*'.format(frames[i][j]))))[0]
                if modeReference in scansarModes:
                    #sensor.imageFile = sorted(glob.glob(os.path.join(dateDirs[i], 'IMG-{}-ALOS2*{}-*-*-F{}'.format(pol.upper(), framesReference[j], k))))[0]
                    sensor.imageFile = sorted(glob.glob(os.path.join(dateDirs[i], 'IMG-{}-ALOS2*{}-*-*-F{}'.format(pol.upper(), frames[i][j], k))))[0]
                else:
                    #sensor.imageFile = sorted(glob.glob(os.path.join(dateDirs[i], 'IMG-{}-ALOS2*{}-*-*'.format(pol.upper(), framesReference[j]))))[0]
                    sensor.imageFile = sorted(glob.glob(os.path.join(dateDirs[i], 'IMG-{}-ALOS2*{}-*-*'.format(pol.upper(), frames[i][j]))))[0]
                sensor.outputFile = date + '.slc'
                sensor.useVirtualFile = useVirtualFile
                #read sensor
                (imageFDR, imageData)=sensor.readImage()
                (leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord)=sensor.readLeader()
                sensor.setSwath(leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData)
                sensor.setFrame(leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData)
                sensor.setTrack(leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData)
                os.chdir('../')
            #!!!frame numbers of all dates are reset to those of reference date
            sensor.track.frames[j].frameNumber = framesReference[j]
            saveProduct(sensor.track.frames[-1], date + '.frame.xml')
            os.chdir('../')
        saveProduct(sensor.track, date + '.track.xml')
        os.chdir('../')
















