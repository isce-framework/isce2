#!/usr/bin/env python3

#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import glob
import shutil
import datetime
import numpy as np
import xml.etree.ElementTree as ET

from StackPulic import stackDateStatistics
from StackPulic import acquisitionModesAlos2


def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='create InSAR pairs')
    parser.add_argument('-idir1', dest='idir1', type=str, required=True,
            help = 'input directory where original data of each date (YYMMDD) is located. only folders are recognized')
    parser.add_argument('-idir2', dest='idir2', type=str, required=True,
            help = 'input directory where resampled data of each date (YYMMDD) is located. only folders are recognized')
    parser.add_argument('-xml', dest='xml', type=str, default=None,
            help = 'alos2App.py input xml file, e.g. alos2App.xml. default: None')
    parser.add_argument('-odir', dest='odir', type=str, required=True,
            help = 'output directory')
    parser.add_argument('-ref_date', dest='ref_date', type=str, required=True,
            help = 'reference date. format: YYMMDD')
    parser.add_argument('-pairs', dest='pairs', type=str, nargs='+', default=None,
            help = 'a number of pairs seperated by blanks. format: YYMMDD-YYMMDD YYMMDD-YYMMDD YYMMDD-YYMMDD... This argument has highest priority. When provided, only process these pairs')
    parser.add_argument('-num', dest='num', type=int, default=None,
            help = 'number of subsequent acquistions for each acquistion to pair up with. default: all pairs')
    parser.add_argument('-exc_date', dest='exc_date', type=str, nargs='+', default=None,
            help = 'a number of secondary dates seperated by blanks, can also include ref_date. format: YYMMDD YYMMDD YYMMDD. If provided, these dates will be excluded from pairing up')
    parser.add_argument('-tsmin', dest='tsmin', type=float, default=None,
            help = 'minimum time span in years for pairing up. default: None')
    parser.add_argument('-tsmax', dest='tsmax', type=float, default=None,
            help = 'maximum time span in years for pairing up. default: None')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()


    #get user parameters from input
    idir1 = inps.idir1
    idir2 = inps.idir2
    alos2AppXml = inps.xml
    odir = inps.odir
    dateReference = inps.ref_date
    pairsUser = inps.pairs
    subsequentNum = inps.num
    dateExcluded = inps.exc_date
    tsmin = inps.tsmin
    tsmax = inps.tsmax
    #######################################################

    DEBUG=False

    spotlightModes, stripmapModes, scansarNominalModes, scansarWideModes, scansarModes = acquisitionModesAlos2()

    #get date statistics, using resampled version
    dateDirs,   dates,   frames,   swaths,   dateIndexReference = stackDateStatistics(idir2, dateReference)
    ndate = len(dates)
    nframe = len(frames)
    nswath = len(swaths)

    if subsequentNum is None:
        subsequentNum = ndate - 1

    #read standard configurations
    if alos2AppXml is not None:
        tree = ET.parse(alos2AppXml)
        root = tree.getroot()

    datefmt = "%y%m%d"
    pairsCreated = []
    for i in range(ndate):
        mdate = dates[i]
        mtime = datetime.datetime.strptime(mdate, datefmt)
        for j in range(subsequentNum):
            if i+j+1 <= ndate - 1:
                sdate = dates[i+j+1]
                stime = datetime.datetime.strptime(sdate, datefmt)
                pair = mdate + '-' + sdate
                ts = np.absolute((stime - mtime).total_seconds()) / (365.0 * 24.0 * 3600)

                #1. determine whether process this pair
                if pairsUser is not None:
                    if pair not in pairsUser:
                        continue
                else:
                    if dateExcluded is not None:
                        if (mdate in dateExcluded) or (sdate in dateExcluded):
                            continue
                    if tsmin is not None:
                        if ts < tsmin:
                            continue
                    if tsmax is not None:
                        if ts > tsmax:
                            continue

                #2. create pair dir
                pairsCreated.append(pair)
                print('creating pair: {}'.format(pair))
                pairDir = os.path.join(odir, pair)
                os.makedirs(pairDir, exist_ok=True)
                #create xml
                if alos2AppXml is not None:
                    safe = root.find("component/property[@name='reference directory']")
                    #safe.text = '{}'.format(os.path.join(inps.dir, mdate))
                    safe.text = 'None'
                    safe = root.find("component/property[@name='secondary directory']")
                    #safe.text = '{}'.format(os.path.join(inps.dir, sdate))
                    safe.text = 'None'
                    tree.write(os.path.join(pairDir, 'alos2App.xml'))

                #3. make frame/swath directories, and copy *.track.xml and *.frame.xml
                if mdate != dates[dateIndexReference]:
                    shutil.copy2(os.path.join(idir1, mdate, mdate+'.track.xml'), pairDir)
                if sdate != dates[dateIndexReference]:
                    shutil.copy2(os.path.join(idir1, sdate, sdate+'.track.xml'), pairDir)
                shutil.copy2(os.path.join(idir2, dates[dateIndexReference], dates[dateIndexReference]+'.track.xml'), pairDir)

                for iframe, frameNumber in enumerate(frames):
                    frameDir = 'f{}_{}'.format(iframe+1, frameNumber)
                    os.makedirs(os.path.join(pairDir, frameDir), exist_ok=True)

                    if mdate != dates[dateIndexReference]:
                        shutil.copy2(os.path.join(idir1, mdate, frameDir, mdate+'.frame.xml'), os.path.join(pairDir, frameDir))
                    if sdate != dates[dateIndexReference]:
                        shutil.copy2(os.path.join(idir1, sdate, frameDir, sdate+'.frame.xml'), os.path.join(pairDir, frameDir))
                    shutil.copy2(os.path.join(idir2, dates[dateIndexReference], frameDir, dates[dateIndexReference]+'.frame.xml'), os.path.join(pairDir, frameDir))

                    for jswath, swathNumber in enumerate(range(swaths[0], swaths[-1] + 1)):
                        swathDir = 's{}'.format(swathNumber)
                        os.makedirs(os.path.join(pairDir, frameDir, swathDir), exist_ok=True)

                        if os.path.isfile(os.path.join(pairDir, frameDir, swathDir, mdate+'.slc')):
                            os.remove(os.path.join(pairDir, frameDir, swathDir, mdate+'.slc'))
                        relpath = os.path.relpath(os.path.join(idir2, mdate, frameDir, swathDir), os.path.join(pairDir, frameDir, swathDir))
                        os.symlink(os.path.join(relpath, mdate+'.slc'), os.path.join(pairDir, frameDir, swathDir, mdate+'.slc'))
                        #os.symlink(os.path.join(idir2, mdate, frameDir, swathDir, mdate+'.slc'), os.path.join(pairDir, frameDir, swathDir, mdate+'.slc'))
                        shutil.copy2(os.path.join(idir2, mdate, frameDir, swathDir, mdate+'.slc.vrt'), os.path.join(pairDir, frameDir, swathDir))
                        shutil.copy2(os.path.join(idir2, mdate, frameDir, swathDir, mdate+'.slc.xml'), os.path.join(pairDir, frameDir, swathDir))

                        if os.path.isfile(os.path.join(pairDir, frameDir, swathDir, sdate+'.slc')):
                            os.remove(os.path.join(pairDir, frameDir, swathDir, sdate+'.slc'))
                        relpath = os.path.relpath(os.path.join(idir2, sdate, frameDir, swathDir), os.path.join(pairDir, frameDir, swathDir))
                        os.symlink(os.path.join(relpath, sdate+'.slc'), os.path.join(pairDir, frameDir, swathDir, sdate+'.slc'))
                        #os.symlink(os.path.join(idir2, sdate, frameDir, swathDir, sdate+'.slc'), os.path.join(pairDir, frameDir, swathDir, sdate+'.slc'))
                        shutil.copy2(os.path.join(idir2, sdate, frameDir, swathDir, sdate+'.slc.vrt'), os.path.join(pairDir, frameDir, swathDir))
                        shutil.copy2(os.path.join(idir2, sdate, frameDir, swathDir, sdate+'.slc.xml'), os.path.join(pairDir, frameDir, swathDir))


    print('total number of pairs created: {}'.format(len(pairsCreated)))
    if pairsUser is not None:
        if sorted(pairsUser) != sorted(pairsCreated):
            print()
            print('WARNING: user has specified pairs to process, but pairs created are different from user specified pairs')
            print('         user specified pairs: {}'.format(', '.join(pairsUser)))
            print('         pairs created: {}'.format(', '.join(pairsCreated)))
            print()
















