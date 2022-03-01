#!/usr/bin/env python3

#
# Author: Cunren Liang
# Copyright 2021
#

import os
import glob
import shutil
import datetime
import numpy as np

import isce, isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml


def datesFromPairs(pairs):
    '''get all dates from pairs
    '''
    dates = []
    for p in pairs:
        for x in p.split('_'):
            if x not in dates:
                dates.append(x)

    dates.sort()

    return dates


def least_sqares(H, S, W=None):
    '''
    #This can make use multiple threads (set environment variable: OMP_NUM_THREADS)
    linear equations:  H theta = s
    W:                 weight matrix
    '''

    S.reshape(H.shape[0], 1)
    if W is None:
        #use np.dot instead since some old python versions don't have matmul
        m1 = np.linalg.inv(np.dot(H.transpose(), H))
        Z = np.dot(       np.dot(m1, H.transpose())           , S)
    else:
        #use np.dot instead since some old python versions don't have matmul
        m1 = np.linalg.inv(np.dot(np.dot(H.transpose(), W), H))
        Z = np.dot(np.dot(np.dot(m1, H.transpose()), W), S)

    return Z.reshape(Z.size)


def interp_2d(data1, numberRangeLooks1, numberRangeLooks2, numberAzimuthLooks1, numberAzimuthLooks2, width2=None, length2=None):
    '''
    interpolate data1 of numberRangeLooks1/numberAzimuthLooks1 to data2 of numberRangeLooks2/numberAzimuthLooks2
    '''
    length1, width1 = data1.shape

    if width2 is None:
        width2 = int(np.around(width1*numberRangeLooks1/numberRangeLooks2))
    if length2 is None:
        length2 = int(np.around(length1*numberAzimuthLooks1/numberAzimuthLooks2))


    #number of range looks input
    nrli = numberRangeLooks1
    #number of range looks output
    nrlo = numberRangeLooks2
    #number of azimuth looks input
    nali = numberAzimuthLooks1
    #number of azimuth looks output
    nalo = numberAzimuthLooks2

    index1 = np.linspace(0, width1-1, num=width1, endpoint=True)
    index2 = np.linspace(0, width2-1, num=width2, endpoint=True) * nrlo/nrli + (nrlo-nrli)/(2.0*nrli)
    data2 = np.zeros((length2, width2), dtype=data1.dtype)
    for i in range(length1):
        f = interp1d(index1, data1[i,:], kind='cubic', fill_value="extrapolate")
        data2[i, :] = f(index2)
    
    index1 = np.linspace(0, length1-1, num=length1, endpoint=True)
    index2 = np.linspace(0, length2-1, num=length2, endpoint=True) * nalo/nali + (nalo-nali)/(2.0*nali)
    for j in range(width2):
        f = interp1d(index1, data2[0:length1, j], kind='cubic', fill_value="extrapolate")
        data2[:, j] = f(index2)

    return data2


def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='least squares estimation')
    parser.add_argument('--idir', dest='idir', type=str, required=True,
            help = 'input directory where each pair (YYYYMMDD_YYYYMMDD) is located. only folders are recognized')
    parser.add_argument('--odir', dest='odir', type=str, required=True,
            help = 'output directory for estimated result of each date')
    parser.add_argument('--zro_date', dest='zro_date', type=str, default=None,
            help = 'date in least squares estimation whose ionospheric phase is assumed to be zero. format: YYYYMMDD. default: first date')

    parser.add_argument('--exc_date', dest='exc_date', type=str, nargs='+', default=[],
            help = 'pairs involving these dates are excluded in least squares estimation. a number of dates seperated by blanks. format: YYYYMMDD YYYYMMDD YYYYMMDD...')
    parser.add_argument('--exc_pair', dest='exc_pair', type=str, nargs='+', default=[],
            help = 'pairs excluded in least squares estimation. a number of pairs seperated by blanks. format: YYYYMMDD-YYYYMMDD YYYYMMDD-YYYYMMDD...')
    parser.add_argument('--tsmax', dest='tsmax', type=float, default=None,
            help = 'maximum time span in years of pairs used in least squares estimation. default: None')

    parser.add_argument('--nrlks1', dest='nrlks1', type=int, default=1,
            help = 'number of range looks of input. default: 1')
    parser.add_argument('--nalks1', dest='nalks1', type=int, default=1,
            help = 'number of azimuth looks of input. default: 1')
    parser.add_argument('--nrlks2', dest='nrlks2', type=int, default=1,
            help = 'number of range looks of output. default: 1')
    parser.add_argument('--nalks2', dest='nalks2', type=int, default=1,
            help = 'number of azimuth looks of output. default: 1')
    parser.add_argument('--width2', dest='width2', type=int, default=None,
            help = 'width of output result. default: None, determined by program')
    parser.add_argument('--length2', dest='length2', type=int, default=None,
            help = 'length of output result. default: None, determined by program')
    parser.add_argument('--merged_geom', dest='merged_geom', type=str, default=None,
            help = 'a merged geometry file for getting width2/length2, e.g. merged/geom_reference/hgt.rdr. if provided, --width2/--length2 will be overwritten')

    parser.add_argument('--interp', dest='interp', action='store_true', default=False,
            help='interpolate estimated result to nrlks2/nalks2 sample size')
    parser.add_argument('--msk_overlap', dest='msk_overlap', action='store_true', default=False,
            help='mask output with overlap of all acquisitions')


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
    dateZero = inps.zro_date
    dateExcluded = inps.exc_date
    pairExcluded = inps.exc_pair
    tsmax = inps.tsmax
    numberRangeLooks1 = inps.nrlks1
    numberAzimuthLooks1 = inps.nalks1
    numberRangeLooks2 = inps.nrlks2
    numberAzimuthLooks2 = inps.nalks2
    width2 = inps.width2
    length2 = inps.length2
    mergedGeom = inps.merged_geom
    interp = inps.interp
    maskOverlap = inps.msk_overlap
    #######################################################

    #all pair folders in order
    pairDirs = sorted(glob.glob(os.path.join(os.path.abspath(idir), '*_*')))
    pairDirs = [x for x in pairDirs if os.path.isdir(x)]

    #all pairs in order
    pairsAll = [os.path.basename(x) for x in pairDirs]
    #all dates in order
    datesAll = datesFromPairs(pairsAll)

    #select pairs
    pairs = []
    for x in pairsAll:
        dateReference, dateSecondary = x.split('_')
        timeReference = datetime.datetime.strptime(dateReference, "%Y%m%d")
        timeSecondary = datetime.datetime.strptime(dateSecondary, "%Y%m%d")
        ts = np.absolute((timeSecondary - timeReference).total_seconds()) / (365.0 * 24.0 * 3600)
        if (dateReference in dateExcluded) and (dateSecondary in dateExcluded):
            continue
        if (x in pairExcluded):
            continue
        if tsmax is not None:
            if ts > tsmax:
                continue
        pairs.append(x)

    dates = datesFromPairs(pairs)
    if dateZero is not None:
        if dateZero not in dates:
            raise Exception('zro_date provided by user not in the dates involved in least squares estimation.')
    else:
        dateZero = dates[0]

    print('all pairs:\n{}'.format(' '.join(pairsAll)))
    print('all dates:\n{}'.format(' '.join(datesAll)))
    print('used pairs:\n{}'.format(' '.join(pairs)))
    print('used dates:\n{}'.format(' '.join(dates)))


####################################################################################
    print('\nSTEP 1. read files')
####################################################################################

    ndate = len(dates)
    npair = len(pairs)

    ionfile = os.path.join(idir, pairs[0], 'ion_cal', 'filt.ion')

    img = isceobj.createImage()
    img.load(ionfile+'.xml')
    width = img.width
    length = img.length

    ionPairs = np.zeros((npair, length, width), dtype=np.float32)
    flag = np.ones((length, width), dtype=np.float32)

    #this is reserved for use
    wls = False
    stdPairs = np.ones((npair, length, width), dtype=np.float32)
    for i in range(npair):
        ionfile = os.path.join(idir, pairs[i], 'ion_cal', 'filt.ion')
        ionPairs[i, :, :] = (np.fromfile(ionfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
        #flag of valid/invalid is defined by amplitde image
        amp = (np.fromfile(ionfile, dtype=np.float32).reshape(length*2, width))[0:length*2:2, :]
        flag *= (amp!=0)


####################################################################################
    print('\nSTEP 2. do least squares')
####################################################################################
    import copy
    from numpy.linalg import matrix_rank
    dates2 = copy.deepcopy(dates)
    dates2.remove(dateZero)

    #observation matrix
    H0 = np.zeros((npair, ndate-1))
    for k in range(npair):
        dateReference = pairs[k].split('_')[0]
        dateSecondary = pairs[k].split('_')[1]
        if dateReference != dateZero:
            dateReference_i = dates2.index(dateReference)
            H0[k, dateReference_i] = 1
        if dateSecondary != dateZero:
            dateSecondary_i = dates2.index(dateSecondary)
            H0[k, dateSecondary_i] = -1
    rank = matrix_rank(H0)
    if rank < ndate-1:
        raise Exception('dates to be estimated are not fully connected by the pairs used in least squares')
    else:
        print('number of pairs to be used in least squares: {}'.format(npair))
        print('number of dates to be estimated: {}'.format(ndate-1))
        print('observation matrix rank: {}'.format(rank))

    ts = np.zeros((ndate-1, length, width), dtype=np.float32)
    for i in range(length):
        if (i+1) % 50 == 0 or (i+1) == length:
            print('processing line: %6d of %6d' % (i+1, length), end='\r')
        if (i+1) == length:
            print()
        for j in range(width):

            #observed signal
            S0 = ionPairs[:, i, j]

            if wls == False:
                #observed signal
                S = S0
                H = H0
            else:
                #add weight
                #https://stackoverflow.com/questions/19624997/understanding-scipys-least-square-function-with-irls
                #https://stackoverflow.com/questions/27128688/how-to-use-least-squares-with-weight-matrix-in-python
                wgt = (stdPairs[:, i, j])**2
                W = np.sqrt(1.0/wgt)
                H = H0 * W[:, None]
                S = S0 * W

            #do least-squares estimation
            #[theta, residuals, rank, singular] = np.linalg.lstsq(H, S)
            #make W full matrix if use W here (which is a slower method)
            #'using W before this' is faster
            theta = least_sqares(H, S, W=None)
            ts[:, i, j] = theta

    # #dump raw estimate
    # cdir = os.getcwd()
    # os.makedirs(odir, exist_ok=True)
    # os.chdir(odir)

    # for i in range(ndate-1):
    #     file_name = 'filt_ion_'+dates2[i]+ml2+'.ion'
    #     ts[i, :, :].astype(np.float32).tofile(file_name)
    #     create_xml(file_name, width, length, 'float')
    # file_name = 'filt_ion_'+dateZero+ml2+'.ion'
    # (np.zeros((length, width), dtype=np.float32)).astype(np.float32).tofile(file_name)
    # create_xml(file_name, width, length, 'float')

    # os.chdir(cdir)


####################################################################################
    print('\nSTEP 3. interpolate ionospheric phase')
####################################################################################
    from scipy.interpolate import interp1d

    width1 = width
    length1 = length

    if width2 is None:
        width2 = int(width1 * numberRangeLooks1 / numberRangeLooks2)
    if length2 is None:
        length2 = int(length1 * numberAzimuthLooks1 / numberAzimuthLooks2)
    if mergedGeom is not None:
        from osgeo import gdal
        ds = gdal.Open(mergedGeom + ".vrt", gdal.GA_ReadOnly)
        width2 = ds.RasterXSize
        length2 = ds.RasterYSize

    os.makedirs(odir, exist_ok=True)
    for idate in range(ndate-1):
        print('interplate {}'.format(dates2[idate]))

        ionrectfile = os.path.join(odir, dates2[idate]+'.ion')
        if interp and ((numberRangeLooks1 != numberRangeLooks2) or (numberAzimuthLooks1 != numberAzimuthLooks2)):
            ionrect = interp_2d(ts[idate, :, :], numberRangeLooks1, numberRangeLooks2, numberAzimuthLooks1, numberAzimuthLooks2, 
                                width2=width2, length2=length2)

            #mask with overlap of all acquistions
            if maskOverlap:
                if idate == 0:
                    flagrect = interp_2d(flag, numberRangeLooks1, numberRangeLooks2, numberAzimuthLooks1, numberAzimuthLooks2, 
                                    width2=width2, length2=length2)
                ionrect *= (flagrect>0.5)

            ionrect.astype(np.float32).tofile(ionrectfile)
            create_xml(ionrectfile, width2, length2, 'float')
        else:
            ionrect = ts[idate, :, :]

            if maskOverlap:
                ionrect *= flag

            ionrect.astype(np.float32).tofile(ionrectfile)
            create_xml(ionrectfile, width1, length1, 'float')

    ionrectfile = os.path.join(odir, dateZero+'.ion')
    if interp and ((numberRangeLooks1 != numberRangeLooks2) or (numberAzimuthLooks1 != numberAzimuthLooks2)):
        (np.zeros((length2, width2), dtype=np.float32)).astype(np.float32).tofile(ionrectfile)
        create_xml(ionrectfile, width2, length2, 'float')
    else:
        (np.zeros((length1, width1), dtype=np.float32)).astype(np.float32).tofile(ionrectfile)
        create_xml(ionrectfile, width1, length1, 'float')




