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

import isce, isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml

from StackPulic import loadProduct
from StackPulic import datesFromPairs


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


def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='unwrap subband interferograms for ionospheric correction')
    parser.add_argument('-idir', dest='idir', type=str, required=True,
            help = 'input directory where each pair (YYMMDD-YYMMDD) is located. only folders are recognized')
    parser.add_argument('-odir', dest='odir', type=str, required=True,
            help = 'output directory for estimated ionospheric phase of each date')
    parser.add_argument('-ref_date_stack', dest='ref_date_stack', type=str, required=True,
            help = 'reference date of stack. format: YYMMDD')
    parser.add_argument('-zro_date', dest='zro_date', type=str, default=None,
            help = 'date in least squares estimation whose ionospheric phase is assumed to be zero. format: YYMMDD. default: first date')
    parser.add_argument('-pairs', dest='pairs', type=str, nargs='+', default=None,
            help = 'pairs to be used in least squares estimation. This has highest priority. a number of pairs seperated by blanks. format: YYMMDD-YYMMDD YYMMDD-YYMMDD...')
    parser.add_argument('-exc_date', dest='exc_date', type=str, nargs='+', default=[],
            help = 'pairs involving these dates are excluded in least squares estimation. a number of dates seperated by blanks. format: YYMMDD YYMMDD YYMMDD...')
    parser.add_argument('-exc_pair', dest='exc_pair', type=str, nargs='+', default=[],
            help = 'pairs excluded in least squares estimation. a number of pairs seperated by blanks. format: YYMMDD-YYMMDD YYMMDD-YYMMDD...')
    parser.add_argument('-tsmax', dest='tsmax', type=float, default=None,
            help = 'maximum time span in years of pairs used in least squares estimation. default: None')
    parser.add_argument('-nrlks1', dest='nrlks1', type=int, default=1,
            help = 'number of range looks 1. default: 1')
    parser.add_argument('-nalks1', dest='nalks1', type=int, default=1,
            help = 'number of azimuth looks 1. default: 1')
    parser.add_argument('-nrlks2', dest='nrlks2', type=int, default=1,
            help = 'number of range looks 2. default: 1')
    parser.add_argument('-nalks2', dest='nalks2', type=int, default=1,
            help = 'number of azimuth looks 2. default: 1')
    parser.add_argument('-nrlks_ion', dest='nrlks_ion', type=int, default=1,
            help = 'number of range looks ion. default: 1')
    parser.add_argument('-nalks_ion', dest='nalks_ion', type=int, default=1,
            help = 'number of azimuth looks ion. default: 1')
    parser.add_argument('-ww', dest='ww', action='store_true', default=False,
            help='use reciprocal of window size as weight')
    parser.add_argument('-interp', dest='interp', action='store_true', default=False,
            help='interpolate ionospheric phase to nrlks2/nalks2 sample size')

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
    dateReferenceStack = inps.ref_date_stack
    dateZero = inps.zro_date
    pairsUser = inps.pairs
    dateExcluded = inps.exc_date
    pairExcluded = inps.exc_pair
    tsmax = inps.tsmax
    numberRangeLooks1 = inps.nrlks1
    numberAzimuthLooks1 = inps.nalks1
    numberRangeLooks2 = inps.nrlks2
    numberAzimuthLooks2 = inps.nalks2
    numberRangeLooksIon = inps.nrlks_ion
    numberAzimuthLooksIon = inps.nalks_ion
    ww = inps.ww
    interp = inps.interp
    #######################################################

    #all pair folders in order
    pairDirs = sorted(glob.glob(os.path.join(os.path.abspath(idir), '*-*')))
    pairDirs = [x for x in pairDirs if os.path.isdir(x)]

    #all pairs in order
    pairsAll = [os.path.basename(x) for x in pairDirs]

    #all dates in order
    datesAll = datesFromPairs(pairsAll)


    if pairsUser is not None:
        pairs = pairsUser
        for x in pairs:
            if x not in pairsAll:
               raise Exception('pair {} provided by user is not in processed pair list'.format(x)) 
    else:
        #exclude
        #pairs = [x for x in pairsAll if (x.split('-')[0] not in dateExcluded) and (x.split('-')[1] not in dateExcluded)]
        #pairs = [x for x in pairsAll if x not in pairExcluded]
        pairs = []
        for x in pairsAll:
            dateReference = x.split('-')[0]
            dateSecondary = x.split('-')[1]
            timeReference = datetime.datetime.strptime(dateReference, "%y%m%d")
            timeSecondary = datetime.datetime.strptime(dateSecondary, "%y%m%d")
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

    ml2 = '_{}rlks_{}alks'.format(numberRangeLooks1*numberRangeLooksIon, numberAzimuthLooks1*numberAzimuthLooksIon)
    ionfiltfile = 'filt_ion'+ml2+'.ion'
    stdfiltfile = 'filt_ion'+ml2+'.std'
    windowsizefiltfile = 'filt_ion'+ml2+'.win'
    ionfiltfile1 = os.path.join(idir, pairs[0], 'ion/ion_cal', ionfiltfile)

    img = isceobj.createImage()
    img.load(ionfiltfile1+'.xml')
    width = img.width
    length = img.length

    ionPairs = np.zeros((npair, length, width), dtype=np.float32)
    stdPairs = np.zeros((npair, length, width), dtype=np.float32)
    winPairs = np.zeros((npair, length, width), dtype=np.float32)
    for i in range(npair):
        ionfiltfile1 = os.path.join(idir, pairs[i], 'ion/ion_cal', ionfiltfile)
        stdfiltfile1 = os.path.join(idir, pairs[i], 'ion/ion_cal', stdfiltfile)
        windowsizefiltfile1 = os.path.join(idir, pairs[i], 'ion/ion_cal', windowsizefiltfile)

        ionPairs[i, :, :] = np.fromfile(ionfiltfile1, dtype=np.float32).reshape(length, width)
        stdPairs[i, :, :] = np.fromfile(stdfiltfile1, dtype=np.float32).reshape(length, width)
        winPairs[i, :, :] = np.fromfile(windowsizefiltfile1, dtype=np.float32).reshape(length, width)


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
        dateReference = pairs[k].split('-')[0]
        dateSecondary = pairs[k].split('-')[1]
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

            if ww == False:
                #observed signal
                S = S0
                H = H0
            else:
                #add weight
                #https://stackoverflow.com/questions/19624997/understanding-scipys-least-square-function-with-irls
                #https://stackoverflow.com/questions/27128688/how-to-use-least-squares-with-weight-matrix-in-python
                wgt = winPairs[:, i, j]
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

    ml3 = '_{}rlks_{}alks'.format(numberRangeLooks1*numberRangeLooks2, 
                              numberAzimuthLooks1*numberAzimuthLooks2)

    width2 = width
    length2 = length

    #ionrectfile1 = os.path.join(idir, pairs[0], 'insar', pairs[0] + ml3 + '.ion')
    #multilookDifferentialInterferogram = os.path.join(idir, pairs[0], 'insar', 'diff_' + pairs[0] + ml3 + '.int')
    #img = isceobj.createImage()
    #img.load(multilookDifferentialInterferogram + '.xml')
    #width3 = img.width
    #length3 = img.length

    trackParameter = os.path.join(idir, pairs[0], dateReferenceStack + '.track.xml')
    trackTmp = loadProduct(trackParameter)
    width3 = int(trackTmp.numberOfSamples / numberRangeLooks2)
    length3 = int(trackTmp.numberOfLines / numberAzimuthLooks2)

    #number of range looks output
    nrlo = numberRangeLooks1*numberRangeLooks2
    #number of range looks input
    nrli = numberRangeLooks1*numberRangeLooksIon
    #number of azimuth looks output
    nalo = numberAzimuthLooks1*numberAzimuthLooks2
    #number of azimuth looks input
    nali = numberAzimuthLooks1*numberAzimuthLooksIon

    cdir = os.getcwd()
    os.makedirs(odir, exist_ok=True)
    os.chdir(odir)

    for idate in range(ndate-1):
        print('interplate {}'.format(dates2[idate]))
        if interp and ((numberRangeLooks2 != numberRangeLooksIon) or (numberAzimuthLooks2 != numberAzimuthLooksIon)):
            ionfilt = ts[idate, :, :]
            index2 = np.linspace(0, width2-1, num=width2, endpoint=True)
            index3 = np.linspace(0, width3-1, num=width3, endpoint=True) * nrlo/nrli + (nrlo-nrli)/(2.0*nrli)
            ionrect = np.zeros((length3, width3), dtype=np.float32)
            for i in range(length2):
                f = interp1d(index2, ionfilt[i,:], kind='cubic', fill_value="extrapolate")
                ionrect[i, :] = f(index3)
            
            index2 = np.linspace(0, length2-1, num=length2, endpoint=True)
            index3 = np.linspace(0, length3-1, num=length3, endpoint=True) * nalo/nali + (nalo-nali)/(2.0*nali)
            for j in range(width3):
                f = interp1d(index2, ionrect[0:length2, j], kind='cubic', fill_value="extrapolate")
                ionrect[:, j] = f(index3)
            
            ionrectfile = 'filt_ion_'+dates2[idate]+ml3+'.ion'
            ionrect.astype(np.float32).tofile(ionrectfile)
            create_xml(ionrectfile, width3, length3, 'float')
        else:
            ionrectfile = 'filt_ion_'+dates2[idate]+ml2+'.ion'
            ts[idate, :, :].astype(np.float32).tofile(ionrectfile)
            create_xml(ionrectfile, width, length, 'float')

    if interp and ((numberRangeLooks2 != numberRangeLooksIon) or (numberAzimuthLooks2 != numberAzimuthLooksIon)):
        ionrectfile = 'filt_ion_'+dateZero+ml3+'.ion'
        (np.zeros((length3, width3), dtype=np.float32)).astype(np.float32).tofile(ionrectfile)
        create_xml(ionrectfile, width3, length3, 'float')
    else:
        ionrectfile = 'filt_ion_'+dateZero+ml2+'.ion'
        (np.zeros((length, width), dtype=np.float32)).astype(np.float32).tofile(ionrectfile)
        create_xml(ionrectfile, width, length, 'float')

    os.chdir(cdir)
