#!/usr/bin/env python3

# Author: Cunren Liang
# Copyright 2022

import os
import copy
import glob
import shutil
import argparse
import numpy as np

import isce
import isceobj
from isceobj.TopsProc.runIon import adaptive_gaussian
from isceobj.TopsProc.runIon import weight_fitting
from isceobj.TopsProc.runIon import fit_surface
from isceobj.TopsProc.runIon import cal_surface
import s1a_isce_utils as ut


def createParser():
    parser = argparse.ArgumentParser(description='compute and filter azimuth ionospheric shift [unit: masBurst.azimuthTimeInterval]')
    parser.add_argument('-k', '--reference_stack', type=str, dest='reference_stack', required=True,
                        help='Directory with the reference image of the stack')
    parser.add_argument('-f', '--reference', type=str, dest='reference', required=True,
                        help='Directory with the reference image (coregistered or reference of the stack)')
    parser.add_argument('-s', '--secondary', type=str, dest='secondary', required=True,
                        help='Directory with the secondary image (coregistered or reference of the stack)')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
            help='input ionosphere')
    parser.add_argument('-c', '--coherence', dest='coherence', type=str, required=True,
            help='coherence, e.g. raw_no_projection.cor')
    parser.add_argument('-o', '--output', dest='output', type=str, required=True,
            help='output azimuth ionospheric shift')
    parser.add_argument('-b', '--win_min', dest='win_min', type=int, default=100, 
            help='minimum filtering window size')
    parser.add_argument('-d', '--win_max', dest='win_max', type=int, default=200, 
            help='maximum filtering window size')
    parser.add_argument('-r', '--nrlks', dest='nrlks', type=int, default=1, 
            help='number of range looks. Default: 1')
    parser.add_argument('-a', '--nalks', dest='nalks', type=int, default=1, 
            help='number of azimuth looks. Default: 1')
    #parser.add_argument('-m', '--masked_areas', dest='masked_areas', type=int, nargs='+', action='append', default=None,
    #        help='This is a 2-d list. Each element in the 2-D list is a four-element list: [firstLine, lastLine, firstColumn, lastColumn], with line/column numbers starting with 1. If one of the four elements is specified with -1, the program will use firstLine/lastLine/firstColumn/lastColumn instead. e.g. two areas masked out: --masked_areas 10 20 10 20 --masked_areas 110 120 110 120')

    return parser


def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def main(iargs=None):
    '''
    check overlap among all acquistions, only keep the bursts that in the common overlap,
    and then renumber the bursts.
    '''
    inps = cmdLineParse(iargs)

    '''
    calculate azimuth shift caused by ionosphere using ionospheric phase
    '''

    #################################################
    #SET PARAMETERS HERE
    #gaussian filtering window size
    #size = np.int(np.around(width / 12.0))
    #size = ionParam.ionshiftFilteringWinsize
    size_max = inps.win_max
    size_min = inps.win_min

    #THESE SHOULD BE GOOD ENOUGH, NO NEED TO SET IN setup(self)
    #if applying polynomial fitting
    #0: no fitting, 1: with fitting
    fit = 0
    corThresholdIonshift = 0.85
    #################################################


####################################################################
    #STEP 1. GET DERIVATIVE OF IONOSPHERE
####################################################################

    #get files
    ionfile = inps.input
    #we are using filtered ionosphere, so we should use coherence file that is not projected.
    #corfile = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname, ionParam.ionCor)
    corfile = inps.coherence
    img = isceobj.createImage()
    img.load(ionfile + '.xml')
    width = img.width
    length = img.length
    amp = (np.fromfile(ionfile, dtype=np.float32).reshape(length*2, width))[0:length*2:2, :]
    ion = (np.fromfile(ionfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    cor = (np.fromfile(corfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]

    ########################################################################################
    #AFTER COHERENCE IS RESAMPLED AT grd2ion, THERE ARE SOME WIRED VALUES
    cor[np.nonzero(cor<0)] = 0.0
    cor[np.nonzero(cor>1)] = 0.0
    ########################################################################################

    #get the azimuth derivative of ionosphere
    dion = np.diff(ion, axis=0)
    dion = np.concatenate((dion, np.zeros((1,width))), axis=0)

    #remove the samples affected by zeros
    flag_ion0 = (ion!=0)
    #moving down by one line
    flag_ion1 = np.roll(flag_ion0, 1, axis=0)
    flag_ion1[0,:] = 0
    #moving up by one line
    flag_ion2 = np.roll(flag_ion0, -1, axis=0)
    flag_ion2[-1,:] = 0
    #now remove the samples affected by zeros
    flag_ion = flag_ion0 * flag_ion1 * flag_ion2
    dion *= flag_ion

    flag = flag_ion * (cor>corThresholdIonshift)
    index = np.nonzero(flag)


####################################################################
    #STEP 2. FIT A POLYNOMIAL TO THE DERIVATIVE OF IONOSPHERE
####################################################################

    order = 3

    #look for data to use
    point_index = np.nonzero(flag)
    m = point_index[0].shape[0]

    #calculate input index matrix
    x0=np.matlib.repmat(np.arange(width), length, 1)
    y0=np.matlib.repmat(np.arange(length).reshape(length, 1), 1, width)

    x = x0[point_index].reshape(m, 1)
    y = y0[point_index].reshape(m, 1)
    z = dion[point_index].reshape(m, 1)
    w = cor[point_index].reshape(m, 1)

    #convert to higher precision type before use
    x=np.asfarray(x,np.float64)
    y=np.asfarray(y,np.float64)
    z=np.asfarray(z,np.float64)
    w=np.asfarray(w,np.float64)
    coeff = fit_surface(x, y, z, w, order)

    rgindex = np.arange(width)
    azindex = np.arange(length).reshape(length, 1)
    #convert to higher precision type before use
    rgindex=np.asfarray(rgindex,np.float64)
    azindex=np.asfarray(azindex,np.float64)
    dion_fit = cal_surface(rgindex, azindex, coeff, order)

    #no fitting
    if fit == 0:
        dion_fit *= 0
    dion_res = (dion - dion_fit)*(dion!=0)


####################################################################
    #STEP 3. FILTER THE RESIDUAL OF THE DERIVATIVE OF IONOSPHERE
####################################################################

    #this will be affected by low coherence areas like water, so not use this.
    #filter the derivation of ionosphere
    #if size % 2 == 0:
    #    size += 1
    #sigma = size / 2.0

    #g2d = gaussian(size, sigma, scale=1.0)
    #scale = ss.fftconvolve((dion_res!=0), g2d, mode='same')
    #dion_filt = ss.fftconvolve(dion_res, g2d, mode='same') / (scale + (scale==0))

    #minimize the effect of low coherence pixels
    cor[np.nonzero( (cor<0.85)*(cor!=0) )] = 0.00001
    dion_filt = adaptive_gaussian(dion_res, cor, size_max, size_min)

    dion = (dion_fit + dion_filt)*(dion!=0)

    #return dion


####################################################################
    #STEP 4. CONVERT TO AZIMUTH SHIFT
####################################################################

    #!!! use number of swaths in reference for now

    #these are coregistered secondaries, so there should be <= swaths in reference of entire stack
    referenceSwathList = ut.getSwathList(inps.reference)
    secondarySwathList = ut.getSwathList(inps.secondary)
    swathList = list(sorted(set(referenceSwathList+secondarySwathList)))

    #swathList = ut.getSwathList(inps.reference)
    firstSwath = None
    for swath in swathList:
        frameReference = ut.loadProduct(os.path.join(inps.reference_stack, 'IW{0}.xml'.format(swath)))

        minBurst = frameReference.bursts[0].burstNumber
        maxBurst = frameReference.bursts[-1].burstNumber
        if minBurst==maxBurst:
            print('Skipping processing of swath {0}'.format(swath))
            continue
        else:
            firstSwath = swath
            break
    if firstSwath is None:
        raise Exception('no valid swaths!')

    midBurstIndex = round((minBurst + maxBurst) / 2.0) - minBurst
    masBurst = frameReference.bursts[midBurstIndex]

    #shift casued by ionosphere [unit: masBurst.azimuthTimeInterval]
    rng = masBurst.rangePixelSize * ((np.arange(width))*inps.nrlks + (inps.nrlks - 1.0) / 2.0) + masBurst.startingRange
    Ka = masBurst.azimuthFMRate(rng)
    ionShift = dion / (masBurst.azimuthTimeInterval * inps.nalks) / (4.0 * np.pi) / Ka[None, :] / masBurst.azimuthTimeInterval

    #output
    outfile = inps.output
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    ion = np.zeros((length*2, width), dtype=np.float32)
    ion[0:length*2:2, :] = amp
    ion[1:length*2:2, :] = ionShift
    ion.astype(np.float32).tofile(outfile)
    img.filename = outfile
    img.extraFilename = outfile + '.vrt'
    img.renderHdr()


if __name__ == '__main__':
    '''
    Main driver.
    '''
    # Main Driver
    main()



