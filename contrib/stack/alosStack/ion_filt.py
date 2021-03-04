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
from isceobj.Alos2Proc.runIonFilt import computeIonosphere
from isceobj.Alos2Proc.runIonFilt import gaussian
#from isceobj.Alos2Proc.runIonFilt import least_sqares
from isceobj.Alos2Proc.runIonFilt import polyfit_2d
from isceobj.Alos2Proc.runIonFilt import adaptive_gaussian
from isceobj.Alos2Proc.runIonFilt import reformatMaskedAreas

from StackPulic import loadTrack
from StackPulic import createObject
from StackPulic import stackDateStatistics
from StackPulic import acquisitionModesAlos2
from StackPulic import subbandParameters

from compute_burst_sync import computeBurstSynchronization


def ionFilt(self, referenceTrack, catalog=None):

    from isceobj.Alos2Proc.runIonSubband import defineIonDir
    ionDir = defineIonDir()
    subbandPrefix = ['lower', 'upper']

    ionCalDir = os.path.join(ionDir['ion'], ionDir['ionCal'])
    os.makedirs(ionCalDir, exist_ok=True)
    os.chdir(ionCalDir)

    log  = ''

    ############################################################
    # STEP 1. compute ionospheric phase
    ############################################################
    from isceobj.Constants import SPEED_OF_LIGHT
    from isceobj.Alos2Proc.Alos2ProcPublic import create_xml

    ###################################
    #SET PARAMETERS HERE
    #THESE SHOULD BE GOOD ENOUGH, NO NEED TO SET IN setup(self)
    corThresholdAdj = 0.97
    corOrderAdj = 20
    ###################################

    print('\ncomputing ionosphere')
    #get files
    ml2 = '_{}rlks_{}alks'.format(self._insar.numberRangeLooks1*self._insar.numberRangeLooksIon, 
                              self._insar.numberAzimuthLooks1*self._insar.numberAzimuthLooksIon)

    lowerUnwfile = subbandPrefix[0]+ml2+'.unw'
    upperUnwfile = subbandPrefix[1]+ml2+'.unw'
    corfile = 'diff'+ml2+'.cor'

    #use image size from lower unwrapped interferogram
    img = isceobj.createImage()
    img.load(lowerUnwfile + '.xml')
    width = img.width
    length = img.length

    lowerUnw = (np.fromfile(lowerUnwfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    upperUnw = (np.fromfile(upperUnwfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    cor = (np.fromfile(corfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    #amp = (np.fromfile(corfile, dtype=np.float32).reshape(length*2, width))[0:length*2:2, :]

    #masked out user-specified areas
    if self.maskedAreasIon != None:
        maskedAreas = reformatMaskedAreas(self.maskedAreasIon, length, width)
        for area in maskedAreas:
            lowerUnw[area[0]:area[1], area[2]:area[3]] = 0
            upperUnw[area[0]:area[1], area[2]:area[3]] = 0
            cor[area[0]:area[1], area[2]:area[3]] = 0

    #remove possible wired values in coherence
    cor[np.nonzero(cor<0)] = 0.0
    cor[np.nonzero(cor>1)] = 0.0

    #remove water body
    wbd = np.fromfile('wbd'+ml2+'.wbd', dtype=np.int8).reshape(length, width)
    cor[np.nonzero(wbd==-1)] = 0.0

    #remove small values
    cor[np.nonzero(cor<corThresholdAdj)] = 0.0

    #compute ionosphere
    fl = SPEED_OF_LIGHT / self._insar.subbandRadarWavelength[0]
    fu = SPEED_OF_LIGHT / self._insar.subbandRadarWavelength[1]
    adjFlag = 1
    ionos = computeIonosphere(lowerUnw, upperUnw, cor**corOrderAdj, fl, fu, adjFlag, 0)

    #dump ionosphere
    ionfile = 'ion'+ml2+'.ion'
    # ion = np.zeros((length*2, width), dtype=np.float32)
    # ion[0:length*2:2, :] = amp
    # ion[1:length*2:2, :] = ionos
    # ion.astype(np.float32).tofile(ionfile)
    # img.filename = ionfile
    # img.extraFilename = ionfile + '.vrt'
    # img.renderHdr()

    ionos.astype(np.float32).tofile(ionfile)
    create_xml(ionfile, width, length, 'float')


    ############################################################
    # STEP 2. filter ionospheric phase
    ############################################################
    import scipy.signal as ss

    #################################################
    #SET PARAMETERS HERE
    #fit and filter ionosphere
    fit = self.fitIon
    filt = self.filtIon
    fitAdaptive = self.fitAdaptiveIon
    filtSecondary = self.filtSecondaryIon
    if (fit == False) and (filt == False):
        raise Exception('either fit ionosphere or filt ionosphere should be True when doing ionospheric correction\n')

    #filtering window size
    size_max = self.filteringWinsizeMaxIon
    size_min = self.filteringWinsizeMinIon
    size_secondary = self.filteringWinsizeSecondaryIon
    if size_min > size_max:
        print('\n\nWARNING: minimum window size for filtering ionosphere phase {} > maximum window size {}'.format(size_min, size_max))
        print('         re-setting maximum window size to {}\n\n'.format(size_min))
        size_max = size_min
    if size_secondary % 2 != 1:
        size_secondary += 1
        print('window size of secondary filtering of ionosphere phase should be odd, window size changed to {}'.format(size_secondary))

    #coherence threshold for fitting a polynomial
    corThresholdFit = 0.25

    #ionospheric phase standard deviation after filtering
    std_out0 = self.filterStdIon
    #std_out0 = 0.1
    #################################################

    print('\nfiltering ionosphere')

    #input files
    ionfile = 'ion'+ml2+'.ion'
    #corfile = 'diff'+ml2+'.cor'
    corLowerfile = subbandPrefix[0]+ml2+'.cor'
    corUpperfile = subbandPrefix[1]+ml2+'.cor'
    #output files
    ionfiltfile = 'filt_ion'+ml2+'.ion'
    stdfiltfile = 'filt_ion'+ml2+'.std'
    windowsizefiltfile = 'filt_ion'+ml2+'.win'

    #read data
    img = isceobj.createImage()
    img.load(ionfile + '.xml')
    width = img.width
    length = img.length

    ion = np.fromfile(ionfile, dtype=np.float32).reshape(length, width)
    corLower = (np.fromfile(corLowerfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    corUpper = (np.fromfile(corUpperfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    cor = (corLower + corUpper) / 2.0
    index = np.nonzero(np.logical_or(corLower==0, corUpper==0))
    cor[index] = 0
    del corLower, corUpper

    #masked out user-specified areas
    if self.maskedAreasIon != None:
        maskedAreas = reformatMaskedAreas(self.maskedAreasIon, length, width)
        for area in maskedAreas:
            ion[area[0]:area[1], area[2]:area[3]] = 0
            cor[area[0]:area[1], area[2]:area[3]] = 0

    #remove possible wired values in coherence
    cor[np.nonzero(cor<0)] = 0.0
    cor[np.nonzero(cor>1)] = 0.0

    #remove water body. Not helpful, just leave it here
    wbd = np.fromfile('wbd'+ml2+'.wbd', dtype=np.int8).reshape(length, width)
    cor[np.nonzero(wbd==-1)] = 0.0

    # #applying water body mask here
    # waterBodyFile = 'wbd'+ml2+'.wbd'
    # if os.path.isfile(waterBodyFile):
    #     print('applying water body mask to coherence used to compute ionospheric phase')
    #     wbd = np.fromfile(waterBodyFile, dtype=np.int8).reshape(length, width)
    #     cor[np.nonzero(wbd!=0)] = 0.00001

    #minimize the effect of low coherence pixels
    #cor[np.nonzero( (cor<0.85)*(cor!=0) )] = 0.00001
    #filt = adaptive_gaussian(ion, cor, size_max, size_min)
    #cor**14 should be a good weight to use. 22-APR-2018
    #filt = adaptive_gaussian_v0(ion, cor**corOrderFilt, size_max, size_min)


    #1. compute number of looks
    azimuthBandwidth = 0
    for i, frameNumber in enumerate(self._insar.referenceFrames):
        for j, swathNumber in enumerate(range(self._insar.startingSwath, self._insar.endingSwath + 1)):
            #azimuthBandwidth += 2270.575 * 0.85
            azimuthBandwidth += referenceTrack.frames[i].swaths[j].azimuthBandwidth
    azimuthBandwidth = azimuthBandwidth / (len(self._insar.referenceFrames)*(self._insar.endingSwath-self._insar.startingSwath+1))

    #azimuth number of looks should also apply to burst mode
    #assume range bandwidth of subband image is 1/3 of orginal range bandwidth, as in runIonSubband.py!!!
    numberOfLooks = referenceTrack.azimuthLineInterval * self._insar.numberAzimuthLooks1*self._insar.numberAzimuthLooksIon / (1.0/azimuthBandwidth) *\
                    referenceTrack.frames[0].swaths[0].rangeBandwidth / 3.0 / referenceTrack.rangeSamplingRate * self._insar.numberRangeLooks1*self._insar.numberRangeLooksIon

    #consider also burst characteristics. In ScanSAR-stripmap interferometry, azimuthBandwidth is from referenceTrack (ScanSAR)
    if self._insar.modeCombination in [21, 31]:
        numberOfLooks /= 5.0
    if self._insar.modeCombination in [22, 32]:
        numberOfLooks /= 7.0
    if self._insar.modeCombination in [21]:
        numberOfLooks *= (self._insar.burstSynchronization/100.0)

    #numberOfLooks checked
    print('number of looks to be used for computing subband interferogram standard deviation: {}'.format(numberOfLooks))
    if catalog is not None:
        catalog.addItem('number of looks of subband interferograms', numberOfLooks, 'runIonFilt')
    log += 'number of looks of subband interferograms: {}\n'.format(numberOfLooks)


    #2. compute standard deviation of the raw ionospheric phase
    #f0 same as in runIonSubband.py!!!
    def ion_std(fl, fu, numberOfLooks, cor):
        '''
        compute standard deviation of ionospheric phase
        fl:  lower band center frequency
        fu:  upper band center frequency
        cor: coherence, must be numpy array
        '''
        f0 = (fl + fu) / 2.0
        interferogramVar = (1.0 - cor**2) / (2.0 * numberOfLooks * cor**2 + (cor==0))
        std = fl*fu/f0/(fu**2-fl**2)*np.sqrt(fu**2*interferogramVar+fl**2*interferogramVar)
        std[np.nonzero(cor==0)] = 0
        return std
    std = ion_std(fl, fu, numberOfLooks, cor)


    #3. compute minimum filter window size for given coherence and standard deviation of filtered ionospheric phase
    cor2 = np.linspace(0.1, 0.9, num=9, endpoint=True)
    std2 = ion_std(fl, fu, numberOfLooks, cor2)
    std_out2 = np.zeros(cor2.size)
    win2 = np.zeros(cor2.size, dtype=np.int32)
    for i in range(cor2.size):
        for size in range(9, 10001, 2):
            #this window must be the same as those used in adaptive_gaussian!!!
            gw = gaussian(size, size/2.0, scale=1.0)
            scale = 1.0 / np.sum(gw / std2[i]**2)
            std_out2[i] = scale * np.sqrt(np.sum(gw**2 / std2[i]**2))
            win2[i] = size
            if std_out2[i] <= std_out0:
                break
    print('if ionospheric phase standard deviation <= {} rad, minimum filtering window size required:'.format(std_out0))
    print('coherence   window size')
    print('************************')
    for x, y in zip(cor2, win2):
        print('  %5.2f       %5d'%(x, y))
    print()
    if catalog is not None:
        catalog.addItem('coherence value', cor2, 'runIonFilt')
        catalog.addItem('minimum filter window size', win2, 'runIonFilt')
    log += 'coherence value: {}\n'.format(cor2)
    log += 'minimum filter window size: {}\n'.format(win2)


    #4. filter interferogram
    #fit ionosphere
    if fit:
        #prepare weight
        wgt = std**2
        wgt[np.nonzero(cor<corThresholdFit)] = 0
        index = np.nonzero(wgt!=0)
        wgt[index] = 1.0/(wgt[index])
        #fit
        ion_fit, coeff = polyfit_2d(ion, wgt, 2)
        ion -= ion_fit * (ion!=0)
    #filter the rest of the ionosphere
    if filt:
        (ion_filt, std_out, window_size_out) = adaptive_gaussian(ion, std, size_min, size_max, std_out0, fit=fitAdaptive)
        if filtSecondary:
            g2d = gaussian(size_secondary, size_secondary/2.0, scale=1.0)
            scale = ss.fftconvolve((ion_filt!=0), g2d, mode='same')
            ion_filt = (ion_filt!=0) * ss.fftconvolve(ion_filt, g2d, mode='same') / (scale + (scale==0))

    if catalog is not None:
        catalog.addItem('standard deviation of filtered ionospheric phase', std_out0, 'runIonFilt')
    log += 'standard deviation of filtered ionospheric phase: {}\n'.format(std_out0)


    #get final results
    if (fit == True) and (filt == True):
        ion_final = ion_filt + ion_fit * (ion_filt!=0)
    elif (fit == True) and (filt == False):
        ion_final = ion_fit
    elif (fit == False) and (filt == True):
        ion_final = ion_filt
    else:
        ion_final = ion

    #output results
    ion_final.astype(np.float32).tofile(ionfiltfile)
    create_xml(ionfiltfile, width, length, 'float')
    if filt == True:
        std_out.astype(np.float32).tofile(stdfiltfile)
        create_xml(stdfiltfile, width, length, 'float')
        window_size_out.astype(np.float32).tofile(windowsizefiltfile)
        create_xml(windowsizefiltfile, width, length, 'float')


    os.chdir('../../')

    return log



def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='unwrap subband interferograms for ionospheric correction')
    parser.add_argument('-idir', dest='idir', type=str, required=True,
            help = 'input directory where resampled data of each date (YYMMDD) is located. only folders are recognized')
    parser.add_argument('-idir2', dest='idir2', type=str, required=True,
            help = 'input directory where original data of each date (YYMMDD) is located. only folders are recognized')
    parser.add_argument('-ref_date_stack', dest='ref_date_stack', type=str, required=True,
            help = 'reference date of stack. format: YYMMDD')
    parser.add_argument('-ref_date', dest='ref_date', type=str, required=True,
            help = 'reference date of this pair. format: YYMMDD')
    parser.add_argument('-sec_date', dest='sec_date', type=str, required=True,
            help = 'reference date of this pair. format: YYMMDD')
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
    parser.add_argument('-fit', dest='fit', action='store_true', default=False,
            help='apply polynomial fit before filtering ionosphere phase')
    parser.add_argument('-filt', dest='filt', action='store_true', default=False,
            help='filtering ionosphere phase')
    parser.add_argument('-fit_adaptive', dest='fit_adaptive', action='store_true', default=False,
            help='apply polynomial fit in adaptive filtering window')
    parser.add_argument('-filt_secondary', dest='filt_secondary', action='store_true', default=False,
            help='secondary filtering of ionosphere phase')
    parser.add_argument('-win_min', dest='win_min', type=int, default=11,
            help = 'minimum filtering window size. default: 11')
    parser.add_argument('-win_max', dest='win_max', type=int, default=301,
            help = 'maximum filtering window size. default: 301')
    parser.add_argument('-win_secondary', dest='win_secondary', type=int, default=5,
            help = 'secondary filtering window size. default: 5')
    parser.add_argument('-filter_std_ion', dest='filter_std_ion', type=float, default=None,
            help = 'standard deviation after ionosphere filtering. default: None, automatically set by the program')
    parser.add_argument('-masked_areas', dest='masked_areas', type=int, nargs='+', action='append', default=None,
            help='This is a 2-d list. Each element in the 2-D list is a four-element list: [firstLine, lastLine, firstColumn, lastColumn], with line/column numbers starting with 1. If one of the four elements is specified with -1, the program will use firstLine/lastLine/firstColumn/lastColumn instead. e.g. two areas masked out: -masked_areas 10 20 10 20 -masked_areas 110 120 110 120')

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
    idir2 = inps.idir2
    dateReferenceStack = inps.ref_date_stack
    dateReference = inps.ref_date
    dateSecondary = inps.sec_date
    numberRangeLooks1 = inps.nrlks1
    numberAzimuthLooks1 = inps.nalks1
    numberRangeLooks2 = inps.nrlks2
    numberAzimuthLooks2 = inps.nalks2
    numberRangeLooksIon = inps.nrlks_ion
    numberAzimuthLooksIon = inps.nalks_ion
    fitIon = inps.fit
    filtIon = inps.filt
    fitAdaptiveIon = inps.fit_adaptive
    filtSecondaryIon = inps.filt_secondary
    filteringWinsizeMinIon = inps.win_min
    filteringWinsizeMaxIon = inps.win_max
    filteringWinsizeSecondaryIon = inps.win_secondary
    filterStdIon = inps.filter_std_ion
    maskedAreasIon = inps.masked_areas

    #######################################################

    pair = '{}-{}'.format(dateReference, dateSecondary)
    ms = pair
    ml1 = '_{}rlks_{}alks'.format(numberRangeLooks1, numberAzimuthLooks1)
    ml2 = '_{}rlks_{}alks'.format(numberRangeLooks1*numberRangeLooks2, numberAzimuthLooks1*numberAzimuthLooks2)
    dateDirs,   dates,   frames,   swaths,   dateIndexReference = stackDateStatistics(idir, dateReferenceStack)
    dateDirs2,   dates2,   frames2,   swaths2,   dateIndexReference2 = stackDateStatistics(idir2, dateReferenceStack)
    spotlightModes, stripmapModes, scansarNominalModes, scansarWideModes, scansarModes = acquisitionModesAlos2()
    trackReferenceStack = loadTrack(os.path.join(idir, dateReferenceStack), dateReferenceStack)
    trackReference = loadTrack(os.path.join(idir2, dateReference), dateReference)
    trackSecondary = loadTrack(os.path.join(idir2, dateSecondary), dateSecondary)
    subbandRadarWavelength, subbandBandWidth, subbandFrequencyCenter, subbandPrefix = subbandParameters(trackReferenceStack)

    self = createObject()
    self._insar = createObject()
    self._insar.numberRangeLooks1 = numberRangeLooks1
    self._insar.numberAzimuthLooks1 = numberAzimuthLooks1
    self._insar.numberRangeLooks2 = numberRangeLooks2
    self._insar.numberAzimuthLooks2 = numberAzimuthLooks2
    self._insar.numberRangeLooksIon = numberRangeLooksIon
    self._insar.numberAzimuthLooksIon = numberAzimuthLooksIon

    self.fitIon = fitIon
    self.filtIon = filtIon
    self.fitAdaptiveIon = fitAdaptiveIon
    self.filtSecondaryIon = filtSecondaryIon
    self.filteringWinsizeMaxIon = filteringWinsizeMaxIon
    self.filteringWinsizeMinIon = filteringWinsizeMinIon
    self.filteringWinsizeSecondaryIon = filteringWinsizeSecondaryIon
    self.maskedAreasIon = maskedAreasIon
    self.applyIon = False

    #ionospheric phase standard deviation after filtering
    if filterStdIon is not None:
        self.filterStdIon = filterStdIon
    else:
        if trackReference.operationMode == trackSecondary.operationMode:
            from isceobj.Alos2Proc.Alos2ProcPublic import modeProcParDict
            self.filterStdIon = modeProcParDict['ALOS-2'][trackReference.operationMode]['filterStdIon']
        else:
            from isceobj.Alos2Proc.Alos2ProcPublic import filterStdPolyIon
            self.filterStdIon = np.polyval(filterStdPolyIon, trackReference.frames[0].swaths[0].rangeBandwidth/(1e6))

    self._insar.referenceFrames = frames
    self._insar.startingSwath = swaths[0]
    self._insar.endingSwath = swaths[-1]
    self._insar.subbandRadarWavelength = subbandRadarWavelength

    self._insar.multilookIon = ms + ml2 + '.ion'
    self._insar.multilookDifferentialInterferogram = 'diff_' + ms + ml2 + '.int'
    self._insar.multilookDifferentialInterferogramOriginal = 'diff_' + ms + ml2 + '_ori.int'

    #usable combinations
    referenceMode = trackReference.operationMode
    secondaryMode = trackSecondary.operationMode
    if (referenceMode in spotlightModes) and (secondaryMode in spotlightModes):
        self._insar.modeCombination = 0
    elif (referenceMode in stripmapModes) and (secondaryMode in stripmapModes):
        self._insar.modeCombination = 1
    elif (referenceMode in scansarNominalModes) and (secondaryMode in scansarNominalModes):
        self._insar.modeCombination = 21
    elif (referenceMode in scansarWideModes) and (secondaryMode in scansarWideModes):
        self._insar.modeCombination = 22
    elif (referenceMode in scansarNominalModes) and (secondaryMode in stripmapModes):
        self._insar.modeCombination = 31
    elif (referenceMode in scansarWideModes) and (secondaryMode in stripmapModes):
        self._insar.modeCombination = 32
    else:
        print('\n\nthis mode combination is not possible')
        print('note that for ScanSAR-stripmap, ScanSAR must be reference\n\n')
        raise Exception('mode combination not supported')

    if self._insar.modeCombination in [21]:
        unsynTimeAll, synPercentageAll = computeBurstSynchronization(trackReference, trackSecondary)
        self._insar.burstSynchronization = np.mean(np.array(synPercentageAll), dtype=np.float64)
    else:
        self._insar.burstSynchronization = 100.0

    #log output info
    log  = '{} at {}\n'.format(os.path.basename(__file__), datetime.datetime.now())
    log += '================================================================================================\n'
    log += ionFilt(self, trackReferenceStack)
    log += '\n'

    logFile = 'process.log'
    with open(logFile, 'a') as f:
        f.write(log)