#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging
import numpy as np
import numpy.matlib

import isceobj

logger = logging.getLogger('isce.alos2insar.runIonFilt')

def runIonFilt(self):
    '''compute and filter ionospheric phase
    '''
    if hasattr(self, 'doInSAR'):
        if not self.doInSAR:
            return

    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    if not self.doIon:
        catalog.printToLog(logger, "runIonFilt")
        self._insar.procDoc.addAllFromCatalog(catalog)
        return

    referenceTrack = self._insar.loadTrack(reference=True)
    secondaryTrack = self._insar.loadTrack(reference=False)

    from isceobj.Alos2Proc.runIonSubband import defineIonDir
    ionDir = defineIonDir()
    subbandPrefix = ['lower', 'upper']

    ionCalDir = os.path.join(ionDir['ion'], ionDir['ionCal'])
    os.makedirs(ionCalDir, exist_ok=True)
    os.chdir(ionCalDir)


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
    if self.filterStdIon is not None:
        std_out0 = self.filterStdIon
    else:
        if referenceTrack.operationMode == secondaryTrack.operationMode:
            from isceobj.Alos2Proc.Alos2ProcPublic import modeProcParDict
            std_out0 = modeProcParDict['ALOS-2'][referenceTrack.operationMode]['filterStdIon']
        else:
            from isceobj.Alos2Proc.Alos2ProcPublic import filterStdPolyIon
            std_out0 = np.polyval(filterStdPolyIon, referenceTrack.frames[0].swaths[0].rangeBandwidth/(1e6))
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
    catalog.addItem('number of looks of subband interferograms', numberOfLooks, 'runIonFilt')


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
    catalog.addItem('coherence value', cor2, 'runIonFilt')
    catalog.addItem('minimum filter window size', win2, 'runIonFilt')


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
            print('applying secondary filtering with window size {}'.format(size_secondary))
            g2d = gaussian(size_secondary, size_secondary/2.0, scale=1.0)
            scale = ss.fftconvolve((ion_filt!=0), g2d, mode='same')
            ion_filt = (ion_filt!=0) * ss.fftconvolve(ion_filt, g2d, mode='same') / (scale + (scale==0))
    catalog.addItem('standard deviation of filtered ionospheric phase', std_out0, 'runIonFilt')

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

    catalog.printToLog(logger, "runIonFilt")
    self._insar.procDoc.addAllFromCatalog(catalog)



def computeIonosphere(lowerUnw, upperUnw, wgt, fl, fu, adjFlag, dispersive):
    '''
    This routine computes ionosphere and remove the relative phase unwrapping errors

    lowerUnw:        lower band unwrapped interferogram
    upperUnw:        upper band unwrapped interferogram
    wgt:             weight
    fl:              lower band center frequency
    fu:              upper band center frequency
    adjFlag:         method for removing relative phase unwrapping errors
                       0: mean value
                       1: polynomial
    dispersive:      compute dispersive or non-dispersive
                       0: dispersive
                       1: non-dispersive
    '''

    #use image size from lower unwrapped interferogram
    (length, width)=lowerUnw.shape

##########################################################################################
    # ADJUST PHASE USING MEAN VALUE
    # #ajust phase of upper band to remove relative phase unwrapping errors
    # flag = (lowerUnw!=0)*(cor>=ionParam.corThresholdAdj)
    # index = np.nonzero(flag!=0)
    # mv = np.mean((lowerUnw - upperUnw)[index], dtype=np.float64)
    # print('mean value of phase difference: {}'.format(mv))
    # flag2 = (lowerUnw!=0)
    # index2 = np.nonzero(flag2)
    # #phase for adjustment
    # unwd = ((lowerUnw - upperUnw)[index2] - mv) / (2.0*np.pi)
    # unw_adj = np.around(unwd) * (2.0*np.pi)
    # #ajust phase of upper band
    # upperUnw[index2] += unw_adj
    # unw_diff = lowerUnw - upperUnw
    # print('after adjustment:')
    # print('max phase difference: {}'.format(np.amax(unw_diff)))
    # print('min phase difference: {}'.format(np.amin(unw_diff)))
##########################################################################################
    #adjust phase using mean value
    if adjFlag == 0:
        flag = (lowerUnw!=0)*(wgt!=0)
        index = np.nonzero(flag!=0)
        mv = np.mean((lowerUnw - upperUnw)[index], dtype=np.float64)
        print('mean value of phase difference: {}'.format(mv))
        diff = mv
    #adjust phase using a surface
    else:
        #diff = weight_fitting(lowerUnw - upperUnw, wgt, width, length, 1, 1, 1, 1, 2)
        diff, coeff = polyfit_2d(lowerUnw - upperUnw, wgt, 2)

    flag2 = (lowerUnw!=0)
    index2 = np.nonzero(flag2)
    #phase for adjustment
    unwd = ((lowerUnw - upperUnw) - diff)[index2] / (2.0*np.pi)
    unw_adj = np.around(unwd) * (2.0*np.pi)
    #ajust phase of upper band
    upperUnw[index2] += unw_adj

    unw_diff = (lowerUnw - upperUnw)[index2]
    print('after adjustment:')
    print('max phase difference: {}'.format(np.amax(unw_diff)))
    print('min phase difference: {}'.format(np.amin(unw_diff)))
    print('max-min: {}'.format(np.amax(unw_diff) - np.amin(unw_diff)    ))

    #ionosphere
    #fl = SPEED_OF_LIGHT / ionParam.radarWavelengthLower
    #fu = SPEED_OF_LIGHT / ionParam.radarWavelengthUpper
    f0 = (fl + fu) / 2.0
    
    #dispersive
    if dispersive == 0:
        ionos = fl * fu * (lowerUnw * fu - upperUnw * fl) / f0 / (fu**2 - fl**2)
    #non-dispersive phase
    else:
        ionos = f0 * (upperUnw*fu - lowerUnw * fl) / (fu**2 - fl**2)

    return ionos


def gaussian(size, sigma, scale = 1.0):

    if size % 2 != 1:
        raise Exception('size must be odd')
    hsize = (size - 1) / 2
    x = np.arange(-hsize, hsize + 1) * scale
    f = np.exp(-x**2/(2.0*sigma**2)) / (sigma * np.sqrt(2.0*np.pi))
    f2d=np.matlib.repmat(f, size, 1) * np.matlib.repmat(f.reshape(size, 1), 1, size)

    return f2d/np.sum(f2d)


def adaptive_gaussian_v0(ionos, wgt, size_max, size_min):
    '''
    This program performs Gaussian filtering with adaptive window size.
    ionos: ionosphere
    wgt: weight
    size_max: maximum window size
    size_min: minimum window size
    '''
    import scipy.signal as ss

    length = (ionos.shape)[0]
    width = (ionos.shape)[1]
    flag = (ionos!=0) * (wgt!=0)
    ionos *= flag
    wgt *= flag

    size_num = 100
    size = np.linspace(size_min, size_max, num=size_num, endpoint=True)
    std = np.zeros((length, width, size_num))
    flt = np.zeros((length, width, size_num))
    out = np.zeros((length, width, 1))

    #calculate filterd image and standard deviation
    #sigma of window size: size_max
    sigma = size_max / 2.0
    for i in range(size_num):
        size2 = np.int(np.around(size[i]))
        if size2 % 2 == 0:
            size2 += 1
        if (i+1) % 10 == 0:
            print('min win: %4d, max win: %4d, current win: %4d'%(np.int(np.around(size_min)), np.int(np.around(size_max)), size2))
        g2d = gaussian(size2, sigma*size2/size_max, scale=1.0)
        scale = ss.fftconvolve(wgt, g2d, mode='same')
        flt[:, :, i] = ss.fftconvolve(ionos*wgt, g2d, mode='same') / (scale + (scale==0))
        #variance of resulting filtered sample
        scale = scale**2
        var = ss.fftconvolve(wgt, g2d**2, mode='same') / (scale + (scale==0))
        #in case there is a large area without data where scale is very small, which leads to wired values in variance
        var[np.nonzero(var<0)] = 0
        std[:, :, i] = np.sqrt(var)

    std_mv = np.mean(std[np.nonzero(std!=0)], dtype=np.float64)
    diff_max = np.amax(np.absolute(std - std_mv)) + std_mv + 1
    std[np.nonzero(std==0)] = diff_max
    
    index = np.nonzero(np.ones((length, width))) + ((np.argmin(np.absolute(std - std_mv), axis=2)).reshape(length*width), )
    out = flt[index]
    out = out.reshape((length, width))

    #remove artifacts due to varying wgt
    size_smt = size_min
    if size_smt % 2 == 0:
        size_smt += 1
    g2d = gaussian(size_smt, size_smt/2.0, scale=1.0)
    scale = ss.fftconvolve((out!=0), g2d, mode='same')
    out2 = ss.fftconvolve(out, g2d, mode='same') / (scale + (scale==0))

    return out2


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


def polyfit_2d(data, weight, order):
    '''
    fit a surface to a 2-d matrix

    data:   input 2-d data
    weight: corresponding 2-d weight
    order:  order. must >= 1

    zero samples in data and weight are OK.
    '''
    #import numpy as np

    if order < 1:
        raise Exception('order must >= 1!\n')

    if data.shape != weight.shape:
        raise Exception('data and weight must be of same size!\n')

    (length, width) = data.shape
    #length*width, but below is better since no need to convert to int
    n = data.size

    #number of coefficients
    ncoeff = 1
    for i in range(1, order+1):
        for j in range(i+1):
            ncoeff += 1

    #row, column
    y, x = np.indices((length, width))
    x = x.flatten()
    y = y.flatten()
    z = data.flatten()
    weight = np.sqrt(weight.flatten())

    #linear functions: H theta = s
    #compute observation matrix H (n*ncoeff)
    H = np.zeros((n, ncoeff))
    H[:,0] += 1
    k = 1
    for i in range(1, order+1):
        for j in range(i+1):
            #x and y do not need to be column vector here
            H[:, k] = x**(i-j)*y**(j)
            k += 1

    #least squares
    #this is robust to singular cases
    coeff = np.linalg.lstsq(H*weight[:,None], z*weight, rcond=-1)[0]
    #this uses multiple threads, should be faster
    #coeff = least_sqares(H*weight[:,None], z*weight, W=None)

    #fit surface
    data_fit = (np.dot(H, coeff)).reshape(length, width)

    return (data_fit, coeff)


def adaptive_gaussian(data, std, size_min, size_max, std_out0, fit=True):
    '''
    This program performs Gaussian filtering with adaptive window size.
    Cunren Liang, 11-JUN-2020

    data:     input raw data, numpy array
    std:      standard deviation of raw data, numpy array
    size_min: minimum filter window size
    size_max: maximum filter window size (size_min <= size_max, size_min == size_max is allowed)
    std_out0: standard deviation of output data
    fit:      whether do fitting before gaussian filtering
    '''
    import scipy.signal as ss


    (length, width) = data.shape

    #assume zero-value samples are invalid
    index = np.nonzero(np.logical_or(data==0, std==0))
    data[index] = 0
    std[index] = 0
    #compute weight using standard deviation
    wgt = 1.0 / (std**2 + (std==0))
    wgt[index] = 0

    #compute number of gaussian filters
    if size_min > size_max:
        raise Exception('size_min: {} > size_max: {}\n'.format(size_min, size_max))

    if size_min % 2 == 0:
        size_min += 1
    if size_max % 2 == 0:
        size_max += 1

    size_num = int((size_max - size_min) / 2 + 1)
    #'size_num == 1' is checked to be OK starting from here


    #create gaussian filters
    print('compute Gaussian filters\n')
    gaussian_filters = []
    for i in range(size_num):
        size = int(size_min + i * 2)
        gaussian_filters.append(gaussian(size, size/2.0, scale=1.0))


    #compute standard deviation after filtering coresponding to each of gaussian_filters
    #if value is 0, there is no valid sample in the gaussian window
    print('compute standard deviation after filtering for each filtering window size')
    std_filt = np.zeros((length, width, size_num))
    for i in range(size_num):
        size = int(size_min + i * 2)
        print('current window size: %4d, min window size: %4d, max window size: %4d' % (size, size_min, size_max), end='\r', flush=True)
        #robust zero value detector. non-zero convolution result at least >= 1, so can use 0.5 
        #as threshold to detect zero-value result
        index = np.nonzero(ss.fftconvolve(wgt!=0, gaussian_filters[i]!=0, mode='same') < 0.5)
        scale = ss.fftconvolve(wgt, gaussian_filters[i], mode='same')
        scale[index] = 0
        #variance of resulting filtered sample
        var_filt = ss.fftconvolve(wgt, gaussian_filters[i]**2, mode='same') / (scale**2 + (scale==0))
        var_filt[index] = 0
        std_filt[:, :, i] = np.sqrt(var_filt)
    print('\n')


    #find gaussian window size (3rd-dimension index of the window size in gaussian_filters)
    #if value is -1, there is no valid sample in any of the gaussian windows
    #and therefore no filtering in the next step is needed
    print('find Gaussian window size to use')
    gaussian_index = np.zeros((length, width), dtype=np.int32)
    std_filt2 = np.zeros((length, width))
    for i in range(length):
        if (((i+1)%50) == 0):
            print('processing line %6d of %6d' % (i+1, length), end='\r', flush=True)
        for j in range(width):
            if np.sum(std_filt[i, j, :]) == 0:
                gaussian_index[i, j] = -1
            else:
                gaussian_index[i, j] = size_num - 1
                for k in range(size_num):
                    if (std_filt[i, j, k] != 0) and (std_filt[i, j, k] <= std_out0):
                        gaussian_index[i, j] = k
                        break
            if gaussian_index[i, j] != -1:
                std_filt2[i, j] = std_filt[i, j, gaussian_index[i, j]]
    del std_filt
    print("processing line %6d of %6d\n" % (length, length))


    #adaptive gaussian filtering
    print('filter image')
    data_out = np.zeros((length, width))
    std_out = np.zeros((length, width))
    window_size_out = np.zeros((length, width), dtype=np.int16)
    for i in range(length):
        #if (((i+1)%5) == 0):
        print('processing line %6d of %6d' % (i+1, length), end='\r', flush=True)
        for j in range(width):
            #if value is -1, there is no valid sample in any of the gaussian windows
            #and therefore no filtering in the next step is needed
            if gaussian_index[i, j] == -1:
                continue

            #1. extract data
            size = int(size_min + gaussian_index[i, j] * 2)
            size_half = int((size - 1) / 2)
            window_size_out[i, j] = size

            #index in original data
            first_line = max(i-size_half, 0)
            last_line = min(i+size_half, length-1)
            first_column = max(j-size_half, 0)
            last_column = min(j+size_half, width-1)
            length_valid = last_line - first_line + 1
            width_valid = last_column - first_column + 1

            #index in filter window
            if first_line == 0:
                last_line2 = size - 1
                first_line2 = last_line2 - (length_valid - 1)
            else:
                first_line2 = 0
                last_line2 = first_line2 + (length_valid - 1)
            if first_column == 0:
                last_column2 = size - 1
                first_column2 = last_column2 - (width_valid - 1)
            else:
                first_column2 = 0
                last_column2 = first_column2 + (width_valid - 1)

            #prepare data and weight within the window
            data_window = np.zeros((size, size))
            wgt_window = np.zeros((size, size))
            data_window[first_line2:last_line2+1, first_column2:last_column2+1] = data[first_line:last_line+1, first_column:last_column+1]
            wgt_window[first_line2:last_line2+1, first_column2:last_column2+1] = wgt[first_line:last_line+1, first_column:last_column+1]
            #number of valid samples in the filtering window
            n_valid = np.sum(data_window!=0)

            #2. fit
            #order, n_coeff = (1, 3)
            order, n_coeff = (2, 6)
            if fit:
                #must have enough samples to do fitting
                #even if order is 2, n_coeff * 3 is much smaller than size_min*size_min in most cases.
                if n_valid > n_coeff * 3:
                    #data_fit = weight_fitting(data_window, wgt_window, size, size, 1, 1, 1, 1, order)
                    data_fit, coeff = polyfit_2d(data_window, wgt_window, order)
                    index = np.nonzero(data_window!=0)
                    data_window[index] -= data_fit[index]

            #3. filter
            wgt_window_2 = wgt_window * gaussian_filters[gaussian_index[i, j]]
            scale = 1.0/np.sum(wgt_window_2)
            wgt_window_2 *= scale
            data_out[i, j] = np.sum(wgt_window_2 * data_window)
            #std_out[i, j] = scale * np.sqrt(np.sum(wgt_window*(gaussian_filters[gaussian_index[i, j]]**2)))
            #already computed
            std_out[i, j] = std_filt2[i, j]
            #print('std_out[i, j], std_filt2[i, j]', std_out[i, j], std_filt2[i, j])

            #4. add back filtered value
            if fit:
                if n_valid > n_coeff * 3:
                    data_out[i, j] += data_fit[size_half, size_half]
    print('\n')

    return (data_out, std_out, window_size_out)


def reformatMaskedAreas(maskedAreas, length, width):
    '''
    reformat masked areas coordinates that are ready to use
    'maskedAreas' is a 2-D list. Each element in the 2-D list is a four-element list: [firstLine,
    lastLine, firstColumn, lastColumn], with line/column numbers starting with 1. If one of the
    four elements is specified with -1, the program will use firstLine/lastLine/firstColumn/
    lastColumn instead.

    output is a 2-D list containing the corresponding python-list/array-format indexes.
    '''
    numberOfAreas = len(maskedAreas)
    maskedAreasReformated = [[0, length, 0, width] for i in range(numberOfAreas)]

    for i in range(numberOfAreas):
        if maskedAreas[i][0] != -1:
            maskedAreasReformated[i][0] = maskedAreas[i][0] - 1
        if maskedAreas[i][1] != -1:
            maskedAreasReformated[i][1] = maskedAreas[i][1]
        if maskedAreas[i][2] != -1:
            maskedAreasReformated[i][2] = maskedAreas[i][2] - 1
        if maskedAreas[i][3] != -1:
            maskedAreasReformated[i][3] = maskedAreas[i][3]
        if (not (0 <= maskedAreasReformated[i][0] <= length-1)) or \
           (not (1 <= maskedAreasReformated[i][1] <= length)) or \
           (not (0 <= maskedAreasReformated[i][2] <= width-1)) or \
           (not (1 <= maskedAreasReformated[i][3] <= width)) or \
           (not (maskedAreasReformated[i][1]-maskedAreasReformated[i][0]>=1)) or \
           (not (maskedAreasReformated[i][3]-maskedAreasReformated[i][2]>=1)):
            raise Exception('area {} masked out in ionospheric phase estimation not correct'.format(i+1))

    return maskedAreasReformated


