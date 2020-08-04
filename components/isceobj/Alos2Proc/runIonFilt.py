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

    #################################################
    #SET PARAMETERS HERE
    #if applying polynomial fitting
    #False: no fitting, True: with fitting
    fit = self.fitIon
    #gaussian filtering window size
    size_max = self.filteringWinsizeMaxIon
    size_min = self.filteringWinsizeMinIon

    if size_min >= size_max:
        print('\n\nWARNING: minimum window size for filtering ionosphere phase {} >= maximum window size {}'.format(size_min, size_max))
        print('         resetting maximum window size to {}\n\n'.format(size_min+5))
        size_max = size_min + 5

    #THESE SHOULD BE GOOD ENOUGH, NO NEED TO SET IN setup(self)
    #corThresholdFit = 0.85

    #Now changed to use lower band coherence. crl, 23-apr-2020.
    useDiffCoherence = False
    if useDiffCoherence:
        #parameters for using diff coherence
        corfile = 'diff'+ml2+'.cor'
        corThresholdFit = 0.95
        # 1 is not good for low coherence case, changed to 20
        #corOrderFit = 1
        corOrderFit = 20
        corOrderFilt = 14
    else:
        #parameters for using lower/upper band coherence
        corfile = subbandPrefix[0]+ml2+'.cor'
        corThresholdFit = 0.4
        corOrderFit = 10
        corOrderFilt = 4

    #################################################

    print('\nfiltering ionosphere')
    ionfile = 'ion'+ml2+'.ion'
    #corfile = 'diff'+ml2+'.cor'
    ionfiltfile = 'filt_ion'+ml2+'.ion'

    img = isceobj.createImage()
    img.load(ionfile + '.xml')
    width = img.width
    length = img.length
    #ion = (np.fromfile(ionfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    ion = np.fromfile(ionfile, dtype=np.float32).reshape(length, width)
    cor = (np.fromfile(corfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    #amp = (np.fromfile(ionfile, dtype=np.float32).reshape(length*2, width))[0:length*2:2, :]

    #masked out user-specified areas
    if self.maskedAreasIon != None:
        maskedAreas = reformatMaskedAreas(self.maskedAreasIon, length, width)
        for area in maskedAreas:
            ion[area[0]:area[1], area[2]:area[3]] = 0
            cor[area[0]:area[1], area[2]:area[3]] = 0

    #remove possible wired values in coherence
    cor[np.nonzero(cor<0)] = 0.0
    cor[np.nonzero(cor>1)] = 0.0

    #remove water body
    wbd = np.fromfile('wbd'+ml2+'.wbd', dtype=np.int8).reshape(length, width)
    cor[np.nonzero(wbd==-1)] = 0.0

    # #applying water body mask here
    # waterBodyFile = 'wbd'+ml2+'.wbd'
    # if os.path.isfile(waterBodyFile):
    #     print('applying water body mask to coherence used to compute ionospheric phase')
    #     wbd = np.fromfile(waterBodyFile, dtype=np.int8).reshape(length, width)
    #     cor[np.nonzero(wbd!=0)] = 0.00001

    if fit:
        import copy
        wgt = copy.deepcopy(cor)
        wgt[np.nonzero(wgt<corThresholdFit)] = 0.0
        ion_fit = weight_fitting(ion, wgt**corOrderFit, width, length, 1, 1, 1, 1, 2)
        ion -= ion_fit * (ion!=0)

    #minimize the effect of low coherence pixels
    #cor[np.nonzero( (cor<0.85)*(cor!=0) )] = 0.00001
    #filt = adaptive_gaussian(ion, cor, size_max, size_min)
    #cor**14 should be a good weight to use. 22-APR-2018
    filt = adaptive_gaussian(ion, cor**corOrderFilt, size_max, size_min)

    if fit:
        filt += ion_fit * (filt!=0)

    # ion = np.zeros((length*2, width), dtype=np.float32)
    # ion[0:length*2:2, :] = amp
    # ion[1:length*2:2, :] = filt
    # ion.astype(np.float32).tofile(ionfiltfile)
    # img.filename = ionfiltfile
    # img.extraFilename = ionfiltfile + '.vrt'
    # img.renderHdr()

    filt.astype(np.float32).tofile(ionfiltfile)
    create_xml(ionfiltfile, width, length, 'float')


    ############################################################
    # STEP 3. resample ionospheric phase
    ############################################################
    from contrib.alos2proc_f.alos2proc_f import rect
    from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
    from scipy.interpolate import interp1d
    import shutil

    #################################################
    #SET PARAMETERS HERE
    #interpolation method
    interpolationMethod = 1
    #################################################

    print('\ninterpolate ionosphere')

    ml3 = '_{}rlks_{}alks'.format(self._insar.numberRangeLooks1*self._insar.numberRangeLooks2, 
                              self._insar.numberAzimuthLooks1*self._insar.numberAzimuthLooks2)

    ionfiltfile = 'filt_ion'+ml2+'.ion'
    #ionrectfile = 'filt_ion'+ml3+'.ion'
    ionrectfile = self._insar.multilookIon

    img = isceobj.createImage()
    img.load(ionfiltfile + '.xml')
    width2 = img.width
    length2 = img.length

    img = isceobj.createImage()
    img.load(os.path.join('../../', ionDir['insar'], self._insar.multilookDifferentialInterferogram) + '.xml')
    width3 = img.width
    length3 = img.length

    #number of range looks output
    nrlo = self._insar.numberRangeLooks1*self._insar.numberRangeLooks2
    #number of range looks input
    nrli = self._insar.numberRangeLooks1*self._insar.numberRangeLooksIon
    #number of azimuth looks output
    nalo = self._insar.numberAzimuthLooks1*self._insar.numberAzimuthLooks2
    #number of azimuth looks input
    nali = self._insar.numberAzimuthLooks1*self._insar.numberAzimuthLooksIon

    if (self._insar.numberRangeLooks2 != self._insar.numberRangeLooksIon) or \
       (self._insar.numberAzimuthLooks2 != self._insar.numberAzimuthLooksIon):
        #this should be faster using fortran
        if interpolationMethod == 0:
            rect(ionfiltfile, ionrectfile,
                width2,length2,
                width3,length3,
                nrlo/nrli, 0.0,
                0.0, nalo/nali,
                (nrlo-nrli)/(2.0*nrli),
                (nalo-nali)/(2.0*nali),
                'REAL','Bilinear')
        #finer, but slower method
        else:
            ionfilt = np.fromfile(ionfiltfile, dtype=np.float32).reshape(length2, width2)
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
            ionrect.astype(np.float32).tofile(ionrectfile)
            del ionrect
        create_xml(ionrectfile, width3, length3, 'float')

        os.rename(ionrectfile, os.path.join('../../insar', ionrectfile))
        os.rename(ionrectfile+'.vrt', os.path.join('../../insar', ionrectfile)+'.vrt')
        os.rename(ionrectfile+'.xml', os.path.join('../../insar', ionrectfile)+'.xml')
        os.chdir('../../insar')
    else:
        shutil.copyfile(ionfiltfile, os.path.join('../../insar', ionrectfile))
        os.chdir('../../insar')
        create_xml(ionrectfile, width3, length3, 'float')
    #now we are in 'insar'


    ############################################################
    # STEP 4. correct interferogram
    ############################################################
    from isceobj.Alos2Proc.Alos2ProcPublic import renameFile
    from isceobj.Alos2Proc.Alos2ProcPublic import runCmd

    if self.applyIon:
        print('\ncorrect interferogram')
        if os.path.isfile(self._insar.multilookDifferentialInterferogramOriginal):
            print('original interferogram: {} is already here, do not rename: {}'.format(self._insar.multilookDifferentialInterferogramOriginal, self._insar.multilookDifferentialInterferogram))
        else:
            print('renaming {} to {}'.format(self._insar.multilookDifferentialInterferogram, self._insar.multilookDifferentialInterferogramOriginal))
            renameFile(self._insar.multilookDifferentialInterferogram, self._insar.multilookDifferentialInterferogramOriginal)

        cmd = "imageMath.py -e='a*exp(-1.0*J*b)' --a={} --b={} -s BIP -t cfloat -o {}".format(
            self._insar.multilookDifferentialInterferogramOriginal,
            self._insar.multilookIon,
            self._insar.multilookDifferentialInterferogram)
        runCmd(cmd)
    else:
        print('\nionospheric phase estimation finished, but correction of interfeorgram not requested')

    os.chdir('../')

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
        diff = weight_fitting(lowerUnw - upperUnw, wgt, width, length, 1, 1, 1, 1, 2)

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


def fit_surface(x, y, z, wgt, order):
    # x: x coordinate, a column vector
    # y: y coordinate, a column vector
    # z: z coordinate, a column vector
    # wgt: weight of the data points, a column vector


    #number of data points
    m = x.shape[0]
    l = np.ones((m,1), dtype=np.float64)

#    #create polynomial
#    if order == 1:
#        #order of estimated coefficents: 1, x, y
#        a1 = np.concatenate((l, x, y), axis=1)
#    elif order == 2:
#        #order of estimated coefficents: 1, x, y, x*y, x**2, y**2
#        a1 = np.concatenate((l, x, y, x*y, x**2, y**2), axis=1)
#    elif order == 3:
#        #order of estimated coefficents: 1, x, y, x*y, x**2, y**2, x**2*y, y**2*x, x**3, y**3
#        a1 = np.concatenate((l, x, y, x*y, x**2, y**2, x**2*y, y**2*x, x**3, y**3), axis=1)
#    else:
#        raise Exception('order not supported yet\n')

    if order < 1:
        raise Exception('order must be larger than 1.\n')

    #create polynomial
    a1 = l;
    for i in range(1, order+1):
        for j in range(i+1):
            a1 = np.concatenate((a1, x**(i-j)*y**(j)), axis=1)

    #number of variable to be estimated
    n = a1.shape[1]

    #do the least squares
    a = a1 * np.matlib.repmat(np.sqrt(wgt), 1, n)
    b = z * np.sqrt(wgt)
    c = np.linalg.lstsq(a, b, rcond=-1)[0]
    
    #type: <class 'numpy.ndarray'>
    return c


def cal_surface(x, y, c, order):
    #x: x coordinate, a row vector
    #y: y coordinate, a column vector
    #c: coefficients of polynomial from fit_surface
    #order: order of polynomial

    if order < 1:
        raise Exception('order must be larger than 1.\n')

    #number of lines
    length = y.shape[0]
    #number of columns, if row vector, only one element in the shape tuple
    #width = x.shape[1]
    width = x.shape[0]

    x = np.matlib.repmat(x, length, 1)
    y = np.matlib.repmat(y, 1, width)
    z = c[0] * np.ones((length,width), dtype=np.float64)

    index = 0
    for i in range(1, order+1):
        for j in range(i+1):
            index += 1
            z += c[index] * x**(i-j)*y**(j)

    return z


def weight_fitting(ionos, weight, width, length, nrli, nali, nrlo, nalo, order):
    '''
    ionos:  input ionospheric phase
    weight: weight
    width:  file width
    length: file length
    nrli:   number of range looks of the input interferograms
    nali:   number of azimuth looks of the input interferograms
    nrlo:   number of range looks of the output ionosphere phase
    nalo:   number of azimuth looks of the ioutput ionosphere phase
    order:  the order of the polynomial for fitting ionosphere phase estimates
    '''

    from isceobj.Alos2Proc.Alos2ProcPublic import create_multi_index2

    lengthi = int(length/nali)
    widthi = int(width/nrli)
    lengtho = int(length/nalo)
    widtho = int(width/nrlo)

    #calculate output index
    rgindex = create_multi_index2(widtho, nrli, nrlo)
    azindex = create_multi_index2(lengtho, nali, nalo)

    #look for data to use
    flag = (weight!=0)*(ionos!=0)
    point_index = np.nonzero(flag)
    m = point_index[0].shape[0]

    #calculate input index matrix
    x0=np.matlib.repmat(np.arange(widthi), lengthi, 1)
    y0=np.matlib.repmat(np.arange(lengthi).reshape(lengthi, 1), 1, widthi)

    x = x0[point_index].reshape(m, 1)
    y = y0[point_index].reshape(m, 1)
    z = ionos[point_index].reshape(m, 1)
    w = weight[point_index].reshape(m, 1)

    #convert to higher precision type before use
    x=np.asfarray(x,np.float64)
    y=np.asfarray(y,np.float64)
    z=np.asfarray(z,np.float64)
    w=np.asfarray(w,np.float64)
    coeff = fit_surface(x, y, z, w, order)

    #convert to higher precision type before use
    rgindex=np.asfarray(rgindex,np.float64)
    azindex=np.asfarray(azindex,np.float64)
    phase_fit = cal_surface(rgindex, azindex.reshape(lengtho, 1), coeff, order)

    #format: widtho, lengtho, single band float32
    return phase_fit


def gaussian(size, sigma, scale = 1.0):

    if size % 2 != 1:
        raise Exception('size must be odd')
    hsize = (size - 1) / 2
    x = np.arange(-hsize, hsize + 1) * scale
    f = np.exp(-x**2/(2.0*sigma**2)) / (sigma * np.sqrt(2.0*np.pi))
    f2d=np.matlib.repmat(f, size, 1) * np.matlib.repmat(f.reshape(size, 1), 1, size)

    return f2d/np.sum(f2d)


def adaptive_gaussian(ionos, wgt, size_max, size_min):
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
