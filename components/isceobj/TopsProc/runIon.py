#
# Author: Cunren Liang
# Copyright 2018
# California Institute of Technology
#

import os
import shutil
import datetime
import numpy as np
import numpy.matlib

import isceobj
import logging
from isceobj.Constants import SPEED_OF_LIGHT
from isceobj.TopsProc.runBurstIfg import loadVirtualArray


logger = logging.getLogger('isce.topsinsar.ion')

#should get rid of the coherence thresholds in the future
##WARNING: when using the original full-bandwidth swath xml file, should also consider burst.image.filename
class dummy(object):
    pass


def setup(self):
    '''
    setup parameters for processing
    '''

    #initialize parameters for ionospheric correction
    ionParam = dummy()
    #The step names in the list below are exactly the function names in 'def runIon(self):'
    #when adding a new step, only put its function name (in right order) in the list,
    #and put the function (in right order) in 'def runIon(self):'
    ionParam.allSteps = ['subband', 'rawion', 'grd2ion', 'filt_gaussian', 'ionosphere_shift', 'ion2grd', 'esd']


    ###################################################################
    #users are supposed to change parameters of this section ONLY
    #SECTION 1. PROCESSING CONTROL PARAMETERS
    #1. suggested default values of the parameters
    ionParam.doIon = False
    ionParam.startStep = ionParam.allSteps[0]
    ionParam.endStep = ionParam.allSteps[-1]

    #ionospheric layer height (km)
    ionParam.ionHeight = 200.0
    #before filtering ionosphere, if applying polynomial fitting
    #False: no fitting
    #True: with fitting
    ionParam.ionFit = True
    #window size for filtering ionosphere
    ionParam.ionFilteringWinsizeMax = 200
    ionParam.ionFilteringWinsizeMin = 100
    #window size for filtering azimuth shift caused by ionosphere
    ionParam.ionshiftFilteringWinsizeMax = 150
    ionParam.ionshiftFilteringWinsizeMin = 75
    #correct phase error caused by non-zero center frequency and azimuth shift caused by ionosphere
    #0: no correction
    #1: use mean value of a burst
    #2: use full burst
    ionParam.azshiftFlag = 1

    #better NOT try changing the following two parameters, since they are related
    #to the filtering parameters above
    #number of azimuth looks in the processing of ionosphere estimation
    ionParam.numberAzimuthLooks = 50
    #number of range looks in the processing of ionosphere estimation
    ionParam.numberRangeLooks = 200
    #number of azimuth looks of the interferogram to be unwrapped
    ionParam.numberAzimuthLooks0 = 5*2
    #number of range looks of the interferogram to be unwrapped
    ionParam.numberRangeLooks0 = 20*2


    #2. accept the above parameters from topsApp.py
    ionParam.doIon = self.ION_doIon
    ionParam.startStep = self.ION_startStep
    ionParam.endStep = self.ION_endStep

    ionParam.ionHeight = self.ION_ionHeight
    ionParam.ionFit = self.ION_ionFit
    ionParam.ionFilteringWinsizeMax = self.ION_ionFilteringWinsizeMax
    ionParam.ionFilteringWinsizeMin = self.ION_ionFilteringWinsizeMin
    ionParam.ionshiftFilteringWinsizeMax = self.ION_ionshiftFilteringWinsizeMax
    ionParam.ionshiftFilteringWinsizeMin = self.ION_ionshiftFilteringWinsizeMin
    ionParam.azshiftFlag = self.ION_azshiftFlag

    ionParam.numberAzimuthLooks = self.ION_numberAzimuthLooks
    ionParam.numberRangeLooks = self.ION_numberRangeLooks
    ionParam.numberAzimuthLooks0 = self.ION_numberAzimuthLooks0
    ionParam.numberRangeLooks0 = self.ION_numberRangeLooks0


    #3. check parameters
    #convert to m
    ionParam.ionHeight *= 1000.0

    #check number of looks
    if not ((ionParam.numberAzimuthLooks % ionParam.numberAzimuthLooks0 == 0) and \
       (1 <= ionParam.numberAzimuthLooks0 <= ionParam.numberAzimuthLooks)):
        raise Exception('numberAzimuthLooks must be integer multiples of numberAzimuthLooks0')
    if not ((ionParam.numberRangeLooks % ionParam.numberRangeLooks0 == 0) and \
       (1 <= ionParam.numberRangeLooks0 <= ionParam.numberRangeLooks)):
        raise Exception('numberRangeLooks must be integer multiples of numberRangeLooks0')

    #check steps for ionospheric correction
    if ionParam.startStep not in ionParam.allSteps:
        print('all steps for ionospheric correction in order: {}'.format(ionParam.allSteps))
        raise Exception('please specify the correct start step for ionospheric correction from above list')
    if ionParam.endStep not in ionParam.allSteps:
        print('all steps for ionospheric correction in order: {}'.format(ionParam.allSteps))
        raise Exception('please specify the correct start step for ionospheric correction from above list')
    if ionParam.allSteps.index(ionParam.startStep) > ionParam.allSteps.index(ionParam.endStep):
        print('correct relationship: start step <= end step')
        raise Exception('error: start step is after end step.')
    ###################################################################

    ###################################################################
    #routines that require setting parameters
    #def ionosphere(self, ionParam):
    #def ionSwathBySwath(self, ionParam):
    #def filt_gaussian(self, ionParam):
    #def ionosphere_shift(self, ionParam):
    #def ion2grd(self, ionParam):
    #def esd(self, ionParam):
    ###################################################################

    #SECTION 2. DIRECTORIES AND FILENAMES
    #directories
    ionParam.ionDirname = 'ion'
    ionParam.lowerDirname = 'lower'
    ionParam.upperDirname = 'upper'
    ionParam.ioncalDirname = 'ion_cal'
    ionParam.ionBurstDirname = 'ion_burst'
    #these are same directory names as topsApp.py/TopsProc.py
    #ionParam.masterSlcProduct = 'master'
    #ionParam.slaveSlcProduct = 'slave'
    #ionParam.fineCoregDirname = 'fine_coreg'
    ionParam.fineIfgDirname = 'fine_interferogram'
    ionParam.mergedDirname = 'merged'
    #filenames
    ionParam.ionRawNoProj = 'raw_no_projection.ion'
    ionParam.ionCorNoProj = 'raw_no_projection.cor'
    ionParam.ionRaw = 'raw.ion'
    ionParam.ionCor = 'raw.cor'
    ionParam.ionFilt = 'filt.ion'
    ionParam.ionShift = 'azshift.ion'
    ionParam.warning = 'warning.txt'

    #SECTION 3. DATA PARAMETERS
    #earth's radius (m)
    ionParam.earthRadius = 6371 * 1000.0
    #reference range (m) for moving range center frequency to zero, center of center swath
    ionParam.rgRef = 875714.0
    #range bandwidth (Hz) for splitting, range processingBandwidth: [5.650000000000000e+07, 4.830000000000000e+07, 4.278991840322842e+07]
    ionParam.rgBandwidthForSplit = 40.0 * 10**6
    ionParam.rgBandwidthSub = ionParam.rgBandwidthForSplit / 3.0

    #SECTION 4. DEFINE WAVELENGTHS AND DETERMINE IF CALCULATE IONOSPHERE WITH MERGED INTERFEROGRAM
    getParamFromData = False
    masterStartingRange = np.zeros(3)
    slaveStartingRange = np.zeros(3)
    swathList = self._insar.getValidSwathList(self.swaths)
    for swath in swathList:
        ####Load slave metadata
        master = self._insar.loadProduct( os.path.join(self._insar.masterSlcProduct, 'IW{0}.xml'.format(swath)))
        slave = self._insar.loadProduct( os.path.join(self._insar.slaveSlcProduct, 'IW{0}.xml'.format(swath)))

        ####Indices w.r.t master
        minBurst, maxBurst = self._insar.commonMasterBurstLimits(swath-1)
        slaveBurstStart, slaveBurstEnd = self._insar.commonSlaveBurstLimits(swath-1)

        if minBurst == maxBurst:
            #print('Skipping processing of swath {0}'.format(swath))
            continue
        else:
            ii = minBurst
            jj = slaveBurstStart + ii - minBurst

            masBurst = master.bursts[ii]
            slvBurst = slave.bursts[jj]

            #use the 1/3, 1/3, 1/3 scheme for splitting
            ionParam.radarWavelength = masBurst.radarWavelength
            ionParam.radarWavelengthLower = SPEED_OF_LIGHT / (SPEED_OF_LIGHT / ionParam.radarWavelength - ionParam.rgBandwidthForSplit / 3.0)
            ionParam.radarWavelengthUpper = SPEED_OF_LIGHT / (SPEED_OF_LIGHT / ionParam.radarWavelength + ionParam.rgBandwidthForSplit / 3.0)
            #use this to determine which polynomial to use to calculate a ramp when calculating ionosphere for cross A/B interferogram
            ionParam.passDirection = masBurst.passDirection.lower()

            masterStartingRange[swath-1] = masBurst.startingRange
            slaveStartingRange[swath-1] = slvBurst.startingRange
            getParamFromData = True

    #determine if calculate ionosphere using merged interferogram
    if np.sum(masterStartingRange==slaveStartingRange) != 3:
        ionParam.calIonWithMerged = False
    else:
        ionParam.calIonWithMerged = True
    #there is no need to process swath by swath when there is only one swath
    #ionSwathBySwath only works when number of swaths >=2
    if len(swathList) == 1:
        ionParam.calIonWithMerged = True
    #for cross Sentinel-1A/B interferogram, always not using merged interferogram
    if master.mission != slave.mission:
        ionParam.calIonWithMerged = False

    #determine if remove an empirical ramp
    if master.mission == slave.mission:
        ionParam.rampRemovel = 0
    else:
        #estimating ionospheric phase for cross Sentinel-1A/B interferogram
        #an empirical ramp will be removed from the estimated ionospheric phase
        if master.mission == 'S1A' and slave.mission == 'S1B':
            ionParam.rampRemovel = 1
        else:
            ionParam.rampRemovel = -1

    if getParamFromData == False:
        raise Exception('cannot get parameters from data')

    return ionParam


def next_pow2(a):
    x=2
    while x < a:
        x *= 2
    return x


def removeHammingWindow(inputfile, outputfile, bandwidth, samplingRate, alpha, virtual=True):
    '''
    This function removes the range Hamming window imposed on the signal
    bandwidth:     range bandwidth
    samplingRate:  range sampling rate
    alpha:         alpha of the Hamming window
    '''
    #(length, width) = slc.shape


    inImg = isceobj.createSlcImage()
    inImg.load( inputfile + '.xml')

    width = inImg.getWidth()
    length = inImg.getLength()

    if not virtual:
        slc = np.memmap(inputfile, dtype=np.complex64, mode='r', shape=(length,width))
    else:
        slc = loadVirtualArray(inputfile + '.vrt')

    #fft length
    nfft = next_pow2(width)
    #Hamming window length
    nwin = np.int(np.around(bandwidth / samplingRate*nfft))
    #make it a even number, since we are going to use even fft length
    nwin = ((nwin+1)//2)*2
    #the starting and ending index of window in the spectrum
    start = np.int(np.around((nfft - nwin) / 2))
    end = np.int(np.around(start + nwin - 1))
    hammingWindow = alpha - (1.0-alpha) * np.cos(np.linspace(-np.pi, np.pi, num=nwin, endpoint=True))
    hammingWindow = 1.0/np.fft.fftshift(hammingWindow)
    spec = np.fft.fft(slc, n=nfft, axis=1)
    spec = np.fft.fftshift(spec, axes=1)
    spec[:, start:end+1] *= hammingWindow[None,:]
    spec = np.fft.fftshift(spec, axes=1)
    spec = np.fft.ifft(spec, n=nfft, axis=1)
    slcd  = spec[:, 0:width] * ((slc.real!=0) | (slc.imag!=0))
    #after these fft and ifft, the values are not scaled by constant.

    slcd.astype(np.complex64).tofile(outputfile)
    inImg.setFilename(outputfile)
    inImg.extraFilename = outputfile + '.vrt'
    inImg.setAccessMode('READ')
    inImg.renderHdr()

    return slcd


def runCmd(cmd, silent=0):

    if silent == 0:
        print("{}".format(cmd))
    status = os.system(cmd)
    if status != 0:
        raise Exception('error when running:\n{}\n'.format(cmd))


def adjustValidLineSample(master,slave):

    master_lastValidLine = master.firstValidLine + master.numValidLines - 1
    master_lastValidSample = master.firstValidSample + master.numValidSamples - 1
    slave_lastValidLine = slave.firstValidLine + slave.numValidLines - 1
    slave_lastValidSample = slave.firstValidSample + slave.numValidSamples - 1

    igram_lastValidLine = min(master_lastValidLine, slave_lastValidLine)
    igram_lastValidSample = min(master_lastValidSample, slave_lastValidSample)

    master.firstValidLine = max(master.firstValidLine, slave.firstValidLine)
    master.firstValidSample = max(master.firstValidSample, slave.firstValidSample)

    master.numValidLines = igram_lastValidLine - master.firstValidLine + 1
    master.numValidSamples = igram_lastValidSample - master.firstValidSample + 1


def multiply2(mastername, slavename, fact, rngname=None, ionname=None, infname=None, overlapBox=None, valid=True, virtual=True):
    '''
    This routine forms interferogram and possibly removes topographic and ionospheric phases.
    all the following indexes start from 1
    overlapBox[0]: first line
    overlapBox[1]: last line
    overlapBox[2]: first sample
    overlapBox[3]: last sample
    '''

    #use master image
    img = isceobj.createSlcImage()
    img.load(mastername + '.xml')
    width = img.getWidth()
    length = img.getLength()

    #master
    if not virtual:
        master = np.memmap(mastername, dtype=np.complex64, mode='r', shape=(length,width))
    else:
        master = loadVirtualArray(mastername + '.vrt')

    #slave
    slave = np.memmap(slavename, dtype=np.complex64, mode='r', shape=(length, width))

    #interferogram
    cJ = np.complex64(-1j)
    inf = master[overlapBox[0]-1:overlapBox[1]-1+1, overlapBox[2]-1:overlapBox[3]-1+1] \
        * np.conj(slave[overlapBox[0]-1:overlapBox[1]-1+1, overlapBox[2]-1:overlapBox[3]-1+1])

    #topography
    if rngname != None:
        rng2 = np.memmap(rngname, dtype=np.float32, mode='r', shape=(length,width))
        inf *= np.exp(cJ*fact*rng2[overlapBox[0]-1:overlapBox[1]-1+1, overlapBox[2]-1:overlapBox[3]-1+1])

    #ionosphere
    if ionname != None:
        ion = np.memmap(ionname, dtype=np.float32, mode='r', shape=(length, width))
        inf *= np.exp(cJ*ion[overlapBox[0]-1:overlapBox[1]-1+1, overlapBox[2]-1:overlapBox[3]-1+1])

    if valid == True:
        inf2 = inf
    else:
        inf2 = np.zeros((length,width), dtype=np.complex64)
        inf2[overlapBox[0]-1:overlapBox[1]-1+1, overlapBox[2]-1:overlapBox[3]-1+1] = inf

    #inf = master[overlapBox[0]-1:overlapBox[1]-1+1, overlapBox[2]-1:overlapBox[3]-1+1] \
    #    * np.conj(slave[overlapBox[0]-1:overlapBox[1]-1+1, overlapBox[2]-1:overlapBox[3]-1+1]) \
    #    * np.exp(cJ*ion[overlapBox[0]-1:overlapBox[1]-1+1, overlapBox[2]-1:overlapBox[3]-1+1]) \
    #    * np.exp(cJ*fact*rng2[overlapBox[0]-1:overlapBox[1]-1+1, overlapBox[2]-1:overlapBox[3]-1+1])

    if infname != None:
        inf2.astype(np.complex64).tofile(infname)
        img = isceobj.createIntImage()
        img.setFilename(infname)
        img.extraFilename = infname + '.vrt'
        if valid == True:
            img.setWidth(overlapBox[3]-overlapBox[2]+1)
            img.setLength(overlapBox[1]-overlapBox[0]+1)
        else:
            img.setWidth(width)
            img.setLength(length)
        img.setAccessMode('READ')
        img.renderHdr()

    return inf2


def subband(self, ionParam):
    '''
    generate subband images
    '''
    from isceobj.Sensor.TOPS import createTOPSSwathSLCProduct
    from isceobj.Util.Poly2D import Poly2D
    from contrib.alos2proc.alos2proc import rg_filter

    from isceobj.TopsProc.runFineResamp import resampSlave
    from isceobj.TopsProc.runFineResamp import getRelativeShifts
    from isceobj.TopsProc.runFineResamp import adjustValidSampleLine
    from isceobj.TopsProc.runFineResamp import getValidLines

    #from isceobj.TopsProc.runBurstIfg import adjustValidLineSample

    print('processing subband burst interferograms')
    virtual = self.useVirtualFiles
    swathList = self._insar.getValidSwathList(self.swaths)
    for swath in swathList:
        ####Load slave metadata
        master = self._insar.loadProduct( os.path.join(self._insar.masterSlcProduct, 'IW{0}.xml'.format(swath)))
        slave = self._insar.loadProduct( os.path.join(self._insar.slaveSlcProduct, 'IW{0}.xml'.format(swath)))

        dt = slave.bursts[0].azimuthTimeInterval
        dr = slave.bursts[0].rangePixelSize

        ###Directory with offsets
        offdir = os.path.join(self._insar.fineOffsetsDirname, 'IW{0}'.format(swath))

        ####Indices w.r.t master
        minBurst, maxBurst = self._insar.commonMasterBurstLimits(swath-1)
        slaveBurstStart, slaveBurstEnd = self._insar.commonSlaveBurstLimits(swath-1)

        if minBurst == maxBurst:
            print('Skipping processing of swath {0}'.format(swath))
            continue

        #create dirs
        lowerDir = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, ionParam.fineIfgDirname, 'IW{0}'.format(swath))
        if not os.path.isdir(lowerDir):
            os.makedirs(lowerDir)
        upperDir = os.path.join(ionParam.ionDirname, ionParam.upperDirname, ionParam.fineIfgDirname, 'IW{0}'.format(swath))
        if not os.path.isdir(upperDir):
            os.makedirs(upperDir)

        ##############################################################
        #for resampling
        relShifts = getRelativeShifts(master, slave, minBurst, maxBurst, slaveBurstStart)
        print('Shifts IW-{0}: '.format(swath), relShifts) 
   
        ####Can corporate known misregistration here
        apoly = Poly2D()
        apoly.initPoly(rangeOrder=0,azimuthOrder=0,coeffs=[[0.]])

        rpoly = Poly2D()
        rpoly.initPoly(rangeOrder=0,azimuthOrder=0,coeffs=[[0.]])

        misreg_az = self._insar.slaveTimingCorrection / dt
        misreg_rg = self._insar.slaveRangeCorrection / dr
        ##############################################################

        fineIfgLower = createTOPSSwathSLCProduct()
        fineIfgLower.configure()

        fineIfgUpper = createTOPSSwathSLCProduct()
        fineIfgUpper.configure()

        #only process common bursts
        for ii in range(minBurst, maxBurst):
            jj = slaveBurstStart + ii - minBurst 
    
            masBurst = master.bursts[ii]
            slvBurst = slave.bursts[jj]

            print('processing master burst: %02d, slave burst: %02d, swath: %d'%(ii+1, jj+1, swath))
            ################################################################
            #1. removing window and subband
            for ms in ['master', 'slave']:
                #setup something
                if ms == 'master':
                    burst = masBurst
                    #put the temporary file in the lower directory
                    tmpFilename = os.path.join(lowerDir, 'master_dw_'+os.path.basename(burst.image.filename))
                    tmpFilename2 = 'master_'+os.path.basename(burst.image.filename)
                else:
                    burst = slvBurst
                    #put the temporary file in the lower directory
                    tmpFilename = os.path.join(lowerDir, 'slave_dw_'+os.path.basename(burst.image.filename))
                    tmpFilename2 = 'slave_'+os.path.basename(burst.image.filename)

                #removing window
                rangeSamplingRate = SPEED_OF_LIGHT / (2.0 * burst.rangePixelSize)
                if burst.rangeWindowType == 'Hamming':
                    removeHammingWindow(burst.image.filename, tmpFilename, burst.rangeProcessingBandwidth, rangeSamplingRate, burst.rangeWindowCoefficient, virtual=virtual)  
                else:
                    raise Exception('Range weight window type: {} is not supported yet!'.format(burst.rangeWindowType))
                
                #subband
                rg_filter(tmpFilename,
                          burst.numberOfSamples,
                          2,
                          [os.path.join(lowerDir, tmpFilename2), os.path.join(upperDir, tmpFilename2)],
                          [ionParam.rgBandwidthSub / rangeSamplingRate, ionParam.rgBandwidthSub / rangeSamplingRate],
                          [-ionParam.rgBandwidthForSplit / 3.0 / rangeSamplingRate, ionParam.rgBandwidthForSplit / 3.0 / rangeSamplingRate],
                          129,
                          512,
                          0.1,
                          0,
                          (burst.startingRange - ionParam.rgRef) / burst.rangePixelSize
                    )

                #remove temporary file
                os.remove(tmpFilename)
                os.remove(tmpFilename+'.xml')
                os.remove(tmpFilename+'.vrt')

            #2. resampling and form interferogram
            #resampling
            try:
                offset = relShifts[jj]
            except:
                raise Exception('Trying to access shift for slave burst index {0}, which may not overlap with master for swath {1}'.format(jj, swath))

            ####Setup initial polynomials
            ### If no misregs are given, these are zero
            ### If provided, can be used for resampling without running to geo2rdr again for fast results
            rdict = {'azpoly' : apoly,
                     'rgpoly' : rpoly,
                     'rangeOff' : os.path.join(offdir, 'range_%02d.off'%(ii+1)),
                    'azimuthOff': os.path.join(offdir, 'azimuth_%02d.off'%(ii+1))}


            ###For future - should account for azimuth and range misreg here .. ignoring for now.
            azCarrPoly, dpoly = slave.estimateAzimuthCarrierPolynomials(slvBurst, offset = -1.0 * offset)

            rdict['carrPoly'] = azCarrPoly
            rdict['doppPoly'] = dpoly


            for lu in ['lower', 'upper']:
                masBurst2 = masBurst.clone()
                slvBurst2 = slvBurst.clone()
                slvBurstResamp2 = masBurst.clone()
                if lu == 'lower':
                    masBurst2.radarWavelength = ionParam.radarWavelengthLower
                    masBurst2.rangeProcessingBandwidth = ionParam.rgBandwidthSub
                    masBurst2.image.filename = os.path.join(lowerDir, 'master_'+os.path.basename(masBurst.image.filename))
                    slvBurst2.radarWavelength = ionParam.radarWavelengthLower
                    slvBurst2.rangeProcessingBandwidth = ionParam.rgBandwidthSub
                    slvBurst2.image.filename = os.path.join(lowerDir, 'slave_'+os.path.basename(slvBurst.image.filename))
                    slvBurstResamp2.radarWavelength = ionParam.radarWavelengthLower
                    slvBurstResamp2.rangeProcessingBandwidth = ionParam.rgBandwidthSub
                    slvBurstResamp2.image.filename = os.path.join(lowerDir, 'master_'+os.path.basename(masBurst.image.filename))
                    outname = os.path.join(lowerDir, 'slave_resamp_'+os.path.basename(slvBurst.image.filename))
                    ifgdir = lowerDir
                else:
                    masBurst2.radarWavelength = ionParam.radarWavelengthUpper
                    masBurst2.rangeProcessingBandwidth = ionParam.rgBandwidthSub
                    masBurst2.image.filename = os.path.join(upperDir, 'master_'+os.path.basename(masBurst.image.filename))
                    slvBurst2.radarWavelength = ionParam.radarWavelengthUpper
                    slvBurst2.rangeProcessingBandwidth = ionParam.rgBandwidthSub
                    slvBurst2.image.filename = os.path.join(upperDir, 'slave_'+os.path.basename(slvBurst.image.filename))
                    slvBurstResamp2.radarWavelength = ionParam.radarWavelengthUpper
                    slvBurstResamp2.rangeProcessingBandwidth = ionParam.rgBandwidthSub
                    slvBurstResamp2.image.filename = os.path.join(upperDir, 'master_'+os.path.basename(masBurst.image.filename))
                    outname = os.path.join(upperDir, 'slave_resamp_'+os.path.basename(slvBurst.image.filename))
                    ifgdir = upperDir
                outimg = resampSlave(masBurst2, slvBurst2, rdict, outname)
                minAz, maxAz, minRg, maxRg = getValidLines(slvBurst2, rdict, outname,
                        misreg_az = misreg_az - offset, misreg_rng = misreg_rg)
                adjustValidSampleLine(slvBurstResamp2, slvBurst2,
                                             minAz=minAz, maxAz=maxAz,
                                             minRng=minRg, maxRng=maxRg)
                slvBurstResamp2.image.filename = outimg.filename
                
                #forming interferogram
                mastername = masBurst2.image.filename
                slavename = slvBurstResamp2.image.filename
                rngname = os.path.join(offdir, 'range_%02d.off'%(ii+1))
                infname = os.path.join(ifgdir, 'burst_%02d.int'%(ii+1))

                fact = 4.0 * np.pi * slvBurstResamp2.rangePixelSize / slvBurstResamp2.radarWavelength
                adjustValidLineSample(masBurst2,slvBurstResamp2)


                #in original runBurstIfg.py, valid samples in the interferogram are the following (indexes in the numpy matrix):
                #masterFrame.firstValidLine:masterFrame.firstValidLine + masterFrame.numValidLines, masterFrame.firstValidSample:masterFrame.firstValidSample + masterFrame.numValidSamples
                #after the following processing, valid samples in the interferogram are the following (indexes in the numpy matrix):
                #[masBurst.firstValidLine:masBurst.firstValidLine + masBurst.numValidLines, masBurst.firstValidSample:masBurst.firstValidSample + masBurst.numValidSamples]
                #SO THEY ARE EXACTLY THE SAME
                firstline   = masBurst2.firstValidLine + 1
                lastline    = firstline + masBurst2.numValidLines - 1
                firstcolumn = masBurst2.firstValidSample + 1
                lastcolumn  = firstcolumn + masBurst2.numValidSamples - 1
                overlapBox = [firstline, lastline, firstcolumn, lastcolumn]
                multiply2(mastername, slavename, fact, rngname=rngname, ionname=None, infname=infname, overlapBox=overlapBox, valid=False, virtual=virtual)

                #directly from multiply() of runBurstIfg.py
                img = isceobj.createIntImage()
                img.setFilename(infname)
                img.setWidth(masBurst2.numberOfSamples)
                img.setLength(masBurst2.numberOfLines)
                img.setAccessMode('READ')
                #img.renderHdr()

                #save it for deleting later
                masBurst2_filename = masBurst2.image.filename
                #change it for interferogram
                masBurst2.image = img

                if lu == 'lower':
                    fineIfgLower.bursts.append(masBurst2)
                else:
                    fineIfgUpper.bursts.append(masBurst2)

                #remove master and slave subband slcs
                os.remove(masBurst2_filename)
                os.remove(masBurst2_filename+'.xml')
                os.remove(masBurst2_filename+'.vrt')
                os.remove(slvBurst2.image.filename)
                os.remove(slvBurst2.image.filename+'.xml')
                os.remove(slvBurst2.image.filename+'.vrt')
                os.remove(slvBurstResamp2.image.filename)
                os.remove(slvBurstResamp2.image.filename+'.xml')
                os.remove(slvBurstResamp2.image.filename+'.vrt')

        fineIfgLower.numberOfBursts = len(fineIfgLower.bursts)
        fineIfgUpper.numberOfBursts = len(fineIfgUpper.bursts)
        self._insar.saveProduct(fineIfgLower, os.path.join(ionParam.ionDirname, ionParam.lowerDirname, ionParam.fineIfgDirname, 'IW{0}.xml'.format(swath)))
        self._insar.saveProduct(fineIfgUpper, os.path.join(ionParam.ionDirname, ionParam.upperDirname, ionParam.fineIfgDirname, 'IW{0}.xml'.format(swath)))


def cal_coherence(inf, win=5, edge=0):
    '''
    compute coherence uisng only interferogram (phase).
    This routine still follows the regular equation for computing coherence,
    but assumes the amplitudes of master and slave are one, so that coherence
    can be computed using phase only.

    inf: interferogram
    win: window size
    edge: 0: remove all non-full convolution samples

          1: remove samples computed from less than half convolution
             (win=5 used to illustration below)
             * * *
             * * *
             * * *
             * * *
             * * *

          2: remove samples computed from less than quater convolution
             (win=5 used to illustration below)
             * * *
             * * *
             * * *

          3: remove non-full convolution samples on image edges

          4: keep all samples
    '''
    import scipy.signal as ss

    if win % 2 != 1:
        raise Exception('window size must be odd!')
    hwin = np.int(np.around((win - 1) / 2))

    filt = np.ones((win, win))
    amp  = np.absolute(inf)

    cnt = ss.convolve2d((amp!=0), filt, mode='same')
    cor = ss.convolve2d(inf/(amp + (amp==0)), filt, mode='same')
    cor = (amp!=0) * np.absolute(cor) / (cnt + (cnt==0))

    #trim edges
    if edge == 0:
        num = win * win
        cor[np.nonzero(cnt < num)] = 0.0
    elif edge == 1:
        num = win * (hwin+1)
        cor[np.nonzero(cnt < num)] = 0.0
    elif edge == 2:
        num = (hwin+1) * (hwin+1)
        cor[np.nonzero(cnt < num)] = 0.0
    elif edge == 3:
        cor[0:hwin, :] = 0.0
        cor[-hwin:, :] = 0.0
        cor[:, 0:hwin] = 0.0
        cor[:, -hwin:] = 0.0
    else:
        pass

    #print("coherence, max: {} min: {}".format(np.max(cor[np.nonzero(cor!=0)]), np.min(cor[np.nonzero(cor!=0)])))
    return cor


def getMergeBox(self, xmlDirname, numberRangeLooks=1, numberAzimuthLooks=1):
    '''
    xmlDirname:         directory containing xml file
    numberRangeLooks:   number of range looks to take after merging
    numberAzimuthLooks: number of azimuth looks to take after merging
    '''

    from isceobj.TopsProc.runMergeBursts import mergeBox
    from isceobj.TopsProc.runMergeBursts import adjustValidWithLooks

    swathList = self._insar.getValidSwathList(self.swaths)

    #get bursts
    frames=[]
    for swath in swathList:
        minBurst, maxBurst = self._insar.commonMasterBurstLimits(swath-1)
        if minBurst==maxBurst:
            #print('Skipping processing of swath {0}'.format(swath))
            continue
        #since burst directory does not necessarily has IW*.xml, we use the following dir
        #ifg = self._insar.loadProduct( os.path.join(self._insar.fineIfgDirname, 'IW{0}.xml'.format(swath)))
        #use lower
        #dirname = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, ionParam.fineIfgDirname)
        ifg = self._insar.loadProduct( os.path.join(xmlDirname, 'IW{0}.xml'.format(swath)))
        frames.append(ifg)

    #determine merged size
    box = mergeBox(frames)
    #adjust valid with looks, 'frames' ARE CHANGED AFTER RUNNING THIS
    (burstValidBox, burstValidBox2, message) = adjustValidWithLooks(frames, box, numberAzimuthLooks, numberRangeLooks, edge=0, avalid='strict', rvalid='strict')

    return (box, burstValidBox, burstValidBox2, frames)


def merge(self, ionParam):
    '''
    merge burst interferograms and compute coherence
    '''
    from isceobj.TopsProc.runMergeBursts import mergeBox
    from isceobj.TopsProc.runMergeBursts import adjustValidWithLooks
    from isceobj.TopsProc.runMergeBursts import mergeBurstsVirtual
    from isceobj.TopsProc.runMergeBursts import multilook as multilook2

    #merge burst interferograms
    mergeFilename = self._insar.mergedIfgname
    xmlDirname = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, ionParam.fineIfgDirname)
    dirs = [ionParam.lowerDirname, ionParam.upperDirname]
    for dirx in dirs:
        mergeDirname = os.path.join(ionParam.ionDirname, dirx, ionParam.mergedDirname)
        burstDirname = os.path.join(ionParam.ionDirname, dirx, ionParam.fineIfgDirname)

        frames=[]
        burstList = []
        swathList = self._insar.getValidSwathList(self.swaths)
        for swath in swathList:
            minBurst, maxBurst = self._insar.commonMasterBurstLimits(swath-1)
            if minBurst==maxBurst:
                continue
            ifg = self._insar.loadProduct( os.path.join(xmlDirname, 'IW{0}.xml'.format(swath)))
            frames.append(ifg)
            burstList.append([os.path.join(burstDirname, 'IW{0}'.format(swath),  'burst_%02d.int'%(x+1)) for x in range(minBurst, maxBurst)])

        if not os.path.isdir(mergeDirname):
            os.makedirs(mergeDirname)

        suffix = '.full'
        if (ionParam.numberRangeLooks0 == 1) and (ionParam.numberAzimuthLooks0 == 1):
            suffix=''

        box = mergeBox(frames)
        #adjust valid with looks, 'frames' ARE CHANGED AFTER RUNNING THIS
        #here numberRangeLooks, instead of numberRangeLooks0, is used, since we need to do next step multilooking after unwrapping. same for numberAzimuthLooks.
        (burstValidBox, burstValidBox2, message) = adjustValidWithLooks(frames, box, ionParam.numberAzimuthLooks, ionParam.numberRangeLooks, edge=0, avalid='strict', rvalid='strict')
        mergeBurstsVirtual(frames, burstList, box, os.path.join(mergeDirname, mergeFilename+suffix))
        if suffix not in ['',None]:
            multilook2(os.path.join(mergeDirname, mergeFilename+suffix),
              outname = os.path.join(mergeDirname, mergeFilename),
              alks = ionParam.numberAzimuthLooks0, rlks=ionParam.numberRangeLooks0)
        #this is never used for ionosphere correction
        else:
            print('Skipping multi-looking ....')

    #The orginal coherence calculated by topsApp.py is not good at all, use the following coherence instead
    lowerintfile = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, ionParam.mergedDirname, self._insar.mergedIfgname)
    upperintfile = os.path.join(ionParam.ionDirname, ionParam.upperDirname, ionParam.mergedDirname, self._insar.mergedIfgname)
    corfile = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, ionParam.mergedDirname, self._insar.correlationFilename)

    img = isceobj.createImage()
    img.load(lowerintfile + '.xml')
    width = img.width
    length = img.length
    lowerint = np.fromfile(lowerintfile, dtype=np.complex64).reshape(length, width)
    upperint = np.fromfile(upperintfile, dtype=np.complex64).reshape(length, width)

    #compute coherence only using interferogram
    #here I use differential interferogram of lower and upper band interferograms
    #so that coherence is not affected by fringes
    cord = cal_coherence(lowerint*np.conjugate(upperint), win=3, edge=4)
    cor = np.zeros((length*2, width), dtype=np.float32)
    cor[0:length*2:2, :] = np.sqrt( (np.absolute(lowerint)+np.absolute(upperint))/2.0 )
    cor[1:length*2:2, :] = cord
    cor.astype(np.float32).tofile(corfile)

    #create xml and vrt
    #img.scheme = 'BIL'
    #img.bands = 2
    #img.filename = corfile
    #img.renderHdr()

    #img = isceobj.Image.createUnwImage()
    img = isceobj.createOffsetImage()
    img.setFilename(corfile)
    img.extraFilename = corfile + '.vrt'
    img.setWidth(width)
    img.setLength(length)
    img.renderHdr()


def renameFile(oldname, newname):
    img = isceobj.createImage()
    img.load(oldname + '.xml')
    img.setFilename(newname)
    img.extraFilename = newname+'.vrt'
    img.renderHdr()

    os.rename(oldname, newname)
    os.remove(oldname + '.xml')
    os.remove(oldname + '.vrt')


def maskUnwrap(unwfile, maskfile):
    tmpfile = 'tmp.unw'
    renameFile(unwfile, tmpfile)
    cmd = "imageMath.py -e='a_0*(abs(b)!=0);a_1*(abs(b)!=0)' --a={0} --b={1} -s BIL -o={2}".format(tmpfile, maskfile, unwfile)
    runCmd(cmd)
    os.remove(tmpfile)
    os.remove(tmpfile+'.xml')
    os.remove(tmpfile+'.vrt')


def snaphuUnwrap(self, xmlDirname, wrapName, corrfile, unwrapName, nrlks, nalks, costMode = 'DEFO',initMethod = 'MST', defomax = 4.0, initOnly = False):
        #runUnwrap(self,                                           costMode = 'SMOOTH',initMethod = 'MCF', defomax = 2,   initOnly = True)
    '''
    xmlDirname:  xml dir name
    wrapName:    input interferogram
    corrfile:    input coherence file
    unwrapName:  output unwrapped interferogram
    nrlks:       number of range looks of the interferogram
    nalks:       number of azimuth looks of the interferogram
    '''

    from contrib.Snaphu.Snaphu import Snaphu
    from isceobj.Planet.Planet import Planet

    img = isceobj.createImage()
    img.load(wrapName + '.xml')
    width = img.getWidth()

    #get altitude
    swathList = self._insar.getValidSwathList(self.swaths)
    for swath in swathList[0:1]:
        ifg = self._insar.loadProduct( os.path.join(xmlDirname, 'IW{0}.xml'.format(swath)))
        wavelength = ifg.bursts[0].radarWavelength

        ####tmid 
        tstart = ifg.bursts[0].sensingStart
        tend   = ifg.bursts[-1].sensingStop
        tmid = tstart + 0.5*(tend - tstart)

        #14-APR-2018
        burst_index = np.int(np.around(len(ifg.bursts)/2))
        orbit = ifg.bursts[burst_index].orbit
        peg = orbit.interpolateOrbit(tmid, method='hermite')

        refElp = Planet(pname='Earth').ellipsoid
        llh = refElp.xyz_to_llh(peg.getPosition())
        hdg = orbit.getENUHeading(tmid)
        refElp.setSCH(llh[0], llh[1], hdg)
        earthRadius = refElp.pegRadCur
        altitude   = llh[2]

    rangeLooks = nrlks
    azimuthLooks = nalks
    azfact = 0.8
    rngfact = 0.8
    corrLooks = rangeLooks * azimuthLooks/(azfact*rngfact) 
    maxComponents = 20

    snp = Snaphu()
    snp.setInitOnly(initOnly)
    snp.setInput(wrapName)
    snp.setOutput(unwrapName)
    snp.setWidth(width)
    snp.setCostMode(costMode)
    snp.setEarthRadius(earthRadius)
    snp.setWavelength(wavelength)
    snp.setAltitude(altitude)
    snp.setCorrfile(corrfile)
    snp.setInitMethod(initMethod)
    snp.setCorrLooks(corrLooks)
    snp.setMaxComponents(maxComponents)
    snp.setDefoMaxCycles(defomax)
    snp.setRangeLooks(rangeLooks)
    snp.setAzimuthLooks(azimuthLooks)
    #snp.setCorFileFormat('FLOAT_DATA')
    snp.prepare()
    snp.unwrap()

    ######Render XML
    outImage = isceobj.Image.createUnwImage()
    outImage.setFilename(unwrapName)
    outImage.setWidth(width)
    outImage.setAccessMode('read')
    outImage.renderVRT()
    outImage.createImage()
    outImage.finalizeImage()
    outImage.renderHdr()

    #####Check if connected components was created
    if snp.dumpConnectedComponents:
        connImage = isceobj.Image.createImage()
        connImage.setFilename(unwrapName+'.conncomp')
        connImage.setWidth(width)
        connImage.setAccessMode('read')
        connImage.setDataType('BYTE')
        connImage.renderVRT()
        connImage.createImage()
        connImage.finalizeImage()
        connImage.renderHdr()

    return


def unwrap(self, ionParam):
    '''
    unwrap lower and upper band interferograms
    '''

    print('unwrapping lower and upper band interferograms')
    dirs = [ionParam.lowerDirname, ionParam.upperDirname]
    #there is only one coherence file in lower directory
    corfile = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, ionParam.mergedDirname, self._insar.correlationFilename)
    for dirx in dirs:
        procdir = os.path.join(ionParam.ionDirname, dirx, ionParam.mergedDirname)
        wrapName = os.path.join(procdir, self._insar.mergedIfgname)
        unwrapName = os.path.join(procdir, self._insar.unwrappedIntFilename)
        xmlDirname = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, ionParam.fineIfgDirname)
        #unwrap
        snaphuUnwrap(self, xmlDirname, wrapName, corfile, unwrapName, ionParam.numberRangeLooks0, ionParam.numberAzimuthLooks0, costMode = 'SMOOTH',initMethod = 'MCF', defomax = 2, initOnly = True)
        #remove wired things in no-data area
        maskUnwrap(unwrapName, wrapName)

    if [ionParam.numberRangeLooks0, ionParam.numberAzimuthLooks0] != [ionParam.numberRangeLooks, ionParam.numberAzimuthLooks]:
        multilook_unw(self, ionParam, ionParam.mergedDirname)


def multilook_unw(self, ionParam, mergedDirname):
    '''
    30-APR-2018
    This routine moves the original unwrapped files to a directory and takes looks
    '''
    from isceobj.TopsProc.runMergeBursts import multilook as multilook2

    oridir0 = '{}rlks_{}alks'.format(ionParam.numberRangeLooks0, ionParam.numberAzimuthLooks0)
    dirs = [ionParam.lowerDirname, ionParam.upperDirname]
    corName = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, ionParam.mergedDirname, oridir0, self._insar.correlationFilename)
    for dirx in dirs:
        procdir = os.path.join(ionParam.ionDirname, dirx, mergedDirname)
        #create a directory for original files
        oridir = os.path.join(procdir, oridir0)
        if not os.path.isdir(oridir):
            os.makedirs(oridir)
        #move files, renameFile uses os.rename, which overwrites if file already exists in oridir. This can support re-run
        filename0 = os.path.join(procdir, self._insar.mergedIfgname)
        filename  = os.path.join(oridir, self._insar.mergedIfgname)
        if os.path.isfile(filename0):
            renameFile(filename0, filename)
        filename0 = os.path.join(procdir, self._insar.unwrappedIntFilename)
        filename  = os.path.join(oridir, self._insar.unwrappedIntFilename)
        if os.path.isfile(filename0):
            renameFile(filename0, filename)
        filename0 = os.path.join(procdir, self._insar.unwrappedIntFilename+'.conncomp')
        filename  = os.path.join(oridir, self._insar.unwrappedIntFilename+'.conncomp')
        if os.path.isfile(filename0):
            renameFile(filename0, filename)
        filename0 = os.path.join(procdir, self._insar.correlationFilename)
        filename  = os.path.join(oridir, self._insar.correlationFilename)
        if os.path.isfile(filename0):
            renameFile(filename0, filename)
        #for topophase.flat.full, move directly
        filename0 = os.path.join(procdir, self._insar.mergedIfgname+'.full.vrt')
        filename  = os.path.join(oridir, self._insar.mergedIfgname+'.full.vrt')
        if os.path.isfile(filename0):
            os.rename(filename0, filename)
        filename0 = os.path.join(procdir, self._insar.mergedIfgname+'.full.xml')
        filename  = os.path.join(oridir, self._insar.mergedIfgname+'.full.xml')
        if os.path.isfile(filename0):
            os.rename(filename0, filename)

        #multi-looking
        nrlks = np.int(np.around(ionParam.numberRangeLooks / ionParam.numberRangeLooks0))
        nalks = np.int(np.around(ionParam.numberAzimuthLooks / ionParam.numberAzimuthLooks0))
        #coherence
        if dirx == ionParam.lowerDirname:
            corName0 = os.path.join(oridir, self._insar.correlationFilename)
            corimg = isceobj.createImage()
            corimg.load(corName0 + '.xml')
            width = corimg.width
            length = corimg.length
            widthNew = np.int(width / nrlks)
            lengthNew = np.int(length / nalks)
            cor0 = (np.fromfile(corName0, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
            amp0 = (np.fromfile(corName0, dtype=np.float32).reshape(length*2, width))[0:length*2:2, :]
            wgt = cor0**2
            a = multilook(wgt, nalks, nrlks)
            b = multilook(cor0, nalks, nrlks)
            c = multilook(amp0**2, nalks, nrlks)
            d = multilook((cor0!=0).astype(np.int), nalks, nrlks)
            #coherence after multiple looking
            cor = np.zeros((lengthNew*2, widthNew), dtype=np.float32)
            cor[0:lengthNew*2:2, :] = np.sqrt(c / (d + (d==0)))
            cor[1:lengthNew*2:2, :] = b / (d + (d==0))
            #output file
            corName = os.path.join(procdir, self._insar.correlationFilename)
            cor.astype(np.float32).tofile(corName)
            corimg.setFilename(corName)
            corimg.extraFilename = corName + '.vrt'
            corimg.setWidth(widthNew)
            corimg.setLength(lengthNew)
            corimg.renderHdr()
        #unwrapped file
        unwrapName0 = os.path.join(oridir, self._insar.unwrappedIntFilename)
        unwimg = isceobj.createImage()
        unwimg.load(unwrapName0 + '.xml')
        unw0 = (np.fromfile(unwrapName0, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
        amp0 = (np.fromfile(unwrapName0, dtype=np.float32).reshape(length*2, width))[0:length*2:2, :]
        e = multilook(unw0*wgt, nalks, nrlks)
        f = multilook(amp0**2, nalks, nrlks)
        unw = np.zeros((lengthNew*2, widthNew), dtype=np.float32)
        unw[0:lengthNew*2:2, :] = np.sqrt(f / (d + (d==0)))
        unw[1:lengthNew*2:2, :] = e / (a + (a==0))
        #output file
        unwrapName = os.path.join(procdir, self._insar.unwrappedIntFilename)
        unw.astype(np.float32).tofile(unwrapName)
        unwimg.setFilename(unwrapName)
        unwimg.extraFilename = unwrapName + '.vrt'
        unwimg.setWidth(widthNew)
        unwimg.setLength(lengthNew)
        unwimg.renderHdr()

    #looks like the above is not a good coherence, re-calculate here
    #here I use differential interferogram of lower and upper band interferograms
    #so that coherence is not affected by fringes
    lowerIntName0 = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, mergedDirname, oridir0, self._insar.mergedIfgname)
    upperIntName0 = os.path.join(ionParam.ionDirname, ionParam.upperDirname, mergedDirname, oridir0, self._insar.mergedIfgname)
    lowerIntName = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, mergedDirname, self._insar.mergedIfgname)
    upperIntName = os.path.join(ionParam.ionDirname, ionParam.upperDirname, mergedDirname, self._insar.mergedIfgname)
    #cmd = 'looks.py -i {} -o {} -r {} -a {}'.format(lowerIntName0, lowerIntName, nrlks, nalks)
    #runCmd(cmd)
    #cmd = 'looks.py -i {} -o {} -r {} -a {}'.format(upperIntName0, upperIntName, nrlks, nalks)
    #runCmd(cmd)
    multilook2(lowerIntName0, outname = lowerIntName, alks = nalks, rlks=nrlks)
    multilook2(upperIntName0, outname = upperIntName, alks = nalks, rlks=nrlks)
    lowerint = np.fromfile(lowerIntName, dtype=np.complex64).reshape(lengthNew, widthNew)
    upperint = np.fromfile(upperIntName, dtype=np.complex64).reshape(lengthNew, widthNew)
    cor = np.zeros((lengthNew*2, widthNew), dtype=np.float32)
    cor[0:length*2:2, :] = np.sqrt( (np.absolute(lowerint)+np.absolute(upperint))/2.0 )
    cor[1:length*2:2, :] = cal_coherence(lowerint*np.conjugate(upperint), win=3, edge=4)
    cor.astype(np.float32).tofile(corName)


def create_multi_index2(width2, l1, l2):
    #for number of looks of l1 and l2
    #calculate the correponding index number of l2 in the l1 array
    #applies to both range and azimuth direction

    return ((l2 - l1) / 2.0  + np.arange(width2) * l2) / l1


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


def weight_fitting(ionos, cor, width, length, nrli, nali, nrlo, nalo, order, coth):
    '''
    ionos:  input ionospheric phase
    cor:    coherence of the interferogram
    width:  file width
    length: file length
    nrli:   number of range looks of the input interferograms
    nali:   number of azimuth looks of the input interferograms
    nrlo:   number of range looks of the output ionosphere phase
    nalo:   number of azimuth looks of the ioutput ionosphere phase
    order:  the order of the polynomial for fitting ionosphere phase estimates
    coth:   coherence threshhold for ionosphere phase estimation
    '''

    lengthi = int(length/nali)
    widthi = int(width/nrli)
    lengtho = int(length/nalo)
    widtho = int(width/nrlo)

    #calculate output index
    rgindex = create_multi_index2(widtho, nrli, nrlo)
    azindex = create_multi_index2(lengtho, nali, nalo)

    #convert coherence to weight
    cor = cor**2/(1.009-cor**2)

    #look for data to use
    flag = (cor>coth)*(ionos!=0)
    point_index = np.nonzero(flag)
    m = point_index[0].shape[0]

    #calculate input index matrix
    x0=np.matlib.repmat(np.arange(widthi), lengthi, 1)
    y0=np.matlib.repmat(np.arange(lengthi).reshape(lengthi, 1), 1, widthi)

    x = x0[point_index].reshape(m, 1)
    y = y0[point_index].reshape(m, 1)
    z = ionos[point_index].reshape(m, 1)
    w = cor[point_index].reshape(m, 1)

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


def computeIonosphere(lowerUnw, upperUnw, cor, fl, fu, adjFlag, corThresholdAdj, dispersive):
    '''
    This routine computes ionosphere and remove the relative phase unwrapping errors

    lowerUnw:        lower band unwrapped interferogram
    upperUnw:        upper band unwrapped interferogram
    cor:             coherence
    fl:              lower band center frequency
    fu:              upper band center frequency
    adjFlag:         method for removing relative phase unwrapping errors
                       0: mean value
                       1: polynomial
    corThresholdAdj: coherence threshold of samples used in removing relative phase unwrapping errors
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
        flag = (lowerUnw!=0)*(cor>=corThresholdAdj)
        index = np.nonzero(flag!=0)
        mv = np.mean((lowerUnw - upperUnw)[index], dtype=np.float64)
        print('mean value of phase difference: {}'.format(mv))
        diff = mv
    #adjust phase using a surface
    else:
        diff = weight_fitting(lowerUnw - upperUnw, cor, width, length, 1, 1, 1, 1, 2, corThresholdAdj)

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


def ionosphere(self, ionParam):

    ###################################
    #SET PARAMETERS HERE
    #THESE SHOULD BE GOOD ENOUGH, NO NEED TO SET IN setup(self)
    corThresholdAdj = 0.85
    ###################################

    print('computing ionosphere')
    #get files
    lowerUnwfile = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, ionParam.mergedDirname, self._insar.unwrappedIntFilename)
    upperUnwfile = os.path.join(ionParam.ionDirname, ionParam.upperDirname, ionParam.mergedDirname, self._insar.unwrappedIntFilename)
    corfile = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, ionParam.mergedDirname, self._insar.correlationFilename)

    #use image size from lower unwrapped interferogram
    img = isceobj.createImage()
    img.load(lowerUnwfile + '.xml')
    width = img.width
    length = img.length

    lowerUnw = (np.fromfile(lowerUnwfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    upperUnw = (np.fromfile(upperUnwfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    lowerAmp = (np.fromfile(lowerUnwfile, dtype=np.float32).reshape(length*2, width))[0:length*2:2, :]
    upperAmp = (np.fromfile(upperUnwfile, dtype=np.float32).reshape(length*2, width))[0:length*2:2, :]
    cor = (np.fromfile(corfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    amp = np.sqrt(lowerAmp**2+upperAmp**2)

    #compute ionosphere
    fl = SPEED_OF_LIGHT / ionParam.radarWavelengthLower
    fu = SPEED_OF_LIGHT / ionParam.radarWavelengthUpper
    adjFlag = 1
    ionos = computeIonosphere(lowerUnw, upperUnw, cor, fl, fu, adjFlag, corThresholdAdj, 0)

    #dump ionosphere
    outDir = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname)
    if not os.path.isdir(outDir):
        os.makedirs(outDir)
    outFilename = os.path.join(outDir, ionParam.ionRawNoProj)
    ion = np.zeros((length*2, width), dtype=np.float32)
    ion[0:length*2:2, :] = amp
    ion[1:length*2:2, :] = ionos
    ion.astype(np.float32).tofile(outFilename)
    img.filename = outFilename
    img.extraFilename = outFilename + '.vrt'
    img.renderHdr()

    #dump coherence
    outFilename = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname, ionParam.ionCorNoProj)
    ion[1:length*2:2, :] = cor
    ion.astype(np.float32).tofile(outFilename)
    img.filename = outFilename
    img.extraFilename = outFilename + '.vrt'
    img.renderHdr()


def cal_cross_ab_ramp(swathList, width, numberRangeLooks, passDirection):
    '''
    calculate an empirical ramp between Sentinel-1A/B
    29-JUN-2018

    swathList:        self._insar.getValidSwathList(self.swaths)
    width:            single-look image width after merging
    numberRangeLooks: number of range looks in the processing of ionosphere estimation
    passDirection:    descending/ascending
    '''
    
    #below is from processing chile_d156_160725(S1A)-160929(S1B)
    #empirical polynomial
    deg = 3
    if passDirection.lower() == 'descending':
        p = np.array([0.95381267, 2.95567604, -4.56047084, 1.05443172])
    elif passDirection.lower() == 'ascending':
        #for ascending, the polynomial is left/right flipped
        p = np.array([-0.95381267, 5.81711404, -4.21231923, 0.40344958])
    else:
        raise Exception('unknown passDirection! should be either descending or ascending')

    #ca/a166/process/160807-170305 also has the swath offset almost equal to these
    #swath offset in single-look range pixels
    swath_offset = [0, 19810, 43519]
    #total number of single-look range pixels
    tnp = 69189

    #getting x
    nswath = len(swathList)
    if nswath == 3:
        width2 = np.int(width/numberRangeLooks)
        x = np.arange(width2) / (width2 - 1.0)
    else:
        width2 = np.int(width/numberRangeLooks)
        #WARNING: what if the some swaths does not have bursts, and are not merged?
        #         here I just simply ignore this case
        offset = swath_offset[swathList[0]-1]
        x = offset / tnp + width / tnp * np.arange(width2) / (width2 - 1.0)
        
    #calculate ramp
    y_fit = x * 0.0
    for i in range(deg+1):
        y_fit += p[i] * x**[deg-i]

    return y_fit


def ionSwathBySwath(self, ionParam):
    '''
    This routine merge, unwrap and compute ionosphere swath by swath, and then
    adjust phase difference between adjacent swaths caused by relative range timing
    error between adjacent swaths.
    
    This routine includes the following steps in the merged-swath processing:
    merge(self, ionParam)
    unwrap(self, ionParam)
    ionosphere(self, ionParam)
    '''

    from isceobj.TopsProc.runMergeBursts import mergeBox
    from isceobj.TopsProc.runMergeBursts import adjustValidWithLooks
    from isceobj.TopsProc.runMergeBursts import mergeBurstsVirtual
    from isceobj.TopsProc.runMergeBursts import multilook as multilook2

    #########################################
    #SET PARAMETERS HERE
    numberRangeLooks = ionParam.numberRangeLooks
    numberAzimuthLooks = ionParam.numberAzimuthLooks
    numberRangeLooks0 = ionParam.numberRangeLooks0
    numberAzimuthLooks0 = ionParam.numberAzimuthLooks0

    #THESE SHOULD BE GOOD ENOUGH, NO NEED TO SET IN setup(self)
    corThresholdSwathAdj = 0.85
    corThresholdAdj = 0.85
    #########################################

    print('computing ionosphere swath by swath')
    #if ionParam.calIonWithMerged == False:
    warningInfo = '{} calculating ionosphere swath by swath, there may be slight phase error between subswaths\n'.format(datetime.datetime.now())
    with open(os.path.join(ionParam.ionDirname, ionParam.warning), 'a') as f:
        f.write(warningInfo)

    #get bursts
    numValidSwaths = 0
    swathList = self._insar.getValidSwathList(self.swaths)
    for swath in swathList:
        minBurst, maxBurst = self._insar.commonMasterBurstLimits(swath-1)
        if minBurst==maxBurst:
            #print('Skipping processing of swath {0}'.format(swath))
            continue
        numValidSwaths += 1

    if numValidSwaths <= 1:
        raise Exception('There are less than one subswaths, no need to use swath-by-swath method to compute ionosphere!')
    else:
        xmlDirname = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, ionParam.fineIfgDirname)
        (box, burstValidBox, burstValidBox2, frames) = getMergeBox(self, xmlDirname, numberRangeLooks=ionParam.numberRangeLooks, numberAzimuthLooks=ionParam.numberAzimuthLooks)

    #compute ionosphere swath by swath
    corList = []
    ampList = []
    ionosList = []
    nswath = len(swathList)
    ii = -1
    for i in range(nswath):
        swath = swathList[i]
        minBurst, maxBurst = self._insar.commonMasterBurstLimits(swath-1)
        if minBurst==maxBurst:
            print('Skipping processing of swath {0}'.format(swath))
            continue
        else:
            ii += 1

        ########################################################
        #STEP 1. MERGE THE BURSTS OF A SWATH
        ########################################################
        dirs = [ionParam.lowerDirname, ionParam.upperDirname]
        for dirx in dirs:
            outputFilename = self._insar.mergedIfgname
            outputDirname = os.path.join(ionParam.ionDirname, dirx, ionParam.mergedDirname + '_IW{0}'.format(swath))
            if not os.path.isdir(outputDirname):
                os.makedirs(outputDirname)
            suffix = '.full'
            if (numberRangeLooks0 == 1) and (numberAzimuthLooks0 == 1):
                suffix=''

            #merge
            burstPattern = 'burst_%02d.int'
            burstDirname = os.path.join(ionParam.ionDirname, dirx, ionParam.fineIfgDirname)
            ifg = self._insar.loadProduct( os.path.join(burstDirname, 'IW{0}.xml'.format(swath)))
            bst = [os.path.join(burstDirname, 'IW{0}'.format(swath), burstPattern%(x+1)) for x in range(minBurst, maxBurst)]
            #doing adjustment before use
            adjustValidWithLooks([ifg], box, numberAzimuthLooks, numberRangeLooks, edge=0, avalid='strict', rvalid=np.int(np.around(numberRangeLooks/8.0)))
            mergeBurstsVirtual([ifg], [bst], box, os.path.join(outputDirname, outputFilename+suffix))

            #take looks
            if suffix not in ['', None]:
                multilook2(os.path.join(outputDirname, outputFilename+suffix),
                           os.path.join(outputDirname, outputFilename),
                           numberAzimuthLooks0,
                           numberRangeLooks0)
            else:
                print('skipping multilooking')

        #The orginal coherence calculated by topsApp.py is not good at all, use the following coherence instead
        lowerintfile = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, ionParam.mergedDirname + '_IW{0}'.format(swath), self._insar.mergedIfgname)
        upperintfile = os.path.join(ionParam.ionDirname, ionParam.upperDirname, ionParam.mergedDirname + '_IW{0}'.format(swath), self._insar.mergedIfgname)
        corfile = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, ionParam.mergedDirname + '_IW{0}'.format(swath), self._insar.correlationFilename)

        img = isceobj.createImage()
        img.load(lowerintfile + '.xml')
        width = img.width
        length = img.length
        lowerint = np.fromfile(lowerintfile, dtype=np.complex64).reshape(length, width)
        upperint = np.fromfile(upperintfile, dtype=np.complex64).reshape(length, width)


        ##########################################################################
        #slight filtering to improve the estimation accurary of swath difference
        if 1 and shutil.which('psfilt1') != None:
            cmd1 = 'mv {} tmp'.format(lowerintfile)
            cmd2 = 'psfilt1 tmp {} {} .3 32 8'.format(lowerintfile, width)
            cmd3 = 'rm tmp'
            cmd4 = 'mv {} tmp'.format(upperintfile)
            cmd5 = 'psfilt1 tmp {} {} .3 32 8'.format(upperintfile, width)
            cmd6 = 'rm tmp'

            runCmd(cmd1)
            runCmd(cmd2)
            runCmd(cmd3)
            runCmd(cmd4)
            runCmd(cmd5)
            runCmd(cmd6)
        ##########################################################################


        #compute coherence only using interferogram
        #here I use differential interferogram of lower and upper band interferograms
        #so that coherence is not affected by fringes
        cord = cal_coherence(lowerint*np.conjugate(upperint), win=3, edge=4)
        cor = np.zeros((length*2, width), dtype=np.float32)
        cor[0:length*2:2, :] = np.sqrt( (np.absolute(lowerint)+np.absolute(upperint))/2.0 )
        cor[1:length*2:2, :] = cord
        cor.astype(np.float32).tofile(corfile)

        #create xml and vrt
        #img.scheme = 'BIL'
        #img.bands = 2
        #img.filename = corfile
        #img.renderHdr()
        
        #img = isceobj.Image.createUnwImage()
        img = isceobj.createOffsetImage()
        img.setFilename(corfile)
        img.extraFilename = corfile + '.vrt'
        img.setWidth(width)
        img.setLength(length)
        img.renderHdr()

        ########################################################
        #STEP 2. UNWRAP SWATH INTERFEROGRAM
        ########################################################
        dirs = [ionParam.lowerDirname, ionParam.upperDirname]
        #there is only one coherence file in lower directory
        corfile = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, ionParam.mergedDirname + '_IW{0}'.format(swath), self._insar.correlationFilename)
        for dirx in dirs:
            procdir = os.path.join(ionParam.ionDirname, dirx, ionParam.mergedDirname + '_IW{0}'.format(swath))
            wrapName = os.path.join(procdir, self._insar.mergedIfgname)
            unwrapName = os.path.join(procdir, self._insar.unwrappedIntFilename)
            xmlDirname = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, ionParam.fineIfgDirname)
            #unwrap
            snaphuUnwrap(self, xmlDirname, wrapName, corfile, unwrapName, numberRangeLooks0, numberAzimuthLooks0, costMode = 'SMOOTH',initMethod = 'MCF', defomax = 2, initOnly = True)
            #remove wired things in no-data area
            maskUnwrap(unwrapName, wrapName)

        if [ionParam.numberRangeLooks0, ionParam.numberAzimuthLooks0] != [ionParam.numberRangeLooks, ionParam.numberAzimuthLooks]:
            multilook_unw(self, ionParam, ionParam.mergedDirname + '_IW{0}'.format(swath))

        ########################################################
        #STEP 3. COMPUTE IONOSPHERE
        ########################################################
        #get files
        lowerUnwfile = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, ionParam.mergedDirname + '_IW{0}'.format(swath), self._insar.unwrappedIntFilename)
        upperUnwfile = os.path.join(ionParam.ionDirname, ionParam.upperDirname, ionParam.mergedDirname + '_IW{0}'.format(swath), self._insar.unwrappedIntFilename)
        corfile = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, ionParam.mergedDirname + '_IW{0}'.format(swath), self._insar.correlationFilename)

        #use image size from lower unwrapped interferogram
        img = isceobj.createImage()
        img.load(lowerUnwfile + '.xml')
        width = img.width
        length = img.length

        lowerUnw = (np.fromfile(lowerUnwfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
        upperUnw = (np.fromfile(upperUnwfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
        lowerAmp = (np.fromfile(lowerUnwfile, dtype=np.float32).reshape(length*2, width))[0:length*2:2, :]
        upperAmp = (np.fromfile(upperUnwfile, dtype=np.float32).reshape(length*2, width))[0:length*2:2, :]
        cor = (np.fromfile(corfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
        amp = np.sqrt(lowerAmp**2+upperAmp**2)

        #compute ionosphere
        fl = SPEED_OF_LIGHT / ionParam.radarWavelengthLower
        fu = SPEED_OF_LIGHT / ionParam.radarWavelengthUpper
        adjFlag = 1
        ionos = computeIonosphere(lowerUnw, upperUnw, cor, fl, fu, adjFlag, corThresholdAdj, 0)

        #dump result
        outDir = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname + '_IW{0}'.format(swath))
        if not os.path.isdir(outDir):
            os.makedirs(outDir)
        outFilename = os.path.join(outDir, ionParam.ionRawNoProj)
        ion = np.zeros((length*2, width), dtype=np.float32)
        ion[0:length*2:2, :] = amp
        ion[1:length*2:2, :] = ionos
        ion.astype(np.float32).tofile(outFilename)
        img.filename = outFilename
        img.extraFilename = outFilename + '.vrt'
        img.renderHdr()

        corList.append(cor)
        ampList.append(amp)
        ionosList.append(ionos)

    #do adjustment between ajacent swaths
    if numValidSwaths == 3:
        adjustList = [ionosList[0], ionosList[2]]
    else:
        adjustList = [ionosList[0]]
    for adjdata in adjustList:
        index = np.nonzero((adjdata!=0) * (ionosList[1]!=0) * (corList[1] > corThresholdSwathAdj))
        if index[0].size < 5:
            print('WARNING: too few samples available for adjustment between swaths: {} with coherence threshold: {}'.format(index[0].size, corThresholdSwathAdj))
            print('         no adjustment made')
            print('         to do ajustment, please consider using lower coherence threshold')
        else:
            print('number of samples available for adjustment in the overlap area: {}'.format(index[0].size))
            #diff = np.mean((ionosList[1] - adjdata)[index], dtype=np.float64)
            
            #use weighted mean instead
            wgt = corList[1][index]**14
            diff = np.sum((ionosList[1] - adjdata)[index] * wgt / np.sum(wgt, dtype=np.float64), dtype=np.float64)

            index2 = np.nonzero(adjdata!=0)
            adjdata[index2] = adjdata[index2] + diff

    #get merged ionosphere
    ampMerged = np.zeros((length, width), dtype=np.float32)
    corMerged = np.zeros((length, width), dtype=np.float32)
    ionosMerged = np.zeros((length, width), dtype=np.float32)
    for i in range(numValidSwaths):
        nBurst = len(burstValidBox[i])
        for j in range(nBurst):

            #index after multi-looking in merged image, index starts from 1
            first_line = np.int(np.around((burstValidBox[i][j][0] - 1) / numberAzimuthLooks + 1))
            last_line = np.int(np.around(burstValidBox[i][j][1] / numberAzimuthLooks))
            first_sample = np.int(np.around((burstValidBox[i][j][2] - 1) / numberRangeLooks + 1))
            last_sample = np.int(np.around(burstValidBox[i][j][3] / numberRangeLooks))

            corMerged[first_line-1:last_line-1+1, first_sample-1:last_sample-1+1] = \
                corList[i][first_line-1:last_line-1+1, first_sample-1:last_sample-1+1]

            ampMerged[first_line-1:last_line-1+1, first_sample-1:last_sample-1+1] = \
                ampList[i][first_line-1:last_line-1+1, first_sample-1:last_sample-1+1]

            ionosMerged[first_line-1:last_line-1+1, first_sample-1:last_sample-1+1] = \
                ionosList[i][first_line-1:last_line-1+1, first_sample-1:last_sample-1+1]

    #remove an empirical ramp
    if ionParam.rampRemovel != 0:
        warningInfo = '{} calculating ionosphere for cross S-1A/B interferogram, an empirical ramp is removed from estimated ionosphere\n'.format(datetime.datetime.now())
        with open(os.path.join(ionParam.ionDirname, ionParam.warning), 'a') as f:
            f.write(warningInfo)

        abramp = cal_cross_ab_ramp(swathList, box[1], numberRangeLooks, ionParam.passDirection)
        if ionParam.rampRemovel == -1:
            abramp *= -1.0
        #currently do not apply this
        #ionosMerged -= abramp[None, :]

    #dump ionosphere
    outDir = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname)
    if not os.path.isdir(outDir):
        os.makedirs(outDir)
    outFilename = os.path.join(outDir, ionParam.ionRawNoProj)
    ion = np.zeros((length*2, width), dtype=np.float32)
    ion[0:length*2:2, :] = ampMerged
    ion[1:length*2:2, :] = ionosMerged
    ion.astype(np.float32).tofile(outFilename)
    img.filename = outFilename
    img.extraFilename = outFilename + '.vrt'
    img.renderHdr()

    #dump coherence
    outFilename = os.path.join(outDir, ionParam.ionCorNoProj)
    ion[1:length*2:2, :] = corMerged
    ion.astype(np.float32).tofile(outFilename)
    img.filename = outFilename
    img.extraFilename = outFilename + '.vrt'
    img.renderHdr()


def multilookIndex(first, last, nl):
    '''
    create the index after multilooking
    the orginal 1-look index can start from any number such as 0, 1 or other number
    after multilooking, the index still starts from the same number.
    first: index of first pixel in the original 1-look array
    last: index of last pixel in the original 1-look array
    nl: number of looks(nl can also be 1). nl >= 1
    '''

    #number of pixels after multilooking
    num = int((last - first + 1)/nl)
    offset = (first + (first + nl - 1)) / 2.0
    index = offset + np.arange(num) * nl

    return index


def computeDopplerOffset(burst, firstline, lastline, firstcolumn, lastcolumn, nrlks=1, nalks=1):
    '''
    compute offset corresponding to center Doppler frequency
    firstline, lastline, firstcolumn, lastcolumn: index of original 1-look burst, index starts from 1.

    output: first lines > 0, last lines < 0
    '''
    from scipy import interpolate
    from scipy.interpolate import interp1d

    Vs = np.linalg.norm(burst.orbit.interpolateOrbit(burst.sensingMid, method='hermite').getVelocity())
    Ks =   2 * Vs * burst.azimuthSteeringRate / burst.radarWavelength 

    #firstcolumn, lastcolumn: index starts from 1
    rng = multilookIndex(firstcolumn-1, lastcolumn-1, nrlks) * burst.rangePixelSize + burst.startingRange
    #firstline, lastline: index starts from 1
    eta = (  multilookIndex(firstline-1, lastline-1, nalks) - (burst.numberOfLines-1.0)/2.0) * burst.azimuthTimeInterval

    f_etac = burst.doppler(rng)
    Ka     = burst.azimuthFMRate(rng)
    eta_ref = (burst.doppler(burst.startingRange) / burst.azimuthFMRate(burst.startingRange) ) - (f_etac / Ka)
    Kt = Ks / (1.0 - Ks/Ka)

    #carr = np.pi * Kt[None,:] * ((eta[:,None] - eta_ref[None,:])**2)
    #center doppler frequency due to rotation
    dopplerOffset1 = (eta[:,None] - eta_ref[None,:]) * Kt / Ka[None,:] / (burst.azimuthTimeInterval * nalks)
    #center doppler frequency due to squint
    dopplerOffset2 = (f_etac[None,:] / Ka[None,:]) / (burst.azimuthTimeInterval * nalks)
    dopplerOffset = dopplerOffset1 + dopplerOffset2
 
    return (dopplerOffset, Ka)


def grd2ion(self, ionParam):

    print('resampling ionosphere from ground to ionospheric layer')
    #get files
    corfile = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname, ionParam.ionCorNoProj)
    ionfile = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname, ionParam.ionRawNoProj)

    #use image size from lower unwrapped interferogram
    img = isceobj.createImage()
    img.load(corfile + '.xml')
    width = img.width
    length = img.length

    cor = (np.fromfile(corfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    amp = (np.fromfile(ionfile, dtype=np.float32).reshape(length*2, width))[0:length*2:2, :]
    ionos = (np.fromfile(ionfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]

    #use the satellite height of the mid burst of first swath of master acquistion
    swathList = self._insar.getValidSwathList(self.swaths)
    master = self._insar.loadProduct( os.path.join(self._insar.masterSlcProduct, 'IW{0}.xml'.format(swathList[0])))
    minBurst, maxBurst = self._insar.commonMasterBurstLimits(swathList[0]-1)
    #no problem with this index at all
    midBurst = np.int(np.around((minBurst+ maxBurst-1) / 2.0))
    masBurst = master.bursts[midBurst]
    #satellite height
    satHeight = np.linalg.norm(masBurst.orbit.interpolateOrbit(masBurst.sensingMid, method='hermite').getPosition())
    #orgininal doppler offset should be multiplied by this ratio
    ratio = ionParam.ionHeight/(satHeight-ionParam.earthRadius)

    xmlDirname = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, ionParam.fineIfgDirname)
    (box, burstValidBox, burstValidBox2, frames) = getMergeBox(self, xmlDirname, numberRangeLooks=ionParam.numberRangeLooks, numberAzimuthLooks=ionParam.numberAzimuthLooks)

##############################################################################################################
    swathList = self._insar.getValidSwathList(self.swaths)
    frames=[]
    #for valid swaths and bursts, consistent with runMergeBursts.py
    for swath in swathList:
        minBurst, maxBurst = self._insar.commonMasterBurstLimits(swath-1)

        if minBurst==maxBurst:
            print('Skipping processing of swath {0}'.format(swath))
            continue

        ifg = self._insar.loadProduct( os.path.join(xmlDirname, 'IW{0}.xml'.format(swath)))
        frames.append(ifg)
##############################################################################################################

    for band in [amp, ionos, cor]:
        nswath = len(frames)
        for i in range(nswath):
            nburst = len(frames[i].bursts)
            for j in range(nburst):
                #according to runBurstIfg.py, this is originally from self._insar.masterSlcProduct, 'IW{0}.xml'
                masBurst = frames[i].bursts[j]
                (dopplerOffset, Ka) = computeDopplerOffset(masBurst, burstValidBox2[i][j][0], burstValidBox2[i][j][1], burstValidBox2[i][j][2], burstValidBox2[i][j][3], nrlks=ionParam.numberRangeLooks, nalks=ionParam.numberAzimuthLooks)
                offset = ratio * dopplerOffset

                #   0              1               2              3
                #firstlineAdj, lastlineAdj, firstcolumnAdj, lastcolumnAdj, 
                #after multiplication, index starts from 1
                firstline = np.int(np.around((burstValidBox[i][j][0] - 1) / ionParam.numberAzimuthLooks + 1))
                lastline = np.int(np.around(burstValidBox[i][j][1] / ionParam.numberAzimuthLooks))
                firstcolumn = np.int(np.around((burstValidBox[i][j][2] - 1) / ionParam.numberRangeLooks + 1))
                lastcolumn = np.int(np.around(burstValidBox[i][j][3] / ionParam.numberRangeLooks))

                #extract image
                burstImage = band[firstline-1:lastline, firstcolumn-1:lastcolumn]
                blength = lastline - firstline + 1
                bwidth = lastcolumn - firstcolumn + 1

                #interpolation
                index0 = np.linspace(0, blength-1, num=blength, endpoint=True)
                for k in range(bwidth):
                    index = index0 + offset[:, k]
                    value = burstImage[:, k]
                    f = interp1d(index, value, kind='cubic', fill_value="extrapolate")
                    
                    index_min = np.int(np.around(np.amin(index)))
                    index_max = np.int(np.around(np.amax(index)))
                    flag = index0 * 0.0
                    flag[index_min:index_max+1] = 1.0
                    #replace the original column with new column in burstImage
                    #this should also replace teh original column with new column in band
                    burstImage[:, k] = (f(index0)) * flag

    #dump ionosphere with projection
    outDir = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname)
    outFilename = os.path.join(outDir, ionParam.ionRaw)
    ion = np.zeros((length*2, width), dtype=np.float32)
    ion[0:length*2:2, :] = amp
    ion[1:length*2:2, :] = ionos
    ion.astype(np.float32).tofile(outFilename)
    img.filename = outFilename
    img.extraFilename = outFilename + '.vrt'
    img.renderHdr()

    #dump coherence with projection
    outFilename = os.path.join(outDir, ionParam.ionCor)
    ion[1:length*2:2, :] = cor
    ion.astype(np.float32).tofile(outFilename)
    img.filename = outFilename
    img.extraFilename = outFilename + '.vrt'
    img.renderHdr()


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


def filt_gaussian(self, ionParam):
    '''
    This function filters image using gaussian filter

    we projected the ionosphere value onto the ionospheric layer, and the indexes are integers.
    this reduces the number of samples used in filtering
    a better method is to project the indexes onto the ionospheric layer. This way we have orginal
    number of samples used in filtering. but this requries more complicated operation in filtering
    currently not implemented.
    a less accurate method is to use ionsphere without any projection
    '''
    from scipy import interpolate
    from scipy.interpolate import interp1d

    #################################################
    #SET PARAMETERS HERE
    #if applying polynomial fitting
    #False: no fitting, True: with fitting
    fit = ionParam.ionFit
    #gaussian filtering window size
    size_max = ionParam.ionFilteringWinsizeMax
    size_min = ionParam.ionFilteringWinsizeMin

    #THESE SHOULD BE GOOD ENOUGH, NO NEED TO SET IN setup(self)
    corThresholdIon = 0.85
    #################################################

    print('filtering ionosphere')
    #I find it's better to use ionosphere that is not projected, it's mostly slowlying changing anyway.
    #this should also be better for operational use.
    ionfile = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname, ionParam.ionRawNoProj)
    #since I decide to use ionosphere that is not projected, I should also use coherence that is not projected.
    corfile = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname, ionParam.ionCorNoProj)

    #use ionosphere and coherence that are projected.
    #ionfile = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname, ionParam.ionRaw)
    #corfile = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname, ionParam.ionCor)

    outfile = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname, ionParam.ionFilt)

    img = isceobj.createImage()
    img.load(ionfile + '.xml')
    width = img.width
    length = img.length
    ion = (np.fromfile(ionfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    cor = (np.fromfile(corfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    amp = (np.fromfile(ionfile, dtype=np.float32).reshape(length*2, width))[0:length*2:2, :]

    ########################################################################################
    #AFTER COHERENCE IS RESAMPLED AT grd2ion, THERE ARE SOME WIRED VALUES
    cor[np.nonzero(cor<0)] = 0.0
    cor[np.nonzero(cor>1)] = 0.0
    ########################################################################################

    ion_fit = weight_fitting(ion, cor, width, length, 1, 1, 1, 1, 2, corThresholdIon)

    #no fitting
    if fit == False:
        ion_fit *= 0

    ion -= ion_fit * (ion!=0)
    
    #minimize the effect of low coherence pixels
    #cor[np.nonzero( (cor<0.85)*(cor!=0) )] = 0.00001
    #filt = adaptive_gaussian(ion, cor, size_max, size_min)
    #cor**14 should be a good weight to use. 22-APR-2018
    filt = adaptive_gaussian(ion, cor**14, size_max, size_min)

    filt += ion_fit * (filt!=0)

    ion = np.zeros((length*2, width), dtype=np.float32)
    ion[0:length*2:2, :] = amp
    ion[1:length*2:2, :] = filt
    ion.astype(np.float32).tofile(outfile)
    img.filename = outfile
    img.extraFilename = outfile + '.vrt'
    img.renderHdr()


def ionosphere_shift(self, ionParam):
    '''
    calculate azimuth shift caused by ionosphere using ionospheric phase
    '''

    #################################################
    #SET PARAMETERS HERE
    #gaussian filtering window size
    #size = np.int(np.around(width / 12.0))
    #size = ionParam.ionshiftFilteringWinsize
    size_max = ionParam.ionshiftFilteringWinsizeMax
    size_min = ionParam.ionshiftFilteringWinsizeMin

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
    ionfile = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname, ionParam.ionFilt)
    #we are using filtered ionosphere, so we should use coherence file that is not projected.
    #corfile = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname, ionParam.ionCor)
    corfile = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname, ionParam.ionCorNoProj)
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

    #use the satellite height of the mid burst of first swath of master acquistion
    swathList = self._insar.getValidSwathList(self.swaths)
    master = self._insar.loadProduct( os.path.join(self._insar.masterSlcProduct, 'IW{0}.xml'.format(swathList[0])))
    minBurst, maxBurst = self._insar.commonMasterBurstLimits(swathList[0]-1)
    #no problem with this index at all
    midBurst = np.int(np.around((minBurst+ maxBurst-1) / 2.0))
    masBurst = master.bursts[midBurst]

    #shift casued by ionosphere [unit: masBurst.azimuthTimeInterval]
    rng = masBurst.rangePixelSize * ((np.arange(width))*ionParam.numberRangeLooks + (ionParam.numberRangeLooks - 1.0) / 2.0) + masBurst.startingRange
    Ka = masBurst.azimuthFMRate(rng)
    ionShift = dion / (masBurst.azimuthTimeInterval * ionParam.numberAzimuthLooks) / (4.0 * np.pi) / Ka[None, :] / masBurst.azimuthTimeInterval

    #output
    outfile = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname, ionParam.ionShift)
    tmp = np.zeros((length*2, width), dtype=np.float32)
    tmp[0:length*2:2, :] = amp
    tmp[1:length*2:2, :] = ionShift
    tmp.astype(np.float32).tofile(outfile)
    img.filename = outfile
    img.extraFilename = outfile + '.vrt'
    img.renderHdr()


def ion2grd(self, ionParam):

    #################################################
    #SET PARAMETERS HERE
    #correct phase error caused by non-zero center frequency
    #and azimuth shift caused by ionosphere
    #0: no correction
    #1: use mean value of a burst
    #2: use full burst
    azshiftFlag = ionParam.azshiftFlag
    #################################################

    print('resampling ionosphere from ionospheric layer to ground')
    #get files
    ionFiltFile = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname, ionParam.ionFilt)
    dionfile = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname, ionParam.ionShift)
    corfile = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname, ionParam.ionCorNoProj)
    img = isceobj.createImage()
    img.load(ionFiltFile + '.xml')
    width = img.width
    length = img.length
    ion = (np.fromfile(ionFiltFile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    dion = (np.fromfile(dionfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    cor = (np.fromfile(corfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]

    print('resampling ionosphere in range')
    #in the following, column index of burst (one look) will never exceed merged image index (one look) on the left side.
    #so we only add one multi-looked sample on the right side in case it exceeds on this side
    #index starts from 0
    ionOneRangeLook = np.zeros((length, (width+1)*ionParam.numberRangeLooks), dtype=np.float32)
    if azshiftFlag == 2:
        dionOneRangeLook = np.zeros((length, (width+1)*ionParam.numberRangeLooks), dtype=np.float32)
    indexRange = np.linspace(1-1, (width+1)*ionParam.numberRangeLooks-1, num=(width+1)*ionParam.numberRangeLooks, endpoint=True)
    indexRange2 = multilookIndex(1-1, width*ionParam.numberRangeLooks-1, ionParam.numberRangeLooks)
    for i in range(length):
        f = interp1d(indexRange2, ion[i, :], kind='cubic', fill_value="extrapolate")
        ionOneRangeLook[i, :] = f(indexRange)
        if azshiftFlag == 2:
            f2 = interp1d(indexRange2, dion[i, :], kind='cubic', fill_value="extrapolate")
            dionOneRangeLook[i, :] = f2(indexRange)
 
    #use the satellite height of the mid burst of first swath of master acquistion
    swathList = self._insar.getValidSwathList(self.swaths)
    master = self._insar.loadProduct( os.path.join(self._insar.masterSlcProduct, 'IW{0}.xml'.format(swathList[0])))
    minBurst, maxBurst = self._insar.commonMasterBurstLimits(swathList[0]-1)
    #no problem with this index at all
    midBurst = np.int(np.around((minBurst+ maxBurst-1) / 2.0))
    masBurst = master.bursts[midBurst]
    #satellite height
    satHeight = np.linalg.norm(masBurst.orbit.interpolateOrbit(masBurst.sensingMid, method='hermite').getPosition())
    #orgininal doppler offset should be multiplied by this ratio
    ratio = ionParam.ionHeight/(satHeight-ionParam.earthRadius)

    xmlDirname = os.path.join(ionParam.ionDirname, ionParam.lowerDirname, ionParam.fineIfgDirname)
    (box, burstValidBox, burstValidBox2, frames) = getMergeBox(self, xmlDirname, numberRangeLooks=ionParam.numberRangeLooks, numberAzimuthLooks=ionParam.numberAzimuthLooks)

    ##############################################################################################################
    swathList = self._insar.getValidSwathList(self.swaths)
    frames=[]
    swathList2 = []
    minBurst2 =[]
    #for valid swaths and bursts, consistent with runMergeBursts.py
    for swath in swathList:
        minBurst, maxBurst = self._insar.commonMasterBurstLimits(swath-1)

        if minBurst==maxBurst:
            print('Skipping processing of swath {0}'.format(swath))
            continue

        ifg = self._insar.loadProduct( os.path.join(xmlDirname, 'IW{0}.xml'.format(swath)))
        frames.append(ifg)
        swathList2.append(swath)
        minBurst2.append(minBurst)
    ##############################################################################################################

    print('resampling ionosphere in azimuth')
    nswath = len(frames)
    for i in range(nswath):
        nburst = len(frames[i].bursts)
        ###output directory for burst ionosphere
        outdir = os.path.join(ionParam.ionDirname, ionParam.ionBurstDirname, 'IW{0}'.format(swathList2[i]))
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        for j in range(nburst):
            #according to runBurstIfg.py, this is originally from self._insar.masterSlcProduct, 'IW{0}.xml'
            masBurst = frames[i].bursts[j]
            (dopplerOffset, Ka) = computeDopplerOffset(masBurst, 1, masBurst.numberOfLines, 1, masBurst.numberOfSamples, nrlks=1, nalks=1)
            offset = ratio * dopplerOffset
            #output ionosphere for this burst
            burstIon = np.zeros((masBurst.numberOfLines, masBurst.numberOfSamples), dtype=np.float32)
            burstDion = np.zeros((masBurst.numberOfLines, masBurst.numberOfSamples), dtype=np.float32)

            #              index in merged           index in burst
            lineOff = burstValidBox[i][j][0] - burstValidBox2[i][j][0]
            columnOff = burstValidBox[i][j][2] - burstValidBox2[i][j][2]
            #use index starts from 0
            #1-look index of burst in the 1-look merged image
            indexBurst0 = np.linspace(0+lineOff, masBurst.numberOfLines-1+lineOff, num=masBurst.numberOfLines, endpoint=True)
            #1-look index of multi-looked merged image in the 1-look merged image
            indexMerged = multilookIndex(1-1, length*ionParam.numberAzimuthLooks-1, ionParam.numberAzimuthLooks)
            for k in range(masBurst.numberOfSamples):
                index = indexMerged
                value = ionOneRangeLook[:, k+columnOff]
                f = interp1d(index, value, kind='cubic', fill_value="extrapolate")

                indexBurst = indexBurst0 + offset[:, k]
                burstIon[:, k] = f(indexBurst)

                if azshiftFlag == 2:
                    value2 = dionOneRangeLook[:, k+columnOff]
                    f2 = interp1d(index, value2, kind='cubic', fill_value="extrapolate")
                    burstDion[:, k] = f2(indexBurst)

            #calculate phase caused by ionospheric shift and non-zero center frequency
            #index after multi-looking in merged image, index starts from 1
            first_line = np.int(np.around((burstValidBox[i][j][0] - 1) / ionParam.numberAzimuthLooks + 1))
            last_line = np.int(np.around(burstValidBox[i][j][1] / ionParam.numberAzimuthLooks))
            first_sample = np.int(np.around((burstValidBox[i][j][2] - 1) / ionParam.numberRangeLooks + 1))
            last_sample = np.int(np.around(burstValidBox[i][j][3] / ionParam.numberRangeLooks))

            burstDionMultilook = dion[first_line-1:last_line-1+1, first_sample-1:last_sample-1+1]
            #for avoid areas with strong decorrelation like water
            burstCorMultilook = cor[first_line-1:last_line-1+1, first_sample-1:last_sample-1+1]
            #index = np.nonzero(burstDionMultilook!=0)
            index = np.nonzero(burstCorMultilook>0.85)
            if len(index[0]) < 10:
                dionMean = 0.0
            else:
                dionMean = np.mean(burstDionMultilook[index], dtype=np.float64)

            if azshiftFlag == 0:
                #no correction
                burstIonShift = 0
            elif azshiftFlag == 1:
                #use mean value
                burstIonShift =  2.0 * np.pi * (dopplerOffset * Ka[None,:] * (masBurst.azimuthTimeInterval)) * (dionMean*masBurst.azimuthTimeInterval)
            elif azshiftFlag == 2:
                #use full burst
                burstIonShift =  2.0 * np.pi * (dopplerOffset * Ka[None,:] * (masBurst.azimuthTimeInterval)) * (burstDion*masBurst.azimuthTimeInterval)
            else:
                raise Exception('unknown option for correcting azimuth shift caused by ionosphere!')

            burstIon += burstIonShift
            print('resampling burst %02d of swath %d, azimuth shift caused by ionosphere: %8.5f azimuth lines'%(minBurst2[i]+j+1, swathList2[i], dionMean))

            #create xml and vrt files
            filename = os.path.join(outdir, '%s_%02d.ion'%('burst', minBurst2[i]+j+1))
            burstIon.astype(np.float32).tofile(filename)
            burstImg = isceobj.createImage()
            burstImg.setDataType('FLOAT')
            burstImg.setFilename(filename)
            burstImg.extraFilename = filename + '.vrt'
            burstImg.setWidth(masBurst.numberOfSamples)
            burstImg.setLength(masBurst.numberOfLines)
            burstImg.renderHdr()
        print('')


def multilook(data, nalks, nrlks):
    '''
    doing multiple looking
    
    ATTENTION:
    NO AVERAGING BY DIVIDING THE NUMBER OF TOTAL SAMPLES IS DONE.
    '''

    (length, width)=data.shape
    width2 = np.int(width/nrlks)
    length2 = np.int(length/nalks)

    tmp2 = np.zeros((length2, width), dtype=data.dtype)
    data2 = np.zeros((length2, width2), dtype=data.dtype)
    for i in range(nalks):
        tmp2 += data[i:length2*nalks:nalks, :]
    for i in range(nrlks):
        data2 += tmp2[:, i:width2*nrlks:nrlks]

    return data2


def get_overlap_box(swath, minBurst, maxBurst):

    #number of burst
    nBurst = maxBurst - minBurst
    if nBurst <= 1:
        print('number of burst: {}, no need to get overlap box'.format(nBurst))
        return None

    overlapBox = []
    overlapBox.append([])
    for ii in range(minBurst+1, maxBurst):
        topBurst = swath.bursts[ii-1]
        curBurst = swath.bursts[ii]

        #overlap lines, line index starts from 1
        offLine = np.int(np.round( (curBurst.sensingStart - topBurst.sensingStart).total_seconds() / curBurst.azimuthTimeInterval))
        firstLineTop = topBurst.firstValidLine + 1
        lastLineTop = topBurst.firstValidLine + topBurst.numValidLines
        firstLineCur = offLine + curBurst.firstValidLine + 1
        lastLineCur = offLine + curBurst.firstValidLine + curBurst.numValidLines

        if lastLineTop < firstLineCur:
            raise Exception('there is not enough overlap between burst {} and burst {}\n'.format(ii-1+1, ii+1))

        firstLine = firstLineCur
        lastLine = lastLineTop

        #overlap samples, sample index starts from 1
        offSample = np.int(np.round(       (curBurst.startingRange - topBurst.startingRange) / curBurst.rangePixelSize         ))
        firstSampleTop = topBurst.firstValidSample + 1
        lastSampleTop = topBurst.firstValidSample + topBurst.numValidSamples
        firstSampleCur = offSample + curBurst.firstValidSample + 1
        lastSampleCur = offSample + curBurst.firstValidSample + curBurst.numValidSamples

        firstSample = max(firstSampleTop, firstSampleCur)
        lastSample = min(lastSampleTop, lastSampleCur)

        #overlap area index. all indexes start from 1.
        #                 |                 top burst                  |                                   current burst                                 |
        #                    0           1          2           3              4                   5                    6                     7
        overlapBox.append([firstLine, lastLine, firstSample, lastSample, firstLine-offLine, lastLine-offLine, firstSample-offSample, lastSample-offSample])

    return overlapBox


def esd(self, ionParam):
    '''
    esd after ionosphere correction
    '''
    ######################################
    #SET PARAMETERS HERE
    #THESE SHOULD BE GOOD ENOUGH, NO NEED TO SET IN setup(self)
    nalks = 5
    nrlks = 30
    corThreshold = 0.75
    ######################################

    print('applying ESD to compensate phase error caused by residual misregistration')

    virtual = self.useVirtualFiles
    swathList = self._insar.getValidSwathList(self.swaths)
    for swath in swathList:

        minBurst, maxBurst = self._insar.commonMasterBurstLimits(swath-1)
        nBurst = maxBurst - minBurst

        if nBurst <= 1:
            continue
    
        ####Load relevant products
        master = self._insar.loadProduct( os.path.join(self._insar.masterSlcProduct, 'IW{0}.xml'.format(swath)))
        slave = self._insar.loadProduct( os.path.join(self._insar.fineCoregDirname, 'IW{0}.xml'.format(swath)))

        #get overlap area
        for ii in range(minBurst, maxBurst):
            jj = ii - minBurst
            ####Process the top bursts
            masBurst = master.bursts[ii] 
            slvBurst = slave.bursts[jj]
            adjustValidLineSample(masBurst,slvBurst)
        overlapBox = get_overlap_box(master, minBurst, maxBurst)
        
        #using esd to calculate mis-registration
        misreg = np.array([])
        totalSamples = 0
        for ii in range(minBurst+1, maxBurst):
            jj = ii - minBurst
            ####Process the top bursts
            masBurstTop = master.bursts[ii-1] 
            slvBurstTop = slave.bursts[jj-1]

            masBurstCur = master.bursts[ii] 
            slvBurstCur = slave.bursts[jj]

            #get info
            mastername = masBurstTop.image.filename
            slavename = slvBurstTop.image.filename
            ionname = os.path.join(ionParam.ionDirname, ionParam.ionBurstDirname, 'IW{0}'.format(swath), '%s_%02d.ion'%('burst',ii+1-1))
            rngname = os.path.join(self._insar.fineOffsetsDirname, 'IW{0}'.format(swath), 'range_%02d.off'%(ii+1-1))
            fact = 4.0 * np.pi * slvBurstTop.rangePixelSize / slvBurstTop.radarWavelength
            #infTop = multiply2(mastername, slavename, ionname, rngname, fact, overlapBox[jj][0:4], virtual=virtual)
            infTop = multiply2(mastername, slavename, fact, rngname=rngname, ionname=ionname, infname=None, overlapBox=overlapBox[jj][0:4], valid=True, virtual=virtual)
            (dopTop, Ka) = computeDopplerOffset(masBurstTop, overlapBox[jj][0], overlapBox[jj][1], overlapBox[jj][2], overlapBox[jj][3], nrlks=nrlks, nalks=nalks)
            #rng = multilookIndex(overlapBox[jj][2]-1, overlapBox[jj][3]-1, nrlks) * masBurstTop.rangePixelSize + masBurstTop.startingRange
            #Ka  = masBurstTop.azimuthFMRate(rng)
            frqTop = dopTop * Ka[None,:] * (masBurstTop.azimuthTimeInterval * nalks)

            mastername = masBurstCur.image.filename
            slavename = slvBurstCur.image.filename
            ionname = os.path.join(ionParam.ionDirname, ionParam.ionBurstDirname, 'IW{0}'.format(swath), '%s_%02d.ion'%('burst',ii+1))
            rngname = os.path.join(self._insar.fineOffsetsDirname, 'IW{0}'.format(swath), 'range_%02d.off'%(ii+1))
            fact = 4.0 * np.pi * slvBurstCur.rangePixelSize / slvBurstCur.radarWavelength
            #infCur = multiply2(mastername, slavename, ionname, rngname, fact, overlapBox[jj][4:8], virtual=virtual)
            infCur = multiply2(mastername, slavename, fact, rngname=rngname, ionname=ionname, infname=None, overlapBox=overlapBox[jj][4:8], valid=True, virtual=virtual)
            (dopCur, Ka) = computeDopplerOffset(masBurstCur, overlapBox[jj][4], overlapBox[jj][5], overlapBox[jj][6], overlapBox[jj][7], nrlks=nrlks, nalks=nalks)
            #rng = multilookIndex(overlapBox[jj][6]-1, overlapBox[jj][7]-1, nrlks) * masBurstCur.rangePixelSize + masBurstCur.startingRange
            #Ka  = masBurstCur.azimuthFMRate(rng)
            frqCur = dopCur * Ka[None,:] * (masBurstCur.azimuthTimeInterval * nalks)

            infTop = multilook(infTop, nalks, nrlks)
            infCur = multilook(infCur, nalks, nrlks)
            infDif = infTop * np.conjugate(infCur)
            cor    = cal_coherence(infDif, win=3, edge=4)
            index = np.nonzero(cor > corThreshold)
            totalSamples += infTop.size

            if index[0].size:
                #misregistration in sec. it should be OK to only use master frequency to compute ESD
                misreg0 = np.angle(infDif[index]) / (2.0 * np.pi * (frqTop[index]-frqCur[index]))
                misreg=np.append(misreg, misreg0.flatten())
                print("misregistration at burst %02d and burst %02d of swath %d: %10.5f azimuth lines"%(ii+1-1, ii+1, swath, np.mean(misreg0, dtype=np.float64)/masBurstCur.azimuthTimeInterval))
            else:
                print("no samples available for ESD at burst %02d and burst %02d of swath %d"%(ii+1-1, ii+1, swath))

        percentage = 100.0 * len(misreg) / totalSamples
        #number of samples per overlap: 100/5*23334/150 = 3111.2
        print("samples available for ESD at swath %d: %d out of %d available, percentage: %5.1f%%"%(swath, len(misreg), totalSamples, percentage))
        if len(misreg) < 1000:
            print("too few samples available for ESD, no ESD correction will be applied\n")
            misreg = 0
            continue
        else:
            misreg = np.mean(misreg, dtype=np.float64)
            print("misregistration from ESD: {} sec, {} azimuth lines\n".format(misreg, misreg/master.bursts[minBurst].azimuthTimeInterval))

        #use mis-registration estimated from esd to compute phase error
        for ii in range(minBurst, maxBurst):
            jj = ii - minBurst
            ####Process the top bursts
            masBurst = master.bursts[ii] 
            slvBurst = slave.bursts[jj]

            ionname = os.path.join(ionParam.ionDirname, ionParam.ionBurstDirname, 'IW{0}'.format(swath), '%s_%02d.ion'%('burst',ii+1))
            ion = np.fromfile(ionname, dtype=np.float32).reshape(masBurst.numberOfLines, masBurst.numberOfSamples)
            (dopplerOffset, Ka) = computeDopplerOffset(masBurst, 1, masBurst.numberOfLines, 1, masBurst.numberOfSamples, nrlks=1, nalks=1)
            centerFrequency = dopplerOffset * Ka[None,:] * (masBurst.azimuthTimeInterval)

            ion += 2.0 * np.pi * centerFrequency * misreg
            #overwrite
            ion.astype(np.float32).tofile(ionname)


def esd_noion(self, ionParam):
    '''
    esd after ionosphere correction
    '''
    ######################################
    #SET PARAMETERS HERE
    #THESE SHOULD BE GOOD ENOUGH, NO NEED TO SET IN setup(self)
    nalks = 5
    nrlks = 30
    corThreshold = 0.75
    ######################################

    print('applying ESD to compensate phase error caused by residual misregistration')


    esddir = 'esd'


    virtual = self.useVirtualFiles
    swathList = self._insar.getValidSwathList(self.swaths)
    for swath in swathList:

        minBurst, maxBurst = self._insar.commonMasterBurstLimits(swath-1)
        nBurst = maxBurst - minBurst

        if nBurst <= 1:
            continue
    
        ####Load relevant products
        master = self._insar.loadProduct( os.path.join(self._insar.masterSlcProduct, 'IW{0}.xml'.format(swath)))
        slave = self._insar.loadProduct( os.path.join(self._insar.fineCoregDirname, 'IW{0}.xml'.format(swath)))

        #get overlap area
        for ii in range(minBurst, maxBurst):
            jj = ii - minBurst
            ####Process the top bursts
            masBurst = master.bursts[ii] 
            slvBurst = slave.bursts[jj]
            adjustValidLineSample(masBurst,slvBurst)
        overlapBox = get_overlap_box(master, minBurst, maxBurst)
        
        #using esd to calculate mis-registration
        misreg = np.array([])
        totalSamples = 0
        for ii in range(minBurst+1, maxBurst):
            jj = ii - minBurst
            ####Process the top bursts
            masBurstTop = master.bursts[ii-1] 
            slvBurstTop = slave.bursts[jj-1]

            masBurstCur = master.bursts[ii] 
            slvBurstCur = slave.bursts[jj]

            #get info
            mastername = masBurstTop.image.filename
            slavename = slvBurstTop.image.filename
            ionname = os.path.join(ionParam.ionDirname, ionParam.ionBurstDirname, 'IW{0}'.format(swath), '%s_%02d.ion'%('burst',ii+1-1))
            rngname = os.path.join(self._insar.fineOffsetsDirname, 'IW{0}'.format(swath), 'range_%02d.off'%(ii+1-1))
            fact = 4.0 * np.pi * slvBurstTop.rangePixelSize / slvBurstTop.radarWavelength
            #infTop = multiply2(mastername, slavename, ionname, rngname, fact, overlapBox[jj][0:4], virtual=virtual)
            infTop = multiply2(mastername, slavename, fact, rngname=rngname, ionname=None, infname=None, overlapBox=overlapBox[jj][0:4], valid=True, virtual=virtual)
            (dopTop, Ka) = computeDopplerOffset(masBurstTop, overlapBox[jj][0], overlapBox[jj][1], overlapBox[jj][2], overlapBox[jj][3], nrlks=nrlks, nalks=nalks)
            #rng = multilookIndex(overlapBox[jj][2]-1, overlapBox[jj][3]-1, nrlks) * masBurstTop.rangePixelSize + masBurstTop.startingRange
            #Ka  = masBurstTop.azimuthFMRate(rng)
            frqTop = dopTop * Ka[None,:] * (masBurstTop.azimuthTimeInterval * nalks)

            mastername = masBurstCur.image.filename
            slavename = slvBurstCur.image.filename
            ionname = os.path.join(ionParam.ionDirname, ionParam.ionBurstDirname, 'IW{0}'.format(swath), '%s_%02d.ion'%('burst',ii+1))
            rngname = os.path.join(self._insar.fineOffsetsDirname, 'IW{0}'.format(swath), 'range_%02d.off'%(ii+1))
            fact = 4.0 * np.pi * slvBurstCur.rangePixelSize / slvBurstCur.radarWavelength
            #infCur = multiply2(mastername, slavename, ionname, rngname, fact, overlapBox[jj][4:8], virtual=virtual)
            infCur = multiply2(mastername, slavename, fact, rngname=rngname, ionname=None, infname=None, overlapBox=overlapBox[jj][4:8], valid=True, virtual=virtual)
            (dopCur, Ka) = computeDopplerOffset(masBurstCur, overlapBox[jj][4], overlapBox[jj][5], overlapBox[jj][6], overlapBox[jj][7], nrlks=nrlks, nalks=nalks)
            #rng = multilookIndex(overlapBox[jj][6]-1, overlapBox[jj][7]-1, nrlks) * masBurstCur.rangePixelSize + masBurstCur.startingRange
            #Ka  = masBurstCur.azimuthFMRate(rng)
            frqCur = dopCur * Ka[None,:] * (masBurstCur.azimuthTimeInterval * nalks)

            infTop = multilook(infTop, nalks, nrlks)
            infCur = multilook(infCur, nalks, nrlks)
            infDif = infTop * np.conjugate(infCur)
            cor    = cal_coherence(infDif, win=3, edge=4)
            index = np.nonzero(cor > corThreshold)
            totalSamples += infTop.size

            if index[0].size:
                #misregistration in sec. it should be OK to only use master frequency to compute ESD
                misreg0 = np.angle(infDif[index]) / (2.0 * np.pi * (frqTop[index]-frqCur[index]))
                misreg=np.append(misreg, misreg0.flatten())
                print("misregistration at burst %02d and burst %02d of swath %d: %10.5f azimuth lines"%(ii+1-1, ii+1, swath, np.mean(misreg0, dtype=np.float64)/masBurstCur.azimuthTimeInterval))
            else:
                print("no samples available for ESD at burst %02d and burst %02d of swath %d"%(ii+1-1, ii+1, swath))

        percentage = 100.0 * len(misreg) / totalSamples
        #number of samples per overlap: 100/5*23334/150 = 3111.2
        print("samples available for ESD at swath %d: %d out of %d available, percentage: %5.1f%%"%(swath, len(misreg), totalSamples, percentage))
        if len(misreg) < 1000:
            print("too few samples available for ESD, no ESD correction will be applied\n")
            misreg = 0
            continue
        else:
            misreg = np.mean(misreg, dtype=np.float64)
            print("misregistration from ESD: {} sec, {} azimuth lines\n".format(misreg, misreg/master.bursts[minBurst].azimuthTimeInterval))


        sdir = os.path.join(ionParam.ionDirname, esddir, 'IW{0}'.format(swath))
        if not os.path.exists(sdir):
            os.makedirs(sdir)

        #use mis-registration estimated from esd to compute phase error
        for ii in range(minBurst, maxBurst):
            jj = ii - minBurst
            ####Process the top bursts
            masBurst = master.bursts[ii] 
            slvBurst = slave.bursts[jj]

            #ionname = os.path.join(ionParam.ionDirname, ionParam.ionBurstDirname, 'IW{0}'.format(swath), '%s_%02d.ion'%('burst',ii+1))
            #ion = np.fromfile(ionname, dtype=np.float32).reshape(masBurst.numberOfLines, masBurst.numberOfSamples)
            
            (dopplerOffset, Ka) = computeDopplerOffset(masBurst, 1, masBurst.numberOfLines, 1, masBurst.numberOfSamples, nrlks=1, nalks=1)
            centerFrequency = dopplerOffset * Ka[None,:] * (masBurst.azimuthTimeInterval)

            ion = 2.0 * np.pi * centerFrequency * misreg
            #overwrite
            ionname = os.path.join(ionParam.ionDirname, esddir, 'IW{0}'.format(swath), '%s_%02d.esd'%('burst',ii+1))
            ion.astype(np.float32).tofile(ionname)



            #create xml and vrt files
            burstImg = isceobj.createImage()
            burstImg.setDataType('FLOAT')
            burstImg.setFilename(ionname)
            burstImg.extraFilename = ionname + '.vrt'
            burstImg.setWidth(masBurst.numberOfSamples)
            burstImg.setLength(masBurst.numberOfLines)
            burstImg.renderHdr()


def rawion(self, ionParam):
    '''
    a simple wrapper
    '''

    if ionParam.calIonWithMerged == True:
        #merge bursts
        merge(self, ionParam)

        #unwrap
        unwrap(self, ionParam)

        #compute ionosphere
        ionosphere(self, ionParam)
    else:
        #an alternative of the above steps: processing swath by swath
        ionSwathBySwath(self, ionParam)


def run_step(currentStep, ionParam):
    return ionParam.allSteps.index(ionParam.startStep) <= ionParam.allSteps.index(currentStep) <= ionParam.allSteps.index(ionParam.endStep)


def runIon(self):

    #get processing parameters
    ionParam = setup(self)

    #if do ionospheric correction
    if ionParam.doIon == False:
        return

    #form subband interferograms
    if run_step('subband', ionParam):
        subband(self, ionParam)

    #compute ionosphere (raw_no_projection.ion) and coherence (raw_no_projection.cor) without projection
    if run_step('rawion', ionParam):
        rawion(self, ionParam)
    #next we move to 'ion_cal' to do the remaining processing

    #resample ionosphere from the ground layer to ionospheric layer
    if run_step('grd2ion', ionParam):
        grd2ion(self, ionParam)

    #filter ionosphere
    if run_step('filt_gaussian', ionParam):
        filt_gaussian(self, ionParam)

    #ionosphere shift
    if run_step('ionosphere_shift', ionParam):
        ionosphere_shift(self, ionParam)

    #resample from ionospheric layer to ground layer, get ionosphere for each burst
    if run_step('ion2grd', ionParam):
        ion2grd(self, ionParam)

    #esd
    if run_step('esd', ionParam):
        esd(self, ionParam)

    #pure esd without applying ionospheric correction
    #esd_noion(self, ionParam)

    return
