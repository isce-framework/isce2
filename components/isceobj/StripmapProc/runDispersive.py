#
# Author: Heresh Fattahi, Cunren Liang
#
#
import logging
import os,gdal
import isceobj
from isceobj.Constants import SPEED_OF_LIGHT
import numpy as np




logger = logging.getLogger('isce.insar.runDispersive')

def getValue(dataFile, band, y_ref, x_ref):
    ds = gdal.Open(dataFile, gdal.GA_ReadOnly)
    length = ds.RasterYSize
    width = ds.RasterXSize

    b = ds.GetRasterBand(band)
    ref = b.ReadAsArray(x_ref,y_ref,1,1)
    
    ds = None
    return ref[0][0]


def dispersive_nonDispersive(lowBandIgram, highBandIgram, f0, fL, fH, outDispersive, outNonDispersive, y_ref=None, x_ref=None, m=None , d=None):
    
    if y_ref and x_ref:
        refL = getValue(lowBandIgram, 2, y_ref, x_ref)
        refH = getValue(highBandIgram, 2, y_ref, x_ref)

    else:
        refL = 0.0
        refH = 0.0
    
    # m : common phase unwrapping error
    # d : differential phase unwrapping error

    if m and d:

        coef = (fL*fH)/(f0*(fH**2 - fL**2))
        cmd = 'imageMath.py -e="{0}*((a_1-2*PI*c)*{1}-(b_1+(2.0*PI)-2*PI*(c+f))*{2})" --a={3} --b={4} --c={5} --f={6} -o {7} -t float32 -s BIL'.format(coef,fH, fL, lowBandIgram, highBandIgram, m , d, outDispersive)
        print(cmd)
        os.system(cmd)

        coefn = f0/(fH**2-fL**2)
        cmd = 'imageMath.py -e="{0}*((a_1+(2.0*PI)-2*PI*c)*{1}-(b_1-2*PI*(c+f))*{2})" --a={3} --b={4} --c={5} --f={6} -o {7} -t float32 -s BIL'.format(coefn,fH, fL, highBandIgram, lowBandIgram, m , d, outNonDispersive)
        print(cmd)
        os.system(cmd)

    else:
        
        coef = (fL*fH)/(f0*(fH**2 - fL**2))
        cmd = 'imageMath.py -e="{0}*(a_1*{1}-(b_1+2.0*PI)*{2})" --a={3} --b={4} -o {5} -t float32 -s BIL'.format(coef,fH, fL, lowBandIgram, highBandIgram, outDispersive)

        print(cmd)
        os.system(cmd)

        coefn = f0/(fH**2-fL**2)
        cmd = 'imageMath.py -e="{0}*((a_1+2.0*PI)*{1}-(b_1)*{2})" --a={3} --b={4} -o {5} -t float32 -s BIL'.format(coefn,fH, fL, highBandIgram, lowBandIgram, outNonDispersive)
        print(cmd)
        os.system(cmd)


    return None

def std_iono_mean_coh(f0,fL,fH,coh_mean,rgLooks,azLooks):
    
    # From Liao et al., Remote Sensing of Environment 2018
    
    # STD sub-band at average coherence value (Eq. 8)
    Nb = (rgLooks*azLooks)/3.0
    coeffA = (np.sqrt(2.0*Nb))**(-1)
    coeffB = np.sqrt(1-coh_mean**2)/coh_mean
    std_subbands = coeffA * coeffB
    
    # STD Ionosphere (Eq. 7)
    coeffC = np.sqrt(1+(fL/fH)**2)
    coeffD = (fH*fL*fH)/(f0*(fH**2-fL**2))
    std_iono = coeffC*coeffD*std_subbands
    
    return std_iono
    
def theoretical_variance_fromSubBands(self, f0, fL, fH, B, Sig_phi_iono, Sig_phi_nonDisp,N):
    
    # Calculating the theoretical variance of the ionospheric phase based on the coherence of the sub-band interferograns 
    ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.lowBandSlcDirname)
    lowBandCoherence = os.path.join(ifgDirname , self.insar.coherenceFilename)
    Sig_phi_L = os.path.join(ifgDirname , 'filt_' + self.insar.ifgFilename + ".sig")

    ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.highBandSlcDirname)
    highBandCoherence = os.path.join(ifgDirname , self.insar.coherenceFilename)
    Sig_phi_H = os.path.join(ifgDirname , 'filt_' + self.insar.ifgFilename + ".sig")
    
    cmd = 'imageMath.py -e="sqrt(1-a**2)/a/sqrt(2.0*{0})" --a={1} -o {2} -t float -s BIL'.format(N, lowBandCoherence, Sig_phi_L)
   
    print(cmd)
    os.system(cmd)
    cmd = 'imageMath.py -e="sqrt(1-a**2)/a/sqrt(2.0*{0})" --a={1} -o {2} -t float -s BIL'.format(N, highBandCoherence, Sig_phi_H)
    print(cmd)
    os.system(cmd)

    coef = (fL*fH)/(f0*(fH**2 - fL**2))

    cmd = 'imageMath.py -e="sqrt(({0}**2)*({1}**2)*(a**2) + ({0}**2)*({2}**2)*(b**2))" --a={3} --b={4} -o {5} -t float -s BIL'.format(coef, fL, fH, Sig_phi_L, Sig_phi_H, Sig_phi_iono)
    os.system(cmd)

    coef_non = f0/(fH**2 - fL**2)
    cmd = 'imageMath.py -e="sqrt(({0}**2)*({1}**2)*(a**2) + ({0}**2)*({2}**2)*(b**2))" --a={3} --b={4} -o {5} -t float -s BIL'.format(coef_non, fL, fH, Sig_phi_L, Sig_phi_H, Sig_phi_nonDisp)
    os.system(cmd)

  
    return None #Sig_phi_iono, Sig_phi_nonDisp

def lowPassFilter(self,dataFile, sigDataFile, maskFile, Sx, Sy, sig_x, sig_y, iteration=5, theta=0.0):
    ds = gdal.Open(dataFile + '.vrt', gdal.GA_ReadOnly)
    length = ds.RasterYSize
    width = ds.RasterXSize

    dataIn = np.memmap(dataFile, dtype=np.float32, mode='r', shape=(length,width))
    sigData = np.memmap(sigDataFile, dtype=np.float32, mode='r', shape=(length,width))
    mask = np.memmap(maskFile, dtype=np.byte, mode='r', shape=(length,width))

    dataF, sig_dataF = iterativeFilter(self,dataIn[:,:], mask[:,:], sigData[:,:], iteration, Sx, Sy, sig_x, sig_y, theta)

    filtDataFile = dataFile + ".filt"
    sigFiltDataFile  = sigDataFile + ".filt"
    filtData = np.memmap(filtDataFile, dtype=np.float32, mode='w+', shape=(length,width))
    filtData[:,:] = dataF[:,:]
    filtData.flush()

    sigFilt= np.memmap(sigFiltDataFile, dtype=np.float32, mode='w+', shape=(length,width))
    sigFilt[:,:] = sig_dataF[:,:]
    sigFilt.flush()

    # writing xml and vrt files
    write_xml(filtDataFile, width, length, 1, "FLOAT", "BIL")
    write_xml(sigFiltDataFile, width, length, 1, "FLOAT", "BIL")

    return filtDataFile, sigFiltDataFile

def write_xml(fileName,width,length,bands,dataType,scheme):

    img = isceobj.createImage()
    img.setFilename(fileName)
    img.setWidth(width)
    img.setLength(length)
    img.setAccessMode('READ')
    img.bands = bands
    img.dataType = dataType
    img.scheme = scheme
    img.renderHdr()
    img.renderVRT()
    
    return None

def iterativeFilter(self,dataIn, mask, Sig_dataIn, iteration, Sx, Sy, sig_x, sig_y, theta=0.0):
    data = np.zeros(dataIn.shape)
    data[:,:] = dataIn[:,:]
    Sig_data = np.zeros(dataIn.shape)
    Sig_data[:,:] = Sig_dataIn[:,:]

    print ('masking the data')
    data[mask==0]=np.nan
    Sig_data[mask==0]=np.nan
    
    if self.dispersive_filling_method == "smoothed":
       print('Filling the holes with smoothed values')
       dataF = fill_with_smoothed(data,3)
       Sig_data = fill_with_smoothed(Sig_data,3) 
    else:
       print ('Filling the holes with nearest neighbor interpolation')
       dataF = fill(data)
       Sig_data = fill(Sig_data)
       
    print ('Low pass Gaussian filtering the interpolated data')
    dataF, Sig_dataF = Filter(dataF, Sig_data, Sx, Sy, sig_x, sig_y, theta=0.0)
    for i in range(iteration):
       print ('iteration: ', i , ' of ',iteration)
       print ('masking the interpolated and filtered data')
       dataF[mask==0]=np.nan
       
       if self.dispersive_filling_method == "smoothed":
          print("Fill the holes with smoothed values")
          dataF = fill_with_smoothed(dataF,3)
       else:
          print('Filling the holes with nearest neighbor interpolation of the filtered data from previous step')
          dataF = fill(dataF)

       print('Replace the valid pixels with original unfiltered data')
       dataF[mask==1]=data[mask==1]
       dataF, Sig_dataF = Filter(dataF, Sig_data, Sx, Sy, sig_x, sig_y, theta=0.0)

    return dataF, Sig_dataF

def Filter(data, Sig_data, Sx, Sy, sig_x, sig_y, theta=0.0):
    
    import cv2

    kernel = Gaussian_kernel(Sx, Sy, sig_x, sig_y) #(800, 800, 15.0, 100.0)
    kernel = rotate(kernel , theta)

    data = data/Sig_data**2
    data = cv2.filter2D(data,-1,kernel)
    W1 = cv2.filter2D(1.0/Sig_data**2,-1,kernel)
    W2 = cv2.filter2D(1.0/Sig_data**2,-1,kernel**2)

    return data/W1, np.sqrt(W2/(W1**2))

def Gaussian_kernel(Sx, Sy, sig_x,sig_y):
    if np.mod(Sx,2) == 0:
        Sx = Sx + 1

    if np.mod(Sy,2) ==0:
            Sy = Sy + 1

    x,y = np.meshgrid(np.arange(Sx),np.arange(Sy))
    x = x + 1
    y = y + 1
    x0 = (Sx+1)/2
    y0 = (Sy+1)/2
    fx = ((x-x0)**2.)/(2.*sig_x**2.)
    fy = ((y-y0)**2.)/(2.*sig_y**2.)
    k = np.exp(-1.0*(fx+fy))
    a = 1./np.sum(k)
    k = a*k
    return k

def rotate(k , theta):

    Sy,Sx = np.shape(k)
    x,y = np.meshgrid(np.arange(Sx),np.arange(Sy))

    x = x + 1
    y = y + 1
    x0 = (Sx+1)/2
    y0 = (Sy+1)/2
    x = x - x0
    y = y - y0

    A=np.vstack((x.flatten(), y.flatten()))
    if theta!=0:
        theta = theta*np.pi/180.
        R = np.array([[np.cos(theta), -1.0*np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        AR = np.dot(R,A)
        xR = AR[0,:].reshape(Sy,Sx)
        yR = AR[1,:].reshape(Sy,Sx)

        k = mlab.griddata(x.flatten(),y.flatten(),k.flatten(),xR,yR, interp='linear')
        #k = f(xR, yR)
        k = k.data
        k[np.isnan(k)] = 0.0
        a = 1./np.sum(k)
        k = a*k
    return k

def fill_with_smoothed(off,filterSize):
    
    from astropy.convolution import convolve
    
    off_2filt=np.copy(off)
    kernel = np.ones((filterSize,filterSize),np.float32)/(filterSize*filterSize)
    loop = 0
    cnt2=1
    
    while (cnt2!=0 & loop<100):
       loop += 1
       idx2= np.isnan(off_2filt)
       cnt2 = np.sum(np.count_nonzero(np.isnan(off_2filt)))
       print(cnt2)
       if cnt2 != 0:
          off_filt= convolve(off_2filt,kernel,boundary='extend',nan_treatment='interpolate')
          off_2filt[idx2]=off_filt[idx2]
          idx3 = np.where(off_filt == 0)
          off_2filt[idx3]=np.nan
          off_filt=None
          
    return off_2filt
    


def fill(data, invalid=None):
    
    from scipy import ndimage
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell
    
    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'.
                 data value are replaced where invalid is True
                 If None (default), use: invalid  = np.isnan(data)
       
    Output:
        Return a filled array.
    """
    if invalid is None: invalid = np.isnan(data)

    ind = ndimage.distance_transform_edt(invalid,
                                    return_distances=False,
                                    return_indices=True)
    return data[tuple(ind)]


def getMask(self, maskFile,std_iono):
    
    from scipy.ndimage import median_filter
    
    ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.lowBandSlcDirname)
    lowBandIgram = os.path.join(ifgDirname , 'filt_' + self.insar.ifgFilename )
    lowBandCor = os.path.join(ifgDirname ,self.insar.coherenceFilename)
    
    if '.flat' in lowBandIgram:
        lowBandIgram = lowBandIgram.replace('.flat', '.unw')
    elif '.int' in lowBandIgram:
        lowBandIgram = lowBandIgram.replace('.int', '.unw')
    else:
        lowBandIgram += '.unw'

    ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.highBandSlcDirname)
    highBandIgram = os.path.join(ifgDirname , 'filt_' + self.insar.ifgFilename )
    highBandCor = os.path.join(ifgDirname ,self.insar.coherenceFilename)

    if '.flat' in highBandIgram:
        highBandIgram = highBandIgram.replace('.flat', '.unw')
    elif '.int' in lowBandIgram:
        highBandIgram = highBandIgram.replace('.int', '.unw')
    else:
        highBandIgram += '.unw'

    if (self.dispersive_filter_mask_type == "coherence") and (not self.dispersive_filter_mask_type == "median_filter"):
        print ('generating a mask based on coherence files of sub-band interferograms with a threshold of {0}'.format(self.dispersive_filter_coherence_threshold))
        cmd = 'imageMath.py -e="(a>{0})*(b>{0})" --a={1} --b={2} -t byte -s BIL -o {3}'.format(self.dispersive_filter_coherence_threshold, lowBandCor, highBandCor, maskFile)
        os.system(cmd)
    elif (self.dispersive_filter_mask_type == "connected_components") and ((os.path.exists(lowBandIgram + '.conncomp')) and (os.path.exists(highBandIgram + '.conncomp'))):
       # If connected components from snaphu exists, let's get a mask based on that. 
       # Regions of zero are masked out. Let's assume that islands have been connected. 
        print ('generating a mask based on .conncomp files')
        cmd = 'imageMath.py -e="(a>0)*(b>0)" --a={0} --b={1} -t byte -s BIL -o {2}'.format(lowBandIgram + '.conncomp', highBandIgram + '.conncomp', maskFile)
        os.system(cmd)

    elif self.dispersive_filter_mask_type == "median_filter":
        print('Generating mask based on median filtering of the raw dispersive component')
	
	# Open raw dispersive component (non-filtered, no unwrapping-error corrected)
        dispFilename = os.path.join(self.insar.ionosphereDirname,self.insar.dispersiveFilename)
        sigFilename = os.path.join(self.insar.ionosphereDirname,self.insar.dispersiveFilename+'.sig')
	
        ds = gdal.Open(dispFilename+'.vrt',gdal.GA_ReadOnly)
        disp = ds.GetRasterBand(1).ReadAsArray()
        ds=None

        mask = (np.abs(disp-median_filter(disp,15))<3*std_iono) 
        
        mask = mask.astype(np.float32)
        mask.tofile(maskFile)
        dims=np.shape(mask)
        write_xml(maskFile,dims[1],dims[0],1,"FLOAT","BIL")

    else:
        print ('generating a mask based on unwrapped files. Pixels with phase = 0 are masked out.')
        cmd = 'imageMath.py -e="(a_1!=0)*(b_1!=0)" --a={0} --b={1} -t byte -s BIL -o {2}'.format(lowBandIgram , highBandIgram , maskFile)
        os.system(cmd)

def unwrapp_error_correction(f0, B, dispFile, nonDispFile,lowBandIgram, highBandIgram, y_ref=None, x_ref=None):

    dFile = os.path.join(os.path.dirname(dispFile) , "dJumps.bil")
    mFile = os.path.join(os.path.dirname(dispFile) , "mJumps.bil")

    if y_ref and x_ref:
        refL = getValue(lowBandIgram, 2, y_ref, x_ref)
        refH = getValue(highBandIgram, 2, y_ref, x_ref)

    else:
        refL = 0.0
        refH = 0.0

    cmd = 'imageMath.py -e="round(((a_1+(2.0*PI)) - (b_1) - (2.0*{0}/3.0/{1})*c + (2.0*{0}/3.0/{1})*f )/2.0/PI)" --a={2} --b={3} --c={4} --f={5}  -o {6} -t float32 -s BIL'.format(B, f0, highBandIgram, lowBandIgram, nonDispFile, dispFile, dFile)
    print(cmd)
    os.system(cmd)
    
    cmd = 'imageMath.py -e="round(((a_1 ) + (b_1+(2.0*PI)) - 2.0*c - 2.0*f )/4.0/PI - g/2)" --a={0} --b={1} --c={2} --f={3} --g={4} -o {5} -t float32 -s BIL'.format(lowBandIgram, highBandIgram, nonDispFile, dispFile, dFile, mFile)
    print(cmd)
    os.system(cmd)

    return mFile , dFile


def runDispersive(self):

    if not self.doDispersive:
        print('Estimating dispersive phase not requested ... skipping')
        return

    ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.lowBandSlcDirname)
    lowBandIgram = os.path.join(ifgDirname , 'filt_' + self.insar.ifgFilename)

    if '.flat' in lowBandIgram:
        lowBandIgram = lowBandIgram.replace('.flat', '.unw')
    elif '.int' in lowBandIgram:
        lowBandIgram = lowBandIgram.replace('.int', '.unw')
    else:
        lowBandIgram += '.unw'

    ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.highBandSlcDirname)
    highBandIgram = os.path.join(ifgDirname , 'filt_' + self.insar.ifgFilename)

    if '.flat' in highBandIgram:
        highBandIgram = highBandIgram.replace('.flat', '.unw')
    elif '.int' in highBandIgram:
        highBandIgram = highBandIgram.replace('.int', '.unw')
    else:
        highBandIgram += '.unw'

    outputDir = self.insar.ionosphereDirname
    if os.path.isdir(outputDir):
        logger.info('Ionosphere directory {0} already exists.'.format(outputDir))
    else:
        os.makedirs(outputDir)

    outDispersive = os.path.join(outputDir, self.insar.dispersiveFilename)
    sigmaDispersive = outDispersive + ".sig"

    outNonDispersive = os.path.join(outputDir, self.insar.nondispersiveFilename)
    sigmaNonDispersive = outNonDispersive + ".sig"

    maskFile = os.path.join(outputDir, "mask.bil")

    masterFrame = self._insar.loadProduct( self._insar.masterSlcCropProduct)

    wvl = masterFrame.radarWavelegth
    wvlL = self.insar.lowBandRadarWavelength
    wvlH = self.insar.highBandRadarWavelength

    
    f0 = SPEED_OF_LIGHT/wvl
    fL = SPEED_OF_LIGHT/wvlL
    fH = SPEED_OF_LIGHT/wvlH

    pulseLength = masterFrame.instrument.pulseLength
    chirpSlope = masterFrame.instrument.chirpSlope
   
    # Total Bandwidth
    B = np.abs(chirpSlope)*pulseLength
    
    
    ###Determine looks
    azLooks, rgLooks = self.insar.numberOfLooks( masterFrame, self.posting,
                                        self.numberAzimuthLooks, self.numberRangeLooks)

    # estimating the dispersive and non-dispersive components
    dispersive_nonDispersive(lowBandIgram, highBandIgram, f0, fL, fH, outDispersive, outNonDispersive)

    # If median filter is selected, compute the ionosphere phase standard deviation at a mean coherence value defined by the user
    if self.dispersive_filter_mask_type == "median_filter":
       coh_thres = self.dispersive_filter_coherence_threshold
       std_iono = std_iono_mean_coh(f0,fL,fH,coh_thres,rgLooks,azLooks)
    else:
       std_iono = None
    
    # generating a mask which will help filtering the estimated dispersive and non-dispersive phase
    getMask(self, maskFile,std_iono)
    
    # Calculating the theoretical standard deviation of the estimation based on the coherence of the interferograms
    theoretical_variance_fromSubBands(self, f0, fL, fH, B, sigmaDispersive, sigmaNonDispersive, azLooks * rgLooks) 
    
    # low pass filtering the dispersive phase
    lowPassFilter(self,outDispersive, sigmaDispersive, maskFile, 
                    self.kernel_x_size, self.kernel_y_size, 
                    self.kernel_sigma_x, self.kernel_sigma_y, 
                    iteration = self.dispersive_filter_iterations, 
                    theta = self.kernel_rotation)


    # low pass filtering the  non-dispersive phase
    lowPassFilter(self,outNonDispersive, sigmaNonDispersive, maskFile, 
                    self.kernel_x_size, self.kernel_y_size,
                    self.kernel_sigma_x, self.kernel_sigma_y,
                    iteration = self.dispersive_filter_iterations,
                    theta = self.kernel_rotation)
            
            
    # Estimating phase unwrapping errors
    mFile , dFile = unwrapp_error_correction(f0, B, outDispersive+".filt", outNonDispersive+".filt", 
                                                    lowBandIgram, highBandIgram)

    # re-estimate the dispersive and non-dispersive phase components by taking into account the unwrapping errors
    outDispersive = outDispersive + ".unwCor"
    outNonDispersive = outNonDispersive + ".unwCor"
    dispersive_nonDispersive(lowBandIgram, highBandIgram, f0, fL, fH, outDispersive, outNonDispersive, m=mFile , d=dFile)

    # low pass filtering the new estimations 
    lowPassFilter(self,outDispersive, sigmaDispersive, maskFile, 
                    self.kernel_x_size, self.kernel_y_size,
                    self.kernel_sigma_x, self.kernel_sigma_y,
                    iteration = self.dispersive_filter_iterations,
                    theta = self.kernel_rotation)

    lowPassFilter(self,outNonDispersive, sigmaNonDispersive, maskFile,
                    self.kernel_x_size, self.kernel_y_size,
                    self.kernel_sigma_x, self.kernel_sigma_y,
                    iteration = self.dispersive_filter_iterations,
                    theta = self.kernel_rotation)

