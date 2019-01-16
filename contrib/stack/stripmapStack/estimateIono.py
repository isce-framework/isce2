#
# Author: Heresh Fattahi, Cunren Liang
#
#
import argparse
import logging
import os
import isce
import isceobj
from isceobj.Constants import SPEED_OF_LIGHT
import numpy as np
import gdal
import shelve

from scipy import ndimage
try:
    import cv2
except ImportError:
    print('OpenCV2 does not appear to be installed / is not importable.')
    print('OpenCV2 is needed for this step. You may experience failures ...')


logger = logging.getLogger('isce.insar.runDispersive')


def createParser():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser( description='split the range spectrum of SLC')
    parser.add_argument('-L', '--low_band_igram_prefix', dest='lowBandIgramPrefix', type=str, required=True,
            help='prefix of unwrapped low band interferogram')
    parser.add_argument('-Lu', '--low_band_igram_unw_method', dest='lowBandIgramUnwMethod', type=str, required=True,
            help='unwrap method used for low band interferogram')
    parser.add_argument('-H', '--high_band_igram_prefix', dest='highBandIgramPrefix', type=str, required=True,
            help='prefix of unwrapped high band interferogram')
    parser.add_argument('-Hu', '--high_band_igram_unw_method', dest='highBandIgramUnwMethod', type=str, required=True,
            help='unwrap method used for high band interferogram')
    parser.add_argument('-o', '--outDir', dest='outDir', type=str, required=True,
            help='output directory')
    parser.add_argument('-a', '--low_band_shelve', dest='lowBandShelve', type=str, default=None,
            help='shelve file used to extract metadata')
    parser.add_argument('-b', '--high_band_shelve', dest='highBandShelve', type=str, default=None,
            help='shelve file used to extract metadata')
    parser.add_argument('-c', '--full_band_coherence', dest='fullBandCoherence', type=str, default=None,
            help='full band coherence')
    parser.add_argument('--low_band_coherence', dest='lowBandCoherence', type=str, default=None,
            help='low band coherence')
    parser.add_argument('--high_band_coherence', dest='highBandCoherence', type=str, default=None,
            help='high band coherence')
    parser.add_argument('--azimuth_looks', dest='azLooks', type=float, default=14.0,
            help='high band coherence')
    parser.add_argument('--range_looks', dest='rngLooks', type=float, default=4.0,
            help='high band coherence')

    parser.add_argument('--dispersive_filter_mask_type', dest='dispersive_filter_mask_type', type=str, default='connected_components',
            help='mask type for iterative low-pass filtering: connected_components or coherence')

    parser.add_argument('--dispersive_filter_coherence_threshold', dest='dispersive_filter_coherence_threshold', type=float, default=0.5,
            help='coherence threshold when mask type for iterative low-pass filtering is coherence')

    #parser.add_argument('-f', '--filter_sigma', dest='filterSigma', type=float, default=100.0,
    #        help='sigma of the gaussian filter')

    parser.add_argument('--filter_sigma_x', dest='kernel_sigma_x', type=float, default=100.0,
                help='sigma of the gaussian filter in X direction, default=100')

    parser.add_argument('--filter_sigma_y', dest='kernel_sigma_y', type=float, default=100.0,
                    help='sigma of the gaussian filter in Y direction, default=100')

    parser.add_argument('--filter_size_x', dest='kernel_x_size', type=float, default=800.0,
                            help='size of the gaussian kernel in X direction, default = 800')

    parser.add_argument('--filter_size_y', dest='kernel_y_size', type=float, default=800.0,
                        help='size of the gaussian kernel in Y direction, default=800')

    parser.add_argument('--filter_kernel_rotation', dest='kernel_rotation', type=float, default=0.0,
                        help='rotation angle of the filter kernel in degrees (default = 0.0)')

    parser.add_argument('-i', '--iteration', dest='dispersive_filter_iterations', type=int, default=5,
            help='number of iteration for filtering and interpolation')

    parser.add_argument('-m', '--mask_file', dest='maskFile', type=str, default=None,
            help='a mask file with one for valid pixels and zero for non valid pixels.')
    parser.add_argument('-u', '--outlier_sigma', dest='outlierSigma', type=float, default=1.0,
            help='number of sigma for removing outliers. data outside (avergae +/- u*sigma) are considered as outliers. sigma is calculated from data/coherence. u is the user input. default u =1')
    parser.add_argument('-p', '--min_pixel_connected_component', dest='minPixelConnComp', type=int, default=1000.0,
            help='minimum number of pixels in a connected component to consider the component as valid. components with less pixel will be masked out')
    parser.add_argument('-r', '--ref', dest='ref', type=str, default=None, help='refernce pixel : row, column')
    return parser


def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def getValue(dataFile, band, y_ref, x_ref):
    ds = gdal.Open(dataFile, gdal.GA_ReadOnly)
    length = ds.RasterYSize
    width = ds.RasterXSize

    b = ds.GetRasterBand(band)
    ref = b.ReadAsArray(x_ref,y_ref,1,1)
    
    ds = None
    return ref[0][0]

def check_consistency(lowBandIgram, highBandIgram, outputDir):


    jumpFile = os.path.join(outputDir , "jumps.bil")
    cmd = 'imageMath.py -e="round((a_1-b_1)/(2.0*PI))" --a={0}  --b={1} -o {2} -t float  -s BIL'.format(lowBandIgram, highBandIgram, jumpFile)
    print(cmd)
    os.system(cmd)

    return jumpFile



def dispersive_nonDispersive(lowBandIgram, highBandIgram, f0, fL, fH, outDispersive, outNonDispersive, jumpFile, y_ref=None, x_ref=None, m=None , d=None):
    
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
        #cmd = 'imageMath.py -e="{0}*((a_1-{8}-2*PI*c)*{1}-(b_1-{9}-2*PI*(c+f))*{2})" --a={3} --b={4} --c={5} --f={6} -o {7} -t float32 -s BIL'.format(coef,fH, fL, lowBandIgram, highBandIgram, m , d, outDispersive, refL, refH)
        cmd = 'imageMath.py -e="{0}*((a_1-2*PI*c)*{1}-(b_1+(2.0*PI*g)-2*PI*(c+f))*{2})" --a={3} --b={4} --c={5} --f={6} --g={7} -o {8} -t float32 -s BIL'.format(coef,fH, fL, lowBandIgram, highBandIgram, m , d, jumpFile, outDispersive)
        print(cmd)
        os.system(cmd)

        coefn = f0/(fH**2-fL**2)
        #cmd = 'imageMath.py -e="{0}*((a_1-{8}-2*PI*c)*{1}-(b_1-{9}-2*PI*(c+f))*{2})" --a={3} --b={4} --c={5} --f={6} -o {7} -t float32 -s BIL'.format(coefn,fH, fL, highBandIgram, lowBandIgram, m , d, outNonDispersive, refH, refL)
        cmd = 'imageMath.py -e="{0}*((a_1+(2.0*PI*g)-2*PI*c)*{1}-(b_1-2*PI*(c+f))*{2})" --a={3} --b={4} --c={5} --f={6} --g={7} -o {8} -t float32 -s BIL'.format(coefn,fH, fL, highBandIgram, lowBandIgram, m , d, jumpFile, outNonDispersive)
        print(cmd)
        os.system(cmd)

    else:
        
        coef = (fL*fH)/(f0*(fH**2 - fL**2))
        #cmd = 'imageMath.py -e="{0}*((a_1-{6})*{1}-(b_1-{7})*{2})" --a={3} --b={4} -o {5} -t float32 -s BIL'.format(coef,fH, fL, lowBandIgram, highBandIgram, outDispersive, refL, refH)
        cmd = 'imageMath.py -e="{0}*(a_1*{1}-(b_1+2.0*PI*c)*{2})" --a={3} --b={4} --c={5}  -o {6} -t float32 -s BIL'.format(coef,fH, fL, lowBandIgram, highBandIgram, jumpFile, outDispersive)

        print(cmd)
        os.system(cmd)

        coefn = f0/(fH**2-fL**2)
        #cmd = 'imageMath.py -e="{0}*((a_1-{6})*{1}-(b_1-{7})*{2})" --a={3} --b={4} -o {5} -t float32 -s BIL'.format(coefn,fH, fL, highBandIgram, lowBandIgram, outNonDispersive, refH, refL) 
        cmd = 'imageMath.py -e="{0}*((a_1+2.0*PI*c)*{1}-(b_1)*{2})" --a={3} --b={4} --c={5} -o {6} -t float32 -s BIL'.format(coefn,fH, fL, highBandIgram, lowBandIgram, jumpFile, outNonDispersive)
        print(cmd)
        os.system(cmd)


    return None

def theoretical_variance_fromSubBands(inps, f0, fL, fH, B, Sig_phi_iono, Sig_phi_nonDisp,N):
    # Calculating the theoretical variance of the 
    # ionospheric phase based on the coherence of
    # the sub-band interferograns 
    #ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.lowBandSlcDirname)
    lowBandCoherence = inps.lowBandCoherence 
    Sig_phi_L = inps.Sig_phi_L 

    #ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.highBandSlcDirname)
    #highBandIgram = os.path.join(ifgDirname , 'filt_' + self.insar.ifgFilename + ".unw")

    #ifgDirname = os.path.dirname(self.insar.lowBandIgram)
    #lowBandCoherence = os.path.join(ifgDirname , self.insar.coherenceFilename)
    #Sig_phi_L = os.path.join(ifgDirname , 'filt_' + self.insar.ifgFilename + ".sig")

    #ifgDirname = os.path.dirname(self.insar.highBandIgram)
    #highBandCoherence = os.path.join(ifgDirname , self.insar.coherenceFilename)
    #Sig_phi_H = os.path.join(ifgDirname , 'filt_' + self.insar.ifgFilename + ".sig")

    highBandCoherence = inps.highBandCoherence
    Sig_phi_H = inps.Sig_phi_H

    #N = self.numberAzimuthLooks*self.numberRangeLooks
    #PI = np.pi
    #fL,f0,fH,B = getBandFrequencies(inps)
    #cL = read(inps.lowBandCoherence,bands=[1])
    #cL = cL[0,:,:]
    #cL[cL==0.0]=0.001
    
    cmd = 'imageMath.py -e="sqrt(1-a**2)/a/sqrt(2.0*{0})" --a={1} -o {2} -t float -s BIL'.format(N, lowBandCoherence, Sig_phi_L)
    print(cmd)
    os.system(cmd)
    #Sig_phi_L = np.sqrt(1-cL**2)/cL/np.sqrt(2.*N)

    #cH = read(inps.highBandCoherence,bands=[1])
    #cH = cH[0,:,:]
    #cH[cH==0.0]=0.001

    cmd = 'imageMath.py -e="sqrt(1-a**2)/a/sqrt(2.0*{0})" --a={1} -o {2} -t float -s BIL'.format(N, highBandCoherence, Sig_phi_H)
    print(cmd)
    os.system(cmd)
    #Sig_phi_H = np.sqrt(1-cH**2)/cH/np.sqrt(2.0*N)

    coef = (fL*fH)/(f0*(fH**2 - fL**2))

    cmd = 'imageMath.py -e="sqrt(({0}**2)*({1}**2)*(a**2) + ({0}**2)*({2}**2)*(b**2))" --a={3} --b={4} -o {5} -t float -s BIL'.format(coef, fL, fH, Sig_phi_L, Sig_phi_H, Sig_phi_iono)
    os.system(cmd)

    #Sig_phi_iono = np.sqrt((coef**2)*(fH**2)*Sig_phi_H**2 + (coef**2)*(fL**2)*Sig_phi_L**2)
    #length, width = Sig_phi_iono.shape

    #outFileIono = os.path.join(inps.outDir, 'Sig_iono.bil')
    #write(Sig_phi_iono, outFileIono, 1, 6)
    #write_xml(outFileIono, length, width)

    coef_non = f0/(fH**2 - fL**2)
    cmd = 'imageMath.py -e="sqrt(({0}**2)*({1}**2)*(a**2) + ({0}**2)*({2}**2)*(b**2))" --a={3} --b={4} -o {5} -t float -s BIL'.format(coef_non, fL, fH, Sig_phi_L, Sig_phi_H, Sig_phi_nonDisp)
    os.system(cmd)

    #Sig_phi_non_dis = np.sqrt((coef_non**2) * (fH**2) * Sig_phi_H**2 + (coef_non**2) * (fL**2) * Sig_phi_L**2)

    #outFileNonDis = os.path.join(inps.outDir, 'Sig_nonDis.bil')
    #write(Sig_phi_non_dis, outFileNonDis, 1, 6)
    #write_xml(outFileNonDis, length, width)

    return None #Sig_phi_iono, Sig_phi_nonDisp

def lowPassFilter(dataFile, sigDataFile, maskFile, Sx, Sy, sig_x, sig_y, iteration=5, theta=0.0):
    ds = gdal.Open(dataFile + '.vrt', gdal.GA_ReadOnly)
    length = ds.RasterYSize
    width = ds.RasterXSize

    dataIn = np.memmap(dataFile, dtype=np.float32, mode='r', shape=(length,width))
    sigData = np.memmap(sigDataFile, dtype=np.float32, mode='r', shape=(length,width))
    mask = np.memmap(maskFile, dtype=np.byte, mode='r', shape=(length,width))

    dataF, sig_dataF = iterativeFilter(dataIn[:,:], mask[:,:], sigData[:,:], iteration, Sx, Sy, sig_x, sig_y, theta)

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

def iterativeFilter(dataIn, mask, Sig_dataIn, iteration, Sx, Sy, sig_x, sig_y, theta=0.0):
    data = np.zeros(dataIn.shape)
    data[:,:] = dataIn[:,:]
    Sig_data = np.zeros(dataIn.shape)
    Sig_data[:,:] = Sig_dataIn[:,:]

    print ('masking the data')
    data[mask==0]=np.nan
    Sig_data[mask==0]=np.nan
    print ('Filling the holes with nearest neighbor interpolation')
    dataF = fill(data)
    Sig_data = fill(Sig_data)
    print ('Low pass Gaussian filtering the interpolated data')
    dataF, Sig_dataF = Filter(dataF, Sig_data, Sx, Sy, sig_x, sig_y, theta=0.0)
    for i in range(iteration):
       print ('iteration: ', i , ' of ',iteration)
       print ('masking the interpolated and filtered data')
       dataF[mask==0]=np.nan
       print('Filling the holes with nearest neighbor interpolation of the filtered data from previous step')
       dataF = fill(dataF)
       print('Replace the valid pixels with original unfiltered data')
       dataF[mask==1]=data[mask==1]
       dataF, Sig_dataF = Filter(dataF, Sig_data, Sx, Sy, sig_x, sig_y, theta=0.0)

    return dataF, Sig_dataF

def Filter(data, Sig_data, Sx, Sy, sig_x, sig_y, theta=0.0):
    kernel = Gaussian_kernel(Sx, Sy, sig_x, sig_y) #(800, 800, 15.0, 100.0)
    kernel = rotate(kernel , theta)

    data = data/Sig_data**2
    data = cv2.filter2D(data,-1,kernel)
    W1 = cv2.filter2D(1.0/Sig_data**2,-1,kernel)
    W2 = cv2.filter2D(1.0/Sig_data**2,-1,kernel**2)

    #data = ndimage.convolve(data,kernel, mode='nearest')
    #W1 = ndimage.convolve(1.0/Sig_data**2,kernel, mode='nearest')
    #W2 = ndimage.convolve(1.0/Sig_data**2,kernel**2, mode='nearest')


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

def fill(data, invalid=None):
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


def getMask(inps, maskFile):
   
    lowBandIgram = inps.lowBandIgram 
    lowBandCor = inps.lowBandCoherence #lowBandIgram.replace("_snaphu.unw", ".cor")

    highBandIgram = inps.highBandIgram
    highBandCor = inps.highBandCoherence #highBandIgram.replace("_snaphu.unw", ".cor")    

    if inps.dispersive_filter_mask_type == "coherence":
        print ('generating a mask based on coherence files of sub-band interferograms with a threshold of {0}'.format(inps.dispersive_filter_coherence_threshold))
        cmd = 'imageMath.py -e="(a>{0})*(b>{0})" --a={1} --b={2} -t byte -s BIL -o {3}'.format(inps.dispersive_filter_coherence_threshold, lowBandCor, highBandCor, maskFile)
        os.system(cmd)
    elif (inps.dispersive_filter_mask_type == "connected_components") and ((os.path.exists(lowBandIgram + '.conncomp')) and (os.path.exists(highBandIgram + '.conncomp'))):
       # If connected components from snaphu exists, let's get a mask based on that. 
       # Regions of zero are masked out. Let's assume that islands have been connected. 
        print ('generating a mask based on .conncomp files')
        cmd = 'imageMath.py -e="(a>0)*(b>0)" --a={0} --b={1} -t byte -s BIL -o {2}'.format(lowBandIgram + '.conncomp', highBandIgram + '.conncomp', maskFile)
        os.system(cmd)

    else:
        print ('generating a mask based on unwrapped files. Pixels with phase = 0 are masked out.')
        cmd = 'imageMath.py -e="(a_1!=0)*(b_1!=0)" --a={0} --b={1} -t byte -s BIL -o {2}'.format(lowBandIgram , highBandIgram , maskFile)
        os.system(cmd)

def unwrapp_error_correction(f0, B, dispFile, nonDispFile,lowBandIgram, highBandIgram, jumpsFile, y_ref=None, x_ref=None):

    dFile = os.path.join(os.path.dirname(dispFile) , "dJumps.bil")
    mFile = os.path.join(os.path.dirname(dispFile) , "mJumps.bil")

    if y_ref and x_ref:
        refL = getValue(lowBandIgram, 2, y_ref, x_ref)
        refH = getValue(highBandIgram, 2, y_ref, x_ref)

    else:
        refL = 0.0
        refH = 0.0

    #cmd = 'imageMath.py -e="round(((a_1-{7}) - (b_1-{8}) - (2.0*{0}/3.0/{1})*c + (2.0*{0}/3.0/{1})*f )/2.0/PI)" --a={2} --b={3} --c={4} --f={5}  -o {6} -t float32 -s BIL'.format(B, f0, highBandIgram, lowBandIgram, nonDispFile, dispFile, dFile, refH, refL)

    cmd = 'imageMath.py -e="round(((a_1+(2.0*PI*g)) - (b_1) - (2.0*{0}/3.0/{1})*c + (2.0*{0}/3.0/{1})*f )/2.0/PI)" --a={2} --b={3} --c={4} --f={5} --g={6}  -o {7} -t float32 -s BIL'.format(B, f0, highBandIgram, lowBandIgram, nonDispFile, dispFile, jumpsFile, dFile)

    print(cmd)

    os.system(cmd)
    #d = (phH - phL - (2.*B/3./f0)*ph_nondis + (2.*B/3./f0)*ph_iono )/2./PI
    #d = np.round(d)

    #cmd = 'imageMath.py -e="round(((a_1 - {6}) + (b_1-{7}) - 2.0*c - 2.0*f )/4.0/PI - g/2)" --a={0} --b={1} --c={2} --f={3} --g={4} -o {5} -t float32 -s BIL'.format(lowBandIgram, highBandIgram, nonDispFile, dispFile, dFile, mFile, refL, refH)

    cmd = 'imageMath.py -e="round(((a_1 ) + (b_1+(2.0*PI*k)) - 2.0*c - 2.0*f )/4.0/PI - g/2)" --a={0} --b={1} --c={2} --f={3} --g={4} --k={5} -o {6} -t float32 -s BIL'.format(lowBandIgram, highBandIgram, nonDispFile, dispFile, dFile, jumpsFile, mFile)

    print(cmd)

    os.system(cmd)


    #m = (phL + phH - 2*ph_nondis - 2*ph_iono)/4./PI - d/2.
    #m = np.round(m)

    return mFile , dFile

def getBandFrequencies(inps):

    with shelve.open(inps.lowBandShelve, flag='r') as db:
          frameL = db['frame']
          wvl0 = frameL.radarWavelegth
          wvlL = frameL.subBandRadarWavelength

    with shelve.open(inps.highBandShelve, flag='r') as db:
       frameH = db['frame']
       wvlH = frameH.subBandRadarWavelength

       pulseLength = frameH.instrument.pulseLength
       chirpSlope = frameH.instrument.chirpSlope
       # Total Bandwidth
       B = np.abs(chirpSlope)*pulseLength

    return wvl0, wvlL, wvlH, B


def main(iargs=None):


    inps = cmdLineParse(iargs)

    '''
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

    '''

    ##########

    # construct the unwrap and  unwrap connected component filenames for both high and low band interferogams
    # allow for different connected component files for the low and high band images depending what the user preferred
    #       for snaphu2stage: use snaphu connected component
    #       for snaphu: use snaphu connected component
    #       for icu: use icu connected component
    # lowband file
    if inps.lowBandIgramUnwMethod == 'snaphu' or inps.lowBandIgramUnwMethod == 'snaphu2stage':
        lowBandconncomp = inps.lowBandIgramPrefix + '_snaphu.unw.conncomp'
    elif inps.lowBandIgramUnwMethod == 'icu':
        lowBandconncomp = inps.lowBandIgramPrefix + '_icu.unw.conncomp'
    inps.lowBandconncomp = lowBandconncomp
    inps.lowBandIgram = inps.lowBandIgramPrefix + '_' + inps.lowBandIgramUnwMethod + '.unw'
    # highband file
    if inps.highBandIgramUnwMethod == 'snaphu' or inps.highBandIgramUnwMethod == 'snaphu2stage':
        highBandconncomp = inps.highBandIgramPrefix + '_snaphu.unw.conncomp'
    elif inps.highBandIgramUnwMethod == 'icu':
        highBandconncomp = inps.highBandIgramPrefix + '_icu.unw.conncomp'
    inps.highBandconncomp = highBandconncomp
    inps.highBandIgram = inps.highBandIgramPrefix + '_' + inps.highBandIgramUnwMethod + '.unw'
    # print a summary for the user
    print('Files to be used for estimating ionosphere:')
    print('**Low band files:')
    print(inps.lowBandIgram)
    print(inps.lowBandconncomp)
    print('**High band files:')
    print(inps.highBandIgram)
    print(inps.highBandconncomp)

    # generate the output directory if it does not exist yet, and back-up the shelve files
    if not os.path.exists(inps.outDir):
       os.makedirs(inps.outDir)
    lowBandShelve = os.path.join(inps.outDir, 'lowBandShelve')
    highBandShelve = os.path.join(inps.outDir, 'highBandShelve')
    if not os.path.exists(lowBandShelve):
       os.makedirs(lowBandShelve)
    if not os.path.exists(highBandShelve):
       os.makedirs(highBandShelve)
    cmdCp = 'cp ' + inps.lowBandShelve + '* ' + lowBandShelve
    os.system(cmdCp)
    cmdCp = 'cp ' + inps.highBandShelve + '* ' + highBandShelve
    os.system(cmdCp)
    inps.lowBandShelve = os.path.join(lowBandShelve, 'data')
    inps.highBandShelve = os.path.join(highBandShelve, 'data')

    
 
    '''
    outputDir = self.insar.ionosphereDirname
    if os.path.isdir(outputDir):
        logger.info('Ionosphere directory {0} already exists.'.format(outputDir))
    else:
        os.makedirs(outputDir)
    '''

    outDispersive = os.path.join(inps.outDir, 'iono.bil')
    sigmaDispersive = outDispersive + ".sig"

    outNonDispersive = os.path.join(inps.outDir, 'nonDispersive.bil') 
    sigmaNonDispersive = outNonDispersive + ".sig"

    inps.Sig_phi_L = os.path.join(inps.outDir, 'lowBand.Sigma')
    inps.Sig_phi_H = os.path.join(inps.outDir, 'highBand.Sigma')

    maskFile = os.path.join(inps.outDir, "mask.bil")

    #masterFrame = self._insar.loadProduct( self._insar.masterSlcCropProduct)
    wvl, wvlL, wvlH, B = getBandFrequencies(inps)
    
    f0 = SPEED_OF_LIGHT/wvl
    fL = SPEED_OF_LIGHT/wvlL
    fH = SPEED_OF_LIGHT/wvlH

    ###Determine looks
    #azLooks, rgLooks = self.insar.numberOfLooks( masterFrame, self.posting,
    #                                    self.numberAzimuthLooks, self.numberRangeLooks)


    #########################################################
    # make sure the low-band and high-band interferograms have consistent unwrapping errors. 
    # For this we estimate jumps as the difference of lowBand and highBand phases divided by 2PI
    # The assumprion is that bothe interferograms are flattened and the phase difference between them
    # is less than 2PI. This assumprion is valid for current sensors. It needs to be evaluated for
    # future sensors like NISAR.
    jumpsFile = check_consistency(inps.lowBandIgram, inps.highBandIgram, inps.outDir)

    #########################################################
    # estimating the dispersive and non-dispersive components
    dispersive_nonDispersive(inps.lowBandIgram, inps.highBandIgram, f0, fL, fH, outDispersive, outNonDispersive, jumpsFile)

    # generating a mask which will help filtering the estimated dispersive and non-dispersive phase
    getMask(inps, maskFile)
    # Calculating the theoretical standard deviation of the estimation based on the coherence of the interferograms
    theoretical_variance_fromSubBands(inps, f0, fL, fH, B, sigmaDispersive, sigmaNonDispersive, inps.azLooks * inps.rngLooks) 

    # low pass filtering the dispersive phase
    lowPassFilter(outDispersive, sigmaDispersive, maskFile, 
                    inps.kernel_x_size, inps.kernel_y_size, 
                    inps.kernel_sigma_x, inps.kernel_sigma_y, 
                    iteration = inps.dispersive_filter_iterations, 
                    theta = inps.kernel_rotation)


    # low pass filtering the  non-dispersive phase
    lowPassFilter(outNonDispersive, sigmaNonDispersive, maskFile, 
                    inps.kernel_x_size, inps.kernel_y_size,
                    inps.kernel_sigma_x, inps.kernel_sigma_y,
                    iteration = inps.dispersive_filter_iterations,
                    theta = inps.kernel_rotation)
            
            
    # Estimating phase unwrapping errors
    mFile , dFile = unwrapp_error_correction(f0, B, outDispersive+".filt", outNonDispersive+".filt", 
                                                    inps.lowBandIgram, inps.highBandIgram, jumpsFile)

    # re-estimate the dispersive and non-dispersive phase components by taking into account the unwrapping errors
    outDispersive = outDispersive + ".unwCor"
    outNonDispersive = outNonDispersive + ".unwCor"
    dispersive_nonDispersive(inps.lowBandIgram, inps.highBandIgram, f0, fL, fH, outDispersive, outNonDispersive, jumpsFile, m=mFile , d=dFile)

    # low pass filtering the new estimations 
    lowPassFilter(outDispersive, sigmaDispersive, maskFile, 
                    inps.kernel_x_size, inps.kernel_y_size,
                    inps.kernel_sigma_x, inps.kernel_sigma_y,
                    iteration = inps.dispersive_filter_iterations,
                    theta = inps.kernel_rotation)

    lowPassFilter(outNonDispersive, sigmaNonDispersive, maskFile,
                    inps.kernel_x_size, inps.kernel_y_size,
                    inps.kernel_sigma_x, inps.kernel_sigma_y,
                    iteration = inps.dispersive_filter_iterations,
                    theta = inps.kernel_rotation)


if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()

