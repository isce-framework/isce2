import sys,os,glob,time
import logging,pickle
import isce,isceobj
import numpy as np
import matplotlib.pyplot as plt

from imageMath import IML
from isceobj.XmlUtil.XmlUtil import XmlUtil
from scipy import stats
from matplotlib import cm

def genRawImg(rawFile,iBias,qBias,rangeStart):
    raw = IML.mmapFromISCE(rawFile,logging)
    cj = np.complex64(1j)
    rawImg = []
    for i in range(len(raw.bands[0])):
        line = raw.bands[0][i,rangeStart::2] + cj*raw.bands[0][i,rangeStart+1::2] - iBias - cj*qBias
        rawImg.append(line)
    rawImg = np.array(rawImg)

    rfat = []
    for i in range(len(rawImg)):
        line = np.fft.fftshift(np.fft.fft(rawImg[i])/len(rawImg[i]))
        rfat.append(line)
    rfat = np.array(rfat)

    return rfat

def processTsnbBlock(img,chop):

    width = len(img[0])
    length = len(img)
    img[:,:int(np.ceil(chop))] = 0. # Zero out around bandwidth
    img[:,int(np.floor(width-chop)):] = 0.
    block = img[:256] # 256 azimuth line block
    mask = np.zeros((256,width)) # TSNB hit mask

    print('Creating raw image and setting params...')
    maskImg = isceobj.createImage()
    maskImg.bands = 1
    maskImg.scheme = 'BIL'
    maskImg.dataType = 'FLOAT'
    maskImg.setWidth(width)
    print('Width set at:',maskImg.width)
    maskImg.setLength(length)
    print('Length set at:',maskImg.length)
    maskImg.setFilename('tsnbMaskImg.bil')
    print('Image name: tsnbMaskImg.bil')
    print('\nStarting image filter processing...')
    with open('tsnbMaskImg.bil','wb') as fid:
        for i in range(length-255):
            if np.mod(i,200) == 0.0:
                print('Progress:',round(((100.*i)/(length-255)),2),'%        ',end='\r')
            zeros = np.zeros(width) # Set up zero-padding
            avg = np.mean(block,axis=0) # Average block
            avg[int(width/2)] = 0. # ignore DC
            bandAvg = avg[int(np.ceil(chop)):int(np.floor(width-chop))] # Pull values only in bandwidth
            zscores = stats.zscore(bandAvg) # Convert block to Z-Scores
            pvalues = stats.norm.sf(np.abs(zscores)) # Convert Z-Scores to one-tailed P-Values
            rfiMask = (pvalues < .005).astype(int) # Isolate significant values (p < .005) and convert to ints
            zeros[int(np.ceil(chop)):int(np.floor(width-chop))] += rfiMask # Pad zeros back into rfiMask line
            mask += zeros.astype(int) # Add rfiMask hits to overall mask

            count = int(np.mod(i,256)) # Calculate place in arrays
            mask[count,int(width/2)] = 0. # Dont mask DC
            mask[count].astype(np.float32).tofile(fid) # Write mask line out to image file
            mask[count] = 0. # Zero out mask line
            if i != (length-256):
                block[count] = img[256+i] # Replace "oldest" line with "next" line
        for i in range(length-255,length): # Write out the rest of the mask to the file
            count = int(np.mod(i,256))
            mask[count].astype(np.float32).tofile(fid)
    print('Image filter successfully written.               \n')
    maskImg.renderHdr()

def processTvwbBlock(img,chop):
  
    width = len(img) # Switched because image will be transposed in a few lines
    length = len(img[0])
    img[:,:int(np.ceil(chop))] = 0. # Zero out around bandwidth
    img[:,int(np.floor(length-chop)):] = 0.
    img = np.transpose(img) # Transposing to match tsnbBlock code
    block = img[:100] # 100 range column block
    mask = np.zeros((100,width)) # TVWB hit mask
    rng = np.arange(width)

    print('Creating raw image and setting params...')
    maskImg = isceobj.createImage()
    maskImg.bands = 1
    maskImg.scheme = 'BIL'
    maskImg.dataType = 'FLOAT'
    maskImg.setWidth(width)
    print('Width set at:',maskImg.width)
    maskImg.setLength(length)
    print('Length set at:',maskImg.length)
    maskImg.setFilename('tvwbMaskImg.bil')
    print('Image name: tvwbMaskImg.bil')
    print('\nStarting image filter processing...')
    with open('tvwbMaskImg.bil','wb') as fid:
        for i in range(length-99):
            if np.mod(i,7) == 0.0:
                print('Progress:',round(((100.*i)/(length-99)),2),'%        ',end='\r')
            avg = np.mean(block,axis=0)
            if np.any(avg): # Possible that 100 columns could be 0s (outside bandwidth)
                avg = np.log10(avg) # Convert to log-space
                fit = np.polyfit(rng,avg,2) # Generate best-fit quadratic
                pfit = np.poly1d(fit) # Generate numpy function with best-fit quadratic coeffs
                avg -= pfit(rng) # Detrend the average with the best-fit (normalize the row's mean)
                zscores = stats.zscore(avg)
                pvalues = stats.norm.sf(np.abs(zscores))
                rfiMask = pvalues < .005
                rfiMask = rfiMask.astype(int)
                mask += rfiMask

            count = int(np.mod(i,100))
            mask[count].astype(np.float32).tofile(fid)
            mask[count] = 0.
            if i != (length-100):
                block[count] = img[100+i]
        for i in range(length-99,length):
            count = int(np.mod(i,100))
            mask[count].astype(np.float32).tofile(fid)
    print('Image filter successfully written.                \n')
    maskImg.renderHdr()
    
def genFinalMask(mName,width):
    print('\nReading and combining masks...')
    with open('tsnbMaskImg.bil','rb') as fid:
        arr = np.fromfile(fid,dtype='float32').reshape(-1,width)
    tsnbHist = plt.hist(arr.flatten(),bins=256)[0][1:] # histogram of values 1-256 in mask
    plt.close()
    tVals = sum(tsnbHist)
    for i in range(1,256):
        if ((sum(tsnbHist[255-i:])/tVals) > 0.8): # Looking to eliminate first sigma of values (just a guess)
            TSNBthresh = 254-i # set threshold
            break
    print('TSNB threshold cutoff set as:',TSNBthresh)
    arr2 = (arr > TSNBthresh).astype(int)

    with open('tvwbMaskImg.bil','rb') as fid:
        tarr = np.transpose(np.fromfile(fid,dtype='float32').reshape(-1,len(arr)))
    tvwbHist = plt.hist(tarr.flatten(),bins=100)[0][1:] # histogram of values 1-100 in mask
    plt.close()
    tVals = sum(tvwbHist)
    for i in range(1,100):
        if ((sum(tvwbHist[99-i:])/tVals) > 0.8):
            TVWBthresh = 98-i
            break
    print('TVWB threshold cutoff set as:',TVWBthresh)
    tarr2 = (tarr > TVWBthresh).astype(int)

    fArr = arr2 | tarr2 # Combine masks

    print('\nPrinting combined and separate masks to',mName,'...')
    # Mask channels as follows:
    # CH 1: Final mask used (combined and thresholded TSNB/TVWB masks)
    # CH 2: TSNB mask pre-threshold
    # CH 3: TSNB mask thresholded
    # CH 4: TVWB mask pre-threshold
    # CH 5: TVWB mask thresholded
    fMaskImg = isceobj.createImage()
    fMaskImg.bands=5
    fMaskImg.scheme='BIL'
    fMaskImg.dataType='FLOAT'
    fMaskImg.setWidth(len(fArr[0]))
    fMaskImg.setLength(len(fArr))
    fMaskImg.setFilename(mName)
    with open(mName,'wb') as fid:
        for i in range(len(fArr)):
            fArr[i].astype(np.float32).tofile(fid) # CH 1
            arr[i].astype(np.float32).tofile(fid) # CH 2
            arr2[i].astype(np.float32).tofile(fid) # CH 3
            tarr[i].astype(np.float32).tofile(fid) # CH 4
            tarr2[i].astype(np.float32).tofile(fid) # CH 5
    fMaskImg.renderHdr()
    print('Finished.')

    # finalRFImasks.bil will contain all masks, so no need for these anymore...
    os.remove('tsnbMaskImg.bil')
    os.remove('tsnbMaskImg.bil.xml')
    os.remove('tvwbMaskImg.bil')
    os.remove('tvwbMaskImg.bil.xml')

    return fArr,np.sum(arr2),np.sum(tarr2)

def runMain(insar,frame):
    tStart = time.time()
    if frame == 'reference':
        rawFrame = insar.referenceFrame
    else:
        rawFrame = insar.secondaryFrame
    rawName = rawFrame.image.filename
    print('\nInput raw image:',rawName)
    # Processed raw image will simply be xxx.raw -> xxx.processed.raw for simplicity
    processedName = rawName.split('.')
    processedName.append(processedName[-1])
    processedName[-2] = 'processed'
    processedName = '.'.join(processedName)
    print('Output raw image name:',processedName)
    maskName = '.'.join(rawName.split('.')[:-1]) + '_RFImasks.bil'
    print('Final masks name:',maskName)
    
    print('\nGenerating raw image:')
    iBias = insar.instrument.inPhaseValue
    qBias = insar.instrument.quadratureValue
    rngStart = rawFrame.image.xmin
    print('iBias:',iBias)
    print('qBias:',qBias)
    print('Image x-min:',rngStart)
    rawImg = genRawImg(rawName,iBias,qBias,rngStart) # Raw image does not have mag-squared transformation
    rawTransImg = np.abs(rawImg)**2 # Transformed image has line-by-line mag-squared xfrm
    
    imWidth = int((rawFrame.image.width - rngStart)/2)
    slope = np.abs(insar.instrument.chirpSlope)
    pulseDir = insar.instrument.pulseLength
    bwidth = slope*pulseDir
    kwidth = (bwidth*imWidth)/insar.instrument.rangeSamplingRate
    chop = (imWidth-kwidth)/2
    print('Processing TSNB components')
    processTsnbBlock(rawTransImg,chop)
    print('Processing TVWB components')
    processTvwbBlock(rawTransImg,chop)
    del rawTransImg # Save space!

    print('Finished generating masks. Combining and processing masks...')
    finalMask,tsnbTot,tvwbTot = genFinalMask(maskName,imWidth)

    # IU.copyattributes() ?
    print('Writing new filtered raw image')
    img = isceobj.createRawImage()
    img.setFilename(processedName)
    img.setXmin(rngStart)
    img.setWidth(rngStart+(2*len(rawImg[0])))
    img.setLength(len(rawImg))
    lcount = 0
    psum = 0
    header = np.zeros(rngStart,dtype=np.uint8)
    outline = np.zeros(2*len(rawImg[0]))
    with open(processedName,'wb') as fid:
        for i in range(len(rawImg)):
            notchLine = (finalMask[i]==0).astype(float) # Aggressive masking (all RFI signals removed completely)
            if np.any(notchLine==0):
                lcount += 1
                psum += (len(notchLine) - sum(notchLine))
            notchLine[0] = 0. # Mask DC value to 0
            ln = rawImg[i] * notchLine # Mask the matching line of image
            header.tofile(fid)
            line = np.fft.ifft(np.fft.ifftshift(ln))*len(ln) # Inverse shift/FFT line and restore magnitude before writing
            outline[0::2] = line.real + iBias
            outline[1::2] = line.imag + qBias
            outline.astype(np.uint8).tofile(fid)
    img.renderHdr()

    tEnd = time.time()
    tstring = str(int((tEnd-tStart)/60))+'m '+str(round((tEnd-tStart)%60,2))+'s'
    aLines = round(100.*lcount/len(rawImg),2)
    fSize = len(rawImg)*len(rawImg[0])
    aPix = round(100.*psum/fSize,2)
    mTSNB = round(100.*(100.*tsnbTot/fSize)/aPix,2)
    mTVWB = round(100.*(100.*tvwbTot/fSize)/aPix,2)

    print('\nTotal run-time:',tstring)
    print('Affected lines in image (%):',aLines)
    print('Affected pixels in image (%):',aPix)
    print('Amount of TSNB RFI in image (%):',mTSNB)
    print('Amount of TVWB RFI in image (%):',mTVWB,'\n')

    return processedName

def RFImask(self):
    print()
    #with open('PICKLE/preprocess','rb') as fid:
    #    insar = pickle.load(fid)
    
    print('Processing',self.insar.referenceFrame.image.filename,':')
    mName = runMain(self.insar,'reference')
    print('\nProcessing',self.insar.secondaryFrame.image.filename,':')
    sName = runMain(self.insar,'secondary')
    if os.path.exists('isce.log'):
        os.remove('isce.log')
    return mName,sName

if __name__ == "__main__":
    RFImask()

