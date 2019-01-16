#!/usr/bin/env python3

#Author: Heresh Fattahi

import os, imp, sys, glob
import argparse
import configparser
import  datetime
import time
import numpy as np
import shelve
import isce
import isceobj
from isceobj.Util.Poly2D import Poly2D
import h5py
from insarPair import insarPair
from insarStack import insarStack
import gdal


#################################################################
def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser( description='extracts the overlap geometry between master bursts')
    parser.add_argument('-i', '--input', type=str, dest='input', required=True,
            help='Directory with the pair directories that includes dense offsets for each pair')
    parser.add_argument('-o', '--output', type=str, dest='output', required=True,
            help='output directory to save dense-offsets for each date with respect to the stack Master date')

    return parser

def cmdLineParse(iargs = None):
    '''
    Command line parser.
    '''

    parser = createParser()
    inps = parser.parse_args(args=iargs)

    return inps


def write2h5(inps):
  # dumping all offset files to an h5 file
  dirs = glob.glob(os.path.join(inps.input,'*'))
  pairsDict = {}
  for dir in dirs:
    #Assuming the directory name for a pair is master_slave dates (eg: 20100506_20101112)
    d12 = os.path.basename(dir)
    #if os.path.exists(os.path.join(dir,d12+'.bil')):
    if os.path.exists(os.path.join(dir,'filtAzimuth.off')):
      obsDict = {'offset-azimuth':os.path.join(dir,'filtAzimuth.off')}
      #qualityDict = {'offset-snr':os.path.join(dir,d12+'_snr.bil')}
      dates = os.path.basename(dir).split('_')
      t1 = time.strptime(dates[0],'%Y%m%d')
      Time1 = datetime.datetime(t1.tm_year,t1.tm_mon,t1.tm_mday)

      t2 = time.strptime(dates[1],'%Y%m%d')
      Time2 = datetime.datetime(t2.tm_year,t2.tm_mon,t2.tm_mday)
      metadataDict = {'platform' : 'platform' , 'processor' : 'ISCE'  }
      pairObj = insarPair(dates=(Time1 , Time2) ,observation = obsDict, metadata=metadataDict)
      #pairObj = insarPair(dates=(Time1 , Time2) ,observation = obsDict, quality = qualityDict, metadata=metadataDict)
      pairObj.get_metadata('offset-azimuth')
      pairsDict[(Time1,Time2)] = pairObj

  ############################################
  stackObj = insarStack(pairsDict = pairsDict)
  stackObj.get_platform_tracks()
  outFile = os.path.join(inps.input,'offsets.h5')
  stackObj.save2h5(output = outFile) 
  return outFile

def date_list(h5file):
  h5=h5py.File(h5file,'r')
  ds = h5['/platform-track/observations'].get('pairs_idx') 
  pairs = ds[:,:]
  numPiars = pairs.shape[0]
  dateList = []
  tbase = []
  masters = [None]*numPiars
  slaves = [None]*numPiars
  for i in range(numPiars):
      master = pairs[i,0].decode("utf-8")
      slave = pairs[i,1].decode("utf-8")
      if master not in dateList: dateList.append(master)
      if slave not in dateList: dateList.append(slave)
      masters[i]=master
      slaves[i]=slave

  dateList.sort()
  d1 = datetime.datetime(*time.strptime(dateList[0],"%Y-%m-%d %H:%M:%S")[0:6])
  for ni in range(len(dateList)):
    d2 = datetime.datetime(*time.strptime(dateList[ni],"%Y-%m-%d %H:%M:%S")[0:6])
    diff = d2-d1
    tbase.append(diff.days)

  dateDict = {}
  for i in range(len(dateList)): dateDict[dateList[i]] = tbase[i]

  return tbase,dateList,dateDict, masters, slaves

#####################################

def design_matrix(h5File):
  tbase,dateList,dateDict, masters, slaves = date_list(h5File)
  numDates = len(dateDict)
  numPairs = len(masters)
  A = np.zeros((numPairs,numDates))
  B = np.zeros_like(A)
  tbase = np.array(tbase)
  for ni in range(numPairs):
     ndxt1 = dateList.index(masters[ni])
     ndxt2 = dateList.index(slaves[ni])
     A[ni,ndxt1] = -1
     A[ni,ndxt2] = 1
     B[ni,ndxt1:ndxt2] = tbase[ndxt1+1:ndxt2+1]-tbase[ndxt1:ndxt2]

  #print('A',A)
  #print('%%%%%%%%%%%%%%% %%%%%')  
  A = A[:,1:]
  B = B[:,:-1]

  return A, B  

def invert_wlq(inps,h5File):
    tbase,dateList,dateDict, masters, slaves = date_list(h5File)
    numPairs = len(masters)
    A,B = design_matrix(h5File)
   
    h5 = h5py.File(h5File,'r')
    data = h5['/platform-track/observations'].get('offset-azimuth')
    snr = h5['/platform-track/quality'].get('offset-snr')
    Nz, Ny, Nx = data.shape
    Npar = A.shape[1]
    A1 = np.linalg.pinv(A)
    A1 = np.array(A1,np.float32)

    ##########
    outName = os.path.join(inps.output,'timeseries.h5')
    h5out = h5py.File(outName,'w')
    ds = h5out.create_dataset('offsets',shape=(len(dateList),Ny,Nx),dtype=np.float32)
    dsq = h5out.create_dataset('quality',shape=(len(dateList),Ny,Nx),dtype=np.float32)

    I = np.eye(Nx)
    #Ak = np.kron(I,A)

    for j in range(Ny):
        print(j, 'out of ',Ny)
        L = data[:,j,:]
        Lsnr = snr[:,j,:]
        mask = np.prod(Lsnr,0)
        ind = mask>0.0
        NumValidPixels = np.sum(ind)
        if NumValidPixels>0:
           Lsnr = Lsnr[:,ind].flatten(1)
           L = L[:,ind].flatten(1)
           Lsnr = Lsnr/np.sum(Lsnr)
           W = np.diag(Lsnr)
           I = np.eye(NumValidPixels)
           Ak = np.kron(I,A)
           Cm = np.linalg.inv(np.dot(np.dot(Ak.T, W),Ak))
           B = np.dot(np.dot(np.linalg.inv(np.dot(np.dot(Ak.T, W),Ak)),Ak.T),W)
           ts = np.dot(B,L)
           Cm = np.sqrt(Cm[range(NumValidPixels*Npar),range(NumValidPixels*Npar)]).reshape([NumValidPixels,Npar]).T
        #Cm = np.vstack((np.zeros((1,ts.shape[1]), dtype=np.float32), ts))

           ts = ts.reshape([NumValidPixels,Npar]).T
        #ts = np.vstack((np.zeros((1,ts.shape[1]), dtype=np.float32), ts))
           ds[1:,j,ind] = ts
           dsq[1:,j,ind] = Cm

    dateListE = [d.encode("ascii", "ignore") for d in dateList]
    dateListE = np.array(dateListE)
    dsDateList = h5out.create_dataset('dateList', data=dateListE, dtype=dateListE.dtype)

    h5out.close()
    h5.close()

    return outName
    ##########

#def invert_old():
    '''    
    for j in range(Ny):
        print(j, 'out of ',Ny)
        L = np.empty((Nz*Nx,1))
        L[:,0] = data[:,j,:].flatten(1)[:]
        Lsnr = snr[:,j,:].flatten(1)
        Lsnr = Lsnr/np.sum(Lsnr)
        W = np.diag(Lsnr)
        #W = np.eye(Nz*Nx)
        B = np.dot(np.dot(np.linalg.inv(np.dot(np.dot(Ak.T, W),Ak)),Ak.T),W)
        ts = np.dot(B,L)
        ts = ts.reshape([Nx,Npar]).T
        ts = np.vstack((np.zeros((1,ts.shape[1]), dtype=np.float32), ts))
        ds[:,j,:] = ts
    '''
    ##########
    '''
    for j in range(Ny):
      print(j, 'out of ',Ny)
      for i in range(Nx):
        L = np.empty((Nz,1))
        L[:,0] = data[:,j,i]
        Lsnr = snr[:,j,i]
        Lsnr = Lsnr/np.sum(Lsnr)
        W = np.diag(Lsnr)
        #print('W',W)
        B = np.dot(np.dot(np.linalg.inv(np.dot(np.dot(A.T, W),A)),A.T),W)
        B = np.array(B,np.float32)
        ts = np.dot(B,L)
        ts = np.vstack((np.zeros((1,ts.shape[1]), dtype=np.float32), ts))
        ds[:,j,i] = ts[:,0]

    '''
def invert(inps,h5File):

    tbase,dateList,dateDict, masters, slaves = date_list(h5File)
    numPairs = len(masters)
    A,B = design_matrix(h5File)

    h5 = h5py.File(h5File,'r')
    data = h5['/platform-track/observations'].get('offset-azimuth')    
    Nz, Ny, Nx = data.shape
    Npar = A.shape[1]
    A1 = np.linalg.pinv(A)
    A1 = np.array(A1,np.float32)

    ##########
    outName = os.path.join(inps.output,'timeseries.h5')
    h5out = h5py.File(outName,'w')
    ds = h5out.create_dataset('offsets',shape=(len(dateList),Ny,Nx),dtype=np.float32)
    #dsq = h5out.create_dataset('quality',shape=(len(dateList),Ny,Nx),dtype=np.float32)
    h5tempCoh = h5py.File(os.path.join(inps.output,'temporal_coherence.h5'),'w')
    dst = h5tempCoh.create_dataset('temporal_coherence', shape=(Ny,Nx),dtype=np.float32)    

    for i in range(Ny):
        print(i, 'out of ',Ny)
        L = data[:,i,:]
        ts = np.dot(A1, L)
        L_residual = L - np.dot(A,ts)
        #dsr[:,i,:] =  L_residual
        dst[i,:] = np.absolute(np.sum(np.exp(1j*L_residual),0))/Nz

        ts = np.vstack((np.zeros((1,ts.shape[1]), dtype=np.float32), ts))
        ds[:,i,:] = ts

    dateListE = [d.encode("ascii", "ignore") for d in dateList]
    dateListE = np.array(dateListE)
    dsDateList = h5out.create_dataset('dateList', data=dateListE, dtype=dateListE.dtype)

    h5out.close()
    h5tempCoh.close()
    h5.close()

    return outName

def writeDateOffsets(inps, h5File):

    h5=h5py.File(h5File, 'r')
    ds = h5.get('offsets')
#    dsq = h5.get('quality')
    dsDates = h5.get('dateList')
    
    dateList = list(dsDates[:])
    print (dateList)
    for i in range(len(dateList)):
        print(dateList[i])
        d = dateList[i].decode("utf-8")
        d = datetime.datetime(*time.strptime(d,"%Y-%m-%d %H:%M:%S")[0:6]).strftime('%Y%m%d')
        outDir = os.path.join(inps.output, d)
        if not os.path.exists(outDir):
           os.makedirs(outDir)
        outName = os.path.join(outDir , d + '.bil')
        write(ds[i,:,:], outName, 1, 6)
 #       outName = os.path.join(outDir , d + '_snr.bil')
 #       write(dsq[i,:,:], outName, 1, 6)

def write(raster, fileName, nbands, bandType):

    ############
    # Create the file
    length, width = raster.shape
    driver = gdal.GetDriverByName( 'ENVI' )
    dst_ds = driver.Create(fileName, raster.shape[1], raster.shape[0], nbands, bandType )
    dst_ds.GetRasterBand(1).WriteArray( raster, 0 ,0 )

    dst_ds = None 

    img = isceobj.createImage()
    img.setFilename(fileName)
    img.setWidth(width)
    img.setLength(length)
    img.setAccessMode('READ')
    img.bands = 1
    img.dataType = 'FLOAT'
    img.scheme = 'BIP'
    img.renderHdr()
    img.renderVRT()


'''
    chunks = getChunks(Ny,Nx, 128, 128)

    print(chunks)

    start = time.time()
    for cnk in chunks:
       ss = np.zeros((128,128))
       for ii in range(numPairs):
         line = data[ii,cnk[0],cnk[1]]
         print(line.shape)
         print(np.mean(line))
         #ss += line
    end = time.time()
    print('3D chunked chunk-by-slice: ', end-start)
    h5.close() 
'''
def getChunks(Ny,Nx, chunk_y, chunk_x):
    # First determine the number of chunks in each dimension
    Ny_chunk = int(Ny // chunk_y)
    Nx_chunk = int(Nx // chunk_x)
    if Ny % chunk_y != 0:
        Ny_chunk += 1
    if Nx % chunk_x != 0:
        Nx_chunk += 1

    # Now construct chunk bounds
    chunks = []
    for i in range(Ny_chunk):
        if i == Ny_chunk - 1:
            nrows = Ny - chunk_y * i
        else:
            nrows = chunk_y
        istart = chunk_y * i
        iend = istart + nrows
        for j in range(Nx_chunk):
            if j == Nx_chunk - 1:
                ncols = Nx - chunk_x * j
            else:
                ncols = chunk_x
            jstart = chunk_x * j
            jend = jstart + ncols
            chunks.append([slice(istart,iend), slice(jstart,jend)])

    return chunks
    
def main(iargs=None):

  inps = cmdLineParse(iargs)
  if not os.path.exists(inps.output):
      os.makedirs(inps.output)

  h5File = write2h5(inps) 

  h5Timeseries = invert(inps, h5File)

  writeDateOffsets(inps, h5Timeseries)
 
if __name__ == '__main__' :
  ''' 
  invert a network of the pair's mis-registrations to
  estimate the mis-registrations wrt the Master date.
  '''

  main()
