#!/usr/bin/env python3

# Author: Heresh Fattahi

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

#################################################################
def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser( description='extracts the overlap geometry between master bursts')
    parser.add_argument('-i', '--input', type=str, dest='input', required=True,
            help='Directory with the overlap directories that has calculated misregistration for each pair')
    parser.add_argument('-o', '--output', type=str, dest='output', required=True,
            help='output directory to save misregistration for each date with respect to the stack Master date')
#    parser.add_argument('-f', '--misregFileName', type=str, dest='misregFileName', default='misreg.txt',
#            help='misreg file name that contains the calculated misregistration for a pair')

    return parser

def cmdLineParse(iargs = None):
    '''
    Command line parser.
    '''

    parser = createParser()
    inps = parser.parse_args(args=iargs)

    return inps


def date_list(pairDirs):
  dateList = []
  tbase = []
  for di in  pairDirs:
    #di = di.replace('.txt','')
    dates = os.path.basename(di).split('_')
    dates1 = os.path.basename(di).split('_')
    if not dates[0] in dateList: dateList.append(dates[0])
    if not dates[1] in dateList: dateList.append(dates[1])
   
  dateList.sort()
  d1 = datetime.datetime(*time.strptime(dateList[0],"%Y%m%d")[0:5])
  for ni in range(len(dateList)):
    d2 = datetime.datetime(*time.strptime(dateList[ni],"%Y%m%d")[0:5])
    diff = d2-d1
    tbase.append(diff.days)
  dateDict = {}
  for i in range(len(dateList)): dateDict[dateList[i]] = tbase[i]
  return tbase,dateList,dateDict

#####################################
def extract_offset(filename):
  print(filename)
  with shelve.open(os.path.join(filename,'misreg'),flag='r') as db:
       print(dir(db))
       azpoly = db['azpoly']
       rgpoly = db['rgpoly']

  azCoefs = np.array(azpoly.getCoeffs())
  rgCoefs = np.array(rgpoly.getCoeffs())

  return azCoefs.flatten(0), rgCoefs.flatten(0)

def getPolyInfo(filename):
  with shelve.open(os.path.join(filename,'misreg'),flag='r') as db:
       azpoly = db['azpoly']
       rgpoly = db['rgpoly']  
  azCoefs = azpoly.getCoeffs()
  rgCoefs = rgpoly.getCoeffs()
  info = {}
  info['sizeOfAzCoefs'] = np.size(azCoefs)
  info['sizeOfRgCoefs'] = np.size(rgCoefs)
  info['shapeOfAzCoefs'] = np.shape(azCoefs)
  info['shapeOfRgCoefs'] = np.shape(rgCoefs)
  info['azazOrder'] = azpoly.getAzimuthOrder()
  info['azrgOrder'] = azpoly.getRangeOrder()
  info['rgazOrder'] = rgpoly.getAzimuthOrder()
  info['rgrgOrder'] = rgpoly.getRangeOrder()

  return info
  #return np.size(azCoefs), np.size(rgCoefs), np.shape(azCoefs), np.shape(rgCoefs)

######################################
def design_matrix(pairDirs):
  '''Make the design matrix for the inversion.  '''
  tbase,dateList,dateDict = date_list(pairDirs)
  numDates = len(dateDict)
  numIfgrams = len(pairDirs)
  A = np.zeros((numIfgrams,numDates))
  B = np.zeros(np.shape(A))

  # numAzCoefs, numRgCoefs, azCoefsShape, rgCoefsShape = getPolyInfo(pairDirs[0])
  polyInfo = getPolyInfo(pairDirs[0])
  Laz = np.zeros((numIfgrams, polyInfo['sizeOfAzCoefs']))
  Lrg = np.zeros((numIfgrams, polyInfo['sizeOfRgCoefs']))
  daysList = []
  for day in tbase:
    daysList.append(day)
  tbase = np.array(tbase)
  t = np.zeros((numIfgrams,2))
  for ni in range(len(pairDirs)):
    date12 = os.path.basename(pairDirs[ni]).replace('.txt','')
    date = date12.split('_')
    ndxt1 = daysList.index(dateDict[date[0]])
    ndxt2 = daysList.index(dateDict[date[1]])
    A[ni,ndxt1] = -1
    A[ni,ndxt2] = 1
    B[ni,ndxt1:ndxt2] = tbase[ndxt1+1:ndxt2+1]-tbase[ndxt1:ndxt2]
    t[ni,:] = [dateDict[date[0]],dateDict[date[1]]]

  #  misreg_dict = extract_offset(os.path.join(overlapDirs[ni],misregName))
    azOff, rgOff = extract_offset(pairDirs[ni])
    Laz[ni,:] = azOff[:]
    Lrg[ni,:] = rgOff[:]

  A = A[:,1:]
  B = B[:,:-1]
  
 # ind=~np.isnan(Laz)
 # return A[ind[:,0],:],B[ind[:,0],:],Laz[ind,:], Lrg[ind]
  return A, B, Laz, Lrg
 
######################################
def main(iargs=None):

  inps = cmdLineParse(iargs)
  if not os.path.exists(inps.output):
      os.makedirs(inps.output)

  pairDirs = glob.glob(os.path.join(inps.input,'*'))
  polyInfo = getPolyInfo(pairDirs[0])

  tbase, dateList, dateDict = date_list(pairDirs)

  A, B, Laz, Lrg = design_matrix(pairDirs)
  A1 = np.linalg.pinv(A)
  A1 = np.array(A1,np.float32)

  zero = np.array([0.],np.float32)
  Saz = np.dot(A1, Laz)

  Saz = np.dot(A1, Laz)
  Srg = np.dot(A1, Lrg)

  residual_az = Laz-np.dot(A,Saz)
  residual_rg = Lrg-np.dot(A,Srg)
  RMSE_az = np.sqrt(np.sum(residual_az**2)/len(residual_az))
  RMSE_rg = np.sqrt(np.sum(residual_rg**2)/len(residual_rg))

  Saz = np.vstack((np.zeros((1,Saz.shape[1]), dtype=np.float32), Saz))
  Srg = np.vstack((np.zeros((1,Srg.shape[1]), dtype=np.float32), Srg))

  print('')
  print('Rank of design matrix: ' + str(np.linalg.matrix_rank(A)))
  if np.linalg.matrix_rank(A)==len(dateList)-1:
     print('Design matrix is full rank.')
  else:
     print('Design matrix is rank deficient. Network is disconnected.')
     print('Using a fully connected network is recommended.')
  print('RMSE in azimuth : '+str(RMSE_az)+' pixels')
  print('RMSE in range : '+str(RMSE_rg)+' pixels')
  print('')
  print('Estimated offsets with respect to the stack master date')    
  print('')
  offset_dict={}
  for i in range(len(dateList)):
     print (dateList[i])
     offset_dict[dateList[i]]=Saz[i]
     azpoly = Poly2D()
     rgpoly = Poly2D()
     azCoefs = np.reshape(Saz[i,:],polyInfo['shapeOfAzCoefs']).tolist()
     rgCoefs = np.reshape(Srg[i,:],polyInfo['shapeOfRgCoefs']).tolist()
     azpoly.initPoly(rangeOrder=polyInfo['azrgOrder'], azimuthOrder=polyInfo['azazOrder'], coeffs=azCoefs)
     rgpoly.initPoly(rangeOrder=polyInfo['rgrgOrder'], azimuthOrder=polyInfo['rgazOrder'], coeffs=rgCoefs)

     if not os.path.exists(os.path.join(inps.output,dateList[i])):
         os.makedirs(os.path.join(inps.output,dateList[i]))

     odb = shelve.open(os.path.join(inps.output,dateList[i]+'/misreg'))
     odb['azpoly'] = azpoly
     odb['rgpoly'] = rgpoly
     odb.close()
 
     #with open(os.path.join(inps.output,dateList[i]+'.txt'), 'w') as f:
     #   f.write(str(Saz[i]))

  print('')  
 
if __name__ == '__main__' :
  ''' 
  invert a network of the pair's mis-registrations to
  estimate the mis-registrations wrt the Master date.
  '''

  main()
