#!/usr/bin/env python3
########################
#Author: Heresh Fattahi
#Copyright 2016
######################
import os, imp, sys, glob
import argparse
import configparser
import  datetime
import time
import numpy as np
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


def date_list(overlapDirs):
  dateList = []
  tbase = []
  for di in  overlapDirs:
    di = di.replace('.txt','')
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
def extract_offset(file):
  
  misreg_dict = {}
  for line in open(file):
    c = line.split(":")
    if len(c) < 2 or line.startswith('%') or line.startswith('#'):
      next #ignore commented lines or those without variables
    else:
      misreg_dict[c[0].strip()] = str.replace(c[1],'\n','').strip() 
    return misreg_dict
######################################
def design_matrix(overlapDirs):
  '''Make the design matrix for the inversion.  '''
  tbase,dateList,dateDict = date_list(overlapDirs)
  numDates = len(dateDict)
  numIfgrams = len(overlapDirs)
  A = np.zeros((numIfgrams,numDates))
  B = np.zeros(np.shape(A))
  L = np.zeros((numIfgrams,1))
  daysList = []
  for day in tbase:
    daysList.append(day)
  tbase = np.array(tbase)
  t = np.zeros((numIfgrams,2))
  for ni in range(len(overlapDirs)):
    date12 = os.path.basename(overlapDirs[ni]).replace('.txt','')
    date = date12.split('_')
    ndxt1 = daysList.index(dateDict[date[0]])
    ndxt2 = daysList.index(dateDict[date[1]])
    A[ni,ndxt1] = -1
    A[ni,ndxt2] = 1
    B[ni,ndxt1:ndxt2] = tbase[ndxt1+1:ndxt2+1]-tbase[ndxt1:ndxt2]
    t[ni,:] = [dateDict[date[0]],dateDict[date[1]]]

  #  misreg_dict = extract_offset(os.path.join(overlapDirs[ni],misregName))
    misreg_dict = extract_offset(overlapDirs[ni])
    L[ni] = float(misreg_dict['median'])
    if (np.isnan(L[ni])):
        L[ni] = 0.0

  A = A[:,1:]
  B = B[:,:-1]
  
  ind=~np.isnan(L)
  return A[ind[:,0],:],B[ind[:,0],:],L[ind]
 
######################################
def main(iargs=None):

  inps = cmdLineParse(iargs)
  if not os.path.exists(inps.output):
      os.makedirs(inps.output)

  overlapPairs = glob.glob(os.path.join(inps.input,'*/*.txt'))

  tbase,dateList,dateDict = date_list(overlapPairs)
  A,B,L = design_matrix(overlapPairs)
#  A,B,L = design_matrix(overlapDirs,inps.misregFileName)
  B1 = np.linalg.pinv(B)
  B1 = np.array(B1,np.float32)

  dS = np.dot(B1,L)
  dtbase = np.diff(tbase)
  dt = np.zeros((len(dtbase),1))
 # dt[:,0]=dtbase
  zero = np.array([0.],np.float32)
 # S = np.concatenate((zero,np.cumsum([dS*dt])))
  S = np.concatenate((zero,np.cumsum([dS*dtbase])))
  
  residual = L-np.dot(B,dS)
 # print (L)
 # print (np.dot(B,dS))
  RMSE = np.sqrt(np.sum(residual**2)/len(residual))
  print('')
  print('Rank of design matrix: ' + str(np.linalg.matrix_rank(B)))
  if np.linalg.matrix_rank(B)==len(dateList)-1:
     print('Design matrix is full rank.')
  else:
     print('Design matrix is rank deficient. Network is disconnected.')
     print('Using a fully connected network is recommended.')
  print('RMSE : '+str(RMSE)+' pixels')
  print('')
  print('Estimated offsets with respect to the stack master date')    
  print('')

  offset_dict={}
  for i in range(len(dateList)):
     print (dateList[i]+' : '+str(S[i]))
     offset_dict[dateList[i]]=S[i]
     with open(os.path.join(inps.output,dateList[i]+'.txt'), 'w') as f:
        f.write(str(S[i]))
  print('')  
 
if __name__ == '__main__' :
  ''' 
  invert a network of the pair's mis-registrations to
  estimate the mis-registrations wrt the Master date.
  '''

  main()
