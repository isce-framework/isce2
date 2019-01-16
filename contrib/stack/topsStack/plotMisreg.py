#!/usr/bin/env python3
########################
import os, imp, sys, glob
import argparse
import configparser
import  datetime
import time
import numpy as np
import matplotlib.pyplot as plt

#################################################################

def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser( description='plots the misregistration time-series')
    parser.add_argument('-i', '--input', type=str, dest='input', required=True,
            help='Directory with the overlap directories that has calculated misregistration for each pair')

    return parser

def cmdLineParse(iargs = None):
    '''
    Command line parser.
    '''

    parser = createParser()
    inps = parser.parse_args(args=iargs)

    return inps

def readMisreg(misregDates):
    dateList = []#np.zeros(len(misregDates))
    
    misreg = np.zeros(len(misregDates))
    for i in range(len(misregDates)):
        misregFile = misregDates[i]
        d = os.path.basename(misregFile).replace('.txt','')    
        #dateList.append(d)
        dd = datetime.datetime(*time.strptime(d, "%Y%m%d")[0:5])
        dateList.append(dd)
        m = np.loadtxt(misregFile)
        misreg[i] = m
    return dateList, misreg
#####################################

def main(iargs=None):

    inps = cmdLineParse(iargs)

    misregDates = glob.glob(os.path.join(inps.input,'*.txt'))
    
    dateList, misreg = readMisreg(misregDates)
    print(dateList)
    print(misreg)
    plt.plot(dateList, misreg, '*', ms=4)
    plt.show()
if __name__ == '__main__' :
  ''' 
  invert a network of the pair's mis-registrations to
  estimate the mis-registrations wrt the Master date.
  '''

  main()

