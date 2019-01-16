#!/usr/bin/env python3

#Author: Heresh Fattahi

import isce
import isceobj
import numpy as np
import argparse
import os
from isceobj.Sensor.TOPS import createTOPSSwathSLCProduct
from mroipac.correlation.correlation import Correlation
import s1a_isce_utils as ut
import gdal
import glob

def createParser():
    parser = argparse.ArgumentParser( description='Extract valid overlap region for the stack')

    parser.add_argument('-m', '--master', dest='master', type=str, required=True,
            help='Directory with master acquisition')

    parser.add_argument('-s', '--slave', dest='slave', type=str, required=True,
            help='Directory with slave acquisition')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)

def updateValidRegion(topMaster, slavePath, swath):

    #slaveSwathList = ut.getSwathList(slave)
    #swathList = list(sorted(set(masterSwathList+slaveSwathList)))

    #for swath in swathList:
        #IWstr = 'IW{0}'.format(swath)
    ####Load relevant products
        #topMaster = ut.loadProduct(os.path.join(inps.master , 'IW{0}.xml'.format(swath)))

    topCoreg = ut.loadProduct(os.path.join(slavePath , 'IW{0}.xml'.format(swath)))

    topIfg = ut.coregSwathSLCProduct()
    topIfg.configure()


    minMaster = topMaster.bursts[0].burstNumber
    maxMaster = topMaster.bursts[-1].burstNumber

    minSlave = topCoreg.bursts[0].burstNumber
    maxSlave = topCoreg.bursts[-1].burstNumber

    minBurst = max(minSlave, minMaster)
    maxBurst = min(maxSlave, maxMaster)
    print ('minSlave,maxSlave',minSlave, maxSlave)
    print ('minMaster,maxMaster',minMaster, maxMaster)
    print ('minBurst, maxBurst: ', minBurst, maxBurst)

    for ii in range(minBurst, maxBurst + 1):

            ####Process the top bursts
        master = topMaster.bursts[ii-minMaster]
        slave  = topCoreg.bursts[ii-minSlave]
        ut.adjustCommonValidRegion(master,slave)
        #topMaster.bursts[ii-minMaster].firstValidLine = master.firstValidLine

    return topMaster


def main(iargs=None):
    '''extract common valid overlap region for the stack.
    '''
    inps=cmdLineParse(iargs)

    stackDir = os.path.join(os.path.dirname(inps.master),'stack')
    if not os.path.exists(stackDir):
        print('creating ', stackDir)
        os.makedirs(stackDir)
    else:
        print(stackDir , ' already exists.')
        print('Replacing master with existing stack.')
        inps.master = stackDir
        print('updating the valid overlap region of:')
        print(stackDir)

    masterSwathList = ut.getSwathList(inps.master)
    slaveList = glob.glob(os.path.join(inps.slave,'2*'))
    slaveSwathList = ut.getSwathList(slaveList[0]) # assuming all slaves have the same swaths
    swathList = list(sorted(set(masterSwathList+slaveSwathList)))

    for swath in swathList:
        print('******************')
        print('swath: ', swath)
        ####Load relevant products
        topMaster = ut.loadProduct(os.path.join(inps.master , 'IW{0}.xml'.format(swath)))
        #print('master.firstValidLine: ', topMaster.bursts[4].firstValidLine)
        for slave in slaveList:
            topMaster = updateValidRegion(topMaster, slave, swath)

        print('writing ', os.path.join(stackDir , 'IW{0}.xml'.format(swath)))
        ut.saveProduct(topMaster, os.path.join(stackDir , 'IW{0}.xml'.format(swath)))
        if not os.path.exists(os.path.join(stackDir ,'IW{0}'.format(swath))):
            os.makedirs(os.path.join(stackDir ,'IW{0}'.format(swath)))


if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()


#swathList = ut.getSwathList(master)
#swathList[2]
#frames = []
#for swath in swathList:
#        ifg = ut.loadProduct(os.path.join(inps.master , 'IW{0}.xml'.format(swath)))

#        if inps.isaligned:
#            reference = ifg.reference
#        else:
#            reference = ifg

#        minBurst = ifg.bursts[0].burstNumber
#        maxBurst = ifg.bursts[-1].burstNumber

#        if minBurst==maxBurst:
#            print('Skipping processing of swath {0}'.format(swath))
#            continue

#        frames.append(ifg)


#swaths = [Swath(x) for x in frame]

'''

slcPath = '/home/hfattahi/PROCESSDIR/MexicoCity_Test/TestStack_offsets/master'
swath = ut.loadProduct(os.path.join(slcPath , 'IW{0}.xml'.format(2)))

tref = swath.sensingStart
rref = swath.bursts[0].startingRange
dt = swath.bursts[0].azimuthTimeInterval 
dr = swath.bursts[0].rangePixelSize



print (slcPath)
for ind, burst in enumerate(swath.bursts):

    xoff = np.int(np.round( (burst.startingRange - rref)/dr))
    yoff = np.int(np.round( (burst.sensingStart - tref).total_seconds() / dt))
    tyoff = int(burst.firstValidLine)
    txoff = int(burst.firstValidSample)
    wysize = int(burst.numValidLines)
    wxsize = int(burst.numValidSamples)
    fyoff = int(yoff + burst.firstValidLine)
    fxoff = int(xoff + burst.firstValidSample)

    #print(xoff, fxoff)
    print(yoff, fyoff)
'''
