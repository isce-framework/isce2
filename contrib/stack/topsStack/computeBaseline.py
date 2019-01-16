#!/usr/bin/env python3
# Author: Piyush Agram
# Copyright 2016
#Heresh Fattahi, Adopted for stack

import argparse
import logging
import isce
import isceobj
import mroipac
import os
import s1a_isce_utils as ut

def createParser():
    parser = argparse.ArgumentParser( description='Use polynomial offsets and create burst by burst interferograms')

    parser.add_argument('-m', '--master', dest='master', type=str, required=True,
            help='Directory with master acquisition')

    parser.add_argument('-s', '--slave', dest='slave', type=str, required=True,
            help='Directory with slave acquisition')

    parser.add_argument('-b', '--baseline_file', dest='baselineFile', type=str, required=True,
                help='An output text file which contains the computed baseline')


    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)



#logger = logging.getLogger('isce.topsinsar.runPreprocessor')

def main(iargs=None):
    '''Compute baseline.
    '''
    inps=cmdLineParse(iargs)
    from isceobj.Planet.Planet import Planet
    import numpy as np



    #swathList = self._insar.getInputSwathList(self.swaths)
    #commonBurstStartMasterIndex = [-1] * self._insar.numberOfSwaths
    #commonBurstStartSlaveIndex = [-1] * self._insar.numberOfSwaths
    #numberOfCommonBursts = [0] * self._insar.numberOfSwaths

    masterSwathList = ut.getSwathList(inps.master)
    slaveSwathList = ut.getSwathList(inps.slave)
    swathList = list(sorted(set(masterSwathList+slaveSwathList)))

    #catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    baselineDir = os.path.dirname(inps.baselineFile)
    if not os.path.exists(baselineDir):
        os.makedirs(baselineDir)

    f = open(inps.baselineFile , 'w')

    for swath in swathList:

        masterxml = os.path.join( inps.master, 'IW{0}.xml'.format(swath))
        slavexml = os.path.join( inps.slave, 'IW{0}.xml'.format(swath))

        if os.path.exists(masterxml) and os.path.exists(slavexml):

            master = ut.loadProduct(os.path.join(inps.master , 'IW{0}.xml'.format(swath)))
            slave = ut.loadProduct(os.path.join(inps.slave , 'IW{0}.xml'.format(swath)))

            minMaster = master.bursts[0].burstNumber
            maxMaster = master.bursts[-1].burstNumber

            minSlave = slave.bursts[0].burstNumber
            maxSlave = slave.bursts[-1].burstNumber

            minBurst = max(minSlave, minMaster)
            maxBurst = min(maxSlave, maxMaster)
            print ('minSlave,maxSlave',minSlave, maxSlave)
            print ('minMaster,maxMaster',minMaster, maxMaster)
            print ('minBurst, maxBurst: ', minBurst, maxBurst)
            refElp = Planet(pname='Earth').ellipsoid
            Bpar = []
            Bperp = []

            for ii in range(minBurst, maxBurst + 1):


                ###Bookkeeping
                #commonBurstStartMasterIndex[swath-1] = minBurst
                #commonBurstStartSlaveIndex[swath-1]  = commonSlaveIndex
                #numberOfCommonBursts[swath-1] = numberCommon


                #catalog.addItem('IW-{0} Number of bursts in master'.format(swath), master.numberOfBursts, 'baseline')
                #catalog.addItem('IW-{0} First common burst in master'.format(swath), minBurst, 'baseline')
                #catalog.addItem('IW-{0} Last common burst in master'.format(swath), maxBurst, 'baseline')
                #catalog.addItem('IW-{0} Number of bursts in slave'.format(swath), slave.numberOfBursts, 'baseline')
                #catalog.addItem('IW-{0} First common burst in slave'.format(swath), minBurst + burstOffset, 'baseline')
                #catalog.addItem('IW-{0} Last common burst in slave'.format(swath), maxBurst + burstOffset, 'baseline')
                #catalog.addItem('IW-{0} Number of common bursts'.format(swath), numberCommon, 'baseline')

                #refElp = Planet(pname='Earth').ellipsoid
                #Bpar = []
                #Bperp = []

                #for boff in [0, numberCommon-1]:
                    ###Baselines at top of common bursts
                mBurst = master.bursts[ii-minMaster]
                sBurst = slave.bursts[ii-minSlave]

                    ###Target at mid range 
                tmid = mBurst.sensingMid
                rng = mBurst.midRange
                masterSV = mBurst.orbit.interpolate(tmid, method='hermite')
                target = mBurst.orbit.rdr2geo(tmid, rng)

                slvTime, slvrng = sBurst.orbit.geo2rdr(target)
                slaveSV = sBurst.orbit.interpolateOrbit(slvTime, method='hermite')

                targxyz = np.array(refElp.LLH(target[0], target[1], target[2]).ecef().tolist())
                mxyz = np.array(masterSV.getPosition())
                mvel = np.array(masterSV.getVelocity())
                sxyz = np.array(slaveSV.getPosition())

                aa = np.linalg.norm(sxyz-mxyz)
                costheta = (rng*rng + aa*aa - slvrng*slvrng)/(2.*rng*aa)

                Bpar.append(aa*costheta)

                perp = aa * np.sqrt(1 - costheta*costheta)
                direction = np.sign(np.dot( np.cross(targxyz-mxyz, sxyz-mxyz), mvel))
                Bperp.append(direction*perp)    


                #catalog.addItem('IW-{0} Bpar at midrange for first common burst'.format(swath), Bpar[0], 'baseline')
                #catalog.addItem('IW-{0} Bperp at midrange for first common burst'.format(swath), Bperp[0], 'baseline')
                #catalog.addItem('IW-{0} Bpar at midrange for last common burst'.format(swath), Bpar[1], 'baseline')
                #catalog.addItem('IW-{0} Bperp at midrange for last common burst'.format(swath), Bperp[1], 'baseline')

            print('Bprep: ', Bperp)
            print('Bpar: ', Bpar)
            f.write('swath: IW{0}'.format(swath) + '\n')
            f.write('Bperp (average): ' + str(np.mean(Bperp))  + '\n')
            f.write('Bpar (average): ' + str(np.mean(Bpar))  + '\n')

    f.close()
        #else:
        #    print('Skipping processing for swath number IW-{0}'.format(swath))

            
    #self._insar.commonBurstStartMasterIndex = commonBurstStartMasterIndex 
    #self._insar.commonBurstStartSlaveIndex = commonBurstStartSlaveIndex   
    #self._insar.numberOfCommonBursts = numberOfCommonBursts


    #if not any([x>=2 for x in self._insar.numberOfCommonBursts]):
    #    print('No swaths contain any burst overlaps ... cannot continue for interferometry applications')

    #catalog.printToLog(logger, "runComputeBaseline")
    #self._insar.procDoc.addAllFromCatalog(catalog)

if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()

