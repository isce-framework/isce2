#
# Author: Piyush Agram
# Copyright 2016
#

import logging
import isceobj
import mroipac
import os
logger = logging.getLogger('isce.topsinsar.runPreprocessor')

def runComputeBaseline(self):
    
    from isceobj.Planet.Planet import Planet
    import numpy as np



    swathList = self._insar.getInputSwathList(self.swaths)
    commonBurstStartMasterIndex = [-1] * self._insar.numberOfSwaths
    commonBurstStartSlaveIndex = [-1] * self._insar.numberOfSwaths
    numberOfCommonBursts = [0] * self._insar.numberOfSwaths


    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)

    for swath in swathList:

        masterxml = os.path.join( self._insar.masterSlcProduct,'IW{0}.xml'.format(swath))
        slavexml = os.path.join( self._insar.slaveSlcProduct, 'IW{0}.xml'.format(swath))

        if os.path.exists(masterxml) and os.path.exists(slavexml):
            master = self._insar.loadProduct(masterxml)
            slave = self._insar.loadProduct(slavexml)

            burstOffset, minBurst, maxBurst = master.getCommonBurstLimits(slave)
            commonSlaveIndex = minBurst + burstOffset
            numberCommon = maxBurst - minBurst

            if numberCommon == 0:
                print('No common bursts found for swath {0}'.format(swath))

            else:
                ###Bookkeeping
                commonBurstStartMasterIndex[swath-1] = minBurst
                commonBurstStartSlaveIndex[swath-1]  = commonSlaveIndex
                numberOfCommonBursts[swath-1] = numberCommon


                catalog.addItem('IW-{0} Number of bursts in master'.format(swath), master.numberOfBursts, 'baseline')
                catalog.addItem('IW-{0} First common burst in master'.format(swath), minBurst, 'baseline')
                catalog.addItem('IW-{0} Last common burst in master'.format(swath), maxBurst, 'baseline')
                catalog.addItem('IW-{0} Number of bursts in slave'.format(swath), slave.numberOfBursts, 'baseline')
                catalog.addItem('IW-{0} First common burst in slave'.format(swath), minBurst + burstOffset, 'baseline')
                catalog.addItem('IW-{0} Last common burst in slave'.format(swath), maxBurst + burstOffset, 'baseline')
                catalog.addItem('IW-{0} Number of common bursts'.format(swath), numberCommon, 'baseline')

                refElp = Planet(pname='Earth').ellipsoid
                Bpar = []
                Bperp = []

                for boff in [0, numberCommon-1]:
                    ###Baselines at top of common bursts
                    mBurst = master.bursts[minBurst + boff]
                    sBurst = slave.bursts[commonSlaveIndex + boff]

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


                catalog.addItem('IW-{0} Bpar at midrange for first common burst'.format(swath), Bpar[0], 'baseline')
                catalog.addItem('IW-{0} Bperp at midrange for first common burst'.format(swath), Bperp[0], 'baseline')
                catalog.addItem('IW-{0} Bpar at midrange for last common burst'.format(swath), Bpar[1], 'baseline')
                catalog.addItem('IW-{0} Bperp at midrange for last common burst'.format(swath), Bperp[1], 'baseline')


        else:
            print('Skipping processing for swath number IW-{0}'.format(swath))


    self._insar.commonBurstStartMasterIndex = commonBurstStartMasterIndex 
    self._insar.commonBurstStartSlaveIndex = commonBurstStartSlaveIndex   
    self._insar.numberOfCommonBursts = numberOfCommonBursts


    if not any([x>=2 for x in self._insar.numberOfCommonBursts]):
        print('No swaths contain any burst overlaps ... cannot continue for interferometry applications')

    catalog.printToLog(logger, "runComputeBaseline")
    self._insar.procDoc.addAllFromCatalog(catalog)

