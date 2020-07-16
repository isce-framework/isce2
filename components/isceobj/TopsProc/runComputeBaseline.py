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
    commonBurstStartReferenceIndex = [-1] * self._insar.numberOfSwaths
    commonBurstStartSecondaryIndex = [-1] * self._insar.numberOfSwaths
    numberOfCommonBursts = [0] * self._insar.numberOfSwaths


    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)

    for swath in swathList:

        referencexml = os.path.join( self._insar.referenceSlcProduct,'IW{0}.xml'.format(swath))
        secondaryxml = os.path.join( self._insar.secondarySlcProduct, 'IW{0}.xml'.format(swath))

        if os.path.exists(referencexml) and os.path.exists(secondaryxml):
            reference = self._insar.loadProduct(referencexml)
            secondary = self._insar.loadProduct(secondaryxml)

            burstOffset, minBurst, maxBurst = reference.getCommonBurstLimits(secondary)
            commonSecondaryIndex = minBurst + burstOffset
            numberCommon = maxBurst - minBurst

            if numberCommon == 0:
                print('No common bursts found for swath {0}'.format(swath))

            else:
                ###Bookkeeping
                commonBurstStartReferenceIndex[swath-1] = minBurst
                commonBurstStartSecondaryIndex[swath-1]  = commonSecondaryIndex
                numberOfCommonBursts[swath-1] = numberCommon


                catalog.addItem('IW-{0} Number of bursts in reference'.format(swath), reference.numberOfBursts, 'baseline')
                catalog.addItem('IW-{0} First common burst in reference'.format(swath), minBurst, 'baseline')
                catalog.addItem('IW-{0} Last common burst in reference'.format(swath), maxBurst, 'baseline')
                catalog.addItem('IW-{0} Number of bursts in secondary'.format(swath), secondary.numberOfBursts, 'baseline')
                catalog.addItem('IW-{0} First common burst in secondary'.format(swath), minBurst + burstOffset, 'baseline')
                catalog.addItem('IW-{0} Last common burst in secondary'.format(swath), maxBurst + burstOffset, 'baseline')
                catalog.addItem('IW-{0} Number of common bursts'.format(swath), numberCommon, 'baseline')

                refElp = Planet(pname='Earth').ellipsoid
                Bpar = []
                Bperp = []

                for boff in [0, numberCommon-1]:
                    ###Baselines at top of common bursts
                    mBurst = reference.bursts[minBurst + boff]
                    sBurst = secondary.bursts[commonSecondaryIndex + boff]

                    ###Target at mid range 
                    tmid = mBurst.sensingMid
                    rng = mBurst.midRange
                    referenceSV = mBurst.orbit.interpolate(tmid, method='hermite')
                    target = mBurst.orbit.rdr2geo(tmid, rng)

                    slvTime, slvrng = sBurst.orbit.geo2rdr(target)
                    secondarySV = sBurst.orbit.interpolateOrbit(slvTime, method='hermite')

                    targxyz = np.array(refElp.LLH(target[0], target[1], target[2]).ecef().tolist())
                    mxyz = np.array(referenceSV.getPosition())
                    mvel = np.array(referenceSV.getVelocity())
                    sxyz = np.array(secondarySV.getPosition())
                    mvelunit = mvel / np.linalg.norm(mvel)
                    sxyz = sxyz - np.dot ( sxyz-mxyz, mvelunit) * mvelunit

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


    self._insar.commonBurstStartReferenceIndex = commonBurstStartReferenceIndex 
    self._insar.commonBurstStartSecondaryIndex = commonBurstStartSecondaryIndex   
    self._insar.numberOfCommonBursts = numberOfCommonBursts


    if not any([x>=2 for x in self._insar.numberOfCommonBursts]):
        print('No swaths contain any burst overlaps ... cannot continue for interferometry applications')

    catalog.printToLog(logger, "runComputeBaseline")
    self._insar.procDoc.addAllFromCatalog(catalog)

