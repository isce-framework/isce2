#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import glob
import logging
import datetime
import numpy as np

import isceobj
import isceobj.Sensor.MultiMode as MultiMode
from isceobj.Planet.Planet import Planet
from isceobj.Alos2Proc.Alos2ProcPublic import runCmd
from isceobj.Alos2Proc.Alos2ProcPublic import getBboxRdr
from isceobj.Alos2Proc.Alos2ProcPublic import getBboxGeo

logger = logging.getLogger('isce.alos2insar.runBaseline')

def runBaseline(self):
    '''compute baseline
    '''
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    referenceTrack = self._insar.loadTrack(reference=True)
    secondaryTrack = self._insar.loadTrack(reference=False)


    ##################################################
    #2. compute burst synchronization
    ##################################################
    #burst synchronization may slowly change along a track as a result of the changing relative speed of the two flights
    #in one frame, real unsynchronized time is the same for all swaths
    unsynTime = 0
    #real synchronized time/percentage depends on the swath burst length (synTime = burstlength - abs(unsynTime))
    #synTime = 0
    synPercentage = 0

    numberOfFrames = len(self._insar.referenceFrames)
    numberOfSwaths = self._insar.endingSwath - self._insar.startingSwath + 1
    
    for i, frameNumber in enumerate(self._insar.referenceFrames):
        for j, swathNumber in enumerate(range(self._insar.startingSwath, self._insar.endingSwath + 1)):
            referenceSwath = referenceTrack.frames[i].swaths[j]
            secondarySwath = secondaryTrack.frames[i].swaths[j]
            #using Piyush's code for computing range and azimuth offsets
            midRange = referenceSwath.startingRange + referenceSwath.rangePixelSize * referenceSwath.numberOfSamples * 0.5
            midSensingStart = referenceSwath.sensingStart + datetime.timedelta(seconds = referenceSwath.numberOfLines * 0.5 / referenceSwath.prf)
            llh = referenceTrack.orbit.rdr2geo(midSensingStart, midRange)
            slvaz, slvrng = secondaryTrack.orbit.geo2rdr(llh)
            ###Translate to offsets
            #note that secondary range pixel size and prf might be different from reference, here we assume there is a virtual secondary with same
            #range pixel size and prf
            rgoff = ((slvrng - secondarySwath.startingRange) / referenceSwath.rangePixelSize) - referenceSwath.numberOfSamples * 0.5
            azoff = ((slvaz - secondarySwath.sensingStart).total_seconds() * referenceSwath.prf) - referenceSwath.numberOfLines * 0.5

            #compute burst synchronization
            #burst parameters for ScanSAR wide mode not estimed yet
            if self._insar.modeCombination == 21:
                scburstStartLine = (referenceSwath.burstStartTime - referenceSwath.sensingStart).total_seconds() * referenceSwath.prf + azoff
                #secondary burst start times corresponding to reference burst start times (100% synchronization)
                scburstStartLines = np.arange(scburstStartLine - 100000*referenceSwath.burstCycleLength, \
                                              scburstStartLine + 100000*referenceSwath.burstCycleLength, \
                                              referenceSwath.burstCycleLength)
                dscburstStartLines = -((secondarySwath.burstStartTime - secondarySwath.sensingStart).total_seconds() * secondarySwath.prf - scburstStartLines)
                #find the difference with minimum absolute value
                unsynLines = dscburstStartLines[np.argmin(np.absolute(dscburstStartLines))]
                if np.absolute(unsynLines) >= secondarySwath.burstLength:
                    synLines = 0
                    if unsynLines > 0:
                        unsynLines = secondarySwath.burstLength
                    else:
                        unsynLines = -secondarySwath.burstLength
                else:
                    synLines = secondarySwath.burstLength - np.absolute(unsynLines)

                unsynTime += unsynLines / referenceSwath.prf
                synPercentage += synLines / referenceSwath.burstLength * 100.0

                catalog.addItem('burst synchronization of frame {} swath {}'.format(frameNumber, swathNumber), '%.1f%%'%(synLines / referenceSwath.burstLength * 100.0), 'runBaseline')

            ############################################################################################
            #illustration of the sign of the number of unsynchronized lines (unsynLines)     
            #The convention is the same as ampcor offset, that is,
            #              secondaryLineNumber = referenceLineNumber + unsynLines
            #
            # |-----------------------|     ------------
            # |                       |        ^
            # |                       |        |
            # |                       |        |   unsynLines < 0
            # |                       |        |
            # |                       |       \ /
            # |                       |    |-----------------------|
            # |                       |    |                       |
            # |                       |    |                       |
            # |-----------------------|    |                       |
            #        Reference Burst          |                       |
            #                              |                       |
            #                              |                       |
            #                              |                       |
            #                              |                       |
            #                              |-----------------------|
            #                                     Secondary Burst
            #
            #
            ############################################################################################
 
            ##burst parameters for ScanSAR wide mode not estimed yet
            elif self._insar.modeCombination == 31:
                #scansar is reference
                scburstStartLine = (referenceSwath.burstStartTime - referenceSwath.sensingStart).total_seconds() * referenceSwath.prf + azoff
                #secondary burst start times corresponding to reference burst start times (100% synchronization)
                for k in range(-100000, 100000):
                    saz_burstx = scburstStartLine + referenceSwath.burstCycleLength * k
                    st_burstx = secondarySwath.sensingStart + datetime.timedelta(seconds=saz_burstx / referenceSwath.prf)
                    if saz_burstx >= 0.0 and saz_burstx <= secondarySwath.numberOfLines -1:
                        secondarySwath.burstStartTime = st_burstx
                        secondarySwath.burstLength = referenceSwath.burstLength
                        secondarySwath.burstCycleLength = referenceSwath.burstCycleLength
                        secondarySwath.swathNumber = referenceSwath.swathNumber
                        break
                #unsynLines = 0
                #synLines = referenceSwath.burstLength
                #unsynTime += unsynLines / referenceSwath.prf
                #synPercentage += synLines / referenceSwath.burstLength * 100.0
                catalog.addItem('burst synchronization of frame {} swath {}'.format(frameNumber, swathNumber), '%.1f%%'%(100.0), 'runBaseline')
            else:
                pass

        #overwrite original frame parameter file
        if self._insar.modeCombination == 31:
            frameDir = 'f{}_{}'.format(i+1, frameNumber)
            self._insar.saveProduct(secondaryTrack.frames[i], os.path.join(frameDir, self._insar.secondaryFrameParameter))

    #getting average
    if self._insar.modeCombination == 21:
        unsynTime /= numberOfFrames*numberOfSwaths
        synPercentage /= numberOfFrames*numberOfSwaths
    elif self._insar.modeCombination == 31:
        unsynTime = 0.
        synPercentage = 100.
    else:
        pass

    #record results
    if (self._insar.modeCombination == 21) or (self._insar.modeCombination == 31):
        self._insar.burstUnsynchronizedTime = unsynTime
        self._insar.burstSynchronization = synPercentage
        catalog.addItem('burst synchronization averaged', '%.1f%%'%(synPercentage), 'runBaseline')


    ##################################################
    #3. compute baseline
    ##################################################
    #only compute baseline at four corners and center of the reference track
    bboxRdr = getBboxRdr(referenceTrack)

    rangeMin = bboxRdr[0]
    rangeMax = bboxRdr[1]
    azimuthTimeMin = bboxRdr[2]
    azimuthTimeMax = bboxRdr[3]

    azimuthTimeMid = azimuthTimeMin+datetime.timedelta(seconds=(azimuthTimeMax-azimuthTimeMin).total_seconds()/2.0)
    rangeMid = (rangeMin + rangeMax) / 2.0

    points = [[azimuthTimeMin, rangeMin],
              [azimuthTimeMin, rangeMax],
              [azimuthTimeMax, rangeMin],
              [azimuthTimeMax, rangeMax],
              [azimuthTimeMid, rangeMid]]

    Bpar = []
    Bperp = []
    #modify Piyush's code for computing baslines
    refElp = Planet(pname='Earth').ellipsoid
    for x in points:
        referenceSV = referenceTrack.orbit.interpolate(x[0], method='hermite')
        target = referenceTrack.orbit.rdr2geo(x[0], x[1])

        slvTime, slvrng = secondaryTrack.orbit.geo2rdr(target)
        secondarySV = secondaryTrack.orbit.interpolateOrbit(slvTime, method='hermite')

        targxyz = np.array(refElp.LLH(target[0], target[1], target[2]).ecef().tolist())
        mxyz = np.array(referenceSV.getPosition())
        mvel = np.array(referenceSV.getVelocity())
        sxyz = np.array(secondarySV.getPosition())

        #to fix abrupt change near zero in baseline grid. JUN-05-2020
        mvelunit = mvel / np.linalg.norm(mvel)
        sxyz = sxyz - np.dot ( sxyz-mxyz, mvelunit) * mvelunit

        aa = np.linalg.norm(sxyz-mxyz)
        costheta = (x[1]*x[1] + aa*aa - slvrng*slvrng)/(2.*x[1]*aa)

        Bpar.append(aa*costheta)

        perp = aa * np.sqrt(1 - costheta*costheta)
        direction = np.sign(np.dot( np.cross(targxyz-mxyz, sxyz-mxyz), mvel))
        Bperp.append(direction*perp)    

    catalog.addItem('parallel baseline at upperleft of reference track', Bpar[0], 'runBaseline')
    catalog.addItem('parallel baseline at upperright of reference track', Bpar[1], 'runBaseline')
    catalog.addItem('parallel baseline at lowerleft of reference track', Bpar[2], 'runBaseline')
    catalog.addItem('parallel baseline at lowerright of reference track', Bpar[3], 'runBaseline')
    catalog.addItem('parallel baseline at center of reference track', Bpar[4], 'runBaseline')

    catalog.addItem('perpendicular baseline at upperleft of reference track', Bperp[0], 'runBaseline')
    catalog.addItem('perpendicular baseline at upperright of reference track', Bperp[1], 'runBaseline')
    catalog.addItem('perpendicular baseline at lowerleft of reference track', Bperp[2], 'runBaseline')
    catalog.addItem('perpendicular baseline at lowerright of reference track', Bperp[3], 'runBaseline')
    catalog.addItem('perpendicular baseline at center of reference track', Bperp[4], 'runBaseline')


    ##################################################
    #4. compute bounding box
    ##################################################
    referenceBbox = getBboxGeo(referenceTrack)
    secondaryBbox = getBboxGeo(secondaryTrack)

    catalog.addItem('reference bounding box', referenceBbox, 'runBaseline')
    catalog.addItem('secondary bounding box', secondaryBbox, 'runBaseline')


    catalog.printToLog(logger, "runBaseline")
    self._insar.procDoc.addAllFromCatalog(catalog)


