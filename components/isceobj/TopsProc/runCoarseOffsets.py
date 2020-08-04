#
# Author: Piyush Agram
# Copyright 2016
#

import numpy as np 
import os
import isceobj
import datetime
import sys
import logging

logger = logging.getLogger('isce.topsinsar.coarseoffsets')

def runGeo2rdr(info, rdict, misreg_az=0.0, misreg_rg=0.0, virtual=False):
    from zerodop.geo2rdr import createGeo2rdr
    from isceobj.Planet.Planet import Planet

    latImage = isceobj.createImage()
    latImage.load(rdict['lat'] + '.xml')
    latImage.setAccessMode('READ')


    lonImage = isceobj.createImage()
    lonImage.load(rdict['lon'] + '.xml')
    lonImage.setAccessMode('READ')

    demImage = isceobj.createImage()
    demImage.load(rdict['hgt'] + '.xml')
    demImage.setAccessMode('READ')

    delta = datetime.timedelta(seconds=misreg_az)
    logger.info('Additional time offset applied in geo2rdr: {0} secs'.format(misreg_az))
    logger.info('Additional range offset applied in geo2rdr: {0} m'.format(misreg_rg))


    #####Run Geo2rdr
    planet = Planet(pname='Earth')
    grdr = createGeo2rdr()
    grdr.configure()

    grdr.slantRangePixelSpacing = info.rangePixelSize
    grdr.prf = 1.0 / info.azimuthTimeInterval
    grdr.radarWavelength = info.radarWavelength
    grdr.orbit = info.orbit
    grdr.width = info.numberOfSamples
    grdr.length = info.numberOfLines
    grdr.demLength = demImage.getLength()
    grdr.demWidth = demImage.getWidth()
    grdr.wireInputPort(name='planet', object=planet)
    grdr.numberRangeLooks = 1
    grdr.numberAzimuthLooks = 1
    grdr.lookSide = -1  
    grdr.setSensingStart(info.sensingStart - delta)
    grdr.rangeFirstSample = info.startingRange - misreg_rg
    grdr.dopplerCentroidCoeffs = [0.]  ###Zero doppler

    grdr.rangeOffsetImageName = rdict['rangeOffName']
    grdr.azimuthOffsetImageName = rdict['azOffName']
    grdr.demImage = demImage
    grdr.latImage = latImage
    grdr.lonImage = lonImage

    grdr.geo2rdr()

    return


def runCoarseOffsets(self):
    '''
    Estimate offsets for the overlap regions of the bursts.
    '''
    
    virtual = self.useVirtualFiles
    if not self.doESD:
        return 


    ##Catalog
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    misreg_az = self._insar.secondaryTimingCorrection
    catalog.addItem('Initial secondary azimuth timing correction', misreg_az, 'coarseoff')

    misreg_rg = self._insar.secondaryRangeCorrection
    catalog.addItem('Initial secondary range timing correction', misreg_rg, 'coarseoff')

    swathList = self._insar.getValidSwathList(self.swaths)
    
    for swath in swathList:

        if self._insar.numberOfCommonBursts[swath-1] < 2:
            print('Skipping coarse offsets for swath IW{0}'.format(swath))
            continue

        ##Load secondary metadata
        secondary = self._insar.loadProduct(os.path.join(self._insar.secondarySlcProduct, 'IW{0}.xml'.format(swath)))

    
        ###Offsets output directory
        outdir = os.path.join(self._insar.coarseOffsetsDirname, self._insar.overlapsSubDirname, 'IW{0}'.format(swath))

        os.makedirs(outdir, exist_ok=True)


        ###Burst indices w.r.t reference
        minBurst = self._insar.commonBurstStartReferenceIndex[swath-1]
        maxBurst =  minBurst + self._insar.numberOfCommonBursts[swath-1] - 1 ###-1 for overlaps
        referenceOverlapDir = os.path.join(self._insar.referenceSlcOverlapProduct, 'IW{0}'.format(swath))
        geomOverlapDir = os.path.join(self._insar.geometryDirname, self._insar.overlapsSubDirname, 'IW{0}'.format(swath))

        secondaryBurstStart = self._insar.commonBurstStartSecondaryIndex[swath-1]

        catalog.addItem('Number of overlap pairs - IW-{0}'.format(swath), maxBurst - minBurst, 'coarseoff')

        for mBurst in range(minBurst, maxBurst):

            ###Corresponding secondary burst
            sBurst = secondaryBurstStart + (mBurst - minBurst)
            burstTop = secondary.bursts[sBurst]
            burstBot = secondary.bursts[sBurst+1]

            logger.info('Overlap pair {0}, IW-{3}: Burst {1} of reference matched with Burst {2} of secondary'.format(mBurst-minBurst, mBurst, sBurst, swath))
            ####Generate offsets for top burst
            rdict = {'lat': os.path.join(geomOverlapDir,'lat_%02d_%02d.rdr'%(mBurst+1,mBurst+2)),
                     'lon': os.path.join(geomOverlapDir,'lon_%02d_%02d.rdr'%(mBurst+1,mBurst+2)),
                     'hgt': os.path.join(geomOverlapDir,'hgt_%02d_%02d.rdr'%(mBurst+1,mBurst+2)),
                     'rangeOffName': os.path.join(outdir, 'range_top_%02d_%02d.off'%(mBurst+1,mBurst+2)),
                    'azOffName': os.path.join(outdir, 'azimuth_top_%02d_%02d.off'%(mBurst+1,mBurst+2))}
       
            runGeo2rdr(burstTop, rdict, misreg_az=misreg_az, misreg_rg=misreg_rg)

            logger.info('Overlap pair {0} - IW-{3}: Burst {1} of reference matched with Burst {2} of secondary'.format(mBurst-minBurst, mBurst+1, sBurst+1, swath))

            ####Generate offsets for bottom burst
            rdict = {'lat': os.path.join(geomOverlapDir,'lat_%02d_%02d.rdr'%(mBurst+1,mBurst+2)),
                     'lon': os.path.join(geomOverlapDir, 'lon_%02d_%02d.rdr'%(mBurst+1,mBurst+2)),
                     'hgt': os.path.join(geomOverlapDir, 'hgt_%02d_%02d.rdr'%(mBurst+1,mBurst+2)),
                     'rangeOffName': os.path.join(outdir, 'range_bot_%02d_%02d.off'%(mBurst+1,mBurst+2)),
                    'azOffName': os.path.join(outdir, 'azimuth_bot_%02d_%02d.off'%(mBurst+1,mBurst+2))}
             
            runGeo2rdr(burstBot, rdict, misreg_az=misreg_az, misreg_rg=misreg_rg, virtual=virtual)

