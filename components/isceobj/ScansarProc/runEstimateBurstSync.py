#
# Author: Piyush Agram
# Copyright 2016
#

import logging
import isceobj
import os
import numpy as np 
import datetime

logger = logging.getLogger('isce.scansarinsar.runEstimateBurstSync')


def runEstimateBurstSync(self):
    '''Estimate burst sync between acquisitions.
    '''

    swathList = self._insar.getValidSwathList(self.swaths)
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)

    print('Common swaths: ', swathList)
   
    if not os.path.isdir(self._insar.burstSyncDirectory):
        os.makedirs(self._insar.burstSyncDirectory)
    
    for ind, swath in enumerate(swathList):
        
        ##Load master swath
        master = self._insar.loadProduct( os.path.join(self._insar.equalizedMasterSlcProduct,
                                                    's{0}.xml'.format(swath)))

        ##Load slave swath
        slave = self._insar.loadProduct( os.path.join(self._insar.equalizedSlaveSlcProduct,
                                                    's{0}.xml'.format(swath)))

        ##Replacing Cunren's original high fidelity implementation with simpler more efficient one
        ##To estimate burst sync, we dont really need the DEM. Hardly changes with topography
        ##Topo impacts range offset but hardly affects azimuth offsets. Hence using a single estimate 
        ##At mid range and start of middle burst of the master. 
        ##In the original implementation - topo and geo2rdr were performed to the mid 100 lines of 
        ##master image. Eventually, offset estimates were averaged to a single number.
        ##Our single estimate gets us to that number in a simpler manner.

        ###Get mid range for middle burst of master
        midRange = master.startingRange + 0.5 * master.numberOfSamples * master.instrument.rangePixelSize
        midLine = master.burstStartLines[len(master.burstStartLines)//2]

        tmaster = master.sensingStart + datetime.timedelta(seconds = midLine / master.PRF)
        llh = master.orbit.rdr2geo(tmaster, midRange)

        slvaz, slvrng = slave.orbit.geo2rdr(llh)


        ###Translate to offsets
        rgoff = ((slvrng - slave.startingRange) / slave.instrument.rangePixelSize) - 0.5 * master.numberOfSamples
        azoff = ((slvaz - slave.sensingStart).total_seconds() * slave.PRF) - midLine

       
        ##Jumping back to Cunren's original code
        scburstStartLine = master.burstStartLines[0] + azoff
        nb = slave.nbraw
        nc = slave.ncraw

        #Slave burst start times corresponding to master burst start times implies 100% synchronization
        scburstStartLines = scburstStartLine + np.arange(-100000, 100000)*nc
        dscburstStartLines = -(slave.burstStartLines[0] - scburstStartLines)

        unsynLines = dscburstStartLines[ np.argmin( np.abs(dscburstStartLines))]

        if np.abs(unsynLines) >= nb:
            synLines = 0
            if unsynLines > 0:
                unsynLines = nb
            else:
                unsynLines = -nb
        else:
            synLines = nb - np.abs(unsynLines)

        
        ##Copy of illustration from Cunren's code
          #############################################################  ###############################
          #illustration of the sign of the number of unsynchronized lin  es (unsynLines)
          #The convention is the same as ampcor offset, that is,
          #              slaveLineNumber = masterLineNumber + unsynLine  s
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
          #        Master Burst          |                       |
          #                              |                       |
          #                              |                       |
          #                              |                       |
          #                              |                       |
          #                              |-----------------------|
          #                                     Slave Burst
          #
          #
          #############################################################  ###############################

        ##For now keeping Cunren's text file format. 
        ##Could be streamlined
        outfile = os.path.join(self._insar.burstSyncDirectory, 's{0}.txt'.format(swath))
        self._insar.writeBurstSyncFile(outfile, rgoff, azoff,
                                                nb, nc,
                                                unsynLines, synLines)

        synPerc = (synLines/nb)*100.0
        
        if synPerc < self.burstOverlapThreshold:
            print('Sync overlap {0} < {1}. Will trigger common azimuth spectra filter for swath {2}'.format(synPerc, self.burstOverlapThreshold, swath))
        else:
            print('Sync overlap {0} >= {1}. No common azimuth spectra filter applied for swath {2}'.format(synPerc, self.burstOverlapThreshold, swath))

    catalog.printToLog(logger, "runEstimateBurstSync")
    self._insar.procDoc.addAllFromCatalog(catalog)
