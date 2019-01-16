#
# Author: Piyush Agram
# Copyright 2016
#

import logging
import isceobj
import os
from isceobj.Constants import SPEED_OF_LIGHT
import numpy as np 

logger = logging.getLogger('isce.scansarinsar.runCommonRangeSpectra')


def createVirtualCopy(infile, outfile):
    '''
    Create a virtual copy as is.
    '''
    from osgeo import gdal
    import shutil

    ds = gdal.Open(infile + '.vrt', gdal.GA_ReadOnly)
    width = ds.RasterXSize
    lgth = ds.RasterYSize

    ds = None


    img = isceobj.createSlcImage()
    img.setWidth(width)
    img.setLength(lgth)
    img.setAccessMode('READ')
    img.setFilename(outfile)
    img.renderHdr()

    ##Copy VRT as is
    shutil.copyfile( infile + '.vrt', outfile + '.vrt')
    


def runCommonRangeSpectra(self):
    '''Align central frequencies and pixel sizes.
    '''

    swathList = self._insar.getValidSwathList(self.swaths)
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)

    print('Common swaths: ', swathList)
    
    for ind, swath in enumerate(swathList):
        
        ##Load master swath
        master = self._insar.loadProduct( os.path.join(self._insar.masterSlcProduct,
                                                    's{0}.xml'.format(swath)))

        ##Load slave swath
        slave = self._insar.loadProduct( os.path.join(self._insar.slaveSlcProduct,
                                                    's{0}.xml'.format(swath)))


        ##Check for overlap frequency
        centerfreq1 = SPEED_OF_LIGHT / master.radarWavelegth
        bandwidth1 = np.abs( master.instrument.pulseLength * master.instrument.chirpSlope)

        centerfreq2 = SPEED_OF_LIGHT / slave.radarWavelegth
        bandwidth2 = np.abs( slave.instrument.pulseLength * slave.instrument.chirpSlope)

        overlapfreq = self._insar.getOverlapFrequency(centerfreq1, bandwidth1,
                                                      centerfreq2, bandwidth2)

        if (overlapfreq is None):
            print('No range bandwidth overlap found for swath {0}'.format(swath))
            raise Exception('No range spectra overlap')

        overlapbw = overlapfreq[1] - overlapfreq[0]

        print('Range spectra overlap for swath {0} : {1}'.format(swath, overlapbw))
        if  overlapbw < self._insar.rangeSpectraOverlapThreshold:
            raise Exception('Not enough range spectra overlap for swath {0}'.format(swath))


        centerfreq = 0.5 * (centerfreq1 + centerfreq2)


        ###Check if master needs range filtering
        if (np.abs(centerfreq1 - centerfreq) < 1.0) and  ((bandwidth1 - 1.0) < overlapbw):
            print('No need to range filter master slc for swath {0}'.format(swath))
            infile = master.image.filename
            outfile = os.path.join(self._insar.commonRangeSpectraSlcDirectory,
                    os.path.sep.join(infile.split(os.path.sep)[-4:]))
            os.makedirs( os.path.dirname(outfile))
            createVirtualCopy(infile, outfile)
            ##Generate product
            master.image.filename = outfile
            self._insar.saveProduct(master, os.path.dirname(outfile) + '.xml')


        else:
            raise NotImplementedError('This feature will be available after porting rg_filter')


        ###Check if slave needs range filtering
        if (np.abs(centerfreq2 - centerfreq) < 1.0) and ((bandwidth2 - 1.0) < overlapbw):
            print('No need to range filter slave slc for swath {0}'.format(swath))

            infile = slave.image.filename
            outfile = os.path.join(self._insar.commonRangeSpectraSlcDirectory,
                        os.path.sep.join(infile.split(os.path.sep)[-4:]))
            os.makedirs( os.path.dirname(outfile))
            createVirtualCopy(infile, outfile)
            ##Generate product
            slave.image.filename = outfile
            self._insar.saveProduct(slave, os.path.dirname(outfile) + '.xml')

        else:
            raise NotImplementedError('This feature will be available after porting rg_filter')


    catalog.printToLog(logger, "runCommonRangeSpectra")
    self._insar.procDoc.addAllFromCatalog(catalog)
