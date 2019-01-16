#
# Author: Piyush Agram
# Copyright 2016
#

import logging
import isceobj
import os
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
    


def runEqualizeSlcs(self):
    '''Align central frequencies and pixel sizes.
    '''

    swathList = self._insar.getValidSwathList(self.swaths)
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)

    print('Common swaths: ', swathList)
    
    for ind, swath in enumerate(swathList):
        
        ##Load master swath
        master = self._insar.loadProduct( os.path.join(self._insar.commonRangeSpectraMasterSlcProduct,
                                                    's{0}.xml'.format(swath)))

        ##Load slave swath
        slave = self._insar.loadProduct( os.path.join(self._insar.commonRangeSpectraSlaveSlcProduct,
                                                    's{0}.xml'.format(swath)))


        ###Check if master needs range filtering
        if (np.abs(master.instrument.rangeSamplingRate - slave.instrument.rangeSamplingRate) < 1.0) and  ((master.PRF - slave.PRF) < 1.0):
            print('No need to equalize pixesl for swath {0}'.format(swath))


            ####Copy master as is
            infile = master.image.filename
            outfile = os.path.join(self._insar.equalizedMasterSlcProduct,
                    's{0}/swath.slc'.format(swath))
            os.makedirs( os.path.dirname(outfile))
            createVirtualCopy(infile, outfile)
            ##Generate product
            master.image.filename = outfile
            self._insar.saveProduct(master, os.path.dirname(outfile) + '.xml')


            ###Copy slave as is
            infile = slave.image.filename
            outfile = os.path.join(self._insar.equalizedSlaveSlcProduct,
                    's{0}/swath.slc'.format(swath))
            os.makedirs( os.path.dirname(outfile))
            createVirtualCopy(infile, outfile)
            ##Generate product
            slave.image.filename = outfile
            self._insar.saveProduct(slave, os.path.dirname(outfile) + '.xml')

        else:
            raise NotImplementedError("We haven't implemented this yet. Maybe we can get around this. To be explored ...")


    catalog.printToLog(logger, "runEqualizeSlcs")
    self._insar.procDoc.addAllFromCatalog(catalog)
