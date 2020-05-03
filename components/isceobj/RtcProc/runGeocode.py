#!/usr/bin/env python3
# coding: utf-8
# Author: Simon Kraatz
# Copyright 2016

import logging
import isceobj
import os
import numpy as np
from isceobj.Util.decorators import use_api
from osgeo import gdal, ogr, osr

logger = logging.getLogger('isce.grdsar.looks')

def runGeocode(self):
    '''
    Geocode a swath file using corresponding lat, lon files
    '''
    sourcexmltmpl = '''    <SimpleSource>
      <SourceFilename>{0}</SourceFilename>
      <SourceBand>{1}</SourceBand>
    </SimpleSource>'''
    
    gcl = [f for f in os.listdir(self._grd.outputFolder) if f.startswith('gamma') and f.endswith('.vrt')] 
    a, b = os.path.split(self._grd.outputFolder)
    latfile = os.path.join(a,self._grd.geometryFolder,'lat.rdr.vrt')
    lonfile = os.path.join(a,self._grd.geometryFolder,'lon.rdr.vrt')
    
    outsrs = 'EPSG:'+str(self.epsg)
    gspacing = self.gspacing
    method = self.intmethod
    insrs = 4326 
    fmt = 'GTiff'
    fl = len(gcl)
    
    for num, val in enumerate(gcl):
        print('****Geocoding file %s out of %s: %s****' %(num+1, fl, val))
        infile = os.path.join(a, self._grd.outputFolder, val)
        outfile = os.path.join(a, self._grd.outputFolder, val[:-3]+'tif')
        
        driver = gdal.GetDriverByName('VRT')
        tempvrtname = os.path.join(a, self._grd.outputFolder, 'geocode.vrt')

        inds = gdal.OpenShared(infile, gdal.GA_ReadOnly)
        tempds = driver.Create(tempvrtname, inds.RasterXSize, inds.RasterYSize, 0)

        for ii in range(inds.RasterCount):
            band = inds.GetRasterBand(1)
            tempds.AddBand(band.DataType)
            tempds.GetRasterBand(ii+1).SetMetadata({'source_0': sourcexmltmpl.format(infile, ii+1)}, 'vrt_sources')
      
        sref = osr.SpatialReference()
        sref.ImportFromEPSG(insrs)
        srswkt = sref.ExportToWkt()
        
        tempds.SetMetadata({'SRS' : srswkt,
                            'X_DATASET': lonfile,
                            'X_BAND' : '1',
                            'Y_DATASET': latfile,
                            'Y_BAND' : '1',
                            'PIXEL_OFFSET' : '0',
                            'LINE_OFFSET' : '0',
                            'PIXEL_STEP' : '1',
                            'LINE_STEP' : '1'}, 'GEOLOCATION')
        
        band = None
        tempds = None 
        inds = None
        bounds = None
        
        spacing = [gspacing, gspacing]
        
        warpOptions = gdal.WarpOptions(format=fmt,
                                       xRes=spacing[0], yRes=spacing[1],
                                       dstSRS=outsrs,
                                       outputBounds = bounds,
                                       resampleAlg=method, geoloc=True)
        gdal.Warp(outfile, tempvrtname, options=warpOptions)
    
    return
