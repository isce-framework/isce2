#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2019 California Institute of Technology. ALL RIGHTS RESERVED.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# United States Government Sponsorship acknowledged. This software is subject to
# U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
# (No [Export] License Required except when exporting to an embargoed country,
# end user, or in support of a prohibited end use). By downloading this software,
# the user agrees to comply with all applicable U.S. export laws and regulations.
# The user has the responsibility to obtain export licenses, or other export
# authority as may be required before exporting this software to any 'EAR99'
# embargoed foreign country or citizen of those countries.
#
# Authors: Piyush Agram, Yang Lei
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






import pdb
import subprocess
import re
import string

class GeogridOptical():
    '''
    Class for mapping regular geographic grid on radar imagery.
    '''

    def runGeogrid(self):
        '''
        Do the actual processing.
        '''

        ##Determine appropriate EPSG system
        self.epsgDem = self.getProjectionSystem(self.demname, self.urlflag)
        self.epsgDat = self.getProjectionSystem(self.dat1name, self.urlflag)
        
        ###Determine extent of data needed
        bbox = self.determineBbox()

        
        ##Run
        self.geogrid()


    def getProjectionSystem(self, filename, urlflag):
        '''
        Testing with Greenland.
        '''
        if not filename:
            raise Exception('File {0} does not exist'.format(filename))

        from osgeo import gdal, osr
        if urlflag == 1:
            ds = gdal.Open('/vsicurl/%s' %(filename))
        else:
            ds = gdal.Open(filename, gdal.GA_ReadOnly)
        srs = osr.SpatialReference()
        srs.ImportFromWkt(ds.GetProjection())
        srs.AutoIdentifyEPSG()
        ds = None
#        pdb.set_trace()

        if srs.IsGeographic():
            epsgstr = srs.GetAuthorityCode('GEOGCS')
        elif srs.IsProjected():
            epsgstr = srs.GetAuthorityCode('PROJCS')
        elif srs.IsLocal():
            raise Exception('Local coordinate system encountered')
        else:
            raise Exception('Non-standard coordinate system encountered')
        if not epsgstr:  #Empty string->use shell command gdalsrsinfo for last trial
            if urlflag == 1:
                cmd = 'gdalsrsinfo -o epsg /vsicurl/{0}'.format(filename)
            else:
                cmd = 'gdalsrsinfo -o epsg {0}'.format(filename)
            epsgstr = subprocess.check_output(cmd, shell=True)
#            pdb.set_trace()
            epsgstr = re.findall("EPSG:(\d+)", str(epsgstr))[0]
#            pdb.set_trace()
        if not epsgstr:  #Empty string
            raise Exception('Could not auto-identify epsg code')
#        pdb.set_trace()
        epsgcode = int(epsgstr)
#        pdb.set_trace()
        return epsgcode

    def determineBbox(self, zrange=[-200,4000]):
        '''
        Dummy.
        '''
        import numpy as np
        import datetime
        from osgeo import osr
        
#        import pdb
#        pdb.set_trace()


        samples = self.startingX + np.array([0, self.numberOfSamples]) * self.XSize
        lines = self.startingY + np.array([0, self.numberOfLines]) * self.YSize

        coordDat = osr.SpatialReference()
        if self.epsgDat:
            coordDat.ImportFromEPSG(self.epsgDat)
        else:
            raise Exception('EPSG code does not exist for image data')
    

        coordDem = osr.SpatialReference()
        if self.epsgDem:
            coordDem.ImportFromEPSG(self.epsgDem)
        else:
            raise Exception('EPSG code does not exist for DEM')
        
        
        trans = osr.CoordinateTransformation(coordDat, coordDem)
        
        

        utms = []
        xyzs = []


        ### Four corner coordinates
        for ss in samples:
            for ll in lines:
                for zz in zrange:
                    utms.append([ss,ll,zz])
                    x,y,z = trans.TransformPoint(ss, ll, zz)
                    xyzs.append([x,y,z])

        utms = np.array(utms)
        xyzs = np.array(xyzs)

        self._xlim = [np.min(xyzs[:,0]), np.max(xyzs[:,0])]
        self._ylim = [np.min(xyzs[:,1]), np.max(xyzs[:,1])]

                
                
                
    
    
    def geogrid(self):
        
        #   For now print inputs that were obtained
        
        urlflag = self.urlflag
        
        if urlflag == 1:
            print("\nReading input images into memory directly from URL's")
        else:
            print("\nReading input images locally from files")
    
        print("\nOptical Image parameters: ")
        print("X-direction coordinate: " + str(self.startingX) + "  " + str(self.XSize))
        print("Y-direction coordinate: " + str(self.startingY) + "  " + str(self.YSize))
        print("Dimensions: " + str(self.numberOfSamples) + "  " + str(self.numberOfLines) + "\n")
        
        print("Map inputs: ")
        print("EPSG: " + str(self.epsgDem))
        print("Smallest Allowable Chip Size in m: " + str(self.chipSizeX0))
        print("Repeat Time: " + str(self.repeatTime))
        print("XLimits: " + str(self._xlim[0]) + "  " + str(self._xlim[1]))
        print("YLimits: " + str(self._ylim[0]) + "  " + str(self._ylim[1]))
        print("Extent in km: " + str((self._xlim[1]-self._xlim[0])/1000.0) + "  " + str((self._ylim[1]-self._ylim[0])/1000.0))
        if (self.demname != ""):
            print("DEM: " + str(self.demname))
        if (self.dhdxname != ""):
            print("Slopes: " + str(self.dhdxname) + "  " + str(self.dhdyname))
        if (self.vxname != ""):
            print("Velocities: " + str(self.vxname) + "  " + str(self.vyname))
        if (self.srxname != ""):
            print("Search Range: " + str(self.srxname) + "  " + str(self.sryname))
        if (self.csminxname != ""):
            print("Chip Size Min: " + str(self.csminxname) + "  " + str(self.csminyname))
        if (self.csmaxxname != ""):
            print("Chip Size Max: " + str(self.csmaxxname) + "  " + str(self.csmaxyname))
        if (self.ssmname != ""):
            print("Stable Surface Mask: " + str(self.ssmname))


        print("\nOutputs: ")

        print("Window locations: " + str(self.winlocname))

        if (self.dhdxname != ""):
            if (self.vxname != ""):
                print("Window offsets: " + str(self.winoffname))
            
            print("Window rdr_off2vel_x vector: " + str(self.winro2vxname))
            print("Window rdr_off2vel_y vector: " + str(self.winro2vyname))
            
            if (self.srxname != ""):
                print("Window search range: " + str(self.winsrname))

        if (self.csminxname != ""):
            print("Window chip size min: " + str(self.wincsminname))
        if (self.csmaxxname != ""):
            print("Window chip size max: " + str(self.wincsmaxname))
        if (self.ssmname != ""):
            print("Window stable surface mask: " + str(self.winssmname))

        print("Output Nodata Value: " + str(self.nodata_out) + "\n")

        
        
        print("Starting processing .... ")
        
        
        
        
        from osgeo import gdal, osr
        import numpy as np
        import struct
        
#        pdb.set_trace()
        if urlflag == 1:
            self.demname = '/vsicurl/%s' %(self.demname)
            self.dhdxname = '/vsicurl/%s' %(self.dhdxname)
            self.dhdyname = '/vsicurl/%s' %(self.dhdyname)
            self.vxname = '/vsicurl/%s' %(self.vxname)
            self.vyname = '/vsicurl/%s' %(self.vyname)
            self.srxname = '/vsicurl/%s' %(self.srxname)
            self.sryname = '/vsicurl/%s' %(self.sryname)
            self.csminxname = '/vsicurl/%s' %(self.csminxname)
            self.csminyname = '/vsicurl/%s' %(self.csminyname)
            self.csmaxxname = '/vsicurl/%s' %(self.csmaxxname)
            self.csmaxyname = '/vsicurl/%s' %(self.csmaxyname)
            self.ssmname = '/vsicurl/%s' %(self.ssmname)
        

        demDS = gdal.Open(self.demname, gdal.GA_ReadOnly)
        
        if (self.dhdxname != ""):
            sxDS = gdal.Open(self.dhdxname, gdal.GA_ReadOnly)
            syDS = gdal.Open(self.dhdyname, gdal.GA_ReadOnly)
        
        if (self.vxname != ""):
            vxDS = gdal.Open(self.vxname, gdal.GA_ReadOnly)
            vyDS = gdal.Open(self.vyname, gdal.GA_ReadOnly)
        
        if (self.srxname != ""):
            srxDS = gdal.Open(self.srxname, gdal.GA_ReadOnly)
            sryDS = gdal.Open(self.sryname, gdal.GA_ReadOnly)
        
        if (self.csminxname != ""):
            csminxDS = gdal.Open(self.csminxname, gdal.GA_ReadOnly)
            csminyDS = gdal.Open(self.csminyname, gdal.GA_ReadOnly)
        
        if (self.csmaxxname != ""):
            csmaxxDS = gdal.Open(self.csmaxxname, gdal.GA_ReadOnly)
            csmaxyDS = gdal.Open(self.csmaxyname, gdal.GA_ReadOnly)
        
        if (self.ssmname != ""):
            ssmDS = gdal.Open(self.ssmname, gdal.GA_ReadOnly)
        
        if demDS is None:
            raise Exception('Error opening DEM file {0}'.format(self.demname))
    
        if (self.dhdxname != ""):
            if (sxDS is None):
                raise Exception('Error opening x-direction slope file {0}'.format(self.dhdxname))
            if (syDS is None):
                raise Exception('Error opening y-direction slope file {0}'.format(self.dhdyname))
        
        if (self.vxname != ""):
            if (vxDS is None):
                raise Exception('Error opening x-direction velocity file {0}'.format(self.vxname))
            if (vyDS is None):
                raise Exception('Error opening y-direction velocity file {0}'.format(self.vyname))

        if (self.srxname != ""):
            if (srxDS is None):
                raise Exception('Error opening x-direction search range file {0}'.format(self.srxname))
            if (sryDS is None):
                raise Exception('Error opening y-direction search range file {0}'.format(self.sryname))

        if (self.csminxname != ""):
            if (csminxDS is None):
                raise Exception('Error opening x-direction chip size min file {0}'.format(self.csminxname))
            if (csminyDS is None):
                raise Exception('Error opening y-direction chip size min file {0}'.format(self.csminyname))

        if (self.csmaxxname != ""):
            if (csmaxxDS is None):
                raise Exception('Error opening x-direction chip size max file {0}'.format(self.csmaxxname))
            if (csmaxyDS is None):
                raise Exception('Error opening y-direction chip size max file {0}'.format(self.csmaxyname))
        
        if (self.ssmname != ""):
            if (ssmDS is None):
                raise Exception('Error opening stable surface mask file {0}'.format(self.ssmname))

        geoTrans = demDS.GetGeoTransform()
        demXSize = demDS.RasterXSize
        demYSize = demDS.RasterYSize
    
    
        #        Get offsets and size to read from DEM
        lOff = int(np.max( [np.floor((self._ylim[1] - geoTrans[3])/geoTrans[5]), 0.]))
#        pdb.set_trace()
        lCount = int(np.min([ np.ceil((self._ylim[0] - geoTrans[3])/geoTrans[5]), demYSize-1.]) - lOff)

        pOff = int(np.max([ np.floor((self._xlim[0] - geoTrans[0])/geoTrans[1]), 0.]))
        pCount = int(np.min([ np.ceil((self._xlim[1] - geoTrans[0])/geoTrans[1]), demXSize-1.]) - pOff)

        print("Xlimits : " + str(geoTrans[0] + pOff * geoTrans[1]) +  "  " + str(geoTrans[0] + (pOff + pCount) * geoTrans[1]))

        print("Ylimits : " + str(geoTrans[3] + (lOff + lCount) * geoTrans[5]) +  "  " + str(geoTrans[3] + lOff * geoTrans[5]))
    
        print("Origin index (in DEM) of geogrid: " + str(pOff) + "   " + str(lOff))
        
        print("Dimensions of geogrid: " + str(pCount) + " x " + str(lCount))
                
        projDem = osr.SpatialReference()
        if self.epsgDem:
            projDem.ImportFromEPSG(self.epsgDem)
        else:
            raise Exception('EPSG code does not exist for DEM')
    
        projDat = osr.SpatialReference()
        if self.epsgDat:
            projDat.ImportFromEPSG(self.epsgDat)
        else:
            raise Exception('EPSG code does not exist for image data')
    
        fwdTrans = osr.CoordinateTransformation(projDem, projDat)
        invTrans = osr.CoordinateTransformation(projDat, projDem)
            
        if (self.vxname != ""):
            nodata = vxDS.GetRasterBand(1).GetNoDataValue()
        else:
            nodata = 0

        nodata_out = self.nodata_out


        pszFormat = "GTiff"
        adfGeoTransform = ( geoTrans[0] + pOff * geoTrans[1], geoTrans[1], 0, geoTrans[3] + lOff * geoTrans[5], 0, geoTrans[5] )
        oSRS = osr.SpatialReference()
        pszSRS_WKT = projDem.ExportToWkt()



        poDriver = gdal.GetDriverByName(pszFormat)
        if( poDriver is None ):
            raise Exception('Cannot create gdal driver for output')

        pszDstFilename = self.winlocname
        poDstDS = poDriver.Create(pszDstFilename, xsize=pCount, ysize=lCount, bands=2, eType=gdal.GDT_Int32)
        poDstDS.SetGeoTransform( adfGeoTransform )
        poDstDS.SetProjection( pszSRS_WKT )

        poBand1 = poDstDS.GetRasterBand(1)
        poBand2 = poDstDS.GetRasterBand(2)
        poBand1.SetNoDataValue(nodata_out)
        poBand2.SetNoDataValue(nodata_out)



        if ((self.dhdxname != "")&(self.vxname != "")):
            poDriverOff = gdal.GetDriverByName(pszFormat)
            if( poDriverOff is None ):
                raise Exception('Cannot create gdal driver for output')
            
            pszDstFilenameOff = self.winoffname
            poDstDSOff = poDriverOff.Create(pszDstFilenameOff, xsize=pCount, ysize=lCount, bands=2, eType=gdal.GDT_Int32)
            poDstDSOff.SetGeoTransform( adfGeoTransform )
            poDstDSOff.SetProjection( pszSRS_WKT )
            
            poBand1Off = poDstDSOff.GetRasterBand(1)
            poBand2Off = poDstDSOff.GetRasterBand(2)
            poBand1Off.SetNoDataValue(nodata_out)
            poBand2Off.SetNoDataValue(nodata_out)


        if ((self.dhdxname != "")&(self.srxname != "")):
            poDriverSch = gdal.GetDriverByName(pszFormat)
            if( poDriverSch is None ):
                raise Exception('Cannot create gdal driver for output')
            
            pszDstFilenameSch = self.winsrname
            poDstDSSch = poDriverSch.Create(pszDstFilenameSch, xsize=pCount, ysize=lCount, bands=2, eType=gdal.GDT_Int32)
            poDstDSSch.SetGeoTransform( adfGeoTransform )
            poDstDSSch.SetProjection( pszSRS_WKT )
            
            poBand1Sch = poDstDSSch.GetRasterBand(1)
            poBand2Sch = poDstDSSch.GetRasterBand(2)
            poBand1Sch.SetNoDataValue(nodata_out)
            poBand2Sch.SetNoDataValue(nodata_out)

        if (self.csminxname != ""):
            poDriverMin = gdal.GetDriverByName(pszFormat)
            if( poDriverMin is None ):
                raise Exception('Cannot create gdal driver for output')
            
            pszDstFilenameMin = self.wincsminname
            poDstDSMin = poDriverMin.Create(pszDstFilenameMin, xsize=pCount, ysize=lCount, bands=2, eType=gdal.GDT_Int32)
            poDstDSMin.SetGeoTransform( adfGeoTransform )
            poDstDSMin.SetProjection( pszSRS_WKT )
            
            poBand1Min = poDstDSMin.GetRasterBand(1)
            poBand2Min = poDstDSMin.GetRasterBand(2)
            poBand1Min.SetNoDataValue(nodata_out)
            poBand2Min.SetNoDataValue(nodata_out)
        
        if (self.csmaxxname != ""):
            poDriverMax = gdal.GetDriverByName(pszFormat)
            if( poDriverMax is None ):
                raise Exception('Cannot create gdal driver for output')
            
            pszDstFilenameMax = self.wincsmaxname
            poDstDSMax = poDriverMax.Create(pszDstFilenameMax, xsize=pCount, ysize=lCount, bands=2, eType=gdal.GDT_Int32)
            poDstDSMax.SetGeoTransform( adfGeoTransform )
            poDstDSMax.SetProjection( pszSRS_WKT )
            
            poBand1Max = poDstDSMax.GetRasterBand(1)
            poBand2Max = poDstDSMax.GetRasterBand(2)
            poBand1Max.SetNoDataValue(nodata_out)
            poBand2Max.SetNoDataValue(nodata_out)


        if (self.ssmname != ""):
            poDriverMsk = gdal.GetDriverByName(pszFormat)
            if( poDriverMsk is None ):
                raise Exception('Cannot create gdal driver for output')
            
            pszDstFilenameMsk = self.winssmname
            poDstDSMsk = poDriverMsk.Create(pszDstFilenameMsk, xsize=pCount, ysize=lCount, bands=1, eType=gdal.GDT_Int32)
            poDstDSMsk.SetGeoTransform( adfGeoTransform )
            poDstDSMsk.SetProjection( pszSRS_WKT )
            
            poBand1Msk = poDstDSMsk.GetRasterBand(1)
            poBand1Msk.SetNoDataValue(nodata_out)




        if (self.dhdxname != ""):
            poDriverRO2VX = gdal.GetDriverByName(pszFormat)
            if( poDriverRO2VX is None ):
                raise Exception('Cannot create gdal driver for output')
            
            pszDstFilenameRO2VX = self.winro2vxname
            poDstDSRO2VX = poDriverRO2VX.Create(pszDstFilenameRO2VX, xsize=pCount, ysize=lCount, bands=2, eType=gdal.GDT_Float64)
            poDstDSRO2VX.SetGeoTransform( adfGeoTransform )
            poDstDSRO2VX.SetProjection( pszSRS_WKT )
            
            poBand1RO2VX = poDstDSRO2VX.GetRasterBand(1)
            poBand2RO2VX = poDstDSRO2VX.GetRasterBand(2)
            poBand1RO2VX.SetNoDataValue(nodata_out)
            poBand2RO2VX.SetNoDataValue(nodata_out)


            poDriverRO2VY = gdal.GetDriverByName(pszFormat)
            if( poDriverRO2VY is None ):
                raise Exception('Cannot create gdal driver for output')
            
            pszDstFilenameRO2VY = self.winro2vyname
            poDstDSRO2VY = poDriverRO2VY.Create(pszDstFilenameRO2VY, xsize=pCount, ysize=lCount, bands=2, eType=gdal.GDT_Float64)
            poDstDSRO2VY.SetGeoTransform( adfGeoTransform )
            poDstDSRO2VY.SetProjection( pszSRS_WKT )
            
            poBand1RO2VY = poDstDSRO2VY.GetRasterBand(1)
            poBand2RO2VY = poDstDSRO2VY.GetRasterBand(2)
            poBand1RO2VY.SetNoDataValue(nodata_out)
            poBand2RO2VY.SetNoDataValue(nodata_out)



        raster1 = np.zeros(pCount,dtype=np.int32)
        raster2 = np.zeros(pCount,dtype=np.int32)
        raster11 = np.zeros(pCount,dtype=np.int32)
        raster22 = np.zeros(pCount,dtype=np.int32)
        sr_raster11 = np.zeros(pCount,dtype=np.int32)
        sr_raster22 = np.zeros(pCount,dtype=np.int32)
        csmin_raster11 = np.zeros(pCount,dtype=np.int32)
        csmin_raster22 = np.zeros(pCount,dtype=np.int32)
        csmax_raster11 = np.zeros(pCount,dtype=np.int32)
        csmax_raster22 = np.zeros(pCount,dtype=np.int32)
        ssm_raster = np.zeros(pCount,dtype=np.int32)
        raster1a = np.zeros(pCount,dtype=np.float64)
        raster1b = np.zeros(pCount,dtype=np.float64)
        raster2a = np.zeros(pCount,dtype=np.float64)
        raster2b = np.zeros(pCount,dtype=np.float64)

        
        
        #   X- and Y-direction pixel size
        X_res = np.abs(self.XSize)
        Y_res = np.abs(self.YSize)
        print("X-direction pixel size: " + str(X_res))
        print("Y-direction pixel size: " + str(Y_res))
        
        ChipSizeX0_PIX_X = np.ceil(self.chipSizeX0 / X_res / 4) * 4
        ChipSizeX0_PIX_Y = np.ceil(self.chipSizeX0 / Y_res / 4) * 4
        
        



        for ii in range(lCount):
            y = geoTrans[3] + (lOff+ii+0.5) * geoTrans[5]
            demLine = demDS.GetRasterBand(1).ReadRaster(xoff=pOff, yoff=lOff+ii, xsize=pCount, ysize=1, buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Float64)
            demLine = struct.unpack('d' * pCount, demLine)
            
            if (self.dhdxname != ""):
                sxLine = sxDS.GetRasterBand(1).ReadRaster(xoff=pOff, yoff=lOff+ii, xsize=pCount, ysize=1, buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Float64)
                sxLine = struct.unpack('d' * pCount, sxLine)
                syLine = syDS.GetRasterBand(1).ReadRaster(xoff=pOff, yoff=lOff+ii, xsize=pCount, ysize=1, buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Float64)
                syLine = struct.unpack('d' * pCount, syLine)
            
            if (self.vxname != ""):
                vxLine = vxDS.GetRasterBand(1).ReadRaster(xoff=pOff, yoff=lOff+ii, xsize=pCount, ysize=1, buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Float64)
                vxLine = struct.unpack('d' * pCount, vxLine)
                vyLine = vyDS.GetRasterBand(1).ReadRaster(xoff=pOff, yoff=lOff+ii, xsize=pCount, ysize=1, buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Float64)
                vyLine = struct.unpack('d' * pCount, vyLine)
            
            if (self.srxname != ""):
                srxLine = srxDS.GetRasterBand(1).ReadRaster(xoff=pOff, yoff=lOff+ii, xsize=pCount, ysize=1, buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Float64)
                srxLine = struct.unpack('d' * pCount, srxLine)
                sryLine = sryDS.GetRasterBand(1).ReadRaster(xoff=pOff, yoff=lOff+ii, xsize=pCount, ysize=1, buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Float64)
                sryLine = struct.unpack('d' * pCount, sryLine)
            
            if (self.csminxname != ""):
                csminxLine = csminxDS.GetRasterBand(1).ReadRaster(xoff=pOff, yoff=lOff+ii, xsize=pCount, ysize=1, buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Float64)
                csminxLine = struct.unpack('d' * pCount, csminxLine)
                csminyLine = csminyDS.GetRasterBand(1).ReadRaster(xoff=pOff, yoff=lOff+ii, xsize=pCount, ysize=1, buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Float64)
                csminyLine = struct.unpack('d' * pCount, csminyLine)
            
            if (self.csmaxxname != ""):
                csmaxxLine = csmaxxDS.GetRasterBand(1).ReadRaster(xoff=pOff, yoff=lOff+ii, xsize=pCount, ysize=1, buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Float64)
                csmaxxLine = struct.unpack('d' * pCount, csmaxxLine)
                csmaxyLine = csmaxyDS.GetRasterBand(1).ReadRaster(xoff=pOff, yoff=lOff+ii, xsize=pCount, ysize=1, buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Float64)
                csmaxyLine = struct.unpack('d' * pCount, csmaxyLine)

            if (self.ssmname != ""):
                ssmLine = ssmDS.GetRasterBand(1).ReadRaster(xoff=pOff, yoff=lOff+ii, xsize=pCount, ysize=1, buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Float64)
                ssmLine = struct.unpack('d' * pCount, ssmLine)

            for jj in range(pCount):
                xyzs = np.array([geoTrans[0] + (jj+pOff+0.5)*geoTrans[1], y, demLine[jj]])
                targxyz0 = xyzs.copy()
                if (self.dhdxname != ""):
                    slp = np.array([sxLine[jj], syLine[jj], -1.0])
                if (self.vxname != ""):
                    vel = np.array([vxLine[jj], vyLine[jj], 0.0])
                else:
                    vel = np.array([0., 0., 0.])
                if (self.srxname != ""):
                    schrng1 = np.array([srxLine[jj], sryLine[jj], 0.0])
                    schrng2 = np.array([-srxLine[jj], sryLine[jj], 0.0])
                targutm0 = np.array(fwdTrans.TransformPoint(targxyz0[0],targxyz0[1],targxyz0[2]))
                xind = np.round((targutm0[0] - self.startingX) / self.XSize) + 1.
                yind = np.round((targutm0[1] - self.startingY) / self.YSize) + 1.
                
                #   x-direction vector
                targutm = targutm0.copy()
                targutm[0] = targutm0[0] + self.XSize
                targxyz = np.array(invTrans.TransformPoint(targutm[0],targutm[1],targutm[2]))
                xunit = (targxyz-targxyz0) / np.linalg.norm(targxyz-targxyz0)
                    
                #   y-direction vector
                targutm = targutm0.copy()
                targutm[1] = targutm0[1] + self.YSize
                targxyz = np.array(invTrans.TransformPoint(targutm[0],targutm[1],targutm[2]))
                yunit = (targxyz-targxyz0) / np.linalg.norm(targxyz-targxyz0)

                #   local normal vector
                if (self.dhdxname != ""):
                    normal = -slp / np.linalg.norm(slp)
                else:
                    normal = np.array([0., 0., 0.])

                if (self.vxname != ""):
                    vel[2] = -(vel[0]*normal[0]+vel[1]*normal[1])/normal[2]

                if (self.srxname != ""):
                    schrng1[2] = -(schrng1[0]*normal[0]+schrng1[1]*normal[1])/normal[2]
                    schrng2[2] = -(schrng2[0]*normal[0]+schrng2[1]*normal[1])/normal[2]
                    


                if ((xind > self.numberOfSamples)|(xind < 1)|(yind > self.numberOfLines)|(yind < 1)):
#                    pdb.set_trace()
                    raster1[jj] = nodata_out
                    raster2[jj] = nodata_out
                    raster11[jj] = nodata_out
                    raster22[jj] = nodata_out
                    
                    sr_raster11[jj] = nodata_out
                    sr_raster22[jj] = nodata_out
                    csmin_raster11[jj] = nodata_out
                    csmin_raster22[jj] = nodata_out
                    csmax_raster11[jj] = nodata_out
                    csmax_raster22[jj] = nodata_out
                    ssm_raster[jj] = nodata_out
                    
                    raster1a[jj] = nodata_out
                    raster1b[jj] = nodata_out
                    raster2a[jj] = nodata_out
                    raster2b[jj] = nodata_out
                else:
                    raster1[jj] = xind;
                    raster2[jj] = yind;
#                    pdb.set_trace()
#                    if ((self.vxname != "")&(vel[0] != nodata)):
##                        pdb.set_trace()
#                        raster11[jj] = np.round(np.dot(vel,xunit)*self.repeatTime/self.XSize/365.0/24.0/3600.0*1)
#                        raster22[jj] = np.round(np.dot(vel,yunit)*self.repeatTime/self.YSize/365.0/24.0/3600.0*1)
#                    else:
#                        raster11[jj] = 0.
#                        raster22[jj] = 0.
                    if (self.dhdxname != ""):

                        if (self.vxname != ""):
                            if (vel[0] == nodata):
                                raster11[jj] = 0.
                                raster22[jj] = 0.
                            else:
                                raster11[jj] = np.round(np.dot(vel,xunit)*self.repeatTime/self.XSize/365.0/24.0/3600.0*1)
                                raster22[jj] = np.round(np.dot(vel,yunit)*self.repeatTime/self.YSize/365.0/24.0/3600.0*1)

                        cross = np.cross(xunit,yunit)
                        cross = cross / np.linalg.norm(cross)
                        cross_check = np.abs(np.arccos(np.dot(normal,cross))/np.pi*180.0-90.0)
                        
                        if (cross_check > 1.0):
                            raster1a[jj] = normal[2]/(self.repeatTime/self.XSize/365.0/24.0/3600.0)*(normal[2]*yunit[1]-normal[1]*yunit[2])/((normal[2]*xunit[0]-normal[0]*xunit[2])*(normal[2]*yunit[1]-normal[1]*yunit[2])-(normal[2]*yunit[0]-normal[0]*yunit[2])*(normal[2]*xunit[1]-normal[1]*xunit[2]));
                            raster1b[jj] = -normal[2]/(self.repeatTime/self.YSize/365.0/24.0/3600.0)*(normal[2]*xunit[1]-normal[1]*xunit[2])/((normal[2]*xunit[0]-normal[0]*xunit[2])*(normal[2]*yunit[1]-normal[1]*yunit[2])-(normal[2]*yunit[0]-normal[0]*yunit[2])*(normal[2]*xunit[1]-normal[1]*xunit[2]));
                            raster2a[jj] = -normal[2]/(self.repeatTime/self.XSize/365.0/24.0/3600.0)*(normal[2]*yunit[0]-normal[0]*yunit[2])/((normal[2]*xunit[0]-normal[0]*xunit[2])*(normal[2]*yunit[1]-normal[1]*yunit[2])-(normal[2]*yunit[0]-normal[0]*yunit[2])*(normal[2]*xunit[1]-normal[1]*xunit[2]));
                            raster2b[jj] = normal[2]/(self.repeatTime/self.YSize/365.0/24.0/3600.0)*(normal[2]*xunit[0]-normal[0]*xunit[2])/((normal[2]*xunit[0]-normal[0]*xunit[2])*(normal[2]*yunit[1]-normal[1]*yunit[2])-(normal[2]*yunit[0]-normal[0]*yunit[2])*(normal[2]*xunit[1]-normal[1]*xunit[2]));
                        else:
                            raster1a[jj] = nodata_out
                            raster1b[jj] = nodata_out
                            raster2a[jj] = nodata_out
                            raster2b[jj] = nodata_out

                        if (self.srxname != ""):
                            if ((self.vxname != "")&(vel[0] == nodata)):
                                sr_raster11[jj] = 0
                                sr_raster22[jj] = 0
                            else:
                                sr_raster11[jj] = np.abs(np.round(np.dot(schrng1,xunit)*self.repeatTime/self.XSize/365.0/24.0/3600.0*1))
                                sr_raster22[jj] = np.abs(np.round(np.dot(schrng1,yunit)*self.repeatTime/self.YSize/365.0/24.0/3600.0*1))
                                if (np.abs(np.round(np.dot(schrng2,xunit)*self.repeatTime/self.XSize/365.0/24.0/3600.0*1)) > sr_raster11[jj]):
                                    sr_raster11[jj] = np.abs(np.round(np.dot(schrng2,xunit)*self.repeatTime/self.XSize/365.0/24.0/3600.0*1))
                                if (np.abs(np.round(np.dot(schrng2,yunit)*self.repeatTime/self.YSize/365.0/24.0/3600.0*1)) > sr_raster22[jj]):
                                    sr_raster22[jj] = np.abs(np.round(np.dot(schrng2,yunit)*self.repeatTime/self.YSize/365.0/24.0/3600.0*1))
                                if (sr_raster11[jj] == 0):
                                    sr_raster11[jj] = 1
                                if (sr_raster22[jj] == 0):
                                    sr_raster22[jj] = 1

                    if (self.csminxname != ""):
                        csmin_raster11[jj] = csminxLine[jj] / self.chipSizeX0 * ChipSizeX0_PIX_X
                        csmin_raster22[jj] = csminyLine[jj] / self.chipSizeX0 * ChipSizeX0_PIX_Y


                    if (self.csmaxxname != ""):
                        csmax_raster11[jj] = csmaxxLine[jj] / self.chipSizeX0 * ChipSizeX0_PIX_X
                        csmax_raster22[jj] = csmaxyLine[jj] / self.chipSizeX0 * ChipSizeX0_PIX_Y



                    if (self.ssmname != ""):
                        ssm_raster[jj] = ssmLine[jj]
                    



                    

#            pdb.set_trace()
            
            poBand1.WriteRaster(xoff=0, yoff=ii, xsize=pCount, ysize=1, buf_len=raster1.tostring(), buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Int32)
            poBand2.WriteRaster(xoff=0, yoff=ii, xsize=pCount, ysize=1, buf_len=raster2.tostring(), buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Int32)
            if ((self.dhdxname != "")&(self.vxname != "")):
                poBand1Off.WriteRaster(xoff=0, yoff=ii, xsize=pCount, ysize=1, buf_len=raster11.tostring(), buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Int32)
                poBand2Off.WriteRaster(xoff=0, yoff=ii, xsize=pCount, ysize=1, buf_len=raster22.tostring(), buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Int32)
            if ((self.dhdxname != "")&(self.srxname != "")):
                poBand1Sch.WriteRaster(xoff=0, yoff=ii, xsize=pCount, ysize=1, buf_len=sr_raster11.tostring(), buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Int32)
                poBand2Sch.WriteRaster(xoff=0, yoff=ii, xsize=pCount, ysize=1, buf_len=sr_raster22.tostring(), buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Int32)
            if (self.csminxname != ""):
                poBand1Min.WriteRaster(xoff=0, yoff=ii, xsize=pCount, ysize=1, buf_len=csmin_raster11.tostring(), buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Int32)
                poBand2Min.WriteRaster(xoff=0, yoff=ii, xsize=pCount, ysize=1, buf_len=csmin_raster22.tostring(), buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Int32)
            if (self.csmaxxname != ""):
                poBand1Max.WriteRaster(xoff=0, yoff=ii, xsize=pCount, ysize=1, buf_len=csmax_raster11.tostring(), buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Int32)
                poBand2Max.WriteRaster(xoff=0, yoff=ii, xsize=pCount, ysize=1, buf_len=csmax_raster22.tostring(), buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Int32)
            if (self.ssmname != ""):
                poBand1Msk.WriteRaster(xoff=0, yoff=ii, xsize=pCount, ysize=1, buf_len=ssm_raster.tostring(), buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Int32)
            if (self.dhdxname != ""):
                poBand1RO2VX.WriteRaster(xoff=0, yoff=ii, xsize=pCount, ysize=1, buf_len=raster1a.tostring(), buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Float64)
                poBand2RO2VX.WriteRaster(xoff=0, yoff=ii, xsize=pCount, ysize=1, buf_len=raster1b.tostring(), buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Float64)
                poBand1RO2VY.WriteRaster(xoff=0, yoff=ii, xsize=pCount, ysize=1, buf_len=raster2a.tostring(), buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Float64)
                poBand2RO2VY.WriteRaster(xoff=0, yoff=ii, xsize=pCount, ysize=1, buf_len=raster2b.tostring(), buf_xsize=pCount, buf_ysize=1, buf_type=gdal.GDT_Float64)

        
        poDstDS = None
        if ((self.dhdxname != "")&(self.vxname != "")):
            poDstDSOff = None
        if ((self.dhdxname != "")&(self.srxname != "")):
            poDstDSSch = None
        if (self.csminxname != ""):
            poDstDSMin = None
        if (self.csmaxxname != ""):
            poDstDSMax = None
        if (self.ssmname != ""):
            poDstDSMsk = None
        if (self.dhdxname != ""):
            poDstDSRO2VX = None
        
            poDstDSRO2VY = None
    
        demDS = None
        
        if (self.dhdxname != ""):
            sxDS = None
            syDS = None
        
        if (self.vxname != ""):
            vxDS = None
            vyDS = None
                
        if (self.srxname != ""):
            srxDS = None
            sryDS = None
    
        if (self.csminxname != ""):
            csminxDS = None
            csminyDS = None

        if (self.csmaxxname != ""):
            csmaxxDS = None
            csmaxyDS = None
                
        if (self.ssmname != ""):
            ssmDS = None
    

            



    def coregister(self,in1,in2,urlflag):
        import os
        import numpy as np
        
        from osgeo import gdal, osr
        import struct
        
        if urlflag == 1:
            DS1 = gdal.Open('/vsicurl/%s' %(in1))
        else:
            DS1 = gdal.Open(in1, gdal.GA_ReadOnly)
        trans1 = DS1.GetGeoTransform()
        xsize1 = DS1.RasterXSize
        ysize1 = DS1.RasterYSize
        
        if urlflag == 1:
            DS2 = gdal.Open('/vsicurl/%s' %(in2))
        else:
            DS2 = gdal.Open(in2, gdal.GA_ReadOnly)
        trans2 = DS2.GetGeoTransform()
        xsize2 = DS2.RasterXSize
        ysize2 = DS2.RasterYSize
        
        W = np.max([trans1[0],trans2[0]])
        N = np.min([trans1[3],trans2[3]])
        E = np.min([trans1[0]+xsize1*trans1[1],trans2[0]+xsize2*trans2[1]])
        S = np.max([trans1[3]+ysize1*trans1[5],trans2[3]+ysize2*trans2[5]])
        
        x1a = int(np.round((W-trans1[0])/trans1[1]))
        x1b = int(np.round((E-trans1[0])/trans1[1]))
        y1a = int(np.round((N-trans1[3])/trans1[5]))
        y1b = int(np.round((S-trans1[3])/trans1[5]))
        
        x2a = int(np.round((W-trans2[0])/trans2[1]))
        x2b = int(np.round((E-trans2[0])/trans2[1]))
        y2a = int(np.round((N-trans2[3])/trans2[5]))
        y2b = int(np.round((S-trans2[3])/trans2[5]))
        
        x1a = np.min([x1a, xsize1-1])
        x1b = np.min([x1b, xsize1-1])
        y1a = np.min([y1a, ysize1-1])
        y1b = np.min([y1b, ysize1-1])
        x2a = np.min([x2a, xsize2-1])
        x2b = np.min([x2b, xsize2-1])
        y2a = np.min([y2a, ysize2-1])
        y2b = np.min([y2b, ysize2-1])
        
        x1a = np.max([x1a, 0])
        x1b = np.max([x1b, 0])
        y1a = np.max([y1a, 0])
        y1b = np.max([y1b, 0])
        x2a = np.max([x2a, 0])
        x2b = np.max([x2b, 0])
        y2a = np.max([y2a, 0])
        y2b = np.max([y2b, 0])
        
        x1a = int(x1a)
        x1b = int(x1b)
        y1a = int(y1a)
        y1b = int(y1b)
        x2a = int(x2a)
        x2b = int(x2b)
        y2a = int(y2a)
        y2b = int(y2b)

        trans = (W, trans1[1], 0.0, N, 0.0, trans1[5])

        if urlflag == 0:
            
            I1 = DS1.ReadAsArray(xoff=x1a, yoff=y1a, xsize=x1b-x1a+1, ysize=y1b-y1a+1)
            I2 = DS2.ReadAsArray(xoff=x2a, yoff=y2a, xsize=x2b-x2a+1, ysize=y2b-y2a+1)

            fileformat = "GTiff"
            driver = gdal.GetDriverByName(fileformat)
            
            DST1 = driver.Create(os.path.basename(in1), xsize=(x1b-x1a+1), ysize=(y1b-y1a+1), bands=1, eType=gdal.GDT_UInt16)
            DST1.SetGeoTransform(trans)
            DST1.SetProjection(DS1.GetProjectionRef())
            DST1.GetRasterBand(1).WriteArray(I1)
            DST1 = None
            
            DST2 = driver.Create(os.path.basename(in2), xsize=(x2b-x2a+1), ysize=(y2b-y2a+1), bands=1, eType=gdal.GDT_UInt16)
            DST2.SetGeoTransform(trans)
            DST2.SetProjection(DS2.GetProjectionRef())
            DST2.GetRasterBand(1).WriteArray(I2)
            DST2 = None
        
        return x1a, y1a, x1b-x1a+1, y1b-y1a+1, x2a, y2a, x2b-x2a+1, y2b-y2a+1, trans




    
        
            

    def __init__(self):
        super(GeogridOptical, self).__init__()

        ##Optical image related parameters
        self.startingY = None
        self.startingX = None
        self.XSize = None
        self.YSize = None
        self.numberOfSamples = None
        self.numberOfLines = None
        self.repeatTime = None
        self.chipSizeX0 = None
        self.urlflag = None

        ##Input related parameters
        self.dat1name = None
        self.demname = None
        self.dhdxname = None
        self.dhdyname = None
        self.vxname = None
        self.vyname = None
        self.srxname = None
        self.sryname = None
        self.csminxname = None
        self.csminyname = None
        self.csmaxxname = None
        self.csmaxyname = None
        self.ssmname = None
        
        ##Output related parameters
        self.winlocname = None
        self.winoffname = None
        self.winsrname = None
        self.wincsminname = None
        self.wincsmaxname = None
        self.winssmname = None
        self.winro2vxname = None
        self.winro2vyname = None

        ##Coordinate system
        self.epsgDem = None
        self.epsgDat = None
        self._xlim = None
        self._ylim = None
        self.nodata_out = None


