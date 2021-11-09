/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright 2019 California Institute of Technology. ALL RIGHTS RESERVED.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * United States Government Sponsorship acknowledged. This software is subject to
 * U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
 * (No [Export] License Required except when exporting to an embargoed country,
 * end user, or in support of a prohibited end use). By downloading this software,
 * the user agrees to comply with all applicable U.S. export laws and regulations.
 * The user has the responsibility to obtain export licenses, or other export
 * authority as may be required before exporting this software to any 'EAR99'
 * embargoed foreign country or citizen of those countries.
 *
 * Authors: Piyush Agram, Yang Lei
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

#include "geogridOptical.h"
#include <gdal.h>
#include <gdal_priv.h>
#include <iostream>
#include <complex>
#include <cmath>




void geoGridOptical::geogridOptical()
{
    //Some constants 
    double deg2rad = M_PI/180.0;

    //For now print inputs that were obtained

    std::cout << "\nOptical Image parameters: \n";
    std::cout << "X-direction coordinate: " << startingX << "  " << XSize << "\n";
    std::cout << "Y-direction coordinate: " << startingY << "  " << YSize << "\n";
    std::cout << "Dimensions: " << nPixels << " " << nLines << "\n";

    std::cout << "\nMap inputs: \n";
    std::cout << "EPSG: " << epsgDem << "\n";
    std::cout << "Smallest Allowable Chip Size in m: " << chipSizeX0 << "\n";
    std::cout << "Grid spacing in m: " << gridSpacingX << "\n";
    std::cout << "Repeat Time: " << dt << "\n";
    std::cout << "XLimits: " << xmin << "  " << xmax << "\n";
    std::cout << "YLimits: " << ymin << "  " << ymax << "\n";
    std::cout << "Extent in km: " << (xmax - xmin)/1000. << "  " << (ymax - ymin)/1000. << "\n";
    if (demname != "")
    {
        std::cout << "DEM: " << demname << "\n";
    }
    if (dhdxname != "")
    {
        std::cout << "Slopes: " << dhdxname << "  " << dhdyname << "\n";
    }
    if (vxname != "")
    {
        std::cout << "Velocities: " << vxname << "  " << vyname << "\n";
    }
    if (srxname != "")
    {
        std::cout << "Search Range: " << srxname << "  " << sryname << "\n";
    }
    if (csminxname != "")
    {
        std::cout << "Chip Size Min: " << csminxname << "  " << csminyname << "\n";
    }
    if (csmaxxname != "")
    {
        std::cout << "Chip Size Max: " << csmaxxname << "  " << csmaxyname << "\n";
    }
    if (ssmname != "")
    {
        std::cout << "Stable Surface Mask: " << ssmname << "\n";
    }
    
    
    std::cout << "\nOutputs: \n";
    std::cout << "Window locations: " << pixlinename << "\n";
    if (dhdxname != "")
    {
        if (vxname != "")
        {
            std::cout << "Window offsets: " << offsetname << "\n";
        }
        
        std::cout << "Window rdr_off2vel_x vector: " << ro2vx_name << "\n";
        std::cout << "Window rdr_off2vel_y vector: " << ro2vy_name << "\n";
        
        if (srxname != "")
        {
            std::cout << "Window search range: " << searchrangename << "\n";
        }
    }
            
    if (csminxname != "")
    {
        std::cout << "Window chip size min: " << chipsizeminname << "\n";
    }
    if (csmaxxname != "")
    {
        std::cout << "Window chip size max: " << chipsizemaxname << "\n";
    }
    if (ssmname != "")
    {
        std::cout << "Window stable surface mask: " << stablesurfacemaskname << "\n";
    }
    
    std::cout << "Output Nodata Value: " << nodata_out << "\n";
    

    std::cout << "\nStarting processing .... \n";

    //Startup GDAL
    GDALAllRegister();

    //DEM related information
    GDALDataset* demDS = NULL;
    GDALDataset* sxDS = NULL;
    GDALDataset* syDS = NULL;
    GDALDataset* vxDS = NULL;
    GDALDataset* vyDS = NULL;
    GDALDataset* srxDS = NULL;
    GDALDataset* sryDS = NULL;
    GDALDataset* csminxDS = NULL;
    GDALDataset* csminyDS = NULL;
    GDALDataset* csmaxxDS = NULL;
    GDALDataset* csmaxyDS = NULL;
    GDALDataset* ssmDS = NULL;

    double geoTrans[6];

    demDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(demname.c_str(), GA_ReadOnly));
    if (dhdxname != "")
    {
        sxDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(dhdxname.c_str(), GA_ReadOnly));
        syDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(dhdyname.c_str(), GA_ReadOnly));
    }
    if (vxname != "")
    {
        vxDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(vxname.c_str(), GA_ReadOnly));
        vyDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(vyname.c_str(), GA_ReadOnly));
    }
    if (srxname != "")
    {
        srxDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(srxname.c_str(), GA_ReadOnly));
        sryDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(sryname.c_str(), GA_ReadOnly));
    }
    if (csminxname != "")
    {
        csminxDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(csminxname.c_str(), GA_ReadOnly));
        csminyDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(csminyname.c_str(), GA_ReadOnly));
    }
    if (csmaxxname != "")
    {
        csmaxxDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(csmaxxname.c_str(), GA_ReadOnly));
        csmaxyDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(csmaxyname.c_str(), GA_ReadOnly));
    }
    if (ssmname != "")
    {
        ssmDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(ssmname.c_str(), GA_ReadOnly));
    }
    if (demDS == NULL)
    {
        std::cout << "Error opening DEM file { " << demname << " }\n";
        std::cout << "Exiting with error code .... (101) \n";
        GDALDestroyDriverManager();
        exit(101);
    }
    if (dhdxname != "")
    {
        if (sxDS == NULL)
        {
            std::cout << "Error opening x-direction slope file { " << dhdxname << " }\n";
            std::cout << "Exiting with error code .... (101) \n";
            GDALDestroyDriverManager();
            exit(101);
        }
        if (syDS == NULL)
        {
            std::cout << "Error opening y-direction slope file { " << dhdyname << " }\n";
            std::cout << "Exiting with error code .... (101) \n";
            GDALDestroyDriverManager();
            exit(101);
        }
    }
    if (vxname != "")
    {
        if (vxDS == NULL)
        {
            std::cout << "Error opening x-direction velocity file { " << vxname << " }\n";
            std::cout << "Exiting with error code .... (101) \n";
            GDALDestroyDriverManager();
            exit(101);
        }
        if (vyDS == NULL)
        {
            std::cout << "Error opening y-direction velocity file { " << vyname << " }\n";
            std::cout << "Exiting with error code .... (101) \n";
            GDALDestroyDriverManager();
            exit(101);
        }
    }
    if (srxname != "")
    {
        if (srxDS == NULL)
        {
            std::cout << "Error opening x-direction search range file { " << srxname << " }\n";
            std::cout << "Exiting with error code .... (101) \n";
            GDALDestroyDriverManager();
            exit(101);
        }
        if (sryDS == NULL)
        {
            std::cout << "Error opening y-direction search range file { " << sryname << " }\n";
            std::cout << "Exiting with error code .... (101) \n";
            GDALDestroyDriverManager();
            exit(101);
        }
    }
    if (csminxname != "")
    {
        if (csminxDS == NULL)
        {
            std::cout << "Error opening x-direction chip size min file { " << csminxname << " }\n";
            std::cout << "Exiting with error code .... (101) \n";
            GDALDestroyDriverManager();
            exit(101);
        }
        if (csminyDS == NULL)
        {
            std::cout << "Error opening y-direction chip size min file { " << csminyname << " }\n";
            std::cout << "Exiting with error code .... (101) \n";
            GDALDestroyDriverManager();
            exit(101);
        }
    }
    if (csmaxxname != "")
    {
        if (csmaxxDS == NULL)
        {
            std::cout << "Error opening x-direction chip size max file { " << csmaxxname << " }\n";
            std::cout << "Exiting with error code .... (101) \n";
            GDALDestroyDriverManager();
            exit(101);
        }
        if (csmaxyDS == NULL)
        {
            std::cout << "Error opening y-direction chip size max file { " << csmaxyname << " }\n";
            std::cout << "Exiting with error code .... (101) \n";
            GDALDestroyDriverManager();
            exit(101);
        }
    }
    if (ssmname != "")
    {
        if (ssmDS == NULL)
        {
            std::cout << "Error opening stable surface mask file { " << ssmname << " }\n";
            std::cout << "Exiting with error code .... (101) \n";
            GDALDestroyDriverManager();
            exit(101);
        }
    }

    demDS->GetGeoTransform(geoTrans);
    int demXSize = demDS->GetRasterXSize();
    int demYSize = demDS->GetRasterYSize();


    //Get offsets and size to read from DEM
//    int lOff = std::max( std::floor((ymax - geoTrans[3])/geoTrans[5]), 0.);
//    int lCount = std::min( std::ceil((ymin - geoTrans[3])/geoTrans[5]), demYSize-1.) - lOff;
//
//    int pOff = std::max( std::floor((xmin - geoTrans[0])/geoTrans[1]), 0.);
//    int pCount = std::min( std::ceil((xmax - geoTrans[0])/geoTrans[1]), demXSize-1.) - pOff;
    lOff = std::max( std::floor((ymax - geoTrans[3])/geoTrans[5]), 0.);
    lCount = std::min( std::ceil((ymin - geoTrans[3])/geoTrans[5]), demYSize-1.) - lOff;
    
    pOff = std::max( std::floor((xmin - geoTrans[0])/geoTrans[1]), 0.);
    pCount = std::min( std::ceil((xmax - geoTrans[0])/geoTrans[1]), demXSize-1.) - pOff;


    std::cout << "Xlimits : " << geoTrans[0] + pOff * geoTrans[1] <<  "  " 
                             << geoTrans[0] + (pOff + pCount) * geoTrans[1] << "\n";


    std::cout << "Ylimits : " << geoTrans[3] + (lOff + lCount) * geoTrans[5] <<  "  "
                             << geoTrans[3] + lOff * geoTrans[5] << "\n";

    std::cout << "Origin index (in DEM) of geogrid: " << pOff << "   " << lOff << "\n";
    
    std::cout << "Dimensions of geogrid: " << pCount << " x " << lCount << "\n";


    //Create GDAL Transformers 
    OGRSpatialReference demSRS(nullptr);
    if (demSRS.importFromEPSG(epsgDem) != 0)
    {
        std::cout << "Could not create OGR spatial reference for DEM EPSG code: " << epsgDem << "\n";
        GDALClose(demDS);
        GDALDestroyDriverManager();
        exit(102);
    }

    OGRSpatialReference datSRS(nullptr);
    if (datSRS.importFromEPSG(epsgDat) != 0)
    {
        std::cout << "Could not create OGR spatil reference for Data EPSG code: " << epsgDat << "\n";
        exit(103);
    }

    OGRCoordinateTransformation *fwdTrans = OGRCreateCoordinateTransformation( &demSRS, &datSRS);
    OGRCoordinateTransformation *invTrans = OGRCreateCoordinateTransformation( &datSRS, &demSRS);

    

    std::vector<double> demLine(pCount);
    std::vector<double> sxLine(pCount);
    std::vector<double> syLine(pCount);
    std::vector<double> vxLine(pCount);
    std::vector<double> vyLine(pCount);
    std::vector<double> srxLine(pCount);
    std::vector<double> sryLine(pCount);
    std::vector<double> csminxLine(pCount);
    std::vector<double> csminyLine(pCount);
    std::vector<double> csmaxxLine(pCount);
    std::vector<double> csmaxyLine(pCount);
    std::vector<double> ssmLine(pCount);
    
    GInt32 raster1[pCount];
    GInt32 raster2[pCount];
    GInt32 raster11[pCount];
    GInt32 raster22[pCount];
    
    GInt32 sr_raster11[pCount];
    GInt32 sr_raster22[pCount];
    GInt32 csmin_raster11[pCount];
    GInt32 csmin_raster22[pCount];
    GInt32 csmax_raster11[pCount];
    GInt32 csmax_raster22[pCount];
    GInt32 ssm_raster[pCount];
    
    double raster1a[pCount];
    double raster1b[pCount];
//    double raster1c[pCount];
    
    double raster2a[pCount];
    double raster2b[pCount];
//    double raster2c[pCount];
    
    GDALRasterBand *poBand1 = NULL;
    GDALRasterBand *poBand2 = NULL;
    GDALRasterBand *poBand1Off = NULL;
    GDALRasterBand *poBand2Off = NULL;
    GDALRasterBand *poBand1Sch = NULL;
    GDALRasterBand *poBand2Sch = NULL;
    GDALRasterBand *poBand1Min = NULL;
    GDALRasterBand *poBand2Min = NULL;
    GDALRasterBand *poBand1Max = NULL;
    GDALRasterBand *poBand2Max = NULL;
    GDALRasterBand *poBand1Msk = NULL;
    GDALRasterBand *poBand1RO2VX = NULL;
    GDALRasterBand *poBand1RO2VY = NULL;
    GDALRasterBand *poBand2RO2VX = NULL;
    GDALRasterBand *poBand2RO2VY = NULL;
    
    GDALDataset *poDstDS = NULL;
    GDALDataset *poDstDSOff = NULL;
    GDALDataset *poDstDSSch = NULL;
    GDALDataset *poDstDSMin = NULL;
    GDALDataset *poDstDSMax = NULL;
    GDALDataset *poDstDSMsk = NULL;
    GDALDataset *poDstDSRO2VX = NULL;
    GDALDataset *poDstDSRO2VY = NULL;
    

    double nodata;
    int* pbSuccess = NULL;
    nodata = demDS->GetRasterBand(1)->GetNoDataValue(pbSuccess);
    
    
    const char *pszFormat = "GTiff";
    char **papszOptions = NULL;
    std::string str = "";
    double adfGeoTransform[6] = { geoTrans[0] + pOff * geoTrans[1], geoTrans[1], 0, geoTrans[3] + lOff * geoTrans[5], 0, geoTrans[5]};
    OGRSpatialReference oSRS;
    char *pszSRS_WKT = NULL;
    demSRS.exportToWkt( &pszSRS_WKT );
    
    
    
    GDALDriver *poDriver;
    poDriver = GetGDALDriverManager()->GetDriverByName(pszFormat);
    if( poDriver == NULL )
    exit(107);
//    GDALDataset *poDstDS;
    
    str = pixlinename;
    const char * pszDstFilename = str.c_str();
    poDstDS = poDriver->Create( pszDstFilename, pCount, lCount, 2, GDT_Int32,
                               papszOptions );
    
    
    poDstDS->SetGeoTransform( adfGeoTransform );
    poDstDS->SetProjection( pszSRS_WKT );
//    CPLFree( pszSRS_WKT );
    
    
//    GDALRasterBand *poBand1;
//    GDALRasterBand *poBand2;
    poBand1 = poDstDS->GetRasterBand(1);
    poBand2 = poDstDS->GetRasterBand(2);
    poBand1->SetNoDataValue(nodata_out);
    poBand2->SetNoDataValue(nodata_out);
    
    
    if ((dhdxname != "")&(vxname != ""))
    {

        GDALDriver *poDriverOff;
        poDriverOff = GetGDALDriverManager()->GetDriverByName(pszFormat);
        if( poDriverOff == NULL )
        exit(107);
//        GDALDataset *poDstDSOff;
    
        str = offsetname;
        const char * pszDstFilenameOff = str.c_str();
        poDstDSOff = poDriverOff->Create( pszDstFilenameOff, pCount, lCount, 2, GDT_Int32,
                                         papszOptions );
    
        poDstDSOff->SetGeoTransform( adfGeoTransform );
        poDstDSOff->SetProjection( pszSRS_WKT );
    //    CPLFree( pszSRS_WKT );
    
//        GDALRasterBand *poBand1Off;
//        GDALRasterBand *poBand2Off;
        poBand1Off = poDstDSOff->GetRasterBand(1);
        poBand2Off = poDstDSOff->GetRasterBand(2);
        poBand1Off->SetNoDataValue(nodata_out);
        poBand2Off->SetNoDataValue(nodata_out);
        
    }
    
    if ((dhdxname != "")&(srxname != ""))
    {
    
        GDALDriver *poDriverSch;
        poDriverSch = GetGDALDriverManager()->GetDriverByName(pszFormat);
        if( poDriverSch == NULL )
        exit(107);
//        GDALDataset *poDstDSSch;
        
        str = searchrangename;
        const char * pszDstFilenameSch = str.c_str();
        poDstDSSch = poDriverSch->Create( pszDstFilenameSch, pCount, lCount, 2, GDT_Int32,
                                         papszOptions );
        
        poDstDSSch->SetGeoTransform( adfGeoTransform );
        poDstDSSch->SetProjection( pszSRS_WKT );
        //    CPLFree( pszSRS_WKT );
        
//        GDALRasterBand *poBand1Sch;
//        GDALRasterBand *poBand2Sch;
        poBand1Sch = poDstDSSch->GetRasterBand(1);
        poBand2Sch = poDstDSSch->GetRasterBand(2);
        poBand1Sch->SetNoDataValue(nodata_out);
        poBand2Sch->SetNoDataValue(nodata_out);
    
    }
    
    if (csminxname != "")
    {
        
        GDALDriver *poDriverMin;
        poDriverMin = GetGDALDriverManager()->GetDriverByName(pszFormat);
        if( poDriverMin == NULL )
        exit(107);
//        GDALDataset *poDstDSMin;
        
        str = chipsizeminname;
        const char * pszDstFilenameMin = str.c_str();
        poDstDSMin = poDriverMin->Create( pszDstFilenameMin, pCount, lCount, 2, GDT_Int32,
                                         papszOptions );
        
        poDstDSMin->SetGeoTransform( adfGeoTransform );
        poDstDSMin->SetProjection( pszSRS_WKT );
        //    CPLFree( pszSRS_WKT );
        
//        GDALRasterBand *poBand1Min;
//        GDALRasterBand *poBand2Min;
        poBand1Min = poDstDSMin->GetRasterBand(1);
        poBand2Min = poDstDSMin->GetRasterBand(2);
        poBand1Min->SetNoDataValue(nodata_out);
        poBand2Min->SetNoDataValue(nodata_out);
        
    }
    
    
    if (csmaxxname != "")
    {
    
        GDALDriver *poDriverMax;
        poDriverMax = GetGDALDriverManager()->GetDriverByName(pszFormat);
        if( poDriverMax == NULL )
        exit(107);
//        GDALDataset *poDstDSMax;
        
        str = chipsizemaxname;
        const char * pszDstFilenameMax = str.c_str();
        poDstDSMax = poDriverMax->Create( pszDstFilenameMax, pCount, lCount, 2, GDT_Int32,
                                         papszOptions );
        
        poDstDSMax->SetGeoTransform( adfGeoTransform );
        poDstDSMax->SetProjection( pszSRS_WKT );
        //    CPLFree( pszSRS_WKT );
        
//        GDALRasterBand *poBand1Max;
//        GDALRasterBand *poBand2Max;
        poBand1Max = poDstDSMax->GetRasterBand(1);
        poBand2Max = poDstDSMax->GetRasterBand(2);
        poBand1Max->SetNoDataValue(nodata_out);
        poBand2Max->SetNoDataValue(nodata_out);
        
    }
    
    
    
    if (ssmname != "")
    {
    
        GDALDriver *poDriverMsk;
        poDriverMsk = GetGDALDriverManager()->GetDriverByName(pszFormat);
        if( poDriverMsk == NULL )
        exit(107);
//        GDALDataset *poDstDSMsk;
        
        str = stablesurfacemaskname;
        const char * pszDstFilenameMsk = str.c_str();
        poDstDSMsk = poDriverMsk->Create( pszDstFilenameMsk, pCount, lCount, 1, GDT_Int32,
                                         papszOptions );
        
        poDstDSMsk->SetGeoTransform( adfGeoTransform );
        poDstDSMsk->SetProjection( pszSRS_WKT );
        //    CPLFree( pszSRS_WKT );
        
//        GDALRasterBand *poBand1Msk;
        poBand1Msk = poDstDSMsk->GetRasterBand(1);
        poBand1Msk->SetNoDataValue(nodata_out);
        
    }
    
    
    if (dhdxname != "")
    {
    
        GDALDriver *poDriverRO2VX;
        poDriverRO2VX = GetGDALDriverManager()->GetDriverByName(pszFormat);
        if( poDriverRO2VX == NULL )
        exit(107);
//        GDALDataset *poDstDSRO2VX;
        
        str = ro2vx_name;
        const char * pszDstFilenameRO2VX = str.c_str();
        poDstDSRO2VX = poDriverRO2VX->Create( pszDstFilenameRO2VX, pCount, lCount, 2, GDT_Float64,
                                         papszOptions );
        
        poDstDSRO2VX->SetGeoTransform( adfGeoTransform );
        poDstDSRO2VX->SetProjection( pszSRS_WKT );
    //    CPLFree( pszSRS_WKT );
        
//        GDALRasterBand *poBand1RO2VX;
//        GDALRasterBand *poBand2RO2VX;
    //    GDALRasterBand *poBand3Los;
        poBand1RO2VX = poDstDSRO2VX->GetRasterBand(1);
        poBand2RO2VX = poDstDSRO2VX->GetRasterBand(2);
    //    poBand3Los = poDstDSLos->GetRasterBand(3);
        poBand1RO2VX->SetNoDataValue(nodata_out);
        poBand2RO2VX->SetNoDataValue(nodata_out);
    //    poBand3Los->SetNoDataValue(nodata_out);
        

        GDALDriver *poDriverRO2VY;
        poDriverRO2VY = GetGDALDriverManager()->GetDriverByName(pszFormat);
        if( poDriverRO2VY == NULL )
        exit(107);
//        GDALDataset *poDstDSRO2VY;
        
        str = ro2vy_name;
        const char * pszDstFilenameRO2VY = str.c_str();
        poDstDSRO2VY = poDriverRO2VY->Create( pszDstFilenameRO2VY, pCount, lCount, 2, GDT_Float64,
                                         papszOptions );
        
        poDstDSRO2VY->SetGeoTransform( adfGeoTransform );
        poDstDSRO2VY->SetProjection( pszSRS_WKT );
//        CPLFree( pszSRS_WKT );
        
//        GDALRasterBand *poBand1RO2VY;
//        GDALRasterBand *poBand2RO2VY;
    //    GDALRasterBand *poBand3Alt;
        poBand1RO2VY = poDstDSRO2VY->GetRasterBand(1);
        poBand2RO2VY = poDstDSRO2VY->GetRasterBand(2);
    //    poBand3Alt = poDstDSAlt->GetRasterBand(3);
        poBand1RO2VY->SetNoDataValue(nodata_out);
        poBand2RO2VY->SetNoDataValue(nodata_out);
    //    poBand3Alt->SetNoDataValue(nodata_out);
        
    }
    
    CPLFree( pszSRS_WKT );

    
    
    
    
    // ground range and azimuth pixel size

    X_res = std::abs(XSize);
    Y_res = std::abs(YSize);
    std::cout << "X-direction pixel size: " << X_res << "\n";
    std::cout << "Y-direction pixel size: " << Y_res << "\n";
//    int ChipSizeX0 = 240;
    double ChipSizeX0 = chipSizeX0;
    int ChipSizeX0_PIX_X = std::ceil(ChipSizeX0 / X_res / 4) * 4;
    int ChipSizeX0_PIX_Y = std::ceil(ChipSizeX0 / Y_res / 4) * 4;
    
    double xind, yind;
    
    for (int ii=0; ii<lCount; ii++)
    {
        double y = geoTrans[3] + (lOff+ii+0.5) * geoTrans[5];
        int status = demDS->GetRasterBand(1)->RasterIO(GF_Read,
                        pOff, lOff+ii,
                        pCount, 1,
                        (void*) (demLine.data()),
                        pCount, 1, GDT_Float64,
                        sizeof(double), sizeof(double)*pCount, NULL);

        if (status != 0)
        {
            std::cout << "Error read line " << lOff + ii << " from DEM file: " << demname << "\n";
            GDALClose(demDS);
            GDALDestroyDriverManager();
            exit(105);
        }
        
        if (dhdxname != "")
        {
            status = sxDS->GetRasterBand(1)->RasterIO(GF_Read,
                                                       pOff, lOff+ii,
                                                       pCount, 1,
                                                       (void*) (sxLine.data()),
                                                       pCount, 1, GDT_Float64,
                                                       sizeof(double), sizeof(double)*pCount, NULL);
            
            if (status != 0)
            {
                std::cout << "Error read line " << lOff + ii << " from x-direction slope file: " << dhdxname << "\n";
                GDALClose(sxDS);
                GDALDestroyDriverManager();
                exit(105);
            }
            
            
            status = syDS->GetRasterBand(1)->RasterIO(GF_Read,
                                                      pOff, lOff+ii,
                                                      pCount, 1,
                                                      (void*) (syLine.data()),
                                                      pCount, 1, GDT_Float64,
                                                      sizeof(double), sizeof(double)*pCount, NULL);
            
            if (status != 0)
            {
                std::cout << "Error read line " << lOff + ii << " from y-direction slope file: " << dhdyname << "\n";
                GDALClose(syDS);
                GDALDestroyDriverManager();
                exit(105);
            }
            
        }
        
        if (vxname != "")
        {
            status = vxDS->GetRasterBand(1)->RasterIO(GF_Read,
                                                      pOff, lOff+ii,
                                                      pCount, 1,
                                                      (void*) (vxLine.data()),
                                                      pCount, 1, GDT_Float64,
                                                      sizeof(double), sizeof(double)*pCount, NULL);
            
            if (status != 0)
            {
                std::cout << "Error read line " << lOff + ii << " from x-direction velocity file: " << vxname << "\n";
                GDALClose(vxDS);
                GDALDestroyDriverManager();
                exit(105);
            }
            
            
            status = vyDS->GetRasterBand(1)->RasterIO(GF_Read,
                                                      pOff, lOff+ii,
                                                      pCount, 1,
                                                      (void*) (vyLine.data()),
                                                      pCount, 1, GDT_Float64,
                                                      sizeof(double), sizeof(double)*pCount, NULL);
            
            if (status != 0)
            {
                std::cout << "Error read line " << lOff + ii << " from y-direction velocity file: " << vyname << "\n";
                GDALClose(vyDS);
                GDALDestroyDriverManager();
                exit(105);
            }
        }
        
        if (srxname != "")
        {
            status = srxDS->GetRasterBand(1)->RasterIO(GF_Read,
                                                      pOff, lOff+ii,
                                                      pCount, 1,
                                                      (void*) (srxLine.data()),
                                                      pCount, 1, GDT_Float64,
                                                      sizeof(double), sizeof(double)*pCount, NULL);
            
            if (status != 0)
            {
                std::cout << "Error read line " << lOff + ii << " from x-direction search range file: " << srxname << "\n";
                GDALClose(srxDS);
                GDALDestroyDriverManager();
                exit(105);
            }
            
            
            status = sryDS->GetRasterBand(1)->RasterIO(GF_Read,
                                                      pOff, lOff+ii,
                                                      pCount, 1,
                                                      (void*) (sryLine.data()),
                                                      pCount, 1, GDT_Float64,
                                                      sizeof(double), sizeof(double)*pCount, NULL);
            
            if (status != 0)
            {
                std::cout << "Error read line " << lOff + ii << " from y-direction search range file: " << sryname << "\n";
                GDALClose(sryDS);
                GDALDestroyDriverManager();
                exit(105);
            }
        }
        
        
        if (csminxname != "")
        {
            status = csminxDS->GetRasterBand(1)->RasterIO(GF_Read,
                                                      pOff, lOff+ii,
                                                      pCount, 1,
                                                      (void*) (csminxLine.data()),
                                                      pCount, 1, GDT_Float64,
                                                      sizeof(double), sizeof(double)*pCount, NULL);
            
            if (status != 0)
            {
                std::cout << "Error read line " << lOff + ii << " from x-direction chip size min file: " << csminxname << "\n";
                GDALClose(csminxDS);
                GDALDestroyDriverManager();
                exit(105);
            }
            
            
            status = csminyDS->GetRasterBand(1)->RasterIO(GF_Read,
                                                      pOff, lOff+ii,
                                                      pCount, 1,
                                                      (void*) (csminyLine.data()),
                                                      pCount, 1, GDT_Float64,
                                                      sizeof(double), sizeof(double)*pCount, NULL);
            
            if (status != 0)
            {
                std::cout << "Error read line " << lOff + ii << " from y-direction chip size min file: " << csminyname << "\n";
                GDALClose(csminyDS);
                GDALDestroyDriverManager();
                exit(105);
            }
        }
        
        
        if (csmaxxname != "")
        {
            status = csmaxxDS->GetRasterBand(1)->RasterIO(GF_Read,
                                                      pOff, lOff+ii,
                                                      pCount, 1,
                                                      (void*) (csmaxxLine.data()),
                                                      pCount, 1, GDT_Float64,
                                                      sizeof(double), sizeof(double)*pCount, NULL);
            
            if (status != 0)
            {
                std::cout << "Error read line " << lOff + ii << " from x-direction chip size max file: " << csmaxxname << "\n";
                GDALClose(csmaxxDS);
                GDALDestroyDriverManager();
                exit(105);
            }
            
            
            status = csmaxyDS->GetRasterBand(1)->RasterIO(GF_Read,
                                                      pOff, lOff+ii,
                                                      pCount, 1,
                                                      (void*) (csmaxyLine.data()),
                                                      pCount, 1, GDT_Float64,
                                                      sizeof(double), sizeof(double)*pCount, NULL);
            
            if (status != 0)
            {
                std::cout << "Error read line " << lOff + ii << " from y-direction chip size max file: " << csmaxyname << "\n";
                GDALClose(csmaxyDS);
                GDALDestroyDriverManager();
                exit(105);
            }
        }
        
        
        
        if (ssmname != "")
        {
            status = ssmDS->GetRasterBand(1)->RasterIO(GF_Read,
                                                          pOff, lOff+ii,
                                                          pCount, 1,
                                                          (void*) (ssmLine.data()),
                                                          pCount, 1, GDT_Float64,
                                                          sizeof(double), sizeof(double)*pCount, NULL);
            
            if (status != 0)
            {
                std::cout << "Error read line " << lOff + ii << " from stable surface mask file: " << ssmname << "\n";
                GDALClose(ssmDS);
                GDALDestroyDriverManager();
                exit(105);
            }
            
        }
        
        
        
    
        
        for (int jj=0; jj<pCount; jj++)
        {
            double xyzs[3];
            
            double targxyz0[3];
            double targutm0[3];
            
            double targutm[3];

            double slp[3];
            double vel[3];
            double normal[3];
            double cross[3];
            double cross_check;
            
            double schrng1[3];
            double schrng2[3];
            
            double xdiff[3], xunit[3];
            double ydiff[3], yunit[3];
            

            //Setup ENU with DEM
            xyzs[0] = geoTrans[0] + (jj+pOff+0.5)*geoTrans[1];
            xyzs[1] = y;
            xyzs[2] = demLine[jj];
            
            for(int pp=0; pp<3; pp++)
            {
                targxyz0[pp] = xyzs[pp];
            }
            
            if (dhdxname != "")
            {
                slp[0] = sxLine[jj];
                slp[1] = syLine[jj];
                slp[2] = -1.0;
            }
            
            if (vxname != "")
            {
                vel[0] = vxLine[jj];
                vel[1] = vyLine[jj];
            }
            
            if (srxname != "")
            {
                schrng1[0] = srxLine[jj];
                schrng1[1] = sryLine[jj];
            
                schrng1[0] *= std::max(max_factor*((dt_unity-1)*max_factor+(max_factor-1)-(max_factor-1)*dt/24.0/3600.0)/((dt_unity-1)*max_factor),1.0);
                schrng1[0] = std::min(std::max(schrng1[0],lower_thld),upper_thld);
                schrng1[1] *= std::max(max_factor*((dt_unity-1)*max_factor+(max_factor-1)-(max_factor-1)*dt/24.0/3600.0)/((dt_unity-1)*max_factor),1.0);
                schrng1[1] = std::min(std::max(schrng1[1],lower_thld),upper_thld);
            
                schrng2[0] = -schrng1[0];
                schrng2[1] = schrng1[1];
            }
            

            //Convert from DEM coordinates to LLH inplace
            fwdTrans->Transform(1, xyzs, xyzs+1, xyzs+2);
            
            for(int pp=0; pp<3; pp++)
            {
                targutm0[pp] = xyzs[pp];
            }
            
            xind = std::round((targutm0[0] - startingX) / XSize) + 0.;
            yind = std::round((targutm0[1] - startingY) / YSize) + 0.;

            
            
            
            
            
            //*********************Slant-range vector
            
            for(int pp=0; pp<3; pp++)
            {
                targutm[pp] = targutm0[pp];
            }
            targutm[0] += XSize;
            
            
            //Convert from LLH inplace to DEM coordinates
            invTrans->Transform(1, targutm, targutm+1, targutm+2);
            
            for(int pp=0; pp<3; pp++)
            {
                xdiff[pp]  = targutm[pp] - targxyz0[pp];
            }
            unitvec_C(xdiff, xunit);
            
            
            
            
            //*********************Along-track vector
            
            for(int pp=0; pp<3; pp++)
            {
                targutm[pp] = targutm0[pp];
            }
            targutm[1] += YSize;
            
            
            //Convert from LLH inplace to DEM coordinates
            invTrans->Transform(1, targutm, targutm+1, targutm+2);
            
            for(int pp=0; pp<3; pp++)
            {
                ydiff[pp]  = targutm[pp] - targxyz0[pp];
            }
            unitvec_C(ydiff, yunit);
            
            
            
            
            //*********************Local normal vector
            if (dhdxname != "")
            {
                unitvec_C(slp, normal);
                for(int pp=0; pp<3; pp++)
                {
                    normal[pp]  = -normal[pp];
                }
            }
            else
            {
                for(int pp=0; pp<3; pp++)
                {
                    normal[pp]  = 0.0;
                }
            }
            
            if (vxname != "")
            {
                vel[2] = -(vel[0]*normal[0]+vel[1]*normal[1])/normal[2];
            }
            
            if (srxname != "")
            {
                schrng1[2] = -(schrng1[0]*normal[0]+schrng1[1]*normal[1])/normal[2];
                schrng2[2] = -(schrng2[0]*normal[0]+schrng2[1]*normal[1])/normal[2];
            }
            
            
            if ((xind > nPixels-1)|(xind < 1-1)|(yind > nLines-1)|(yind < 1-1))
            {
                raster1[jj] = nodata_out;
                raster2[jj] = nodata_out;
                raster11[jj] = nodata_out;
                raster22[jj] = nodata_out;
                
                sr_raster11[jj] = nodata_out;
                sr_raster22[jj] = nodata_out;
                csmin_raster11[jj] = nodata_out;
                csmin_raster22[jj] = nodata_out;
                csmax_raster11[jj] = nodata_out;
                csmax_raster22[jj] = nodata_out;
                ssm_raster[jj] = nodata_out;
                
                raster1a[jj] = nodata_out;
                raster1b[jj] = nodata_out;
//                raster1c[jj] = nodata_out;
                raster2a[jj] = nodata_out;
                raster2b[jj] = nodata_out;
//                raster2c[jj] = nodata_out;
                
            }
            else
            {
                raster1[jj] = xind;
                raster2[jj] = yind;
                
                if (dhdxname != "")
                {
                    
                    if (vxname != "")
                    {
                        if (vel[0] == nodata)
                        {
                            raster11[jj] = 0.;
                            raster22[jj] = 0.;
                        }
                        else
                        {
                            raster11[jj] = std::round(dot_C(vel,xunit)*dt/XSize/365.0/24.0/3600.0*1);
                            raster22[jj] = std::round(dot_C(vel,yunit)*dt/YSize/365.0/24.0/3600.0*1);
                        }
                      
                    }
                    
                    cross_C(xunit,yunit,cross);
                    unitvec_C(cross, cross);
                    cross_check = std::abs(std::acos(dot_C(normal,cross))/deg2rad-90.0);
                    
                    if (cross_check > 1.0)
                    {
                        raster1a[jj] = normal[2]/(dt/XSize/365.0/24.0/3600.0)*(normal[2]*yunit[1]-normal[1]*yunit[2])/((normal[2]*xunit[0]-normal[0]*xunit[2])*(normal[2]*yunit[1]-normal[1]*yunit[2])-(normal[2]*yunit[0]-normal[0]*yunit[2])*(normal[2]*xunit[1]-normal[1]*xunit[2]));
                        raster1b[jj] = -normal[2]/(dt/YSize/365.0/24.0/3600.0)*(normal[2]*xunit[1]-normal[1]*xunit[2])/((normal[2]*xunit[0]-normal[0]*xunit[2])*(normal[2]*yunit[1]-normal[1]*yunit[2])-(normal[2]*yunit[0]-normal[0]*yunit[2])*(normal[2]*xunit[1]-normal[1]*xunit[2]));
                        raster2a[jj] = -normal[2]/(dt/XSize/365.0/24.0/3600.0)*(normal[2]*yunit[0]-normal[0]*yunit[2])/((normal[2]*xunit[0]-normal[0]*xunit[2])*(normal[2]*yunit[1]-normal[1]*yunit[2])-(normal[2]*yunit[0]-normal[0]*yunit[2])*(normal[2]*xunit[1]-normal[1]*xunit[2]));
                        raster2b[jj] = normal[2]/(dt/YSize/365.0/24.0/3600.0)*(normal[2]*xunit[0]-normal[0]*xunit[2])/((normal[2]*xunit[0]-normal[0]*xunit[2])*(normal[2]*yunit[1]-normal[1]*yunit[2])-(normal[2]*yunit[0]-normal[0]*yunit[2])*(normal[2]*xunit[1]-normal[1]*xunit[2]));
                    }
                    else
                    {
                        raster1a[jj] = nodata_out;
                        raster1b[jj] = nodata_out;
                        raster2a[jj] = nodata_out;
                        raster2b[jj] = nodata_out;
                    }
                    
                    if (srxname != "")
                    {
                        if ((schrng1[0] == nodata)|(schrng1[0] == 0))
                        {
                            sr_raster11[jj] = 0;
                            sr_raster22[jj] = 0;
                        }
                        else
                        {
                            sr_raster11[jj] = std::abs(std::round(dot_C(schrng1,xunit)*dt/XSize/365.0/24.0/3600.0*1));
                            sr_raster22[jj] = std::abs(std::round(dot_C(schrng1,yunit)*dt/YSize/365.0/24.0/3600.0*1));
                            if (std::abs(std::round(dot_C(schrng2,xunit)*dt/XSize/365.0/24.0/3600.0*1)) > sr_raster11[jj])
                            {
                                sr_raster11[jj] = std::abs(std::round(dot_C(schrng2,xunit)*dt/XSize/365.0/24.0/3600.0*1));
                            }
                            if (std::abs(std::round(dot_C(schrng2,yunit)*dt/YSize/365.0/24.0/3600.0*1)) > sr_raster22[jj])
                            {
                                sr_raster22[jj] = std::abs(std::round(dot_C(schrng2,yunit)*dt/YSize/365.0/24.0/3600.0*1));
                            }
                            if (sr_raster11[jj] == 0)
                            {
                                sr_raster11[jj] = 1;
                            }
                            if (sr_raster22[jj] == 0)
                            {
                                sr_raster22[jj] = 1;
                            }
                        }
                    }
 
                }
                
                
                
                if (csminxname != "")
                {
                    if (csminxLine[jj] == nodata)
                    {
                        csmin_raster11[jj] = nodata_out;
                        csmin_raster22[jj] = nodata_out;
                    }
                    else
                    {
                        csmin_raster11[jj] = csminxLine[jj] / ChipSizeX0 * ChipSizeX0_PIX_X;
                        csmin_raster22[jj] = csminyLine[jj] / ChipSizeX0 * ChipSizeX0_PIX_Y;
                    }
                }
                
                
                if (csmaxxname != "")
                {
                    if (csmaxxLine[jj] == nodata)
                    {
                        csmax_raster11[jj] = nodata_out;
                        csmax_raster22[jj] = nodata_out;
                    }
                    else
                    {
                        csmax_raster11[jj] = csmaxxLine[jj] / ChipSizeX0 * ChipSizeX0_PIX_X;
                        csmax_raster22[jj] = csmaxyLine[jj] / ChipSizeX0 * ChipSizeX0_PIX_Y;
                    }
                }
                
                
                if (ssmname != "")
                {
                    if (ssmLine[jj] == nodata)
                    {
                        ssm_raster[jj] = nodata_out;
                    }
                    else
                    {
                        ssm_raster[jj] = ssmLine[jj];
                    }
                }
                
                
                
//                raster1a[jj] = los[0]*dt/dr/365.0/24.0/3600.0;
//                raster1b[jj] = los[1]*dt/dr/365.0/24.0/3600.0;
//                raster1c[jj] = los[2]*dt/dr/365.0/24.0/3600.0;
//                raster2a[jj] = temp[0]*dt/norm_C(alt)/365.0/24.0/3600.0;
//                raster2b[jj] = temp[1]*dt/norm_C(alt)/365.0/24.0/3600.0;
//                raster2c[jj] = temp[2]*dt/norm_C(alt)/365.0/24.0/3600.0;
            }
            
            
//            std::cout << ii << " " << jj << "\n";
//            std::cout << rgind << " " << azind << "\n";
//            std::cout << raster1[jj][ii] << " " << raster2[jj][ii] << "\n";
//            std::cout << raster1[ii][jj] << "\n";
        }
        
        
        
        poBand1->RasterIO( GF_Write, 0, ii, pCount, 1,
                          raster1, pCount, 1, GDT_Int32, 0, 0 );
        poBand2->RasterIO( GF_Write, 0, ii, pCount, 1,
                          raster2, pCount, 1, GDT_Int32, 0, 0 );
        
        if ((dhdxname != "")&(vxname != ""))
        {
            poBand1Off->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 raster11, pCount, 1, GDT_Int32, 0, 0 );
            poBand2Off->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 raster22, pCount, 1, GDT_Int32, 0, 0 );
        }
        
        if ((dhdxname != "")&(srxname != ""))
        {
            poBand1Sch->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 sr_raster11, pCount, 1, GDT_Int32, 0, 0 );
            poBand2Sch->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 sr_raster22, pCount, 1, GDT_Int32, 0, 0 );
        }
        
        if (csminxname != "")
        {
            poBand1Min->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 csmin_raster11, pCount, 1, GDT_Int32, 0, 0 );
            poBand2Min->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 csmin_raster22, pCount, 1, GDT_Int32, 0, 0 );
        }
        
        if (csmaxxname != "")
        {
            poBand1Max->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 csmax_raster11, pCount, 1, GDT_Int32, 0, 0 );
            poBand2Max->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 csmax_raster22, pCount, 1, GDT_Int32, 0, 0 );
        }
        
        if (ssmname != "")
        {
            poBand1Msk->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 ssm_raster, pCount, 1, GDT_Int32, 0, 0 );
        }
        
        if (dhdxname != "")
        {
            poBand1RO2VX->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 raster1a, pCount, 1, GDT_Float64, 0, 0 );
            poBand2RO2VX->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 raster1b, pCount, 1, GDT_Float64, 0, 0 );
    //        poBand3Los->RasterIO( GF_Write, 0, ii, pCount, 1,
    //                             raster1c, pCount, 1, GDT_Float64, 0, 0 );
            poBand1RO2VY->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 raster2a, pCount, 1, GDT_Float64, 0, 0 );
            poBand2RO2VY->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 raster2b, pCount, 1, GDT_Float64, 0, 0 );
    //        poBand3Alt->RasterIO( GF_Write, 0, ii, pCount, 1,
    //                             raster2c, pCount, 1, GDT_Float64, 0, 0 );
        }
        
        
    }
    
    /* Once we're done, close properly the dataset */
    GDALClose( (GDALDatasetH) poDstDS );
    
    if ((dhdxname != "")&(vxname != ""))
    {
        /* Once we're done, close properly the dataset */
        GDALClose( (GDALDatasetH) poDstDSOff );
    }
    
    if ((dhdxname != "")&(srxname != ""))
    {
        /* Once we're done, close properly the dataset */
        GDALClose( (GDALDatasetH) poDstDSSch );
    }
    
    if (csminxname != "")
    {
        /* Once we're done, close properly the dataset */
        GDALClose( (GDALDatasetH) poDstDSMin );
    }
    
    if (csmaxxname != "")
    {
        /* Once we're done, close properly the dataset */
        GDALClose( (GDALDatasetH) poDstDSMax );
    }
    
    if (ssmname != "")
    {
        /* Once we're done, close properly the dataset */
        GDALClose( (GDALDatasetH) poDstDSMsk );
    }
    
    if (dhdxname != "")
    {
        /* Once we're done, close properly the dataset */
        GDALClose( (GDALDatasetH) poDstDSRO2VX );
        
        /* Once we're done, close properly the dataset */
        GDALClose( (GDALDatasetH) poDstDSRO2VY );
    }
    
    
    GDALClose(demDS);
    
    if (dhdxname != "")
    {
        GDALClose(sxDS);
        GDALClose(syDS);
    }
    
    if (vxname != "")
    {
        GDALClose(vxDS);
        GDALClose(vyDS);
    }
    
    if (srxname != "")
    {
        GDALClose(srxDS);
        GDALClose(sryDS);
    }
    
    if (csminxname != "")
    {
        GDALClose(csminxDS);
        GDALClose(csminyDS);
    }
    
    if (csmaxxname != "")
    {
        GDALClose(csmaxxDS);
        GDALClose(csmaxyDS);
    }
    
    if (ssmname != "")
    {
        GDALClose(ssmDS);
    }
    
    GDALDestroyDriverManager();
    
}

void geoGridOptical::computeBbox(double *wesn)
{
    std::cout << "\nEstimated bounding box: \n" 
              << "West: " << wesn[0] << "\n"
              << "East: " << wesn[1] << "\n"
              << "South: " << wesn[2] << "\n"
              << "North: " << wesn[3] << "\n";
}

double geoGridOptical::dot_C(double r_v[3], double r_w[3])
{
    double dot;
    dot = r_v[0]*r_w[0] + r_v[1]*r_w[1] + r_v[2]*r_w[2];
    return dot;
}

void geoGridOptical::cross_C(double r_u[3], double r_v[3], double r_w[3])
{
    r_w[0] = r_u[1]*r_v[2] - r_u[2]*r_v[1];
    r_w[1] = r_u[2]*r_v[0] - r_u[0]*r_v[2];
    r_w[2] = r_u[0]*r_v[1] - r_u[1]*r_v[0];
}

double geoGridOptical::norm_C(double r_v[3])
{
    double norm;
    norm = std::sqrt(r_v[0]*r_v[0] + r_v[1]*r_v[1] + r_v[2]*r_v[2]);
    return norm;
}


void geoGridOptical::unitvec_C(double r_v[3], double r_w[3])
{
    double norm;
    norm = std::sqrt(r_v[0]*r_v[0] + r_v[1]*r_v[1] + r_v[2]*r_v[2]);
    r_w[0] = r_v[0] / norm;
    r_w[1] = r_v[1] / norm;
    r_w[2] = r_v[2] / norm;
}
