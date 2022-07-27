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

#include "geogrid.h"
#include <gdal.h>
#include <gdal_priv.h>
#include <iostream>
#include <complex>
#include <cmath>


extern "C"
{
#include "linalg3.h"
#include "geometry.h"
}

void geoGrid::geogrid()
{
    //Some constants 
    double deg2rad = M_PI/180.0;

    //For now print inputs that were obtained

    std::cout << "\nRadar parameters: \n";
    std::cout << "Range: " << startingRange << "  " << dr << "\n";
    std::cout << "Azimuth: " << sensingStart << "  " << prf << "\n";
    std::cout << "Dimensions: " << nPixels << " " << nLines << "\n";
    std::cout << "Incidence Angle: " << incidenceAngle/deg2rad << "\n";

    std::cout << "\nMap inputs: \n";
    std::cout << "EPSG: " << epsgcode << "\n";
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
    if (demSRS.importFromEPSG(epsgcode) != 0)
    {
        std::cout << "Could not create OGR spatial reference for EPSG code: " << epsgcode << "\n";
        GDALClose(demDS);
        GDALDestroyDriverManager();
        exit(102);
    }

    OGRSpatialReference llhSRS(nullptr);
    if (llhSRS.importFromEPSG(4326) != 0)
    {
        std::cout << "Could not create OGR spatil reference for EPSG code: 4326 \n";
        GDALClose(demDS);
        GDALDestroyDriverManager();
        exit(103);
    }

    OGRCoordinateTransformation *fwdTrans = OGRCreateCoordinateTransformation( &demSRS, &llhSRS);
    OGRCoordinateTransformation *invTrans = OGRCreateCoordinateTransformation( &llhSRS, &demSRS);

    //WGS84 ellipsoid only
    cEllipsoid wgs84;
    wgs84.a = 6378137.0;
    wgs84.e2 = 0.0066943799901;

    //Initial guess for solution
    double tmid = sensingStart + 0.5 * nLines / prf;
    double satxmid[3];
    double satvmid[3];

    if (interpolateWGS84Orbit(orbit, tmid, satxmid, satvmid) != 0)
    {
        std::cout << "Error with orbit interpolation for setup. \n";
        GDALClose(demDS);
        GDALDestroyDriverManager();
        exit(104);
    }
//    std::cout << "Center Satellite Velocity: " << satvmid[0] << " " << satvmid[1] << " " << satvmid[2] << "\n";
//    std::cout << satxmid[0] << " " << satxmid[1] << " " << satxmid[2] << "\n";

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
    double raster1c[pCount];
    
    double raster2a[pCount];
    double raster2b[pCount];
    double raster2c[pCount];
    

    
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
    GDALRasterBand *poBand3RO2VX = NULL;
    GDALRasterBand *poBand3RO2VY = NULL;
    
    
    GDALDataset *poDstDS = NULL;
    GDALDataset *poDstDSOff = NULL;
    GDALDataset *poDstDSSch = NULL;
    GDALDataset *poDstDSMin = NULL;
    GDALDataset *poDstDSMax = NULL;
    GDALDataset *poDstDSMsk = NULL;
    GDALDataset *poDstDSRO2VX = NULL;
    GDALDataset *poDstDSRO2VY = NULL;

    

    double nodata;
//    double nodata_out;
    if (vxname != "")
    {
        int* pbSuccess = NULL;
        nodata = vxDS->GetRasterBand(1)->GetNoDataValue(pbSuccess);
    }
//    nodata_out = -2000000000;
    
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
        poDstDSRO2VX = poDriverRO2VX->Create( pszDstFilenameRO2VX, pCount, lCount, 3, GDT_Float64,
                                         papszOptions );
        
        poDstDSRO2VX->SetGeoTransform( adfGeoTransform );
        poDstDSRO2VX->SetProjection( pszSRS_WKT );
    //    CPLFree( pszSRS_WKT );
        
//        GDALRasterBand *poBand1RO2VX;
//        GDALRasterBand *poBand2RO2VX;
    //    GDALRasterBand *poBand3Los;
        poBand1RO2VX = poDstDSRO2VX->GetRasterBand(1);
        poBand2RO2VX = poDstDSRO2VX->GetRasterBand(2);
        poBand3RO2VX = poDstDSRO2VX->GetRasterBand(3);
        poBand1RO2VX->SetNoDataValue(nodata_out);
        poBand2RO2VX->SetNoDataValue(nodata_out);
        poBand3RO2VX->SetNoDataValue(nodata_out);
        

        GDALDriver *poDriverRO2VY;
        poDriverRO2VY = GetGDALDriverManager()->GetDriverByName(pszFormat);
        if( poDriverRO2VY == NULL )
        exit(107);
//        GDALDataset *poDstDSRO2VY;
        
        str = ro2vy_name;
        const char * pszDstFilenameRO2VY = str.c_str();
        poDstDSRO2VY = poDriverRO2VY->Create( pszDstFilenameRO2VY, pCount, lCount, 3, GDT_Float64,
                                         papszOptions );
        
        poDstDSRO2VY->SetGeoTransform( adfGeoTransform );
        poDstDSRO2VY->SetProjection( pszSRS_WKT );
//        CPLFree( pszSRS_WKT );
        
//        GDALRasterBand *poBand1RO2VY;
//        GDALRasterBand *poBand2RO2VY;
    //    GDALRasterBand *poBand3Alt;
        poBand1RO2VY = poDstDSRO2VY->GetRasterBand(1);
        poBand2RO2VY = poDstDSRO2VY->GetRasterBand(2);
        poBand3RO2VY = poDstDSRO2VY->GetRasterBand(3);
        poBand1RO2VY->SetNoDataValue(nodata_out);
        poBand2RO2VY->SetNoDataValue(nodata_out);
        poBand3RO2VY->SetNoDataValue(nodata_out);
        
        
    }
    
    CPLFree( pszSRS_WKT );

    
    
    
    
    // ground range and azimuth pixel size
//    double grd_res, azm_res;
    
//    double incang = 38.0*deg2rad;
    double incang = incidenceAngle;
    grd_res = dr / std::sin(incang);
    azm_res = norm_C(satvmid) / prf;
    std::cout << "Ground range pixel size: " << grd_res << "\n";
    std::cout << "Azimuth pixel size: " << azm_res << "\n";
//    int ChipSizeX0 = 240;
    double ChipSizeX0 = chipSizeX0;
    int ChipSizeX0_PIX_grd = std::ceil(ChipSizeX0 / grd_res / 4) * 4;
    int ChipSizeX0_PIX_azm = std::ceil(ChipSizeX0 / azm_res / 4) * 4;
    
    
    
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
        
        
        
        
        int rgind;
        int azind;
        
        for (int jj=0; jj<pCount; jj++)
        {
            double xyz[3];
            double llh[3];
            double targllh0[3];
            double llhi[3];
            double drpos[3];
            double slp[3];
            double vel[3];
            double schrng1[3];
            double schrng2[3];
            

            //Setup ENU with DEM
            llh[0] = geoTrans[0] + (jj+pOff+0.5)*geoTrans[1];
            llh[1] = y;
            llh[2] = demLine[jj];
            
            for(int pp=0; pp<3; pp++)
            {
                targllh0[pp] = llh[pp];
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
            fwdTrans->Transform(1, llh, llh+1, llh+2);

            //Bringing it into ISCE
            if (GDAL_VERSION_MAJOR == 2)
            {
                llhi[0] = deg2rad * llh[1];
                llhi[1] = deg2rad * llh[0];
            }
            else
            {
                llhi[0] = deg2rad * llh[0];
                llhi[1] = deg2rad * llh[1];
            }
            
            llhi[2] = llh[2];

            //Convert to ECEF
            latlon_C(&wgs84, xyz, llhi, LLH_2_XYZ);
            
//            if ((ii == (lCount+1)/2)&(jj == pCount/2)){
//                std::cout << xyz[0] << " " << xyz[1] << " " << xyz[2] << "\n";
//            }
            
            //Start the geo2rdr algorithm
            double satx[3];
            double satv[3];
            double tprev;
            
            double tline = tmid;
            double rngpix;
            double los[3];
            double alt[3];
            double normal[3];
            double cross[3];
            double cross_check;
            
            double dopfact;
            double height;
            double vhat[3], that[3], chat[3], nhat[3], delta[3], targVec[3], targXYZ[3], diffvec[3], temp[3], satvc[3], altc[3];
            double vmag;
            double major, minor;
            double satDist;
            double alpha, beta, gamma;
            double radius, hgt, zsch;
            double a, b, costheta, sintheta;
            double rdiff;
            
            for(int kk=0; kk<3; kk++) 
            {
                satx[kk]  = satxmid[kk];
            }

            for(int kk=0; kk<3; kk++)
            {
                satv[kk]  = satvmid[kk];
            }

            //Iterations
            for (int kk=0; kk<51;kk++)
            {
                tprev = tline;
                
                for(int pp=0; pp<3; pp++)
                {
                    drpos[pp] = xyz[pp] - satx[pp];
                }
                
                rngpix = norm_C(drpos);
                double fn = dot_C(drpos, satv);
                double fnprime = -dot_C(satv, satv);
                
                tline = tline - fn/fnprime;
                
                if (interpolateWGS84Orbit(orbit, tline, satx, satv) != 0)
                {
                    std::cout << "Error with orbit interpolation. \n";
                    GDALClose(demDS);
                    GDALDestroyDriverManager();
                    exit(106);
                }

            }
//            if ((ii==600)&&(jj==600))
//            {
//                std::cout << "\n" << lOff+ii << " " << pOff+jj << " " << demLine[jj] << "\n";
//            }
            rgind = std::round((rngpix - startingRange) / dr) + 0.;
            azind = std::round((tline - sensingStart) * prf) + 0.;
            
            
            //*********************Slant-range vector
            
            
            unitvec_C(drpos, los);
            
            for(int pp=0; pp<3; pp++)
            {
                llh[pp]  = xyz[pp] + los[pp] * dr;
            }
            
            latlon_C(&wgs84, llh, llhi, XYZ_2_LLH);
            
            //Bringing it from ISCE into LLH
            if (GDAL_VERSION_MAJOR == 2)
            {
                llh[0] = llhi[1] / deg2rad;
                llh[1] = llhi[0] / deg2rad;
            }
            else
            {
                llh[0] = llhi[0] / deg2rad;
                llh[1] = llhi[1] / deg2rad;
            }
            
            llh[2] = llhi[2];
            
            //Convert from LLH inplace to DEM coordinates
            invTrans->Transform(1, llh, llh+1, llh+2);
            
            for(int pp=0; pp<3; pp++)
            {
                drpos[pp]  = llh[pp] - targllh0[pp];
            }
            unitvec_C(drpos, los);
            
            //*********************Along-track vector
            
            tline = tline + 1/prf;
            
            if (interpolateWGS84Orbit(orbit, tline, satx, satv) != 0)
            {
                std::cout << "Error with orbit interpolation. \n";
                GDALClose(demDS);
                GDALDestroyDriverManager();
                exit(106);
            }
            //run the topo algorithm for new tline
            dopfact = 0.0;
            height = demLine[jj];
            unitvec_C(satv, vhat);
            vmag = norm_C(satv);
            
            //Convert position and velocity to local tangent plane
            major = wgs84.a;
            minor = major * std::sqrt(1 - wgs84.e2);
            
            //Setup ortho normal system right below satellite
            satDist = norm_C(satx);
            temp[0] = (satx[0] / major);
            temp[1] = (satx[1] / major);
            temp[2] = (satx[2] / minor);
            alpha = 1 / norm_C(temp);
            radius = alpha * satDist;
            hgt = (1.0 - alpha) * satDist;
            
            //Setup TCN basis - Geocentric
            unitvec_C(satx, nhat);
            for(int pp=0; pp<3; pp++)
            {
                nhat[pp]  = -nhat[pp];
            }
            cross_C(nhat,satv,temp);
            unitvec_C(temp, chat);
            cross_C(chat,nhat,temp);
            unitvec_C(temp, that);
            
            
            //Solve the range doppler eqns iteratively
            //Initial guess
            zsch = height;
            
            for (int kk=0; kk<10;kk++)
            {
                a = satDist;
                b = (radius + zsch);
                
                costheta = 0.5 * (a / rngpix + rngpix / a - (b / a) * (b / rngpix));
                sintheta = std::sqrt(1-costheta*costheta);
                
                gamma = rngpix * costheta;
                alpha = dopfact - gamma * dot_C(nhat,vhat) / dot_C(vhat,that);
                beta = -lookSide * std::sqrt(rngpix * rngpix * sintheta * sintheta - alpha * alpha);
                for(int pp=0; pp<3; pp++)
                {
                    delta[pp] = alpha * that[pp] + beta * chat[pp] + gamma * nhat[pp];
                }
                
                for(int pp=0; pp<3; pp++)
                {
                    targVec[pp] = satx[pp] + delta[pp];
                }
                
                latlon_C(&wgs84, targVec, llhi, XYZ_2_LLH);
                llhi[2] = height;
                latlon_C(&wgs84, targXYZ, llhi, LLH_2_XYZ);
                
                zsch = norm_C(targXYZ) - radius;
                
                for(int pp=0; pp<3; pp++)
                {
                    diffvec[pp] = satx[pp] - targXYZ[pp];
                }
                rdiff  = rngpix - norm_C(diffvec);
            }
            
            //Bringing it from ISCE into LLH
            
            if (GDAL_VERSION_MAJOR == 2)
            {
                llh[0] = llhi[1] / deg2rad;
                llh[1] = llhi[0] / deg2rad;
            }
            else
            {
                llh[0] = llhi[0] / deg2rad;
                llh[1] = llhi[1] / deg2rad;
            }
            
            llh[2] = llhi[2];
            
            //Convert from LLH inplace to DEM coordinates
            invTrans->Transform(1, llh, llh+1, llh+2);
            
            for(int pp=0; pp<3; pp++)
            {
                alt[pp]  = llh[pp] - targllh0[pp];
            }
            unitvec_C(alt, temp);
            
            
            if (dhdxname != "")
            {
                //*********************Local normal vector
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
            
            
            if ((rgind > nPixels-1)|(rgind < 1-1)|(azind > nLines-1)|(azind < 1-1))
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
                raster1c[jj] = nodata_out;
                raster2a[jj] = nodata_out;
                raster2b[jj] = nodata_out;
                raster2c[jj] = nodata_out;
                
            }
            else
            {
                raster1[jj] = rgind;
                raster2[jj] = azind;
                
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
                            raster11[jj] = std::round(dot_C(vel,los)*dt/dr/365.0/24.0/3600.0*1);
                            raster22[jj] = std::round(dot_C(vel,temp)*dt/norm_C(alt)/365.0/24.0/3600.0*1);
                        }
                      
                    }
                    
                    cross_C(los,temp,cross);
                    unitvec_C(cross, cross);
                    cross_check = std::abs(std::acos(dot_C(normal,cross))/deg2rad-90.0);
                    
                    if (cross_check > 1.0)
                    {
                        raster1a[jj] = normal[2]/(dt/dr/365.0/24.0/3600.0)*(normal[2]*temp[1]-normal[1]*temp[2])/((normal[2]*los[0]-normal[0]*los[2])*(normal[2]*temp[1]-normal[1]*temp[2])-(normal[2]*temp[0]-normal[0]*temp[2])*(normal[2]*los[1]-normal[1]*los[2]));
                        raster1b[jj] = -normal[2]/(dt/norm_C(alt)/365.0/24.0/3600.0)*(normal[2]*los[1]-normal[1]*los[2])/((normal[2]*los[0]-normal[0]*los[2])*(normal[2]*temp[1]-normal[1]*temp[2])-(normal[2]*temp[0]-normal[0]*temp[2])*(normal[2]*los[1]-normal[1]*los[2]));
                        raster2a[jj] = -normal[2]/(dt/dr/365.0/24.0/3600.0)*(normal[2]*temp[0]-normal[0]*temp[2])/((normal[2]*los[0]-normal[0]*los[2])*(normal[2]*temp[1]-normal[1]*temp[2])-(normal[2]*temp[0]-normal[0]*temp[2])*(normal[2]*los[1]-normal[1]*los[2]));
                        raster2b[jj] = normal[2]/(dt/norm_C(alt)/365.0/24.0/3600.0)*(normal[2]*los[0]-normal[0]*los[2])/((normal[2]*los[0]-normal[0]*los[2])*(normal[2]*temp[1]-normal[1]*temp[2])-(normal[2]*temp[0]-normal[0]*temp[2])*(normal[2]*los[1]-normal[1]*los[2]));
                    }
                    else
                    {
                        raster1a[jj] = nodata_out;
                        raster1b[jj] = nodata_out;
                        raster2a[jj] = nodata_out;
                        raster2b[jj] = nodata_out;
                    }
                    
                    for(int pp=0; pp<3; pp++)
                    {
                        targXYZ[pp] -= xyz[pp];
                    }
                    raster1c[jj] = dr/dt*365.0*24.0*3600.0*1;
                    raster2c[jj] = norm_C(targXYZ)/dt*365.0*24.0*3600.0*1;
                    
                    
                    if (srxname != "")
                    {
                        if ((schrng1[0] == nodata)|(schrng1[0] == 0))
                        {
                            sr_raster11[jj] = 0;
                            sr_raster22[jj] = 0;
                        }
                        else
                        {
                            sr_raster11[jj] = std::abs(std::round(dot_C(schrng1,los)*dt/dr/365.0/24.0/3600.0*1));
                            sr_raster22[jj] = std::abs(std::round(dot_C(schrng1,temp)*dt/norm_C(alt)/365.0/24.0/3600.0*1));
                            if (std::abs(std::round(dot_C(schrng2,los)*dt/dr/365.0/24.0/3600.0*1)) > sr_raster11[jj])
                            {
                                sr_raster11[jj] = std::abs(std::round(dot_C(schrng2,los)*dt/dr/365.0/24.0/3600.0*1));
                            }
                            if (std::abs(std::round(dot_C(schrng2,temp)*dt/norm_C(alt)/365.0/24.0/3600.0*1)) > sr_raster22[jj])
                            {
                                sr_raster22[jj] = std::abs(std::round(dot_C(schrng2,temp)*dt/norm_C(alt)/365.0/24.0/3600.0*1));
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
                        csmin_raster11[jj] = csminxLine[jj] / ChipSizeX0 * ChipSizeX0_PIX_grd;
                        csmin_raster22[jj] = csminyLine[jj] / ChipSizeX0 * ChipSizeX0_PIX_azm;
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
                        csmax_raster11[jj] = csmaxxLine[jj] / ChipSizeX0 * ChipSizeX0_PIX_grd;
                        csmax_raster22[jj] = csmaxyLine[jj] / ChipSizeX0 * ChipSizeX0_PIX_azm;
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
            poBand3RO2VX->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 raster1c, pCount, 1, GDT_Float64, 0, 0 );
            poBand1RO2VY->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 raster2a, pCount, 1, GDT_Float64, 0, 0 );
            poBand2RO2VY->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 raster2b, pCount, 1, GDT_Float64, 0, 0 );
            poBand3RO2VY->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 raster2c, pCount, 1, GDT_Float64, 0, 0 );
            
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
void geoGrid::computeBbox(double *wesn)
{
    std::cout << "\nEstimated bounding box: \n" 
              << "West: " << wesn[0] << "\n"
              << "East: " << wesn[1] << "\n"
              << "South: " << wesn[2] << "\n"
              << "North: " << wesn[3] << "\n";
}
