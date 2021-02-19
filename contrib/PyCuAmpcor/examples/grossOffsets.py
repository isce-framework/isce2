#!/usr/bin/env python3
# Generate pixel offsets based on Antarctica velocity model (MEaSUREs InSAR-Based Antarctica Ice Velocity Map, Version 2 doi:https://doi.org/10.5067/D7GK8F5J8M8R)
# Author: Minyan Zhong
import os
import argparse
import isce
import isceobj
import gdal
import pyproj
import numpy as np
import matplotlib.pyplot as plt

EXAMPLE = '''
grossOffsets.py --model_file antarctica_ice_velocity_450m_v2.nc --lon lon.rdr --lat lat.rdr --los los.rdr --los_scheme bil --ww 64 --wh 64 --sw 10 --sh 10 --mm 50 --kw 32 --kh 32 --startpixeldw 50 --startpixelac 50 --rangePixelSize 0.930 --azimuthPixelSize 2.286 --interval 1
'''

def createParser():
    '''
    Command line parser.
    '''
    
    parser = argparse.ArgumentParser(description='Generate pixel offsets (integer pixel) based on Antarctica ice velocity model (MEaSUREs InSAR-Based Antarctica Ice Velocity Map, Version 2 doi:https://doi.org/10.5067/D7GK8F5J8M8R)', formatter_class=argparse.RawTextHelpFormatter, epilog=EXAMPLE)

    # path to antarctica velocity model
    parser.add_argument('--model_file', type=str, dest='model_file', required=True)

    # lat, lon, los
    parser.add_argument('--lat', type=str, dest='lat', required=True,
                        help='latitude file')
    parser.add_argument('--lon', type=str, dest='lon', required=True,
                        help='longitude fie')

    parser.add_argument('--los', type=str, dest='los', required=True,
                        help='two bands raster data in float. band1: incidence angle; bands: satellite flight direction (ISCE2 convention)')

    parser.add_argument('--los_scheme', type=str, dest='los_scheme', required=True,
                        help='interleave scheme of los (bil, bsq or bip)')

    # window size settings
    parser.add_argument('--ww', type=int, dest='winwidth', default=64,
                        help='Window width (default: %(default)s).')
    parser.add_argument('--wh', type=int, dest='winhgt', default=64,
                        help='Window height (default: %(default)s).')
    parser.add_argument('--sw', type=int, dest='srcwidth', default=20,
                        help='Half search range along width, (default: %(default)s, recommend: 4-32).')
    parser.add_argument('--sh', type=int, dest='srchgt', default=20,
                        help='Half search range along height (default: %(default)s, recommend: 4-32).')
    parser.add_argument('--kw', type=int, dest='skipwidth', default=64,
                        help='Skip across (default: %(default)s).')
    parser.add_argument('--kh', type=int, dest='skiphgt', default=64,
                        help='Skip down (default: %(default)s).')

    # determine the number of windows
    # either specify the starting pixel and the number of windows,
    # or by setting them to -1, let the script to compute these parameters
    parser.add_argument('--mm', type=int, dest='margin', default=0,
                        help='Margin (default: %(default)s).')
    
    parser.add_argument('--spa','--startpixelac', dest='startpixelac', type=int, default=-1, help='Starting Pixel across of the reference image(default: %(default)s to be determined by margin and search range).')
    
    parser.add_argument('--spd','--startpixeldw', dest='startpixeldw', type=int, default=-1, help='Starting Pixel down of the reference image (default: %(default)s).')
    
    parser.add_argument('--aps', '--azimuthPixelSize', dest='azimuthPixelSize', type=float, required=True, help='azimuth pixel size')
    
    parser.add_argument('--rps', '--rangePixelSize', dest='rangePixelSize', type=float, required=True, help='range pixel size')

    parser.add_argument('--interval', dest='interval', type=float, required=True, help='interval between reference and secondary scene (unit: day)')

    parser.add_argument('--outdir', dest='outdir', type=str, default='.', help='output directory')

    parser.add_argument('--outname', dest='outname', type=str, default='grossOffsets.bin', help='output name of gross pixel offsets (integer)')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    inps = parser.parse_args(args=iargs)
    return inps

class grossOffsets:
    def __init__(self, inps):
        model_path = inps.model_file
        self.model_file = model_path
        self.latfile = inps.lat
        self.lonfile = inps.lon
        self.losfile = inps.los

        ds = gdal.Open(self.losfile)
        self.XSize = ds.RasterXSize
        self.YSize = ds.RasterYSize
        ds = None
        
        self.los_scheme = inps.los_scheme.lower()
        assert(self.los_scheme in ['bil','bsq', 'bip']), print('interleave scheme of los')

        self.margin = inps.margin
        self.winSizeHgt = inps.winhgt
        self.winSizeWidth = inps.winwidth
        self.searchSizeHgt = inps.srchgt
        self.searchSizeWidth = inps.srcwidth
        self.skipSizeHgt = inps.skiphgt
        self.skipSizeWidth = inps.skipwidth

        self.startpixelac = inps.startpixelac if inps.startpixelac != -1 else self.margin + self.searchSizeWidth
 
        self.startpixeldw = inps.startpixeldw if inps.startpixeldw != -1 else self.margin + self.searchSizeHgt
 
        self.azPixelSize = inps.azimuthPixelSize
        self.rngPixelSize = inps.rangePixelSize

        self.interval = inps.interval

        self.outdir = inps.outdir
        self.outname = inps.outname

        self.get_veloData()
        self.vProj = pyproj.Proj('+init=EPSG:3031')

    def get_veloData(self):
        assert os.path.exists(self.model_file), print("Please download MEaSUREs InSAR-Based Antarctica Ice Velocity Map, Version 2 at https://nsidc.org/data/NSIDC-0484/versions")

        data_read = 0
        ds = gdal.Open("NETCDF:{0}:{1}".format(self.model_file, 'VX'))
        self.vx = ds.ReadAsArray()

        ds = gdal.Open("NETCDF:{0}:{1}".format(self.model_file, 'VY'))
        self.vy = ds.ReadAsArray()

        self.vx = np.flipud(self.vx)
        self.vy = np.flipud(self.vy) 

        self.v = np.sqrt(np.multiply(self.vx,self.vx)+np.multiply(self.vy,self.vy))

        self.model_spacing = 450
        self.x0 = np.arange(-2800000,2800000,step=450)
        self.y0 = np.arange(-2800000,2800000,step=450)+200

    def runGrossOffsets(self):
        ## Step 0: Set up projection transformers for ease of use
        self.llhProj = pyproj.Proj('+init=EPSG:4326')
        self.xyzProj = pyproj.Proj('+init=EPSG:4978')

        # From xy to lat lon.
        refPt = self.vProj(0.0, 0.0, inverse=True)
       
        ### Step 2: Cut the data
        print('Extract the data to this radar scene...')
        # The following code is to be consistent with "get_offset_geometry" in dense_offset.py

        numWinDown = (self.YSize - self.margin*2 - self.searchSizeHgt*2 - self.winSizeHgt) // self.skipSizeHgt
        numWinAcross = (self.XSize - self.margin*2 - self.searchSizeWidth*2 - self.winSizeWidth) // self.skipSizeWidth
        
        lat = np.zeros(shape=(numWinDown,numWinAcross),dtype=np.float64) 
        lon = np.zeros(shape=(numWinDown,numWinAcross),dtype=np.float64)
        inc = np.zeros(shape=(numWinDown,numWinAcross),dtype=np.float32)
        azi = np.zeros(shape=(numWinDown,numWinAcross),dtype=np.float32)

        self.centerOffsetHgt = self.winSizeHgt//2-1
        self.centerOffsetWidth = self.winSizeWidth//2-1

        print("Number of winows in down direction, Number of window in across direction: ")
        print(numWinDown, numWinAcross)

        cut_vx = np.zeros(shape=(numWinDown,numWinAcross))
        cut_vy = np.zeros(shape=(numWinDown,numWinAcross))
        cut_v = np.zeros(shape=(numWinDown,numWinAcross))
        pixel = np.zeros(shape=(numWinDown,numWinAcross))
        line = np.zeros(shape=(numWinDown,numWinAcross))

        for iwin in range(numWinDown):
            # Need to calculate lat lon in the interior mode.
            print('Processing line: ',iwin, 'out of', numWinDown)
            down = self.margin + self.skipSizeHgt * iwin  + self.centerOffsetHgt
            off = down*self.XSize

            across_indices = self.margin + np.arange(numWinAcross)*self.skipSizeWidth + self.centerOffsetWidth

            # latitude
            latline = np.memmap(filename=self.latfile,dtype='float64',offset=8*off,shape=(self.XSize))
            # longitude
            lonline = np.memmap(filename=self.lonfile,dtype='float64',offset=8*off,shape=(self.XSize))

            # incidence angle and satellite flight direction
            # bil
            if self.los_scheme == "bil":
                off2 = down * self.XSize * 2
                losline = np.memmap(filename=self.losfile,dtype='float32',offset=4*off2,shape=(self.XSize*2))

                incline = losline[0:self.XSize]
                aziline = losline[self.XSize:self.XSize*2]
            # bsq
            elif self.los_scheme == 'bsq':
                off2 = self.YSize * self.XSize + down * self.XSize
                incline = np.memmap(filename=self.losfile,dtype='float32',offset=4*off,shape=(self.XSize))
                aziline = np.memmap(filename=self.losfile,dtype='float32',offset=4*off2,shape=(self.XSize))
            # bip 
            else:
               off2 = down * self.XSize * 2
               losline = np.memmap(filename=self.losfile,dtype='float32',offset=4*off2,shape=(self.XSize*2))
               incline = losline[0:self.XSize*2:2]
               aziline = losline[1:self.XSize*2:2]

            # Subset the line
            lat[iwin,:] = latline[across_indices]
            lon[iwin,:] = lonline[across_indices]
            inc[iwin,:] = incline[across_indices]
            azi[iwin,:] = aziline[across_indices]

            #print(iwin,'lat: ',lat[iwin,:])
            #print(iwin,'lon: ',lon[iwin,:])
            #print(iwin,'inc: ',inc[iwin,:])
            #print(iwin,'azi: ',azi[iwin,:])

            #### Look up in MEaSUREs InSAR-Based Antarctica Ice Velocity Map

            # Convert lat lon to grid coordinates in polar stereographic projection.
            xyMap = pyproj.transform(self.llhProj, self.vProj, lon[iwin,:], lat[iwin,:])
 
            # Extract the values in the velocity model.
            model_spacing = self.model_spacing
            pixel[iwin,:] = np.clip((xyMap[0]-self.x0[0])/model_spacing, 0, self.vx.shape[1]-1)
            line[iwin,:] = np.clip((xyMap[1]-self.y0[0])/model_spacing, 0, self.vx.shape[0]-1)

            pixel_int = pixel[iwin,:].astype(int)
            line_int = line[iwin,:].astype(int)

            cut_vx[iwin,:] = self.vx[line_int,pixel_int]
            cut_vy[iwin,:] = self.vy[line_int,pixel_int]

        cut_v = np.sqrt(np.multiply(cut_vx,cut_vx),np.multiply(cut_vy,cut_vy))
        valid = np.logical_and(inc!=0, cut_v!=0)

        ### Mask out invalid values ###
        # 1. Mask out invalid values at margin.
        cut_vx[inc==0] = np.nan
        cut_vy[inc==0] = np.nan
 
        # Get Interpolated speed.
        cut_v = np.sqrt(np.multiply(cut_vx,cut_vx),np.multiply(cut_vy,cut_vy))

        print("The speed matrix") 
        print(cut_v)
        print("The shape of speed matrix")
        print(cut_v.shape)

        ### Step 3: Convert XY velocity to EN velocity (clockwise rotation)
        print('Coverting XY to EN...')

        lonr = np.radians(lon - refPt[0])
        cut_ve = np.multiply(cut_vx, np.cos(lonr)) - np.multiply(cut_vy, np.sin(lonr))
        cut_vn = np.multiply(cut_vy, np.cos(lonr)) + np.multiply(cut_vx, np.sin(lonr))

        print('Polar stereographic velocity: ', [cut_vx, cut_vy])
        print('Local ENU velocity: ', [cut_ve, cut_vn])
 
        ####Step 4: Convert EN velocity to rng and azimuth
        #Local los and azi vector in ENU coordinate
        print(' Coverting EN to rdr...')
        incr = np.radians(inc)
        azir = np.radians(azi)
        losr = np.radians(azi-90.0)

        losenu=[ np.multiply(np.sin(incr),np.cos(losr)),
                 np.multiply(np.sin(incr),np.sin(losr)),
                 -np.cos(incr) ]
        
        azienu=[ np.cos(azir),
                 np.sin(azir),
                 0.0 ]

        # unit: pixel per day
        grossRangeOffset = (self.interval/365.25) * (cut_ve * losenu[0] + cut_vn * losenu[1])/ self.rngPixelSize
        grossAzimuthOffset = (self.interval/365.25) * (cut_ve * azienu[0] + cut_vn * azienu[1]) / self.azPixelSize

        # Mask out invalid values at margin.
        grossRangeOffset[inc==0] = np.nan
        grossAzimuthOffset[inc==0] = np.nan
 
        print('Gross azimuth offset: ', grossAzimuthOffset)
        print('Gross range offset: ', grossRangeOffset)
        print('Shape of gross offsets: ', grossRangeOffset.shape)

        ### Show FLOAT results ###
        fig=plt.figure(21,figsize=(9,9))
        ax = fig.add_subplot(121)
        ax.set_title('gross azimuth offset',fontsize=15)
        cax = ax.imshow(grossAzimuthOffset,cmap=plt.cm.coolwarm)
        cbar = fig.colorbar(cax,shrink=0.8)
        cbar.set_label("pixel",fontsize=15)

        ax = fig.add_subplot(122)
        ax.set_title('gross range offset',fontsize=15)
        cax = ax.imshow(grossRangeOffset,cmap=plt.cm.coolwarm)
        cbar = fig.colorbar(cax,shrink=0.8)
        cbar.set_label("pixel",fontsize=15)

        figname = os.path.join(self.outdir,'pixel_offsets.png')
        fig.savefig(figname,format='png')
        plt.close()

        # Save grossRangeOffset and grossAzimuthOffset as ISCE supported images.
        # Range
        rangeFileName = os.path.join(self.outdir, 'grossRange.off')
        driver = gdal.GetDriverByName('ENVI')
        dst_ds = driver.Create(rangeFileName, xsize=grossRangeOffset.shape[1], ysize=grossRangeOffset.shape[0], bands=1, eType=gdal.GDT_Float32)
        dst_ds.GetRasterBand(1).WriteArray(grossRangeOffset,0,0)
        dst_ds = None

        outImage = isceobj.createImage()
        outImage.setDataType('FLOAT')
        outImage.setFilename(rangeFileName)
        outImage.setBands(1)
        outImage.scheme='BIL'
        outImage.setLength(grossRangeOffset.shape[0])
        outImage.setWidth(grossRangeOffset.shape[1])
        outImage.setAccessMode('read')
        outImage.renderHdr()

        # Azimuth
        azimuthFileName = os.path.join(self.outdir, 'grossAzimuth.off')
        driver = gdal.GetDriverByName('ENVI')
        dst_ds = driver.Create(azimuthFileName, xsize=grossAzimuthOffset.shape[1], ysize=grossAzimuthOffset.shape[0], bands=1, eType=gdal.GDT_Float32)
        dst_ds.GetRasterBand(1).WriteArray(grossAzimuthOffset,0,0)
        dst_ds = None

        outImage = isceobj.createImage()
        outImage.setDataType('FLOAT')
        outImage.setFilename(azimuthFileName)
        outImage.setBands(1)
        outImage.scheme='BIL'
        outImage.setLength(grossAzimuthOffset.shape[0])
        outImage.setWidth(grossAzimuthOffset.shape[1])
        outImage.setAccessMode('read')
        outImage.renderHdr()

        ### Round to integer ###
        grossAzimuthOffset_int = np.rint(grossAzimuthOffset).astype(np.int32)
        grossRangeOffset_int = np.rint(grossRangeOffset).astype(np.int32)

        ### Show Integer results ###
        fig=plt.figure(22,figsize=(9,9))
        ax = fig.add_subplot(121)
        ax.set_title('gross azimuth offset (int)',fontsize=15)
        cax = ax.imshow(grossAzimuthOffset_int,cmap=plt.cm.coolwarm)
        cbar = fig.colorbar(cax,shrink=0.8)
        cbar.set_label("pixel",fontsize=15)

        ax = fig.add_subplot(122)
        ax.set_title('gross range offset (int)',fontsize=15)
        cax = ax.imshow(grossRangeOffset_int,cmap=plt.cm.coolwarm)
        cbar = fig.colorbar(cax,shrink=0.8)
        cbar.set_label("pixel",fontsize=15)

        figname = os.path.join(self.outdir,'pixel_offsets_int.png')
        fig.savefig(figname,format='png')
        plt.close()

        # Save grossRangeOffset and grossAzimuthOffset as ISCE supported images.
        # Range
        rangeFileName = os.path.join(self.outdir, 'grossRange_int.off')
        driver = gdal.GetDriverByName('ENVI')
        dst_ds = driver.Create(rangeFileName, xsize=grossRangeOffset.shape[1], ysize=grossRangeOffset.shape[0], bands=1, eType=gdal.GDT_Int32)
        dst_ds.GetRasterBand(1).WriteArray(grossRangeOffset_int,0,0)
        dst_ds = None

        outImage = isceobj.createImage()
        outImage.setDataType('INT')
        outImage.setFilename(rangeFileName)
        outImage.setBands(1)
        outImage.scheme='BIL'
        outImage.setLength(grossRangeOffset.shape[0])
        outImage.setWidth(grossRangeOffset.shape[1])
        outImage.setAccessMode('read')
        outImage.renderHdr()

        # Azimuth
        azimuthFileName = os.path.join(self.outdir, 'grossAzimuth_int.off')
        driver = gdal.GetDriverByName('ENVI')
        dst_ds = driver.Create(azimuthFileName, xsize=grossAzimuthOffset.shape[1], ysize=grossAzimuthOffset.shape[0], bands=1, eType=gdal.GDT_Int32)
        dst_ds.GetRasterBand(1).WriteArray(grossAzimuthOffset_int,0,0)
        dst_ds = None

        outImage = isceobj.createImage()
        outImage.setDataType('INT')
        outImage.setFilename(azimuthFileName)
        outImage.setBands(1)
        outImage.scheme='BIL'
        outImage.setLength(grossAzimuthOffset.shape[0])
        outImage.setWidth(grossAzimuthOffset.shape[1])
        outImage.setAccessMode('read')
        outImage.renderHdr()

        # Round to integer and write to raw binary file
        numTotal = numWinDown * numWinAcross
        grossOffsets_int = np.hstack((grossAzimuthOffset_int.reshape(numTotal,1), grossRangeOffset_int.reshape(numTotal,1)))
        print("grossOffsets: \n", grossOffsets_int, grossOffsets_int.dtype)
        grossOffsets_int.tofile(os.path.join(self.outdir, self.outname))

        return 0

def main(iargs=None):
    inps = cmdLineParse(iargs)
    grossObj = grossOffsets(inps)
    grossObj.runGrossOffsets()

if __name__=='__main__':
    main()
