#!/usr/bin/env python3
########################
#Author: Heresh Fattahi
#Copyright 2016
######################
import argparse
import isce
import isceobj
import os
import gdal
import numpy as np
import xml.etree.ElementTree as ET

def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser( description='Create DEM simulation for merged images')
    parser.add_argument('-l','--lat', dest='latFile', type=str, required=True,
            help = 'latitude file in radar coordinate')
    parser.add_argument('-L','--lon', dest='lonFile', type=str, required=True,
            help = 'longitude file in radar coordinate')
    parser.add_argument('-f', '--filelist', dest='prodlist', type=str, required=True,
            help='Input file to be geocoded')
    parser.add_argument('-o', '--xoff', dest='xOff', type=int, default=0,
            help='Offset from the begining of geometry files in x direction. Default 0.0')
    parser.add_argument('-p', '--yoff', dest='yOff', type=int, default=0,
            help='Offset from the begining of geometry files in y direction. Default 0.0')
    parser.add_argument('-r', '--resampling_method', dest='resamplingMethod', type=str, default='near',
            help='Resampling method (gdalwarp resamplin methods)')

    
    parser.add_argument('-b', '--bbox', dest='bbox', type=str, default='',
            help='Bounding box (SNWE)')
    parser.add_argument('-x', '--lon_step', dest='lonStep', type=str, default=0.001,
            help='output pixel size (longitude) in degrees. Default 0.001')
    parser.add_argument('-y', '--lat_step', dest='latStep', type=str, default=0.001,
            help='output pixel size (latitude) in degrees. Default 0.001')


    parser.add_argument('-t', '--tiff', dest='istiff', action='store_true', default=False,
            help='Create GEOTIFF instead of standard ENVI / ISCE files')
    parser.add_argument('--alex', dest='isAlexGrid', default=False, 
            action='store_true', help='Geocode to the Antarctica grid for optical offsets used by Alex Gardner')


    return parser

def cmdLineParse(iargs = None):
    '''
    Command line parser.
    '''

    parser = createParser()
    inps =  parser.parse_args(args = iargs)


    if not inps.isAlexGrid:
        inps.bbox = [val for val in inps.bbox.split()]
        if len(inps.bbox) != 4:
            raise Exception('Bbox should contain 4 floating point values')
        inps.outproj = 'EPSG:4326'
    else:
        print('Ignoring bbox and spacing inputs. Using standard grid for Antarctica.')
        inps.lonStep = 240.0
        inps.latStep = 240.0
        inps.outproj = 'EPSG:3031'

    inps.prodlist = inps.prodlist.split()
    return inps

def prepare_lat_lon(inps):

    latFile = os.path.abspath(inps.latFile)
    lonFile = os.path.abspath(inps.lonFile)
    #cmd = 'isce2gis.py vrt -i ' + latFile
    #os.system(cmd)
    #cmd = 'isce2gis.py vrt -i ' + lonFile
    #os.system(cmd)

   
    width, length =  getSize(latFile)
    widthFile , lengthFile = getSize(inps.prodlist[0])

    print("size of lat and lon files (width, length) ", width, length)
    print("size of input file to be geocoded (width, length): ", widthFile , lengthFile)

    xOff = inps.xOff
    yOff = inps.yOff

    cmd = 'gdal_translate -of VRT -srcwin ' + str(xOff) + ' ' + str(yOff) \
           +' '+ str(width - xOff) +' '+ str(length - yOff) +' -outsize ' + str(widthFile) + \
           ' '+ str(lengthFile)  + ' -a_nodata 0 ' + latFile +'.vrt' + ' tempLAT.vrt'

    os.system(cmd)

    cmd = 'gdal_translate -of VRT -srcwin ' + str(xOff) + ' ' + str(yOff) \
          +' '+ str(int(width-xOff)) +' '+ str(int(length-yOff)) +' -outsize ' + str(widthFile) +\
           ' '+ str(lengthFile)  + ' -a_nodata 0 ' + lonFile +'.vrt' + ' tempLON.vrt'

    os.system(cmd)

    return 'tempLAT.vrt', 'tempLON.vrt' 

    # gdal_translate -of VRT -srcwin  384 384 64889 12785 -outsize 1013 199 ../../COMBINED/GEOM_MASTER/LAT.rdr LAT_off.vrt
    

def writeVRT(infile, latFile, lonFile):
#This function is modified from isce2gis.py

            #cmd = 'isce2gis.py vrt -i ' + infile
            #os.system(cmd)

            tree = ET.parse(infile + '.vrt')
            root = tree.getroot()

            meta = ET.SubElement(root, 'metadata')
            meta.attrib['domain'] = "GEOLOCATION"
            meta.tail = '\n'
            meta.text = '\n    '


            rdict = { 'Y_DATASET' : os.path.relpath(latFile , os.path.dirname(infile)),
                      'X_DATASET' :  os.path.relpath(lonFile , os.path.dirname(infile)),
                      'X_BAND' : "1",
                      'Y_BAND' : "1",
                      'PIXEL_OFFSET': "0",
                      'LINE_OFFSET' : "0",
                      'LINE_STEP' : "1",
                      'PIXEL_STEP' : "1" }

            for key, val in rdict.items():
                data = ET.SubElement(meta, 'mdi')
                data.text = val
                data.attrib['key'] = key
                data.tail = '\n    '

            data.tail = '\n'
            tree.write(infile + '.vrt')


def runGeo(inps):

    #for file in inps.prodlist:
       #cmd = 'isce2gis.py envi -i ' + file
       #os.system(cmd)


    if not inps.isAlexGrid:
        WSEN = str(inps.bbox[2]) + ' ' + str(inps.bbox[0]) + ' ' + str(inps.bbox[3]) + ' ' + str(inps.bbox[1])
        latFile, lonFile = prepare_lat_lon(inps)

        getBound(latFile,float(inps.bbox[0]),float(inps.bbox[1]),'lat')
        getBound(lonFile,float(inps.bbox[2]),float(inps.bbox[3]),'lon')
    
        for infile in inps.prodlist:
            infile = os.path.abspath(infile)
            print ('geocoding ' + infile)
            outFile = os.path.join(os.path.dirname(infile), "geo_" + os.path.basename(infile))
            #cmd = 'isce2gis.py vrt -i '+ file + ' --lon ' + lonFile + ' --lat '+ latFile
            #os.system(cmd)
            writeVRT(infile, latFile, lonFile)

            cmd = 'gdalwarp -of ENVI -geoloc  -te '+ WSEN + ' -tr ' + str(inps.latStep) + ' ' + str(inps.lonStep) + ' -srcnodata 0 -dstnodata 0 ' + ' -r ' + inps.resamplingMethod + ' ' + infile +'.vrt '+ outFile
            print (cmd)
            os.system(cmd)

            write_xml(outFile)


    else:
        from geo2ant import getGridLimits
        ylims, xlims = getGridLimits(latfile=latFile, lonfile=lonFile)

        WSEN = str(xlim[0]) + ' ' + str(ylim[0]) + ' ' + str(xlim[1]) + ' ' + str(ylim[1])
        if inps.istiff:
            ext = '.tif'
            outformat = 'GTiff'
        else:
            ext = '.ant'
            outformat = 'ENVI'

        for infile in inps.prodlist:
            print('geocoding: ' + infile)

            writeVRT(infile, latFile, lonFile)

            cmd = 'gdalwarp -of ' + outformat + ' -t_srs ' + inps.outproj + ' -geoloc -te ' + WSEN + ' -tr ' + str(inps.lonStep) + ' ' + str(inps.latStep) + ' -srcnodata 0 -dstnodata 0 -r ' + inps.resamplingMethod + ' ' + infile + '.vrt ' + infile+'ext'
            status = os.system(cmd)
            if status:
                raise Exception('Command {0} Failed'.format(cmd))

            if not inps.istiff:
                write_xml(infile+ext)
        

def getSize(infile):    

    ds=gdal.Open(infile + ".vrt")
    b=ds.GetRasterBand(1)
    return b.XSize, b.YSize

def getBound(infile,minval,maxval,latlon): #added by Minyan Zhong
    
    ds=gdal.Open(infile)
    b=ds.GetRasterBand(1)
    data=b.ReadAsArray()

    idx=np.where((data>=minval) & (data<=maxval))
   
    if latlon=='lat':
        print('latitide bound in cliped area:')
    else:
        print('longitude bound in cliped area:')
    print(np.min(data[idx]),np.max(data[idx]))


def get_lat_lon(infile):

    ds=gdal.Open(infile)
    b=ds.GetRasterBand(1)
    width  = b.XSize
    length = b.YSize
    minLon = ds.GetGeoTransform()[0]
    deltaLon = ds.GetGeoTransform()[1]
    maxLat = ds.GetGeoTransform()[3]
    deltaLat = ds.GetGeoTransform()[5]
    minLat = maxLat + (b.YSize)*deltaLat

    return maxLat, deltaLat, minLon, deltaLon, width, length

def write_xml(outFile): 

    maxLat, deltaLat, minLon, deltaLon, width, length = get_lat_lon(outFile)

    unwImage = isceobj.Image.createImage()
    unwImage.setFilename(outFile)
    unwImage.setWidth(width)
    unwImage.setLength(length)
    unwImage.bands = 1
    unwImage.scheme = 'BIL'
    unwImage.dataType = 'FLOAT'
    unwImage.setAccessMode('read')
    
    unwImage.coord2.coordDescription = 'Latitude'
    unwImage.coord2.coordUnits = 'degree'
    unwImage.coord2.coordStart = maxLat 
    unwImage.coord2.coordDelta = deltaLat 
    unwImage.coord1.coordDescription = 'Longitude'
    unwImage.coord1.coordUnits = 'degree'
    unwImage.coord1.coordStart = minLon 
    unwImage.coord1.coordDelta = deltaLon 

   # unwImage.createImage()
    unwImage.renderHdr()
    unwImage.renderVRT()

def main(iargs=None):
    '''
    Main driver.
    '''
    inps = cmdLineParse(iargs)
    runGeo(inps)
 
   
if __name__ == '__main__':
    main()


