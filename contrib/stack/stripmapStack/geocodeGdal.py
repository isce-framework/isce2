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
    parser.add_argument('-b', '--bbox', dest='bbox', type=str, required=True,
            help='Bounding box (SNWE)')
    parser.add_argument('-x', '--lon_step', dest='lonStep', type=str, default=0.001,
            help='output pixel size (longitude) in degrees. Default 0.001')
    parser.add_argument('-y', '--lat_step', dest='latStep', type=str, default=0.001,
            help='output pixel size (latitude) in degrees. Default 0.001')
    parser.add_argument('-o', '--xoff', dest='xOff', type=int, default=0,
            help='Offset from the begining of geometry files in x direction. Default 0.0')
    parser.add_argument('-p', '--yoff', dest='yOff', type=int, default=0,
            help='Offset from the begining of geometry files in y direction. Default 0.0')
    parser.add_argument('-r', '--resampling_method', dest='resamplingMethod', type=str, default='near',
            help='Resampling method (gdalwarp resamplin methods)')

    return parser

def cmdLineParse(iargs = None):
    '''
    Command line parser.
    '''

    parser = createParser()
    inps =  parser.parse_args(args = iargs)

    inps.bbox = [val for val in inps.bbox.split()]
    if len(inps.bbox) != 4:
        raise Exception('Bbox should contain 4 floating point values')

    inps.prodlist = inps.prodlist.split()
    return inps

def prepare_lat_lon(inps):

    latFile = os.path.abspath(inps.latFile)
    lonFile = os.path.abspath(inps.lonFile)
    cmd = 'isce2gis.py vrt -i ' + latFile
    os.system(cmd)
    cmd = 'isce2gis.py vrt -i ' + lonFile
    os.system(cmd)
    
    width, length =  getSize(latFile)
    widthFile , lengthFile = getSize(inps.prodlist[0])
    
    xOff = inps.xOff
    yOff = inps.yOff

    tempLat = os.path.join(os.path.dirname(inps.prodlist[0]), 'tempLAT.vrt')
    tempLon = os.path.join(os.path.dirname(inps.prodlist[0]), 'tempLON.vrt')

    cmd = 'gdal_translate -of VRT -srcwin ' + str(xOff) + ' ' + str(yOff) \
           +' '+ str(width - xOff) +' '+ str(length - yOff) +' -outsize ' + str(widthFile) + \
           ' '+ str(lengthFile)  + ' -a_nodata 0 ' + latFile +'.vrt ' +  tempLat

    os.system(cmd)

    cmd = 'gdal_translate -of VRT -srcwin ' + str(xOff) + ' ' + str(yOff) \
          +' '+ str(int(width-xOff)) +' '+ str(int(length-yOff)) +' -outsize ' + str(widthFile) +\
           ' '+ str(lengthFile)  + ' -a_nodata 0 ' + lonFile +'.vrt ' +  tempLon

    os.system(cmd)

    return tempLat, tempLon

    # gdal_translate -of VRT -srcwin  384 384 64889 12785 -outsize 1013 199 ../../COMBINED/GEOM_MASTER/LAT.rdr LAT_off.vrt
    

def writeVRT(infile, latFile, lonFile):
#This function is modified from isce2gis.py
            latFile = os.path.abspath(latFile)
            lonFile = os.path.abspath(lonFile)
            infile = os.path.abspath(infile)
            cmd = 'isce2gis.py vrt -i ' + infile
            os.system(cmd)

            tree = ET.parse(infile + '.vrt')
            root = tree.getroot()

            meta = ET.SubElement(root, 'metadata')
            meta.attrib['domain'] = "GEOLOCATION"
            meta.tail = '\n'
            meta.text = '\n    '


            #rdict = { 'Y_DATASET' : os.path.relpath(latFile , os.path.dirname(infile)),
            #          'X_DATASET' :  os.path.relpath(lonFile , os.path.dirname(infile)),

            rdict = { 'Y_DATASET' : latFile ,
                      'X_DATASET' :  lonFile ,
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

    for rfile in inps.prodlist:
       cmd = 'isce2gis.py envi -i ' + rfile
       os.system(cmd)

    WSEN = str(inps.bbox[2]) + ' ' + str(inps.bbox[0]) + ' ' + str(inps.bbox[3]) + ' ' + str(inps.bbox[1])
    latFile, lonFile = prepare_lat_lon(inps)
    
    for rfile in inps.prodlist:
       rfile = os.path.abspath(rfile)
       print ('geocoding ' + rfile)
       #cmd = 'isce2gis.py vrt -i '+ rfile + ' --lon ' + lonFile + ' --lat '+ latFile
       #os.system(cmd)
       writeVRT(rfile, latFile, lonFile)

       cmd = 'gdalwarp -of ENVI -geoloc  -te '+ WSEN + ' -tr ' + str(inps.latStep) + ' ' + str(inps.lonStep) + ' -srcnodata 0 -dstnodata 0 ' + ' -r ' +inps.resamplingMethod +' ' + rfile +'.vrt ' + rfile + '.geo'
       print (cmd)
       os.system(cmd)
       write_xml(rfile + '.geo')

def getSize(f):

    ds=gdal.Open(f, gdal.GA_ReadOnly)
    b=ds.GetRasterBand(1)
    width = b.XSize
    length = b.YSize
    ds = None
    return width, length
       
def get_lat_lon(f):

    ds=gdal.Open(f, gdal.GA_ReadOnly)
    b=ds.GetRasterBand(1)
    width  = b.XSize
    length = b.YSize
    minLon = ds.GetGeoTransform()[0]
    deltaLon = ds.GetGeoTransform()[1]
    maxLat = ds.GetGeoTransform()[3]
    deltaLat = ds.GetGeoTransform()[5]
    minLat = maxLat + (b.YSize)*deltaLat
    ds = None
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


