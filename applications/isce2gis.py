#!/usr/bin/env python3

import isce
import isceobj
import argparse
import os
import xml.etree.ElementTree as ET
from imageMath import IML

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Export ISCE products directly to ENVI / VRT formats')
 
    subparsers = parser.add_subparsers(help='Output format options', dest='fmt')

    vrtparser = subparsers.add_parser( 'vrt', help='Export with VRT file')
    vrtparser.add_argument('-i', '--input', dest='infile', type=str, required=True,
            help='ISCE product file to export')
    vrtparser.add_argument('--lat', dest='latvrt', type=str, default=None,
            help='Location of the latitude file')
    vrtparser.add_argument('--lon', dest='lonvrt', type=str, default=None,
            help='Location of the longitude file')

    enviparser = subparsers.add_parser('envi', help='Export with ENVI hdr file')
    enviparser.add_argument('-i', '--input', dest='infile', type=str, required=True,
            help='ISCE product file to export')

    vals = parser.parse_args()
#    print(vals)
    return vals


def isce2envi(inname):
    '''
    Create ENVI hdr for ISCSE product.
    '''
    img, dataname, metaname = IML.loadImage(inname)
    img.renderEnviHDR()

    return


def isce2vrt(inname):
    '''
    Create VRT for ISCE product.
    '''
    img, dataname, metaname = IML.loadImage(inname)
    img.renderVRT()
    return


def getVRTinfo(inname):
    '''
    Verify if the lat / lon VRT info is appropriate.
    '''

    tree = ET.parse(inname.strip() + '.vrt')
    root = tree.getroot()

    width = int(root.attrib['rasterXSize'])
    length  = int(root.attrib['rasterYSize'])

    bands = len(root.find('VRTRasterBand'))

    if bands != 1:
        raise Exception('%s is not a one band image'%(inname+'.vrt'))

    return (width, length)
    
    

if __name__ == '__main__':
    '''
    Main driver.
    '''

    inps = cmdLineParse()

    if inps.fmt == 'envi':
        isce2envi(inps.infile)

    elif inps.fmt == 'vrt':
        
        if (inps.latvrt is None) or (inps.lonvrt is None):
            isce2vrt(inps.infile)

        else:
#            latf = inps.latvrt + '.vrt'
#            if not os.path.exists(latf):
            isce2vrt(inps.latvrt)

#            lonf = inps.lonvrt + '.vrt'
#            if not os.path.exists(lonf):
            isce2vrt(inps.lonvrt)
            
            latimg, dummy, dummy = IML.loadImage(inps.latvrt)
            latwid = latimg.getWidth()
            latlgt = latimg.getLength()
            if latimg.getBands() != 1:
                raise Exception('Latitude image should be single band')


            lonimg, dummy, dummy = IML.loadImage(inps.lonvrt)
            lonwid = lonimg.getWidth()
            lonlgt = lonimg.getLength()

            if lonimg.getBands() != 1:
                raise Exception('Longitude image should be single band')
            
            img = isceobj.createImage()
            img.load(inps.infile + '.xml')
            wid = img.getWidth()
            lgt = img.getLength()

            if any([(latwid - wid) != 0, (lonwid - wid) != 0]):
                raise Exception('Widths of image, lat and lon files dont match')

            if any([(latlgt - lgt) != 0, (lonlgt - lgt) != 0]):
                raise Exception('Lengths of image, lat and lon files dont match')

            ####Create prelim XML
            isce2vrt(inps.infile)
            tree = ET.parse(inps.infile + '.vrt')
            root = tree.getroot()

            meta = ET.SubElement(root, 'metadata')
            meta.attrib['domain'] = "GEOLOCATION"
            meta.tail = '\n'
            meta.text = '\n    '

            
            rdict = { 'Y_DATASET' : os.path.relpath(inps.latvrt + '.vrt', os.path.dirname(inps.infile)),
                      'X_DATASET' :  os.path.relpath(inps.lonvrt + '.vrt', os.path.dirname(inps.infile)),
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
            tree.write(inps.infile + '.vrt')
