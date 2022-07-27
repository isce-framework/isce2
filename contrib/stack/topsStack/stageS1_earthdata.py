#!/usr/bin/env python3

# Authors: Piyush Agram, Emre Havazli
# Copyright 2021

import os
import netrc
import base64
import zipfile
import logging
import argparse
from osgeo import gdal

from urllib.request import build_opener, install_opener, Request, urlopen
from urllib.request import HTTPHandler, HTTPSHandler, HTTPCookieProcessor
from urllib.error import HTTPError, URLError
from http.cookiejar import MozillaCookieJar


class SentinelVRT:
    """
    Class for virtual download of S1 products.
    """
    def __init__(self, url, dest):
        """
        Constructor with URL.
        """
        # URL
        self.url = url

        # Destination folder
        self.dest = os.path.join(dest, os.path.basename(url))

        # Product Type
        if "IW_GRD" in self.url:
            self.productType = "GRD"
        elif "IW_SLC" in self.url:
            self.productType = "SLC"
        else:
            raise Exception("Product type could not be determined for: "
                            "{0}".format(self.url))

        # Write dummy zip file to test output can be written
        if os.path.exists(self.dest):
            print("Destination zip file already exists. "
                  "Will be overwritten ....")
            os.remove(self.dest)
        self.createZip()

        # Fetch manifest
        self.IPF = None  # TODO: Get calibration XML for IPF 2.36-low priority
        self.fetchManifest()

        # Fetch annotation
        self.fetchAnnotation()

        # Fetch images - TODO: GRD support
        if self.productType == "SLC":
            self.fetchSLCImagery()

    def createZip(self):
        """
        Create local zip file to populate.
        """
        try:
            with zipfile.ZipFile(self.dest, mode='w') as myzip:
                with myzip.open('download.log', 'w') as myfile:
                    myfile.write('Downloaded with ISCE2\n'.encode('utf-8'))
        except:
            raise Exception('Could not create zipfile: {0}'.format(self.dest))

    def fetchManifest(self):
        """
        Fetch manifest.safe
        """
        try:
            res = gdal.ReadDir(self.srcsafe)
            if 'manifest.safe' not in res:
                raise Exception("Manifest file not found in "
                                "{0}".format(self.srcsafe))
        except:
            raise Exception("Could not fetch manifest from "
                            "{0}".format(self.srcsafe))

        try:
            with zipfile.ZipFile(self.dest, mode='a') as myzip:
                with myzip.open(os.path.join(self.zip2safe,'manifest.safe'),
                                             'w') as myfile:
                    logging.info('Fetching manifest.safe')
                    self.downloadFile(os.path.join(self.srcsafe,
                                      'manifest.safe'), myfile)

        except:
            raise Exception("Could not download manifest.safe from "
                            "{0} to {1}".format(self.url, self.dest))

    def fetchAnnotation(self):
        """
        Fetch annotation files.
        """
        dirname = os.path.join(self.srcsafe, 'annotation')
        res = gdal.ReadDir(dirname)

        try:
            with zipfile.ZipFile(self.dest, mode='a') as myzip:
                for ii in res:
                    if ii.endswith('.xml'):
                        srcname = os.path.join(dirname, ii)
                        destname = os.path.join(self.zip2safe,
                                                'annotation', ii)
                        logging.info('Fetching {0}'.format(srcname))
                        with myzip.open(destname, 'w') as myfile:
                            self.downloadFile(srcname, myfile)
        except:
            raise Exception("Could not download {0} from {1} to "
                            "{2}".format(ii, self.url, self.dest))

    def fetchSLCImagery(self):
        """
        Create VRTs for TIFF files.
        """
        import isce
        from isceobj.Sensor.TOPS.Sentinel1 import Sentinel1

        dirname = os.path.join(self.srcsafe, 'measurement')
        res = gdal.ReadDir(dirname)

        # If more were known about the tiff, this can be improved
        vrt_template = """<VRTDataset rasterXSize="{samples}" rasterYSize="{lines}">
    <VRTRasterBand dataType="CInt16" band="1">
        <NoDataValue>0.0</NoDataValue>
        <SimpleSource>
            <SourceFilename relativeToVRT="0">{tiffname}</SourceFilename>
            <SourceBand>1</SourceBand>
            <SourceProperties RasterXSize="{samples}" RasterYSize="{lines}" DataType="CInt16" BlockXSize="{samples}" BlockYSize="1"/>
            <SrcRect xOff="0" yOff="0" xSize="{samples}" ySize="{lines}"/>
            <DstRect xOff="0" yOff="0" xSize="{samples}" ySize="{lines}"/>
        </SimpleSource>
    </VRTRasterBand>
</VRTDataset>"""

        # Parse annotation files to have it ready with information
        for ii in res:
            parts = ii.split('-')
            swath = int(parts[1][-1])
            pol = parts[3]

            # Read and parse metadata for swath
            xmlname = ii.replace('.tiff', '.xml')

            try:
                reader = Sentinel1()
                reader.configure()
                reader.xml = [os.path.join("/vsizip", self.dest,
                                           self.zip2safe, 'annotation',
                                           xmlname)]
                reader.manifest = [os.path.join("/vsizip", self.dest,
                                                self.zip2safe,
                                                'manifest.safe')]
                reader.swathNumber = swath
                reader.polarization = pol
                reader.parse()

                vrtstr = vrt_template.format(
                    samples=reader.product.bursts[0].numberOfSamples,
                    lines=(reader.product.bursts[0].numberOfLines *
                           len(reader.product.bursts)),
                    tiffname=os.path.join(self.srcsafe, 'measurement', ii))

                #Write the VRT to zip file
                with zipfile.ZipFile(self.dest, mode='a') as myzip:
                    destname = os.path.join(self.zip2safe, 'measurement',
                                            ii)
                    with myzip.open(destname, 'w') as myfile:
                        myfile.write(vrtstr.encode('utf-8'))
            except:
                raise Exception("Could not create vrt for {0} at {1} in "
                                "{2}".format(ii, self.url, self.dest))

    @property
    def vsi(self):
        return os.path.join('/vsizip/vsicurl', self.url)

    @property
    def srcsafe(self):
        return os.path.join(self.vsi, self.zip2safe)

    @property
    def zip2safe(self):
        """
        Get safe directory path from zip name.
        """
        return os.path.basename(self.url).replace('.zip', '.SAFE')

    @staticmethod
    def downloadFile(inname, destid):

        # Get file size
        stats = gdal.VSIStatL(inname)
        if stats is None:
            raise Exception('Could not get stats for {0}'.format(inname))

        # Copy file to local folder
        success = False
        while not success:
            try:
                vfid = gdal.VSIFOpenL(inname, 'rb')
                data = gdal.VSIFReadL(1, stats.size, vfid)
                gdal.VSIFCloseL(vfid)
                success = True
            except AttributeError as errmsg:
                if errmsg.endswith('307'):
                    print('Redirected on {0}. Retrying ... '.format(inname))
            except Exception as err:
                print(err)
                raise Exception('Could not download file: {0}'.format(inname))

        # Write to destination id
        destid.write(data)


def cmdLineParse():
    """
    Command line parser.
    """

    parser = argparse.ArgumentParser(
             description='Download S1 annotation files with VRT pointing to '
                         'tiff files')
    parser.add_argument('-i', '--input', dest='inlist', type=str,
                        required=True, help='Text file with URLs to fetch')
    parser.add_argument('-o', '--output', dest='outdir', type=str,
                        default='.', help='Output folder to store the data in')
    parser.add_argument('-c', '--cookies', dest='cookies', type=str,
                        default='asfcookies.txt', help='Path to cookies file')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        default=False, help='Set to CPL_DEBUG to ON')

    return parser.parse_args()


def main(inps=None):
    """
    Main driver.
    """

    # check if output directory exists
    if os.path.isdir(inps.outdir):
        print('Output directory {0} exists'.format(inps.outdir))
    else:
        print('Creating output directory {0}'.format(inps.outdir))
        os.mkdir(inps.outdir)

    # Setup GDAL with cookies
    gdal.UseExceptions()

    gdal.SetConfigOption('GDAL_HTTP_COOKIEFILE', inps.cookies)
    gdal.SetConfigOption('GDAL_HTTP_COOKIEJAR', inps.cookies)
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
    if inps.debug:
        gdal.SetConfigOption('CPL_DEBUG', 'ON')
        gdal.SetConfigOption('CPL_CURL_VERBOSE', 'YES')
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Read in URLs into a list
    urlList = []
    try:
        with open(inps.inlist, 'r') as fid:
            for cnt, line in enumerate(fid):
                urlList.append(line.strip())

    except:
        raise Exception('Could not parse input file "{0}" as a list of line '
                        'separated URLs'.format(inps.inlist))

    for url in urlList:
        logging.info('Downloading: {0}'.format(url))
        downloader = SentinelVRT(url, inps.outdir)


if __name__ == '__main__':
    # Parse command line
    inps = cmdLineParse()

    # Process
    main(inps)
