#!/usr/bin/env python3


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2012 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






from __future__ import print_function
import isce
from ctypes import cdll, c_char_p, c_int, byref
from array import array
import struct
import zipfile
import os
import sys
import math
import urllib.request, urllib.parse, urllib.error
from isce import logging
from iscesys.Component.Component import Component

import xml.etree.ElementTree as ET
from html.parser import HTMLParser
class DemDirParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self._results = []
        self._filterList = []
        self._removeList = []
    @property
    def filterList(self):
        return self._filterList
    @filterList.setter
    def filterList(self,filterList):
        self._filterList = filterList
    @property
    def removeList(self):
        return self._removeList
    @removeList.setter
    def removeList(self,removeList):
        self._removeList = removeList
    @property
    def results(self):
        return self._results
    #implement the call back from data received
    def handle_data(self,data):
        #check that the data is one of the expected type
        #based on filtesList
        for filt in self.filterList:
            isOk = True
            #check that the data is not one that needs to be removed
            for rm in self.removeList:
                if data.count(rm):
                    isOk = False
                    break
            if isOk and data.count(filt):
                self._results.append(data.strip())



#Parameters definitions
URL = Component.Parameter('_url',
    public_name = 'URL',default = 'http://dds.cr.usgs.gov',
    type = str,
    mandatory = False,
    doc = "Top part of the url where the DEMs are stored. Used for SRTM version2")
USERNAME = Component.Parameter('_un',
    public_name='username',
    default = None,
    type = str,
    mandatory = False,
    doc = "Username in case the url is password protected")
PASSWORD = Component.Parameter('_pw',
    public_name='password',
    default = None,
    type = str,
    mandatory = False,
    doc = "Password in case the url is password protected")
KEEP_AFTER_FAILED = Component.Parameter('_keepAfterFailed',
    public_name='keepAfterFailed',
    default = False,
    type = bool,
    mandatory = False,
    doc = "If the stitching for some reason fails, it keeps the downloaded files.\n" +\
      "If 'useLocalDirectory' is set then this flag is forced to True to avoid \n" +\
      "accidental deletion of files (default: False)")
DIRECTORY = Component.Parameter('_downloadDir',
    public_name='directory',
    default = './',
    type = str,
    mandatory = False,
    doc = "If useLocalDirectory is False,it is used to download\n" + \
     "the files and create the stitched file, otherwise it assumes that this is the\n" + \
     "the local directory where the DEMs are  (default: current working directory)")
ACTION = Component.Parameter('_action',
    public_name='action',
    default = 'stitch',
    type = str,
    mandatory = False,
    doc = "Action to perform. Possible values are 'stitch' to stitch DEMs together\n" + \
        "or 'download' to download the DEMs  (default: 'stitch')")
CORRECT = Component.Parameter('_correct',
    public_name='correct',
    default = False,
    type = bool,
    mandatory = False,
    doc = "Apply correction  EGM96 -> WGS84 (default: True). The output metadata is in xml \n" +
    "format only")
META = Component.Parameter('_meta',
    public_name='meta',
    default = 'xml',
    type = str,
    mandatory = False,
    doc = "What type of metadata file is created. Possible values: xml  or rsc (default: xml)")
SOURCE = Component.Parameter('_source',
    public_name='source',
    default = 1,
    type = int,
    mandatory = False,
    doc = "DEM SRTM source. Possible values 1  or 3 (default: 1)")
NO_FILLING = Component.Parameter('_noFilling',
    public_name='noFilling',
    default = True,
    type = bool,
    mandatory = False,
    doc = "If the flag is False the missing DEMs are filled with null values \n" + \
        "(default: True, default null value -32768.")
FILLING_VALUE = Component.Parameter('_fillingValue',
    public_name='fillingValue',
    default = -32768,
    type = int,
    mandatory = False,
    doc = "Value used to fill missing DEMs (default: -32768)")
BBOX = Component.Parameter('_bbox',
    public_name='bbox',
    default = None,
    type = list,
    mandatory = False,
    doc = "Defines the spatial region in the format south north west east.\n" + \
        "The values should be integers from (-90,90) for latitudes and (0,360) or " +\
        "(-180,180) for longitudes.")
PAIRS = Component.Parameter('_pairs',
    public_name='pairs',
    default = None,
    type = list,
    mandatory =  False,
    doc = "Set of latitude and longitude pairs for which action = 'download' is performed.\n" +\
         "The values should be integers from (-90,90)\n" + \
          "for latitudes and (0,360) or (-180,180) for longitudes")
KEEP_DEMS = Component.Parameter('_keepDems',
    public_name='keepDems',
    default = False,
    type = bool,
    mandatory = False,
    doc = "If the option is present then the single files used for stitching are kept.\n" + \
         "If 'useLocalDirectory' is set then this flag is forced to True to avoid\n" + \
         "accidental deletion of files (default: False)'")
REPORT = Component.Parameter('_report',
    public_name='report',
    default = False,
    type = bool,
    mandatory = False ,
    doc = "If the option is present then failed and succeeded downloads are printed (default: False)")
USE_LOCAL_DIRECTORY = Component.Parameter('_useLocalDirectory',
    public_name='useLocalDirectory',
    default = False,
    type = bool,
    mandatory = False,
    doc = "If the option is True then use the files that are in the location\n" + \
         "specified by 'directory'. If not present 'directory' indicates\n" + \
         "the directory where the files are downloaded (default: False)")
OUTPUT_FILE = Component.Parameter('_outputFile',
    public_name='outputFile',
    default = None,
    type = str,
    mandatory = False,
    doc = "Name of the output file to be created in 'directory'.\n" + \
         "If not provided the system generates one based on the bbox extremes")

REGIONS = Component.Parameter('_regions',
    public_name='regions',
    default = None,
    type = list,
    mandatory = False,
    doc = "Regions where to look for the DEM files")

## This class provides a set of convenience method to retrieve and possibly combine different DEMs from  the USGS server.
# \c NOTE: the latitudes and the longitudes that describe the DEMs refer to the bottom left corner of the image.
class DemStitcher(Component):





    ##
    # Given a latitude and longitude in degrees it returns the expected filename.
    # @param lat \c int latitude in the range (-90,90). Actual data are restricted to (-60,60) or so.
    # @param lon \c int longitude in the range [-180,180) or [0,360).
    # @return \c string the filename for that location

    def createFilename(self,lat,lon,source = None):

        if lon > 180:
            lon = -(360 - lon)
        else:
            lon = lon
        ns,ew = self.convertCoordinateToString(lat,lon)
        return ns + ew +  self._extension +  self._zip
    ##
    # Given a rectangle (in latitude,longitude space) defined by a maximum and minimum latitude   and by a maximum and minimum longitude (in degrees) it returns
    # an ordered  list of the filenames defining the rectangle. The list is ordered first in ascending longitudes and teh ascending latitudes.
    # @param lats \c list \c int list containing the minimum and maximum latitudes in the range (-90,90). Actual data are restricted to (-60,60) or so.
    # @param lons \c list \c int list containing the minimum and maximum longitudes in the range [-180,180) or [0,360).
    # @return \c tuple (\list strings the list of filenames covering the specified area, \c int the number of frames found along the longitude direction,
    # \c int the number of frames found along the latitude direction)

    def createNameList(self,lats,lons,source = None):
        self._inputFileList = []
        if lons[0] > 180:
            lons[0] = -(360 - lons[0])
        else:
            lons[0] = lons[0]
        if lons[1] > 180:
            lons[1] = -(360 - lons[1])
        else:
            lons[1] = lons[1]

        lonMin = min(lons[0],lons[1])
        lons[1] = int(math.ceil(max(lons[0],lons[1])))
        lons[0] = int(math.floor(lonMin))
        #sanity check for lat
        latMin = min(lats[0],lats[1])
        lats[1] = int(math.ceil(max(lats[0],lats[1])))
        lats[0] = int(math.floor(latMin))
        # give error if crossing 180 and -180.
        latList = []
        lonList = []
        for i in range(lats[0],lats[1]): # this leave out lats[1], but is ok because the last frame will go up to that point
            latList.append(i)
        #for lat go north to south
        latList.reverse()
        # create the list starting from the min to the max
        if(lons[1] - lons[0] < 180):
            for i in range(lons[0],lons[1]): # this leave out lons[1], but is ok because the last frame will go up to that point
                lonList.append(i)
        else:
            print("Error. The crossing of E180 and W180 is not handled.")
            raise Exception
        self._latLonList = []
        for lat in latList:
            for lon in lonList:
                name = self.createFilename(lat,lon,source)
                self._inputFileList.append(name)
                self._latLonList.append([lat,lon])
        return self._inputFileList,len(latList),len(lonList)

    ##
    # Given a rectangle (in latitude,longitude space) defined by a maximum and minimum
    # latitude   and by a maximum and minimum longitude (in degrees) it fetches
    # the compressed (zip format) DEMs contained in that rectangle.
    # @param lats \c list \c ints list containing the minimum and maximum latitudes
    # in the range (-90,90). Actual data are restricted to (-60,60) or so.
    # @param lons \c list \c ints list containing the minimum and maximum longitudes
    # in the range [-180,180) or [0,360).
    # @param source \c int the type of DEM. source = 1 for 1 arcsec resolution data,
    # source = 3 for 3 arcsec resolution data.
    # @param downloadDir \c string the directory where the DEMs are downloaded.
    # If the directory does not exists it will be created. If the argument is not provided
    # then the files are downloaded in the location defined by the self._downloadDir
    # that is defaulted to the current directory.
    # @param region \c string region where to look for the files. If not provided the files
    # are searched by scanning the content of each region. Use method getRagionList to get
    # the list of possible region for a given source. Set region only if sure that all the
    # requested file are contained in it.
    def getDemsInBox(self,lats,lons,source,downloadDir = None,region = None):
        nameList,numLat,numLon, = self.createNameList(lats,lons,source)

        if downloadDir is None:
            downloadDir = self._downloadDir
        else:
            self._downloadDir = downloadDir
        #hackish. needs major refactoring. If self._useLocalDirectory is set we
        #need only the nameList, no need to download
        if not self._useLocalDirectory:
            if region:
                regionList = [region]*len(nameList)
            else:
                regionList = None

            self.getDems(source,nameList,downloadDir,regionList)

        else:
            #create a fake download report from the nameList
            files = os.listdir(downloadDir)
            for fileNow in nameList:
                #if file present then report success, failure otherwise
                if files.count(fileNow):
                    self._downloadReport[fileNow] = self._succeded
                else:
                    self._downloadReport[fileNow] = self._failed

        return nameList,numLat,numLon


    ##
    # Given a list of filenames  it fetches the corresponding
    # compressed (zip format) DEMs.
    # @param source \c int the type of DEM. source = 1 for 1 arcsec resolution data,
    # source = 3 for 3 arcsec resolution data.
    # @param listFile \c list of the filenames to be retrieved.
    # @param downloadDir \c string the directory where the DEMs are downloaded.
    # If the directory does not exists it will be created. If the argument is not
    # provided then the files are downloaded in the location defined by the
    # self._downloadDir that is defaulted to the current directory.
    # @param region \c  list \c strings regions where to look for the files. It must
    # have the same length of \c listFile. If not provided the files are searched by
    # scanning the content of each region. Use method getRegionList to get the list of
    # possible regions for a given source. Set region only if sure that all the requested
    # file are contained in it.

    def getDems(self,source,listFile,downloadDir = None,region = None):
        if downloadDir is None:
            downloadDir = self._downloadDir
        else:
            self._downloadDir = downloadDir

        if not (downloadDir) is  None:
            try:
                os.makedirs(downloadDir)
            except:
                #dir already exists
                pass
        if region:
            regionList = region
        #region unknown, so try all of them
        else:
            # the scanning of the regions is time comsuming. get all the files in all region and create a big list
            regionList = self.getRegionList(source)

        regionMapping = []
        fullList = []
        for regionNow in regionList:
            fileListUrl = self.getFileListPerRegion(source,regionNow)
            if fileListUrl:
                listNow = [file for file in fileListUrl]
                fullList.extend(listNow)
                regionNowMap = [regionNow]*len(fileListUrl)
                regionMapping.extend(regionNowMap)

        for fileNow in listFile:
            url = ''
            for i in range(len(fullList)):
                if fileNow == fullList[i]:
                    regionNow = regionMapping[i]
                    url = self.getFullHttp(source,regionNow)
                    break
            if not  (url == ''):
                try:
                    if not os.path.exists(os.path.join(downloadDir,fileNow)):
                        if(self._un is None or self._pw is None):
                            if os.path.exists(os.path.join(os.environ['HOME'],'.netrc')):
                                command = 'curl -n  -L -c $HOME/.earthdatacookie -b $HOME/.earthdatacookie -k -f -O ' + os.path.join(url,fileNow)
                            else:
                                self.logger.error('Please create a .netrc file in your home directory containing\nmachine urs.earthdata.nasa.gov\n\tlogin yourusername\n\tpassword yourpassword')
                                sys.exit(1)
                        else:
                            command = 'curl -k -f -u ' + self._un + ':' + self._pw + ' -O ' + os.path.join(url,fileNow)
                        # curl with -O download in working dir, so save current, move to donwloadDir
                        # nd get back once download is finished
                        cwd = os.getcwd()
                        os.chdir(downloadDir)
                        if os.system(command):
                            os.chdir(cwd)
                            raise Exception
                        os.chdir(cwd)
                    self._downloadReport[fileNow] = self._succeded
                except Exception as e:
                    self.logger.warning('There was a problem in retrieving the file  %s. Exception %s'%(os.path.join(url,fileNow),str(e)))
                    self._downloadReport[fileNow] = self._failed

            else:
                self._downloadReport[fileNow] = self._failed
    ##
    # After retriving DEMs this funtion prints the status of the download for each file, which could be 'succeded' or 'failed'

    def printDownloadReport(self):
        for k,v in self._downloadReport.items():
            print('Download of file',k,v,'.')
    ##
    # This function returns a dictionary whose keys are the attemped downloaded files and
    # the values are the status of teh download, 'succeed' or 'failed'.
    # @return \c dictionary whose keys are the attemped downloaded files and the values are
    # the status of teh download, 'succeed' or 'failed'.

    def getDownloadReport(self):
        return self._downloadReport

    ##
    # Given a list of latitudes and longitudes  it fetches the corresponding
    # compressed (zip format) DEMs.
    # @param lats \c list \c int list containing set of latitudes in the range (-90,90).
    # Actual data are restricted to (-60,60) or so.
    # @param lons \c list \c int list containing set of longitudes in the range [-180,180)
    # or [0,360).
    # @param source \c int the type of DEM. source = 1 for 1 arcsec resolution data,
    # source = 3 for 3 arcsec resolution data.
    # @param downloadDir \c string the directory where the DEMs are downloaded. If the
    # directory does not exists it will be created. If the argument is not provided then
    # the files are downloaded in the location defined by the self._downloadDir that is
    # defaulted to the current directory.
    # @param region \c  list \c strings regions where to look for the files. It must have
    # the same length of \c listFile. If not provided the files are searched by scanning
    # the content of each region. Use method getRagionList to get the list of possible
    # regions for a given source. Set region only if sure that all the requested file are
    # contained in it.

    def downloadFilesFromList(self,lats,lons,source,downloadDir = None,region = None):

        inputFileList = []
        for lat,lon in zip(lats,lons):
            name = self.createFilename(lat,lon,source)
            inputFileList.append(name)
        self.getDems(source,inputFileList,downloadDir,region)
    ##
    # Given a latitude and longitude  it fetches the corresponding
    # compressed (zip format) DEM.
    # @param lat \c list \c int  latitude in the range (-90,90). Actual data are restricted to (-60,60) or so.
    # @param lons \c list \c int longitude in the range [-180,180) or [0,360).
    # @param source \c int the type of DEM. source = 1 for 1 arcsec resolution data, source = 3 for 3 arcsec resolution data.
    # @param downloadDir \c string the directory where the DEMs are downloaded. If the directory does not exists it will be created. If the argument is not provided then the files are downloaded in the location defined by the self._downloadDir that is defaulted to the current directory.
    # @param region \c  list \c strings regions where to look for the files. It must have the same length of \c listFile. If not provided the files are searched by scanning the content of each region. Use method getRagionList to get the list of possible regions for a given source. Set region only if sure that all the requested file are contained in it.
    def downloadFile(self,lat,lon,source,downloadDir = None,region = None):
        name = self.createFilename(lat,lon,source)
        inputFileList =  [name]
        regionList = [region]
        self.getDems(source,inputFileList,downloadDir,regionList)
    ##
    # It returns the list of DEMs for a give source and region (if provided). If the region is not provided the full list of files for that source type is returned.
    # @param source \c int the type of DEM. source = 1 for 1 arcsec resolution data, source = 3 for 3 arcsec resolution data.
    # @param region \c  list \c strings regions where to look for the files. If the region is not provided the full list of files for that source type is returned.
    # @return \c list \c string list containing the the filenames found for the specific source and (if specified) region.

    def getFileList(self,source,region = None):
        retList = []
        if region:
            regionList = self.getRegionList(source)
            foundRegion = False
            for el in regionList:
                if el == region:
                    foundRegion = True
            if foundRegion:
                retList = self.getFileListPerRegion(source,region)
        else:
            regionList = self.getRegionList(source)
            for el in regionList:
                retList.extend(self.getFileListPerRegion(source,el))
        return retList

    ##
    # It returns the list of DEMs for a given source and region.
    # @param source \c int the type of DEM. source = 1 for 1 arcsec resolution data, source = 3 for 3 arcsec resolution data.
    # @param region \c  list \c strings regions where to look for the files.
    # @return \c list \c string list containing the the filenames found for the specific source and region.
    def getFileListPerRegion(self,source,region):
        url = self.getFullHttp(source,region)
        return self.getUrlList(url,self._filters['fileExtension'], self._remove)


    ##
    # It returns the list of regions for a given source.
    # @param source \c int the type of DEM. source = 1 for 1 arcsec resolution data, source = 3 for 3 arcsec resolution data.
    # @return \c list \c string list of region for the specified source.
    def getRegionList(self,source):
        # check first if it has been computed before
        if self._regionList[str(source)] == []:
            url = self.http + str(source)
            self._regionList[str(source)] = self.getUrlList(url,self._filters['region'+str(source)], self._remove)
        return self._regionList[str(source)]

    def getUrlList(self,url,filterList = None, removeList = None):
        if filterList is None:
            filterList = []
        if removeList is None:
            removeList = []
        if self._un is None or self._pw is None:
            fp = urllib.request.urlopen(url)
            allUrl = fp.read()
            fp.close()
        else:
            # create a password manager
            password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
            # Add the username and password.
            password_mgr.add_password(None,url,self._un,self._pw)
            handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
            # create "opener" (OpenerDirector instance)
            opener = urllib.request.build_opener(handler)
            # use the opener to fetch a URL
            allUrl = opener.open(url).read()

        ddp = DemDirParser()
        # feed the data from the read() of the url to the parser. It will call the DemDirParser.handle_data everytime t
        # a data type is parsed
        ddp.filterList = filterList
        ddp.removeList = removeList
        ddp.feed(allUrl.decode('utf-8', 'replace'))
        return ddp.results

    ##
    # Setter function for the download directory.
    # @param ddir \c string directory where the DEMs are downloaded. In self.stitchDem defines also the directory where the output stiched file is saved.

    def setDownloadDirectory(self,ddir):
        self._downloadDir = ddir

    ##
    # Fuction that decompress the given file in zip format.
    # @param filename \c strig  the name of the file to decompress.
    # @param downloadDir \c string the directory where the DEMs are downloaded. If the directory does not exists it will be created. If the argument is not provided then the files are downloaded in the location defined by the self._downloadDir that is defaulted to the current directory.
    def decompress(self,filename,downloadDir = None,keep = None):

        # keep .zip by default
        if keep == None:
            keep = True
        if downloadDir is None:
            downloadDir = self._downloadDir
        else:
            self._downloadDir = downloadDir

        filen = os.path.join(downloadDir,filename)
        try:
            #some system might not have zlib so you a system call to unzip
            zip = zipfile.ZipFile(filen,'r')
            import zlib
            zip.extractall(downloadDir)
        except:
            self.extract(downloadDir,filen)

        if not keep:
            os.remove(filen)

    def extract(self,downloadDir,filen):
        os.system('unzip -o -qq '    + os.path.join(filen)  + ' -d ' + downloadDir)


    def defaultName(self,snwe):
        latMin = math.floor(snwe[0])
        latMax = math.ceil(snwe[1])
        lonMin = math.floor(snwe[2])
        lonMax = math.ceil(snwe[3])
        nsMin,ewMin = self.convertCoordinateToString(latMin, lonMin)
        nsMax,ewMax = self.convertCoordinateToString(latMax, lonMax)
        demName = (
            'demLat_' + nsMin + '_' +nsMax +
            '_Lon_' + ewMin +
            '_' + ewMax  + '.dem'
            )

        return demName

    def convertCoordinateToString(self,lat,lon):

        if(lon > 180):
            lon = -(360 - lon)
        if(lon < 0):
            ew = 'W'
        else:
            ew = 'E'
        lonAbs = int(math.fabs(lon))
        if(lonAbs >= 100):
            ew += str(lonAbs)
        elif(lonAbs < 10):
            ew +=  '00' + str(lonAbs)
        else:
            ew +=  '0' + str(lonAbs)

        if(int(lat) >= 0):
            ns = 'N'
        else:
            ns = 'S'
        latAbs = int(math.fabs(lat))
        if(latAbs >= 10):
            ns += str(latAbs)
        else:
            ns += '0' +str(latAbs)

        return ns,ew


    #based on the source predict the width of the dem
    def getDemWidth(self,lon,source):
        if source == 3:
            factor = 1200
        else:
            factor = 3600
        return int(math.fabs((lon[1] - lon[0]))*factor)

    #this method also create an actual DeimImage object that is returned by the getImage() method
    def createXmlMetadata(self,lat,lon,source,outname):

        demImage = self.createImage(lat,lon,source,outname)
        demImage.renderHdr()

    def createImage(self,lat,lon,source,outname):
        from isceobj.Image import createDemImage

        demImage = createDemImage()
        if source == 3:
            delta = 1/1200.0
        else:
            delta = 1/3600.0

        try:
            os.makedirs(self._downloadDir)
        except:
            #dir already exists
            pass

        width = self.getDemWidth(lon,source)
        demImage.initImage(outname,'read',width)
        length = demImage.getLength()
        dictProp = {'METADATA_LOCATION':outname+'.xml','REFERENCE':self._reference,'Coordinate1':{'size':width,'startingValue':min(lon[0],lon[1]),'delta':delta},'Coordinate2':{'size':length,'startingValue':max(lat[0],lat[1]),'delta':-delta},'FILE_NAME':outname}
        #no need to pass the dictionaryOfFacilities since init will use the default one
        demImage.init(dictProp)
        self._image = demImage
        return demImage

##
#Function to indent an element of an ElementTree object. If the element passed is the root element, then all the ElementTree object is indented.
#@param elem element of an ElementTree object.

    def indent(self,elem, depth = None,last = None):
        if depth == None:
            depth = [0]
        if last == None:
            last = False
        tab = ' '*4
        if(len(elem)):
            depth[0] += 1
            elem.text = '\n' + (depth[0])*tab
            lenEl = len(elem)
            lastCp = False
            for i in range(lenEl):
                if(i == lenEl - 1):
                    lastCp = True
                self.indent(elem[i],depth,lastCp)
            if(not last):
                elem.tail = '\n' + (depth[0])*tab
            else:
                depth[0] -= 1
                elem.tail = '\n' + (depth[0])*tab
        else:
            if(not last):
                elem.tail = '\n' + (depth[0])*tab
            else:
                depth[0] -= 1
                elem.tail = '\n' + (depth[0])*tab

    def writeFileFromDictionary(self,file,dict, name = None):
        if not name:
            name = ''
        root = ET.Element('component')
        nameSubEl = ET.SubElement(root,'name')
        nameSubEl.text = name
        for key, val in dict.items():
            propSubEl = ET.SubElement(root,'property')
            ET.SubElement(propSubEl, 'name').text = key
            ET.SubElement(propSubEl, 'value').text = str(val)


        self.indent(root)
        etObj = ET.ElementTree(root)
        etObj.write(file)
    def createRscMetadata(self,lat,lon,source,outname):

        demImage = self.createImage(lat,lon,source,outname)

        dict = {'WIDTH':demImage.width,'LENGTH':demImage.length,'X_FIRST':demImage.coord1.coordStart,'Y_FIRST':demImage.coord2.coordStart,'X_STEP':demImage.coord1.coordDelta,'Y_STEP':-demImage.coord2.coordDelta,'X_UNIT':'degrees','Y_UNIT':'degrees'}
        try:
            os.makedirs(self._downloadDir)
        except:
            #dir already exists
            pass
        extension = '.rsc'
        outfile = outname + extension
        fp = open(outfile,'w')
        for k,v in dict.items():
            fp.write(str(k) + '\t' + str(v) + '\n')
        fp.close()

    def setKeepDems(self,val):
        self._keepDems = val

    def setCreateXmlMetadata(self,val):
        self._createXmlMetadata = val

    def setCreateRscMetadata(self,val):
        self._createRscMetadata = val

    def setMetadataFilename(self,demName):
        self._metadataFilename = demName

    def setFillingFilename(self,name):
        self._fillingFilename = name

    def setFillingValue(self,val):
        self._fillingValue = val

    def setFilling(self):
        self._noFilling = False

    def setNoFilling(self):
        self._noFilling = True

    def setUseLocalDirectory(self,val):
        self._useLocalDirectory = val
    def getUrl(self):
        return self._url
    def setUrl(self,url):
        self._url = url
        #after the url has been set generate the full path
        self._http = self._url + '/srtm/version2_1/SRTM'

    def setUsername(self,un):
        self._un = un

    def setPassword(self,pw):
        self._pw = pw

    def createFillingTile(self,source,swap,filename):
        fp = open(filename,'wb')
        numSamples = 1201
        if (source == 1):
            numSamples = 3601

        if swap:
            # pack it as a big endian and unpack it, and get the swapped number
            fillingValue = struct.unpack('h',struct.pack('>h',self._fillingValue))[0]
        else:
            fillingValue = self._fillingValue
        fullTile = [fillingValue]*numSamples*numSamples
        tile = array('h')
        tile.fromlist(fullTile)
        tile.tofile(fp)
        fp.close()


    #allow to overwrite from subclasses the nameing convention of the unzipped
    def getUnzippedName(self,name,source = None):
        return name.replace(self._zip,'')
    def stitchDems(self,lat,lon,source, outname, downloadDir = None,region = None, keep = None, swap = None):
        if downloadDir is None:
            downloadDir = self._downloadDir
        else:
            self._downloadDir = downloadDir

        swapFlag = 0
        if swap:
            if swap == True:#might be true or false
                swapFlag = 1
        else: # do it by default
            swapFlag = 1


        listNames,nLat,nLon = self.getDemsInBox(lat,lon,source,downloadDir,region)
        unzip = True
        #keep track of the synthetic ones since they don't need to be unzipped
        syntheticTiles = []
        if self._noFilling:
            #make sure that we have all the file to cover the region. check if some download failed
            for k,v in self._downloadReport.items():
                if v == self._failed:
                    unzip = False
                    #clean up the dowloaded files if it failed since when trying  a second source it might endup
                    #stitching them together beacaiuse it does not re-download the ones present and unfortunately
                    #the dems with different resolution have the same name convention
                    if not self._keepAfterFailed:
                        os.system("rm -rf " + downloadDir + "/*.hgt*")
                    break
        else:
            syntTileCreated = False
            #check and send a warning if the full region is not available
            if not self._succeded in self._downloadReport.values():
                self.logger.warning('The full region of interested is not available. A DEM with all null values will be created.')
            for k,v in self._downloadReport.items():
                if v == self._failed:#symlink each missing file to the reference one created in createFillingFile
                    if not syntTileCreated:#create the synthetic Tile the first time around
                        #get the abs path otherwise the symlink doesn't work
                        tileName = os.path.abspath(os.path.join(downloadDir,self._fillingFilename))
                        self.createFillingTile(source,swapFlag,tileName)
                        syntTileCreated = True

                    syntheticTiles.append(k)
                    demName = os.path.join(downloadDir,self.getUnzippedName(k,source))
                    #check for lexists so it returns also broken links, just in case something went wrong before
                    if os.path.lexists(demName):#clean up to make sure that old names are not there. will cause problem if use old one and the resolution od the dem is changed
                        os.remove(demName)
                    os.symlink(tileName,demName)

        if unzip:
            decompressedList = []
            for name in listNames:
                if not name in syntheticTiles:#synthetic tiles don't need to be decompressed
                    self.decompress(name,downloadDir,keep)

                newName = self.getUnzippedName(name,source)
                if downloadDir:
                    newName = os.path.join(downloadDir,newName)

                decompressedList.append(bytes(newName, 'utf-8'))
            numSamples = 1201
            if (source == 1):
                numSamples = 3601

            outname = os.path.join(downloadDir,outname)
            numFiles = [nLat,nLon]
            fileListIn_c = (c_char_p * len(decompressedList))()
            fileListIn_c[:] = decompressedList
            numFiles_c = (c_int * len(numFiles))()
            numFiles_c[:] = numFiles
            fileOut_c = c_char_p(bytes(outname, 'utf-8'))
            numSamples_c = c_int(numSamples)
            swapFlag_c = c_int(swapFlag)
            self._lib.concatenateDem(fileListIn_c,numFiles_c,fileOut_c,byref(numSamples_c),byref(swapFlag_c))

            if not self._keepDems:
                for dem in decompressedList:
                    os.remove(dem)
            if self._createXmlMetadata:
                self.createXmlMetadata(lat,lon,source,outname)
            if self._createRscMetadata:
                self.createRscMetadata(lat,lon,source,outname)

        return unzip #if False it means that failed

    ## Corrects the self._image from EGM96 to WGS84 and viceversa.
    #@param image \c Image if provided is used instead of the instance attribute self._image
    #@param conversionType \c int -1 converts from  EGM96 to WGS84, 1 converts from  WGS84 to EGM96
    #@return \c Image instance the converted Image
    def correct(self,image = None,conversionType=-1):
        '''Corrects the self._image from EGM96 to WGS84 and viceversa.'''
        from contrib.demUtils.Correct_geoid_i2_srtm import (
            Correct_geoid_i2_srtm
            )
        cg = Correct_geoid_i2_srtm()
        return cg(image,conversionType) if image else cg(self._image,conversionType)

    #still need to call it since the initialization calls the _url so the setter of
    #url does not get called
    def _configure(self):
        #after the url has been set generate the full path
        self._http = self._url + '/srtm/version2_1/SRTM'

    def getImage(self):
        return self._image

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        del d['_lib']
        return d

    def __setstate__(self,d):
        self.__dict__.update(d)
        self.logger = logging.getLogger('isce.contrib.demUtils.DemStitcher')
        libName = os.path.join(os.path.dirname(__file__),self._loadLibName)
        ##self._keepAfterFailed = False #if True keeps the downloaded files even if the stitching failed.
        self._lib = cdll.LoadLibrary(libName)
        return

    def main(self):
        # prevent from deliting local files
        if(self._useLocalDirectory):
            self._keepAfterFailed = True
            self._keepDems = True
        # is a metadata file is created set the right type
        if(self._meta == 'xml'):
            self.setCreateXmlMetadata(True)
        elif(self._meta == 'rsc'):
            self.setCreateRscMetadata(True)
        # check for the action to be performed
        if(self._action == 'stitch'):
            if(self._bbox):
                lat = self._bbox[0:2]
                lon = self._bbox[2:4]
                if (self._outputFile is None):
                    self._outputFile = self.defaultName(self._bbox)

                if not(self.stitchDems(lat,lon,self._source,self._outputFile,self._downloadDir, \
                        keep=self._keepDems)):
                    print('Could not create a stitched DEM. Some tiles are missing')
                else:
                    if(self._correct):
                        width = self.getDemWidth(lon,self._source)
                        self.correct()
                        #self.correct(self._output,self._source,width,min(lat[0],lat[1]),min(lon[0],lon[1]))
            else:
                print('Error. The "bbox" attribute must be specified when the action is "stitch"')
                raise ValueError
        elif(self._action == 'download'):
            if(self._bbox):
                lat = self._bbox[0:2]
                lon = self._bbox[2:4]
                self.getDemsInBox(lat,lon,self._source,self._downloadDir)
            #can make the bbox and pairs mutually esclusive if replace the if below with elif
            if(self._pairs):
                self.downloadFilesFromList(self._pairs[::2],self._pairs[1::2],self._source,self._downloadDir)
            if(not (self._bbox or self._pairs)):
                print('Error. Either the "bbox" attribute or the "pairs" attribute must be specified when --action download is used')
                raise ValueError

        else:
            print('Unrecognized action ',self._action)
            return

        if(self._report):
            for k,v in self._downloadReport.items():
                print(k,'=',v)


    def _facilities(self):
        super(DemStitcher,self)._facilities()
    def getFullHttp(self,source, region = None):
        toAppend = ''
        if region:
            toAppend = ('/' + region + '/')
        return self._http  + str(source)  + toAppend
    @property
    def http(self):
        return self._http
    family = 'demstitcher'
    parameter_list = (
                      URL,
                      USERNAME,
                      PASSWORD,
                      KEEP_AFTER_FAILED,
                      DIRECTORY,
                      ACTION,
                      CORRECT,
                      META,
                      SOURCE,
                      NO_FILLING,
                      FILLING_VALUE,
                      BBOX,
                      PAIRS,
                      KEEP_DEMS,
                      REPORT,
                      USE_LOCAL_DIRECTORY,
                      OUTPUT_FILE,
                      REGIONS
                      )
    def __init__(self,family = '', name = ''):

        self._loadLibName = "demStitch.so"
        libName = os.path.join(os.path.dirname(__file__),self._loadLibName)
        ##self._keepAfterFailed = False #if True keeps the downloaded files even if the stitching failed.
        self._lib = cdll.LoadLibrary(libName)
        self._downloadReport = {}
        # Note if _useLocalDirectory is True then the donwloadDir is the local directory
        ##self._downloadDir = os.getcwd()#default to the cwd
        self._inputFileList = []
        ##self._useLocalDirectory = False
        ##self._outputFile = ''
        ##self._un = un
        ##self._pw = pw
        self._extension = '.hgt'
        self._zip = '.zip'

        #to make it working with other urls, make sure that the second part of the url
        #it's /srtm/version2_1/SRTM(1,3)
        self._filters = {'region1':['Region'],'region3':['Africa','Australia','Eurasia','Islands','America'],'fileExtension':['.hgt.zip']}
        self._remove = ['.jpg']
        self._metadataFilename = 'fileDem.dem'
        self._createXmlMetadata = None
        self._createRscMetadata = None
        self._regionList = {'1':[],'3':[]}
        ##self._keepDems = False
        self._fillingFilename = 'filling.hgt' # synthetic tile to cover holes
        ##self._fillingValue = -32768 # fill the synthetic tile with this value
        ##self._noFilling = False
        self._failed = 'failed'
        self._succeded = 'succeded'
        self._image = None
        self._reference = 'EGM96'
        super(DemStitcher, self).__init__(family if family else  self.__class__.family, name=name)
        # logger not defined until baseclass is called

        if not self.logger:
            self.logger = logging.getLogger('isce.contrib.demUtils.DemStitcher')

    url = property(getUrl,setUrl)
