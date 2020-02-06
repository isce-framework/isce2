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
# Author: Piyush Agram
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




import numpy as np
import sys
import math
from html.parser import HTMLParser
import urllib.request, urllib.parse, urllib.error
from isce import logging
from iscesys.Component.Component import Component
import zipfile
import os
import glob


class WaterBody(object):
    '''
    Class for dealing with SRTM water body shapes.
    '''

    def __init__(self, shapefiles=None):

        self.shapefiles = shapefiles

    def mergeShapes(self, outname):
        '''
        Merge all input shapefiles into a single shape file.
        '''
        try:
            from osgeo import ogr, osr
        except:
            raise Exception('Need OGR/GDAL python bindings to deal with shapefiles.')

        driver = ogr.GetDriverByName('ESRI Shapefile')
        if os.path.exists(outname):
            driver.DeleteDataSource(outname)

        layername = os.path.splitext(os.path.basename(outname))[0]

        dstshp = driver.CreateDataSource(outname)

        for num,infile in enumerate(self.shapefiles):
            srcshp = ogr.Open(infile)
            lyrshp = srcshp.GetLayer()
            srs = lyrshp.GetSpatialRef()
            if srs is None:
                srs = osr.SpatialReference()
                srs.SetWellKnownGeogCS("WGS84")

            inLayerDefn = lyrshp.GetLayerDefn()

            if num==0:
                dstlayer = dstshp.CreateLayer(layername, geom_type=lyrshp.GetGeomType(), srs=srs)

                for i in range(inLayerDefn.GetFieldCount()):
                    fieldDefn = inLayerDefn.GetFieldDefn(i)
                    dstlayer.CreateField(fieldDefn)

            for feat in lyrshp:
                out_feat = ogr.Feature(inLayerDefn)
                out_feat.SetGeometry(feat.GetGeometryRef().Clone())
                for i in range(inLayerDefn.GetFieldCount()):
                    out_feat.SetField(inLayerDefn.GetFieldDefn(i).GetNameRef(), feat.GetField(i))

                dstlayer.CreateFeature(out_feat)


            lyrshp = None
            srcshp = None

        dstshp = None

    def rasterize(self, snwe, dims, shapefile, outname):

        try:
            from osgeo import ogr, osr, gdal
        except:
            raise Exception('Need OGR/GDAL python bindings to deal with shapefiles.')


        src = ogr.Open(shapefile)
        lyr = src.GetLayer()
        
        srs = lyr.GetSpatialRef()
        deltax = np.abs((snwe[3] - snwe[2])/(dims[0]*1.0))
        deltay = np.abs((snwe[1] - snwe[0])/(1.0*dims[1]))

        geotransform = [snwe[2], deltax, 0.0, snwe[1], 0.0, -deltay]

        driver = gdal.GetDriverByName('MEM')
        dst = driver.Create('', dims[0], dims[1], 1, gdal.GDT_Byte)
        dst.SetGeoTransform(geotransform)
        dst.SetProjection(srs.ExportToWkt())
        dst.GetRasterBand(1).Fill(1) 
        err = gdal.RasterizeLayer(dst, [1], lyr,
                burn_values=[0],options = ["ALL_TOUCHED=TRUE"])

        edriver = gdal.GetDriverByName('ENVI')
        edriver.CreateCopy(outname, dst, 0)
    
        lyr = None
        src = None
        dst = None


        return



class SWBDDirParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self._results = []
        self._filterList = []
        self._removeList = []

    @property
    def filterlist(self):
        return self._filterList

    @filterlist.setter
    def filterList(self,filterList):
        self._filterList = filterList

    @property
    def removeList(self):
        return self._removeList

    @removeList.setter
    def removeList(self, removeList):
        self._removeList = removeList

    @property
    def results(self):
        return self._results

    def handle_data(self,data):
        for filt in self.filterList:
            isOk = True
            for rm in self._removeList:
                if data.count(rm):
                    isOk = False
                    break

            if isOk and data.count(filt):
                self._results.append(data.strip())

####Actual land water mask parameters
#Parameters definitions
URL = Component.Parameter('_url',
    public_name = 'URL',default = 'http://dds.cr.usgs.gov',
    type = str,
    mandatory = False,
    doc = "Top part of the url where the Masks are stored  (default: http://dds.cr.usgs.gov)")
USERNAME = Component.Parameter('_un',
    public_name='username',
    default = '',
    type = str,
    mandatory = False,
    doc = "Username in case the url is password protected")
PASSWORD = Component.Parameter('_pw',
    public_name='password',
    default = '',
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
    default = os.getcwd(),
    type = str,
    mandatory = False,
    doc = "If useLocalDirectory is False,it is used to download\n" + \
     "the files and create the stitched file, otherwise it assumes that this is the\n" + \
     "the local directory where the Masks are  (default: current working directory)")
ACTION = Component.Parameter('_action',
    public_name='action',
    default = 'stitch',
    type = str,
    mandatory = False,
    doc = "Action to perform. Possible values are 'stitch' to stitch Masks together\n" + \
        "or 'download' to download the Masks  (default: 'stitch')")
META = Component.Parameter('_meta',
    public_name='meta',
    default = 'xml',
    type = str,
    mandatory = False,
    doc = "What type of metadata file is created. Possible values: xml  or rsc (default: xml)")
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
KEEP_MASKS = Component.Parameter('_keepMasks',
    public_name='keepMasks',
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
    doc = "Regions where to look for the Mask files")

WIDTH = Component.Parameter('_width',
        public_name='width',
        default = None,
        type=int,
        mandatory=True,
        doc='Width of output mask')

LENGTH = Component.Parameter('_length',
        public_name='length',
        default = None,
        type=int,
        mandatory=True,
        doc='Length of output mask')

LAT_FIRST = Component.Parameter('_firstLatitude',
        public_name = 'firstLatitude',
        default=None,
        type=float,
        mandatory=True,
        doc='First latitude')

LON_FIRST = Component.Parameter('_firstLongitude',
        public_name = 'firstLongitude',
        default=None,
        type=float,
        mandatory=True,
        doc='First longitude')

LAT_LAST = Component.Parameter('_lastLatitude',
        public_name='lastLatitude',
        default=None,
        type=float,
        mandatory=True,
        doc='Last Latitude')

LON_LAST = Component.Parameter('_lastLongitude',
        public_name='lastLongitude',
        default=None,
        type=float,
        mandatory=True,
        doc='Last Longitude')


class MaskStitcher(Component):

    def createFilename(self, lat, lon):
        '''
        Creates the file name for the archive containing the given point.
        Based on DEM stitcher's functions.
        '''

        if lon > 180:
            lon = -(360 - lon)
        else:
            lon = lon

        ns, ew = self.convertCoordinateToString(lat,lon)
        return ew+ns


    def createNameList(self, lats, lons):
        '''
        Creates the list of tiles that need to be downloaded.
        '''

        inputList = []

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
            print("Error. The crossing of E180 and E0 is not handled.")
            raise Exception

        for lat in latList:
            for lon in lonList:
                name = self.createFilename(lat,lon)
                inputList.append(name)

        prestring = inputList[0][0]
        for kk in inputList:
            if not kk.startswith(prestring):
                raise Exception('Cross of the date line / meridian not handled')

        return inputList,len(latList),len(lonList)


    def convertCoordinateToString(self,lat,lon):
        '''
        Based on dem stitcher.
        '''

        if(lon > 180):
            lon = -(360 - lon)
        if(lon < 0):
            ew = 'w'
        else:
            ew = 'e'
        lonAbs = int(math.fabs(lon))
        if(lonAbs >= 100):
            ew += str(lonAbs)
        elif(lonAbs < 10):
            ew +=  '00' + str(lonAbs)
        else:
            ew +=  '0' + str(lonAbs)

        if(int(lat) >= 0):
            ns = 'n'
        else:
            ns = 's'
        latAbs = int(math.fabs(lat))
        if(latAbs >= 10):
            ns += str(latAbs)
        else:
            ns += '0' +str(latAbs)

        return ns,ew

    def getMasksInBox(self,lats,lons,downloadDir = None,region = None):
        nameList,numLat,numLon, = self.createNameList(lats,lons)

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

            fetchList = self.getMasks(nameList,downloadDir,regionList)

        else:
            #create a fake download report from the nameList
            files = os.listdir(downloadDir)
            for fileNow in nameList:
                #if file present then report success, failure otherwise
                if files.count(fileNow):
                    self._downloadReport[fileNow] = self._succeded
                else:
                    self._downloadReport[fileNow] = self._failed

        return fetchList,numLat,numLon

    def getMasks(self,listFile,downloadDir = None,region = None):
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
            regionList = self.getRegionList()

        regionMapping = []
        fullList = []
        for regionNow in regionList:
#            fileListUrl = self.getFileListPerRegion(regionNow)
            fileListUrl = regionList[regionNow]
            if fileListUrl:
                listNow = [file for file in fileListUrl]
                fullList.extend(listNow)
                regionNowMap = [regionNow]*len(fileListUrl)
                regionMapping.extend(regionNowMap)

        downloadList = []
        for fileNow in listFile:
            url = ''
            fileMatch = ''
            for i in range(len(fullList)):
#                if fileNow == fullList[i]:

                 if fullList[i].startswith(fileNow):
                    regionNow = regionMapping[i]
                    fileMatch = fullList[i]
                    if regionNow == 'W':
                        url = self._http + '/SWBDwest/'
                    elif regionNow == 'E':
                        url = self._http + '/SWBDeast/'
                    else:
                        raise Exception('Unknown region: %s'%regionNow)

                    break
            if not  (url == '') and not (fileMatch == ''):
                opener = urllib.request.URLopener()
                try:
                    if not os.path.exists(os.path.join(downloadDir,fileMatch)):
                        if(self._un is None or self._pw is None):
                            opener.retrieve(url + fileMatch,os.path.join(downloadDir,fileMatch))
                        else:
                            # curl with -O download in working dir, so save current, move to donwloadDir
                            # nd get back once download is finished
                            cwd = os.getcwd()
                            os.chdir(downloadDir)
                            command = 'curl -k -u ' + self._un + ':' + self._pw + ' -O ' + os.path.join(url,fileMatch)
                            if os.system(command):
                                raise Exception

                            os.chdir(cwd)


                    print('Unzipping : ', fileMatch)
                    command = 'unzip ' + os.path.join(downloadDir,fileMatch)
                    if os.system(command):
                        raise Exception

                    self._downloadReport[fileMatch] = self._succeded
                except Exception as e:
                    self.logger.warning('There was a problem in retrieving the file  %s. Exception %s'%(os.path.join(url,fileNow),str(e)))
                    self._downloadReport[fileMatch] = self._failed

                downloadList.append(fileMatch)
            else:
                self._downloadReport[fileMatch] = self._failed

        return downloadList

    def printDownloadReport(self):
        for k,v in self._downloadReport.items():
            print('Download of file',k,v,'.')


    def getDownloadReport(self):
        return self._downloadReport


    def downloadFilesFromList(self,lats,lons,downloadDir = None,region = None):

        inputFileList = []
        for lat,lon in zip(lats,lons):
            name = self.createFilename(lat,lon)
            inputFileList.append(name)
        self.getDems(inputFileList,downloadDir,region)

    def getFileList(self,region = None):
        retList = []
        if region:
            regionList = self.getRegionList()
            foundRegion = False
            for el in regionList:
                if el == region:
                    foundRegion = True
            if foundRegion:
                retList = self.getFileListPerRegion(region)
        else:
            regionList = self.getRegionList()
            for el in regionList:
                retList.extend(self.getFileListPerRegion(el))
        return retList

    def getFileListPerRegion(self,region):
        if region=='W':
            url = self._http + '/SWBDwest/'
        elif region=='E':
            url = self._http + '/SWBDeast/'
        else:
            raise Exception('Unknown region: %s'%region)

        print('Url: ', url)
        return self.getUrlList(url,self._filters['fileExtension'])


    def getRegionList(self):
        # check first if it has been computed before

        for kk in  self._regionList:
            self._regionList[kk] = self.getFileListPerRegion(kk)

        return self._regionList

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

        ddp = SWBDDirParser()
        # feed the data from the read() of the url to the parser. It will call the DemDirParser.handle_data everytime t
        # a data type is parsed
        ddp.filterList = filterList
        ddp.removeList = removeList
        ddp.feed(allUrl.decode('utf-8', 'replace'))
        return ddp.results

    def setDownloadDirectory(self,ddir):
        self._downloadDir = ddir


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
            '_' + ewMax  + '.msk'
            )

        return demName

    def createXmlMetadata(self,outname):

        demImage = self.createImage(outname)
        demImage.renderHdr()

    def createImage(self,outname):
        from isceobj.Image import createDemImage

        demImage = createDemImage()

        try:
            os.makedirs(self._downloadDir)
        except:
            #dir already exists
            pass

        width = self._width
        demImage.initImage(outname,'read',width)
        demImage.dataType='BYTE'
        length = demImage.getLength()
        deltaLat = (self._lastLatitude - self._firstLatitude)/ (length-1.0)
        deltaLon = (self._lastLongitude - self._firstLongitude)/ (width-1.0)
        dictProp = {'REFERENCE':self._reference,'Coordinate1':{'size':width,'startingValue':self._firstLongitude,'delta':deltaLon},'Coordinate2':{'size':length,'startingValue':self._firstLatitude,'delta':-deltaLat},'FILE_NAME':outname}
        #no need to pass the dictionaryOfFacilities since init will use the default one
        demImage.init(dictProp)
        self._image = demImage
        return demImage


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


    def createRscMetadata(self,lat,lon,outname):

        demImage = self.createImage(lat,lon,outname)

        dict = {'WIDTH':demImage.width,'LENGTH':demImage.length,'X_FIRST':demImage.coord1.coordStart,'Y_FIRST':demImage.coord2.coordStart,'X_STEP':demImage.coord1.coordDelta,'Y_STEP':-demImage.coord2.coordDelta,'X_UNIT':'degrees','Y_UNIT':'degrees'}
        try:
            os.makedirs(self._downloadDir)
        except:
            #dir already exists
            pass
        extension = '.rsc'
        outfile = os.path.join(self._downloadDir,outname + extension)
        fp = open(outfile,'w')
        for k,v in dict.items():
            fp.write(str(k) + '\t' + str(v) + '\n')
        fp.close()


    def setKeepMasks(self,val):
        self._keepMasks = val

    def setCreateXmlMetadata(self,val):
        self._createXmlMetadata = val

    def setCreateRscMetadata(self,val):
        self._createRscMetadata = val

    def setMetadataFilename(self,demName):
        self._metadataFilename = demName

    def setFirstLatitude(self, val):
        self._firstLatitude = float(val)

    def setFirstLongitude(self, val):
        self._firstLongitude = float(val)

    def setLastLatitude(self,val):
        self._lastLatitude = float(val)

    def setLastLongitude(self,val):
        self._lastLongitude = float(val)

    def setWidth(self, val):
        self._width = int(val)

    def setLength(self, val):
        self._length = int(val)

    def setUseLocalDirectory(self,val):
        self._useLocalDirectory = val
    def getUrl(self):
        return self._url
    def setUrl(self,url):
        self._url = url
        #after the url has been set generate the full path
        self._http = self._url + '/srtm/version2_1/SWBD'

    def setUsername(self,un):
        self._un = un

    def setPassword(self,pw):
        self._pw = pw

    def stitchMasks(self,lat,lon,outname, downloadDir = None,region = None, keep = None):

        if downloadDir is None:
            downloadDir = self._downloadDir
        else:
            self._downloadDir = downloadDir


        listNames,nLat,nLon = self.getMasksInBox(lat,lon,downloadDir,region)
        print(listNames)
        unzip = True

        outname = os.path.join(downloadDir,outname)
        print('Output: ', outname)
        if self._firstLatitude is None:
            self._firstLatitude = max(lat)

        if self._lastLatitude is None:
            self._lastLatitude = min(lat)

        if self._firstLongitude is None:
            self._firstLongitude = min(lon)

        if self._lastLongitude is None:
            self._lastLongitude = max(lon)

        if self._width is None:
            self._width = int(1200 * (self._lastLatitude - self._firstLatitude))

        if self._length is None:
            self._length = int(1200* (self._lastLongitude - self._firstLatitude))


        #####Deals with rasterization
        fixedNames = []
        for name in listNames:
            fixedNames.append(name.replace('.zip','.shp'))

        sh = WaterBody(fixedNames)
        shpname = os.path.splitext(outname)[0] + '.shp'
        sh.mergeShapes(shpname)

        sh.rasterize([self._lastLatitude, self._firstLatitude,self._firstLongitude, self._lastLongitude], [self._width, self._length], shpname, outname)



        if not self._keepMasks:
            for kk in listNames:
                os.remove(os.path.join(downloadDir,kk))

            for kk in glob.glob(os.path.join(downloadDir,'*.shp')):
                os.remove(kk)
            for kk in glob.glob(os.path.join(downloadDir,'*.shx')):
                os.remove(kk)

            for kk in glob.glob(os.path.join(downloadDir,'*.dbf')):
                os.remove(kk)

        if self._createXmlMetadata:
                self.createXmlMetadata(outname)
        if self._createRscMetadata:
            self.createRscMetadata(outname)

        return unzip #if False it means that failed


    def _configure(self):
        #after the url has been set generate the full path
        self._http = self._url + '/srtm/version2_1/SWBD'

    def getImage(self):
        return self._image

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        del d['_lib']
        return d

    def __setstate__(self,d):
        self.__dict__.update(d)
        self.logger = logging.getLogger('isce.contrib.demUtils.MaskStitcher')
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
                        self.correct(os.path.join(self._dir,self._outputFile), \
                                     self._source,width,min(lat[0],lat[1]),min(lon[0],lon[1]))
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


    family = 'maskstitcher'
    parameter_list = (
                      URL,
                      USERNAME,
                      PASSWORD,
                      KEEP_AFTER_FAILED,
                      DIRECTORY,
                      ACTION,
                      META,
                      BBOX,
                      PAIRS,
                      KEEP_MASKS,
                      REPORT,
                      USE_LOCAL_DIRECTORY,
                      OUTPUT_FILE,
                      REGIONS,
                      WIDTH,
                      LENGTH,
                      LAT_FIRST,
                      LON_FIRST,
                      LAT_LAST,
                      LON_LAST
                      )


    def __init__(self,family = '', name = ''):

        self._downloadReport = {}
        # Note if _useLocalDirectory is True then the donwloadDir is the local directory
        self._inputFileList = []
        self._extension = '.shp'
        self._zip = '.zip'

        #to make it working with other urls, make sure that the second part of the url
        #it's /srtm/version2_1/SRTM(1,3)
        #self._filters = {'region1':['Region'],'region3':['Africa','Australia','Eurasia','Islands','America'],'fileExtension':['.hgt.zip']}
        self._filters = {'fileExtension' : ['.zip']}
        self._remove = ['.jpg']
        self._metadataFilename = 'fileDem.dem'
        self._createXmlMetadata = None
        self._createRscMetadata = None
        self._regionList = {'W':[],'E':[]}
        self._failed = 'failed'
        self._succeded = 'succeded'
        self._image = None
        self._reference = 'EGM96'
        super(MaskStitcher, self).__init__(family if family else  self.__class__.family, name=name)
        # logger not defined until baseclass is called

        if not self.logger:
            self.logger = logging.getLogger('isce.contrib.demUtils.MaskStitcher')

    utl = property(getUrl,setUrl)


if __name__ == '__main__':

    '''
    Testing with w123n37.shp
    '''
    sh = WaterBody()
    sh.push('w123n37.shp')
#    sh.plot()
    sh.createGrid([37.0,38.0,-123.0,-122.0], 1201, 1201, 'test.msk')
#    Can view test.msk using "mdx -s 1201
