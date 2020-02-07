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







import isce
from ctypes import cdll
import os
import sys
import urllib.request, urllib.error, urllib.parse
from isce import logging
from iscesys.Component.Component import Component
from contrib.demUtils.DemStitcher import DemStitcher as DS
#Parameters definitions
URL1 = Component.Parameter('_url1',
    public_name = 'URL1',default = 'http://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11',
    type = str,
    mandatory = False,
    doc = "Url for the high resolution DEM. Used for SRTM version3")
URL3 = Component.Parameter('_url3',
    public_name = 'URL3',default = 'http://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL3.003/2000.02.11',
    type = str,
    mandatory = False,
    doc = "Url for the low resolution DEM. Used for SRTM version3")
EXTRA_EXT1 = Component.Parameter('_extraExt1',
    public_name = 'extra extension 1',default = 'SRTMGL1',
    type = str,
    mandatory = False,
    doc = "The actual file name might have some extra string compared to the conventional one." \
          + "This is for the high resolution files. Used for SRTM version3")
EXTRA_EXT3 = Component.Parameter('_extraExt3',
    public_name = 'extra extension 3',default = 'SRTMGL3',
    type = str,
    mandatory = False,
    doc = "The actual file name might have some extra string compared to the conventional one." \
          + "This is for the low resolution files. Used for SRTM version3")
HAS_EXTRAS = Component.Parameter('_hasExtras',
    public_name = 'has extras',default = True,
    type = bool,
    mandatory = False,
    doc = "Instead of having to provide the EXTRA_EXT empty when the extra extension " \
+ "is not present, turn on this flag. Used for SRTM version3")

## This class provides a set of convenience method to retrieve and possibly combine different DEMs from  the USGS server.
# \c NOTE: the latitudes and the longitudes that describe the DEMs refer to the bottom left corner of the image.
class DemStitcher(DS):


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
        if(self._hasExtras):
            if(source and source == 1):
                toAppend = '.' + self._extraExt1
            elif(source and source == 3):
                toAppend = '.' + self._extraExt3
            else:
                print('Unrecognized dem source',source)
                raise Exception

            return ns + ew +  toAppend + self._extension +  self._zip

        else:
            return ns + ew +  self._extension +  self._zip


    def getUnzippedName(self,name,source = None):
        if(self._hasExtras):
            if(source and source == 1):
                name =  name.replace('.' + self._extraExt1,'')
            elif(source and source == 3):
                name =  name.replace('.' + self._extraExt3,'')

            else:
                print('Unrecognized dem source',source)
                raise Exception
        return name.replace(self._zip,'')

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
        for fileNow in listFile:
            url = self.getFullHttp(source)
            opener = urllib.request.URLopener()
            try:
                if not os.path.exists(os.path.join(downloadDir,fileNow)):
                        if(self._un is None or self._pw is None):
                            #opener.retrieve(url + fileNow,os.path.join(downloadDir,fileNow))
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
                        print(command)
                        if os.system(command):
                            os.chdir(cwd)
                            raise Exception
                        os.chdir(cwd)
                self._downloadReport[fileNow] = self._succeded
            except Exception as e:
                self.logger.warning('There was a problem in retrieving the file  %s. Exception %s'%(os.path.join(url,fileNow),str(e)))
                self._downloadReport[fileNow] = self._failed


    #still need to call it since the initialization calls the _url so the setter of
    #url does not get called
    def _configure(self):
        pass

    def _facilities(self):
        super(DemStitcher,self)._facilities()

    def __setstate__(self,d):
        self.__dict__.update(d)
        self.logger = logging.getLogger('isce.contrib.demUtils.DemStitcherV3')
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
                        self.correct(os.path.join(self._downloadDir,self._outputFile), \
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
            for k,v in list(self._downloadReport.items()):
                print(k,'=',v)

    #use this logic so the right http is returned

    def getFullHttp(self,source):
        return self._url1 if source == 1 else self._url3
    '''
    parameter_list = (
                      URL1,
                      URL3,
                      EXTRA_EXT1,
                      EXTRA_EXT3,
                      HAS_EXTRAS,
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
                      OUTPUT_FILE
                     )
    '''
    parameter_list = (
                      URL1,
                      URL3,
                      EXTRA_EXT1,
                      EXTRA_EXT3,
                      HAS_EXTRAS
                     ) + DS.parameter_list

    family = 'demstitcher'

    def __init__(self,family = '', name = ''):

        super(DemStitcher, self).__init__(family if family else  self.__class__.family, name=name)
        # logger not defined until baseclass is called
        self._extension = '.hgt'
        self._zip = '.zip'

        #to make it working with other urls, make sure that the second part of the url
        #it's /srtm/version2_1/SRTM(1,3)
        self._remove = ['.jpg','.xml']
        if not self.logger:
            self.logger = logging.getLogger('isce.contrib.demUtils.DemStitcherV3')
