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



from iscesys.Component.Component import Component
import numpy as np
import os
import abc
from iscesys.Stitcher.Stitcher import Stitcher as ST
from iscesys.DataRetriever.DataRetriever import DataRetriever as DR
DTYPE = Component.Parameter('_dtype',
    public_name = 'dtype',
    default = '',
    type = str,
    mandatory = True,
    doc = 'Data type')
OUTPUT_FILE = Component.Parameter('_outputFile',
    public_name='outputFile',
    default = '',
    type = str,
    mandatory = True,
    doc = 'Output file.')
TILE_SIZE = Component.Parameter('_tileSize',
    public_name = 'tileSize',
    default = [],
    container=list,
    type=int,
    mandatory = True,
    doc = 'Two element list with the number of row and columns of the tile.')
OVERLAP = Component.Parameter('_overlap',
    public_name = 'overlap',
    default = [1,1],
    container=list,
    type=int,
    mandatory = False,
    doc = 'Number of overlapping pixels  between two tiles along the rows and columns.')
URL = Component.Parameter('_url',
    public_name = 'URL',default = '',
    type = str,
    mandatory = False,
    doc = "URL where to get the data from")
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
DIRECTORY = Component.Parameter('_downloadDir',
    public_name='directory',
    default = './',
    type = str,
    mandatory = False,
    doc = "Location where the files are downloaded")
KEEP = Component.Parameter('_keep',
    public_name='keep',
    default = False,
    type = bool,
    mandatory = False,
    doc = "Keep the files downloaded after stitching")
ENDIAN = Component.Parameter('_endian',
    public_name = 'endian',
    default = '>',
    type = str,
    mandatory = False,
    doc = 'Data endianness. > big endian, < small endian')
USE_LOCAL = Component.Parameter('_useLocal',
    public_name='useLocal',
    default = False,
    type = bool,
    mandatory = False,
    doc = "If the option is True then use the files that are in the location\n" + \
         "specified by 'directory' for stitching. If not present 'directory' indicates\n" + \
         "the directory where the files are downloaded.\n " + \
         "When 'useLocal' is True then 'keep' is considered False\n " +\
         "to avoid accidental removal  of user files (default: False)")
FILLING_VALUE = Component.Parameter('_fillingValue',
    public_name = 'fillingValue',
    default = 0,
    type=float,
    mandatory = True,
    doc = 'Value used for missing tiles.')
NO_FILLING = Component.Parameter('_noFilling',
    public_name='noFilling',
    default = True,
    type = bool,
    mandatory = False,
    doc = "If the flag is False the missing tiles are filled with 'fillingValue' values" )
PROCEED_IF_NO_SERVER = Component.Parameter(
    '_proceedIfNoServer',
    public_name='proceed if no server',
    default=False,
    type=bool,
    mandatory=False,
    doc='Flag to continue even if server is down.'
)

class TileManager(Component,metaclass=abc.ABCMeta):
    family = 'tilemanager'
    parameter_list = (
                      URL,
                      USERNAME,
                      PASSWORD,
                      DIRECTORY,
                      DTYPE,
                      OUTPUT_FILE,
                      TILE_SIZE,
                      OVERLAP,
                      KEEP,
                      ENDIAN,
                      FILLING_VALUE,
                      USE_LOCAL,
                      NO_FILLING,
                      PROCEED_IF_NO_SERVER
                      )
    ##
    # Abstract method to create a filename based on lat and lon
    # Given a latitude and longitude in degrees it returns the expected filename.
    # @param lat \c int latitude in the range (-90,90). 
    # @param lon \c int longitude in the range [-180,180)
    # @return \c string the filename for that location
    @abc.abstractmethod
    def createFilename(self,lat,lon):
        pass
    ##
    #Abstract method to create an image instance
    #@return \c Image instance
    @abc.abstractmethod
    def createImage(self,lats,lons):
        pass

    ## Convenience method to create a list of file names from bounding box
    # which can be generated by the lat and lon.
    # Given a rectangle (latitude,longitude) defined by a maximum and minimum latitude   and by a maximum and minimum longitude (in degrees) it returns
    # an ordered  list of the filenames defining the rectangle. The list is ordered first in ascending longitudes and then ascending latitudes.
    # @param lats \c list \c int list containing the minimum and maximum latitudes in the range (-90,90). 
    # @param lons \c list \c int list containing the minimum and maximum longitudes in the range [-180,180).
    # @return \c tuple (\list strings the list of filenames covering the specified area, \c int the number of frames found along the longitude direction,
    # \c int the number of frames found along the latitude direction)
    #NOTE: createFilename needs to be implemented
    def createNameListFromBounds(self,lats,lons):
        self._inputFileList = []
        
        lons = sorted(lons)
        lats = sorted(lats)
        lons[1] = int(np.ceil(lons[1]))
        lons[0] = int(np.floor(lons[0]))
        lats[1] = int(np.ceil(lats[1]))
        lats[0] = int(np.floor(lats[0]))
        #lats are from larger to smaller
        latList = np.arange(lats[0],lats[1])[::-1]
        lonList = np.arange(lons[0],lons[1])
        # give error if crossing 180 and -180.

        if(lons[1] - lons[0] < 180):
             lonList = np.arange(lons[0],lons[1])
        else:
            print("Error. The crossing of E180 and W180 is not handled.")
            raise Exception
        for lat in latList:
            for lon in lonList:
                name = self.createFilename(lat,lon)
                self._inputFileList.append(name)
        return self._inputFileList,len(latList),len(lonList)
    ## Convenience method to create a list of file names from two lists of lats and lons.   
    # @param lats \c list \c int list containing the minimum and maximum latitudes in the range (-90,90). 
    # @param lons \c list \c int list containing the minimum and maximum longitudes in the range [-180,180).
    # @return \c tuple (\list strings the list of filenames covering the specified area, \c int the number of frames found along the longitude direction,
    # \c int the number of frames found along the latitude direction)
    #NOTE: createFilename needs to be implemented
    def createNameList(self,lats,lons):        
        return [self.createFilename(lat, lon) for lat,lon in zip(lats,lons)]

    def configureStitcher(self,names,arrangement):
        self._stitcher.configure()
        self._stitcher.arrangement = arrangement
        self._stitcher.tileSize = self._tileSize
        self._stitcher.fileList = names
        self._stitcher.dtype = self._dtype
        self._stitcher.outputFile = self._outputFile
        self._stitcher.endian = self._endian 
        self._stitcher.directory = self._downloadDir
        self._stitcher._fillingValue = self._fillingValue

    
    def configureRetriever(self):
        self._retriever.configure()
        self._retriever.url = self._url
        self._retriever.pw = self._pw
        self._retriever.un = self._un
        self._retriever.downloadDir = self._downloadDir
        self._retriever.proceedIfNoServer = self._proceedIfNoServer

    def getFileList(self,names,report,map):
        ret = []
        for name in names:
            if name in report and report[name] == self._retriever._succeded:
                #the map returns a list of file that normally should have only
                #one element
                ret.append(map[name][0])
            else:
                ret.append(self._stitcher._toSkipName)
        return ret
    
    def stitch(self,lats,lons):
        result = True
        names,nlats,nlons = self.createNameListFromBounds(lats, lons)
        self.configureStitcher(names, [nlats,nlons])
        if not self._useLocal: 
            self.configureRetriever()
            self._retriever.getFiles(names)
            self._stitcher.fileList = self.getFileList(names,self._retriever._downloadReport,
                                                        self._retriever._namesMapping)
            
            #the second part checks that everything was downloaded
            if self._noFilling and self._stitcher._toSkipName in self._stitcher.fileList:
                result = False
                self.clean()
        else:
            self._stitcher.fileList = names
        if result:                                        
            self._stitcher.stitch()
            self.createXml(lats,lons)
            if (not self._keep) and (not self._useLocal):
                self.clean()
        return result     
    
    def clean(self):
        for name in self._stitcher.fileList:
            if not name == self._stitcher._toSkipName:
                os.remove(name)
    def createXml(self,lats,lons):
        image = self.createImage(lats,lons,self.outputFile)
        self._image = image
        image.dump(self.outputFile + '.xml')
    
    def download(self,lats,lons,fromBounds=True):
        if fromBounds:
            names,nlats,nlons = self.createNameListFromBounds(lats,lons)
        else:
            names = self.createNameList(lats,lons)
        self.configureRetriever()
        self._retriever.getFiles(names)
    
    @property
    def proceedIfNoServer(self):
        return self._proceedIfNoServer
    @proceedIfNoServer.setter
    def proceedIfNoServer(self,proceedIfNoServer):
        self._proceedIfNoServer = proceedIfNoServer
    @property
    def url(self):
        return self._url
    @url.setter
    def url(self,url):
        self._url = url
    @property
    def un(self):
        return self._un   
    @un.setter
    def un(self,un):
        self._un = un
    @property
    def pw(self):
        return self._pw
    @pw.setter
    def pw(self,pw):
        self._pw = pw
    @property
    def dtype(self):
        return self._dtype
    @dtype.setter
    def dtype(self,val):
        self._dtype = val
    @property
    def outputFile(self):
        return self._outputFile
    @outputFile.setter
    def outputFile(self,val):
        self._outputFile = val
    @property
    def tileSize(self):
        return self._tileSize
    @tileSize.setter
    def tileSize(self,val):
        self._tileSize = val
    @property
    def overlap(self):
        return self._overlap
    @overlap.setter
    def overlap(self,val):
        self._overlap = val
    @property
    def keep(self):
        return self._keep
    @keep.setter
    def keep(self,val):
        self._keep = val
    @property
    def endian(self):
        return self._endian
    @endian.setter
    def endian(self,val):
        self._endian = val
    @property
    def fillValue(self):
        return self._fillValue
    @fillValue.setter
    def fillValue(self,val):
        self._fillValue = val
    @property
    def useLocal(self):
        return self._useLocal
    @useLocal.setter
    def useLocal(self,val):
        self._useLocal = val
    @property
    def noFilling(self):
        return self._noFilling
    @noFilling.setter
    def noFilling(self,val):
        self._noFilling = val
    @property
    def image(self):
        return self._image
    @image.setter
    def image(self,val):
        self._image = val
    ##
    # Setter function for the download directory.
    # @param ddir \c string directory where the data are downloaded.
    @property
    def downloadDir(self):
        return self._downloadDir
    @downloadDir.setter
    def downloadDir(self,ddir):
        self._downloadDir = ddir
    def __init__(self,family = '', name = ''):
        #the .configure() methods are called in  configureStitcher/Retriever
        self._retriever = DR()
        self._stitcher = ST()
        self._image = None

        super(TileManager, self).__init__(family if family else  self.__class__.family, name=name)
