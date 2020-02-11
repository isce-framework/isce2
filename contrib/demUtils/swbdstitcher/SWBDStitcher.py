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
import numpy as np
import os
import sys
from isce import logging
import math
import urllib.request, urllib.parse, urllib.error
from iscesys.Component.Component import Component
from contrib.demUtils.DemStitcher import DemStitcher
from isceobj.Image import createImage
#Parameters definitions
URL = Component.Parameter('_url',
    public_name = 'URL',default = 'http://e4ftl01.cr.usgs.gov/SRTM/SRTMSWBD.003/2000.02.11',
    type = str,
    mandatory = False,
    doc = "Url for the high resolution water body mask")

KEEP_WBDS = Component.Parameter('_keepWbds',
    public_name='keepWbds',
    default = False,
    type = bool,
    mandatory = False,
    doc = "If the option is present then the single files used for stitching are kept.\n" + \
         "If 'useLocalDirectory' is set then this flag is forced to True to avoid\n" + \
         "accidental deletion of files (default: False)'")
## This class provides a set of convenience method to retrieve and possibly combine different DEMs from  the USGS server.
# \c NOTE: the latitudes and the longitudes that describe the DEMs refer to the bottom left corner of the image.
class SWBDStitcher(DemStitcher):

    def getUnzippedName(self,name,source = None):
        name =  name.replace('.' + self._extraExt,'')
        return name.replace(self._zip,'')
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
        toAppend = '.' + self._extraExt
        return ns + ew +  toAppend + self._extension +  self._zip

    def defaultName(self,snwe):
        latMin = math.floor(snwe[0])
        latMax = math.ceil(snwe[1])
        lonMin = math.floor(snwe[2])
        lonMax = math.ceil(snwe[3])
        nsMin,ewMin = self.convertCoordinateToString(latMin, lonMin)
        nsMax,ewMax = self.convertCoordinateToString(latMax, lonMax)
        swbdName = (
            'swbdLat_' + nsMin + '_' +nsMax +
            '_Lon_' + ewMin +
            '_' + ewMax  + '.wbd'
            )

        return swbdName
    @staticmethod
    def toRadar(maskin,latin,lonin,output):
        maskim = createImage()
        maskim.load(maskin + '.xml')
        latim = createImage()
        latim.load(latin + '.xml')
        lonim = createImage()
        lonim.load(lonin + '.xml')
        mask = np.fromfile(maskin,maskim.toNumpyDataType())
        lat = np.fromfile(latin,latim.toNumpyDataType())
        lon = np.fromfile(lonin,lonim.toNumpyDataType())
        mask = np.reshape(mask,[maskim.coord2.coordSize,maskim.coord1.coordSize])
        startLat  = maskim.coord2.coordStart
        deltaLat  = maskim.coord2.coordDelta
        startLon  = maskim.coord1.coordStart
        deltaLon  = maskim.coord1.coordDelta
        #remember mask starts from top left corner
        #deltaLat < 0
        lati = np.clip(((lat - startLat)/deltaLat).astype(np.int), 0, mask.shape[0]-1)
        loni = np.clip(((lon - startLon)/deltaLon).astype(np.int), 0, mask.shape[1]-1)
        cropped = (mask[lati,loni] + 1).astype(maskim.toNumpyDataType())
        cropped = np.reshape(cropped,(latim.coord2.coordSize,latim.coord1.coordSize))
        cropped.tofile(output)
        croppedim = createImage()
        croppedim.initImage(output,'read',cropped.shape[1],maskim.dataType)
        croppedim.renderHdr()

    def createImage(self,lat,lon,source,outname):


        image = createImage()

        delta = 1/3600.0

        try:
            os.makedirs(self._downloadDir)
        except:
            #dir already exists
            pass

        width = self.getDemWidth(lon,1)
        image.initImage(outname,'read',width,'BYTE')
        length = image.getLength()

        dictProp = {'METADATA_LOCATION':outname+'.xml','Coordinate1':{'size':width,'startingValue':min(lon[0],lon[1]),'delta':delta},'Coordinate2':{'size':length,'startingValue':max(lat[0],lat[1]),'delta':-delta},'FILE_NAME':outname}
        #no need to pass the dictionaryOfFacilities since init will use the default one
        image.init(dictProp)
        self._image = image
        return image
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
    def stitchWbd(self,lat,lon,outname, downloadDir = None, keep = None):
        if downloadDir is None:
            downloadDir = self._downloadDir
        else:
            self._downloadDir = downloadDir

        tileSize = 3600
        source = 1
        listNames,nLat,nLon = self.getDemsInBox(lat,lon,source,downloadDir)
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
                        os.system("rm -rf " + downloadDir + "/*.raw*")
                    break
        else:
            syntTileCreated = False
            #check and send a warning if the full region is not available
            if not self._succeded in self._downloadReport.values():
                self.logger.warning('The full region of interested is not available. Missing region is assumed to be land')
            for k,v in self._downloadReport.items():
                if v == self._failed:#symlink each missing file to the reference one created in createFillingFile
                    if not syntTileCreated:#create the synthetic Tile the first time around
                        #get the abs path otherwise the symlink doesn't work
                        syntTileCreated = True

                    syntheticTiles.append(k)

        if unzip:
            mmap = np.memmap(outname,np.int8,'w+',shape=(nLat*tileSize,nLon*tileSize))
            mmap[:,:] = 0
            decompressedList = []
            pos = 0
            for i in range(nLat):
                for j in range(nLon):
                    name =  listNames[pos]
                    if  name in syntheticTiles:#synthetic tiles don't need to be decompressed
                        pos += 1
                        continue
                    self.decompress(name,downloadDir,keep)

                    newName = self.getUnzippedName(name,source)
                    if downloadDir:
                        newName = os.path.join(downloadDir,newName)

                    decompressedList.append(bytes(newName, 'utf-8'))
                    data = np.reshape(np.fromfile(newName,np.int8),(3601,3601))
                    mmap[i*tileSize:(i+1)*tileSize,j*tileSize:(j+1)*tileSize] = data[:-1,:-1]
                    pos += 1

            if not self._keepWbds:
                for f in decompressedList:
                    os.remove(f)
            if self._createXmlMetadata:
                self.createXmlMetadata(lat,lon,source,outname)
        return unzip #if False it means that failed

    #still need to call it since the initialization calls the _url so the setter of
    #url does not get called
    def _configure(self):
        pass

    def _facilities(self):
        super(DemStitcher,self)._facilities()

    def __setstate__(self,d):
        self.__dict__.update(d)
        self.logger = logging.getLogger('isce.contrib.demUtils.SWBDStitcher')

        return

    def getWbdsInBox(self,lat,lon,downloadDir=None):
        self.getDemsInBox(lat,lon,1,downloadDir)




    def updateParameters(self):
        self.extendParameterList(DemStitcher,SWBDStitcher)
        super(SWBDStitcher,self).updateParameters()

    #use this logic so the right http is returned


    def getFullHttp(self,source):
        return self._url

    parameter_list = (
                      URL,
                      KEEP_WBDS
                     )

    family = 'swbdstitcher'

    def __init__(self,family = '', name = ''):

        super(SWBDStitcher, self).__init__(family if family else  self.__class__.family, name=name)
        # logger not defined until baseclass is called
        self._extension = '.raw'
        self._zip = '.zip'
        self._extraExt = 'SRTMSWBD'
        #to make it working with other urls, make sure that the second part of the url
        #it's /srtm/version2_1/SRTM(1,3)
        self._remove = ['.jpg','.xml']
        if not self.logger:
            self.logger = logging.getLogger('isce.contrib.demUtils.SWBDStitcher')

        self.parameter_list = self.parameter_list + super(DemStitcher,self).parameter_list
        self.updateParameters()
