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
import zipfile
import os
import sys
from isce import logging
from iscesys.Component.Component import Component
import shutil
from urllib import request
from urllib.parse import urlparse
import time
#Parameters definitions
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
    doc = "Location where the file are downloaded")
WAIT = Component.Parameter('_wait',
    public_name='wait',
    default = 5,
    type = float,
    mandatory = False,
    doc = "Wait time between trials when server is down")
NUM_TRIALS = Component.Parameter('_numTrials',
    public_name='number of trials',
    default = 3,
    type = int,
    mandatory = False,
    doc = "Number of times it tries to download the file when server is down")
PROCEED_IF_NO_SERVER = Component.Parameter(
    '_proceedIfNoServer',
    public_name='proceed if no server',
    default=False,
    type=bool,
    mandatory=False,
    doc='Flag to continue even if server is down.'
)
## This class provides a set of convenience method to retrieve and possibly combine different DEMs from  the USGS server.
# \c NOTE: the latitudes and the longitudes that describe the DEMs refer to the bottom left corner of the image.
class DataRetriever(Component):

    def serverUp(self,url,needCredentials=False):
        urlp = urlparse(url)
        server = urlp.scheme + "://" + urlp.netloc
        ret = False
        if needCredentials:
            try:
                request.urlopen(server)
                ret = True
            except Exception as e:
                try:
                    #when server needs credentials trying the url open fails
                    #with one of the below messages
                    if e.reason.reason.count('CERTIFICATE_VERIFY_FAILED'):
                        ret = True
                except:
                    try:
                         if ''.join(e.reason.split()).lower() == 'authorizationrequired':
                             ret = True
                    except:
                        #then assume that the exception was due to the server down
                        ret = False
        else:
            try:
                request.urlopen(server)
                ret = True
            except Exception:
                #in this case assume directly server down
                ret = False

        return ret

    ##
    # Fetches the files in listFiles from  URL
    # @param listFile \c list of the filenames to be retrieved.

    def getFiles(self,listFile):
        try:
            os.makedirs(self._downloadDir)
        except:
            #dir already exists
            pass
        #curl with -O downloads in working dir, so save cwd
        cwd = os.getcwd()
        #move to _downloadDir
        os.chdir(self._downloadDir)
        for fileNow in listFile:
            reason = 'file'
            for i in range(self._numTrials):
                try:
                    if not os.path.exists(fileNow):
                        if(self._un is None or self._pw is None):
                            if not self.serverUp(self._url):
                                reason = 'server'
                                raise Exception
                            if os.path.exists(os.path.join(os.environ['HOME'],'.netrc')):
                                command = 'curl -n  -L -c $HOME/.earthdatacookie -b $HOME/.earthdatacookie -k -f -O ' + os.path.join(self._url,fileNow)
                                print("command = {}".format(command))
                            else:
                                self.logger.error('Please create a .netrc file in your home directory containing\nmachine urs.earthdata.nasa.gov\n\tlogin yourusername\n\tpassword yourpassword')
                                sys.exit(1)

                        else:
                            if not self.serverUp(self._url,True):
                                reason = 'server'
                                raise Exception
                            command = 'curl -k -f -u ' + self._un + ':' + self._pw + ' -O ' + os.path.join(self._url,fileNow)
                        if os.system(command):
                            raise Exception
                    self._downloadReport[fileNow] = self._succeded
                    break
                except Exception as e:
                    if reason == 'file':
                        self.logger.warning('There was a problem in retrieving the file  %s. Requested file seems not present on server.'%(os.path.join(self._url,fileNow)))
                        #if the problem is file missing break the loop that tries when the server is down
                        self._downloadReport[fileNow] = self._failed
                        break
                    elif reason == 'server':
                        if i == self._numTrials - 1 and not self._proceedIfNoServer:
                            self.logger.error('There was a problem in retrieving the file  %s. Check the name of the server or try again later in case the server is momentarily down.'%(os.path.join(self._url,fileNow)))
                            sys.exit(1)
                        if i == self._numTrials - 1 and  self._proceedIfNoServer:
                            self._downloadReport[fileNow] = self._failed
                        else:
                            time.sleep(self._wait)
        #move back to original directory
        self.decompressFiles(listFile,self._downloadReport,os.getcwd())
        self.clean(listFile,self._downloadReport)
        os.chdir(cwd)


    def decompressFiles(self,listFile,report,cwd='./'):
        import tempfile as tf
        for file in listFile:
            if report[file] == self._succeded:
                td = tf.TemporaryDirectory()
                self.decompress(file,td.name)
                self._namesMapping[file] = os.listdir(td.name)
                for name in self._namesMapping[file]:
                    try:
                        shutil.move(os.path.join(td.name,name),cwd)
                    except Exception:
                        #probably file already exists. Remove it and try again
                        try:
                            os.remove(os.path.join(cwd,name))
                            shutil.move(os.path.join(td.name,name),cwd)
                        except Exception:
                            print('Cannot decompress file',name)
                            raise Exception



    def clean(self,listFile,report):
        for file in listFile:
            if report[file] == self._succeded:
                os.remove(file)
    ##
    #After retrieving the files this function prints the status of the download for each file,
    #which could be 'succeeded' or 'failed'

    def printDownloadReport(self):
        for k,v in self._downloadReport.items():
            print('Download of file',k,v,'.')
    ##
    # This function returns a dictionary whose keys are the attempted downloaded files and
    # the values are the status of the download, 'succeed' or 'failed'.
    # @return \c dictionary whose keys are the attempted downloaded files and the values are
    # the status of teh download, 'succeed' or 'failed'.

    def getDownloadReport(self):
        return self._downloadReport



    ##
    # Function that decompresses the file.
    # @param filename \c string  the name of the file to decompress.
    def decompress(self,filename,ddir):
        ex = self.getExtractor(filename)
        ex.extractall(ddir)

    ##
    #Inspecting the file determine the right extractor. If it cannot be determined then assume
    #no compression was used

    def getExtractor(self,filename):
        import tarfile
        import zipfile
        from . import gzipfile

        ret = None
        if(tarfile.is_tarfile(filename)):
            ret = tarfile.TarFile(filename)
        elif(zipfile.is_zipfile(filename)):
            ret = zipfile.ZipFile(filename)
        elif(gzipfile.is_gzipfile(filename)):
            ret = gzipfile.GZipFile(filename)
        else:
            print('Unrecognized archive type')
            raise Exception
        return ret

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
    ##
    # Setter function for the download directory.
    # @param ddir \c string directory where the data are downloaded.
    @property
    def downloadDir(self):
        return self._downloadDir
    @downloadDir.setter
    def downloadDir(self,ddir):
        self._downloadDir = ddir

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self,d):
        self.__dict__.update(d)
        self.logger = logging.getLogger('isce.iscesys.DataRetriever')
        return





    family = 'dataretriever'
    parameter_list = (
                      URL,
                      USERNAME,
                      PASSWORD,
                      DIRECTORY,
                      WAIT,
                      NUM_TRIALS,
                      PROCEED_IF_NO_SERVER
                      )
    def __init__(self,family = '', name = ''):

        #map of the names before and after decompression
        self._namesMapping = {}
        self._downloadReport = {}
        # Note if _useLocalDirectory is True then the donwloadDir is the local directory
        ##self._downloadDir = os.getcwd()#default to the cwd

        self._failed = 'failed'
        self._succeded = 'succeeded'
        super(DataRetriever, self).__init__(family if family else  self.__class__.family, name=name)
        # logger not defined until baseclass is called

        if not self.logger:
            self.logger = logging.getLogger('isce.iscesys.DataRetriever')
