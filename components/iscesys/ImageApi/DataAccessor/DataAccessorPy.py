#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
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
from iscesys.ImageApi import DataAccessor as DA
import os
## If you finalize more than once, do you get an error?
ERROR_CHECK_FINALIZE = False

class DataAccessor(object):
    _accessorType = ''

    @staticmethod
    def getTypeSizeS(type_):
        return DA.getTypeSize(type_)
    def __init__(self):
        self._accessor = None
        self._factory = None
        self.scheme = ''
        self.caster = ''
        self.width = None
        self.bands = None
        self.length = None
        self.accessMode = ''
        self.filename = ''
        self.dataType = ''
        self._size = None
        #instead of creating a new function for each type of Accessor to be created
        #in the c bindings, pass a dictionary which contains the key 'type' to know the accessor that
        #needs to be instanciated
        self._extraInfo = {}
        self._extra_reader = 'vrt'
        return None

    ## Experimental
    def __int__(self):
        return self.getAccessor()

    def initAccessor(self, filename, filemode, width,
                     type=None, bands=None, scheme=None, caster=None):
        self.filename = filename
        self.accessMode = filemode
        self.width = int(width)
        if type:
            self.dataType = type
        if bands:
            self.bands = int(bands)
        if scheme:
            self.scheme = scheme
        if caster:
            self.caster = caster
        return None
    def getGDALDataTypeId(self,type_):
        #from GDALDataType enum
        map = {'byte':1,'ciqbyte':1,'short':3,'int':4,'float':6,'double':7,
               'cshort':8,'cint':9,'cfloat':10,'cdouble':11}
        try:
            return map[type_.lower()]
        except:
            print('Unsupported  datatype',type_)
            raise Exception

    def checkLocation(self):
        from iscesys.Parsers.FileParserFactory import createFileParser
        parser = createFileParser('xml')
        #get the properties from the file
        prop, fac, misc = parser.parse(self.metadatalocation)
        #first check if it exists as it is
        filename = ''

        if not (os.path.exists(prop['file_name'])):
            name = os.path.basename(prop['file_name'])
            #check the path relative to the xml file
            filename = os.path.join(os.path.split(self.metadatalocation)[0],name)
            #check if relative to cwd
            if not (os.path.exists(filename)):
                filename = os.path.join(os.getcwd(),name)
                if not (os.path.exists(filename)):
                    filename = ''
        else:
            filename = prop['file_name']
        if not filename:
            paths = self.uniquePath([os.path.split(prop['file_name'])[0],os.path.split(self.metadatalocation)[0],
                  os.getcwd()])
            toptr = '\n'.join(paths)
            print('The image file',name,'specified in the metadata file',self.metadatalocation,
                  'cannot be found in', 'any of the following default locations:' if len(paths) > 1 else 'in the following location:' ,
                  toptr)
            raise Exception

        return filename
    def uniquePath(self,paths):
        ret = []
        for pth in paths:
            if not pth in ret:
                ret.append(pth)
        return ret

    def methodSelector(self):
        selection = ''
        if self._accessorType.lower() == 'api':
            selection = 'api'
        elif self._accessorType.lower() == self._extra_reader:
            selection = self._extra_reader
        elif self.accessMode.lower() == 'write':
            selection='api'
        elif self.accessMode.lower() == 'read':
            selection = self._extra_reader

        return selection

    def createAccessor(self):
        if(not self.filename and hasattr(self,'metadatalocation') and self.metadatalocation and not self.accessMode.lower().count('write')):
            #it will only keep going if all ok
            self.filename = self.checkLocation()
        caster = '' or self.caster
        filename = self.filename
        scheme = self.scheme
        #if the filename is a URL, the extraFilename should indicate the file from the local machine
        #instead of from the remote server.
        if self.filename.startswith('http'):
            self.extraFilename = os.path.basename(self.filename) + '.' + self._extra_reader
        else:
            self.extraFilename = self.filename + '.' + self._extra_reader

        if self._accessor is None:#to avoid creating duplicates
            selection = self.methodSelector()
            if selection == 'api':
                size = DA.getTypeSize(self.dataType)
                #to optimize bip access per band we read in memory all bands and then
                #set the right band and write the content back leaving the other bands untouched
                #this requires a read and write which only works if the file is opened in
                #writeread (or readwrite) mode and not just write
                if(self.accessMode.lower() == 'write'):
                #if(self.scheme.lower() == 'bip' and self.accessMode.lower() == 'write'):
                    self.accessMode = 'writeread'
            elif selection == self._extra_reader:
                size = self.getGDALDataTypeId(self.dataType)
                filename = self._extraFilename
                #GDALAccessor handles all the different scheme in the same way since it reads
                #always in BSQ scheme regardless of the under laying scheme
                scheme = 'GDAL'
            else:
                print('Cannot select appropruiate image API')
                raise Exception
            self._accessor, self._factory = DA.createAccessor(
                filename, self.accessMode, size, self.bands,
                self.width,scheme,caster,self._extraInfo
                )
        return None

    def finalizeAccessor(self):
        try:
            DA.finalizeAccessor(self._accessor, self._factory)
        except TypeError:
            message = "Image %s is already finalized" % str(self)
            if ERROR_CHECK_FINALIZE:
                raise RuntimeError(message)
            else:
                print(message)

        self._accessor = None
        self._factory = None
        return None

    def getTypeSize(self):
        return DA.getTypeSize(self.dataType)
    def rewind(self):
        DA.rewind(self._accessor)

    def createFile(self, lines):
        DA.createFile(self._accessor, lines)

    def getFileLength(self):
        openedHere = False

        if self._accessor is None:
            openedHere = True
            self.initAccessor(self.filename, 'read', int(self.width),
                              self.dataType, int(self.bands), self.scheme)
            self.createAccessor()
        length = DA.getFileLength(self._accessor)

        if openedHere:
            self.finalizeAccessor()

        return length

    def getAccessor(self):
        return self._accessor

    def getFilename(self):
        return self.filename

    def getAccessMode(self):
        return self.accessMode

    def getSize(self):
        return self.size

    def getBands(self):
        return self.bands

    ## Get the width associated to the DataAccessor.DataAccessor object created.
    #@return \c int width of the DataAccessor.DataAccessor object.
    def getWidth(self):
        return self.width

    def getInterleavedScheme(self):
        return self.scheme

    def getCaster(self):
        return self.caster

    def getDataType(self):
        return self.dataType

    def setFilename(self, val):
        self.filename = str(val)

    def setAccessMode(self, val):
        self.accessMode = str(val)

    def setBands(self, val):
        self.bands = int(val)

    def setWidth(self, val):
        self.width = int(val)

    def setInterleavedScheme(self, val):
        self.scheme = str(val)

    def setCaster(self, val):
        self.caster = val

    def setDataType(self, val):
        self.dataType = val

    def setExtraInfo(self,ei):
        self._extraInfo = ei
    pass
