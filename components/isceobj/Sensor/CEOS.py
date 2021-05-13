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




import os
import struct
from xml.etree.ElementTree import ElementTree as ET

class CEOSDB(object):

    typeMap = ['Skip', 'An', 'In', 'B1', 'B4', 'Fn', 'B2', 'Debug', 'BF4', 'B8']

    def __init__(self, xml=None, dataFile=None):
        self.xml = xml
        self.dataFile = dataFile
        self.startPosition = dataFile.tell()
        self.recordLength = 0
        self.metadata = {}
        if not  xml == None:
            self.xmlFP = open(self.xml, 'r')
            self.rootChildren = list(ET(file=self.xmlFP).getroot())
        else:
            self.xmlFP = None
            self.rootChildren = []
    def getMetadata(self):
        return self.metadata

    def getEndOfRecordPosition(self):
        return self.startPosition + self.recordLength


    def finalizeParser(self):
        self.xmlFP.close()

    def parseFast(self):
        """
            Use the xml definition of the field positions, names and lengths to
            parse a CEOS data file
        """

        for z in self.rootChildren:
            # If the tag name is 'rec', this is a plain old record
            if z.tag == 'rec':
                (key,data) = self.decodeNode(z)
                self.metadata[key] = data
            # If the tag name is 'struct', we need to loop over some other
            # records
            elif z.tag == "struct":
                try:
                    loopCounterName = z.attrib['loop']
                    loopCount = self.metadata[loopCounterName]
                except KeyError:
                    loopCount = int(z.attrib['nloop'])
                key = z.attrib['name']
                self.metadata[key] = [None]*loopCount
                for i in range(loopCount):
                    struct = {}
                    for node in z:
                        (subkey,data) = self.decodeNode(node)
                        struct[subkey] = data
                    self.metadata[key][i] = struct



        self.recordLength = self.metadata['Record Length']

    def parse(self):
        """
            Use the xml definition of the field positions, names and lengths to
            parse a CEOS data file
        """
        xmlFP = open(self.xml, 'r')

        self.root = ET(file=xmlFP).getroot()
        for z in self.root:
            # If the tag name is 'rec', this is a plain old record
            if z.tag == 'rec':
                (key,data) = self.decodeNode(z)
                self.metadata[key] = data
            # If the tag name is 'struct', we need to loop over some other
            #records
            if z.tag == "struct":
                try:
                    loopCounterName = z.attrib['loop']
                    loopCount = self.metadata[loopCounterName]
                except KeyError:
                    loopCount = int(z.attrib['nloop'])

                key = z.attrib['name']
                self.metadata[key] = [None]*loopCount
                for i in range(loopCount):
                    struct = {}
                    for node in z:
                        (subkey,data) = self.decodeNode(node)
                        struct[subkey] = data
                    self.metadata[key][i] = struct

        xmlFP.close()
        self.recordLength = self.metadata['Record Length']

    def decodeNode(self,node):
        """
            Create an entry in the metadata dictionary
        """
        key = node.attrib['name']
        size = int(node.attrib['num'])
        format = int(node.attrib['type'])
        data = self.readData(key, size, format)
        return key, data

    def readData(self, key, size, format):
        """
            Read data from a node and return it
        """
        formatString = ''
        strp_3 = lambda x: str.strip(x.decode('utf-8')).rstrip('\x00')
        convertFunction = None
        if (self.typeMap[format] == "Skip"):
            self.dataFile.seek(size, os.SEEK_CUR)
            return
        elif (self.typeMap[format] == "An"):
            formatString = "%ss" % size
            convertFunction = strp_3
        elif (self.typeMap[format] == "In"):
            formatString = "%ss" % size
            convertFunction = int
        elif (self.typeMap[format] == "Fn"):
            formatString = "%ss" % size
            convertFunction = float
        elif (self.typeMap[format] == "Debug"):
            print (key, size, format, self.dataFile.tell())
        elif (self.typeMap[format] == "B4"):
            formatString = ">I"
            convertFunction = int
            size = 4
        elif (self.typeMap[format] == "BF4"):
            formatString = ">f"
            convertFunction = None
            size = 4
        elif (self.typeMap[format] == "B2"):
            formatString = ">H"
            convertFunction = int
            size = 2
        elif (self.typeMap[format] == "B1"):
            formatString = ">B"
            convertFunction = int
            size = 1
        elif (self.typeMap[format] == "B8"):
            formatString = ">Q"
            convertFunction = None
            size = 8
        else:
            raise TypeError("Unknown format %s" % format)

        data = self._readAndUnpackData(length=size, format=formatString,
            typefunc=convertFunction)
        return data


    def _readAndUnpackData(self, length=None, format=None, typefunc=None,
        numberOfFields=1):
        """
        Convenience method for reading and unpacking data.

        length is the length of the field in bytes [required]
        format is the format code to use in struct.unpack() [required]
        numberOfFields is the number of fields expected from the call to
            struct.unpack() [default = 1]
        typefunc is the function through which the output of struct.unpack will
            be passed [default = None]
        """
        line = self.dataFile.read(length)
        try:
            data = struct.unpack(format, line)
        except struct.error as strerr:
            print(strerr)
            return
        if (numberOfFields == 1):
            data = data[0]
            if (typefunc == float):
                data = data.decode('utf-8').replace('D','E')
            if(typefunc):
                try:
                    data = typefunc(data)
                except ValueError:
                    data = 0

        return data
