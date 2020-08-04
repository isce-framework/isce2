#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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




from collections import OrderedDict
import xml.etree.ElementTree as ET

class Component(OrderedDict):
    '''
    Class for storing component information.
    '''
    def __init__(self, name=None,data=None):

        if name in [None, '']:
            raise Exception('Component must have a name')

        self.name = name

        if data is None:
            self.data = OrderedDict()
        elif isinstance(data, OrderedDict):
            self.data = data
        elif isinstance(data, dict):
            self.data = OrderedDict()
            for key, val in data.items():
                self.data[key] = val
        else:
            raise Exception('Component data in __init__ should be a dict or ordereddict')
            

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self,key,value):
        if not isinstance(key, str):
            raise Exception('Component key must be a string')

        self.data[key] = value

    def toXML(self):
        '''
        Creates an XML element from the component.
        '''
        root = ET.Element('component')
        root.attrib['name'] = self.name

        for key, val in self.data.items():
            if isinstance(val, Catalog):
                compSubEl = ET.SubElement(root, 'component')
                compSubEl.attrib['name'] = key
                ET.SubElement(compSubEl, 'catalog').text = str(val.xmlname)

            elif isinstance(val, Component):
                if key != val.name:
                    print('WARNING: dictionary name and Component name dont match')
                    print('Proceeding with Component name')
                root.append(val.toXML())

            elif (isinstance(val,dict) or isinstance(val, OrderedDict)):
                obj = Component(name=key, data=val)
                root.append(obj.toXML())

            elif (not isinstance(val, dict)) and (not isinstance(val, OrderedDict)):
                propSubEl = ET.SubElement(root,'property')
                propSubEl.attrib['name'] = key
                ET.SubElement(propSubEl, 'value').text = str(val)

        return root

    def writeXML(self, filename, root='dummy', noroot=False):
        '''
        Write the component information to an XML file.
        '''
        if root in [None, '']:
            raise Exception('Root name cannot be blank')

        if noroot:
            fileRoot = self.toXML()
        else:
            fileRoot = ET.Element(root)

            ####Convert component to XML
            root = self.toXML()
            fileRoot.append(root)

        print(fileRoot)

        indentXML(fileRoot)

        ####Write file
        etObj = ET.ElementTree(fileRoot)
        etObj.write(filename, encoding='unicode') 

class Catalog(object):
    '''
    Class for storing catalog key.
    '''
    def __init__(self, name):
        self.xmlname = name

def indentXML(elem, depth = None,last = None):
    if depth == None:
        depth = [0]
    if last == None:
        last = False
    tab =u' '*4
    if(len(elem)):
        depth[0] += 1
        elem.text = u'\n' + (depth[0])*tab
        lenEl = len(elem)
        lastCp = False
        for i in range(lenEl):
            if(i == lenEl - 1):
                lastCp = True
            indentXML(elem[i],depth,lastCp)
        if(not last):
            elem.tail = u'\n' + (depth[0])*tab
        else:
            depth[0] -= 1
            elem.tail = u'\n' + (depth[0])*tab
    else:
        if(not last):
            elem.tail = u'\n' + (depth[0])*tab
        else:
            depth[0] -= 1
            elem.tail = u'\n' + (depth[0])*tab

def test():
    '''
    Test method to demonstrate utility.
    '''

    insar = Component('insar')

    ####Reference info
    reference = {}
    reference['hdf5'] = 'reference.h5'
    reference['output'] = 'reference.raw'

    ####Secondary info
    secondary = {}
    secondary['hdf5'] = 'secondary.h5'
    secondary['output'] = 'secondary.raw'

    insar['reference'] = reference
    insar['secondary'] = secondary

    ####Set properties
    insar['doppler method'] = 'useDEFAULT'
    insar['sensor name'] = 'COSMO_SKYMED'
    insar['range looks'] = 3
    insar['dem'] = Catalog('dem.xml')

    insar.writeXML('test.xml', root='insarApp')
