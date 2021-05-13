#!/usr/bin/env python3

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
import sys
import xml.etree.ElementTree as ET
##
#This class offers a set of methods that allow the reading and writing of xml files using the ElementTree python module.
#It can be used to create xml file suitable to initialize object using the initializer  Component::InitFromXmlFile . In the  ISCE  package each object will be defined with the following elements
#\verbatim
#<component>
#        <name>NameOfTheObject<\name>
#        <property>
#                <name>VARIABLE1<\name>
#                <value>value1<\value>
#                <doc>"documentation VARIABLE1"<\doc>
#        <\property>
#        <property>
#                <name>VARIABLE2<\name>
#                <value>value2<\value>
#                <doc>"documentation VARIABLE2"<\doc>
#                <units>m/s</units>
#                <otherattributes>SomeOtherAttribute<\otherattributes>
#        <\property>
#</component>
#\endverbatim
#Each paramenter of the object named "NameOfTheObject" will be defined by a "property element". Inside the property element there are other elements
#that characterize the specific paramenter. The element "name" defines the name of the variable and is the same as the key in
#the dictionaryOfVariables of the object at hand (see Component::Component). The "value" is the value that the specific variable will be initialized to.
#All the other elements will be part of the descriptionOfVariables dictionary (see Component::Component).
#@see http://effbot.org/zone/element-index.htm
#@see Component::Component
class XmlUtil:

##
# Reads an xml file and turns it into an ElementTree object.
#@param file either a file name or a file object
#@return an ElementTree object
    def readFile(self,file):

        tree = ET.parse(file)
        return tree
##
# Writes a dictionary into an indented xml file
# @param file \c string  filename to be used.
# @param dict \c dictionary to be saved in xml format
# @param name \c string the name to be set in the 'name' field
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
##
# Writes an ElementTree object or a root element of a ElementTree object into an indented xml file
    def writeFile(self,file,object):
        root = None
        try:
            root = object.getroot()
        except Exception:
            root = object

        self.indent(root)
        etObj = ET.ElementTree(root)
        etObj.write(file)
    #if the string contained in obj is an actual object, when is exec there is no problem. if it was supposed to be a string, the name will not be defined aand an exception is thrown. put in a function to reduce chance that the name is actually defined
    def isStr(self,obj):
        retVal = False
        try:
            exec (obj)
            return False
        except:
            return True

##
#Given an ElementTree object it creates a dictionary of dictionaries where each entry corresponds to a "property" element. For instance in the example
#\verbatim
#<component>
#        <name>NameOfTheObject<\name>
#        <property>
#                <name>VARIABLE1<\name>
#                <value>value1<\value>
#                <units>m/s</units>
#                <doc>"documentation VARIABLE1"<\doc>
#        <\property>
#        <property>
#                <name>VARIABLE2<\name>
#                <value>value2<\value>
#                <doc>"documentation VARIABLE2"<\doc>
#                <otherattributes>SomeOtherAttribute<\otherattributes>
#        <\property>
#</component>
#\endverbatim
#the returned dictionary is
#\verbatim
# rectDict = {VAIRABLE1:{'value':value1,'doc':"documentation VARIABLE1",'units':'m/s'},VARIABLE2:{'value':value2,'doc':"documentation VARIABLE2"},'otherattributes':"SomeOtherAttibutes:}
#\endverbatim
#@param tree ElementTree object
#@return dictionary of dictionaries
    def createDictionary(self,tree):

        retDict = {}
        property = ''
        if(self.property):
            property = self.property
        else:
            property = 'property'
        root = tree.getroot()
        listEl = root.findall(property)
        for var in listEl:
            keyWord = ''
            if (var.find('name') == None):#need at least to have the name of the variable
                print('Error. Expecting the tag \'name\' to be present for each', property, 'element')
                raise Exception
            else:
                keyWord = var.find('name').text

            listChildren = list(var)
            tmpDict = {}
            for description in listChildren:
                if(description.tag == 'name'):
                    continue

                    #since description.text is a string by doing exec it will put the actual value into the dictionary.
                if (not self.isStr(description.text)):# if value is a string when execed it will see it as a name that has not been defined -> exception so try and catch
                    exec('tmpDict[description.tag] = '  + description.text)
                else:
                    tmpDict[description.tag] =  description.text

            retDict[keyWord] = tmpDict

        return retDict
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

##
# Construnctor. It the optional string keyword is provided,  the createDictionary() function will use the string keyword instead of the
# string "property" to create the dictionary.
#@param keyword string to be used in createDictionary()
#@see createDictionary()

    def __init__(self,keyword = None):

        ##
        #String used to create the dictionar of dictionaries in createDictionary()
        #@see __init__()
        #@see createDictionary()
        self.property = keyword

