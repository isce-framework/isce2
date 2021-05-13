#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2011 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Eric Gurrola
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




import os, errno, itertools
from .OrderedDict import OrderedDict
from io import StringIO

HEADER = "\n%s\n    %%s\n%s\n" % ("#"*100, '-'*100)
FOOTER = "#"*100
MAX_LIST_SIZE = 20

class Catalog(OrderedDict):
    # This bigArrayNum variable is used to ensure that big array files are
    # unique on disk. Some components that use the Catalog class are used
    # multiple times.
    bigArrayNum = 0

    def __init__(self, name, nodePath=None):
        OrderedDict.__init__(self)
        self.name = name
        if nodePath is None:
            self.fullName = name
        else:
            self.fullName = '.'.join(nodePath)
        return

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for other_k, other_v in other.items():
            try:
                self_v = self[other_k]
            except KeyError as e:
                return False
            if not (self_v == other_v):
                return False
        return True

    def addItem(self, key, value, node):
        """
        Adds given key/value pair to the specified node. If the node does not
        exist, it is created.
        """
        nodePath = node.split('.')
        self._addItem(key, value, nodePath)

    def hasNode(self, node):
        """
        Indicates whether a node exists in this catalog (such as "foo.bar.baz")
        """
        if not isinstance(node,str):
            raise TypeError("'node' must be a string")
        nodeList = node.split('.')
        return self._hasNodes(nodeList)

    def _hasNodes(self, nodeList):
        catalog = self
        for node in nodeList:
            if (node not in catalog) or (not isinstance(catalog[node], Catalog)):
                return False
            catalog = catalog[node]
        return True

    def addAllFromCatalog(self, otherCatalog):
        """Adds all the entries from the other catalog into this catalog."""
        if not isinstance(otherCatalog, Catalog):
            raise TypeError("'otherCatalog' must be of type Catalog")
        self._addAllFromCatalog(otherCatalog, [])

    def _addAllFromCatalog(self, otherCatalog, nodePath):
        for k, v in otherCatalog.items():
            if isinstance(v, Catalog):
                nodePath.append(v.name)
                self._addAllFromCatalog(v, nodePath)
                nodePath.pop()
            else:
                self._addItem(k, v, nodePath)

    def addInputsFrom(self, obj, node):
        """
        Given an object, attempts to import its dictionaryOfVariables attribute
        into this catalog under the given node.
        """
        if not hasattr(obj, 'dictionaryOfVariables'):
            raise AttributeError(
                "The object of type {} ".format(obj.__class__.__name__)
                 +  "does not have a dictionaryOfVariables attribute!")
        nodePath = node.split('.')
        for k, v in obj.dictionaryOfVariables.items():
            #check for new and old style dictionaryOfVariables
            try:
                attr = v['attrname']
                #only dump input or inoutput
                if(v['type'] == 'component' or v['intent'] == 'output'):
                    continue
            except Exception:
                attr = v[0].replace('self.', '', 1)
            self._addItem(k, getattr(obj, attr), nodePath)
        if 'constants' in iter(list(obj.__dict__.keys())):
            for k, v in list(obj.constants.items()):
                self._addItem(k, v, nodePath)

    def addOutputsFrom(self, obj, node):
        """
        Given an object, attempts to import its dictionaryOfOutputVariables
        attribute into this catalog under the given node.
        """
        if not hasattr(obj, 'dictionaryOfOutputVariables'):
            #it's probably the new type of dictionary
            for k, v in obj.dictionaryOfVariables.items():
                nodePath = node.split('.')
                #check for new and old style dictionaryOfVariables
                try:
                    attr = v['attrname']
                    #only dump output or inoutput
                    if(v['intent'] == 'input'):
                        continue
                except Exception:
                    continue
                self._addItem(k, getattr(obj, attr), nodePath)
        else: 
            #old style/. To be removed once everything is turned into a Configurable           
            nodePath = node.split('.')
            for k, v in obj.dictionaryOfOutputVariables.items():
                attr = v.replace('self.', '', 1)
                self._addItem(k, getattr(obj, attr), nodePath)

    def _addItem(self, key, value, nodePath):
        catalog = self
        partialPath = []
        for node in nodePath:
            partialPath.append(node)
            # Instantiate a new catalog if the node does not already exist
            if node not in catalog:
                catalog[node] = Catalog(node, partialPath)
            catalog = catalog[node]
        # Just record the file info if this value is actually a large array
        catalog[key] = self._dumpValueIfBigArray(key, value, nodePath)

    def _dumpValueIfBigArray(self, key, v, nodePath):
        """Checks to see if the value is a list greater than the defined length threshhold. If so,
        dump the array to a file and return a string value indictating the file name. Otherwise,
        return the normal value."""
        if self._isLargeList(v):
            # Make the catalog directory if it doesn't already exist
            os.makedirs('catalog', exist_ok=True)
            fileName = 'catalog/%s.%s.%03i' % ('.'.join(nodePath), key, Catalog.bigArrayNum)
            Catalog.bigArrayNum += 1
            f = open(fileName, 'w')
            self.writeArray(f, v)
            f.close()
            v = fileName
        return v

    def writeArray(self, file, array):
        """Attempts to output arrays in a tabular format as neatly as possible. It tries
        to determine whether or not it needs to transpose an array based on if an array is
        multidimensional and if each sub-array is longer than the main array."""
        # The arrya is guaranteed to be > 0 by the caller of this method
        multiDim = isinstance(array[0], list) or isinstance(array[0], tuple)
        # 'transpose' the array if each element array is longer than the main array
        # this isn't fool proof and might produce incorrect results for short multi-dim
        # arrays, but it work in practice
        if multiDim and len(array[0]) > len(array):
            array = zip(*array)
        for e in array:
            if multiDim:
                e = '\t'.join(str(x) for x in e)
            else:
                e = str(e)
            file.write("%s\n" % e)


    def _isLargeList(self, l):
        """This handles the fact that a list might contain lists. It returns True if the list
        itself or any of its sublists are longer than MAX_LIST_SIZE. If 'l' is not a list,
        False is returned. This method does assume that all sublists will be the same size."""
        while (isinstance(l, list) or isinstance(l, tuple)) and len(l) > 0:
            if len(l) > MAX_LIST_SIZE:
                return True
            l = l[0]
        return False


    def printToLog(self, logger, title):
        """Prints this catalog to the given logger, one entry per line.
        Example output line: foo.bar = 1"""
        file = StringIO()
        file.write(HEADER % title)
        self._printToLog(file, self)
        file.write(FOOTER)
        logger.info(file.getvalue())

    def _printToLog(self, file, catalog):
        for k in sorted(catalog.keys()):
            v = catalog[k]
            if isinstance(v, Catalog):
                self._printToLog(file, v)
            else:
                file.write("%s.%s = %s\n" % (catalog.fullName, k, str(v)))

    def renderXml(self, file=None, nodeTag=None, elementTag=None):
        if not file:
            file = self.fullName+'.xml'

        adict = {self.fullName:self}

#        from isceobj.XmlUtil import xmlUtils as xmlu
        dict_to_xml(adict,file,nodeTag=nodeTag,elementTag=elementTag)




import xml.etree.ElementTree as ET
from collections import UserDict

def dict_to_xml(adict,file,nodeTag=None,elementTag=None):
    a = ET.Element(nodeTag)  # something to hang nodes on
    a = dict_to_et(a,adict,nodeTag,elementTag)
    et = list(a)[0]
    indent(et)
    tree = ET.ElementTree(et)
    tree.write(file)

def space_repl(key):
    return key.replace(' ','_')

def slash_repl(key):
    return key.replace('/','_dirslash_')

def key_clean(key):
    return slash_repl(space_repl(key))

def dict_to_et(node,adict,nodeTag,elementTag):
    for key, val in adict.items():
        if isinstance(val,UserDict) or isinstance(val,dict):
            if nodeTag:
               subnode = ET.Element(nodeTag)
               node.append(subnode)
               name = ET.Element('name')
               subnode.append(name)
               name.text = key_clean(str(key))
            else:
               subnode = ET.Element(key_clean(str(key)))
               node.append(subnode)
            subnode = dict_to_et(subnode,val,nodeTag,elementTag)
        else:
            if elementTag:
               subnode = ET.Element(elementTag)
               node.append(subnode)
               name = ET.Element('name')
               subnode.append(name)
               name.text = key_clean(str(key))
               value = ET.Element('value')
               subnode.append(value)
               value.text = str(val).replace('\n', '\\n')
            else:
               lmnt = ET.Element(key_clean(str(key)))
               node.append(lmnt)
               lmnt.text = str(val).replace('\n', '\\n')
    return node

def indent(elem, depth = None,last = None):
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
            indent(elem[i],depth,lastCp)

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
