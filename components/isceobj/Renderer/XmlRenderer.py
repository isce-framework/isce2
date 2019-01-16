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
# Author: Walter Szeliga
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




import types
import datetime
from xml.etree import ElementTree
from iscesys.Component.Component import Component
from iscesys.Util.decorators import type_check

##
# This class provides a basis for image and metadata renderers (How?)
#
class BaseRenderer(Component):

    ## This is overridden, so why does it exisit?
    dictionaryOfVariables = {'COMPONENT': ['component',
                                           Component,
                                           True]}

    @type_check(Component)
    def setComponent(self, component):
        self._component = component

    def getComponent(self):
        return self._component

    component = property(getComponent, setComponent)
    pass

##
# A class to render metadata in xml format
#
#class XmlRenderer(Component,BaseRenderer):
class XmlRenderer(BaseRenderer):

    dictionaryOfVariables = {'OUTPUT': ['output','str','mandatory']}

    def __init__(self):
        super(XmlRenderer, self).__init__()

        self.component = None
        self.output = None
        self.variables = {}
        self.documentation = {}
        self.descriptionOfVariables = {}
        return None


    def setComponent(self, component):
        self.component = component
        self.variables = component.dictionaryOfVariables
        self.documentation = component.descriptionOfVariables

    def setOutput(self,output):
        self.output = output

    def getOutput(self):
        return self.output

    def render(self):
        root = ElementTree.Element('component')
        self.subrender(root)
        from isceobj.XmlUtil.xmlUtils import indent
        indent(root)
        tree = ElementTree.ElementTree(root)
        tree.write(self.output)
        pass

    # change note; how can the attribute.replace work with 'self' removed from
    # the dict?
    # Why the exec statements?  There are assig nments (eval works). Why use eval.
    # there are on variables in them? the needs refactoring
    def subrender(self,root):
        value = 'value'
        selfPos = 0
        typePos = 1
        nameSubEl = ElementTree.SubElement(root,'name')
        nameSubEl.text = self.component.__class__.__name__
        for key, val in self.variables.items():
            propSubEl = ElementTree.SubElement(root,'property')
            isUndef = False
            ElementTree.SubElement(propSubEl, 'name').text = key
            attribute = val[selfPos]
            attribute = attribute.replace('self.','self.component.')
            exec('type = ' + attribute + '.__class__')
#            type = attribute.__class__
            if (type in [types.NoneType,types.IntType,types.LongType,types.StringType,types.FloatType,datetime.datetime]):
                exec('ElementTree.SubElement(propSubEl, \'value\').text = str(' + attribute + ')')
            elif (type == types.ListType):
                exec('valList = ' + attribute)
                # Test by printing to the screen
                #for val in valList:
                #    print val
            else:
                exec('component = ' + attribute)
                subrenderer = XmlRenderer()
                subrenderer.setComponent(component)
                subrenderer.subrender(root)
            if key in self.documentation:
                for keyDict,valDict in self.documentation[key].items():
                    ElementTree.SubElement(propSubEl,keyDict).text = str(valDict)




