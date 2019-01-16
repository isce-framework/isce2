#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2009 California Institute of Technology. ALL RIGHTS RESERVED.
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
import logging
import os
const_key = '__const__'
const_name = 'constant'
const_marker = '\$'  #\ required to escape special character for re

class Parser(object):
    """Parser

    A class containing commandLineParser, componentParser, and propertyParser
    methods.
    """


    def command_line_parser(self, *args):
        return self.commandLineParser(args)
    ## get it?
    def commandLineParser(self, args):
        from iscesys.DictUtils.DictUtils import DictUtils as DU
        """commandLineParser

        Parses a command line, which may include files and command line options
        and returns dictionaries containing propDict, factDict, miscDict, and
        listOfOptions where

        propDict contains the input values for the properties of an ISCE
        application as well as those for the components declared as facilities
        in the application

        factDict contains input values for the factories used in constructing
        instances of the components declared as facilities in the application.

        miscDict contains the above two types of information that are entered
        in-line on the command line.  These will override those given in the
        files during component initialization if there are conflicts.

        listOfOptions contains the '--' style options such as '--help'.
        """

        propDict = {}
        factDict = {}
        miscDict = {}
        listOfOptions = []
        for arg in args:
            if arg.startswith('--'):
                listOfOptions.append(arg)
                continue

            isFile = False
            for filetype in self._filetypes:
                if arg.endswith('.' + filetype):
                    ## imports
                    from iscesys.DictUtils.DictUtils import DictUtils as DU
                    from iscesys.Parsers.FileParserFactory import createFileParser
                    FP = createFileParser(filetype)
                    tmpProp, tmpFact, tmpMisc = FP.parse(arg)

                    if tmpProp:
                        DU.updateDictionary(propDict, tmpProp, replace=True)
                    if tmpFact:
                        DU.updateDictionary(factDict, tmpFact, replace=True)
                    if tmpMisc:
                        DU.updateDictionary(miscDict,tmpMisc,replace=True)

                    isFile = True
                    break

            if isFile:
                continue

            #if it gets here the argument is not a file
            #assume a form like,
            #component1.component2 .... .componentN.attribute=value .
            #no space otherwise the split above will not work properly
            #probably it is better if we specify from the top component so it
            #is easier to handle the case in which the files come after
            #(otherwise the key of the first node is not defined)


            tmpProp, tmpFact, tmpMisc = self.dotStringToDicts(arg)

            if tmpProp:
                DU.updateDictionary(propDict, tmpProp, replace=True)
            if tmpFact:
                DU.updateDictionary(factDict, tmpFact, replace=True)
            if tmpMisc:
                DU.updateDictionary(miscDict,tmpMisc,replace=True)

        return (DU.renormalizeKeys(propDict),DU.renormalizeKeys(factDict),DU.renormalizeKeys(miscDict),listOfOptions)

    def dotStringToDicts(self, arg):
        tmpProp = {}
        tmpFact = {}
        tmpMisc = {}
        if not (arg == '-h' or  arg == '--help'):

            compAndVal = arg.split('=')
            if len(compAndVal) != 2:
                logging.error('Error. The argument',
                              arg,
                              'is neither an input file nor a sequence object.param=val')
                raise TypeError('Error. The argument %s is neither an input file nor a sequence object.param=val' % str(arg))

            if self.isStr(compAndVal[1]):
                val = compAndVal[1]
            else:
                val = eval(compAndVal[1])

            listOfComp = compAndVal[0].split('.')

            d = {}
            self.nodeListValToDict(listOfComp, val, d)
            innerNode = listOfComp[-1]


            if innerNode in ('doc', 'units'):
                tmpMisc = d
            elif innerNode in ('factorymodule', 'factoryname'):
                tmpFact = d
            else:
                tmpProp = d

        return tmpProp, tmpFact, tmpMisc

    def nodeListValToDict(self, l, v, d):
        if len(l) > 1:
            k = self.normalize_comp_name(l[0])
            d.update({k:{}})
            self.nodeListValToDict(l[1:], v, d[k])
        else:
            d.update({self.normalize_prop_name(l[0]):v})


    #root is the node we are parsing.
    #dictIn is the dict where the value of that node is set.
    #dictFact is the dict where the informations relative to the factory for that node are set.
    #dictMisc is a miscellaneus dictionary where we put other info about the property such as doc,units etc

    def parseComponent(self,root,dictIn,dictFact,dictMisc = None,metafile=None):
        # Check for constants
        self.parseConstants(root, dictIn, dictMisc)
        self.apply_consts_dict(dictIn[const_key], dictIn[const_key])
        # check if it has some property to set. it will overwrite the ones possibly present in the catalog
        self.parseProperty(root,dictIn,dictMisc)

        nodes = root.findall('component')

        for node in nodes:
            #Normalize the input node name per our convention
            name = self.getNormalizedComponentName(node)
            factoryname = self.getComponentElement(node, 'factoryname')
            factorymodule = self.getComponentElement(node, 'factorymodule')
            args = node.find('args')
            kwargs = node.find('kwargs')
            doc = node.find('doc')
            #check if any of the facility attributes are defined
            # don't ask me why but checking just "if factoryname or factorymodule .. " did not work

            if (not factoryname == None) or (not factorymodule == None) or (not args == None) or (not kwargs == None) or (not doc == None):
                if not name in dictFact:
                    dictFact.update({name:{}})
            if not factoryname == None:
                dictFact[name].update({'factoryname': factoryname})
            if not factorymodule == None:
                dictFact[name].update({'factorymodule': factorymodule})
            if not  args == None:
                #this must be a tuple
                argsFact = eval(args.text)
                dictFact[name].update({'args':argsFact})
            if not  kwargs == None:
                #this must be a dictionary
                kwargsFact = eval(kwargs.text)
                dictFact[name].update({'kwargs':kwargsFact})
            if not  doc is None:
                #the doc should be a list of strings. if not create a list
                if self.isStr(doc.text):
                    dictFact[name].update({'doc':[doc.text]})
                else:#if not a string it should be a list
                    exec("dictFact[name].update({'doc': " + doc.text + "})")

            catalog = node.find('catalog')
            if not catalog == None:
                parser = node.find('parserfactory')

                # if a parser is present than call the factory otherwise use default.
                #it should return a dictionary (of dictionaries possibly) with name,value.
                #complex objects are themselves rendered into dictionaries
                tmpDictIn = {}
                tmpDictFact = {}
                tmpDictMisc = {}

                #the catalog can be a string i.e. a filename (that will be parsed) or a dictionary
                catalog_text = catalog.text.strip()
                if self.isStr(catalog_text):
                    #Create a file parser in XP
                    if parser:
                        #If the inputs specified a parser, then use it
                        filetype = node.find('filetype').text
                        XP = eval(parser.text + '(\"' + filetype + '\")')

                    else:
                        #If the inputs did not specify a parser, then create one from an input extension type
                        #or, if not given as input, from the extension of the catalog
                        filetype = node.find('filetype')
                        if filetype:
                            ext = filetype.text
                        else:
                            ext = catalog_text.split('.')[-1]

                        from .FileParserFactory import createFileParser
                        XP = createFileParser(ext)
                    self._metafile = catalog_text
                    (tmpDictIn,tmpDictFact,tmpDictMisc) =  XP.parse(catalog_text)

                    #the previous parsing will return dict of dicts with all the subnodes of that entry, so update the  node.
                    if not tmpDictIn == {}:
                        if not name in dictIn:
                            dictIn.update({name:tmpDictIn})
                        else:
                            dictIn[name].update(tmpDictIn)
                    if not tmpDictFact == {}:
                        if not name in dictFact:
                            dictFact.update({name:tmpDictFact})
                        else:
                            dictFact[name].update(tmpDictFact)
                    if not tmpDictMisc == {}:
                        if not name in dictMisc:
                            dictMisc.update({name:tmpDictMisc})
                        else:
                            dictMisc[name].update(tmpDictMisc)

                else:
                    #the catalog is a dictionary of type {'x1':val1,'x2':val2}
                    tmpDictIn = eval(catalog_text)
                    if isinstance(tmpDictIn,dict):
                        if not tmpDictIn == {}:
                            if not name in dictIn:
                                dictIn.update({name:tmpDictIn})
                            else:
                                dictIn[name].update(tmpDictIn)

                    else:
                        logging.error("Error. catalog must be a filename or  a dictionary")
                        raise

            tmpDict = {}
            tmpDict[const_key] = dictIn[const_key] #pass the constants down
            tmpDictFact= {}
            tmpDictMisc= {}

            #add the attribute metalocation to the object paramenter
            tmpDict['metadata_location'] = os.path.abspath(self._metafile)
            self.parseComponent(node,tmpDict,tmpDictFact,tmpDictMisc)
            if not tmpDict == {}:
                if not name in dictIn:
                    dictIn.update({name:tmpDict})
                else:
                    dictIn[name].update(tmpDict)
            if not tmpDictFact == {}:
                if not name in dictFact:
                    dictFact.update({name:tmpDictFact})
                else:
                    dictFact[name].update(tmpDictFact)
            if not tmpDictMisc == {}:
                if not name in dictMisc:
                    dictMisc.update({name:tmpDictMisc})
                else:
                    dictMisc[name].update(tmpDictMisc)


    def getNormalizedComponentName(self, node):
        """
        getNormalizedComponentName(self, node)
        return the normalized component name.
        """
        name = self.normalize_comp_name(self.getPropertyName(node))
        return name

    def getComponentElement(self, node, elementName):
        """
        getComponentElement(self, node, elementName)
        Given an input node and the node elementName return
        the value of that elementName of the property.
        Look for the 'property' element either as a sub-tag or
        as an attribute of the property tag.  Raise an exception
        if both are used.
        """
        return self.getPropertyElement(node, elementName)


    def parseConstants(self, root, dictIn, dictMisc=None):
        """
        Parse constants.
        """

        if not const_key in dictIn.keys():
            dictIn[const_key] = {}

        nodes = root.findall(const_name)
        for node in nodes:
            #get the name of the constant
            name = self.getPropertyName(node)
            #get the value of the constant
            value = self.getPropertyValue(node)
            #get the other possible constant elements
            units = self.getPropertyElement(node, 'units')
            doc = self.getPropertyElement(node, 'doc')

            dictIn[const_key].update({name:value})

            if (not units == None) and (not dictMisc == None):
                if not const_key in dictMisc.keys():
                    dictMisc[const_key] = {}
                if not name in dictMisc[const_key]:#create the node
                    dictMisc[const_key].update({name:{'units':units}})
                else:
                    dictMisc[const_key][name].update({'units':units})
            if (not doc == None) and (not dictMisc[const_key] == None):
                if not name in dictMisc[const_key]:#create the node
                    dictMisc[const_key].update({name:{'doc':doc}})
                else:
                    dictMisc[const_key][name].update({'doc':doc})

        return

    def apply_consts_dict(self, dconst, d):
        for k, v in d.items():
            d[k] = self.apply_consts(dconst, v)

    def apply_consts(self, dconst, s):
        """
        Apply value of constants defined in dconst to the string s
        """
        import re
        for k, v in dconst.items():
            var = const_marker+k+const_marker
            s = re.sub(var, v, s)
        return s

    def parseProperty(self,root,dictIn,dictMisc = None):
        nodes = root.findall('property')
        for node in nodes:
            #Normalize the input property names per our convention
            name = self.getNormalizedPropertyName(node)
            #get the property value
            value = self.getPropertyValue(node)
            #substitute constants
            value = self.apply_consts(dictIn[const_key], value)
            #get the other possible property elements
            units = self.getPropertyElement(node, 'units')
            doc = self.getPropertyElement(node, 'doc')
            value = self.checkException(name,value)
            #Try to update the input dictionary
            if self.isStr(value): # it is actually a string
                dictIn.update({name:value})
            else: # either simple ojbect, including list, or a dictionary
                try:
                    dictIn.update({name:eval(value)})
                except:
                    pass
            if units and (not dictMisc is None):              
                if units:                
                    if not name in dictMisc:#create the node
                        dictMisc.update({name:{'units':units}})
                    else:
                        dictMisc[name].update({'units':units})
            if doc and (not dictMisc == None):
               
                if not name in dictMisc:#create the node
                    dictMisc.update({name:{'doc':doc}})
                else:
                    dictMisc[name].update({'doc':doc})

    ## Use this function to handle specific keywords that need to be interpreted as string
    ## but they might be reserved words (like 'float')
    def checkException(self,name,value):
        if(name.lower() == 'data_type'):
            return value.upper()
        else:
            return value



    def getNormalizedPropertyName(self, node):
        """
        getPropertyName(self, node)
        return the normalized property name
        (remove spaces and capitalizations).
        """
        name = self.normalize_prop_name(self.getPropertyName(node))
        return name

    def getPropertyName(self, node):
        """
        getPropertyName(self, node)
        Look for the 'property' public name either as an
        attribute of the 'property' tag or as a separate
        tag named 'name'.
        """
        name = self.getPropertyElement(node, 'name')
        return name

    def getPropertyValue(self, node):
        """
        getPropertyValue(self, node)
        Given an input node, return the value of the property.
        The value may either be given in a 'value' tag, a
        'value' attribute, or as the unnamed text contained in
        the property tag.  In the last of these three options,
        all other elements of the property tag must be given as
        attributes of the tag.
        Only one of the three possible styles for any given
        property is allowed.  An exception is raised if more
        than one style ('value' tag, 'value' attribute, or unnamed)
        is given.
        """

        v1 = None

        #unnamed option.
        #If other tags are given, element tree returns None
        v1 = node.text
        if v1:
            v1 = v1.strip()

        #attribute and/or tag options handled by getPropertyElement
        try:
            v2 = self.getPropertyElement(node, 'value')
        except IOError as msg:
            msg1 = None
            if v1:
                msg1 = "Input xml file uses unnamed 'value' style.\n"
            msg = msg1 + msg
            raise IOError(msg)

        if v1 and v2:
            msg = "Input xml file uses 'unnamed' value style and also either"
            msg += "\n    the 'attribute' or 'tag' value style "
            msg += "for property '{0}'.".format(self.getPropertyName(node))
            msg += "\n    Choose only one of these styles."
            logging.error(msg)
            raise IOError(msg)


        if not v1 and not v2:
            msg = "No valid value given for property "
            msg += "'{0}'in the input file.".format(self.getPropertyName(node))
            msg += "\n    A possible mistake that could cause this problem is"
            msg += "\n    the use of 'unnamed value' style along with other"
            msg += "\n    tags (as opposed to attributes) in a property tag."
            msg += "\n    The 'unnamed value' style works best is all other"
            msg += "\n    property elements are attributes of the property tag."
            logging.warning(msg)
#            raise IOError(msg)

        return v1 if v1 else v2

    def getPropertyElement(self, node, elementName):
        """
        getPropertyElement(self, node, elementName)
        Given an input node and the node elementName return
        the value of that elementName of the property.
        Look for the 'property' element either as a sub-tag or
        as an attribute of the property tag.  Raise an exception
        if both are used.
        """
        e1 = e2 = None

        #attribute style, returns None if no such attribute
        e1 = node.get(elementName)

        #tag style, not so forgiving if absent
        #also need to strip leading and trailing spaces
        try:
            e2 = node.find(elementName).text.strip()
        except:
            pass

        if e1 and e2:
            msg  = "Input xml file uses attribute and tag styles"
            msg += "for element {0} = '{1}'.".format(elementName, e1)
            msg += "\n   Choose one style only."
            raise IOError(msg)
            return

        return e1 if e1 else e2

    # listComp is the list of nodes that we need to follow in propDict.
    # at the last one we set the val
    def updateParameter(self,propDict,listComp,val):
        if len(listComp) > 1:#more node to explore
            if not listComp[0] in propDict:#create if node not present
                propDict.update({listComp[0]:{}})
            #go down to the next passing the remaining list of components
            self.updateParameter(propDict[listComp[0]],listComp[1:],val)
        else:#we reached the end of the dictionary
            propDict[listComp[0]] = val


    def isStr(self, obj):
        try:
            eval(obj)
            return False
        except:
            return True

    def normalize_comp_name(self, comp_name):
        """
        normalize_comp_name removes extra white spaces and
        capitalizes first letter of each word
        """
        from isceobj.Util.StringUtils import StringUtils
        return StringUtils.capitalize_single_spaced(comp_name)

    def normalize_prop_name(self, prop_name):
        """
        normalize_prop_name removes extra white spaces and
        converts words to lower case
        """
        from isceobj.Util.StringUtils import StringUtils
        return StringUtils.lower_single_spaced(prop_name)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d
    def __setstate__(self,d):
        self.__dict__.update(d)
        self.logger = logging.getLogger('isce.iscesys.Parser')
    def __init__(self):
        self._filetypes = ['xml'] # add all the types here
        self.logger = logging.getLogger('isce.iscesys.Parser')
        self._metafile = None

def main(argv):
    # test xml Parser. run ./Parser.py testXml1.xml
    #from XmlParser import XmlParser
    #XP = XmlParser()
    #(propDict,factDict,miscDict) = XP.parse(argv[0])
    PA = Parser()
    #(propDict,factDict,miscDict,opts) = PA.commandLineParser(argv[:-1])
    (propDict,factDict,miscDict,opts) = PA.commandLineParser(argv)

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv[1:]))
