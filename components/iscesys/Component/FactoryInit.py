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
from inspect import getmodule
from getopt import getopt
import xml.etree.ElementTree as ET
# remember to keep this updated. would be better the have an import of
# components


## JEB: Cannot remove these until we have unittestin for each app.
from isceobj import *
from iscesys import *
from stdproc import *
from mroipac import *

from isceobj.XmlUtil.XmlUtil import XmlUtil
from iscesys.Component.InitFromDictionary import InitFromDictionary


# This class provides a set of methods that allow objects to be initialized
# from command line. The standard way is to create a FactoryInit object and
# invoke the initFactory(arglist) where the arglist is the argument list
# passed at command line which can be obtained directly from sys.argv.
# The first argument could be the name of an xml file. This file contains the
# objects to be initialized and the initializers adopted.
# A possible example of such a file is: \n
#\verbatim
#<component>
#    <name>nameOfThisComponent</name>
#    <property>
#        <name>FirstObjectName</name>
#        <factoryname>NameOfTheClassOfTheObject</factoryname>
#        <factorymodule>NameOfTheFileWhereTheClassIs</factorymodule>
#        <factorylocation>DirectoryWhereTheFileIs</factorylocation>
#        <initclass>NameOfTheClassOfTheInitilaizer</initclass>
#        <initmodule>NameOfTheFileOfTheInitializer</initmodule>
#         <initlocation>LocationWhereTheFileOfTheInitializerIs</initlocation>
#        <value>ValuePassedToTheInitializer</value>
#    </property>
#    <property>
#        <name>SecondObjectName</name>
#        <factoryname>NameOfTheClassOfTheObject</factoryname>
#        <factorymodule>NameOfTheFileWhereTheClassIs</factorymodule>
#        <factorylocation>DirectoryWhereTheFileIs</factorylocation>
#        <initclass>NameOfTheClassOfTheInitilaizer</initclass>
#        <initmodule>NameOfTheFileOfTheInitializer</initmodule>
#         <initlocation>LocationWhereTheFileOfTheInitializerIs</initlocation>
#        <value>ValuePassedToTheInitializer</value>
#    </property>
#</component>
#\endverbatim
# The factory name can be omitted if there is only one class defined in the
# "factorymodule" file. If the "initlocation" is moroisys.Component,  then it
# could be omitted as well or set to "default". If the "initclass" is omitted
# then the InitFromDictionary is understood. The "value" indicates the object
# that the initializer needs. For instance if the "initclass" is
# InitFromXmlFile, then "value" is the name of the xml file from where the
# object is going to be initialized. If "initclass" is InitFromDictionary,
# then value will be a python dictionary. All the locations could be specified
# as directories (i.e. names separated by forward slashes) or as python paths
# (i.e. names separated by dots).\n
# The rest of the commad line arguments is a list of keyword/value pair
# starting with --name. All the keywords are the same as the one  shown in
# the above example with the modification <keyword> ==> --keyword, so
# <name> ==> --name, <factoryname> ==> --factoryname and so on.
# After each keyword a value must be specified.\n
# The arguments following the xml initialization file, will overwrite the
# values of the objects already specified in such a file.
# Note: from command line put in quotes the value following the keyword
# --value. For instace for a dictionary write
#\verbatim
# --value "{'VARIABLE1':value1,'VARIABLE2':value2}"
# or
# --value '{"VARIABLE1":value1,"VARIABLE2":value2}'
#\endverbatim
#or for a filename
#\verbatim
# --value "filename" or --value 'filename'
#\endverbatim
# Note also that if a quantity inside the dictionary is a string, then use
# different quotes from the ones enclosing the dictionary. Any other
# combination seems to fail. It has to do with the interpretation of the
# shell of single and double quotes.
class FactoryInit(object):


    ## More Better format for initFactory
    def init_factory(self, *argv):
        return self.initFactory(argv)

    ## Invoke this method by passing the argument list that can be obtained
    ## from the sys module by invoking sys.argv. The object specified in
    ## the initlialization xml file and in the following argumnt list will
    ## be initialized.
    ## @param argv command line argument list.
    def initFactory(self, argv):
        if isinstance(argv, basestring):
            argv = [argv]
        if not argv:
            raise ValueError('Error. The argument list is empty')
        argList = []
        # separate the arg for each component. there is always the name
        # followed by the other info
        argComp = []
        for arg in argv:
            if arg == '--name':
                # append previous. the first time could be the xml file with
                # default that are superseeded by the command line list
                argList.append(argComp)
                argComp = ['--name']
            else:
                argComp.append(arg)
        #add lat one
        argList.append(argComp)
        for arg in argList:
            if arg:
                if arg[0] == '--name':# is a component
                    # for the command line it will make it easier to change
                    # parameters using a dictionary
                    self.defaultInitModule = 'InitFromDictionary'
                    #all long args i.e. preceeded by --
                    optlist, args = getopt(arg, '', self.listOfArgs)
                    # put the result in a dictionary format in which the --
                    # is removed from the key
                    tmpDict = {}
                    name = ''
                    for pair in optlist:
                        if pair[0] == '--name': #name is always the first one
                            name = pair[1]
                        else:
                            tmpDict[pair[0][2:]] = pair[1] #remove the -- from pair[0]
                    self.optDict[name] = tmpDict
                    if args:# must be empty
                        raise ValueError('Error. Not expecting single argument')

                else: #first part with file info
                    self.fileInit = arg[0]
                    #use initFromXmlFile as default
                    self.defaultInitModule = 'InitFromXmlFile'
                    self.initComponentFromFile()

        # loop though the list of components that need to be updated after
        # being initialized from file. otherwise create a new one
        for key, comp in self.optDict.items():
            #the component already exists -> update it
            if key in self.dictionaryOfComponents:
                instance = self.dictionaryOfComponents[key]
                self.factoryInitComponent(key, comp, instance)
            else:
                self.factoryInitComponent(key, comp)


    def initComponentFromFile(self):
        objXmlUtil = XmlUtil()
        retDict =  objXmlUtil.createDictionary(
            objXmlUtil.readFile(self.fileInit)
            )
        # each component "key" has a dictionary where the keys are the
        # listOfArgs values (not necessarily all)
        for key, comp in retDict.items():
            self.factoryInitComponent(key, comp)
        return

    # if the string contained in obj is an actual object, when is exec there is
    # no problem. if it was supposed to be a string, the name will not be
    # defined aand an exception is thrown. put in a function to reduce chance
    # that the name is actually defined (smaller scope)
    def isStr(self, obj):
        """ A function always called with str() arguemnt-- so
        what happens- what do I do?"""
        retVal = False
        try:
            exec('a = ' + obj)
        except:
            retVal = True
#        raw_input("<"+str(retVal)+"|"+obj+">")
        return retVal


    def factoryInitComponent(self,name,comp, *args, **kwargs):

        #Python 3 will make this unnecessary with its new function syntax to
        #indicate end of positional arguments so that keyword arguments
        # can not suck up extra positional arguments
        # (see http://www.python.org/dev/peps/pep-3102/)
        # for now it is necessary that all named arguments appear in kwargs,
        # including the known instanceObj.
        if kwargs.has_key('instanceObj'):
           instanceObj = kwargs.pop('instanceObj')
        else:
           instanceObj = None

        initDictionary = {} #dictionary with the initializers
        instanceInit = None
        value = None

        # admit the possibility of not ititializing at this point. if value
        # does not exist that just instanciate the object with the factory
        # name and do not initialize
        if 'value' in comp:
            value = comp['value']
            initLocation = ''
            if 'initlocation' in comp:
                if  comp['initlocation'] == 'default':
                    initLocation = self.defaultInitLocation
                else:
                    initLocation = comp['initlocation']
            else:
                initLocation = self.defaultInitLocation

            initLocation = initLocation.replace('/', '.')

            initModule = ''
            if 'initmodule' in comp:
                initModule = comp['initmodule']
            else:
                initModule = self.defaultInitModule
            if initModule.endswith('.py'):
                initModule = initModule.replace('.py','')
            try:
                command = 'from ' + initLocation + ' import ' + initModule
                exec(command)
            except ImportError:
                print('Error. Cannot import the module',
                      initModule,'from',initLocation)
                raise ImportError

            initClass = None
            if 'initclass' in comp:
                initClass = comp['initclass']
                instance = None
                if self.isStr(str(value)):
                    exec('instance = ' + initModule + '.' + initClass + '(value)')
                else:
                    exec('instance = ' + initModule + '.' + initClass + '(' + str(value) + ')')

                instanceInit = instance

            else:
                exec('listMembers = dir(' + initModule + ')')
                instance = None
                #the following finds the initilizers
                for member in listMembers:
                    #given only the file where the class initializer is
                    # defined,get all the members in that file, then
                    try:# try to instantiate the object from that members  and, if it exists, see if that object was defined in that file i.e. initModule
                        if self.isStr(str(value)):
                            exec('instance = ' + initModule + '.' + member + '(value)')
                        else:
                            exec('instance = ' + initModule + '.' + member + '(' + str(value) + ')')
                        modName = getmodule(instance).__name__
                        modNameList = modName.split('.')#just want the last part
                        modName = modNameList.pop()
                        if modName == initModule:# found right object. create instance
                            instanceInit = instance
                            break
                    except Exception:# the instantiation failed
                        continue


            if instanceObj:
                instanceObj.initComponent(instanceInit)

            else:

                #do the same thing for the object that needs to be instantiated
                factoryLocation = None
                factoryModule = None
                if 'factorylocation' in comp:#if present use it otherwise allow to specify like package1.package2.....packageN.factoryModule
                                             #and extract the necessary information from the factoryModule
                    factoryLocation = comp['factorylocation']
                    factoryLocation = factoryLocation.replace("/",".")
                    try:
                        factoryModule = comp['factorymodule']
                    except KeyError:
                        print('The \'factorymodule\' keyword is not present for the component',name)
                        raise KeyError
                    if factoryModule.endswith('.py'):
                        factoryModule = factoryModule.replace('.py','')
                    try:
                        command = 'from ' + factoryLocation + ' import ' + factoryModule
                        exec(command)
                    except ImportError:
                        print('Error. Cannot import the module',factoryModule,'from',factoryLocation)
                        raise ImportError
                else:
                    try:
                        factoryModule = comp['factorymodule']
                    except KeyError:
                        #print('The \'factorymodule\' keyword is not present for the component',name)
                        #raise KeyError
                        factoryModule = None
                    if not factoryModule == None:
                        if factoryModule.endswith('.py'):
                            factoryModule = factoryModule.replace('.py','')
                        factoryModule = factoryModule.replace("/",".")
                        splitFactoryModule = factoryModule.rpartition(".") #split from last "." in a 3-tuple containing first part, "." and last second part
                        factoryLocation = splitFactoryModule[0]
                        factoryModule = splitFactoryModule[2]
                        try:
                            command = 'from ' + factoryLocation + ' import ' + factoryModule
                            exec(command)
                        except ImportError:
                        #if also acquiring the factoryLocation from the factoryModule didn't work
                        # try to see if the factory name is sufficient
                            factoryModule = None
                            pass
                            #print('Error. Cannot import the module',factoryModule,'from',factoryLocation)
                            #raise ImportError


                factoryName = None
                if 'factoryname' in comp:
                    factoryName = comp['factoryname']
                    #instance = None
                    if factoryModule == None:# than assume that factory name is actually a factory method that does the import and returns the right object
                        exec('instanceObj = ' + factoryName + '(*args,**kwargs)')
                    else:
                        exec('instanceObj = ' + factoryModule + '.' + factoryName + '(*args,**kwargs)')

                    instanceObj.initComponent(instanceInit)
                    self.dictionaryOfComponents[name] = instanceObj

                else:
                    exec('listMembers = dir(' + factoryModule + ')')
                    #instance = None
                    #the following finds the initilizers
                    for member in listMembers:
                        #given only the file where the class initializer is defined,get all the members in that file, then
                        try:# try to instantiate the object from that members  and, if it exists, see if that object was defined in that file i.e. factoryModule
                            exec('instanceObj = ' + factoryModule + '.' + member + '()')
                            modName = getmodule(instanceObj).__name__
                            modNameList = modName.split('.')#just want the last part
                            modName = modNameList.pop()
                            if modName == factoryModule:# found right object. crate instance
                                self.dictionaryOfComponents[name] = instanceObj
                                instanceObj.initComponent(instanceInit)
                                break
                        except Exception:# the instantiation failed
                            continue
        else:# if there is no value keyword that assume that the object doen not need to be init, at least at this time. moreover here we assume that factoryName is actually a factory method

            try:
                factoryName = comp['factoryname']
            except KeyError:
                print('The \'factoryname\' keyword is not present for the component',name)
                raise KeyError
            exec('instanceObj = ' + factoryName + '(*args,**kwargs)')
            self.dictionaryOfComponents[name] = instanceObj
##
#Set a different default "initlocation". The default one is iscesys.Component

    def setDefaultInitLocation(self,default):
        self.defaultInitLocation = default
##
# Get an instance of the object "factoryname". The name of the instance is the one used in the initialization xml file (<name>ObjectName</name>) and/or in the command line --name.
#@param name name of the particular object.
    def getComponent(self,name):
        try:
            return self.dictionaryOfComponents[name]
        except KeyError:
            print('The requested component',name,'is not present')
            raise KeyError
        pass

    ## debug counter to see if it is being used
    _count = 0
    ## Default init location
    defaultInitLocation = 'iscesys.Component'
    ## Default initializer
    defaultInitModule = 'InitFromDictionary'
    ## list of args
    listOfArgs = ['name=','value=', 'factoryname=', 'factorymodule=',
                           'factorylocation=','initclass=','initlocation=',
                           'initmodule=']
    def __init__(self):
        self.optDict = {}
        self.fileInit = ''
        self.dictionaryOfComponents = {}
        self._count + 1
        return None
