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
import os
import sys
import operator
from isce import logging
from iscesys.DictUtils.DictUtils import DictUtils as DU
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from isceobj.Util import key_of_same_content


## Flag to (dis/en)- able exec statements for True/False- this is for
## development, since the "exec" imports certainly work, while the
## __import__() calls *should* work, though they whole thing should
## use 2.7's importlib module (I don't have it- JEB).
EXEC = False

from iscesys.Traits import traits

def containerize(a, ttyp, ctyp):
    """
    Convert a string version of a list, tuple, or comma-/space-separated
    string into a Python list of ttyp objects.
    """
    if not isinstance(a, str):
        return a

    #strip off the container indicator ('[', ']' for list, '(', ')' for tuple)
    if '[' in a:
        a = a.split('[')[1].split(']')[0]
    elif '(' in a:
        a = a.split('(')[1].split(')')[0]

    #At this point a is a string of one item or several items separated by
    #commas or spaces. This is converted to a list of one or more items
    #of type ttyp and then cast to the container type (ctyp).  It is
    #required that the constructor of the container type takes a list
    #as argument (as is the case for list, tuple, numpy.array to name a few).
    if ',' in a:
        return ctyp([ttyp(x.strip()) for x in a.split(',')])
    else:
        return ctyp([ttyp(x.strip()) for x in a.split()])


def apply_type(value, dtype, ctype=None):
    '''
    Function to convert a string representation of an entity's
    value to the dtype given as input.  Handles an optional
    argument named 'container' to convert the input string
    value into a list of entities of the type 'dtype'.
    '''
    if isinstance(dtype, str):
        dtype = traits[dtype]

    #Check if container is defined
    if ctype:
        if isinstance(ctype, str):
            #if ctype is a string get the actual trait
            ctype = traits[ctype]
        return containerize(value, dtype, ctype)
    else:
        return dtype(value)

    print("dtype {} not in known traits".format(dtype))
    return


## A metaclass for all configurables
class configurable(type):

    ## Bitwise verbose flag
    VERBOSE = 0

    ## All Configurable class creations go through this method-- ALL of them
    def __new__(mcs, *args, **kwargs):
        if mcs.VERBOSE & 1: print("NEW:", mcs, args, kwargs)
        cls =  type.__new__(mcs, *args, **kwargs)

#        ## Experimental dictionaryOfVariables manipulation,
#       ToDO: build deriviative dictionaries here
        if ( 0 and
            hasattr(cls, 'dictionaryOfVariables') and
            not isinstance(cls.dictionaryOfVariables, dict)
            ):
            cls.dictionaryOfVariables = DictionaryOfVariables(
                dict_ = cls.dictionaryOfVariables
                )

        return cls


    ## All Configurable instantiations go through this method-- ALL of them
    def __call__(cls, *args, **kwargs):
        if cls.VERBOSE &  2: print("CALL:", cls, args, kwargs)
        inst =  super(configurable, cls).__call__(*args, **kwargs)
        return inst
    pass





class EmptyFacility(object):
    """EmptyFacility used in initial component creation of a component declared
    as a facility so that its type at least indicates what is intended until the
    actual component is created"""
    pass


## class for the ubiquitous dictionaryOfVariables-- it emulates the dictionary
## format (mostly) in place now, and add functionality and structure that should
## make usage clearer.
class DictionaryOfVariables(object):
    """DictionaryOfVariables(var1, var2, ..., varN, dict_={})

    makes a dictionary of variables.
    """
    ## Construct from a dictionary (dict=...) or from a variable argument
    ## list (*args)-- but they better be Variables -or else
    def __init__(self, *args, **kwargs):
        self.dict_ = kwargs.get("dict_") or {}
        try:
            for item in args:
                self.dict_.update(item.to_dict())
        except (AttributeError, TypeError) as err:
            if not hasattr(self.dict_, 'update'):
                raise TypeError("dict_ keyword is not a dictionary")
            else:
                if not isinstance(item, Configurable.Variable):
                    raise TypeError("argument is not a Variable instance")
            raise err("Undiagnosed Error in __init__")
        return None

    ## Trivial extensions pass behavior to dict_
    def __iter__(self): return iter(self.dict_)
    def __eq__(self, other): return self.dict_ == other.dict_
    def __getitem__(self, index): return self.dict_.__getitem__(index)
    def __setitem__(self, index, value):
        return self.dict_.__setitem__(index, value)
    def itervalues(self): return self.dict_.itervalues()
    def iterkeys(self): return self.dict_.iterkeys()
    def iteritem(self): return self.dict_.iteritem()
    def values(self): return self.dict_.values()
    def keys(self): return self.dict_.keys()
    def item(self): return self.dict_.item()


    ## Private filter of dict_'s items with a function, func.
    def _filter(self, func):
        result = {}
        for key, value in self.dict_.iteritems():
            if func(value):
                result.update({key:value})
                pass
            pass
        return self.__class__(dict_=result)

    ## get a DictionaryOfVariables of mandatory variables
    def mandatory(self):
        return self._filter(bool)

    ## get a DictionaryOfVariables of optional variables
    def optional(self):
        return self._filter(operator.not_)

    pass


class SELF():
    """
    A class to use for Facility declaration to indicate
    that an argument is self.  A kludge to work with
    _RunWrapper class objects that are Facilities.
    """
    def __init__(self):
        pass

## The base Framework object that implements confugurability.
class Configurable(object):


    ## A Parameter class- supports all types, not just primiatives.
    class Parameter(object):
        '''Parameter( attrname,
        public_name="",
        default=None,
        container=None,
        type=type,
        mandatory=False,
        units=None,
        doc="""Please provide a docstring""",
        private=False):
        '''
        ## if True, do type checking in __init__().
        warn = False
        def __init__(self, attrname,
                     public_name="",
                     default=None,
                     container=None,
                     type=type,
                     mandatory=False,
                     units=None,
                     doc="""Please provide a docstring""",
                     private=False,
                     intent='input'):

            if self.__class__.warn:
                raise NotImplementedError

            ## This name will be assigned to the Configurable's subclass's
            ## __dict__ -- it *will* be an instance attribute.
            self.attrname = str(attrname)
            ## This name will be used at the command line or xml interface
            ## to indentify the parameter
            self.public_name = public_name or attrname
            ## Handle container option.  The container attribute is a container
            ## type that can cast a list into an instance of the container.
            ## The elements of the list are specified by the 'type' attribute.
            self.container = container
            ## This is the paramater's data type-- or a tuple of allowable
            ## data types, though that is not fully implemented
            self.type = type
            ## This is the default value - should be of type self.type,
            ## in theory
            self.default = default
            ## units may be used someday
            self.units = units
            ## if private = False -> Parameter is mandatory iff True. It's optional (i.e. if not provided is set
            # to a default iff False). It still must be set before running
            # if private = True -> Parameter COULD be provided by the user if mandatory is False but it does not
            #need to be set before running
            # User cannot set it if mandatory = True and private is True
            self.mandatory = mandatory
            ## A helpful docstring for the user, check PEP257.
            self.doc = doc
            self.private = private
            self.intent = intent
            return None

        ## Calling a parameter makes an instance of its type
        def __call__(self, *args, **kwargs):
            return self.type(*args, **kwargs)

        ## str is attrname
        def __str__(self):
            return self.attrname

        def __repr__(self):
            result = self.__class__.__name__ + "('" + str(self) + "'"
            result += ", public_name='%s'"  % self.public_name
            result += ", default=%s"  % str(self.default)
            try:
                s = self.type.__name__
            except AttributeError:
                s = str(self.type)
            result += ", type=%s"  % s
            result +=", units=%s" % str(self.units)
            result +=", mandatory=%s" % str(self.mandatory)
            result +=", private=%s" % str(self.private)

            return result + ")"

        ## bool is mandatory
        def __nonzero__(self):
            return self.mandatory

        ## A way to map camelCase to CAMEL_CASE (well, all caps).
        def upper(self):
            result = ""
            for n, c in enumerate(str(self)):
                result += "_"+c if n and c.isupper() else c.upper()
            return result

        ## is default even the right type?
        def test_default(self):
            return isinstance(self.default, self.type)

        pass


    ## Facility imports itself (module) and sets up its own function (factory)
    ## and then can execute it self with a list of parameters (__call__),
    ## finally, it can assign it self to a component-- or is that an
    ##  antipattern?
    class Facility(object):
        '''Parameter(attrname,
        public_name="",
        module=,
        factory=,
        args=(),
        kwargs={},
        mandatory=False,
        doc="""Please provide a docstring""",
        private=False):
        '''
        ## if True, do type checking in __init__().
        warn = False

        ## The callable factory is None, until it is set.
        factory = None

        def __init__(self,
                     attrname,
                     public_name="",
                     module=None,
                     factory=None,
                     parameter_names=(),
                     args=(),
                     kwargs={},
                     mandatory=False,
                     doc="""Please provide a docstring""",
                     private=False):

            if self.__class__.warn:
                raise NotImplementedError

            if args and parameter_names:
                message = "Cannot set args keyword if parameter_names is set"
                raise ValueError(message)

            ## This name will be assigned to the Configurable's subclass's
            ## __dict__ -- it *will* be an instance attribute.
            self.attrname = str(attrname)
            ## This name will be used at the command line or xml interface
            ## to indentify the parameter
            self.public_name = public_name or attrname

            self.args = args
            self.parameter_names = parameter_names

            self.module_name = module
            self.factory_name = factory

            ## Parameter is (not) mandatory iff True (False).
            self.mandatory = bool(mandatory)
            ## A helpful docstring for the user, check PEP257.
            self.doc = doc

            self.private = private

            return None

        ## Got get the factory in the name
        def import_factory(self, fromlist=None):
            self.factorymodule = __import__(self.module_name,
                                            fromlist=fromlist or [''])
            self.factory = getattr(self.factorymodule, self.factory_name)
            return None


        ## Get arguments from the component's parameters
        def extract_args_from_component(self, component):
            return [
                getattr(component, attr) for attr in self.arg_names
                ]

        ## get args- however they are defined
        def _get_args(self, component=None):
            return self.args or self.extract_args_from_component(component)

        ## Calling a facility runs it with its arguments
        def __call__(self, component=None):
            if not self.factory:
                self.import_factory()
            result = self.factory(*self._get_args(component))
            return result

        ## call it and assign it to component--whether this is good idea is
        ## TBD-- maybe the componenet's method should modify itself?
        def execute(self, component=None):
            try:
                result = setattr(component, str(self), self(component))
            except AttributeError:
                ## Since this is wrapped in a sys.exit call, this should
                ## give it a meaningful number. maybe.
                import errno
                result = errno.EOPNOTSUPP
            return result

        ## str is attrname
        def __str__(self):
            return self.attrname

        ## bool is mandatory
        def __nonzero__(self):
            return self.mandatory

        ## A way to map camelCase to CAMEL_CASE (well, all caps).
        def upper(self):
            result = ""
            for n, c in enumerate(str(self)):
                result += "_"+c if n and c.isupper() else c.upper()
            return result
        pass

    ## A way to hold variables in the dictionary of variables
    ## experimental implementation in Formslc.
    class Variable(object):
        """Variable(name, dtype, mandatory, key=None)

        name is a sting
        dtype is a type
        mandatory is a bool

        If key is set to a name, then that name will be associated
        with the variable; otherwise, it is computed from "name"
        in the NAME method.
        """
        selfPos = 0
        typePos = 2

        ## name is attrname, key is public name
        def __init__(self, name, dtype, mandatory, key=None):
            self.name = str(name)
            self.dtype = dtype
            self.mandatory = bool(mandatory)
            self.key = key
            return None

        ## Create object, of any type dtype.
        def __call__(self, *args, **kwargs):
            return self.dtype(*args, **kwargs)

        ## Bool is the mandatory flag None case is not supported
        def __nonzero__(self):
            return self.mandatory

        ## String is the name.
        def __str__(self):
            return self.name

        ## repr is like a tuple
        def __repr__(self):
            return repr(self.to_tuple())

        ## iter is like a tuple
        def __iter__(self):
            return iter(self.to_tuple())

        ## like a tuple
        def __getitem__(self, index):
            return self.to_tuple()[index]

        ## a tuple
        def to_tuple(self):
            return (self.name, self.dtype, self.mandatory)

        ## Default name convention:
        ## camelCase --> CAMEL_CASE
        def NAME(self):
            if self.key:
                return self.key
            result = ""
            for n, c in enumerate(self.name):
                result += "_"+c if n and c.isupper() else  c.upper()
            return result

        ## Make self into a dictionary item
        def to_dict(self):
            return {self.NAME(): self}

        pass
    #has to be inside the class in this case since the Parameter is defined inside
    METADATA_LOCATION  = Parameter('metadatalocation',
                              public_name='METADATA_LOCATION',
                              default='',
                              type=str,
                              mandatory=False,
                              private=True,
                              doc='Location of the metadata file where the instance was defined')
    ## Metaclass allows control of Class/Instance creation.
    __metaclass__ = configurable

    ## A configurable objects parameter list
    parameter_list = (METADATA_LOCATION,)
    ## A configurable objects facilities list (TBD)
    facility_list = ()
    ## A configurable objects facilities list (TBD)
    port_list = ()
    def get_parameter_names(self, func=lambda x: True):
        return map(str, filter(func, self.parameter_list))

    def get_parameter_values(self, func=lambda x: True):
        return map(self.attrgetter, self.get_parameter_names(func=func))

    ## Build a dictionary of {attrname:value} -- basically a __dict__
    def get_parameter_dictionary(self, func=lambda x:True):
        return dict(
            zip(self.get_parameter_names(func=func),
                self.get_parameter_values(func=func))
            )

    def get_mandatory_parameters(self):
        return filter(bool, self.parameter_list)

    def get_optional_parameters(self):
        return filter(operator.not_, self.parameter_list)


    ## TBD method: passing a facility to youself should call it
    ## and assign it? Thus, I think,
    #map(self.set_facility_attr, self.facility_list) #should run everything?
    def set_facility_attr(self, facility):
        result = facility(self)
        setattr(self, str(facility), result)
        return result

##
# The object to be initialized calls this inherited method, passing the
# initializer object (see initFromFile.py or InitFromDictionary.py as
# examples of initializers). and gets initialized.
# @param initObject instance of a particular initializer class.
##
    def initComponent(self,initObject):
        retDict = initObject.init(self)
        self.init(retDict)

##
# This method extracts the information returned in a dictionary of dictionaries
# by the "init(): method of the initializer object. If for example the returned
# dictionary is:
#\verbatim
#{'SPACECRAFT_NAME':{'value':'ERS1','doc': 'European Remote Sensing Satellite'}, 'BODY_FIXED_VELOCITY',:{'value':7552.60745017,'doc':'velocity of the spacecraft','units':'m/s'}}
#\endverbatim

# and the self.dictionaryOfVariables is:
#\verbatim
# self.dictionaryOfVariables = {'SPACECRAFT_NAME':{'attrname':'spacecraftName','type':'str','mandatory':True},
# 'BODY_FIXED_VELOCITY':{'attrname':'bodyFixedVelocity', 'type':'float','mandatory':True]}
#\endverbatim
# the self.spacecraftName  is set to 'ERS1' and self.bodyFixedVelocity is set to 7552.60745017, while the self.descriptionOfVariables will be set to
#\verbatim
#self.descriptionOfVariables = {'SPACECRAFT_NAME':{'doc': 'European Remote Sensing Satellite'}, 'BODY_FIXED_VELOCITY',:{'doc':'velocity of the spacecraft','units':'m/s'}}
#\endverbatim

    from datetime import datetime as dt

    def renderToDictionary(self,obj,propDict,factDict,miscDict):
        obj.reformatDictionaryOfVariables()
    
        #remove meaningless values from the dictionaries
        for k,v in obj.dictionaryOfVariables.items():
            val = getattr(obj, v['attrname'])
            #Ignore the EmptyFacilities
            if isinstance(val,EmptyFacility):
                continue
            if v['type'] == 'component':#variable type
                propDict[k] = {}
                miscDict[k] = {}
                #check if the key are equivalent and possible replace the one in the dict with k
                if DU.keyIsIn(k, obj._dictionaryOfFacilities, True):
                    factDict[k] = obj._dictionaryOfFacilities[k]
                    if factDict[k]['factoryname'] == 'default':
                        module,factory = self._getFacilityInfoFromObject(val)
                        factDict[k] =  {
                                    'factorymodule':module,
                                    'factoryname':factory
                                    }
                else:
                    factDict[k] = {}

                #see method comment for detail
                if val is not None:
                    val.adaptToRender()
                    self.renderToDictionary(val,propDict[k],factDict[k],miscDict[k])
                    val.restoreAfterRendering()
                else:
                    if self.logger:
                        self.logger.warning(
                            "component {} is empty in object of type {}".format(
                                v['attrname'], type(obj))
                        )
                    else:
                        print(("***information: "+
                               "component {} is empty in object of type {}").format(
                                   v['attrname'], type(obj))
                        )
            else:

                propDict.update({k:val})
                if k in obj.unitsOfVariables:
                    miscDict[k] = {'units':obj.unitsOfVariables[k]['units']}
                if k in obj.descriptionOfVariables:
                    try:
                        miscDict[k].update({'doc':obj.descriptionOfVariables[k]['doc']})
                    except KeyError:
                        miscDict[k] = {'doc':obj.descriptionOfVariables[k]['doc']}


    def _getFacilityInfoFromObject(self,obj):
        module = obj.__module__
        fact = obj.__class__.__name__
        return module,fact
    #abstract method if the object needs to do some reformatting
    #which might be needed if some of the attributes cannot be serialized correctly
    def adaptToRender(self):
        pass

    #abstract method to be called after adaptToRender to repristinate the original format
    def restoreAfterRendering(self):
        pass
    def reformatDictionaryOfVariables(self):
        newDict = {}
        for k,v in self.dictionaryOfVariables.items():
            if isinstance(v,list):
                if k in self.dictionaryOfOutputVariables:
                    intent = 'output'
                else:
                    intent = 'input'
                newDict[k] = {'attrname':v[0].replace('self.',''),'type':v[1],
                              'mandatory':True if v[2] == 'mandatory' else False,'private':False,
                              'intent':intent}
            elif isinstance(v, dict):
                newDict[k] = v
            else:
                continue
            self.dictionaryOfVariables = newDict


    def init(self,propDict=None,factDict=None,docDict=None,unitsDict=None):

        if propDict is None:
            propDict = {}
        else:
            propDict = DU.renormalizeKeys(propDict)

        if factDict is None:
            factDict = {}
        else:
            factDict = DU.renormalizeKeys(factDict)

        if docDict is None:
            docDict = {}
        else:
            docDict = DU.renormalizeKeys(docDict)

        if unitsDict is None:
            unitsDict = {}
        else:
            unitsDict = DU.renormalizeKeys(unitsDict)

        self.catalog = DU.renormalizeKeys(self.catalog)
        self._dictionaryOfFacilities = DU.renormalizeKeys(
            self._dictionaryOfFacilities
            )

        self.descriptionOfVariables = DU.renormalizeKeys(
            self.descriptionOfVariables
            )

        self.unitsOfVariables = DU.renormalizeKeys(self.unitsOfVariables)

        #update the various dictionaries with what was read from command line
        if not propDict == {}:
            # the top level has only one key that is the appicaltion name
            DU.updateDictionary(self.catalog,propDict,replace=True)

        if not factDict == {}:
            # the top level has only one key that is the appicaltion name
            #note: the _dictionaryOfFacilities has also a doc str. add this as a spare keyword so the
            # content will be appended instead of replaced
            DU.updateDictionary(self._dictionaryOfFacilities,factDict,replace=True,spare='doc')

        if not docDict == {}:
            #The top level has only one key that is the appicaltion name.
            #the update does a append if there is already an entry with a particular key
            DU.updateDictionary(self.descriptionOfVariables,docDict)

        if not unitsDict == {}:
            #The top level has only one key, the application name. In this case replace and hopefully they put the same units!!!
            DU.updateDictionary(self.unitsOfVariables,unitsDict,replace = True)

        #init recursively
        self.initRecursive(self.catalog,self._dictionaryOfFacilities)


    def initProperties(self,dictProp):
        """ as for calling _facilities, if we make sure that this method is
        called in the contructure we don't have to worry about this part
        #set the defaults first and then overwrite with the values in dictProp
        if property  present
        try:
            self._parameters()
        except:# not implemented
            pass
        """

        self.reformatDictionaryOfVariables()
        from iscesys.Parsers.Parser import const_key
        for k,v in dictProp.items():
            if k == const_key:
                continue
            #if it is  a property than it should be in dictionary of variables
            try:
                #pure property are only present in dictProp
                if(k.upper() not in list(map(str.upper, list(self._dictionaryOfFacilities.keys())))):
                    kp, vp = key_of_same_content(k, self.dictionaryOfVariables)
                    compName = vp['attrname']
                    dtype = vp['type']
                    ctype = vp['container'] if 'container' in vp.keys() else None
                    v = apply_type(v, dtype, ctype)
                    setattr(self, compName, v)

            except:#if it is not try to see if it implements a the _parameter
                if k not in self._parametersExceptions:
                    warnOrErr = 'Error'
                    if self._ignoreMissing:
                        warnOrErr = 'Warning'
                    message='%s. The attribute corresponding to the key '%warnOrErr + \
                        '"%s" is not present in the object "%s".\nPossible causes are the definition'%(str(k),str(self.__class__)) + \
                        ' in the xml file of such attribute that is no longer defined \nin the '+ \
                        'object "%s" or a spelling error'%str(self.__class__)

                    if self.logger:
                        if self._ignoreMissing:
                            self.logger.warning(message)
                        else:
                            self.logger.error(message)
                    else:
                        print(message)
                    if not self._ignoreMissing:
                        sys.exit(1)

    def initRecursive(self,dictProp,dictFact):
        #separate simple properties from factories.
        #first init the properties since some time they might be used by the factories

        self.initProperties(dictProp)

        for k, dFk  in dictFact.items():
            #create an instance of the object
            factorymodule = ''
            factoryname = ''
            args = ()
            kwargs = {}
            mandatory = ''

#            try:
#                kp, dFk = key_of_same_content(k,dictFact)
#            except:
#                if self.logger:
#                    self.logger.error('No entry in the factory dictionary for %s. Cannot create the object.' % k)
#                else:
#                    print('No entry in the factory dictionary for %s. Cannot create the object.' % k)
#                raise Exception

            try:
                factorymodule = dFk['factorymodule']
            except:
                pass

            try:
                factoryname = dFk['factoryname']
                #factoryname = default means that the object is private, it does not need
                #to be initialized and when dumped the factory info will be extracted from
                #the object itself
                if(factoryname == 'default'):
                    continue
            except:
                if self.logger:
                    self.logger.error('Cannot create object without a factory method.')
                else:
                    print('Cannot create object without a factory method.')
                raise Exception

            try:
                args = dFk['args']
            except:
                pass

            try:
                kwargs = dFk['kwargs']
            except:
                pass

            if factorymodule:
                statement= 'from ' + factorymodule + ' import ' + factoryname
#                raw_input("1:"+statement)
                if EXEC:
                    exec(statement)
                else:
                    factoryobject = getattr(
                        __import__(factorymodule, fromlist=['']),
                        factoryname
                        )
                    pass
                pass

#            raw_input("2:"+statement)
            if EXEC:
                factoryMethod = factoryname + '(*args,**kwargs)'
                statement = 'comp = ' + factoryMethod
                exec(statement)
            else:
#                raw_input("1:"+str(factoryobject))
                comp = factoryobject(*args, **kwargs)
                pass

            try:
                p, v = key_of_same_content(k,dictProp)
            except:
                v = {} # no property for this object. eventually the default will be set in initProperties

            if not isinstance(v,dict):
                # something wrong since it should be a complex object and therefore should be defined by a dict.
                if self.logger:
                    self.logger.error('Expecting a dictionary for the attribute',k,'. Instead received',v)
                else:
                    print('Expecting a dictionary for the attribute',k,'. Instead received',v)

            #now look for all the complex objects that are in dictFact and extract the factory
            nextDict = {}
            keyList = ['attrname','factorymodule','factoryname','kwargs','doc','args','mandatory','private']
            for k1, v1 in dFk.items():
                #check that it is not one of the reserved factory keys
                isReserved = False
                for k2check in keyList:
                    if k1 == k2check:
                        isReserved = True
                        break
                if not isReserved:
                    nextDict.update({k1:v1})

            # update with what has been set into _configureThis. Notice that some are not real Configurable, such as the runMethods
            # so they don't have catalog and _dictionaryOfFacilities
            if(hasattr(comp,'catalog') and hasattr(comp,'_dictionaryOfFacilities')):
                #configure the component first
                comp._configureThis()
                falseList =  [True]*2
                self._updateFromDicts([comp.catalog,comp._dictionaryOfFacilities],[v,nextDict],falseList)
                v = comp.catalog
                nextDict = comp._dictionaryOfFacilities
            if not (v == {} and nextDict == {}):#if they are both empty don't do anything
                comp.initRecursive(v,nextDict)

            # now the component is initialized. let's set it into the comp object giving the prescribed name
            kp, vp = key_of_same_content(k,self._dictionaryOfFacilities)

            try:
                #try the dictionaryOfFacilities to see if it is defined
                #and has the attrname.
                #for private parameters that are object the facility method in
                #not implemented, just the property. When reloding the dictionaryOfFacility
                #is updated with the info from the xml file but the 'attrname' is missing
                #so check is tha k was defined in the dictionaryOfVariables since it contains
                #all the parameters
                try:
                    compName = vp['attrname']
                except Exception:
                    if kp in [x.lower() for x in  self.dictionaryOfVariables.keys()]:
                        compName = k

                compName = compName.replace('self.','')# the dictionary of variables used to contain the self.
                setattr(self, compName, comp)
            except:
                if self.logger:
                    self.logger.error('The attribute',k,',is not present in the  _dictionaryOfFacilities.')
                else:
                    print('The attribute',k,',is not present in the _dictionaryOfFacilities.')

##
# This method checks if all the variables are initialized to a meaningful value. It throws an exception if at least one variable is not properly initialzed.
##
    def checkInitialization(self):
        self.reformatDictionaryOfVariables()
        for key , val in self.dictionaryOfVariables.items():
            #when private or when intent is output (which defaults to private False and mandatory False)
            #do not check
            if val['private'] == True or val['type'] == 'component' or  val['intent'] == 'output':
                continue
            attrName = val['attrname']
            valNow = getattr(self,attrName)
            if not valNow and not (valNow == 0):
                raise Exception('The variable %s must be initialized'%key)

    def _parameters(self):
        """Define the user configurable parameters for this application"""

        for item in self.__class__.parameter_list:
            try:
                try:
                    from copy import deepcopy
                    default = deepcopy(item.default)
                except:
                    default = item.default

                setattr(self,
                        item.attrname,
                        self.parameter(item.attrname,
                                       public_name=item.public_name,
                                       default=default,
                                       units=None,
                                       doc=item.doc,
                                       container=item.container,
                                       type=item.type,
                                       mandatory=item.mandatory,
                                       private=item.private,
                                       intent=item.intent
                                       )
                        )
            except AttributeError:
                message = (
                    "Failed to set parameter %s type %s in %s" %
                    (str(item), item.__class__.__name__, repr(self))
                    )
                raise AttributeError(message)
            pass
        return None

    def _facilitiesEmpty(self):
        """
        First pass in configuring a Component requires placeholder facilities
        to be defined before running the _parameters method to create the
        dictionaryOfVariables from Parameters.  This method will do this with
        the EmptyFacility class.
        """

        #Check if the facility_list tuple is empty
        if not self.facility_list:
            #If so, then let _facilities handle this case
            #in case the component redefined _facilities
            self._facilities()

        #Create the facilities as attributes of the component
        #without unpacking the arguments; that will happen in
        #_facilities after the parameters are handled
        for item in self.__class__.facility_list:
            try:
                setattr(self,
                    item.attrname,
                    self.facility(
                        item.attrname,
                        public_name=item.public_name,
                        module=item.module_name,
                        factory=item.factory_name,
                        args=item.args,
                        mandatory=item.mandatory,
                        doc=item.doc
                    )
                )
            except AttributeError:
                message = (
                    "Failed to set facility %s type %s in %s" %
                    (str(item), item.__class__.__name__, repr(self))
                    )
                raise AttributeError(message)
            pass

        return

    def _facilities(self):
        """
        Method that the developer should replace in order to define the facilities of the application
        """

        #Don't do anything if the facility_list is empty
        if not self.facility_list:
            return

        for item in self.__class__.facility_list:
            try:
                #convert item.args that are Parameter instances to the
                #corresponding attribute value that was set in self_parameters
                #also check if one of the args is an instance of SELF class
                #which is sometimes required as an argument to the facility
                #constructor
                largs = list(item.args)
                for i, arg in enumerate(largs):
                    if isinstance(arg, SELF):
                        largs[i] = self
                    elif isinstance(arg, Parameter):
                        largs[i] = getattr(self, arg.attrname)
                    else:
                        largs[i] = arg
                targs = tuple(largs)
                setattr(self,
                    item.attrname,
                    self.facility(
                        item.attrname,
                        public_name=item.public_name,
                        module=item.module_name,
                        factory=item.factory_name,
                        args=targs,
                        private=item.private,
                        mandatory=item.mandatory,
                        doc=item.doc
                    )
                )
            except AttributeError:
                message = (
                    "Failed to set facility %s type %s in %s" %
                    (str(item), item.__class__.__name__, repr(self))
                    )
                raise AttributeError(message)
            pass

        return
    
    def _init(self):
        """
        Method that the developer may replace in order to do anything after parameters are set and before facilities are created
        """
        return

    def _configure(self):
        """
        Method that the developer may replace in order to do anything after facilities are created and before his main method is called.
        """
        return

    def _finalize(self):
        """
        Method that the developer may replace in order to do anything after main is called such as finalizing objects that were created.
        """
        return

    def _processFacilities(self, cmdLineDict):

        self._cmdLineDict = cmdLineDict
        factDict = self._cmdLineDict[0]
        docDict = self._cmdLineDict[1]
        unitsDict = self._cmdLineDict[2]
        #The first key is just the name of the top component, so pass the associated dictionary
        if factDict:
            passFact = factDict[list(factDict.keys())[0]]
        else:
            passFact = {}
        if docDict:
            passDoc = docDict[list(docDict.keys())[0]]
        else:
            passDoc = {}
        if unitsDict:
            passUnits = unitsDict[list(unitsDict.keys())[0]]
        else:
            passUnits = {}
        self.init(self.catalog,passFact,passDoc,passUnits)

        return

    #Note: mandatory private
    #      True      True  -> must be set by the framework before running
    #      True      False -> must be set by the user before running
    #      False     True  -> could be set by the user or framework but is not necessary
    #      False     False -> could be set by user, if not the framework sets a default


    def parameter(self,attrname,public_name=None,default=None,units=None,
                  doc=None,container=None,type=None,mandatory=False,
                  private=False,intent='input'):
        public_name = DU.renormalizeKey(public_name)
        if units:
            # Required to be a dictionary of dictionaries in
            # DictUtils.updateDictionary to match structure
            # created from user inputs in Parser
            self.unitsOfVariables[public_name] = {'units':units}
        if doc:
            # Required to be a dictionary of dictionaries in
            # DictUtils.updateDictionary to match structure
            # created from user inputs in Parser
            self.descriptionOfVariables[public_name] = {'doc':doc}
        if type:
            self.typeOfVariables[public_name] = type

        #for backward compatibility we need to map the mandatory/private to some string
        if (mandatory is True or mandatory == 'True') and private is False:
            mand = 'mandatory'
            self.mandatoryVariables.append(public_name)
        elif (mandatory is False or mandatory == 'False') and private is False:
            mand = 'optional'
            self.optionalVariables.append(public_name)
        #need to add this case. optional means that is needed by if the user does not set it then a default result is used.
        #None means that if not given then it is not used. For instance for the ImageAPI the Caster might not be needed when no casting is required
        elif (mandatory is None or mandatory == 'None') or (mandatory is False and private is True):
            mand = 'None'
        elif (mandatory is True and private is True):
            mand = 'private'
        self.dictionaryOfVariables[public_name] = {'attrname':attrname,
                                              'mandatory':mandatory,
                                              'private':private,
                                              'container':container,
                                              'type':type,
                                              'intent':intent
                                              }
        return default


    def facility(self, attrname, public_name=None, module=None, factory=None,
                 doc=None, args=(), kwargs={}, mandatory=False, private=False):
                
        public_name = DU.renormalizeKey(public_name)

        #Enter the facility in the dictionaryOfFacilities
        self._dictionaryOfFacilities[public_name] = {'attrname':attrname,
                                                     'factorymodule':module,
                                                     'factoryname':factory,
                                                     'args':args,
                                                     'kwargs':kwargs,
                                                     'mandatory':mandatory,
                                                     'private':private
                                                     }

        #check also for string. should change it to make it consistent between
        #parameter and facility
        if (mandatory is True or mandatory == 'True') and private is False:
            mand = 'mandatory'
            self.mandatoryVariables.append(public_name)
        elif (mandatory is False or mandatory == 'False') and private is False:
            mand = 'optional'
            self.optionalVariables.append(public_name)

        #need to add this case. optional means that is needed by if the user
        #does not set it then a default result is used.
        #None means that if not given then it is not used. For instance for the
        #ImageAPI the Cater might not be needed when no casting is required
        elif ((mandatory is None or mandatory == 'None') or
              (mandatory is False and private is True)):
            mand = 'None'
        elif (mandatory is True and private is True):
            mand = 'private'

        #Add to dictionaryOfVariables
        self.dictionaryOfVariables[public_name] = {'attrname':attrname,
                                                   'mandatory':mandatory,
                                                   'private':private,
                                                   'type':'component'
                                                    }
        #Add doc string if given
        if doc:
            self._dictionaryOfFacilities[public_name].update({'doc':doc})

        #Delay creating the instance until we parse the command line and check
        #for alternate factory
        return EmptyFacility()


    def _instanceInit(self):
        # each class has to call this method after calling the super __init__
        # in case of many level of inheritance
        self._parameters()
        self._facilities()

        #init with what we have so far. other callers might overwrite some
        #parameters.
        #note self.catalog is empty. any empty dict would do it
        self.init(self.catalog, self._dictionaryOfFacilities,
                  self.descriptionOfVariables, self.unitsOfVariables)
        self.initOptionalAndMandatoryLists()
        
    ## Given an application the we expose only the "mandatory" attribute which could be True or False.
    # In order to take care of the fact that mandatory = False consists of two cases, i.e. private = False
    # or private = True, we convine that if private is False than the attributes only appears in the
    # application file. If private is True than a parameter with the same name appears also in
    # the private file which is a file with the same name as the application but preceeded by the underscores (like Foo and __Foo)
    # If mandatory = True and private = False it only needs to appear in the appication file
    # without specifying the private attribute since is False by default.
    # Finally for  the case mandatory = True and private = True the attribute only appears in the
    # private file (like __Foo) and it's not exposed to the user

    def updatePrivate(self):
        #Not all instances need to call this, so try
        try:
            import importlib
            module = '.'.join(self.__module__.split('.')[0:-1])
            imp = importlib.import_module(module + '.__' + self.__class__.__name__)
            #truth table for mandatary, private attributes
            #False,False = attribute could be set by user and if not the system must sets it
            #True,False = attribute must be set by user
            #False,True = attribute could be set by user, if not no one sets it because not needed (like Caster)
            #True,True = attribute must be set and the system and not the user i responsible for that

            #if a parameter appears in both lists then sets private = True otherwise add it to the
            # object parameter_list
            toAdd = []
            #NOTE: the import is not a class so no imp.__class__.parameter_list
            for ppar in imp.parameter_list:
                found = False
                for par in self.__class__.parameter_list:
                    if par.attrname == ppar.attrname:
                        par.private = True
                        found = True
                        break

                if not found:
                    toAdd.append(ppar)
            self.__class__.parameter_list += tuple(toAdd)
            #same for facilities
            toAdd = []
            for ppar in imp.facility_list:
                found = False
                for par in self.__class__.facility_list:
                    if par.attrname == ppar.attrname:
                        par.private = True
                        found = True
                        break

                if not found:
                    toAdd.append(ppar)
            self.__class__.facility_list += tuple(toAdd)
        except Exception:
            pass
##
# This method sets self.warning = True. All the warnings are enabled.
#@see unsetWarnings()
#@see self.warnings
##
    def setWarnings(self):
        self.warnings = True

##
# This method sets self.warning = False. All the warnings are disabled.
#@see setWarnings()
#@see self.warnings
##
    def unsetWarnings(self):
        self.warnings = False

    def initOptionalAndMandatoryLists(self):

        self.reformatDictionaryOfVariables()
        for key, val in self.dictionaryOfVariables.items():
            if val['mandatory'] is True:
                self.mandatoryVariables.append(key)
            elif val['mandatory'] is False and val['private'] is False:
                self.optionalVariables.append(key)
            elif val['private'] is True:
                continue
            else:
                if self.logger:
                    self.logger.error('Error. Variable can only be optional or mandatory or None')
                else:
                    print('Error. Variable can only be optional or mandatory or None')
                raise Exception

    def _selectFromDicts(self,dblist):
        ''' Select all the relevant information for this instance from the
        different dictionaries and merge them. Returns a tuple with
        (propDict,factDict,miscDict,unitsDict,docDict)
        with proctDict already with the top key removed
        '''
        #Parse the dblist into the separate configuration dictionaries
        from iscesys.Parsers.Parser import Parser
        PA = Parser()
        propDict, factDict, miscDict, argopts = PA.commandLineParser(
            dblist
            )
        #extract doc from miscDict
        docDict = DU.extractDict(miscDict, 'doc')
        #extract units from miscDict
        unitsDict = DU.extractDict(miscDict, 'units')
        from iscesys.Component.Application import CmdLinePropDict
        from iscesys.Component.Application import CmdLineFactDict
        from iscesys.Component.Application import CmdLineMiscDict
        from iscesys.Component.Application import CmdLineDocDict
        from iscesys.Component.Application import CmdLineUnitsDict

        cmdLinePropDict = DU.renormalizeKeys(CmdLinePropDict())
        cmdLineFactDict = DU.renormalizeKeys(CmdLineFactDict())
        cmdLineMiscDict = DU.renormalizeKeys(CmdLineMiscDict())
        cmdLineUnitsDict = DU.renormalizeKeys(CmdLineUnitsDict())
        cmdLineDocDict = DU.renormalizeKeys(CmdLineDocDict())


        propName = {}
        factName = {}
        miscName = {}
        unitsName = {}
        docName = {}
        # NOTE: all dicts have the key used for search removed

        #NOTE CmdLine... have highest priority
        #extract everything that belongs to self.name from the command line.
        #this has the top priority
        if(self.keyname):
            propName,factName,miscName,unitsName,docName = \
            self._extractFromDicts([cmdLinePropDict,cmdLineFactDict,cmdLineMiscDict,
                                   cmdLineUnitsDict,cmdLineDocDict],self.keyname)

        #extract everything that belongs to self.family from the command line.
        #this has the second highest priority
        propFamily = {}
        factFamily = {}
        miscFamily = {}
        unitsFamily = {}
        docFamily = {}
        if(self.keyfamily):
            propFamily,factFamily,miscFamily,unitsFamily,docFamily =\
            self._extractFromDicts([cmdLinePropDict,cmdLineFactDict,cmdLineMiscDict,
                                   cmdLineUnitsDict,cmdLineDocDict],self.keyfamily)


        propDictF = {}
        factDictF = {}
        miscDictF = {}
        unitsDictF = {}
        docDictF = {}

        #extract everything that belongs to self.family from the dblist that include local and db directory files
        #this has the second highest priority
        if(self.keyfamily in propDict):
            propDictF,factDictF,miscDictF,unitsDictF,docDictF =\
            self._extractFromDicts(
                [propDict,factDict,miscDict,unitsDict,docDict],self.keyfamily
            )

        propDictN = {}
        factDictN = {}
        miscDictN = {}
        unitsDictN = {}
        docDictN = {}
        if(self.keyname in propDict):
            propDictN,factDictN,miscDictN,unitsDictN,docDictN =\
            self._extractFromDicts(
                [propDict,factDict,miscDict,unitsDict,docDict],self.keyname
            )

        self._updateFromDicts([propDictF,factDictF,miscDictF,unitsDictF,docDictF],
                              [propDictN,factDictN,miscDictN,unitsDictN,docDictN],
                              [True,True,True,True,False])

        self._updateFromDicts([propDictF,factDictF,miscDictF,unitsDictF,docDictF],
                              [propFamily,factFamily,miscFamily,unitsFamily,docFamily],
                              [True,True,True,True,False])

        self._updateFromDicts([propDictF,factDictF,miscDictF,unitsDictF,docDictF],
                              [propName,factName,miscName,unitsName,docName],
                              [True,True,True,True,False])


        return propDictF,factDictF,miscDictF,unitsDictF,docDictF


    def help(self):
        """Method that the developer may replace in order to give a helpful
        message to the user

        """

    def dump(self,filename='',dumper='xml'):
        #if not provided use self.name and if not
        if not filename:
            if not self.name:
                if not self.family:
                    message = "Configurable.py:dump(). The filename is not specified"
                    if self.logger:
                        self.logger.error(message)
                    else:
                        print(message)
                    raise Exception

        from iscesys.Dumpers.DumperFactory import createFileDumper
        from isceobj.XmlUtil import xmlUtils as xml
        odProp = xml.OrderedDict()
        odFact = xml.OrderedDict()
        odMisc = xml.OrderedDict()
        dump = createFileDumper(dumper)
        self.renderToDictionary(self, odProp,odFact,odMisc)
        # remove key,value parir with empty value (except if value is zero)
        DU.cleanDictionary(odProp)
        DU.cleanDictionary(odFact)
        DU.cleanDictionary(odMisc)
        firstTag = self.name if self.name else self.family
        dump.dump(filename, odProp, odFact, odMisc, firstTag)

    def load(self,filename,parser='xml'):
        if not filename:
            if not self.name:
                if not self.family:
                    message = "Configurable.py:load(). The filename is not specified"
                    if self.logger:
                        self.logger.error(message)
                    else:
                        print(message)
                    raise Exception

        from iscesys.Parsers.FileParserFactory import createFileParser
        FP = createFileParser(parser)
        tmpProp, tmpFact, tmpMisc = FP.parse(filename)
        docDict = DU.extractDict(tmpMisc, 'doc')
        #extract units from miscDict
        unitsDict = DU.extractDict(tmpMisc, 'units')
        self._parameters()
        self._updateFromDicts([self.catalog],[tmpProp],[True])
        #just to be sure that the facilities, even if default ones,
        #are defined so we can check against the dictionaryOfFacilities
        #to make sure that a property is indeed a property and
        #not a facility (used in initProperties to validate
        #the property)
        self._facilitiesEmpty()
        self.initProperties(self.catalog)

        self._init()

        self._facilities()
        self._dictionaryOfFacilities = DU.renormalizeKeys(self._dictionaryOfFacilities)
        self._updateFromDicts([self._dictionaryOfFacilities],[tmpFact],[True])
        self.init(self.catalog,self._dictionaryOfFacilities,docDict,unitsDict)

        # Run the user's _configure to transfer user-configured facilities to
        # the instance variables
        self._configure()

    def _extractFromDicts(self,listIn,name):
        listOut = []
        for dictIn in listIn:
            listOut.append(DU.getDictWithKey(dictIn,name,False))

        return tuple(listOut)
    def _updateFromDicts(self,toUpgrade,upgrade,replace=None):
        if not replace:
            replace = [False]*len(toUpgrade)
        for dictT,dictU,rep in zip(toUpgrade,upgrade,replace):
            DU.updateDictionary(dictT,dictU, replace=rep)
    ##Method called by sub class to update the parameter_list
    ##@param supclass the super class
    def updateParameters(self):
        unique = {}
        for par in self.__class__.parameter_list:
            if par.attrname in  unique:
                continue
            unique[par.attrname] = par

        self.__class__.parameter_list = tuple(unique.values())
    def extendParameterList(self,sup,sub):
        if self.__class__ == sub:
            self.__class__.parameter_list = self.__class__.parameter_list + sup.parameter_list
        else:
            self.__class__.parameter_list = self.__class__.parameter_list + sub.parameter_list + sup.parameter_list
    ## Call this function after creating the instance to initialize it
    def configure(self):
        """ Public alias to _configureThis"""
        self._configureThis()

    def _configureThis(self):

        #temp hack when the instance does not support the configurability
        #from local files

        if(self.name or self.family or self.normname or self.normfamily):
            #Determine the possible configuration file names.
            #x refers to the files in the install directory where
            #the component is installed.
            #r refers to a directory defined through the
            #environment variable ISCEDB.
            #l refers to the local directory.
            #family refers to the name given to a component in its definition.
            #name refers to an instance name of the component.

            xfamilydb = rfamilydb = lfamilydb = ''
            xnamedb = rnamedb = lnamedb = ''

            import inspect, os
            xpath = os.path.split(inspect.getfile(self.__class__))[0]
            lpath = os.curdir

            #rpath, rafmilydb, and rnamedb are only used if environment
            #variable ISCEDB is defined
            rpath = ''
            try:
                rpath = os.environ['ISCEDB']
            except:
                pass


            #the family name remote and local db filenames
            if self.family:
                familydb = self.family+self.ext
                xfamilydb = os.path.join(xpath, familydb)
                lfamilydb = os.path.join(lpath, familydb)
                if rpath:
                    rfamilydb = os.path.join(rpath, familydb)

            #the instance name remote and local db filenames
            if self.name:
                namedb = self.name+self.ext
                xnamedb = os.path.join(xpath, namedb)
                lnamedb = os.path.join(lpath, namedb)
                if rpath:
                    rnamedb = os.path.join(rpath, namedb)

            #Build the configuration data base list
            #ordered in increasing order of priorities.
            dblist = []

            #Lowest priority: from the install directory
            #family-name db
            if os.path.exists(xfamilydb):
                dblist.append(xfamilydb)

            #instance-name db
            if os.path.exists(xnamedb):
                dblist.append(xnamedb)

            #Second priority: remote ISCEDB directory
            #family-name db
            if os.path.exists(rfamilydb):
                dblist.append(rfamilydb)

            #instance-name db
            if os.path.exists(rnamedb):
                dblist.append(rnamedb)

            #Third priority: current directory
            #family-name db
            if os.path.exists(lfamilydb):
                dblist.append(lfamilydb)

            #instance-name db
            if os.path.exists(lnamedb):
                dblist.append(lnamedb)
            self._parameters()

            propDict,factDict,miscDict, unitsDict,docDict = self._selectFromDicts(dblist)
            propDict = DU.renormalizeKeys(propDict)
            factDict = DU.renormalizeKeys(factDict)
            miscDict = DU.renormalizeKeys(miscDict)
            unitsDict = DU.renormalizeKeys(unitsDict)
            docDict = DU.renormalizeKeys(docDict)            
            self.catalog = DU.renormalizeKeys(self.catalog)
            self._updateFromDicts([self.catalog],[propDict],[True])
            #just to be sure that the facilities, even if default ones,
            #are defined so we can check against the dictionaryOfFacilities
            #to make sure that a property is indeed a property and
            #not a facilities (used in initProperties to validate
            #the property)
            self._facilitiesEmpty()
            self.initProperties(self.catalog)
            self.dictionaryOfVariables = DU.renormalizeKeys(self.dictionaryOfVariables)

            self._init()
            self._facilities()

            self._dictionaryOfFacilities = DU.renormalizeKeys(self._dictionaryOfFacilities)
            self._updateFromDicts([self._dictionaryOfFacilities],[factDict],[True])
            self.init(self.catalog,self._dictionaryOfFacilities,unitsDict,docDict)

            # Run the user's _configure to transfer user-configured facilities to
            # the instance variables
            self._configure()
        return



    def isAskingHelp(self):
        import inspect
        return (os.path.basename(inspect.getfile(self.__class__)) == os.path.basename(sys.argv[0]))
## Constructor

    def __init__(self, family = None, name = None):


        # bool variable set to True if the user wants to ignore warning for key specified in
        # the xml that are not present in the dictionaryOfVariables. Default False
        self._ignoreMissing = False
        #Some parameters might not be defined in the class yet so if it does not exist, ignore it
        self._parametersExceptions = ['metadata_location','delta_latitude','delta_longitude',
                                      'first_latitude','first_longitude','width','length']


        ##
        # bool variable set True by default. If True all warnings are enabled.
        self.warnings = True
        ##
        #
        if not family:
            family = '_family'
        self.family = self.normfamily = self.keyfamily =  family
        #provide a non empty default  otherwise the checkInitialization will complain
        if not name:
            name = family + '_name'
        self.name = self.normname = self.keyname = name
        from iscesys.Parsers.Parser import Parser
        from iscesys.DictUtils.DictUtils import DictUtils
        #####
        #become hard to keep track of the name
        ####
        if self.normfamily:
            self.normfamily = Parser().normalize_comp_name(self.normfamily)
        if self.normname:
            self.normname = Parser().normalize_comp_name(self.normname)
        if self.keyfamily:
            self.keyfamily = DU.renormalizeKey(self.family)
        if self.keyname:
            self.keyname = DU.renormalizeKey(self.name)

        self.ext = '.xml'
        self.logger = None
        self.catalog = {}
        self.descriptionOfVariables = {}
        self.descriptionOfFacilities = {}
        self._dictionaryOfFacilities = {}
        self._cmdLineDict = None

        self.typeOfVariables = {}
        self.unitsOfVariables = {}
        self.dictionaryOfOutputVariables = {}
        if not hasattr(self, 'dictionaryOfVariables'):
            self.dictionaryOfVariables = {}
        self.mandatoryVariables = []
        self.optionalVariables = []

        self.updatePrivate()

        #Add the parameters and facilities to the instance
        self._parameters()
        #First pass add empty facilities
        self._facilitiesEmpty()

        self.initOptionalAndMandatoryLists()
        #add the family and name as parameters so they get registered into the
        #dictionaryOfVariables
        self.family = self.parameter('family',public_name='family',default=self.family,
                                     type=str,mandatory=False,doc='Instance family name')
        self.name = self.parameter('name',public_name='name',default=self.name,
                                       type=str,mandatory=False,doc='Instance name')

        if (("--help" in sys.argv or "-h" in sys.argv) and  self.isAskingHelp()):
            #assume that the factory for which we want to get the help
            # is after the keyword --help or -h
            from iscehelp import Helper
            help = Helper()
            if ("--steps" or "-s") in sys.argv:
                help.askHelp(self, steps=True)
            else:
                help.askHelp(self, steps=False)

#Parameter = Configurable.Parameter

#if __name__ == "__main__":
#    import sys
#    sys.exit(main())



Variable = Configurable.Variable
Parameter = Configurable.Parameter
Facility = Configurable.Facility
