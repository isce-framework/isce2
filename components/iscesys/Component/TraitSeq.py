

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
# Author: Ravi Lanka
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




from collections.abc import MutableSequence
from iscesys.Component.Component import Component
import numpy as N
import re

# Factor or Parameter
FACTORY = Component.Parameter(
    '_factory',
    public_name='_factory',
    default=None,
    type=bool,
    mandatory=False,
    doc='Flag - Factory/Parameter'
)

# Factory Related
FACTORY_NAME = Component.Parameter(
    '_factory_name',
    public_name='_factorname',
    default=None,
    type=str,
    mandatory=False,
    doc='Factory Name used in the Trait Sequence'
)

MODULE_NAME = Component.Parameter(
    '_module_name',
    public_name='_modulename',
    default=None,
    type=str,
    mandatory=False,
    doc='Module name used in Trait Sequence'
)

# Parameter Related
CONTAINER = Component.Parameter(
    '_container',
    public_name='_container',
    default=None,
    type=str,
    mandatory=False,
    doc='Container Name of the Factory used in the Trait Sequence'
)

TYPE = Component.Parameter(
    '_intent',
    public_name='_intent',
    default=None,
    type=str,
    mandatory=False,
    doc='intent of the parameter used in the Trait Sequence'
)

TYPE = Component.Parameter(
    '_type',
    public_name='_type',
    default=None,
    type=str,
    mandatory=False,
    doc='Type of the parameter used in the Trait Sequence'
)

# Common Parameters
MANDATORY = Component.Parameter(
    '_mandatory',
    public_name='_mandatory',
    default=False,
    type=bool,
    mandatory=False,
    doc='Mandatory Field of the module used in Trait Sequence'
)

PRIVATE = Component.Parameter(
    '_private',
    public_name='_private',
    default=True,
    type=bool,
    mandatory=False,
    doc='Private Field of the module used in Trait Sequence'
)

NAME = Component.Parameter(
    '_name',
    public_name='NAME',
    default=[],
    container=list,
    type=str,
    mandatory=False,
    doc='Holds the sequence of names'
)

class TraitSeq(Component, MutableSequence):
    family = 'TraitSeq'
    parameter_list = (FACTORY,
                      FACTORY_NAME,
                      MODULE_NAME,
                      CONTAINER,
                      TYPE,
                      MANDATORY,
                      PRIVATE,
                      NAME)
    facility_list = ()

    def __init__(self, name = ''):
        super().__init__(family=self.__class__.family, name=name if name else self.__class__.family)
        self.configure()
        self.list          = list()
        self.objid         = list()
        self.facility_list = ()
        return

    def _instantiate_(self, obj):
        from iscesys.Component.Configurable import Configurable
        self._factory = isinstance(obj, Configurable)
        if self._factory:
            # Flag for element
            self._factory       = True

            # Parse module name and factory
            module_name, factory_name = TraitSeq._extractTraits_(obj)

            # Setting Factory to default
            self._factory_name  = 'default'
            self._module_name   = module_name
        else:
            # Parameter
            raise Exception("Yet to be supported")
            self._factory       = False
            self._container     = obj.container
            self._intent        = obj.intent
            self.type           = obj.type

        return

    def set_aux(self, obj):
        if self._factory is None:
          # Called for the first time to set
          # objects of the class
          self._instantiate_(obj)

        if self._factory is True:
          self._createFacility_(obj)
        else:
          self._createParameter_(obj)
        return

    def _createParameter_(self, obj):
        """
        Creates Parameter class object and updates Dictionary
        """
        objn = self.__getName__(obj.name)
        self.objid.append(id(obj))
        self._name.append(objn)
        self.parameter_list += (objn,)
        self.dictionaryOfVariables[objn] = {
              'attrname' : objn,
              'container': self._container,
              'type'     : self._type,
              'intent'   : self._intent}
        setattr(self, objn, obj)

    def _updateDict_(self, objn):
        self._dictionaryOfFacilities[objn] = {
             'attrname'     : objn,
             'public_name'  : objn,
             'factorymodule': self._module_name,
             'factoryname'  : self._factory_name,
             'mandatory'    : self._mandatory,
             'private'      : self._private,
             'args'         : (),
             'kwargs'       : None,
             'doc'          : ''}
        self.dictionaryOfVariables[objn] = {
             'attrname' : objn,
             'type'     : 'component',
             'mandatory': self._mandatory,
             'private'  : self._private}
        return

    def _createFacility_(self, obj):
        """
        Creates Facility class object and updates dictionary
        """
        objn = self.__getName__(obj.name)
        self.objid.append(id(obj))
        self._name.append(objn)
        self.facility_list += (objn,)
        self._updateDict_(objn)
        setattr(self, objn, obj)
        return

    def updateDict(self, obj, i, objn):
        print(objn)
        self.objid[i] = id(obj)
        self._name[i] = objn

        self._updateDict_(objn)

        # Handle facility list differently as it is a tuple
        cFacility = list(self.facility_list)
        cFacility = objn
        self.facility_list = tuple(cFacility)
        return

    def _copyFacility(self):
        """
        Fixes the Variable of Variables to contain Facilities
        """
        facility_list = list(self._dictionaryOfFacilities.keys())
        variable_list = list(self.dictionaryOfVariables.keys())
        for name in facility_list:
          if name not in variable_list:
            self.dictionaryOfVariables[name] = {
                'attrname' : name,
                'type'     : 'component',
                'mandatory': self._mandatory,
                'private'  : self._private}

        return

    def __getName__(self, name, _next_=0):
        if name.lower() != 'traitseq_name':
            objn = name.lower()
        else:
            objn = '{}{}'.format(self.name, len(self.list) + _next_)
        objn = '{}{}'.format(self.name, len(self.list) + _next_)
        return objn

    @staticmethod
    def _extractTraits_(obj):
        # Parse module name and factory
        module = re.findall("'([^']*)'", str(type(obj)))[0]
        module_name = module.split('.')[-1]
        factory_name = '.'.join(module.split('.')[:-1])
        return (module_name, factory_name)

    def _checkTrait_(self, obj):
        '''
        Checks if the element added is of the same type
        as in the list
        '''
        #Set the ith element of self.list to value object
        if self._factory is not None:
          # Already the first element is added to the list
          if self._factory:
            module_name, factory_name = TraitSeq._extractTraits_(obj)
            if (self._module_name != module_name):
              raise Exception("""Incorrect object type added \
                                 TraitSeq currently supports only objects of single type""")
          else:
            raise Exception('Not Yet supported')

    ###################
    # fixes on basic methods because Configurability used properties to fetch
    # some details on about facilities
    ###################

    def renderToDictionary(self,obj,propDict,factDict,miscDict):
        '''
        Overloading rendering to preprocess before writting
        '''
        self._copyFacility()
        super(Component, self).renderToDictionary(obj,propDict,factDict,miscDict)
        return

    def initRecursive(self,dictProp,dictFact):
        '''
        Fixing Properties dictionary before initializing
        '''
        self._copyFacility()
        super(Component, self).initRecursive(dictProp,dictFact)

        try:
          # Fixing object ID and the list
          if len(self._name) != len(self.objid):
            self.objid = []
            self.list = []
            for name in self._name:
              obj = getattr(self, name.lower())
              cid = id(obj)
              self.objid.append(cid)
              self.list.append(obj)
        except:
          # Elements not initialized from xml
          pass


    ##################
    # List Methods
    ##################

    def __add__(self, other):
        #Add lists contained in other TraitSeq object
        if self._checkEQ_(other):
          for i in range(len(other)):
            self.append(other.list[i])
        else:
          raise Exception("""Object are of different types
                             TraitSeq currently supports only objects of a single type""")

        return self

    def __contains__(self, x):
        #Check if x is contained in self.list
        return x in self.list

    def __delitem__(self, i, flag=True):
        #Delete item at index i from self.list
        #Update the Component dictionaries and facility_list
        if flag:
          del self.list[i]
        del self.dictionaryOfVariables[self._name[i]]
        del self._dictionaryOfFacilities[self._name[i]]
        del self._name[i]
        del self.objid[i]

        # Handle facility list differently as it is a tuple
        cFacility = list(self.facility_list)
        del cFacility[i]
        self.facility_list = tuple(cFacility)

        return

    def __getitem__(self, i):
        #Return the item in self.list at index i
        return self.list[i]

    def __len__(self):
        #Return the length of self.list
        return len(self.list)

    def __str__(self):
        #Return a string version of self.list
        return str(self.list)

    def __setitem__(self, i, obj):
        self._checkTrait_(obj)
        self.list[i] = obj
        name = self.__getName__(obj.name, _next_=1)
        setattr(self, name, obj)
        self.objid    = id(obj)
        if self._name[i] != name:

          # Update Facility List
          cFacility = list(self.facility_list)
          cFacility[i] = name
          self.facility_list = tuple(cFacility)

          # Remove old
          del self.dictionaryOfVariables[self._name[i]]
          del self._dictionaryOfFacilities[self._name[i]]

          self._updateDict_(name)
          self._name[i] = name

        return

    def append(self, obj):
        #Append an element to self.list
        self._checkTrait_(obj)
        self.list.append(obj)
        self.set_aux(obj)

    def clear(self):
        #Clear all items from self.list
        self.list.clear()
        self.dictionaryOfVariables.clear()
        self._dictionaryOfFacilities.clear()
        self._name.clear()
        self.objid.clear()

        # Handle facility list differently as it is a tuple
        self.facility_list = ()
        return

    def copy(self):
        #Return a copy of self.list
        return self.copy()

    def count(self, x):
        #return count of how many times x occurs in self.list
        return self.list.count(x)

    def extend(self, other):
        #Extend self.list with other list
        raise Exception('Not Yet supported')
        self.list.extend(other)

    def index(self, x):
        #return the index of x in self.list;
        return self.list.index(x)

    def insert(self, i, v):
        self._checkTrait_(v)
        self.list.insert(i, v)
        objn = self.__getName__(v.name)
        setattr(self, objn, v)
        self._updateDict_(objn)

        # Update Facility List
        self._name.insert(i, objn)
        self.objid.insert(i, id(v))
        cFacility = list(self.facility_list)
        cFacility.insert(i, objn)
        self.facility_list = tuple(cFacility)

        return

    def pop(self, i=None):
        #pop item off the specified index if given, else off the end of list
        self.__delitem__(i if i else len(self)-1)
        return

    def remove(self, x):
        #remove item x from the list
        self.list.remove(x)
        flag = False

        # Update bookmark list
        cidx = [id(x) for x in self.list]
        setdiff = [obj for obj in self.objid + cidx if obj not in cidx]
        if (len(setdiff) == 1):
          self.__delitem__(self.objid.index(setdiff[0]), flag)
        else:
          raise Exception('Not Yet supported')

        return

    def reverse(self):
        #reverse the items in the list
        self.list.reverse()
        self.facility_list = self.facility_list[::-1]
        self._name.reverse()
        self.objid.reverse()
        return

    @staticmethod
    def _orderSeq_(x, idx):
        if len(x) != len(idx):
          raise Exception('Index of different length')

        x = N.array(x)
        return list(x[N.array(idx, dtype=int)])

    def sort(self, key=None):
        #Sort self.list according to the ordering relations (lt, gt, eq) of
        #the type of elements in self.list.
        self.list.sort(key=key)

        # Find the order to update dictionary
        pid = N.array(self.objid)
        cid = N.empty((len(self.list)))
        for i, obj in enumerate(self.list):
          cid[i] = N.where(pid == id(obj))[0][0]

        # Update internal list for proper sequencing
        self._name          = self._orderSeq_(self._name, cid)
        self.objid          = self._orderSeq_(self.objid, cid)
        self.facility_list  = tuple(self._orderSeq_(self.facility_list, cid))
        return

    def __eq__(self, other):
        return self.list == other.list

    def _checkEQ_(self, other):
        if self._factory:
          return ((self._module_name, self._factory_name, self._mandatory, self._private) ==
                    (other._module_name, other._factory_name, other._mandatory, other._private))
        else:
          return ((self._container, self._type, self._intent) == \
                    (other._container, other._type, other._intent))
