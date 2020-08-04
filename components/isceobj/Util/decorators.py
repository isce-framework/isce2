

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
# Author: Eric Belz
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




## \namespace isceobj.Util.decorators Utility decorators
"""This module has generic decorators, such as:

float_setter         (makes sure a setter calls it's args __float__)
type_check           (a type checker for methods)
object_wrapper       (for making methods that interface to a C for Fortran library)
dov                  (deal with dictionaryOfVariables in a DRY way)
profiler             (a method profile-- TBD).
etc
"""
from functools import wraps

## force a setter to use a type
def force(type_):
    def enforcer(setter):
        @wraps(setter)
        def checked_setter(self, value):
            try:
                checked = type_(value)
            except (ValueError, TypeError, AttributeError):
                message = "Expecting a %s, but got: %s, in setter: %s, %s"%(
                    str(type_),
                    str(value),
                    self.__class__.__name__,
                    setter.__name__)
                if hasattr(self, "logger"):
                    self.logger.warn(message)
                else:
                    raise ValueError(message)
                pass
            return setter(self, checked)
        return checked_setter
    return enforcer

## for floating point
float_setter = force(float)

## decorator to wrap a 1-argument **methods** and check if the argument is an instance
# of "cls"
def type_check(cls):
    """decorator=type_check(cls)

    "cls" is a class that the decorated method's sole argument must be an instance of (or
    a TypeError is raised-- a string explaining the problem is included"
    
    "decorator" is the decorator with which to decorate the method-- the decorator
    protocol is that a decorator with arguments returns a decorator.

    USAGE WARNING: CANNOT DECORATE FUNCTIONS or STATICMETHODS (yet).
    """
    ## The checker knows "cls", and takes the method to make, an return, checked_method
    def checker(method):
        ## The interpretor installs this method in method's place--
        ## it checks and raises a TypeEror if needed
        def checked_method(*args):
            obj = args[-1]
            if not isinstance(obj, cls):
                raise TypeError(
                    method.__name__+
                    " excpected: "+
                    cls.__name__ +
                    ", got:" +
                    str(obj.__class__)
                    )
            return method(*args)
        return checked_method
    return checker




## If self.method(*args) return self.object.method(*args), then decorate method with:
## @object_wrapper("object")
def object_wrapper(objname):
    """If self.method(*args) return self.object.method(*args), then decorate method with
    @object_wrapper("object") and make sure that the method looks like this:

    @object_wrapprt("object")
    def method(self, x1, ..., xn.):
        return object.method

    where object.method(x1,..., xn) is object.method's signature. See LineAccesorPy.py
    for a concrete example.
    """

    ## functools.wraps prevents the decorator from overiding the method's:
    ## __name__, __doc__ attributes. 
    def accessor(func):
        """This is a method decorator. The bare method returns "func", while the
        decorated method ("new_method") calls it, with self.<objname> as the implicit 1st
        argument. """
        @wraps(func)
        def new_method(self, *args):
            return func(getattr(self, objname), *args)
        return new_method
    return accessor


## This decorator decorates __init__ methods for classes that need their mandatory and
## optional variables computed -- it may removed when Parameters replace variable tuples.
def dov(init):
    """Usage:

        dictionaryOfVariables = {....}

        @dov
        def __init__(self,...):


    Decorates __init__ so that it takes a STATIC dictionary of variables and computes
    dynamic mandatoryVariables and optionalVariables dictionies. Obviously easy to
    rewrite to take a dynamic dictionaryOfVariables.

    Nevertheless, it should be a class decorator that only handles static variables.
    That's TBD.
    """
    def constructor(self, *args, **kwargs):
        self.descriptionOfVariables = {}
        self.mandatoryVariables = []
        self.optionalVariables = []
        typePos = 2
        for key , val in self.dictionaryOfVariables.items():
            value = val[typePos]
            if value is True or value == 'mandatory':
                self.mandatoryVariables.append(key)
            elif value is False or value == 'optional':
                self.optionalVariables.append(key)
            else:
                raise ValueError(
                    'Error. Variable can only be "optional"/False or "mandatory"/True'
                    )
            pass
        return init(self, *args, **kwargs)
    return constructor




## Decorator to add logging to a class's __init__.
def logged(init):
    """Usage:
    -------------------------------------------------
   | class CLS(object):                             |
   |                                                |
   |      logging_name = 'isce.cls' # or whatever   |
   |                                                |
   |     @logged                                    |
   |     def __init__(self, ...)                    |
   -------------------------------------------------
   This decorator adds a logger names "logging_name"
   to any instance of CLS.
   """
    
    import logging
    def constructor(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.logger = logging.getLogger(self.__class__.logging_name)
        return None
    return constructor


## The no-pickle list, may be moved to a /library.
DONT_PICKLE_ME = ('logger', '_inputPorts', '_outputPorts')
## Decorator to add pickling to a class without using inheritance
def pickled(cls):
    """Usage:
    -------------------------------------------------
   | @pickled                                       |
   | class CLS(object):                             |
   |                                                |
   |      logging_name = 'isce.cls' # or whatever   |
   |                                                |
   -------------------------------------------------
   This decorator adds pickling to class CLS.
   By default, it also invokes @logged on the CLS.__init__,
   so you need to 
   """

    ## reject objects bases on name
    def __getstate__(self):
        d = dict(self.__dict__)
        ## for future use: modify no pickle list.
        skip = (
            () if not hasattr(self.__class__, 'dont_pickle_me') else
            self.__class__.dont_pickle_me
            )
        for key in DONT_PICKLE_ME+skip:
            if key in d:
                del d[key]
                pass
            pass
        return d
    
    def __setstate__(self,d):
        self.__dict__.update(d)
        import logging
        self.logger = logging.getLogger(self.__class__.logging_name)
        pass

    if not hasattr(cls, '__setstate__'): cls.__setstate__ = __setstate__
    if not hasattr(cls, '__getstate__'): cls.__getstate__ = __getstate__

    return cls


## A decorator for making a port out of a trivial method named add<port>
def port(check):
    """port(check) makes a decorator.
    
    if "check" is a str [type] it enforces:
    hasattr(port, check) [isintanace(port, check)].

    The decorated method should be as follows, for port "spam"

    @port("eggs")
    def addspam(self):
        pass
    
    That will setup:
    
    self.spam from self.inputPorts['spam'] and ensure:
    self.spam.eggs exists.

    Of course, the method canbe notrivial, too.
    """
    def port_decorator(method):
        port_name = method.__name__[3:].lower()
        attr = port_name
        @wraps(method)
        def port_method(self):
            local_object = self.inputPorts[port_name]
            setattr(self, attr, local_object)
            if check is not None:
                if isinstance(check, str):
                    if not hasattr(local_object, check):
                        raise AttributeError(check+" failed")
                    pass
                else:
                    if not isinstance(local_object, check):
                        raise TypeError(str(check)+" failed")
                    pass
                pass
            return method(self) # *args, **kwargs is TBD.
        return port_method
    return port_decorator

##Provide a decorator for those methods that need to use the old api.
##at one point one might just turn it off by simply returning the original function 
def use_api(func):
    from iscesys.ImageApi.DataAccessorPy import DataAccessor
    def use_api_decorator(*args,**kwargs):
        #turn on the use of the old image api
        if DataAccessor._accessorType == 'api':
            leave = True
        else:
            DataAccessor._accessorType = 'api'
            leave = False
        ret = func(*args,**kwargs)
        #turn off. The default will be used, i.e. api for write and gdal for read
        if not leave:
            DataAccessor._accessorType = ''
        return ret
    return use_api_decorator
        
        
