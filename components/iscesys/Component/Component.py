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
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import logging
import types
from iscesys.Component.Configurable import Configurable
from iscesys.StdOEL.StdOELPy import _WriterInterface
from isceobj.Util.decorators import type_check


## A function transformation to convert func(self, *args) into func(self)(*args)
def curried(func):
    """curried(func) takes function with signature:
    
    func(self, *args)

    and makes it:

    curried_func(*args) --> func(self, *args)

    That is, it makes the self implicit
    """
    def curried_func(self, *args):
        return func(self, *args)
    curried_func.__doc__ = """Curried version of:\n%s""" % func.__doc__
    return curried_func

## A function transformation to convert func(self, *args) into func(self)(*args)
def delayed(method):
    """delayed(method) takes method with signature:
    
    func(arg, *args)

    and makes a new methodwith signature:

    delayed_func(arg)(*args)

    That is, it delays the evaluation of the method until the returned
    function is called.
    """
    from functools import partial
    def delayed_func(self, attr):
        return partial(method, self, attr)
    delayed_func.__doc__ =  (
        """Delayed execution (via call) version of:\n%s""" %
        method.__doc__
        )
    return delayed_func


## A mixin for ANY object that need to have attribute access done by metacode.
class StepHelper(object):
    """This Mixin helps get subclasses attributes evalauted at a later time--
    during the steps process.
    """
    @staticmethod
    def compose(f, g, fargs=(), gargs=(), fkwargs={}, gkwargs={}):
        """fog = compose(f, g [fargs=(), gargs=(), fkwargs={}, gkwargs={}])

        f        g        callable objects
        fargs,   gargs    the callable's fixed arguments
        fkwargs  gkwargs  the callable's fixed keywords

        fog:              a callable object that will signatur:

        fog(*args, **kwargs) that will evaluate

        f(g(*(args+gargs), **(kwargs+gkwargs)), *fargs, **fkwargs)
        """
        from functools import partial
        def fog(*args, **kwargs):
            return  (
                partial(f, *fargs, **fkwargs)(
                    partial(g, *gargs, **gkwargs)(
                        *args, **kwargs
                         )
                    )
                )
        return fog
    
    ## self.attrgetter(attr) --> getattr(self, attr)
    attrgetter = curried(getattr)
    ## self.attrsetter(attr) --> setattr(self, attr, value)
    attrsetter = curried(setattr)
    ## self.delayed_attrgetter(attr) --> partial(self.attrgetter, attr)
    delayed_attrgetter = delayed(attrgetter)
    ## self.delayed_attrsetter(attr) --> partial(self.attrsetter, attr)
    delayed_attrsetter = delayed(attrsetter)

    ## Return a function with no arguments that will copy attrf to attr i,
    ## when called.
    def delayed_attrcopy_from_to(self, attri, attrf):
        """f = self.delayed_attrcopy_from_to(attri, attrif)

        attri     inital attribute name
        attrf     final (target) attribute name

        f         a function of 0 arguements that will do:
        
        self.attrf = self.attri

        when called. hasattr(self, attr..) need only be True
        when f is called.
        """
        return lambda : self.attrsetter(attrf, self.attrgetter(attri))

    pass


## Decorator (with arguments) to run a Port method with flow keyword set
def flow(flow):
    """Decorator:
    decorator = flow(flow)

    The decorator then transforms a method:

    method = decorator(method)

    so that the "flow" kwarg is set the argument to the decorator.
    A nonsense value of "flow" will raise and Exception in
    Componenet.__select_flow
    """
    from functools import wraps
    ## flow(flow) returns this (that's the python decorator protocal)
    def decorator(method):
        ## The decorator returns the method with the flow keyword set;
        ## @wraps makes the docstring and __name__
        @wraps(method)
        def method_with_flow(self, *args, **kwargs):
            kwargs["flow"] = flow
            return method(self)(*args, **kwargs)
        return method_with_flow
    return decorator


class Component(Configurable, StepHelper, _WriterInterface):
    

    def __init__(self, family=None, name=None):
        super(Component, self).__init__(family, name)
        self._inputPorts =  InputPorts()
        self._outputPorts = OutputPorts()
        self.createPorts()
        self.createLogger()
        return None

    ## This is how you call a component:
    ## args are passed to the method
    ## kwargs are ports.
    def __call__(self, *args, **kwargs):
        for key, value in kwargs.items():
            self.wireInputPort(name=key, object=value)
        return getattr(self, self.__class__.__name__.lower())(*args)

    ## Default pickle behavior
    def __getstate__(self):
        d = dict(self.__dict__)
        for key in ('logger', '_inputPorts', '_outputPorts'):
            try:
                del d[key]
            except KeyError:
                pass
        return d

    ## Default unpickle behavior
    def __setstate__(self, d):
        from iscesys.Component.Component import InputPorts, OutputPorts
        self.__dict__.update(d)
        self.createLogger()
        self._inputPorts = InputPorts()
        self._outputPorts = OutputPorts()
        self.createPorts()
        return None

    ## Place holder for portless components.
    def createPorts(self):
        pass

    ## Moving all logging to here, you must have a logging_name to get logged.
    ## this is not optimal-- and indicates a logging decorator is the
    ## appropriate thing to do.
    def createLogger(self):
        try:
            name = self.__class__.logging_name
            self.logger = logging.getLogger(name)
        except AttributeError:
            pass

        pass

    @property
    def inputPorts(self):
        return self._inputPorts
    @inputPorts.setter
    def inputPorts(self, value):
        return setattr(self._inputPorts, value)
    
    @property
    def outputPorts(self):
        return self._outputPorts
    @outputPorts.setter
    def outputPorts(self, value):
        return setattr(self._outputPorts, value)



    ## Private helper method (not for humans): Get correct port
    ## (input or output)
    def __select_flow(self, flow):
        """private method get "input" or "output" port depending on keyword
        'flow'."""
        try:
            attr = "_" + flow + "Ports"
            result = getattr(self, attr)
        except AttributeError as err:
            ## On exception: figure out what went wrong and explain.
            allowed_values = [cls.flow for cls in (InputPorts, OutputPorts)]
            if flow not in allowed_values:
                raise ValueError(
                    "flow keyword (%s) must be allowed values:%s" %
                    (str(flow), str(allowed_values))
                    )
            raise err
        return result

    ## private helper method for WIRING, flow keyword uses __selecet_flow()
    def _wirePort(self, name, object, flow):
        port_iterator = self.__select_flow(flow)
        if name in port_iterator:
            port = port_iterator.getPort(name=name)
            port.setObject(object)
        else:
            raise PortError("No %s port named %s" % (port_iterator.flow, name))
        return None  
    
    ## private helper method for LIST, flow keyword uses __selecet_flow()
    def _listPorts(self, flow):
        for port in self.__select_flow(flow):
            print(flow + "Port Name:" + port.getName())
            pass
        return None
    
    ## private help to get ports, flow keyword uses __selecet_flow()
    def _getPort(self, name=None, flow=None):
        return self.__select_flow(flow).getPort(name)

    ## helper method to activate a port, flow keyword uses __selecet_flow()
    def _activePort(self, flow):
        for port in self.__select_flow(flow):
            port()
            pass
        return None

    ## wire InputPorts using a flow() decorated _wirePort()
    @flow("input")
    def wireInputPort(self, name=None, object=None):
        """wireInputPort([name=None [, object=None]])
        _inputPorts.getPort(name).setObject(object)
        """
        return self._wirePort
    
    ## wire OutputPorts using a flow() decorated _wirePort()
    @flow("output")
    def wireOutputPort(self, name=None, object=None):
        """wireOutputPort([name=None [, object=None]])
        _outputPorts.getPort(name).setObject(object)
        """
        return self._wirePort

    ## Since wiring a port is getting a string and object-- that a dictionary item,
    ## this does it via a dictionary of {name:object} pairs, and provides a string
    ## free interface to wiring ports
    def wire_input_ports(**kwargs):
        """wire_input_port(**kwargs)  kwargs={name: object, ...}"""
        for key, value in kwargs.items():
            self.wireInputPort(name=key, object=value)
        return self

    ## list InputPorts using a flow() decorated _listPort()
    @flow("input")
    def listInputPorts(self):
        """prints items in a list of _.inputPorts """
        return self._listPorts

    ## list OutputPorts using a flow() decorated _listPort()
    @flow("output")
    def listOutputPorts(self):
        """prints items in a list of _.outputPorts """
        return self._listPorts
    
    ## get an InputPort with flow() decorated _getPort()
    @flow("input")
    def getInputPort(self, name=None):
        """getInputPort([name=None]) -->

        _inputPorts.getPort(name)
        """
        return self._getPort
    ## get an OutputPort with flow() decorated _getPort()
    @flow("output")
    def getOutputPort(self, name=None):
        """getOutputPort([name=None]) -->

        _outputPorts.getPort(name)
        """
        return self._getPort

    ## get an InputPort with flow() decorated _activePort()
    @flow("input")
    def activateInputPorts(self):
        """call each port in _inputPorts"""
        return self._activePort

    ## get an OutputPort with flow() decorated _activePort()
    @flow("output")
    def activateOutputPorts(self):
        """call each port in _outputPorts"""
        return self._activePort
 
    pass

class Port(object):
    
    def __init__(self, name, method=None, doc=""):
        self.name = name   # Name with which to reference the port
        self._method = method # Function which implements the port
        self._object = None
        self.doc = doc # A documentation string for the port
        
    @type_check(str)
    def setName(self, name):
        self._name = name
        
    def getName(self):
        return self._name
    
#    @type_check(new.instancemethod)
    def setMethod(self, method=None):
        self._method = method
    
    def getMethod(self):
        return self._method
    
    def setObject(self, object=None):
        self._object = object
        
    def getObject(self):
        return self._object
    
    def __str__(self):
        return str(self._doc)

    def __call__(self):
        return self._method()

    @property
    def doc(self):
        return str(self._doc)
    
    @doc.setter
    @type_check(str)
    def doc(self, value):
        self._doc = value
        



    name = property(getName, setName)
    object = property(getObject, setObject)
    method = property(getMethod, setMethod)

    
    pass

class PortIterator(object):
    """PortIterator() uses:
    add() method to add ports. Note: it is also
    a port container (__contains__) and mapping (__getitem__)
    """
    def __init__(self):
        self._last = 0
        self._ports = []
        self._names = []
        
    def add(self, port):
        """add(port)
        appends port to _ports
        appends port.getName() to _names"""
        if isinstance(port, Port):
            self._ports.append(port)
            self._names.append(port.getName())
            pass
        return None

    def getPort(self, name=None):
        try:
            result = self._ports[self._names.index(name)]
        except IndexError:
            result =  None
        return result
  
    def hasPort(self, name=None):
        return name in self._names

    ## Make PortIterator a container: name in port_iterator
    def __contains__(self, name):
        """name in port --> port.hasPort(name)"""
        return self.hasPort(name)

    ## Make PortIterator a mapping (port_iterator[name] --> port)
    def __getitem__(self, name):
        """port[name] --> port.getPort(name).getObject()"""
        return self.getPort(name).getObject()

    ## port_iterator[name]=method --> port.add(Port(name=name, method=method)
    def __setitem__(self, name, method):
        return self.add(Port(name=name, method=method))

    ## iter(port_iterator()) returns an iterator over port_iterator._list
    def __iter__(self):
        return iter(self._ports)
    
    ## Len(PortIterator) is the len(PortIterator._ports)
    def __len__(self):
        return len(self._ports)

    def next(self):
        if(self._last < len(self._ports)):
            next_ = self._ports[self._last]
            self._last += 1
            return next_
        else:
            self._last = 0 # This is so that we can restart iteration
            raise StopIteration()
        

    pass


class InputPorts(PortIterator):
    ## flow tells Component's generic methods that this in for input
    flow="input"
    pass

                                    
class OutputPorts(PortIterator):
    ## flow tells Component's generic methods that this in for output
    flow="output"
    pass


class PortError(Exception):
    """Raised when an invalid port operation is attempted"""
    def __init__(self, value):
        self.value = value
        return None

    def __str__(self):
        return repr(self.value)
    

