#!/usr/bin/env python3
from .Datetime import datetimeType

#local traits
mytraits = {'datetimeType':datetimeType
           }

#traits dictionary filled with builitin traits first and
#then local traits

#traits dictionary defines conversions from string to type
traits = {}

#load the builtin traits
import builtins
for k in builtins.__dict__.keys():
    try:
        traits[k] = builtins.__dict__[k]
    except:
        #there is some strangeness in some of the entries in the __builtins__
        pass

#local traits
traits.update(mytraits)
