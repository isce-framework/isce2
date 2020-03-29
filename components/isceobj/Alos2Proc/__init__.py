#
# Author: Piyush Agram
# Copyright 2016
#

from .Alos2Proc import *
from .Factories import *

def getFactoriesInfo():
    return  {'Alos2Proc':
                     {'args':
                           {
                            'procDoc':{'value':None,'type':'Catalog','optional':True}
                            },
                     'factory':'createAlos2Proc'                     
                     }
              
              }

def createAlos2Proc(name=None, procDoc= None):
    from .Alos2Proc import Alos2Proc
    return Alos2Proc(name = name,procDoc = procDoc)
