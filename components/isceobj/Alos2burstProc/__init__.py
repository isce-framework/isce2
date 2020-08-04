#
# Author: Piyush Agram
# Copyright 2016
#

from .Alos2burstProc import *
from .Factories import *

def getFactoriesInfo():
    return  {'Alos2burstProc':
                     {'args':
                           {
                            'procDoc':{'value':None,'type':'Catalog','optional':True}
                            },
                     'factory':'createAlos2burstProc'                     
                     }
              
              }

def createAlos2burstProc(name=None, procDoc= None):
    from .Alos2burstProc import Alos2burstProc
    return Alos2burstProc(name = name,procDoc = procDoc)
