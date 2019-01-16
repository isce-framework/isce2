#
# Author: Piyush Agram
# Copyright 2016
#

from .TopsProc import *
from .Factories import *

def getFactoriesInfo():
    return  {'TopsProc':
                     {'args':
                           {
                            'procDoc':{'value':None,'type':'Catalog','optional':True}
                            },
                     'factory':'createTopsProc'                     
                     }
              
              }

def createTopsProc(name=None, procDoc= None):
    from .TopsProc import TopsProc
    return TopsProc(name = name,procDoc = procDoc)
