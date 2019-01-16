#!/usr/bin/env python3

def createUnwrapComp(name=''):
    from .unwrapComponents import unwrapComponents
    instance = unwrapComponents(name=name)
    return instance

def getFactoriesInfo():
    return  {'UnwrapComp':
                     {
                     'factory':'createUnwrapComp'                     
                     }
              }
