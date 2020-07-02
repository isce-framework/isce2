#!/usr/bin/env python3

def createDataRetriever(name=''):
    from .DataRetriever import DataRetriever
    return DataRetriever(name=name)
        
def getFactoriesInfo():
    """
    Returns a dictionary with information on how to create an object Sensor from its factory
    """
    return  {'DataRetriever':
                    {          
                     'factory':'createDataRetriever'
                     }
              }
