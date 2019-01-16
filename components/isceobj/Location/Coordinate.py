'''
Copyright 2010, by the California Institute of Technology. 
ALL RIGHTS RESERVED. 
United States Government Sponsorship acknowledged. 
Any commercial use must be negotiated with the Office of 
Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By 
accepting this software, the user agrees to comply with all applicable 
U.S. export laws and regulations. User has the responsibility to obtain 
export licenses, or other export authority as may be required before 
exporting such information to foreign countries or providing access 
to foreign persons.
'''

class Coordinate(object):
    """A class to hold peg point information"""
        
    def __init__(self,latitude=None,longitude=None,height=None):
        self._latitude = latitude
        self._longitude = longitude
        self._height = height       
        
    def getLatitude(self):
        return self._latitude

    def getLongitude(self):
        return self._longitude
    
    def getHeight(self):
        return self._height
   
    def setLatitude(self, value):
        self._latitude = value

    def setLongitude(self, value):
        self._longitude = value
        
    def setHeight(self,height):
        self._height = height
        
    def __str__(self):
        retstr = 'Latitude: %s\n'
        retlst = (self._latitude,)
        retstr += 'Longitude: %s\n'
        retlst += (self._longitude,)
        retstr += 'Height: %s'
        retlst += (self._height,)
        return retstr % retlst

    latitude = property(getLatitude, setLatitude)
    longitude = property(getLongitude, setLongitude)
    height = property(getHeight,setHeight)
        
