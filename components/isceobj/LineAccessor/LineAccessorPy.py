#!/usr/bin/env python3 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                                  Giangi Sacco
#                        NASA Jet Propulsion Laboratory
#                      California Institute of Technology
#                        (C) 2009  All Rights Reserved
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


from __future__ import print_function
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from isceobj.LineAccessor import LineAccessor
from isceobj.Util.decorators import object_wrapper
# translation between BandSchemeType and integer: BNULL = 0, BSQ = 1, BIP = 2, BIL = 3

## This Class provides a set of convinient methods to initialize or use some of the LineAccessor.ccp methods.
# @see LineAccessor.cpp
##
 


## Make a local decorator from the generic one.
accessor = object_wrapper("LineAccessorObj")

class LineAccessorPy(object):
    
    def __init__(self, lazy=True):
        self.LineAccessorObj = None if lazy else self.createLineAccessorObject()
        return None
    
    def createLineAccessorObject(self):
        self.LineAccessorObj = LineAccessor.getLineAccessorObject()
        return None

    def getLineAccessorPointer(self):
        return self.LineAccessorObj

    @accessor
    def initLineAccessor(self, filename, filemode, endian, type, row, col):
        return LineAccessor.initLineAccessor
        
    @accessor
    def createFile(self, length):
        return LineAccessor.createFile

    @accessor
    def rewindImage(self):
        return LineAccessor.rewindImage

    @accessor
    def getMachineEndianness(self):
        return LineAccessor.getMachineEndianness
        
    @accessor
    def finalizeLineAccessor(self):
        return LineAccessor.finalizeLineAccessor
           
    @accessor
    def changeBandScheme(self, filein, fileout, type, width, numBands, bandIn, bandOut):
        return LineAccessor.changeBandScheme
        
    @accessor
    def convertFileEndianness(self, filein, fileout, type):
        return LineAccessor.convertFileEndianness
    
    @accessor
    def getTypeSize(self, type):
        return LineAccessor.getTypeSize

    @accessor
    def getFileLength(self):
        return LineAccessor.getFileLength
    
    @accessor
    def getFileWidth(self):
        return LineAccessor.getFileWidth
    
    @accessor
    def printObjectInfo(self):
        return LineAccessor.printObjectInfo
    
    @accessor
    def printAvailableDataTypesAndSizes(self):
        return LineAccessor.printAvailableDataTypesAndSizes

    pass

