#!/usr/bin/env python

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Piyush Agram
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



from iscesys.Component.Configurable import Configurable


class Polynomial(Configurable):
    '''
    Class to store 1D polynomials in ISCE.
    Implented as a list of coefficients:

    [    1,     x^1,     x^2, ...., x^n]

    The size of the 1D list will correspond to 
    [order+1].
    '''
    family = 'polynomial'
    def __init__(self, family='', name=''):
        '''
        Constructor for the polynomial object.
        '''
        self._coeffs = []       
        self._accessor = None
        self._factory = None
        self._poly = None
        self._width = 0
        self._length = 0
        super(Polynomial,self).__init__(family if family else  self.__class__.family, name)
        
        
        return
    def initPoly(self,image = None):
        
        if(image):
            self._width = image.width
            self._length = image.length
            
    def setCoeffs(self, parms):
        '''
        Set the coefficients using another nested list.
        '''
        raise NotImplementedError("Subclasses should implement setCoeffs!")

    def getCoeffs(self):
        return self._coeffs


    def setImage(self, width):
        self._width = image.width
        self._length = image.length

   
    def exportToC(self):
        '''
        Use the extension module and return a pointer in C.
        '''
        raise NotImplementedError("Subclasses should implement exportToC!")


    def importFromC(self, pointer, clean=True):
        pass

    def copy(self):
        pass
        
    def setWidth(self, var):
        self._width = int(var)
        return

    @property
    def width(self):
        return self._width

    def setLength(self, var):
        self._length = int(var)
        return

    @property
    def length(self):
        return self._length

    def getPointer(self):
        return self._accessor
