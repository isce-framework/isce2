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




from iscesys.ImageApi import DataAccessor as DA
from isceobj.Util.Polynomial import Polynomial
from iscesys.Component.Component import Component

ERROR_CHECK_FINALIZE = False
WIDTH = Component.Parameter('_width',
    public_name='width',
    default = 0,
    type=float,
    mandatory=False,
    doc="Width of the image associated with the polynomial"
)
LEGNTH = Component.Parameter('_length',
    public_name='length',
    default = 0,
    type=float,
    mandatory=False,
    doc="Length of the image associated with the polynomial"
)
ORDER = Component.Parameter('_order',
    public_name='order',
    default = None,
    type=int,
    mandatory=False,
    doc="Polynomial order"
)

NORM = Component.Parameter('_norm',
    public_name='norm',
    default = 1.,
    type=float,
    mandatory=False,
    doc=""
)
MEAN = Component.Parameter('_mean',
    public_name='mean',
    default = 0.,
    type=float,
    mandatory=False,
    doc=""
)
COEFFS = Component.Parameter('_coeffs',
    public_name='coeffs',
    default = [],
    container=list,
    type=float,
    mandatory=False,
    doc=""
)
class Poly1D(Polynomial):
    '''
    Class to store 1D polynomials in ISCE.
    Implented as a list of coefficients:

    [    1,     x^1,     x^2, ...., x^n]

    The size of the 1D list will correspond to
    [order+1].
    '''
    family = 'poly1d'
    parameter_list = (WIDTH,
                      LEGNTH,
                      ORDER,
                      NORM,
                      MEAN,
                      COEFFS)

    def __init__(self, family='', name='', order=None, image=None,direction = 'x'):
        '''
        Constructor for the polynomial object. The base class Polynomial set width and length
        if image not None.
        direction 'x' or 'y'. 'x' the line width = image.width otherwise line width = image.length
        Basically x is for range doppler and y for azimuth doppler
        '''
        #at the moment all poly work with doubles
        self._dataSize = 8
        super(Poly1D,self).__init__(family if family else  self.__class__.family, name)

    def initPoly(self, order=None, coeffs=None, image=None,direction = 'x'):
        super(Poly1D,self).initPoly(image)

        if(direction == 'y'):#swap direction
            tmp = self._width
            self._width = self._length
            self._length = tmp
        if coeffs:
            import copy
            self._coeffs  = copy.deepcopy(coeffs)
        self._order = int(order) if order else order

        if (self._coeffs is not None) and (len(self._coeffs)>0):
            self.createPoly1D()

        return

    def dump(self,filename):
        from copy import deepcopy
        toDump = deepcopy(self)
        self._poly = None
        self._accessor= None
        self._factory = None
        super(Poly1D,self).dump(filename)
        #tried to do self = deepcopy(toDump) but did not work
        self._poly = toDump._poly
        self._accessor = toDump._accessor
        self._factory = toDump._factory

    def load(self,filename):
        super(Poly1D,self).load(filename)
        #recreate the pointer objcts _poly, _accessor, _factory
        self.createPoly1D()

    def setCoeffs(self, parms):
        '''
        Set the coefficients using another nested list.
        '''
        self._coeffs = [0. for j in parms]
        for ii,row in enumerate(parms):
            self._coeffs[ii] = float(row)

        return

    def getCoeffs(self):
        return self._coeffs

    def setNorm(self, parm):
        self._norm = float(parm)

    def setMean(self, parm):
        self._mean = float(parm)

    def getNorm(self):
        return self._norm

    def getMean(self):
        return self._mean

    def getWidth(self):
        return self._width

    def getLength(self):
        return self._length

    def __call__(self, rng):
        '''
        Evaluate the polynomial.
        This is much slower than the C implementation - only for sparse usage.
        '''
        x = (rng - self._mean)/self._norm
        res = 0.
        for ii,row in enumerate(self._coeffs):
            res += row * (x**ii)

        return res

    def exportToC(self):
        '''
        Use the extension module and return a pointer in C.
        '''
        from isceobj.Util import combinedlibmodule as CL

        g = CL.exportPoly1DToC(self._order, self._mean, self._norm, self._coeffs)

        return g

    def importFromC(self, pointer, clean=True):
        '''
        Uses information from the  extension module structure to create Python object.
        '''
        from isceobj.Util import combinedlibmodule as CL

        order,mean,norm,coeffs = CL.importPoly1DFromC(pointer)
        self._order = order
        self._mean = mean
        self._norm = norm
        self._coeffs = coeffs.copy()

        if clean:
            CL.freeCPoly1D(pointer)
        pass

    def copy(self):
        '''
        Create a copy of the given polynomial instance.
        Do not carry any associated image information.
        Just the coefficients etc for scaling and manipulation.
        '''

        newObj = Poly1D()
        g = self.exportToC()
        newObj.importFromC(g)
        return newObj

    def createPoly1D(self):
        if self._accessor is None:
            self._poly = self.exportToC()
            self._accessor, self._factory = DA.createPolyAccessor(self._poly,"poly1d",
                                                              self._width,self._length,self._dataSize)
        else:
            print('C pointer already created. Finalize and recreate if image dimensions changed.')

    def finalize(self):
        from isceobj.Util import combinedlibmodule as CL
        CL.freeCPoly1D(self._poly)
        try:
            DA.finalizeAccessor(self._accessor, self._factory)
        except TypeError:
            message = "Poly1D %s is already finalized" % str(self)
            if ERROR_CHECK_FINALIZE:
               raise RuntimeError(message)
            else:
                print(message)

        self._accessor = None
        self._factory = None
        return None
    def polyfit(self, xin, yin, sig=None,cond=1.0e-12):
        '''
        Fit a 1D polynomial.
        x = np.arange(5,85)
        y = 1.23 + 4.5*x + 0.03*x*x
        g = Poly1D(order=2)
        g.polyfit(x,y)

        print(g(5), g(8), g(11))
        '''

        import numpy as np

        x = np.array(xin)
        y = np.array(yin)
        Npts = x.size

        ####Scale inputs
        xmin = np.min(xin)
        xnorm = np.max(xin) - xmin

        if xnorm == 0:
            xnorm = 1.0

        x = (x-xmin)/xnorm

        A = np.ones((Npts, self._order + 1))

        for poww in range(1,self._order+1):
            A[:,poww] = np.power(x, poww)

        if sig is not None:
            snr = 1.0 + 1.0/np.array(sig)
            A = A /snr[:,None]
            y = y/snr

        val, res, rank, eigs = np.linalg.lstsq(A,y,rcond=cond)
        if len(res) > 0:
            print('Chi squared: %f'%(np.sqrt(res/(1.0*Npts))))

        self.setCoeffs(val)
        self.setMean(xmin)
        self.setNorm(xnorm)
