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
LENGTH = Component.Parameter('_length',
    public_name='length',
    default = 0,
    type=float,
    mandatory=False,
    doc="Length of the image associated with the polynomial"
)
RANGE_ORDER = Component.Parameter('_rangeOrder',
    public_name='rangeOrder',
    default = None,
    type=int,
    mandatory=False,
    doc="Polynomial order in the range direction"
)
AZIMUTH_ORDER = Component.Parameter('_azimuthOrder',
    public_name='azimuthOrder',
    default = None,
    type=int,
    mandatory=False,
    doc="Polynomial order in the azimuth direction"
)
NORM_RANGE = Component.Parameter('_normRange',
    public_name='normRange',
    default = 1.,
    type=float,
    mandatory=False,
    doc=""
)
MEAN_RANGE = Component.Parameter('_meanRange',
    public_name='meanRange',
    default = 0.,
    type=float,
    mandatory=False,
    doc=""
)
NORM_AZIMUTH = Component.Parameter('_normAzimuth',
    public_name='normAzimuth',
    default = 1.,
    type=float,
    mandatory=False,
    doc=""
)
MEAN_AZIMUTH = Component.Parameter('_meanAzimuth',
    public_name='meanAzimuth',
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
class Poly2D(Polynomial):
    '''
    Class to store 2D polynomials in ISCE.
    Implented as a list of lists, the coefficients
    are stored as shown below:

    [ [    1,     x^1,     x^2, ....],
      [  y^1, x^1 y^1, x^2 y^1, ....],
      [  y^2, x^1 y^2, x^2 y^2, ....],
      [    :        :        :     :]]

    where "x" corresponds to pixel index in range and
    "y" corresponds to pixel index in azimuth.

    The size of the 2D matrix will correspond to
    [rangeOrder+1, azimuthOrder+1].
    '''
    family = 'poly2d'
    parameter_list = (WIDTH,
                      LENGTH,
                      RANGE_ORDER,
                      AZIMUTH_ORDER,
                      NORM_RANGE,
                      MEAN_RANGE,
                      NORM_AZIMUTH,
                      MEAN_AZIMUTH,
                      COEFFS)

    def __init__(self,  family='', name=''):
        '''
        Constructor for the polynomial object. . The base class Polynomial set width and length
        if image not None
        '''
        #at the moment all poly work with doubles
        self._dataSize = 8
        super(Poly2D,self).__init__(family if family else  self.__class__.family, name)
        self._instanceInit()

        return

    def initPoly(self,rangeOrder=None, azimuthOrder=None,coeffs=None, image=None):
        super(Poly2D,self).initPoly(image)

        if coeffs:
            import copy
            self._coeffs  = copy.deepcopy(coeffs)

        self._rangeOrder = int(rangeOrder) if rangeOrder else rangeOrder
        self._azimuthOrder = int(azimuthOrder) if azimuthOrder else azimuthOrder
        if (self._coeffs is not None) and  (len(self._coeffs) > 0):
            self.createPoly2D()

    def dump(self,filename):
        from copy import deepcopy
        toDump = deepcopy(self)
        self._poly = None
        self._accessor= None
        self._factory = None
        super(Poly2D,self).dump(filename)
        #tried to do self = deepcopy(toDump) but did not work
        self._poly = toDump._poly
        self._accessor = toDump._accessor
        self._factory = toDump._factory

    def load(self,filename):
        super(Poly2D,self).load(filename)
        #recreate the pointer objcts _poly, _accessor, _factory
        self.createPoly2D()

    def setCoeff(self, row, col, val):
        """
        Set the coefficient at specified row, column.
        """
        self._coeffs[row][col] = val
        return

    def setCoeffs(self, parms):
        '''
        Set the coefficients using another nested list.
        '''
        self._coeffs = [[0. for i in j] for j in parms]
        for ii,row in enumerate(parms):
            for jj,col in enumerate(row):
                self._coeffs[ii][jj] = float(col)

        return

    def getCoeffs(self):
        return self._coeffs

    def setNormRange(self, parm):
        self._normRange = float(parm)

    def setMeanRange(self, parm):
        self._meanRange = float(parm)

    def getNormRange(self):
        return self._normRange

    def getMeanRange(self):
        return self._meanRange

    def setNormAzimuth(self, parm):
        self._normAzimuth = float(parm)

    def setMeanAzimuth(self, parm):
        self._meanAzimuth = float(parm)

    def getNormAzimuth(self):
        return self._normAzimuth

    def getMeanAzimuth(self):
        return self._meanAzimuth

    def getRangeOrder(self):
        return self._rangeOrder

    def getAzimuthOrder(self):
        return self._azimuthOrder

    def getWidth(self):
        return self._width

    def getLength(self):
        return self._length

    def __call__(self, azi,rng):
        '''
        Evaluate the polynomial.
        This is much slower than the C implementation - only for sparse usage.
        '''
        y = (azi - self._meanAzimuth)/self._normAzimuth
        x = (rng - self._meanRange)/self._normRange
        res = 0.
        for ii,row in enumerate(self._coeffs):
            yfact = y**ii
            for jj,col in enumerate(row):
                res += self._coeffs[ii][jj] * yfact * (x**jj)

        return res

    def copy(self):
        '''
        Create a copy of the given polynomial instance.
        Do not carry any associated image information.
        Just the coefficients etc for scaling and manipulation.
        '''

        newObj = Poly2D()
        g = self.exportToC()
        newObj.importFromC(g)
        return newObj

    def exportToC(self):
        '''
        Use the extension module and return a pointer in C.
        '''
        from isceobj.Util import combinedlibmodule as CL
        order = [self._azimuthOrder, self._rangeOrder]
        means = [self._meanAzimuth, self._meanRange]
        norms = [self._normAzimuth, self._normRange]
        ptr = CL.exportPoly2DToC(order, means, norms, self._coeffs)
        return ptr

    def importFromC(self, pointer, clean=True):
        '''
        Uses information from the  extension module structure to create Python object.
        '''
        from isceobj.Util import combinedlibmodule as CL
        orders, means, norms, coeffs = CL.importPoly2DFromC(pointer)
        self._azimuthOrder, self._rangeOrder = orders
        self._meanAzimuth, self._meanRange = means
        self._normAzimuth, self._normRange = norms
        self._coeffs = []

        for ii in range(self._azimuthOrder+1):
            ind = ii * (self._rangeOrder+1)
            self._coeffs.append(coeffs[ind:ind+self._rangeOrder+1])

        if clean:
            CL.freeCPoly2D(pointer)

        return


    def createPoly2D(self):
        if self._accessor is None:
            self._poly = self.exportToC()
            self._accessor, self._factory = DA.createPolyAccessor(self._poly,"poly2d",
                                                              self._width,self._length,self._dataSize)
        else:
            print('C pointer already created. Finalize and recreate if image dimensions changed.')

    def finalize(self):
        from isceobj.Util import combinedlibmodule as CL
        CL.freeCPoly2D(self._poly)
        try:
            DA.finalizeAccessor(self._accessor, self._factory)
        except TypeError:
            message = "Poly2D %s is already finalized" % str(self)
            if ERROR_CHECK_FINALIZE:
                raise RuntimeError(message)
            else:
                print(message)

        self._accessor = None
        self._factory = None
        return None

    def polyfit(self,xin,yin,zin,
            sig=None,snr=None,cond=1.0e-12,
            maxOrder=True):
        '''
        2D polynomial fitting.

xx = np.random.random(75)*100
yy = np.random.random(75)*200

z = 3000 + 1.0*xx + 0.2*xx*xx + 0.459*yy + 0.13 * xx* yy + 0.6*yy*yy

gg = Poly2D(rangeOrder=2, azimuthOrder=2)
gg.polyfit(xx,yy,z,maxOrder=True)

print(xx[5], yy[5], z[5], gg(yy[5], xx[5]))
print(xx[23], yy[23], z[23], gg(yy[23], xx[23]))
        '''
        import numpy as np

        x = np.array(xin)
        xmin = np.min(x)
        xnorm = np.max(x) - xmin
        if xnorm == 0:
            xnorm = 1.0

        x = (x - xmin)/ xnorm

        y=np.array(yin)
        ymin = np.min(y)
        ynorm = np.max(y) - ymin
        if ynorm == 0:
            ynorm = 1.0

        y = (y-ymin)/ynorm

        z = np.array(zin)
        bigOrder = max(self._azimuthOrder, self._rangeOrder)

        arrList = []
        for ii in range(self._azimuthOrder + 1):
            yfact = np.power(y, ii)
            for jj in range(self._rangeOrder+1):
                xfact = np.power(x,jj) * yfact

                if maxOrder:
                    if ((ii+jj) <= bigOrder):
                        arrList.append(xfact.reshape((x.size,1)))
                else:
                    arrList.append(xfact.reshape((x.size,1)))

        A = np.hstack(arrList)

        if sig is not None and snr is not None:
            raise Exception('Only one of sig / snr can be provided')

        if sig is not None:
            snr = 1.0 + 1.0/sig

        if snr is not None:
            A = A / snr[:,None]
            z = z / snr



        returnVal = True

        val, res, rank, eigs = np.linalg.lstsq(A,z, rcond=cond)
        if len(res)> 0:
            print('Chi squared: %f'%(np.sqrt(res/(1.0*len(z)))))
        else:
            print('No chi squared value....')
            print('Try reducing rank of polynomial.')
            returnVal = False

        self.setMeanRange(xmin)
        self.setMeanAzimuth(ymin)
        self.setNormRange(xnorm)
        self.setNormAzimuth(ynorm)

        coeffs = []
        count = 0
        for ii in range(self._azimuthOrder+1):
            row = []
            for jj in range(self._rangeOrder+1):
                if maxOrder:
                    if (ii+jj) <= bigOrder:
                        row.append(val[count])
                        count = count+1
                    else:
                        row.append(0.0)
                else:
                    row.append(val[count])
                    count = count+1
            coeffs.append(row)

        self.setCoeffs(coeffs)
        
        return returnVal
    
def createPolynomial(order=None,
        norm=None, offset=None):
    '''
    Create a polynomial with given parameters.
    Order, Norm and Offset are iterables.
    '''

    poly = Poly2D(rangeOrder=order[0], azimuthOrder=order[1])

    if norm:
        poly.setNormRange(norm[0])
        poly.setNormAzimuth(norm[1])

    if offset:
        poly.setMeanRange(offset[0])
        poly.setMeanAzimuth(offset[1])

    return poly

def createRangePolynomial(order=None, offset=None, norm=None):
    '''
    Create a polynomial in range.
    '''
    poly = Poly2D(rangeOrder=order, azimuthOrder=0)

    if offset:
        poly.setMeanRange(offset)

    if norm:
        poly.setNormRange(norm)

    return poly

def createAzimuthPolynomial(order=None, offset=None, norm=None):
    '''
    Create a polynomial in azimuth.
    '''
    poly = Poly2D(rangeOrder=0, azimuthOrder=order)

    if offset:
        poly.setMeanAzimuth(offset)

    if norm:
        poly.setNormAzimuth(norm)

    return poly
