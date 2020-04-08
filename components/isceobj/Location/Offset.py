#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Walter Szeliga
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




import math
from isce import logging

from isceobj.Util.decorators import type_check, force, pickled, logged
import numpy as np

from iscesys.Component.Component import Component

X = Component.Parameter('x',
        public_name='x',
        default=None,
        type=float,
        mandatory=True,
        doc = 'Range location')

DX = Component.Parameter('dx',
        public_name='dx',
        default=None,
        type=float,
        mandatory=True,
        doc = 'Range offset')

Y = Component.Parameter('y',
        public_name='y',
        default=None,
        type=float,
        mandatory=True,
        doc = 'Azimuth location')

DY = Component.Parameter('dy',
        public_name='dy',
        default=None,
        type=float,
        mandatory=True,
        doc = 'Azimuth offset')

SIGMA_X = Component.Parameter('sigmax',
        public_name='sigmax',
        default=0,
        type=float,
        mandatory=True,
        doc = 'Range covariance')

SIGMA_Y = Component.Parameter('sigmay',
        public_name='sigmay',
        default=0,
        type=float,
        mandatory=True,
        doc = 'Azimuth covariance')

SIGMA_XY = Component.Parameter('sigmaxy',
        public_name='sigmaxy',
        default=0,
        type=float,
        mandatory=True,
        doc = 'Cross covariance')

SNR = Component.Parameter('snr',
        public_name='snr',
        default=0,
        type=float,
        mandatory=True,
        doc = 'Signal to Noise Ratio')

@pickled
class Offset(Component):
    """A class to represent the two-dimensional offset of a particular
    location"""

    logging_name = "isceobj.Location.Offset"

    family = 'offset'
    parameter_list = (
                      X,
                      Y,
                      DX,
                      DY,
                      SIGMA_X,
                      SIGMA_Y,
                      SIGMA_XY,
                      SNR
                     )
    @logged
    def __init__(self, x=None, y=None, dx=None, dy=None, snr=0.0,
                 sigmax=0.0, sigmay=0.0, sigmaxy=0.0,
                 family = None, name = None):
        super(Offset, self).__init__(family=family if family else  self.__class__.family, name=name)
        self.x = x
        self.dx = dx
        self.y = y
        self.dy = dy
        self.setSignalToNoise(snr)
        self.setCovariance(sigmax, sigmay, sigmaxy)
        return None

    def setCoordinate(self, x, y):
        self.x = x
        self.y = y

    def setOffset(self, dx, dy):
        self.dx = dx
        self.dy = dy
        pass
    def setCovariance(self, covx, covy, covxy):
        self.sigmax = covx
        self.sigmay = covy
        self.sigmaxy = covxy

    @force(float)
    def setSignalToNoise(self, snr):
        self.snr = snr if not math.isnan(snr) else 0.0

    def getCoordinate(self):
        return self.x,self.y

    def getOffset(self):
        return self.dx,self.dy

    def getSignalToNoise(self):
        return self.snr

    def getCovariance(self):
        return self.sigmax, self.sigmay, self.sigmaxy


    def __str__(self):
        retstr = "%s %s %s %s %s %s %s %s" % (self.x,self.dx,self.y,self.dy,self.snr, self.sigmax, self.sigmay, self.sigmaxy)
        return retstr


OFFSETS = Component.Parameter('_offsets',
        public_name='offsets',
        default=[],
        type=float,
        mandatory=False,
        intent='output',
        doc = 'List of offsets')

@pickled
class OffsetField(Component):
    """A class to represent a collection of offsets defining an offset field"""
    logging_name = "isceobj.Location.OffsetField"

    family = 'offsetfield'
    parameter_list = (
                      OFFSETS,
                     )
    @logged
    def __init__(self,family=None,name=None):

        super(OffsetField, self).__init__(
            family=family if family else  self.__class__.family, name=name)
        self._last = 0
        self._cpOffsets = None
        return None

    #extend dump method. Convert Offset object to list before dumping it
    def dump(self,filename):

        self.adaptToRender()
        super(OffsetField,self).dump(filename)
        #restore to list of Offset
        self.restoreAfterRendering()

    def load(self,filename):
        import copy

        super(OffsetField,self).load(filename)
        #make a copy
        cpOffsets = copy.deepcopy(self._offsets)
        self.packOffsetswithCovariance(cpOffsets)

    def adaptToRender(self):
        import copy
        #make a copy before dumping
        self._cpOffsets = copy.deepcopy(self._offsets)
        #change the offsets to a list on numbers instead of Offset
        self._offsets = self.unpackOffsetswithCovariance()

    def restoreAfterRendering(self):
        self._offsets = self._cpOffsets

    def initProperties(self,catalog):
        if 'offsets' in catalog:
            offsets = catalog['offsets']
            import numpy as np
            offsets = np.array(offsets)
            self.packOffsetswithCovariance(offsets.T)
            catalog.pop('offsets')
        super().initProperties(catalog)

    def getLocationRanges(self):
        xdxydysnr = self.unpackOffsets()
        numEl = len(xdxydysnr)
        x = np.zeros(numEl)
        y = np.zeros(numEl)
        for i in range(numEl):
            x[i] =  xdxydysnr[i][0]
            y[i] =  xdxydysnr[i][2]
        xr = sorted(x)
        yr = sorted(y)
        return [xr[0],xr[-1],yr[0],yr[-1]]

    def plot(self,type,xmin = None, xmax = None, ymin = None, ymax = None):
        try:
            import numpy as np
            from  scipy.interpolate import griddata
            import matplotlib.pyplot as plt
            from pylab import quiver,quiverkey
        except ImportError:
            self.logger.error('This method requires scipy, numpy and matplotlib to be installed.')
        xdxydysnr = self.unpackOffsets()
        numEl = len(xdxydysnr)
        x = np.zeros(numEl)
        y = np.zeros(numEl)
        dx = np.zeros(numEl)
        dy = np.zeros(numEl)
        for i in range(numEl):
            x[i] =  xdxydysnr[i][0]
            dx[i] =  xdxydysnr[i][1]
            y[i] =  xdxydysnr[i][2]
            dy[i] =  xdxydysnr[i][3]
        if xmin is None: xmin = np.min(x)
        if xmax is None: xmax = np.max(x)
        if ymin is None: ymin = np.min(y)
        if ymax is None: ymax = np.max(y)
        legendL = np.floor(max(np.max(dx),np.max(dy)))
        #normally the width in range is much smaller that the length in azimuth, so normalize so that we have the same number os sample for each axis
        step = min(np.min(int(np.ceil(((ymax - ymin)/(xmax - xmin))))),5)
        X , Y = np.mgrid[xmin:xmax,ymin:ymax:step]
        skip = int(np.ceil(xmax - xmin)/100)*5
        if type == 'field':
            U = griddata(np.array([x,y]).T,dx, (X,Y), method='linear')
            V = griddata(np.array([x,y]).T,dy, (X,Y), method='linear')
            Q = quiver(X[::skip,::skip], Y[::skip,::skip],
                       U[::skip,::skip], V[::skip,::skip],
                       pivot='mid', color='g')
            arrow = str(legendL) + ' pixles'
            qk = quiverkey(Q, 0.8, 0.95, legendL, arrow,
                           labelpos='E',
                           coordinates='figure',
                           fontproperties={'weight':'bold'})
            ax = Q.axes
            ax.set_xlabel('range location')
            ax.set_ylabel('azimuth location')
        elif(type == 'pcolor'):
            M = griddata(np.array([x,y]).T,
                         np.sqrt(dx**2 + dy**2),
                         (X,Y),
                         method='linear')
            P = griddata(np.array([x,y]).T,
                         np.arctan2(dy, dx),
                         (X,Y)
                         ,method='linear')
            plt.subplot(2, 1, 1)
            plt.imshow(M,aspect='auto', extent=[xmin, xmax, ymin, ymax])
            plt.colorbar()
            ax1 = plt.gca()
            ax1.set_ylabel('azimuth location')
            ax1.set_title('offset magnitude')
            plt.subplot(2, 1, 2)
            plt.imshow(P, aspect='auto', extent=[xmin,xmax,ymin,ymax])
            plt.colorbar()
            ax2 = plt.gca()
            ax2.set_xlabel('range location')
            ax2.set_ylabel('azimuth location')
            ax2.set_title('offset phase')
        plt.show()
        return plt

    @type_check(Offset)
    def addOffset(self, offset):
        self._offsets.append(offset)
        pass

    def __next__(self):
        if self._last < len(self._offsets):
            next = self._offsets[self._last]
            self._last += 1
            return next
        else:
            self._last = 0 # This is so that we can restart iteration
            raise StopIteration()

    def packOffsets(self, offsets):#create an offset field from a list of offets
        self._offset = []
        for i in range(len(offsets[0])):
            #note that different ordering (x,y,dx,dy,snr) instead of (x,dx,y,dy,snr)
            self.addOffset(
                Offset(x=offsets[0][i],
                       y=offsets[2][i],
                       dx=offsets[1][i],
                       dy=offsets[3][i],
                       snr=offsets[4][i])
                )

    def packOffsetswithCovariance(self, offsets):
        self._offset = []
        for i in range(len(offsets[0])):
            self.addOffset(
                    Offset(x=offsets[0][i],
                           y=offsets[2][i],
                           dx=offsets[1][i],
                           dy=offsets[3][i],
                           snr=offsets[4][i],
                           sigmax=offsets[5][i],
                           sigmay=offsets[6][i],
                           sigmaxy=offsets[7][i])
                    )

    def unpackOffsets(self):
        """A convenience method for converting our iterator into a flat
        list for use in Fortran and C code"""
        offsetArray = []
        for offset in self.offsets:
            x, y = offset.getCoordinate()
            dx, dy = offset.getOffset()
            snr = offset.getSignalToNoise()
            offsetArray.append([x,dx,y,dy,snr])
            pass
        return offsetArray

    def unpackOffsetswithCovariance(self):
        offsetArray = []
        for offset in self.offsets:
            x,y = offset.getCoordinate()
            dx,dy = offset.getOffset()
            snr = offset.getSignalToNoise()
            sx, sy, sxy = offset.getCovariance()
            offsetArray.append([x,dx,y,dy,snr,sx,sy,sxy])
            pass
        return offsetArray

    def cull(self, snr=0.0):
        """Cull outliers based on their signal-to-noise ratio.

        @param snr: the signal-to-noise ratio to use in the culling.  Values with greater signal-to-noise will be kept.
        """
        culledOffsetField = OffsetField()
        i = 0
        for offset in self.offsets:
            if (offset.getSignalToNoise() < snr):
                i += 1
            else:
                culledOffsetField.addOffset(offset)

        self.logger.info("%s offsets culled" % (i))
        return culledOffsetField

    def __iter__(self):
        return self

    def __str__(self):
        return '\n'.join(map(str, self.offsets))+'\n' #2013-06-03 Kosal: wrong use of map

    @property
    def offsets(self):
        return self._offsets

    pass

    def getFitPolynomials(self,rangeOrder=2,azimuthOrder=2,maxOrder=True, usenumpy=False):
        from stdproc.stdproc.offsetpoly.Offsetpoly import Offsetpoly
        from isceobj.Util import Poly2D

        numCoeff = 0
        ####Try and use Howard's polynomial fit code whenever possible
        if (rangeOrder == azimuthOrder) and (rangeOrder <= 3):
            if (rangeOrder == 1):
                if maxOrder:
                    numCoeff = 3
                else:
                    numCoeff = 4
            elif (rangeOrder == 2):
                if maxOrder:
                    numCoeff = 6
            elif (rangeOrder == 3):
                if maxOrder:
                    numCoeff = 10


        inArr = np.array(self.unpackOffsets())
        azmin = np.min(inArr[:,2])
        inArr[:,2] -= azmin

        ####Use Howard's code
        if (numCoeff != 0) and not usenumpy:
            x = list(inArr[:,0])
            y = list(inArr[:,2])
            dx = list(inArr[:,1])
            dy = list(inArr[:,3])
            sig = list(inArr[:,4])

            ####Range Offsets
            obj = Offsetpoly()
            obj.setLocationAcross(x)
            obj.setLocationDown(y)
            obj.setSNR(sig)
            obj.setOffset(dx)
            obj.numberFitCoefficients = numCoeff
            obj.offsetpoly()

            val = obj.offsetPoly

            #####Unpack into 2D array
            if numCoeff == 3:
                coeffs = [[val[0], val[1]],
                          [val[2], 0.0]]

            elif numCoeff == 4:
                coeffs = [[val[0], val[1]],
                          [val[2], val[3]]]

            elif numCoeff == 6:
                coeffs = [[val[0], val[1], val[4]],
                          [val[2], val[3], 0.0],
                          [val[5], 0.0, 0.0]]

            elif numCoeff == 10:
                ####Unpacking needs to be checked.
                coeffs = [[val[0], val[1], val[4], val[8]],
                          [val[2], val[3], val[6], 0.0],
                          [val[5], val[7],0.0, 0.0],
                          [val[9], 0.0, 0.0, 0.0]]


            rangePoly = Poly2D.Poly2D()
            rangePoly.setMeanAzimuth(azmin)
            rangePoly.initPoly(rangeOrder=rangeOrder, azimuthOrder=azimuthOrder, coeffs=coeffs)

            ####Azimuth Offsets
            obj.setOffset(dy)
            obj.offsetpoly()
            val = obj.offsetPoly

            #####Unpack into 2D array
            if numCoeff == 3:
                coeffs = [[val[0], val[1]],
                          [val[2], 0.0]]

            elif numCoeff == 4:
                coeffs = [[val[0], val[1]],
                          [val[2], val[3]]]

            elif numCoeff == 6:
                coeffs = [[val[0], val[1], val[4]],
                          [val[2], val[3], 0.0],
                          [val[5], 0.0, 0.0]]

            elif numCoeff == 10:
                ####Unpacking needs to be checked.
                coeffs = [[val[0], val[1], val[4], val[8]],
                          [val[2], val[3], val[6], 0.0],
                          [val[5], val[7],0.0, 0.0],
                          [val[9], 0.0, 0.0, 0.0]]

            azimuthPoly = Poly2D.Poly2D()
            azimuthPoly.setMeanAzimuth(azmin)
            azimuthPoly.initPoly(rangeOrder=rangeOrder, azimuthOrder=azimuthOrder, coeffs=coeffs)

        ####Fallback to numpy based polynomial fitting
        else:

            x = inArr[:,0]
            y = inArr[:,2]
            dx = inArr[:,1]
            dy = inArr[:,3]
            sig = inArr[:,4]


            azimuthPoly = Poly2D.Poly2D()
            azimuthPoly.initPoly(rangeOrder=rangeOrder, azimuthOrder=azimuthOrder)
            azimuthPoly.polyfit(x,y,dy, sig=sig)
            azimuthPoly._meanAzimuth += azmin

            rangePoly = Poly2D.Poly2D()
            rangePoly.initPoly(rangeOrder=rangeOrder, azimuthOrder=azimuthOrder)
            rangePoly.polyfit(x,y,dx,sig=sig)
            rangePoly._meanAzimuth += azmin

        return (azimuthPoly, rangePoly)
