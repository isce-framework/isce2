#
# Author: Joshua Cohen
# Copyright 2016
#

from libc.stdint cimport uint64_t
from libcpp cimport bool
from libcpp.vector cimport vector

cdef extern from "Poly2d.h":
    cdef cppclass Poly2d:
        int rangeOrder
        int azimuthOrder
        double rangeMean
        double azimuthMean
        double rangeNorm
        double azimuthNorm
        vector[double] coeffs

        Poly2d() except +
        Poly2d(int,int,double,double,double,double) except +
        Poly2d(const Poly2d&) except +
        void setCoeff(int,int,double)
        void getCoeff(int,int)
        double eval(double,double)
        void printPoly()

cdef class PyPoly2d:
    cdef Poly2d *c_poly2d
    cdef bool owner

    def __cinit__(self, int azimuthOrder=-1, int rangeOrder=-1, double azimuthMean=0., double rangeMean=0., double azimuthNorm=1., double rangeNorm=1.):
        self.c_poly2d = new Poly2d(rangeOrder, azimuthOrder, rangeMean, azimuthMean, rangeNorm, azimuthNorm)
        self.owner = True
    def __dealloc__(self):
        if (self.owner):
            del self.c_poly2d

    @property
    def azimuthOrder(self):
        return self.c_poly2d.azimuthOrder
    @azimuthOrder.setter
    def azimuthOrder(self, int a):
        if (a < 0):
            return
        else:
            c = self.coeffs
            for i in range((a-self.azimuthOrder)*(self.rangeOrder+1)):
                c.append(0.)
            nc = []
            for i in range((a+1)*(self.rangeOrder+1)):
                nc.append(c[i])
            self.c_poly2d.azimuthOrder = a
            self.c_poly2d.coeffs.resize((self.azimuthOrder+1)*(self.rangeOrder+1))
            self.coeffs = nc
    @property
    def rangeOrder(self):
        return self.c_poly2d.rangeOrder
    @rangeOrder.setter
    def rangeOrder(self, int a):
        if (a < 0):
            return
        else:
            c = self.coeffs
            nc = []
            # Cleanest is to first form 2D array of coeffs from 1D
            for i in range(self.azimuthOrder+1):
                ncs = []
                for j in range(self.rangeOrder+1):
                    ncs.append(c[i*(self.rangeOrder+1)+j])
                nc.append(ncs)
            # nc is now the 2D reshape of coeffs
            for i in range(self.azimuthOrder+1): # Go row-by-row...
                for j in range(a-self.rangeOrder): # Add 0s to each row (if
                    nc[i].append(0.)               # a > self.rangeOrder)
            self.c_poly2d.rangeOrder = a
            self.c_poly2d.coeffs.resize((self.azimuthOrder+1)*(self.rangeOrder+1))
            c = []
            for i in range(self.azimuthOrder+1):
                for j in range(self.rangeOrder+1):
                    c.append(nc[i][j])
            self.coeffs = c
    @property
    def azimuthMean(self):
        return self.c_poly2d.azimuthMean
    @azimuthMean.setter
    def azimuthMean(self, double a):
        self.c_poly2d.azimuthMean = a
    @property
    def rangeMean(self):
        return self.c_poly2d.rangeMean
    @rangeMean.setter
    def rangeMean(self, double a):
        self.c_poly2d.rangeMean = a
    @property
    def azimuthNorm(self):
        return self.c_poly2d.azimuthNorm
    @azimuthNorm.setter
    def azimuthNorm(self, double a):
        self.c_poly2d.azimuthNorm = a
    @property
    def rangeNorm(self):
        return self.c_poly2d.rangeNorm
    @rangeNorm.setter
    def rangeNorm(self, double a):
        self.c_poly2d.rangeNorm = a
    @property
    def coeffs(self):
        a = []
        for i in range((self.azimuthOrder+1)*(self.rangeOrder+1)):
            a.append(self.c_poly2d.coeffs[i])
        return a
    @coeffs.setter
    def coeffs(self, a):
        if ((self.azimuthOrder+1)*(self.rangeOrder+1) != len(a)):
            print("Error: Invalid input size (expected 1D list of length "+str(self.azimuthOrder+1)+"*"+str(self.rangeOrder+1)+")")
            return
        for i in range((self.azimuthOrder+1)*(self.rangeOrder+1)):
            self.c_poly2d.coeffs[i] = a[i]
    def dPrint(self):
        self.printPoly()
    @staticmethod
    cdef boundTo(Poly2d *poly):
        cdef PyPoly2d newpoly = PyPoly2d()
        del newpoly.c_poly2d
        newpoly.c_poly2d = poly
        newpoly.owner = False
        return newpoly

    def setCoeff(self, int a, int b, double c):
        self.c_poly2d.setCoeff(a,b,c)
    def getCoeff(self, int a, int b):
        return self.c_poly2d.getCoeff(a,b)
    def eval(self, double a, double b):
        return self.c_poly2d.eval(a,b)
    def printPoly(self):
        self.c_poly2d.printPoly()

cdef extern from "ResampSlc.h":
    cdef cppclass ResampSlc:
        uint64_t slcInAccessor, slcOutAccessor, residRgAccessor, residAzAccessor
        double wvl, slr, r0, refwvl, refslr, refr0
        int outWidth, outLength, inWidth, inLength
        bool isComplex, flatten, usr_enable_gpu
        Poly2d *rgCarrier
        Poly2d *azCarrier
        Poly2d *rgOffsetsPoly
        Poly2d *azOffsetsPoly
        Poly2d *dopplerPoly

        ResampSlc() except +
        ResampSlc(const ResampSlc&) except +
        void setRgCarrier(Poly2d*)
        void setAzCarrier(Poly2d*)
        void setRgOffsets(Poly2d*)
        void setAzOffsets(Poly2d*)
        void setDoppler(Poly2d*)
        Poly2d* releaseRgCarrier()
        Poly2d* releaseAzCarrier()
        Poly2d* releaseRgOffsets()
        Poly2d* releaseAzOffsets()
        Poly2d* releaseDoppler()
        void clearPolys()
        void resetPolys()
        void resamp()


cdef class PyResampSlc:
    cdef ResampSlc *c_resamp

    def __cinit__(self):
        self.c_resamp = new ResampSlc()
    #def __dealloc__(self):
    #    del self.c_resamp

    @property
    def slcInAccessor(self):
        return self.c_resamp.slcInAccessor
    @slcInAccessor.setter
    def slcInAccessor(self, uint64_t a):
        self.c_resamp.slcInAccessor = a
    @property
    def slcOutAccessor(self):
        return self.c_resamp.slcOutAccessor
    @slcOutAccessor.setter
    def slcOutAccessor(self, uint64_t a):
        self.c_resamp.slcOutAccessor = a
    @property
    def residRgAccessor(self):
        return self.c_resamp.residRgAccessor
    @residRgAccessor.setter
    def residRgAccessor(self, uint64_t a):
        self.c_resamp.residRgAccessor = a
    @property
    def residAzAccessor(self):
        return self.c_resamp.residAzAccessor
    @residAzAccessor.setter
    def residAzAccessor(self, uint64_t a):
        self.c_resamp.residAzAccessor = a
    @property
    def wvl(self):
        return self.c_resamp.wvl
    @wvl.setter
    def wvl(self, double a):
        self.c_resamp.wvl = a
    @property
    def slr(self):
        return self.c_resamp.slr
    @slr.setter
    def slr(self, double a):
        self.c_resamp.slr = a
    @property
    def r0(self):
        return self.c_resamp.r0
    @r0.setter
    def r0(self, double a):
        self.c_resamp.r0 = a
    @property
    def refwvl(self):
        return self.c_resamp.refwvl
    @refwvl.setter
    def refwvl(self, double a):
        self.c_resamp.refwvl = a
    @property
    def refslr(self):
        return self.c_resamp.refslr
    @refslr.setter
    def refslr(self, double a):
        self.c_resamp.refslr = a
    @property
    def refr0(self):
        return self.c_resamp.refr0
    @refr0.setter
    def refr0(self, double a):
        self.c_resamp.refr0 = a
    @property
    def outWidth(self):
        return self.c_resamp.outWidth
    @outWidth.setter
    def outWidth(self, int a):
        self.c_resamp.outWidth = a
    @property
    def outLength(self):
        return self.c_resamp.outLength
    @outLength.setter
    def outLength(self, int a):
        self.c_resamp.outLength = a
    @property
    def inWidth(self):
        return self.c_resamp.inWidth
    @inWidth.setter
    def inWidth(self, int a):
        self.c_resamp.inWidth = a
    @property
    def inLength(self):
        return self.c_resamp.inLength
    @inLength.setter
    def inLength(self, int a):
        self.c_resamp.inLength = a
    @property
    def isComplex(self):
        return self.c_resamp.isComplex
    @isComplex.setter
    def isComplex(self, bool a):
        self.c_resamp.isComplex = a
    @property
    def flatten(self):
        return self.c_resamp.flatten
    @flatten.setter
    def flatten(self, bool a):
        self.c_resamp.flatten = a
    @property
    def usr_enable_gpu(self):
        return self.c_resamp.usr_enable_gpu
    @usr_enable_gpu.setter
    def usr_enable_gpu(self, bool a):
        self.c_resamp.usr_enable_gpu = a
    # Note: The property accessors here return a PyPoly2d object that is
    #       "bound" to the ResampSlc's Poly2d object. That means when the
    #       returned PyPoly2d object goes out of scope, it will not try
    #       to delete the contained Poly2d object.
    @property
    def rgCarrier(self):
        return PyPoly2d.boundTo(self.c_resamp.rgCarrier)
    @rgCarrier.setter
    def rgCarrier(self, PyPoly2d poly):
        self.c_resamp.setRgCarrier(poly.c_poly2d)
    @property
    def azCarrier(self):
        return PyPoly2d.boundTo(self.c_resamp.azCarrier)
    @azCarrier.setter
    def azCarrier(self, PyPoly2d poly):
        self.c_resamp.setAzCarrier(poly.c_poly2d)
    @property
    def rgOffsetsPoly(self):
        return PyPoly2d.boundTo(self.c_resamp.rgOffsetsPoly)
    @rgOffsetsPoly.setter
    def rgOffsetsPoly(self, PyPoly2d poly):
        self.c_resamp.setRgOffsets(poly.c_poly2d)
    @property
    def azOffsetsPoly(self):
        return PyPoly2d.boundTo(self.c_resamp.azOffsetsPoly)
    @azOffsetsPoly.setter
    def azOffsetsPoly(self, PyPoly2d poly):
        self.c_resamp.setAzOffsets(poly.c_poly2d)
    @property
    def dopplerPoly(self):
        return PyPoly2d.boundTo(self.c_resamp.dopplerPoly)
    @dopplerPoly.setter
    def dopplerPoly(self, PyPoly2d poly):
        self.c_resamp.setDoppler(poly.c_poly2d)

    # Note: The "release" functions will return a PyPoly2d object that is bound
    #       to the corresponding ResampSlc's Poly2d, but the difference between
    #       this and the regular PyPoly2d property is that the PyPoly2d object
    #       becomes "unbound" (i.e. when the PyPoly2d object goes out of scope,
    #       it will destroy the Poly2d it's bound to)
    def releaseRgCarrier(self):
        cdef PyPoly2d poly = PyPoly2d.boundTo(self.c_resamp.releaseRgCarrier())
        poly.owner = True
        return poly
    def releaseAzCarrier(self):
        cdef PyPoly2d poly = PyPoly2d.boundTo(self.c_resamp.releaseAzCarrier())
        poly.owner = True
        return poly
    def releaseRgOffsets(self):
        cdef PyPoly2d poly = PyPoly2d.boundTo(self.c_resamp.releaseRgOffsets())
        poly.owner = True
        return poly
    def releaseAzOffsets(self):
        cdef PyPoly2d poly = PyPoly2d.boundTo(self.c_resamp.releaseAzOffsets())
        poly.owner = True
        return poly
    def releaseDoppler(self):
        cdef PyPoly2d poly = PyPoly2d.boundTo(self.c_resamp.releaseDoppler())
        poly.owner = True
        return poly

    def resamp_slc(self):
        self.c_resamp.resamp()

