#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

#################################################################

cdef extern from "Poly2d.h" namespace "isceLib":
    cdef cppclass Poly2d:
        int azimuthOrder
        int rangeOrder
        double azimuthMean
        double rangeMean
        double azimuthNorm
        double rangeNorm
        double *coeffs

        Poly2d() except +
        Poly2d(int,int,double,double,double,double) except +
        Poly2d(int,int,double,double,double,double,double*) except +
        Poly2d(const Poly2d&) except +
        int isNull()
        void resetCoeffs()
        void setCoeff(int,int,double)
        double getCoeff(int,int)
        double eval(double,double)

cdef class PyPoly2d:
    cdef Poly2d c_poly2d

    def __cinit__(self, a=None, b=None, c=None, d=None, e=None, f=None, g=None):
        if (g): # init with coeffs
            if (a and b):
                self.c_poly2d.azimuthOrder = a # Have to avoid the setters due to the 2D nature
                self.c_poly2d.rangeOrder = b
                self.resetCoeffs()
                self.coeffs = g
            else:
                print("Error: Cannot init Poly2d with coefficients without specifying range and azimuth order.")
                self.resetCoeffs()
                self.coeffs = []
            self.azimuthMean = c
            self.rangeMean = d
            self.azimuthNorm = e
            self.rangeNorm = f
        elif (a): # Init without coeffs
            if (a and b):
                self.c_poly2d.azimuthOrder = a
                self.c_poly2d.rangeOrder = b
            elif (a):
                self.c_poly2d.azimuthOrder = a
                self.c_poly2d.rangeOrder = 0
            else:
                self.c_poly2d.azimuthOrder = 0
                self.c_poly2d.rangeOrder = b
            self.azimuthMean = c
            self.rangeMean = d
            self.azimuthNorm = e
            self.rangeNorm = f
            self.resetCoeffs()
    
    @property
    def azimuthOrder(self):
        return self.c_poly2d.azimuthOrder
    @azimuthOrder.setter
    def azimuthOrder(self, int a):
        # Need a better way to do this...
        if (a < 0):
            return
        if (self.rangeOrder == -1): # only on empty constructor
            self.c_poly2d.azimuthOrder = a
        else:
            c = self.coeffs
            for i in range((a-self.azimuthOrder)*(self.rangeOrder+1)):
                c.append(0.)
            nc = []
            for i in range((a+1)*(self.rangeOrder+1)):
                nc.append(c[i])
            self.c_poly2d.azimuthOrder = a
            self.resetCoeffs()
            self.coeffs = nc
    @property
    def rangeOrder(self):
        return self.c_poly2d.rangeOrder
    @rangeOrder.setter
    def rangeOrder(self, int a):
        # Need a better way to do this...
        if (a < 0):
            return
        if (self.azimuthOrder == -1):
            self.c_poly2d.rangeOrder = a
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
            self.resetCoeffs()
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
        if (self.isNull() == 1):
            return a
        for i in range((self.azimuthOrder+1)*(self.rangeOrder+1)):
            a.append(self.c_poly2d.coeffs[i])
        return a
    @coeffs.setter
    def coeffs(self, a):
        if ((self.azimuthOrder+1)*(self.rangeOrder+1) != len(a)):
            print("Error: Invalid input size (expected 1D list of length "+str(self.azimuthOrder+1)+"*"+str(self.rangeOrder+1)+")")
            return
        if (self.isNull() == 1): # Only happens if you try to immediately set coefficients after calling the empty constructor
            print("Warning: Memory was not malloc'd for coefficients. Range/azimuth order cannot be inferred, so coefficients will not be set.")
            return
        for i in range((self.azimuthOrder+1)*(self.rangeOrder+1)):
            self.c_poly2d.coeffs[i] = a[i]
    def copy(self, poly):
        try:
            self.azimuthOrder = poly.azimuthOrder
            self.rangeOrder = poly.rangeOrder
            self.azimuthMean = poly.azimuthMean
            self.rangeMean = poly.rangeMean
            self.azimuthNorm = poly.azimuthNorm
            self.rangeNorm = poly.rangeNorm
            self.resetCoeffs()
            self.coeffs = poly.coeffs
        except:
            print("Error: Object passed in to copy is not of type PyPoly2d.")
    def dPrint(self):
        print("AzimuthOrder = "+str(self.azimuthOrder)+", rangeOrder = "+str(self.rangeOrder)+", azimuthMean = "+str(self.azimuthMean)+", rangeMean = "+str(self.rangeMean)+
                ", azimuthNorm = "+str(self.azimuthNorm)+", rangeNorm = "+str(self.rangeNorm)+", coeffs = "+str(self.coeffs))

    def isNull(self):
        return self.c_poly2d.isNull()
    def resetCoeffs(self):
        self.c_poly2d.resetCoeffs()
    def setCoeff(self, int a, int b, double c):
        self.c_poly2d.setCoeff(a,b,c)
    def getCoeff(self, int a, int b):
        return self.c_poly2d.getCoeff(a,b)
    def eval(self, double a, double b):
        return self.c_poly2d.eval(a,b)


