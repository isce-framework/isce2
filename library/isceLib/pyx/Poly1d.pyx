#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

#################################################################

cdef extern from "Poly1d.h" namespace "isceLib":
    cdef cppclass Poly1d:
        int order
        double mean
        double norm
        double *coeffs

        Poly1d() except +
        Poly1d(int,double,double) except +
        Poly1d(int,double,double,double*) except +
        Poly1d(const Poly1d&) except +
        int isNull()
        void resetCoeffs()
        void setCoeff(int,double)
        double getCoeff(int)
        double eval(double)

cdef class PyPoly1d:
    cdef Poly1d c_poly1d

    def __cinit__(self, a=None, b=None, c=None, d=None):
        if (d): # Init with coeffs
            if (a):
                self.order = a
            else:
                self.order = len(d) - 1
            self.mean = b
            self.norm = c
            self.coeffs = d
        elif (a): # Init without coeffs
            self.order = a
            self.mean = b
            self.norm = c
    
    @property
    def order(self):
        return self.c_poly1d.order
    @order.setter
    def order(self, int a):
        if (a < 0):
            return
        if (a+1 != len(self.coeffs)):
            c = self.coeffs
            for i in range(a+1-len(c)): # If new order is higher than current order
                c.append(0.)
            nc = []
            for i in range(a+1): # Truncate coeffs as necesary
                nc.append(c[i])
            self.c_poly1d.order = a
            self.resetCoeffs()
            self.coeffs = nc
    @property
    def mean(self):
        return self.c_poly1d.mean
    @mean.setter
    def mean(self, double a):
        self.c_poly1d.mean = a
    @property
    def norm(self):
        return self.c_poly1d.norm
    @norm.setter
    def norm(self, double a):
        self.c_poly1d.norm = a
    @property
    def coeffs(self):
        a = []
        if (self.isNull() == 1):
            return a
        for i in range(self.order+1):
            a.append(self.c_poly1d.coeffs[i])
        return a
    @coeffs.setter
    def coeffs(self, a):
        if (self.order+1 != len(a)):
            print("Error: Invalid input size (expected list of length "+str(self.order+1)+")")
            return
        if (self.isNull() == 1):
            print("Warning: Memory was not malloc'd for coefficients. Order will be set appropriately.")
            self.order = len(a) - 1
            self.resetCoeffs()
        for i in range(self.order+1):
            self.c_poly1d.coeffs[i] = a[i]
    def copy(self, poly):
        try:
            self.order = poly.order
            self.mean = poly.mean
            self.norm = poly.norm
            self.resetCoeffs()
            self.coeffs = poly.coeffs
        except:
            print("Error: Object passed in to copy is not of type PyPoly1d.")
    def dPrint(self):
        print("Order = "+str(self.order)+", mean = "+str(self.mean)+", norm = "+str(self.norm)+", coeffs = "+str(self.coeffs))

    def isNull(self):
        return self.c_poly1d.isNull()
    def resetCoeffs(self):
        self.c_poly1d.resetCoeffs()
    def setCoeff(self, int a, double b):
        self.c_poly1d.setCoeff(a,b)
    def getCoeff(self, int a):
        return self.c_poly1d.getCoeff(a)
    def eval(self, double a):
        return self.c_poly1d.eval(a)

