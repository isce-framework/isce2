#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

cdef extern from "Ellipsoid.h" namespace "isceLib":
    cdef cppclass Ellipsoid:
        double a,e2

        Ellipsoid() except +
        Ellipsoid(double,double) except +
        Ellipsoid(const Ellipsoid&) except +
        double rEast(double)
        double rNorth(double)
        double rDir(double,double)
        void latLon(double[3],double[3],int)
        void getAngs(double[3],double[3],double[3],double&,double&)
        void getTCN_TCvec(double[3],double[3],double[3],double[3])


cdef class PyEllipsoid:
    cdef Ellipsoid c_ellipsoid

    def __cinit__(self, a=None, b=None): # Handles empty and non-empty constructors
        if (a and b): # Non-empty constructor call
            self.a = a
            self.e2 = b
    
    @property
    def a(self): # Access to the properties of the underlying c_ellipsoid object w/o needing a getter/setter
        return self.c_ellipsoid.a
    @a.setter
    def a(self, double a):
        self.c_ellipsoid.a = a
    @property
    def e2(self):
        return self.c_ellipsoid.e2
    @e2.setter
    def e2(self, double a):
        self.c_ellipsoid.e2 = a
    def copy(self, elp): # Replaces copy-constructor functionality
        try:
            self.a = elp.a
            self.e2 = elp.e2
        except: # Note: this allows for a dummy class object to be passed in that just has a and e2 as parameters!
            print("Error: Object passed in to copy is not of type PyEllipsoid.")
    def dPrint(self):
        print('a = '+str(self.a)+', e2 = '+str(self.e2))

    def rEast(self, double a):
        return self.c_ellipsoid.rEast(a)
    def rNorth(self, double a):
        return self.c_ellipsoid.rNorth(a)
    def rDir(self, double a, double b):
        return self.c_ellipsoid.rDir(a,b)
    def latLon(self, list a, list b, int c):
        cdef double _a[3]
        cdef double _b[3]
        for i in range(3):
            _a[i] = a[i]
            _b[i] = b[i]
        self.c_ellipsoid.latLon(_a,_b,c)
        for i in range(3):
            a[i] = _a[i]
            b[i] = _b[i]
    def getAngs(self, list a, list b, list c, d, e=None):
        cdef double _a[3]
        cdef double _b[3]
        cdef double _c[3]
        cdef double _d,_e
        if (e):
            print("Error: Python cannot pass primitives by reference.")
            print("To call this function, please pass the function an empty tuple as the fourth")
            print("argument (no fifth argument). The first element of the list will be the azimuth")
            print("angle, the second element will be the look angle.")
        else:
            _d = 0.
            _e = 0.
            for i in range(3):
                _a[i] = a[i]
                _b[i] = b[i]
                _c[i] = c[i]
            self.c_ellipsoid.getAngs(_a,_b,_c,_d,_e)
            for i in range(3):
                a[i] = _a[i]
                b[i] = _b[i]
                c[i] = _c[i]
            d[0] = _d
            d[1] = _e
    def getTCN_TCvec(self, list a, list b, list c, list d):
        cdef double _a[3]
        cdef double _b[3]
        cdef double _c[3]
        cdef double _d[3]
        for i in range(3):
            _a[i] = a[i]
            _b[i] = b[i]
            _c[i] = c[i]
            _d[i] = d[i]
        self.c_ellipsoid.getTCN_TCvec(_a,_b,_c,_d)
        for i in range(3):
            a[i] = _a[i]
            b[i] = _b[i]
            c[i] = _c[i]
            d[i] = _d[i]

