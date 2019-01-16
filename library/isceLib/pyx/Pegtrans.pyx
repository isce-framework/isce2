#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

cdef extern from "Pegtrans.h" namespace "isceLib":
    cdef cppclass Pegtrans:
        double mat[3][3]
        double matinv[3][3]
        double ov[3]
        double radcur

        Pegtrans() except +
        Pegtrans(const Pegtrans&) except +
        void radarToXYZ(Ellipsoid&,Peg&)
        void convertSCHtoXYZ(double[3],double[3],int)
        void convertSCHdotToXYZdot(double[3],double[3],double[3],double[3],int)
        void SCHbasis(double[3],double[3][3],double[3][3])


cdef class PyPegtrans:
    cdef Pegtrans c_pegtrans

    def __cinit__(self): # Never will be initialized with values, so no need to check
        return
    
    @property
    def mat(self):
        a = [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]
        for i in range(3):
            for j in range(3):
                a[i][j] = self.c_pegtrans.mat[i][j]
        return a
    @mat.setter
    def mat(self, a):
        if ((len(a) != 3) or (len(a[0]) != 3)):
            print("Error: Invalid input size.")
            return
        for i in range(3):
            for j in range(3):
                self.c_pegtrans.mat[i][j] = a[i][j]
    @property
    def matinv(self):
        a = [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]
        for i in range(3):
            for j in range(3):
                a[i][j] = self.c_pegtrans.matinv[i][j]
        return a
    @matinv.setter
    def matinv(self, a):
        if ((len(a) != 3) or (len(a[0]) != 3)):
            print("Error: Invalid input size.")
            return
        for i in range(3):
            for j in range(3):
                self.c_pegtrans.matinv[i][j] = a[i][j]
    @property
    def ov(self):
        a = [0.,0.,0.]
        for i in range(3):
            a[i] = self.c_pegtrans.ov[i]
        return a
    @ov.setter
    def ov(self, a):
        if (len(a) != 3):
            print("Error: Invalid input size.")
            return
        for i in range(3):
            self.c_pegtrans.ov[i] = a[i]
    @property
    def radcur(self):
        return self.c_pegtrans.radcur
    @radcur.setter
    def radcur(self, double a):
        self.c_pegtrans.radcur = a
    def dPrint(self):
        m = self.mat
        mi = self.matinv
        o = self.ov
        r = self.radcur
        print("Mat = "+str(m)+", matinv = "+str(mi)+", ov = "+str(o)+", radcur = "+str(r))
    def copy(self, pt):
        try:
            self.mat = pt.mat
            self.matinv = pt.matinv
            self.ov = pt.ov
            self.radcur = pt.radcur
        except:
            print("Error: Object passed in is not of type PyPegtrans.")

    def radarToXYZ(self, PyEllipsoid a, PyPeg b):
        self.c_pegtrans.radarToXYZ(a.c_ellipsoid,b.c_peg)
    def convertSCHtoXYZ(self, list a, list b, int c):
        cdef double _a[3]
        cdef double _b[3]
        for i in range(3):
            _a[i] = a[i]
            _b[i] = b[i]
        self.c_pegtrans.convertSCHtoXYZ(_a,_b,c)
        for i in range(3):
            a[i] = _a[i]
            b[i] = _b[i]
    def convertSCHdotToXYZdot(self, list a, list b, list c, list d, int e):
        cdef double _a[3]
        cdef double _b[3]
        cdef double _c[3]
        cdef double _d[3]
        for i in range(3):
            _a[i] = a[i]
            _b[i] = b[i]
            _c[i] = c[i]
            _d[i] = d[i]
        self.c_pegtrans.convertSCHdotToXYZdot(_a,_b,_c,_d,e)
        for i in range(3):
            a[i] = _a[i]
            b[i] = _b[i]
            c[i] = _c[i]
            d[i] = _d[i]
    def SCHbasis(self, list a, list b, list c):
        cdef double _a[3]
        cdef double _b[3][3]
        cdef double _c[3][3]
        for i in range(3):
            _a[i] = a[i]
            for j in range(3):
                _b[i][j] = b[i][j]
                _c[i][j] = c[i][j]
        self.c_pegtrans.SCHbasis(_a,_b,_c)
        for i in range(3):
            a[i] = _a[i]
            for j in range(3):
                b[i][j] = _b[i][j]
                c[i][j] = _c[i][j]
