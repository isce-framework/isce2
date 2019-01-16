#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

#################################################################

cdef extern from "LinAlg.h" namespace "isceLib":
    cdef cppclass LinAlg:
        LinAlg() except +
        void cross(double[3],double[3],double[3])
        double dot(double[3],double[3])
        void linComb(double,double[3],double,double[3],double[3])
        void matMat(double[3][3],double[3][3],double[3][3])
        void matVec(double[3][3],double[3],double[3])
        double norm(double[3])
        void tranMat(double[3][3],double[3][3])
        void unitVec(double[3],double[3])

cdef class PyLinAlg:
    cdef LinAlg c_linAlg

    def __cinit__(self):
        return

    def cross(self, list a, list b, list c):
        cdef double _a[3]
        cdef double _b[3]
        cdef double _c[3]
        for i in range(3):
            _a[i] = a[i]
            _b[i] = b[i]
            _c[i] = c[i]
        self.c_linAlg.cross(_a,_b,_c)
        for i in range(3):
            a[i] = _a[i]
            b[i] = _b[i]
            c[i] = _c[i]
    def dot(self, list a, list b):
        cdef double _a[3]
        cdef double _b[3]
        for i in range(3):
            _a[i] = a[i]
            _b[i] = b[i]
        return self.c_linAlg.dot(_a,_b)
    def linComb(self, double a, list b, double c, list d, list e):
        cdef double _b[3]
        cdef double _d[3]
        cdef double _e[3]
        for i in range (3):
            _b[i] = b[i]
            _d[i] = d[i]
            _e[i] = e[i]
        self.c_linAlg.linComb(a,_b,c,_d,_e)
        for i in range(3):
            b[i] = _b[i]
            d[i] = _d[i]
            e[i] = _e[i]
    def matMat(self, list a, list b, list c):
        cdef double _a[3][3]
        cdef double _b[3][3]
        cdef double _c[3][3]
        for i in range(3):
            for j in range(3):
                _a[i][j] = a[i][j]
                _b[i][j] = b[i][j]
                _c[i][j] = c[i][j]
        self.c_linAlg.matMat(_a,_b,_c)
        for i in range(3):
            for j in range(3):
                a[i][j] = _a[i][j]
                b[i][j] = _b[i][j]
                c[i][j] = _c[i][j]
    def matVec(self, list a, list b, list c):
        cdef double _a[3][3]
        cdef double _b[3]
        cdef double _c[3]
        for i in range(3):
            for j in range(3):
                _a[i][j] = a[i][j]
            _b[i] = b[i]
            _c[i] = c[i]
        self.c_linAlg.matVec(_a,_b,_c)
        for i in range(3):
            for j in range(3):
                a[i][j] = _a[i][j]
            b[i] = _b[i]
            c[i] = _c[i]
    def norm(self, list a):
        cdef double _a[3]
        for i in range(3):
            _a[i] = a[i]
        return self.c_linAlg.norm(_a)
    def tranMat(self, list a, list b):
        cdef double _a[3][3]
        cdef double _b[3][3]
        for i in range(3):
            for j in range(3):
                _a[i][j] = a[i][j]
                _b[i][j] = b[i][j]
        self.c_linAlg.tranMat(_a,_b)
        for i in range(3):
            for j in range(3):
                a[i][j] = _a[i][j]
                b[i][j] = _b[i][j]
    def unitVec(self, list a, list b):
        cdef double _a[3]
        cdef double _b[3]
        for i in range(3):
            _a[i] = a[i]
            _b[i] = b[i]
        self.c_linAlg.unitVec(_a,_b)
        for i in range(3):
            a[i] = _a[i]
            b[i] = _b[i]

