#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

cdef extern from "Position.h" namespace "isceLib":
    cdef cppclass Position:
        double j[3]
        double jdot[3]
        double jddt[3]

        Position() except +
        Position(const Position&) except +
        void lookVec(double,double,double[3])


cdef class PyPosition:
    cdef Position c_position

    def __cinit__(self):
        return
    
    @property
    def j(self):
        a = [0.,0.,0.]
        for i in range(3):
            a[i] = self.c_position.j[i]
        return a
    @j.setter
    def j(self, a):
        if (len(a) != 3):
            print("Error: Invalid input size.")
            return
        for i in range(3):
            self.c_position.j[i] = a[i]
    @property
    def jdot(self):
        a = [0.,0.,0.]
        for i in range(3):
            a[i] = self.c_position.jdot[i]
        return a
    @jdot.setter
    def jdot(self, a):
        if (len(a) != 3):
            print("Error: Invalid input size.")
            return
        for i in range(3):
            self.c_position.jdot[i] = a[i]
    @property
    def jddt(self):
        a = [0.,0.,0.]
        for i in range(3):
            a[i] = self.c_position.jddt[i]
        return a
    @jddt.setter
    def jddt(self, a):
        if (len(a) != 3):
            print("Error: Invalid input size.")
            return
        for i in range(3):
            self.c_position.jddt[i] = a[i]
    def dPrint(self):
        print("J = "+str(self.j)+", jdot = "+str(self.jdot)+", jddt = "+str(self.jddt))
    def copy(self, ps):
        self.j = ps.j
        self.jdot = ps.jdot
        self.jddt = ps.jddt

    def lookVec(self, double a, double b, list c):
        cdef double _c[3]
        for i in range(3):
            _c[i] = c[i]
        self.c_position.lookVec(a,b,_c)
        for i in range(3):
            c[i] = _c[i]
