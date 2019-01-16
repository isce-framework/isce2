#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

#################################################################

cdef extern from "Orbit.h" namespace "isceLib":
    cdef cppclass Orbit:
        int basis
        int nVectors
        double *position
        double *velocity
        double *UTCtime

        Orbit() except +
        Orbit(int,int) except +
        Orbit(const Orbit&) except +
        int isNull()
        void resetStateVectors()
        void getPositionVelocity(double,double[3],double[3])
        void getStateVector(int,double&,double[3],double[3])
        void setStateVector(int,double,double[3],double[3])
        int interpolate(double,double[3],double[3],int)
        int interpolateWGS84Orbit(double,double[3],double[3])
        int interpolateLegendreOrbit(double,double[3],double[3])
        int interpolateSCHOrbit(double,double[3],double[3])
        int computeAcceleration(double,double[3])
        void printOrbit()
        void loadFromHDR(const char*,int)
        void dumpToHDR(const char*)

cdef class PyOrbit:
    cdef Orbit c_orbit

    def __cinit__(self, a=None, b=None):
        if (a): # Init with basis/nvec
            self.basis = a
            if (b):
                self.nVectors = b
            else:
                self.nVectors = 0
        elif (b):
            if (a):
                self.basis = a
            else:
                self.basis = 1
            self.nVectors = b

    @property
    def basis(self):
        return self.c_orbit.basis
    @basis.setter
    def basis(self, int a):
        self.c_orbit.basis = a
    @property
    def nVectors(self):
        return self.c_orbit.nVectors
    @nVectors.setter
    def nVectors(self, int a):
        if (a < 0):
            return
        if (a == 0):
            self.c_orbit.nVectors = 0
            self.resetStateVectors()
            self.UTCtime = []
            self.position = []
            self.velocity = []
            return
        t = self.UTCtime
        p = self.position
        v = self.velocity
        for i in range(a-self.nVectors):
            t.append(0.)
            for j in range(3):
                p.append(0.)
                v.append(0.)
        nt = []
        np = []
        nv = []
        for i in range(a):
            nt.append(t[i])
            for j in range(3):
                np.append(p[3*i+j])
                nv.append(v[3*i+j])
        self.c_orbit.nVectors = a
        self.resetStateVectors()
        self.UTCtime = nt
        self.position = np
        self.velocity = nv
    @property
    def UTCtime(self):
        a = []
        if (self.isNull() == 1):
            return a
        for i in range(self.nVectors):
            a.append(self.c_orbit.UTCtime[i])
        return a
    @UTCtime.setter
    def UTCtime(self, a):
        if (self.isNull() == 1):
            print("Warning: Memory was not malloc'd for storage. nVectors will be set appropriately.")
            self.nVectors = len(a) # internal call to resetStateVectors()
        if (self.nVectors != len(a)):
            print("Error: Invalid input size (expected list of length "+str(self.nVectors)+")")
            return
        for i in range(self.nVectors):
            self.c_orbit.UTCtime[i] = a[i]
    @property
    def position(self):
        a = []
        if (self.isNull() == 1):
            return a
        for i in range(3*self.nVectors):
            a.append(self.c_orbit.position[i])
        return a
    @position.setter
    def position(self, a):
        if (len(a)%3 != 0):
            print("Error: Expected list with length of a multiple of 3.")
            return
        if (self.isNull() == 1):
            print("Warning: Memory was not malloc'd for storage. nVectors will be set appropriately.")
            self.nVectors = len(a) / 3
        if (3*self.nVectors != len(a)):
            print("Error: Invalid input size (expected list of length "+str(3*self.nVectors)+")")
            return
        for i in range(3*self.nVectors):
            self.c_orbit.position[i] = a[i]
    @property
    def velocity(self):
        a = []
        if (self.isNull() == 1):
            return a
        for i in range(3*self.nVectors):
            a.append(self.c_orbit.velocity[i])
        return a
    @velocity.setter
    def velocity(self, a):
        if (len(a)%3 != 0):
            print("Error: Expected list with length of a multiple of 3.")
            return
        if (self.isNull() == 1):
            print("Warning: Memory was not malloc'd for storage. nVectors will be set appropriately.")
            self.nVectors = len(a) / 3
        if (3*self.nVectors != len(a)):
            print("Error: Invalid input size (expected list of length "+str(3*self.nVectors)+")")
            return
        for i in range(3*self.nVectors):
            self.c_orbit.velocity[i] = a[i]
    def copy(self, orb):
        try:
            self.basis = orb.basis
            self.nVectors = orb.nVectors
            self.UTCtime = orb.UTCtime
            self.position = orb.position
            self.velocity = orb.velocity
        except:
            print("Error: Object passed in to copy is not of type PyOrbit.")
    def dPrint(self):
        self.printOrbit()

    def isNull(self):
        return self.c_orbit.isNull()
    def resetStateVectors(self):
        self.c_orbit.resetStateVectors()
    def getPositionVelocity(self, double a, list b, list c):
        cdef double _b[3]
        cdef double _c[3]
        for i in range(3):
            _b[i] = b[i]
            _c[i] = c[i]
        self.c_orbit.getPositionVelocity(a,_b,_c)
        for i in range(3):
            b[i] = _b[i]
            c[i] = _c[i]
    def getStateVector(self, int a, b, list c, list d):
        cdef double _c[3]
        cdef double _d[3]
        cdef double _b
        if (type(b) != type([])):
            print("Error: Python cannot pass primitives by reference.")
            print("To call this function, please pass the function an empty 1-tuple in the")
            print("second argument slot. The function will store the resulting time value")
            print("as the first (and only) element in the 1-tuple.")
        else:
            _b = 0.
            for i in range(3):
                _c[i] = c[i]
                _d[i] = d[i]
            self.c_orbit.getStateVector(a,_b,_c,_d)
            for i in range(3):
                c[i] = _c[i]
                d[i] = _d[i]
            b[0] = _b
    def setStateVector(self, int a, double b, list c, list d):
        cdef double _c[3]
        cdef double _d[3]
        for i in range(3):
            _c[i] = c[i]
            _d[i] = d[i]
        self.c_orbit.setStateVector(a,b,_c,_d)
        for i in range(3):
            c[i] = _c[i]
            d[i] = _d[i]
    def interpolate(self, double a, list b, list c, int d):
        cdef double _b[3]
        cdef double _c[3]
        cdef int ret
        for i in range(3):
            _b[i] = b[i]
            _c[i] = c[i]
        ret = self.c_orbit.interpolate(a,_b,_c,d)
        for i in range(3):
            b[i] = _b[i]
            c[i] = _c[i]
        return ret
    def interpolateWGS84Orbit(self, double a, list b, list c):
        cdef double _b[3]
        cdef double _c[3]
        cdef int ret
        for i in range(3):
            _b[i] = b[i]
            _c[i] = c[i]
        ret = self.c_orbit.interpolateWGS84Orbit(a,_b,_c)
        for i in range(3):
            b[i] = _b[i]
            c[i] = _c[i]
        return ret
    def interpolateLegendreOrbit(self, double a, list b, list c):
        cdef double _b[3]
        cdef double _c[3]
        cdef int ret
        for i in range(3):
            _b[i] = b[i]
            _c[i] = c[i]
        ret = self.c_orbit.interpolateLegendreOrbit(a,_b,_c)
        for i in range(3):
            b[i] = _b[i]
            c[i] = _c[i]
        return ret
    def interpolateSCHOrbit(self, double a, list b, list c):
        cdef double _b[3]
        cdef double _c[3]
        cdef int ret
        for i in range(3):
            _b[i] = b[i]
            _c[i] = c[i]
        ret = self.c_orbit.interpolateSCHOrbit(a,_b,_c)
        for i in range(3):
            b[i] = _b[i]
            c[i] = _c[i]
        return ret
    def computeAcceleration(self, double a, list b):
        cdef double _b[3]
        cdef int ret
        for i in range(3):
            _b[i] = b[i]
        ret = self.c_orbit.computeAcceleration(a,_b)
        for i in range(3):
            b[i] = _b[i]
        return ret
    def printOrbit(self):
        self.c_orbit.printOrbit()
    def loadFromHDR(self, a, int b=1):
        cdef bytes _a = a.encode()
        cdef char *cstring = _a
        self.c_orbit.loadFromHDR(cstring,b)
    def dumpToHDR(self, a):
        cdef bytes _a = a.encode()
        cdef char *cstring = _a
        self.c_orbit.dumpToHDR(cstring)

