#
# Author: Joshua Cohen
# Copyright 2017
#
from libc.stdint cimport uint64_t
from libcpp cimport bool

cdef extern from "GeoController.h":
    cdef cppclass GeoController:
        cppclass Geo2rdr:
            cppclass Orbit:
                int nVectors,basis
                double *position
                double *velocity
                double *UTCtime
                Orbit() except +
                Orbit(const Orbit&) except +
                void setOrbit(int,int)
                void setOrbit(char*,int)
                void getPositionVelocity(double,double[3],double[3])
                void setStateVector(int,double,double[3],double[3])
                void getStateVector(int,double&,double[3],double[3])
                int interpolateOrbit(double,double[3],double[3],int)
                int interpolateSCHOrbit(double,double[3],double[3])
                int interpolateWGS84Orbit(double,double[3],double[3])
                int interpolateLegendreOrbit(double,double[3],double[3])
                int computeAcceleration(double,double[3])
                void orbitHermite(double[4][3],double[4][3],double[4],double,double[3],double[3])
                void dumpToHDR(char*)
                void printOrbit()
            cppclass Poly1d:
                double *coeffs
                double mean,norm
                int order
                Poly1d() except +
                Poly1d(int) except +
                Poly1d(const Poly1d&) except +
                void setPoly(int,double,double)
                double eval(double)
                void setCoeff(int,double)
                double getCoeff(int)
                void printPoly()
            double major,eccentricitySquared,drho,rngstart,wvl,tstart,prf
            uint64_t latAccessor,lonAccessor,hgtAccessor,azAccessor,rgAccessor,azOffAccessor,rgOffAccessor
            int imgLength,imgWidth,demLength,demWidth,nRngLooks,nAzLooks,orbit_nvecs,orbit_basis,orbitMethod
            int poly_order,poly_mean,poly_norm
            bool bistatic
            Orbit orb
            Poly1d dop
            Geo2rdr() except +
            void geo2rdr()
            void createOrbit()
            void createPoly()
        Geo2rdr geo
        GeoController() except +
        void runGeo2rdr()
        void createOrbit()
        void createPoly()
        void setEllipsoidMajorSemiAxis(double)
        void setEllipsoidEccentricitySquared(double)
        void setRangePixelSpacing(double)
        void setRangeFirstSample(double)
        void setPRF(double)
        void setRadarWavelength(double)
        void setSensingStart(double)
        void setLatAccessor(uint64_t)
        void setLonAccessor(uint64_t)
        void setHgtAccessor(uint64_t)
        void setAzAccessor(uint64_t)
        void setRgAccessor(uint64_t)
        void setAzOffAccessor(uint64_t)
        void setRgOffAccessor(uint64_t)
        void setLength(int)
        void setWidth(int)
        void setDemLength(int)
        void setDemWidth(int)
        void setNumberRangeLooks(int)
        void setNumberAzimuthLooks(int)
        void setBistaticFlag(int)
        void setOrbitMethod(int)
        void setOrbitNvecs(int)
        void setOrbitBasis(int)
        void setOrbitVector(int,double,double,double,double,double,double,double)
        void setPolyOrder(int)
        void setPolyMean(double)
        void setPolyNorm(double)
        void setPolyCoeff(int,double)

cdef class PyGeo2rdr:
    cdef GeoController c_geoController
    cdef bool orbSet, polySet
    
    def __cinit__(self):
        orbSet = False
        polySet = False
        return
    def geo2rdr(self):
        self.c_geoController.runGeo2rdr()
    def setEllipsoidMajorSemiAxis(self, double v):
        self.c_geoController.setEllipsoidMajorSemiAxis(v)
    def setEllipsoidEccentricitySquared(self, double v):
        self.c_geoController.setEllipsoidEccentricitySquared(v)
    def setRangePixelSpacing(self, double v):
        self.c_geoController.setRangePixelSpacing(v)
    def setRangeFirstSample(self, double v):
        self.c_geoController.setRangeFirstSample(v)
    def setPRF(self, double v):
        self.c_geoController.setPRF(v)
    def setRadarWavelength(self, double v):
        self.c_geoController.setRadarWavelength(v)
    def setSensingStart(self, double v):
        self.c_geoController.setSensingStart(v)
    def setLatAccessor(self, uint64_t v):
        self.c_geoController.setLatAccessor(v)
    def setLonAccessor(self, uint64_t v):
        self.c_geoController.setLonAccessor(v)
    def setHgtAccessor(self, uint64_t v):
        self.c_geoController.setHgtAccessor(v)
    def setAzAccessor(self, uint64_t v):
        self.c_geoController.setAzAccessor(v)
    def setRgAccessor(self, uint64_t v):
        self.c_geoController.setRgAccessor(v)
    def setAzOffAccessor(self, uint64_t v):
        self.c_geoController.setAzOffAccessor(v)
    def setRgOffAccessor(self, uint64_t v):
        self.c_geoController.setRgOffAccessor(v)
    def setLength(self, int v):
        self.c_geoController.setLength(v)
    def setWidth(self, int v):
        self.c_geoController.setWidth(v)
    def setDemLength(self, int v):
        self.c_geoController.setDemLength(v)
    def setDemWidth(self, int v):
        self.c_geoController.setDemWidth(v)
    def setNumberRangeLooks(self, int v):
        self.c_geoController.setNumberRangeLooks(v)
    def setNumberAzimuthLooks(self, int v):
        self.c_geoController.setNumberAzimuthLooks(v)
    def setBistaticFlag(self, int v):
        self.c_geoController.setBistaticFlag(v)
    def setOrbitMethod(self, int v):
        self.c_geoController.setOrbitMethod(v)
    def createOrbit(self, int basis, int nvec):
        self.c_geoController.setOrbitBasis(basis)
        self.c_geoController.setOrbitNvecs(nvec)
        self.c_geoController.createOrbit()
        self.orbSet = True
    def setOrbitVector(self, int idx, double t, double px, double py, double pz, double vx, double vy, double vz):
        if (self.orbSet):
            if (idx < self.c_geoController.geo.orb.nVectors):
                self.c_geoController.setOrbitVector(idx,t,px,py,pz,vx,vy,vz)
            else:
                print("Error: Trying to set state vector "+str(idx+1)+" out of "+str(self.c_geoController.geo.orb.nVectors)+".")
        else:
            print("Error: Orbit has not been set with 'createOrbit', therefore state vectors cannot be added (memory space has not been malloc'ed).")
    def createPoly(self, int order, double mean, double norm):
        self.c_geoController.setPolyOrder(order)
        self.c_geoController.setPolyMean(mean)
        self.c_geoController.setPolyNorm(norm)
        self.c_geoController.createPoly()
        self.polySet = True
    def setPolyCoeff(self, int idx, double coeff):
        if (self.polySet):
            if (idx <= self.c_geoController.geo.dop.order):
                self.c_geoController.setPolyCoeff(idx,coeff)
            else:
                print("Error: Trying to set poly coefficient "+str(idx+1)+" out of "+str(self.c_geoController.geo.dop.order+1)+".")
        else:
            print("Error: Poly has not been set with 'createPoly', therefore coefficients cannot be added (memory space has not been malloc'ed).")
    def printPoly(self):
        print(self.c_geoController.geo.dop.order)

