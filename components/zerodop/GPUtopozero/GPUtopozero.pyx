#
# Author: Joshua Cohen
# Copyright 2016
#
# Basic interface using the Cython builder to wrap the Topozero() object in Python
# To use the module (after it's been compiled, see the setup_note in the same directory),
# call 'import gpu_topozero' or 'from gpu_topozero import PyTopozero' to be able to create
# a PyTopozero() object. Once the object has been wired correctly, call the 'createOrbit()'
# and 'runTopo()' methods to run the underlying C++ code.
#
# Note: The cdef is being pulled from 'include/Topozero.h' so this file must be sitting in
#       a directory that contains the src/ and include/ folders in order to link properly!

from libc.stdint cimport uint64_t # Needed to pass accessor pointers
from libcpp.vector cimport vector # Never used in Python, needed for the Orbit object definitions

cdef extern from "Topozero.h":
    cdef cppclass Topozero:
        cppclass Topo:
            cppclass Orbit:
                int nVectors,basis
                vector[double] position,velocity,UTCtime
                Orbit() except +
                void setOrbit(int,int)
                void setOrbit(char*,int)
                void getPositionVelocity(double,vector[double]&,vector[double]&)
                void setStateVector(int,double,vector[double]&,vector[double]&)
                void getStateVector(int,double&,vector[double]&,vector[double]&)
                int interpolateOrbit(double,vector[double]&,vector[double]&,int)
                int interpolateSCHOrbit(double,vector[double]&,vector[double]&)
                int interpolateWGS84Orbit(double,vector[double]&,vector[double]&)
                int interpolateLegendreOrbit(double,vector[double]&,vector[double]&)
                int computeAcceleration(double,vector[double]&)
                void orbitHermite(vector[vector[double]]&,vector[vector[double]]&,vector[double]&,vector[double]&,vector[double]&)
                void dumpToHDR(char*)
                void printOrbit()
            double firstlat,firstlon,deltalat,deltalon,major,eccentricitySquared
            double rspace,r0,peghdg,prf,t0,wvl,thresh
            uint64_t demAccessor,dopAccessor,slrngAccessor,latAccessor,lonAccessor
            uint64_t losAccessor,heightAccessor,incAccessor,maskAccessor
            int numiter,idemwidth,idemlength,ilrl,extraiter,length,width,Nrnglooks
            int Nazlooks,dem_method,orbit_method,orbit_nvecs,orbit_basis
            Orbit orb
            void createOrbit()
            void writeToFile(void**,double**,bool,bool,int,int,bool)
            void topo()
        Topo topo
        Topozero() except + # Just in case there are exceptions on creation, this allows them to be passed through Python
        void runTopo()
        void createOrbit()
        void setFirstLat(double)
        void setFirstLon(double)
        void setDeltaLat(double)
        void setDeltaLon(double)
        void setMajor(double)
        void setEccentricitySquared(double)
        void setRspace(double)
        void setR0(double)
        void setPegHdg(double)
        void setPrf(double)
        void setT0(double)
        void setWvl(double)
        void setThresh(double)
        void setDemAccessor(uint64_t)
        void setDopAccessor(uint64_t)
        void setSlrngAccessor(uint64_t)
        void setLatAccessor(uint64_t)
        void setLonAccessor(uint64_t)
        void setLosAccessor(uint64_t)
        void setHeightAccessor(uint64_t)
        void setIncAccessor(uint64_t)
        void setMaskAccessor(uint64_t)
        void setNumIter(int)
        void setIdemWidth(int)
        void setIdemLength(int)
        void setIlrl(int)
        void setExtraIter(int)
        void setLength(int)
        void setWidth(int)
        void setNrngLooks(int)
        void setNazLooks(int)
        void setDemMethod(int)
        void setOrbitMethod(int)
        void setOrbitNvecs(int)
        void setOrbitBasis(int)
        void setOrbitVector(int,double,double,double,double,double,double,double)

cdef class PyTopozero:
    cdef Topozero c_topozero
    def __cinit__(self):
        return
    def runTopo(self):
        self.c_topozero.runTopo()
    def createOrbit(self):
        self.c_topozero.createOrbit()
    def set_firstlat(self,double v):
        self.c_topozero.setFirstLat(v)
    def set_firstlon(self,double v):
        self.c_topozero.setFirstLon(v)
    def set_deltalat(self,double v):
        self.c_topozero.setDeltaLat(v)
    def set_deltalon(self,double v):
        self.c_topozero.setDeltaLon(v)
    def set_major(self,double v):
        self.c_topozero.setMajor(v)
    def set_eccentricitySquared(self,double v):
        self.c_topozero.setEccentricitySquared(v)
    def set_rSpace(self,double v):
        self.c_topozero.setRspace(v)
    def set_r0(self,double v):
        self.c_topozero.setR0(v)
    def set_pegHdg(self,double v):
        self.c_topozero.setPegHdg(v)
    def set_prf(self,double v):
        self.c_topozero.setPrf(v)
    def set_t0(self,double v):
        self.c_topozero.setT0(v)
    def set_wvl(self,double v):
        self.c_topozero.setWvl(v)
    def set_thresh(self,double v):
        self.c_topozero.setThresh(v)
    def set_demAccessor(self,uint64_t v):
        self.c_topozero.setDemAccessor(v)
    def set_dopAccessor(self,uint64_t v):
        self.c_topozero.setDopAccessor(v)
    def set_slrngAccessor(self,uint64_t v):
        self.c_topozero.setSlrngAccessor(v)
    def set_latAccessor(self,uint64_t v):
        self.c_topozero.setLatAccessor(v)
    def set_lonAccessor(self,uint64_t v):
        self.c_topozero.setLonAccessor(v)
    def set_losAccessor(self,uint64_t v):
        self.c_topozero.setLosAccessor(v)
    def set_heightAccessor(self,uint64_t v):
        self.c_topozero.setHeightAccessor(v)
    def set_incAccessor(self,uint64_t v):
        self.c_topozero.setIncAccessor(v)
    def set_maskAccessor(self,uint64_t v):
        self.c_topozero.setMaskAccessor(v)
    def set_numIter(self,int v):
        self.c_topozero.setNumIter(v)
    def set_idemWidth(self,int v):
        self.c_topozero.setIdemWidth(v)
    def set_idemLength(self,int v):
        self.c_topozero.setIdemLength(v)
    def set_ilrl(self,int v):
        self.c_topozero.setIlrl(v)
    def set_extraIter(self,int v):
        self.c_topozero.setExtraIter(v)
    def set_length(self,int v):
        self.c_topozero.setLength(v)
    def set_width(self,int v):
        self.c_topozero.setWidth(v)
    def set_nRngLooks(self,int v):
        self.c_topozero.setNrngLooks(v)
    def set_nAzLooks(self,int v):
        self.c_topozero.setNazLooks(v)
    def set_demMethod(self,int v):
        self.c_topozero.setDemMethod(v)
    def set_orbitMethod(self,int v):
        self.c_topozero.setOrbitMethod(v)
    def set_orbitNvecs(self,int v):
        self.c_topozero.setOrbitNvecs(v)
    def set_orbitBasis(self,int v):
        self.c_topozero.setOrbitBasis(v)
    def set_orbitVector(self,int idx, double t, double px, double py, double pz, double vx, double vy, double vz): 
        self.c_topozero.setOrbitVector(idx,t,px,py,pz,vx,vy,vz)

