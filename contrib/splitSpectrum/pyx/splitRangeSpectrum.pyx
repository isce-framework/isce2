#cython: language_level=3
#
# Author: Heresh Fattahi
# Copyright 2017
#

from libcpp.string cimport string

cdef extern from "splitRangeSpectrum.h" namespace "splitSpectrum":
    cdef cppclass splitRangeSpectrum:
        splitRangeSpectrum() except +
        string inputDS
        string lbDS
        string hbDS
        int blocksize
        int memsize
        float rangeSamplingRate
        double lowBandWidth
        double highBandWidth
        double lowCenterFrequency
        double highCenterFrequency
        int split_spectrum_process()


cdef class PySplitRangeSpectrum:
    '''
    Python wrapper for splitRangeSpectrum
    '''
    cdef splitRangeSpectrum thisptr

    def __cinit__(self):
        return

    @property
    def inputDS(self):
        return self.thisptr.inputDS.decode('utf-8')

    @inputDS.setter
    def inputDS(self,x):
        self.thisptr.inputDS = x.encode('utf-8')

    @property
    def lbDS(self):
        return self.thisptr.lbDS.decode('utf-8')

    @lbDS.setter
    def lbDS(self,x):
        self.thisptr.lbDS = x.encode('utf-8')

    @property
    def hbDS(self):
        return self.thisptr.hbDS.decode('utf-8')

    @hbDS.setter
    def hbDS(self,x):
        self.thisptr.hbDS = x.encode('utf-8')

    @property
    def memsize(self):
        return self.thisptr.memsize

    @memsize.setter
    def memsize(self,x):
        self.thisptr.memsize = x

    @property
    def blocksize(self):
        return self.thisptr.blocksize

    @blocksize.setter
    def blocksize(self,x):
        self.thisptr.blocksize = x 

    @property
    def rangeSamplingRate(self):
        return self.thisptr.rangeSamplingRate

    @rangeSamplingRate.setter
    def rangeSamplingRate(self,x):
        self.thisptr.rangeSamplingRate = x 

    @property
    def lowBandWidth(self):
        return self.thisptr.lowBandWidth

    @lowBandWidth.setter
    def lowBandWidth(self,x):
        self.thisptr.lowBandWidth = x

    @property
    def highBandWidth(self):
        return self.thisptr.highBandWidth

    @highBandWidth.setter
    def highBandWidth(self,x):
        self.thisptr.highBandWidth = x

    @property
    def lowCenterFrequency(self):
        return self.thisptr.lowCenterFrequency

    @lowCenterFrequency.setter
    def lowCenterFrequency(self,x):
        self.thisptr.lowCenterFrequency = x

    @property
    def highCenterFrequency(self):
        return self.thisptr.highCenterFrequency

    @highCenterFrequency.setter
    def highCenterFrequency(self,x):
        self.thisptr.highCenterFrequency = x


    def split(self):
        return self.thisptr.split_spectrum_process()
