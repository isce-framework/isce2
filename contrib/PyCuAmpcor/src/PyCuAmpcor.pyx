#
# PYX file to control Python module interface to underlying CUDA-Ampcor code
#
from libcpp.string cimport string
import numpy as np
cimport numpy as np


cdef extern from "cudaUtil.h":
    int gpuDeviceInit(int)
    void gpuDeviceList()
    int gpuGetMaxGflopsDeviceId()

def listGPU():
    gpuDeviceList()

def findGPU():
    return gpuGetMaxGflopsDeviceId()

def setGPU(int id):
    return gpuDeviceInit(id)


cdef extern from "cuAmpcorParameter.h":
    cdef cppclass cuAmpcorParameter:
        cuAmpcorParameter() except +
        int algorithm      					## Cross-correlation algorithm: 0=freq domain 1=time domain
        int deviceID       					## Targeted GPU device ID: use -1 to auto select
        int nStreams       					## Number of streams to asynchonize data transfers and compute kernels
        int derampMethod   					## Method for deramping 0=None, 1=average, 2=phase gradient

        ## chip or window size for raw data
        int windowSizeHeightRaw         	## Template window height (original size)
        int windowSizeWidthRaw          	## Template window width (original size)
        int searchWindowSizeHeightRaw   	## Search window height (original size)
        int searchWindowSizeWidthRaw    	## Search window width (orignal size)
        int halfSearchRangeDownRaw    		##(searchWindowSizeHeightRaw-windowSizeHeightRaw)/2
        int halfSearchRangeAcrossRaw     	##(searchWindowSizeWidthRaw-windowSizeWidthRaw)/2
        ## chip or window size after oversampling
        int rawDataOversamplingFactor   	## Raw data overampling factor (from original size to oversampled size)

        ## strides between chips/windows
        int skipSampleDownRaw    			## Skip size between neighboring windows in Down direction (original size)
        int skipSampleAcrossRaw  			## Skip size between neighboring windows in across direction (original size)

        ## Zoom in region near location of max correlation
        int zoomWindowSize       			## Zoom-in window size in correlation surface (same for down and across directions)
        int oversamplingFactor   			## Oversampling factor for interpolating correlation surface
        int oversamplingMethod

        float thresholdSNR       			## Threshold of Signal noise ratio to remove noisy data

        ##reference image
        string referenceImageName     			## reference SLC image name
        int imageDataType1              	## reference image data type, 2=cfloat=complex=float2 1=float
        int referenceImageHeight           	## reference image height
        int referenceImageWidth            	## reference image width

        ##secondary image
        string secondaryImageName      			## secondary SLC image name
        int imageDataType2              	## secondary image data type, 2=cfloat=complex=float2 1=float
        int secondaryImageHeight            	## secondary image height
        int secondaryImageWidth            		## secondary image width

        int useMmap                         ## whether to use mmap
        int mmapSizeInGB                    ## mmap buffer size in unit of Gigabytes (if not mmmap, the buffer size)

        ## total number of chips/windows
        int numberWindowDown            	## number of total windows (down)
        int numberWindowAcross          	## number of total windows (across)
        int numberWindows  					## numberWindowDown*numberWindowAcross

        ## number of chips/windows in a batch/chunk
        int numberWindowDownInChunk     	## number of windows processed in a chunk (down)
        int numberWindowAcrossInChunk   	## number of windows processed in a chunk (across)
        int numberWindowsInChunk  			## numberWindowDownInChunk*numberWindowAcrossInChunk
        int numberChunkDown             	## number of chunks (down)
        int numberChunkAcross           	## number of chunks (across)
        int numberChunks

        int *referenceStartPixelDown   		## reference starting pixels for each window (down)
        int *referenceStartPixelAcross 		## reference starting pixels for each window (across)
        int *secondaryStartPixelDown    		## secondary starting pixels for each window (down)
        int *secondaryStartPixelAcross  		## secondary starting pixels for each window (across)
        int *grossOffsetDown 				## Gross offsets between reference and secondary windows (down) : secondaryStartPixel - referenceStartPixel
        int *grossOffsetAcross      		## Gross offsets between reference and secondary windows (across)
        int grossOffsetDown0				## constant gross offset (down)
        int grossOffsetAcross0				## constant gross offset (across)
        int referenceStartPixelDown0           ## the first pixel of reference image (down), be adjusted with margins and gross offset
        int referenceStartPixelAcross0         ## the first pixel of reference image (across)
        int *referenceChunkStartPixelDown 		## array of starting pixels for all reference chunks (down)
        int *referenceChunkStartPixelAcross 	## array of starting pixels for all reference chunks (across)
        int *secondaryChunkStartPixelDown 		## array of starting pixels for all secondary chunks (down)
        int *secondaryChunkStartPixelAcross 	## array of starting pixels for all secondary chunks (across)
        int *referenceChunkHeight 				## array of heights of all reference chunks, required when loading chunk to GPU
        int *referenceChunkWidth 				## array of width of all reference chunks
        int *secondaryChunkHeight 				## array of width of all reference chunks
        int *secondaryChunkWidth 				## array of width of all secondary chunks
        int maxReferenceChunkHeight 			## max height for all reference/secondary chunks, determine the size of reading cache in GPU
        int maxReferenceChunkWidth 			## max width for all reference chunks, determine the size of reading cache in GPU
        int maxSecondaryChunkHeight
        int maxSecondaryChunkWidth

        string grossOffsetImageName
        string offsetImageName     ## Output Offset fields filename
        string snrImageName        ## Output SNR filename
        string covImageName        ## Output COV filename
        void setStartPixels(int*, int*, int*, int*)
        void setStartPixels(int, int, int*, int*)
        void setStartPixels(int, int, int, int)
        void checkPixelInImageRange()  ## check whether

        void setupParameters()      ## Process other parameters after Python Inpu

cdef extern from "cuAmpcorController.h":
    cdef cppclass cuAmpcorController:
        cuAmpcorController() except +
        cuAmpcorParameter *param
        void runAmpcor()

cdef class PyCuAmpcor(object):
    '''
    Python interface for cuda Ampcor
    '''
    cdef cuAmpcorController c_cuAmpcor
    def __cinit__(self):
        return

    @property
    def algorithm(self):
        return self.c_cuAmpcor.param.algorithm
    @algorithm.setter
    def algorithm(self, int a):
        self.c_cuAmpcor.param.algorithm = a
    @property
    def deviceID(self):
        return self.c_cuAmpcor.param.deviceID
    @deviceID.setter
    def deviceID(self, int a):
        self.c_cuAmpcor.param.deviceID = a
    @property
    def nStreams(self):
        return self.c_cuAmpcor.param.nStreams
    @nStreams.setter
    def nStreams(self, int a):
        self.c_cuAmpcor.param.nStreams = a
    @property
    def useMmap(self):
        return self.c_cuAmpcor.param.useMmap
    @useMmap.setter
    def useMmap(self, int a):
        self.c_cuAmpcor.param.useMmap = a
    @property
    def mmapSize(self):
        return self.c_cuAmpcor.param.mmapSizeInGB
    @mmapSize.setter
    def mmapSize(self, int a):
        self.c_cuAmpcor.param.mmapSizeInGB = a
    @property
    def derampMethod(self):
        return self.c_cuAmpcor.param.derampMethod
    @derampMethod.setter
    def derampMethod(self, int a):
        self.c_cuAmpcor.param.derampMethod = a
    @property
    def windowSizeHeight(self):
        return self.c_cuAmpcor.param.windowSizeHeightRaw
    @windowSizeHeight.setter
    def windowSizeHeight(self, int a):
        self.c_cuAmpcor.param.windowSizeHeightRaw = a
    @property
    def windowSizeWidth(self):
        return self.c_cuAmpcor.param.windowSizeWidthRaw
    @windowSizeWidth.setter
    def windowSizeWidth(self, int a):
        self.c_cuAmpcor.param.windowSizeWidthRaw = a
    @property
    def halfSearchRangeDown(self):
        """half of the search range"""
        return self.c_cuAmpcor.param.halfSearchRangeDownRaw
    @halfSearchRangeDown.setter
    def halfSearchRangeDown(self, int a):
        """set half of the search range"""
        self.c_cuAmpcor.param.halfSearchRangeDownRaw = a
    @property
    def halfSearchRangeAcross(self):
        """half of the search range"""
        return self.c_cuAmpcor.param.halfSearchRangeAcrossRaw
    @halfSearchRangeAcross.setter
    def halfSearchRangeAcross(self, int a):
        """set half of the search range"""
        self.c_cuAmpcor.param.halfSearchRangeAcrossRaw = a
    @property
    def searchWindowSizeHeight(self):
        return self.c_cuAmpcor.param.searchWindowSizeHeightRaw
    @property
    def searchWindowSizeWidth(self):
        return self.c_cuAmpcor.param.searchWindowSizeWidthRaw
    @property
    def skipSampleDown(self):
        return self.c_cuAmpcor.param.skipSampleDownRaw
    @skipSampleDown.setter
    def skipSampleDown(self, int a):
        self.c_cuAmpcor.param.skipSampleDownRaw = a
    @property
    def skipSampleAcross(self):
        return self.c_cuAmpcor.param.skipSampleAcrossRaw
    @skipSampleAcross.setter
    def skipSampleAcross(self, int a):
        self.c_cuAmpcor.param.skipSampleAcrossRaw = a

    @property
    def rawDataOversamplingFactor(self):
        """anti-aliasing oversampling factor"""
        return self.c_cuAmpcor.param.rawDataOversamplingFactor
    @rawDataOversamplingFactor.setter
    def rawDataOversamplingFactor(self, int a):
        self.c_cuAmpcor.param.rawDataOversamplingFactor = a
    @property
    def corrSurfaceZoomInWindow(self):
        """Zoom-In Window Size for correlation surface"""
        return self.c_cuAmpcor.param.zoomWindowSize
    @corrSurfaceZoomInWindow.setter
    def corrSurfaceZoomInWindow(self, int a):
        self.c_cuAmpcor.param.zoomWindowSize = a
    @property
    def corrSurfaceOverSamplingFactor(self):
        """Oversampling factor for correlation surface"""
        return self.c_cuAmpcor.param.oversamplingFactor
    @corrSurfaceOverSamplingFactor.setter
    def corrSurfaceOverSamplingFactor(self, int a):
        self.c_cuAmpcor.param.oversamplingFactor = a
    @property
    def corrSufaceOverSamplingMethod(self):
        """Oversampling method for correlation surface(0=fft,1=sinc)"""
        return self.c_cuAmpcor.param.oversamplingMethod
    @corrSufaceOverSamplingMethod.setter
    def corrSufaceOverSamplingMethod(self, int a):
        self.c_cuAmpcor.param.oversamplingMethod = a
    @property
    def referenceImageName(self):
        return self.c_cuAmpcor.param.referenceImageName
    @referenceImageName.setter
    def referenceImageName(self, str a):
        self.c_cuAmpcor.param.referenceImageName = <string> a.encode()
    @property
    def secondaryImageName(self):
        return self.c_cuAmpcor.param.secondaryImageName
    @secondaryImageName.setter
    def secondaryImageName(self, str a):
        self.c_cuAmpcor.param.secondaryImageName = <string> a.encode()
    @property
    def referenceImageName(self):
        return self.c_cuAmpcor.param.referenceImageName
    @referenceImageName.setter
    def referenceImageName(self, str a):
        self.c_cuAmpcor.param.referenceImageName = <string> a.encode()
    @property
    def referenceImageHeight(self):
        return self.c_cuAmpcor.param.referenceImageHeight
    @referenceImageHeight.setter
    def referenceImageHeight(self, int a):
        self.c_cuAmpcor.param.referenceImageHeight=a
    @property
    def referenceImageWidth(self):
        return self.c_cuAmpcor.param.referenceImageWidth
    @referenceImageWidth.setter
    def referenceImageWidth(self, int a):
        self.c_cuAmpcor.param.referenceImageWidth=a
    @property
    def secondaryImageHeight(self):
        return self.c_cuAmpcor.param.secondaryImageHeight
    @secondaryImageHeight.setter
    def secondaryImageHeight(self, int a):
        self.c_cuAmpcor.param.secondaryImageHeight=a
    @property
    def secondaryImageWidth(self):
        return self.c_cuAmpcor.param.secondaryImageWidth
    @secondaryImageWidth.setter
    def secondaryImageWidth(self, int a):
        self.c_cuAmpcor.param.secondaryImageWidth=a

    @property
    def numberWindowDown(self):
        return self.c_cuAmpcor.param.numberWindowDown
    @numberWindowDown.setter
    def numberWindowDown(self, int a):
        self.c_cuAmpcor.param.numberWindowDown = a
    @property
    def numberWindowAcross(self):
        return self.c_cuAmpcor.param.numberWindowAcross
    @numberWindowAcross.setter
    def numberWindowAcross(self, int a):
        self.c_cuAmpcor.param.numberWindowAcross = a
    @property
    def numberWindows(self):
        return  self.c_cuAmpcor.param.numberWindows

    @property
    def numberWindowDownInChunk(self):
        return  self.c_cuAmpcor.param.numberWindowDownInChunk
    @numberWindowDownInChunk.setter
    def numberWindowDownInChunk(self, int a):
        self.c_cuAmpcor.param.numberWindowDownInChunk = a
    @property
    def numberWindowAcrossInChunk(self):
        return  self.c_cuAmpcor.param.numberWindowAcrossInChunk
    @numberWindowAcrossInChunk.setter
    def numberWindowAcrossInChunk(self, int a):
        self.c_cuAmpcor.param.numberWindowAcrossInChunk = a
    @property
    def numberChunkDown(self):
        return  self.c_cuAmpcor.param.numberChunkDown
    @property
    def numberChunkAcross(self):
        return  self.c_cuAmpcor.param.numberChunkAcross
    @property
    def numberChunks(self):
        return  self.c_cuAmpcor.param.numberChunks


    ## gross offets
    @property
    def grossOffsetImageName(self):
        return self.c_cuAmpcor.param.grossOffsetImageName
    @grossOffsetImageName.setter
    def grossOffsetImageName(self, str a):
        self.c_cuAmpcor.param.grossOffsetImageName = <string> a.encode()
    @property
    def offsetImageName(self):
        return self.c_cuAmpcor.param.offsetImageName
    @offsetImageName.setter
    def offsetImageName(self, str a):
        self.c_cuAmpcor.param.offsetImageName = <string> a.encode()

    @property
    def snrImageName(self):
        return self.c_cuAmpcor.param.snrImageName
    @snrImageName.setter
    def snrImageName(self, str a):
        self.c_cuAmpcor.param.snrImageName = <string> a.encode()

    @property
    def covImageName(self):
        return self.c_cuAmpcor.param.covImageName
    @covImageName.setter
    def covImageName(self, str a):
        self.c_cuAmpcor.param.covImageName = <string> a.encode()

    @property
    def referenceStartPixelDownStatic(self):
        return self.c_cuAmpcor.param.referenceStartPixelDown0
    @referenceStartPixelDownStatic.setter
    def referenceStartPixelDownStatic(self, int a):
        self.c_cuAmpcor.param.referenceStartPixelDown0 = a
    @property
    def referenceStartPixelAcrossStatic(self):
        return self.c_cuAmpcor.param.referenceStartPixelAcross0
    @referenceStartPixelAcrossStatic.setter
    def referenceStartPixelAcrossStatic(self, int a):
        self.c_cuAmpcor.param.referenceStartPixelAcross0 = a
    @property
    def grossOffsetDownStatic(self):
        return self.c_cuAmpcor.param.grossOffsetDown0
    @grossOffsetDownStatic.setter
    def grossOffsetDownStatic(self, int a):
        self.c_cuAmpcor.param.grossOffsetDown0 =a
    @property
    def grossOffsetAcrossStatic(self):
        return self.c_cuAmpcor.param.grossOffsetAcross0
    @grossOffsetAcrossStatic.setter
    def grossOffsetAcrossStatic(self, int a):
        self.c_cuAmpcor.param.grossOffsetAcross0 =a

    @property
    def grossOffsetDownDynamic(self):
        cdef int *c_data
        c_data = self.c_cuAmpcor.param.grossOffsetDown
        p_data = np.zeros(self.numberWindows, dtype = np.float32)
        for i in range (self.numberWindows):
            p_data[i] = c_data[i]
        return p_data
    @grossOffsetDownDynamic.setter
    def grossOffsetDownDynamic (self, np.ndarray[np.int32_t,ndim=1,mode="c"] pa):
        cdef int *c_data
        cdef int *p_data
        c_data = self.c_cuAmpcor.param.grossOffsetDown
        p_data = <int *> pa.data
        for i in range (self.numberWindows):
            c_data[i] = p_data[i]
    @property
    def grossOffsetAcrossDynamic(self):
        cdef int *c_data
        c_data = self.c_cuAmpcor.param.grossOffsetAcross
        p_data = np.zeros(self.numberWindows, dtype = np.float32)
        for i in range (self.numberWindows):
            p_data[i] = c_data[i]
        return p_data
    @grossOffsetAcrossDynamic.setter
    def grossOffsetAcrossDynamic (self, np.ndarray[np.int32_t,ndim=1,mode="c"] pa):
        cdef int *c_data
        cdef int *p_data
        c_data = self.c_cuAmpcor.param.grossOffsetAcross
        p_data = <int *> pa.data
        for i in range (self.numberWindows):
            c_data[i] = p_data[i]
        return


    def setConstantGrossOffset(self, int goDown, int goAcross):
        """
        constant gross offsets
        param goDown gross offset in azimuth direction
        param goAcross gross offset in range direction
        """
        self.c_cuAmpcor.param.setStartPixels(<int>self.referenceStartPixelDownStatic, <int>self.referenceStartPixelAcrossStatic, goDown, goAcross)

    def setVaryingGrossOffset(self, np.ndarray[np.int32_t,ndim=1,mode="c"] vD, np.ndarray[np.int32_t,ndim=1,mode="c"] vA):
        """
        varying gross offsets for each window
        param vD numpy 1d array of size numberWindows, gross offsets in azimuth direction
        param vA numpy 1d array of size numberWindows, gross offsets in azimuth direction
        static part should be included
        """
        self.c_cuAmpcor.param.setStartPixels(<int>self.referenceStartPixelDownStatic, <int>self.referenceStartPixelAcrossStatic, <int *> vD.data, <int *> vA.data)

    def checkPixelInImageRange(self):
        """ check whether each window is with image range """
        self.c_cuAmpcor.param.checkPixelInImageRange()

    def setupParams(self):
        """
        set up constant parameters and allocate array parameters (offsets)
        should be called after number of windows is set and before setting varying gross offsets
        """
        self.c_cuAmpcor.param.setupParameters()

    def runAmpcor(self):
        """ main procedure to run ampcor """
        self.c_cuAmpcor.runAmpcor()







