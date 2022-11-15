CXX ?= g++
NVCC ?= nvcc

CUDA_ROOT ?= $(dir $(shell which $(NVCC)))..

LDFLAGS = -L$(CUDA_ROOT)/lib64 -L$(CUDA_ROOT)/lib64/stubs -lcuda -lcudart -lcufft -lgdal
CXXFLAGS = -std=c++11 -fPIC -shared -I$(CUDA_ROOT)/include
NVCCFLAGS = -std=c++11 -m64 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_60,code=sm_60 \
		-Xcompiler -fPIC -shared -Wno-deprecated-gpu-targets \
		-ftz=false -prec-div=true -prec-sqrt=true \
		-I/usr/include/gdal

CXXFLAGS += -O2 -DNDEBUG
NVCCFLAGS += -O2 -DNDEBUG

# pybind11 configuration
PYTHON ?= python3
PYTHON_CONFIG ?= python3-config
PYTHON_EXT_SUFFIX := $(shell "$(PYTHON_CONFIG)" --extension-suffix)
PYTHON_INCLUDES := $(shell "$(PYTHON)" -m pybind11 --includes) \
                   $(shell "$(PYTHON_CONFIG)" --includes)

DEPS = cudaUtil.h cudaError.h cuArrays.h GDALImage.h cuAmpcorParameter.h
OBJS =  GDALImage.o cuArrays.o cuArraysCopy.o cuArraysPadding.o cuOverSampler.o \
		cudaError.o cudaUtil.o \
		cuSincOverSampler.o cuDeramp.o cuOffset.o \
		cuCorrNormalization.o cuCorrNormalizationSAT.o cuCorrNormalizer.o \
		cuAmpcorParameter.o cuCorrTimeDomain.o cuCorrFrequency.o \
		cuAmpcorChunk.o cuAmpcorController.o cuEstimateStats.o

all: pyampcor

pyampcor: PyCuAmpcor$(PYTHON_EXT_SUFFIX)

PyCuAmpcor$(PYTHON_EXT_SUFFIX): PyCuAmpcor.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(PYTHON_INCLUDES) $^ -o $@

%.o: %.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -rf *.o *.so build *~
