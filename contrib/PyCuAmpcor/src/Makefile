PROJECT = CUAMPCOR

LDFLAGS =  -lcuda -lcudart -lcufft -lcublas
CXXFLAGS = -std=c++11 -fpermissive -fPIC -shared
NVCCFLAGS = -ccbin g++ -m64 \
    -gencode arch=compute_35,code=sm_35 \
	-gencode arch=compute_60,code=sm_60 \
    -Xcompiler -fPIC -shared -Wno-deprecated-gpu-targets \
    -ftz=false -prec-div=true -prec-sqrt=true

CXX=g++
NVCC=nvcc

DEPS = cudaUtil.h cudaError.h cuArrays.h GDALImage.h cuAmpcorParameter.h
OBJS =  GDALImage.o cuArrays.o cuArraysCopy.o cuArraysPadding.o cuOverSampler.o \
	    cuSincOverSampler.o cuDeramp.o cuOffset.o \
        cuCorrNormalization.o cuAmpcorParameter.o cuCorrTimeDomain.o cuCorrFrequency.o \
        cuAmpcorChunk.o cuAmpcorController.o cuEstimateStats.o

all: pyampcor

GDALImage.o: GDALImage.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) -c -o $@ GDALImage.cu

cuArrays.o: cuArrays.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) -c -o $@ cuArrays.cu

cuArraysCopy.o: cuArraysCopy.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) -c -o $@ cuArraysCopy.cu

cuArraysPadding.o: cuArraysPadding.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) -c -o $@ cuArraysPadding.cu

cuSincOverSampler.o: cuSincOverSampler.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) -c -o $@ cuSincOverSampler.cu

cuOverSampler.o: cuOverSampler.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) -c -o $@ cuOverSampler.cu

cuDeramp.o: cuDeramp.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) -c -o $@ cuDeramp.cu

cuOffset.o: cuOffset.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) -c -o $@ cuOffset.cu

cuCorrNormalization.o: cuCorrNormalization.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) -c -o $@ cuCorrNormalization.cu

cuAmpcorParameter.o: cuAmpcorParameter.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ cuAmpcorParameter.cu

cuCorrTimeDomain.o: cuCorrTimeDomain.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) -c -o $@ cuCorrTimeDomain.cu

cuCorrFrequency.o: cuCorrFrequency.cu $(DEPS) cuCorrFrequency.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ cuCorrFrequency.cu

cuAmpcorChunk.o: cuAmpcorChunk.cu cuAmpcorUtil.h $(DEPS)
	$(NVCC) $(NVCCFLAGS) -c -o $@ cuAmpcorChunk.cu

cuAmpcorController.o: cuAmpcorController.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ cuAmpcorController.cu

cuEstimateStats.o: cuEstimateStats.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ cuEstimateStats.cu


pyampcor: $(OBJS)
	rm -f PyCuAmpcor.cpp && python3 setup.py build_ext --inplace

clean:
	rm -rf *.o *so build *~  PyCuAmpcor.cpp ctest *.dat
