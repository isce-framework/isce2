/*
 * maxlocation.cu
 * Purpose: find the location of maximum for a batch of images/vectors
 *          this uses the reduction algorithm similar to summations  
 *  
 * Author : Lijun Zhu
 *          Seismo Lab, Caltech
 * Version 1.0 10/01/16  
*/ 
	
#include "cuAmpcorUtil.h"
#include <cfloat>

/*
__device__ float atomicMaxf(float* address, float val)
{
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
        }
    return __int_as_float(old);
}*/


// comapre two elements
inline static __device__ void maxPairReduce(volatile float* maxval, volatile int* maxloc, 
      size_t gid, size_t strideid)
{
	if(maxval[gid] < maxval[strideid]) {
		maxval[gid] = maxval[strideid];
		maxloc[gid] = maxloc[strideid];
	}
}   

// max reduction kernel, save the results to shared memory 
template<const int BLOCKSIZE>
__device__ void max_reduction(const float* const images, 
    const size_t imageSize,
    const size_t nImages, 
    volatile float* shval, 
    volatile int* shloc)
{
	int tid = threadIdx.x;
    shval[tid] = -FLT_MAX; 
	int imageStart = blockIdx.x*imageSize;
	int imagePixel;

	// reduction for elements with i, i+BLOCKSIZE, i+2*BLOCKSIZE ... 
	// 
	for(int gid = tid; gid < imageSize; gid+=blockDim.x)
	{
		imagePixel = imageStart+gid;
		if(shval[tid] < images[imagePixel]) {
			shval[tid] = images[imagePixel];
			shloc[tid] = gid;
		}
	}
    __syncthreads();
    
    //reduction within a block
    if (BLOCKSIZE >=1024){ if (tid < 512) { maxPairReduce(shval, shloc, tid, tid + 512); } __syncthreads(); }
    if (BLOCKSIZE >=512) { if (tid < 256) { maxPairReduce(shval, shloc, tid, tid + 256); } __syncthreads(); }
    if (BLOCKSIZE >=256) { if (tid < 128) { maxPairReduce(shval, shloc, tid, tid + 128); } __syncthreads(); }
    if (BLOCKSIZE >=128) { if (tid < 64 ) { maxPairReduce(shval, shloc, tid, tid + 64 ); } __syncthreads(); }
    //reduction within a warp 
    if (tid < 32)
    {
		maxPairReduce(shval, shloc, tid, tid + 32);
		maxPairReduce(shval, shloc, tid, tid + 16);
		maxPairReduce(shval, shloc, tid, tid +  8);
		maxPairReduce(shval, shloc, tid, tid +  4);
		maxPairReduce(shval, shloc, tid, tid +  2);
		maxPairReduce(shval, shloc, tid, tid +  1);
    }
    __syncthreads();
}

//kernel and function for 1D array, find both max value and location
template <const int BLOCKSIZE>
__global__ void  cuMaxValLoc_kernel( const float* const images, float *maxval, int* maxloc, const size_t imageSize, const size_t nImages)
{
    __shared__ float shval[BLOCKSIZE];
    __shared__ int shloc[BLOCKSIZE];    
    int bid = blockIdx.x; 
    if(bid >= nImages) return;
    
    max_reduction<BLOCKSIZE>(images, imageSize, nImages, shval, shloc);
    
    if (threadIdx.x == 0) {
        maxloc[bid] = shloc[0];
        maxval[bid] = shval[0];
    }      
}

void cuArraysMaxValandLoc(cuArrays<float> *images, cuArrays<float> *maxval, cuArrays<int> *maxloc, cudaStream_t stream)
{
    const size_t imageSize = images->size;
    const size_t nImages = images->count; 
    dim3 threadsperblock(NTHREADS);
    dim3 blockspergrid(nImages);
    cuMaxValLoc_kernel<NTHREADS><<<blockspergrid, threadsperblock, 0, stream>>>
        (images->devData, maxval->devData, maxloc->devData, imageSize, nImages);
    getLastCudaError("cudaKernel fine max location error\n");
}

//kernel and function for 1D array, find max location only 
template <const int BLOCKSIZE>
__global__ void  cudaKernel_maxloc(const float* const images, int* maxloc,
                                   const size_t imageSize, const size_t nImages)
{
    __shared__ float shval[BLOCKSIZE];
    __shared__ int shloc[BLOCKSIZE];
    
    int bid = blockIdx.x; 
    if(bid >=nImages) return;
    
    max_reduction<BLOCKSIZE>(images, imageSize, nImages, shval, shloc);
    
    if (threadIdx.x == 0) {
        maxloc[bid] = shloc[0];
    }
}

void cuArraysMaxLoc(cuArrays<float> *images, cuArrays<int> *maxloc, cudaStream_t stream) 
{
    int imageSize = images->size;
    int nImages = maxloc->size;
    
    cudaKernel_maxloc<NTHREADS><<<nImages, NTHREADS,0, stream>>>
        (images->devData, maxloc->devData, imageSize, nImages);
    getLastCudaError("cudaKernel find max location 1D error\n");
}

//kernel and function for 2D array(image), find max location only
template <const int BLOCKSIZE>
__global__ void  cudaKernel_maxloc2D(const float* const images, int2* maxloc, float* maxval, const size_t imageNX, const size_t imageNY, const size_t nImages)
{
    __shared__ float shval[BLOCKSIZE];
    __shared__ int shloc[BLOCKSIZE];
    
    int bid = blockIdx.x; 
    if(bid >= nImages) return;
    
    const int imageSize = imageNX * imageNY;
    max_reduction<BLOCKSIZE>(images, imageSize, nImages, shval, shloc);
    
    if (threadIdx.x == 0) {
        maxloc[bid] = make_int2(shloc[0]/imageNY, shloc[0]%imageNY); 
        maxval[bid] = shval[0];
    }
}

void cuArraysMaxloc2D(cuArrays<float> *images, cuArrays<int2> *maxloc,
                      cuArrays<float> *maxval, cudaStream_t stream) 
{
    cudaKernel_maxloc2D<NTHREADS><<<images->count, NTHREADS, 0, stream>>>
        (images->devData, maxloc->devData, maxval->devData, images->height, images->width, images->count);
    getLastCudaError("cudaKernel find max location 2D error\n");
}

//kernel and function for 2D array(image), find max location only, use overload
template <const int BLOCKSIZE>
__global__ void  cudaKernel_maxloc2D(const float* const images, int2* maxloc, const size_t imageNX, const size_t imageNY, const size_t nImages)
{
    __shared__ float shval[BLOCKSIZE];
    __shared__ int shloc[BLOCKSIZE];
    
    int bid = blockIdx.x; 
    if(bid >= nImages) return;
        
    const int imageSize = imageNX * imageNY;
    max_reduction<BLOCKSIZE>(images, imageSize, nImages, shval, shloc);
    
    if (threadIdx.x == 0) {
        int xloc = shloc[0]/imageNY;
        int yloc = shloc[0]%imageNY;
        maxloc[bid] = make_int2(xloc, yloc); 
    }
}

void cuArraysMaxloc2D(cuArrays<float> *images, cuArrays<int2> *maxloc, cudaStream_t stream) 
{
    cudaKernel_maxloc2D<NTHREADS><<<images->count, NTHREADS, 0, stream>>>
        (images->devData, maxloc->devData, images->height, images->width, images->count);
    getLastCudaError("cudaKernel find max location 2D error\n");
}




//determine final offset values
__global__ void cuSubPixelOffset_kernel(const int2 *offsetInit, const int2 *offsetZoomIn,
                                        float2 *offsetFinal,
                                        const float OSratio,
                                        const float xoffset, const float yoffset, const int size)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= size) return;
    offsetFinal[idx].x = OSratio*(offsetZoomIn[idx].x ) + offsetInit[idx].x  - xoffset;
    offsetFinal[idx].y = OSratio*(offsetZoomIn[idx].y ) + offsetInit[idx].y - yoffset;
}  


/// determine the final offset value
/// @param[in] 

void cuSubPixelOffset(cuArrays<int2> *offsetInit, cuArrays<int2> *offsetZoomIn, cuArrays<float2> *offsetFinal, 
                      int OverSampleRatioZoomin, int OverSampleRatioRaw,
                      int xHalfRangeInit,  int yHalfRangeInit, 
                      int xHalfRangeZoomIn, int yHalfRangeZoomIn,
                      cudaStream_t stream)
{
    int size = offsetInit->getSize();
    float OSratio = 1.0f/(float)(OverSampleRatioZoomin*OverSampleRatioRaw);
    float xoffset = xHalfRangeInit ;
    float yoffset = yHalfRangeInit ;
    //std::cout << "subpixel" << xoffset << " " << yoffset << " ratio " << OSratio << std::endl;
    
    cuSubPixelOffset_kernel<<<IDIVUP(size, NTHREADS), NTHREADS, 0, stream>>>
        (offsetInit->devData, offsetZoomIn->devData, 
         offsetFinal->devData, OSratio, xoffset, yoffset, size);
    getLastCudaError("cuSubPixelOffset_kernel");
    //offsetInit->debuginfo(stream);
    //offsetZoomIn->debuginfo(stream);
    
}

static inline __device__ int dev_padStart(const size_t padDim, const size_t imageDim, const size_t maxloc)
{
    int halfPadSize = padDim/2;
    int start = maxloc - halfPadSize;
    if(start <0) start =0;
    else if(maxloc > imageDim-halfPadSize-1) start = imageDim-padDim-1;
    return start;
}
	 	
//cuda kernel for cuda_determineInterpZone
__global__ void cudaKernel_determineInterpZone(const int2* maxloc, const size_t nImages,
                                               const size_t imageNX,  const size_t imageNY, 
                                               const size_t padNX, const size_t padNY,  int2* padOffset)
{
    int imageIndex = threadIdx.x + blockDim.x *blockIdx.x; //image index
    if (imageIndex < nImages) {
        padOffset[imageIndex].x = dev_padStart(padNX, imageNX, maxloc[imageIndex].x);
        padOffset[imageIndex].y = dev_padStart(padNY, imageNY, maxloc[imageIndex].y);
    }
} 

/*
 * determine the interpolation area (pad) from the max location and the padSize
 *    the pad will be (maxloc-padSize/2, maxloc+padSize/2-1)  
 * @param[in] maxloc[nImages]   
 * @param[in] padSize   
 * @param[in] imageSize 
 * @param[in] nImages
 * @param[out] padStart[nImages] return values of maxloc-padSize/2
 */
void cuDetermineInterpZone(cuArrays<int2> *maxloc, cuArrays<int2> *zoomInOffset, cuArrays<float> *corrOrig, cuArrays<float> *corrZoomIn, cudaStream_t stream) 
{
	int threadsperblock=NTHREADS;
	int blockspergrid=IDIVUP(corrOrig->count, threadsperblock);
	cudaKernel_determineInterpZone<<<blockspergrid, threadsperblock, 0, stream>>>
	    (maxloc->devData, maxloc->size, corrOrig->height, corrOrig->width, corrZoomIn->height, corrZoomIn->width, zoomInOffset->devData);
}


static inline __device__ int dev_adjustOffset(const size_t newRange, const size_t oldRange, const size_t maxloc)
{
    int maxloc_cor = maxloc;
    if(maxloc_cor < newRange) {maxloc_cor = oldRange;}
    else if(maxloc_cor > 2*oldRange-newRange) {maxloc_cor = oldRange;} 
    int start = maxloc_cor - newRange;
    return start;
}

__global__ void cudaKernel_determineSecondaryExtractOffset(int2 * maxloc, 
    const size_t nImages, int xOldRange, int yOldRange, int xNewRange, int yNewRange)
{
    int imageIndex = threadIdx.x + blockDim.x *blockIdx.x; //image index
	if (imageIndex < nImages) 
	{
        maxloc[imageIndex].x = dev_adjustOffset(xNewRange, xOldRange, maxloc[imageIndex].x);
        maxloc[imageIndex].y = dev_adjustOffset(yNewRange, yOldRange, maxloc[imageIndex].y);
	}
}

///@param[in] xOldRange, yOldRange are (half) search ranges in first step
///@param[in] x
void cuDetermineSecondaryExtractOffset(cuArrays<int2> *maxLoc, 
    int xOldRange, int yOldRange, int xNewRange, int yNewRange, cudaStream_t stream) 
{
	int threadsperblock=NTHREADS;
	int blockspergrid=IDIVUP(maxLoc->size, threadsperblock);
	cudaKernel_determineSecondaryExtractOffset<<<blockspergrid, threadsperblock, 0, stream>>>
	    (maxLoc->devData, maxLoc->size, xOldRange, yOldRange, xNewRange, yNewRange);
}




__global__ void cudaKernel_maxlocPlusZoominOffset(float *offset, const int * padStart, const int * maxlocUpSample, 
        const size_t nImages, float zoomInRatioX, float zoomInRatioY)
{
	int imageIndex = threadIdx.x + blockDim.x *blockIdx.x; //image index
	if (imageIndex < nImages) 
	{
		int index=2*imageIndex;
		offset[index] = padStart[index] + maxlocUpSample[index] * zoomInRatioX;
		index++;
		offset[index] = padStart[index] + maxlocUpSample[index] * zoomInRatioY;
	}
} 

void cuda_maxlocPlusZoominOffset(float *offset, const int * padStart, const int * maxlocUpSample, 
        const size_t nImages, float zoomInRatioX, float zoomInRatioY)
{
	int threadsperblock=NTHREADS;
	int blockspergrid = IDIVUP(nImages, threadsperblock);
	cudaKernel_maxlocPlusZoominOffset<<<blockspergrid,threadsperblock>>>(offset, padStart, maxlocUpSample, 
        nImages, zoomInRatioX, zoomInRatioY);
}


