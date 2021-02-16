/*
 * @file cuOffset.cu
 * @brief Utilities used to determine the offset field
 *
 */

// my module dependencies
#include "cuAmpcorUtil.h"
// for FLT_MAX
#include <cfloat>

// find the max between two elements
inline static __device__ void maxPairReduce(volatile float* maxval, volatile int* maxloc,
      size_t gid, size_t strideid)
{
    if(maxval[gid] < maxval[strideid]) {
        maxval[gid] = maxval[strideid];
        maxloc[gid] = maxloc[strideid];
    }
}

// max reduction kernel
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

    // reduction for intra-block elements
    // i.e., for elements with i, i+BLOCKSIZE, i+2*BLOCKSIZE ...
    for(int gid = tid; gid < imageSize; gid+=blockDim.x)
    {
        imagePixel = imageStart+gid;
        if(shval[tid] < images[imagePixel]) {
            shval[tid] = images[imagePixel];
            shloc[tid] = gid;
        }
    }
    __syncthreads();

    // reduction within a block
    if (BLOCKSIZE >=1024){ if (tid < 512) { maxPairReduce(shval, shloc, tid, tid + 512); } __syncthreads(); }
    if (BLOCKSIZE >=512) { if (tid < 256) { maxPairReduce(shval, shloc, tid, tid + 256); } __syncthreads(); }
    if (BLOCKSIZE >=256) { if (tid < 128) { maxPairReduce(shval, shloc, tid, tid + 128); } __syncthreads(); }
    if (BLOCKSIZE >=128) { if (tid < 64 ) { maxPairReduce(shval, shloc, tid, tid + 64 ); } __syncthreads(); }
    // reduction within a warp
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


// kernel for 2D array(image), find max location only
template <const int BLOCKSIZE>
__global__ void  cudaKernel_maxloc2D(const float* const images, int2* maxloc, float* maxval,
    const size_t imageNX, const size_t imageNY, const size_t nImages)
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

/**
 * Find both the maximum value and the location for a batch of 2D images
 * @param[in] images input batch of images
 * @param[out] maxval arrays to hold the max values
 * @param[out] maxloc arrays to hold the max locations
 * @param[in] stream cudaStream
 * @note This routine is overloaded with the routine without maxval
 */
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

/**
 * Find (only) the maximum location for a batch of 2D images
 * @param[in] images input batch of images
 * @param[out] maxloc arrays to hold the max locations
 * @param[in] stream cudaStream
 * @note This routine is overloaded with the routine with maxval
 */
void cuArraysMaxloc2D(cuArrays<float> *images, cuArrays<int2> *maxloc, cudaStream_t stream)
{
    cudaKernel_maxloc2D<NTHREADS><<<images->count, NTHREADS, 0, stream>>>
        (images->devData, maxloc->devData, images->height, images->width, images->count);
    getLastCudaError("cudaKernel find max location 2D error\n");
}

// cuda kernel for cuSubPixelOffset
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


/**
 * Determine the final offset value
 * @param[in] offsetInit max location (adjusted to the starting location for extraction) determined from
 *   the cross-correlation before oversampling, in dimensions of pixel
 * @param[in] offsetZoomIn max location from the oversampled cross-correlation surface
 * @param[out] offsetFinal the combined offset value
 * @param[in] OversampleRatioZoomIn the correlation surface oversampling factor
 * @param[in] OversampleRatioRaw the oversampling factor of reference/secondary windows before cross-correlation
 * @param[in] xHalfRangInit the original half search range along x, to be subtracted
 * @param[in] yHalfRangInit the original half search range along y, to be subtracted
 *
 * 1. Cross-correlation is performed at first for the un-oversampled data with a larger search range.
 *   The secondary window is then extracted to a smaller size (a smaller search range) around the max location.
 *   The extraction starting location (offsetInit) - original half search range (xHalfRangeInit, yHalfRangeInit)
 *        = pixel size offset
 * 2. Reference/secondary windows are then oversampled by OversampleRatioRaw, and cross-correlated.
 * 3. The correlation surface is further oversampled by OversampleRatioZoomIn.
 *    The overall oversampling factor is OversampleRatioZoomIn*OversampleRatioRaw.
 *    The max location in oversampled correlation surface (offsetZoomIn) / overall oversampling factor
 *        = subpixel offset
 *    Final offset =  pixel size offset +  subpixel offset
 */
void cuSubPixelOffset(cuArrays<int2> *offsetInit, cuArrays<int2> *offsetZoomIn,
    cuArrays<float2> *offsetFinal,
    int OverSampleRatioZoomin, int OverSampleRatioRaw,
    int xHalfRangeInit,  int yHalfRangeInit,
    cudaStream_t stream)
{
    int size = offsetInit->getSize();
    float OSratio = 1.0f/(float)(OverSampleRatioZoomin*OverSampleRatioRaw);
    float xoffset = xHalfRangeInit ;
    float yoffset = yHalfRangeInit ;

    cuSubPixelOffset_kernel<<<IDIVUP(size, NTHREADS), NTHREADS, 0, stream>>>
        (offsetInit->devData, offsetZoomIn->devData,
         offsetFinal->devData, OSratio, xoffset, yoffset, size);
    getLastCudaError("cuSubPixelOffset_kernel");

}

// cuda device function to compute the shift of center
static inline __device__ int2 dev_adjustOffset(
    const int oldRange, const int newRange, const int maxloc)
{
    // determine the starting point around the maxloc
    // oldRange is the half search window size, e.g., = 32
    // newRange is the half extract size, e.g., = 4
    // maxloc is in range [0, 64]
    // we want to extract \pm 4 centered at maxloc
    // Examples:
    // 1. maxloc = 40: we set start=maxloc-newRange=36, and extract [36,44), shift=0
    // 2. maxloc = 2, start=-2: we set start=0, shift=-2,
    //   (shift means the max is -2 from the extracted center 4)
    // 3. maxloc =64, start=60: set start=56, shift = 4
    //   (shift means the max is 4 from the extracted center 60).

    // shift the max location by -newRange to find the start
    int start = maxloc - newRange;
    // if start is within the range, the max location will be in the center
    int shift = 0;
    // right boundary
    int rbound = 2*(oldRange-newRange);
    if(start<0)     // if exceeding the limit on the left
    {
        // set start at 0 and record the shift of center
        shift = -start;
        start = 0;
    }
    else if(start > rbound ) // if exceeding the limit on the right
    {
        //
        shift = start-rbound;
        start = rbound;
    }
    return make_int2(start, shift);
}

// cuda kernel for cuDetermineSecondaryExtractOffset
__global__ void cudaKernel_determineSecondaryExtractOffset(int2 * maxLoc, int2 *shift,
    const size_t nImages, int xOldRange, int yOldRange, int xNewRange, int yNewRange)
{
    int imageIndex = threadIdx.x + blockDim.x *blockIdx.x; //image index
    if (imageIndex < nImages)
    {
        // get the starting pixel (stored back to maxloc) and shift
        int2 result = dev_adjustOffset(xOldRange, xNewRange, maxLoc[imageIndex].x);
        maxLoc[imageIndex].x = result.x;
        shift[imageIndex].x = result.y;
        result = dev_adjustOffset(yOldRange, yNewRange, maxLoc[imageIndex].y);
        maxLoc[imageIndex].y = result.x;
        shift[imageIndex].y = result.y;
    }
}

/**
 * Determine the secondary window extract offset from the max location
 * @param[in] xOldRange, yOldRange are (half) search ranges in first step
 * @param[in] xNewRange, yNewRange are (half) search range
 *
 * After the first run of cross-correlation, with a larger search range,
 *  We now choose a smaller search range around the max location for oversampling.
 *  This procedure is used to determine the starting pixel locations for extraction.
 */
void cuDetermineSecondaryExtractOffset(cuArrays<int2> *maxLoc, cuArrays<int2> *maxLocShift,
    int xOldRange, int yOldRange, int xNewRange, int yNewRange, cudaStream_t stream)
{
    int threadsperblock=NTHREADS;
    int blockspergrid=IDIVUP(maxLoc->size, threadsperblock);
    cudaKernel_determineSecondaryExtractOffset<<<blockspergrid, threadsperblock, 0, stream>>>
        (maxLoc->devData, maxLocShift->devData, maxLoc->size, xOldRange, yOldRange, xNewRange, yNewRange);
    getLastCudaError("cuDetermineSecondaryExtractOffset");
}

// end of file


