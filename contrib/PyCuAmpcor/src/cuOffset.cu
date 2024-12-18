/*
 * @file cuOffset.cu
 * @brief Utilities used to determine the offset field
 *
 */

// my module dependencies
#include "cuAmpcorUtil.h"

// find the max between two elements
inline static __device__ void maxPairReduce(volatile real_type* maxval, volatile int* maxloc,
      size_t gid, size_t strideid)
{
    if(maxval[gid] < maxval[strideid]) {
        maxval[gid] = maxval[strideid];
        maxloc[gid] = maxloc[strideid];
    }
}

// max reduction kernel for a 2d image
// start from (start.x, start.y) in a rectangle range (range.x, range.y)
// start + range <= imageSize
template<const int BLOCKSIZE>
__device__ void max_reduction_2d(const real_type* const image,
    const int2 imageSize,
    const int2 start, const int2 range,
    volatile real_type* shval,
    volatile int* shloc)
{
    int tid = threadIdx.x;
    shval[tid] = -REAL_MAX;

    // reduction for intra-block elements
    // i.e., for elements with i, i+BLOCKSIZE, i+2*BLOCKSIZE ...
    for(int gid = tid; gid < range.x*range.y; gid+=blockDim.x)
    {
        // gid is the flattened pixel id in the search range
        // get the pixel 2d coordinate in whole image
        int idx = start.x + gid / range.y;
        int idy = start.y + gid % range.y;
        // get the flattened 1d coordinate
        int pixelId = IDX2R(idx, idy, imageSize.y);
        real_type pixelValue = image[pixelId];
        if(shval[tid] < pixelValue) {
            shval[tid] = pixelValue;
            shloc[tid] = pixelId;
        }
    }
    __syncthreads();

    // reduction within a block with shared memory
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
__global__ void  cudaKernel_maxloc2D(const real_type* const images, int2* maxloc, real_type* maxval,
    const int2 imageSize, const size_t nImages, const int2 start, const int2 range)
{
    __shared__ real_type shval[BLOCKSIZE];
    __shared__ int shloc[BLOCKSIZE];

    int imageIdx = blockIdx.x;
    if(imageIdx >= nImages) return;

    // get the starting pointer for this image
    const real_type *image = images + imageIdx*imageSize.x*imageSize.y;

    max_reduction_2d<BLOCKSIZE>(image, imageSize, start, range, shval, shloc);

    // thread 0 contains the final result, convert it to 2d coordinate
    if (threadIdx.x == 0) {
        maxloc[imageIdx] = make_int2(shloc[0]/imageSize.y, shloc[0]%imageSize.y);
        maxval[imageIdx] = shval[0];
    }
}

/**
 * Find both the maximum value and the location for a batch of 2D images
 * @param[in] images input batch of images
 * @param[out] maxval arrays to hold the max values
 * @param[out] maxloc arrays to hold the max locations
 * @param[in] start starting search pixel
 * @param[in] range search range
 * @param[in] stream cudaStream
 * @note This routine is overloaded with the routine without maxval
 */
void cuArraysMaxloc2D(cuArrays<real_type> *images,
                      const int2 start, const int2 range,
                      cuArrays<int2> *maxloc, cuArrays<real_type> *maxval, cudaStream_t stream)
{
    cudaKernel_maxloc2D<NTHREADS><<<images->count, NTHREADS, 0, stream>>>
        (images->devData,
        maxloc->devData, maxval->devData,
        make_int2(images->height, images->width), images->count,
        start, range
        );
    getLastCudaError("cudaKernel find max location 2D error\n");
}

/**
 * Find both the maximum value and the location for a batch of 2D images
 * @param[in] images input batch of images
 * @param[out] maxval arrays to hold the max values
 * @param[out] maxloc arrays to hold the max locations
 * @param[in] stream cudaStream
 */
void cuArraysMaxloc2D(cuArrays<real_type> *images,
                      cuArrays<int2> *maxloc, cuArrays<real_type> *maxval, cudaStream_t stream)
{
    // if no start and range are provided, use the whole image
    int2 start = make_int2(0, 0);
    int2 imageSize = make_int2(images->height, images->width);
    cudaKernel_maxloc2D<NTHREADS><<<images->count, NTHREADS, 0, stream>>>
        (images->devData,
        maxloc->devData, maxval->devData,
        imageSize, images->count,
        start, imageSize
        );
    getLastCudaError("cudaKernel find max location 2D error\n");
}

// cuda kernel for cuSubPixelOffset
__global__ void cuSubPixelOffset_kernel(const int2 *offsetInit, const int2 *offsetZoomIn,
                                        real2_type *offsetFinal, const int size,
                                        const int2 initOrigin, const real_type initRatio,
                                        const int2 zoomInOrigin, const real_type zoomInRatio)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= size) return;
    offsetFinal[idx].x = initRatio*(offsetInit[idx].x-initOrigin.x) + zoomInRatio*(offsetZoomIn[idx].x - zoomInOrigin.x);
    offsetFinal[idx].y = initRatio*(offsetInit[idx].y-initOrigin.y) + zoomInRatio*(offsetZoomIn[idx].y - zoomInOrigin.y);
}


/**
 * Determine the final offset value
 * @param[in] offsetInit max location (adjusted to the starting location for extraction) determined from
 *   the cross-correlation before oversampling, in dimensions of pixel
 * @param[in] offsetZoomIn max location from the oversampled cross-correlation surface
 * @param[out] offsetFinal the combined offset value
 */
void cuSubPixelOffset(cuArrays<int2> *offsetInit,
    cuArrays<int2> *offsetZoomIn,
    cuArrays<real2_type> *offsetFinal,
    const int2 initOrigin, const int initFactor,
    const int2 zoomInOrigin, const int zoomInFactor,
    cudaStream_t stream)
{
    int size = offsetInit->getSize();

    // GPU performs multiplication faster
    real_type initRatio = 1.0f/(real_type)(initFactor);
    real_type zoomInRatio = 1.0f/(real_type)(zoomInFactor);


    cuSubPixelOffset_kernel<<<IDIVUP(size, NTHREADS), NTHREADS, 0, stream>>>
        (offsetInit->devData, offsetZoomIn->devData,
         offsetFinal->devData, size, initOrigin, initRatio, zoomInOrigin, zoomInRatio);
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


