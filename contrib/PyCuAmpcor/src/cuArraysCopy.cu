/**
 * @file cuArraysCopy.cu
 * @brief Utilities for copying/converting images to different format
 *
 * All methods are declared in cuAmpcorUtil.h
 * cudaArraysCopyToBatch to extract a batch of windows from the raw image
 *   various implementations include:
 *   1. fixed or varying offsets, as start pixels for windows
 *   2. complex to complex, usually
 *   3. complex to (amplitude,0), for TOPS
 *   4. real to complex, for real images
 * cuArraysCopyExtract to extract(shrink in size) from a batch of windows to another batch
 *   overloaded for different data types
 * cuArraysCopyInsert to insert a batch of windows (smaller in size) to another batch
 *   overloaded for different data types
 * cuArraysCopyPadded to insert a batch of windows to another batch while padding 0s for rest elements
 *   used for fft oversampling
 *   see also cuArraysPadding.cu for other zero-padding utilities
 * cuArraysR2C cuArraysC2R cuArraysAbs to convert between different data types
 */


// dependencies
#include "cuArrays.h"
#include "cudaUtil.h"
#include "cudaError.h"
#include "float2.h"
#include "data_types.h"

// cuda kernel for cuArraysCopyToBatch
__global__ void cuArraysCopyToBatch_kernel(const image_complex_type *imageIn, const int inNX, const int inNY,
    complex_type *imageOut, const int outNX, const int outNY,
    const int nImagesX, const int nImagesY,
    const int strideX, const int strideY)
{
    int idxImage = blockIdx.z;
    int outx = threadIdx.x + blockDim.x*blockIdx.x;
    int outy = threadIdx.y + blockDim.y*blockIdx.y;
    if(idxImage >=nImagesX*nImagesY|| outx >= outNX || outy >= outNY) return;
    int idxOut = idxImage*outNX*outNY + outx*outNY + outy;
    int idxImageX = idxImage/nImagesY;
    int idxImageY = idxImage%nImagesY;
    int idxIn = (idxImageX*strideX+outx)*inNY + idxImageY*strideY+outy;
    imageOut[idxOut] = make_complex_type(imageIn[idxIn].x, imageIn[idxIn].y);
}

/**
 * Copy a chunk into a batch of chips for a given stride
 * @note used to extract chips from a raw image
 * @param image1 Input image as a large chunk
 * @param image2 Output images as a batch of chips
 * @param strideH stride along height to extract chips
 * @param strideW stride along width to extract chips
 * @param stream cudaStream
 */
void cuArraysCopyToBatch(cuArrays<image_complex_type> *image1, cuArrays<complex_type> *image2,
    int strideH, int strideW, cudaStream_t stream)
{
    const int nthreads = NTHREADS2D;
    dim3 blockSize(nthreads, nthreads, 1);
    dim3 gridSize(IDIVUP(image2->height,nthreads), IDIVUP(image2->width,nthreads), image2->count);
    cuArraysCopyToBatch_kernel<<<gridSize,blockSize, 0 , stream>>> (
        image1->devData, image1->height, image1->width,
        image2->devData, image2->height, image2->width,
        image2->countH, image2->countW,
        strideH, strideW);
    getLastCudaError("cuArraysCopyToBatch_kernel");
}

// kernel for cuArraysCopyToBatchWithOffset
template<typename T_in, typename T_out>
__global__ void cuArraysCopyToBatchWithOffset_kernel(const T_in *imageIn, const int inNX, const int inNY,
    T_out *imageOut, const int outNX, const int outNY, const int nImages,
    const int *offsetX, const int *offsetY)
{
    // get image index
    int idxImage = blockIdx.z;
    // check the image index within range
    if(idxImage>=nImages ) return;
    // get the output pixel location
    int outx = threadIdx.x + blockDim.x*blockIdx.x;
    int outy = threadIdx.y + blockDim.y*blockIdx.y;
     // check the output location within range (due to cuda threads)
    if(outx >= outNX || outy >= outNY) return;
    // flatten the output location to 1d
    int idxOut = idxImage*outNX*outNY + outx*outNY + outy;
    // find the input pixel location
    int inx = offsetX[idxImage] + outx;
    int iny = offsetY[idxImage] + outy;
    // check whether the location is within the input image range
    if(inx>=0 && inx<inNX && iny>=0 && iny<inNY) {
        int idxIn = inx*inNY+iny;
        imageOut[idxOut] = T_out{imageIn[idxIn]}; //true, copy
    }
    else
        imageOut[idxOut] = T_out{0.0}; //false, fill with 0
}

// kernel for cuArraysCopyToBatchWithOffset
template<>
__global__ void cuArraysCopyToBatchWithOffset_kernel<image_complex_type, complex_type>(const image_complex_type *imageIn, const int inNX, const int inNY,
    complex_type *imageOut, const int outNX, const int outNY, const int nImages,
    const int *offsetX, const int *offsetY)
{
    // get image index
    int idxImage = blockIdx.z;
    // check the image index within range
    if(idxImage>=nImages ) return;
    // get the output pixel location
    int outx = threadIdx.x + blockDim.x*blockIdx.x;
    int outy = threadIdx.y + blockDim.y*blockIdx.y;
     // check the output location within range (due to cuda threads)
    if(outx >= outNX || outy >= outNY) return;
    // flatten the output location to 1d
    int idxOut = idxImage*outNX*outNY + outx*outNY + outy;
    // find the input pixel location
    int inx = offsetX[idxImage] + outx;
    int iny = offsetY[idxImage] + outy;
    // check whether the location is within the input image range
    if(inx>=0 && inx<inNX && iny>=0 && iny<inNY) {
        int idxIn = inx*inNY+iny;
        imageOut[idxOut] = make_complex_type(imageIn[idxIn].x, imageIn[idxIn].y);
    }
    else
        imageOut[idxOut] = make_complex_type(0.0, 0.0); //false, fill with 0
}

/**
 * Copy a chunk into a batch of chips with varying offsets/strides
 * @note used to extract chips from a raw secondary image with varying offsets
 * @param image1 Input image as a large chunk
 * @param  the leading dimension of image1, usually, its width inNY
 * @param image2 Output images as a batch of chips
 * @param strideH (varying) offsets along height to extract chips
 * @param strideW (varying) offsets along width to extract chips
 * @param stream cudaStream
 */
void cuArraysCopyToBatchWithOffset(cuArrays<image_complex_type> *image1, const int inNX, const int inNY,
    cuArrays<complex_type> *image2, const int *offsetH, const int* offsetW, cudaStream_t stream)
{
    const int nthreads = 16;
    dim3 blockSize(nthreads, nthreads, 1);
    dim3 gridSize(IDIVUP(image2->height,nthreads), IDIVUP(image2->width,nthreads), image2->count);
    cuArraysCopyToBatchWithOffset_kernel<<<gridSize,blockSize, 0 , stream>>> (
        image1->devData, inNX, inNY,
        image2->devData, image2->height, image2->width, image2->count,
        offsetH, offsetW);
    getLastCudaError("cuArraysCopyToBatchAbsWithOffset_kernel");
}

// same as above, but from complex to real(take amplitudes)
__global__ void cuArraysCopyToBatchAbsWithOffset_kernel(const image_complex_type *imageIn, const int inNX, const int inNY,
    complex_type *imageOut, const int outNX, const int outNY, const int nImages,
    const int *offsetX, const int *offsetY)
{

    // get image index
    int idxImage = blockIdx.z;
    // check the image index within range
    if(idxImage>=nImages ) return;
    // get the output pixel location
    int outx = threadIdx.x + blockDim.x*blockIdx.x;
    int outy = threadIdx.y + blockDim.y*blockIdx.y;
     // check the output location within range (due to cuda threads)
    if(outx >= outNX || outy >= outNY) return;
    // flatten the output location to 1d
    int idxOut = idxImage*outNX*outNY + outx*outNY + outy;
    // find the input pixel location
    int inx = offsetX[idxImage] + outx;
    int iny = offsetY[idxImage] + outy;
    // check whether the location is within the input image range
    if(inx>=0 && inx<inNX && iny>=0 && iny<inNY)
    {
        int idxIn = inx*inNY+iny;
        imageOut[idxOut] = make_complex_type(complexAbs(imageIn[idxIn]), 0.0); //true, copy with abs value
    }
    else
        imageOut[idxOut] = make_complex_type(0.0, 0.0); //false, fill with 0
}

/**
 * Copy a chunk into a batch of chips with varying offsets/strides
 * @note similar to cuArraysCopyToBatchWithOffset, but take amplitudes instead
 * @param image1 Input image as a large chunk
 * @param lda1 the leading dimension of image1, usually, its width inNY
 * @param image2 Output images as a batch of chips
 * @param strideH (varying) offsets along height to extract chips
 * @param strideW (varying) offsets along width to extract chips
 * @param stream cudaStream
 */
void cuArraysCopyToBatchAbsWithOffset(cuArrays<image_complex_type> *image1, const int inNX, const int inNY,
    cuArrays<complex_type> *image2, const int *offsetH, const int* offsetW, cudaStream_t stream)
{
    const int nthreads = 16;
    dim3 blockSize(nthreads, nthreads, 1);
    dim3 gridSize(IDIVUP(image2->height,nthreads), IDIVUP(image2->width,nthreads), image2->count);
    cuArraysCopyToBatchAbsWithOffset_kernel<<<gridSize,blockSize, 0 , stream>>> (
        image1->devData, inNX, inNY,
        image2->devData, image2->height, image2->width, image2->count,
        offsetH, offsetW);
    getLastCudaError("cuArraysCopyToBatchAbsWithOffset_kernel");
}

/**
 * Copy a chunk into a batch of chips with varying offsets/strides
 * @note used to load real images
 * @param image1 Input image as a large chunk
 * @param lda1 the leading dimension of image1, usually, its width inNY
 * @param image2 Output images as a batch of chips
 * @param strideH (varying) offsets along height to extract chips
 * @param strideW (varying) offsets along width to extract chips
 * @param stream cudaStream
 */
void cuArraysCopyToBatchWithOffsetR2C(cuArrays<image_real_type> *image1, const int inNX, const int inNY,
    cuArrays<complex_type> *image2, const int *offsetH, const int* offsetW, cudaStream_t stream)
{
    const int nthreads = 16;
    dim3 blockSize(nthreads, nthreads, 1);
    dim3 gridSize(IDIVUP(image2->height,nthreads), IDIVUP(image2->width,nthreads), image2->count);

    cuArraysCopyToBatchWithOffset_kernel<<<gridSize,blockSize, 0 , stream>>> (
        image1->devData, inNX, inNY,
        image2->devData, image2->height, image2->width, image2->count,
        offsetH, offsetW);
    getLastCudaError("cuArraysCopyToBatchWithOffsetR2C_kernel");
}

//copy a chunk into a series of chips, from complex to real
__global__ void cuArraysCopyC2R_kernel(const complex_type *imageIn, const int inNX, const int inNY,
    real_type *imageOut, const int outNX, const int outNY,
    const int nImagesX, const int nImagesY,
    const int strideX, const int strideY, const real_type factor)
{
    int idxImage = blockIdx.z;
    int outx = threadIdx.x + blockDim.x*blockIdx.x;
    int outy = threadIdx.y + blockDim.y*blockIdx.y;
    if(idxImage >=nImagesX*nImagesY|| outx >= outNX || outy >= outNY) return;
    int idxOut = idxImage*outNX*outNY + outx*outNY + outy;
    int idxImageX = idxImage/nImagesY;
    int idxImageY = idxImage%nImagesY;
    int idxIn = (idxImageX*strideX+outx)*inNY + idxImageY*strideY+outy;
    imageOut[idxOut] = complexAbs(imageIn[idxIn])*factor;
}

/**
 * Copy a chunk into a batch of chips with varying offsets/strides
 * @note similar to cuArraysCopyToBatchWithOffset, but take amplitudes instead
 * @param image1 Input image as a large chunk
 * @param image2 Output images as a batch of chips
 * @param strideH offsets along height to extract chips
 * @param strideW offsets along width to extract chips
 * @param stream cudaStream
 */
void cuArraysCopyC2R(cuArrays<complex_type> *image1, cuArrays<real_type> *image2,
    int strideH, int strideW, cudaStream_t stream)
{
    const int nthreads = 16;
    dim3 blockSize(nthreads, nthreads, 1);
    dim3 gridSize(IDIVUP(image2->height,nthreads), IDIVUP(image2->width,nthreads), image2->count);
    real_type factor = 1.0f/image1->size; //the FFT factor
    cuArraysCopyC2R_kernel<<<gridSize,blockSize, 0 , stream>>> (
        image1->devData, image1->height, image1->width,
        image2->devData, image2->height, image2->width,
        image2->countH, image2->countW,
        strideH, strideW, factor);
    getLastCudaError("cuda Error: cuArraysCopyC2R_kernel");
}

//copy a chunk into a series of chips with varying strides
template<typename T>
__global__ void cuArraysCopyExtractVaryingOffset(const T *imageIn, const int inNX, const int inNY,
     T *imageOut, const int outNX, const int outNY, const int nImages,
     const int2 *offsets)
{
    int outx = threadIdx.x + blockDim.x*blockIdx.x;
    int outy = threadIdx.y + blockDim.y*blockIdx.y;

    if(outx < outNX && outy < outNY)
    {
        int idxImage = blockIdx.z;
        int idxOut = (blockIdx.z * outNX + outx)*outNY+outy;
        int idxIn = (blockIdx.z*inNX + outx + offsets[idxImage].x)*inNY + outy + offsets[idxImage].y;
        imageOut[idxOut] = imageIn[idxIn];
    }
}

/**
 * Copy a tile of images to another image, with starting pixels offsets
 * @param[in] imageIn input images of dimension nImages*inNX*inNY
 * @param[out] imageOut output images of dimension nImages*outNX*outNY
 * @param[in] offsets, varying offsets for extraction
 */
template<typename T>
void cuArraysCopyExtract(cuArrays<T> *imagesIn, cuArrays<T> *imagesOut, cuArrays<int2> *offsets, cudaStream_t stream)
{
    //assert(imagesIn->height >= imagesOut && inNY >= outNY);
    const int nthreads = 16;
    dim3 threadsperblock(nthreads, nthreads,1);
    dim3 blockspergrid(IDIVUP(imagesOut->height,nthreads), IDIVUP(imagesOut->width,nthreads), imagesOut->count);
    cuArraysCopyExtractVaryingOffset<<<blockspergrid, threadsperblock,0, stream>>>(imagesIn->devData, imagesIn->height, imagesIn->width,
        imagesOut->devData, imagesOut->height, imagesOut->width, imagesOut->count, offsets->devData);
    getLastCudaError("cuArraysCopyExtract error");
}

// instantiate the above template for the data types we need
template void cuArraysCopyExtract(cuArrays<real_type> *in, cuArrays<real_type> *out, cuArrays<int2> *offsets, cudaStream_t);
template void cuArraysCopyExtract(cuArrays<complex_type> *in, cuArrays<complex_type> *out, cuArrays<int2> *offsets, cudaStream_t);

// correlation surface extraction (Minyan Zhong)
__global__ void cuArraysCopyExtractVaryingOffsetCorr(const real_type *imageIn, const int inNX, const int inNY,
     real_type *imageOut, const int outNX, const int outNY, const int nImages,
     const int2 *maxloc)
{

    // get the image index
    int idxImage = blockIdx.z;

    // One thread per out point. Find the coordinates within the current image.
    int outx = threadIdx.x + blockDim.x*blockIdx.x;
    int outy = threadIdx.y + blockDim.y*blockIdx.y;

    // check whether thread is within output image range
    if (outx < outNX && outy < outNY)
    {
        // Find the corresponding input.
        int inx = outx + maxloc[idxImage].x - outNX/2;
        int iny = outy + maxloc[idxImage].y - outNY/2;

        // Find the location in flattened array.
        int idxOut = (idxImage * outNX + outx) * outNY + outy;
        int idxIn = (idxImage * inNX + inx) * inNY + iny;
        imageOut[idxOut] = imageIn[idxIn];
    }
}

/**
 * copy a tile of images to another image, with starting pixels offsets accouting for boundary
 * @param[in] imageIn inut images
 * @param[out] imageOut output images of dimension nImages*outNX*outNY
 */
void cuArraysCopyExtractCorr(cuArrays<real_type> *imagesIn, cuArrays<real_type> *imagesOut, cuArrays<int2> *maxloc, cudaStream_t stream)
{
    //assert(imagesIn->height >= imagesOut && inNY >= outNY);
    const int nthreads = 16;

    dim3 threadsperblock(nthreads, nthreads,1);

    dim3 blockspergrid(IDIVUP(imagesOut->height,nthreads), IDIVUP(imagesOut->width,nthreads), imagesOut->count);

    cuArraysCopyExtractVaryingOffsetCorr<<<blockspergrid, threadsperblock,0, stream>>>(
        imagesIn->devData, imagesIn->height, imagesIn->width,
        imagesOut->devData, imagesOut->height, imagesOut->width,
        imagesOut->count, maxloc->devData);
    getLastCudaError("cuArraysCopyExtract error");
}

__global__ void cuArraysCopyExtractVaryingOffsetCorr(const real_type *imageIn, const int inNX, const int inNY,
     real_type *imageOut, const int outNX, const int outNY, int *imageValid, const int nImages,
     const int2 *maxloc)
{

    // get the image index
    int idxImage = blockIdx.z;

    // One thread per out point. Find the coordinates within the current image.
    int outx = threadIdx.x + blockDim.x*blockIdx.x;
    int outy = threadIdx.y + blockDim.y*blockIdx.y;

    // check whether thread is within output image range
    if (outx < outNX && outy < outNY)
    {
        // Find the corresponding input.
        int inx = outx + maxloc[idxImage].x - outNX/2;
        int iny = outy + maxloc[idxImage].y - outNY/2;

        // Find the location in flattened array.
        int idxOut = ( blockIdx.z * outNX + outx ) * outNY + outy;
        int idxIn = ( blockIdx.z * inNX + inx ) * inNY + iny;

        // check whether inside of the input image
        if (inx>=0 && iny>=0 && inx<inNX && iny<inNY)
        {
            // inside the boundary, copy over and mark the pixel as valid (1)
            imageOut[idxOut] = imageIn[idxIn];
            imageValid[idxOut] = 1;
        }
        else {
            // outside, set it to 0 and mark the pixel as invalid (0)
            imageOut[idxOut] = 0.0f;
            imageValid[idxOut] = 0;
        }
    }
}

/**
 * copy a tile of images to another image, with starting pixels offsets accouting for boundary
 * @param[in] imageIn inut images
 * @param[out] imageOut output images of dimension nImages*outNX*outNY
 */
void cuArraysCopyExtractCorr(cuArrays<real_type> *imagesIn, cuArrays<real_type> *imagesOut, cuArrays<int> *imagesValid, cuArrays<int2> *maxloc, cudaStream_t stream)
{
    //assert(imagesIn->height >= imagesOut && inNY >= outNY);
    const int nthreads = 16;

    dim3 threadsperblock(nthreads, nthreads,1);

    dim3 blockspergrid(IDIVUP(imagesOut->height,nthreads), IDIVUP(imagesOut->width,nthreads), imagesOut->count);

    cuArraysCopyExtractVaryingOffsetCorr<<<blockspergrid, threadsperblock,0, stream>>>(imagesIn->devData, imagesIn->height, imagesIn->width,
        imagesOut->devData, imagesOut->height, imagesOut->width, imagesValid->devData, imagesOut->count, maxloc->devData);
    getLastCudaError("cuArraysCopyExtract error");
}


// end of correlation surface extraction (Minyan Zhong)



template<typename T>
__global__ void cuArraysCopyExtractFixedOffset(const T *imageIn, const int inNX, const int inNY,
     T *imageOut, const int outNX, const int outNY, const int nImages,
     const int offsetX, const int offsetY)
{
    int outx = threadIdx.x + blockDim.x*blockIdx.x;
    int outy = threadIdx.y + blockDim.y*blockIdx.y;

    if(outx < outNX && outy < outNY)
    {
        int idxOut = (blockIdx.z * outNX + outx)*outNY+outy;
        int idxIn = (blockIdx.z*inNX + outx + offsetX)*inNY + outy + offsetY;
        imageOut[idxOut] = imageIn[idxIn];
    }
}

__global__ void cuArraysCopyExtractFixedOffset(const complex_type *imageIn, const int inNX, const int inNY,
     real_type *imageOut, const int outNX, const int outNY, const int nImages,
     const int offsetX, const int offsetY)
{
    int outx = threadIdx.x + blockDim.x*blockIdx.x;
    int outy = threadIdx.y + blockDim.y*blockIdx.y;

    if(outx < outNX && outy < outNY)
    {
        int idxOut = (blockIdx.z * outNX + outx)*outNY+outy;
        int idxIn = (blockIdx.z*inNX + outx + offsetX)*inNY + outy + offsetY;
        imageOut[idxOut] = imageIn[idxIn].x;
    }
}

/**
 * copy/extract images from a large size to
 * a smaller size from the location (offsetX, offsetY)
 */
template<typename T_in, typename T_out>
void cuArraysCopyExtract(cuArrays<T_in> *imagesIn, cuArrays<T_out> *imagesOut, int2 offset, cudaStream_t stream)
{
    //assert(imagesIn->height >= imagesOut && inNY >= outNY);
    const int nthreads = NTHREADS2D;
    dim3 threadsperblock(nthreads, nthreads,1);
    dim3 blockspergrid(IDIVUP(imagesOut->height,nthreads), IDIVUP(imagesOut->width,nthreads), imagesOut->count);
    cuArraysCopyExtractFixedOffset<<<blockspergrid, threadsperblock,0, stream>>>
        (imagesIn->devData, imagesIn->height, imagesIn->width,
        imagesOut->devData, imagesOut->height, imagesOut->width, imagesOut->count, offset.x, offset.y);
    getLastCudaError("cuArraysCopyExtract error");
}

// instantiate the above template for the data types we need
template void cuArraysCopyExtract(cuArrays<real_type> *in, cuArrays<real_type> *out, int2 offset, cudaStream_t);
template void cuArraysCopyExtract(cuArrays<complex_type> *in, cuArrays<real_type> *out, int2 offset, cudaStream_t);
template void cuArraysCopyExtract(cuArrays<complex_type> *in, cuArrays<complex_type> *out, int2 offset, cudaStream_t);
template void cuArraysCopyExtract(cuArrays<real3_type> *in, cuArrays<real3_type> *out, int2 offset, cudaStream_t);


__global__ void cuArraysCopyExtractAbsFixedOffset(const complex_type *imageIn, const int inNX, const int inNY,
     real_type *imageOut, const int outNX, const int outNY, const int nImages,
     const int offsetX, const int offsetY)
{
    int outx = threadIdx.x + blockDim.x*blockIdx.x;
    int outy = threadIdx.y + blockDim.y*blockIdx.y;

    if(outx < outNX && outy < outNY)
    {
        int idxOut = (blockIdx.z * outNX + outx)*outNY+outy;
        int idxIn = (blockIdx.z*inNX + outx + offsetX)*inNY + outy + offsetY;
        imageOut[idxOut] = complexAbs(imageIn[idxIn]);
    }
}
/**
 * copy/extract images from a large size to
 * a smaller size from the location (offsetX, offsetY), take amplitude
 */
void cuArraysCopyExtractAbs(cuArrays<complex_type> *imagesIn, cuArrays<real_type> *imagesOut, int2 offset, cudaStream_t stream)
{
    //assert(imagesIn->height >= imagesOut && inNY >= outNY);
    const int nthreads = NTHREADS2D;
    dim3 threadsperblock(nthreads, nthreads,1);
    dim3 blockspergrid(IDIVUP(imagesOut->height,nthreads), IDIVUP(imagesOut->width,nthreads), imagesOut->count);
    cuArraysCopyExtractAbsFixedOffset<<<blockspergrid, threadsperblock,0, stream>>>
        (imagesIn->devData, imagesIn->height, imagesIn->width,
        imagesOut->devData, imagesOut->height, imagesOut->width, imagesOut->count, offset.x, offset.y);
    getLastCudaError("cuArraysCopyExtractAbs error");
}



template<typename T>
__global__ void cuArraysCopyInsert_kernel(const T* imageIn, const int inNX, const int inNY,
    T* imageOut, const int outNY, const int offsetX, const int offsetY)
{
    int inx = threadIdx.x + blockDim.x*blockIdx.x;
    int iny = threadIdx.y + blockDim.y*blockIdx.y;
    if(inx < inNX && iny < inNY) {
        int idxOut = IDX2R(inx+offsetX, iny+offsetY, outNY);
        int idxIn = IDX2R(inx, iny, inNY);
        imageOut[idxOut] = imageIn[idxIn];
    }
}

/**
 * copy/insert images from a smaller size to a larger size from the location (offsetX, offsetY)
 */
template<typename T>
void cuArraysCopyInsert(cuArrays<T> *imageIn, cuArrays<T> *imageOut, int offsetX, int offsetY, cudaStream_t stream)
{
    const int nthreads = 16;
    dim3 threadsperblock(nthreads, nthreads);
    dim3 blockspergrid(IDIVUP(imageIn->height,nthreads), IDIVUP(imageIn->width,nthreads));
    cuArraysCopyInsert_kernel<<<blockspergrid, threadsperblock,0, stream>>>(imageIn->devData, imageIn->height, imageIn->width,
           imageOut->devData,  imageOut->width, offsetX, offsetY);
    getLastCudaError("cuArraysCopyInsert error");
}

// instantiate the above template for the data types we need
template void cuArraysCopyInsert(cuArrays<complex_type>* in, cuArrays<complex_type>* out, int offX, int offY, cudaStream_t);
template void cuArraysCopyInsert(cuArrays<real3_type>* in, cuArrays<real3_type>* out, int offX, int offY, cudaStream_t);
template void cuArraysCopyInsert(cuArrays<real_type>* in, cuArrays<real_type>* out, int offX, int offY, cudaStream_t);
template void cuArraysCopyInsert(cuArrays<int>* in, cuArrays<int>* out, int offX, int offY, cudaStream_t);

template<typename T_in, typename T_out>
__global__ void cuArraysCopyPadded_kernel(T_in *imageIn, int inNX, int inNY, int sizeIn,
    T_out *imageOut, int outNX, int outNY, int sizeOut, int nImages)
{
    int outx = threadIdx.x + blockDim.x*blockIdx.x;
    int outy = threadIdx.y + blockDim.y*blockIdx.y;

    if(outx < outNX && outy < outNY)
    {
        int idxImage = blockIdx.z;
        int idxOut = IDX2R(outx, outy, outNY)+idxImage*sizeOut;
        if(outx < inNX && outy <inNY) {
            int idxIn = IDX2R(outx, outy, inNY)+idxImage*sizeIn;
            imageOut[idxOut] = T_out{imageIn[idxIn]};
        } else {
            imageOut[idxOut] = T_out{0};
        }
    }
}

/**
 * copy images from a smaller size to a larger size while padding 0 for extra elements
 */
template<typename T_in, typename T_out>
void cuArraysCopyPadded(cuArrays<T_in> *imageIn, cuArrays<T_out> *imageOut, cudaStream_t stream)
{
    const int nthreads = 16;
    int nImages = imageIn->count;
    dim3 blockSize(nthreads, nthreads,1);
    dim3 gridSize(IDIVUP(imageOut->height,nthreads), IDIVUP(imageOut->width,nthreads), nImages);
    cuArraysCopyPadded_kernel<<<gridSize, blockSize, 0, stream>>>(imageIn->devData, imageIn->height, imageIn->width, imageIn->size,
       imageOut->devData, imageOut->height, imageOut->width, imageOut->size, nImages);
    getLastCudaError("cuArraysCopyPadded error");
}

// instantiate the above template for the data types we need
template void cuArraysCopyPadded(cuArrays<real_type> *imageIn, cuArrays<real_type> *imageOut, cudaStream_t);
template void cuArraysCopyPadded(cuArrays<real_type> *imageIn, cuArrays<complex_type> *imageOut, cudaStream_t);
template void cuArraysCopyPadded(cuArrays<complex_type> *imageIn, cuArrays<complex_type> *imageOut, cudaStream_t);

// cuda kernel for setting a constant value
__global__ void cuArraysSetConstant_kernel(real_type *image, int size, real_type value)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if(idx < size)
    {
        image[idx] = value;
    }
}

/**
 * Set real images to a constant value
 * @note use setZero if value=0 because cudaMemset is faster
 */
void cuArraysSetConstant(cuArrays<real_type> *imageIn, real_type value, cudaStream_t stream)
{
    const int nthreads = 256;
    int size = imageIn->getSize();

    cuArraysSetConstant_kernel<<<IDIVUP(size, nthreads), nthreads, 0, stream>>>
        (imageIn->devData, imageIn->size, value);
     getLastCudaError("cuArraysSetConstant error");
}


// convert real_type to complex_type(complex)
__global__ void cuArraysR2C_kernel(real_type *image1, complex_type *image2, int size)
{
    int idx =  threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < size)
    {
        image2[idx].x = image1[idx];
        image2[idx].y =  0.0;
    }
}

/**
 * Convert real images to complex images (set imaginary parts to 0)
 * @param[in] image1 input images
 * @param[out] image2 output images
 */
void cuArraysR2C(cuArrays<real_type> *image1, cuArrays<complex_type> *image2, cudaStream_t stream)
{
    int size = image1->getSize();
    cuArraysR2C_kernel<<<IDIVUP(size, NTHREADS), NTHREADS, 0, stream>>>(image1->devData, image2->devData, size);
    getLastCudaError("cuArraysR2C");
}


// take real part of complex_type to real_type
__global__ void cuArraysC2R_kernel(complex_type *image1, real_type *image2, int size)
{
    int idx =  threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < size)
    {
        image2[idx] = image1[idx].x;
    }
}

/**
 * Take real part of complex images
 * @param[in] image1 input images
 * @param[out] image2 output images
 */
void cuArraysC2R(cuArrays<complex_type> *image1, cuArrays<real_type> *image2, cudaStream_t stream)
{
    int size = image1->getSize();
    cuArraysC2R_kernel<<<IDIVUP(size, NTHREADS), NTHREADS, 0, stream>>>(image1->devData, image2->devData, size);
    getLastCudaError("cuArraysC2R");
}

// cuda kernel for cuArraysAbs
__global__ void cuArraysAbs_kernel(complex_type *image1, real_type *image2, int size)
{
    int idx =  threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < size)
    {
        image2[idx] = complexAbs(image1[idx]);
    }
}

/**
 * Obtain abs (amplitudes) of complex images
 * @param[in] image1 input images
 * @param[out] image2 output images
 */
void cuArraysAbs(cuArrays<complex_type> *image1, cuArrays<real_type> *image2, cudaStream_t stream)
{
    int size = image1->getSize();
    cuArraysAbs_kernel<<<IDIVUP(size, NTHREADS), NTHREADS, 0, stream>>>(image1->devData, image2->devData, size);
    getLastCudaError("cuArraysAbs_kernel");
}

// end of file
