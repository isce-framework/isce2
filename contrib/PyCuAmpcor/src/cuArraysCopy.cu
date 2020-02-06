/* imagecopy.cu
 * various utitilies for copying images in device memory
 *
 * Lijun Zhu @ Seismo Lab, Caltech
 * v1.0 Jan 2017
 */

#include "cuArrays.h"
#include "cudaUtil.h"
#include "cudaError.h"
#include "float2.h"

/*
inline __device__ float cuAbs(float2 a)
{
	return sqrtf(a.x*a.x+a.y*a.y);
}*/

// copy a chunk into a batch of chips for a given stride
__global__ void cuArraysCopyToBatch_kernel(const float2 *imageIn, const int inNX, const int inNY,
    float2 *imageOut, const int outNX, const int outNY,
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
	imageOut[idxOut] = imageIn[idxIn];
}

void cuArraysCopyToBatch(cuArrays<float2> *image1, cuArrays<float2> *image2,
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


// copy a chunk into a batch of chips for a set of offsets (varying strides), from complex to complex
__global__ void cuArraysCopyToBatchWithOffset_kernel(const float2 *imageIn, const int inNY,
    float2 *imageOut, const int outNX, const int outNY, const int nImages,
	const int *offsetX, const int *offsetY)
{
	int idxImage = blockIdx.z;
	int outx = threadIdx.x + blockDim.x*blockIdx.x;
	int outy = threadIdx.y + blockDim.y*blockIdx.y;
	if(idxImage>=nImages || outx >= outNX || outy >= outNY) return;
	int idxOut = idxImage*outNX*outNY + outx*outNY + outy;
	int idxIn = (offsetX[idxImage]+outx)*inNY + offsetY[idxImage] + outy;
	imageOut[idxOut] = imageIn[idxIn];
}

// lda1 (inNY) is the leading dimension of image1, usually, its width
void cuArraysCopyToBatchWithOffset(cuArrays<float2> *image1, const int lda1, cuArrays<float2> *image2,
	const int *offsetH, const int* offsetW, cudaStream_t stream)
{
	const int nthreads = 16;
	dim3 blockSize(nthreads, nthreads, 1);
	dim3 gridSize(IDIVUP(image2->height,nthreads), IDIVUP(image2->width,nthreads), image2->count);
    //fprintf(stderr, "copy tile to batch, %d %d\n", lda1, image2->count);
	cuArraysCopyToBatchWithOffset_kernel<<<gridSize,blockSize, 0 , stream>>> (
		image1->devData, lda1,
		image2->devData, image2->height, image2->width, image2->count,
		offsetH, offsetW);
	getLastCudaError("cuArraysCopyToBatchAbsWithOffset_kernel");
}

// copy a chunk into a batch of chips for a set of offsets (varying strides), from complex to real(take amplitudes)
__global__ void cuArraysCopyToBatchAbsWithOffset_kernel(const float2 *imageIn, const int inNY,
    float2 *imageOut, const int outNX, const int outNY, const int nImages,
	const int *offsetX, const int *offsetY)
{
	int idxImage = blockIdx.z;
	int outx = threadIdx.x + blockDim.x*blockIdx.x;
	int outy = threadIdx.y + blockDim.y*blockIdx.y;
	if(idxImage>=nImages || outx >= outNX || outy >= outNY) return;
	int idxOut = idxImage*outNX*outNY + outx*outNY + outy;
	int idxIn = (offsetX[idxImage]+outx)*inNY + offsetY[idxImage] + outy;
	imageOut[idxOut] = make_float2(complexAbs(imageIn[idxIn]), 0.0);
}

void cuArraysCopyToBatchAbsWithOffset(cuArrays<float2> *image1, const int lda1, cuArrays<float2> *image2,
	const int *offsetH, const int* offsetW, cudaStream_t stream)
{
	const int nthreads = 16;
	dim3 blockSize(nthreads, nthreads, 1);
	dim3 gridSize(IDIVUP(image2->height,nthreads), IDIVUP(image2->width,nthreads), image2->count);
    //fprintf(stderr, "copy tile to batch, %d %d\n", lda1, image2->count);
	cuArraysCopyToBatchAbsWithOffset_kernel<<<gridSize,blockSize, 0 , stream>>> (
		image1->devData, lda1,
		image2->devData, image2->height, image2->width, image2->count,
		offsetH, offsetW);
	getLastCudaError("cuArraysCopyToBatchAbsWithOffset_kernel");
}

// copy a chunk into a batch of chips for a set of offsets (varying strides), from real to complex(to real part)
__global__ void cuArraysCopyToBatchWithOffsetR2C_kernel(const float *imageIn, const int inNY,
    float2 *imageOut, const int outNX, const int outNY, const int nImages,
	const int *offsetX, const int *offsetY)
{
	int idxImage = blockIdx.z;
	int outx = threadIdx.x + blockDim.x*blockIdx.x;
	int outy = threadIdx.y + blockDim.y*blockIdx.y;
	if(idxImage>=nImages || outx >= outNX || outy >= outNY) return;
	int idxOut = idxImage*outNX*outNY + outx*outNY + outy;
	int idxIn = (offsetX[idxImage]+outx)*inNY + offsetY[idxImage] + outy;
	imageOut[idxOut] = make_float2(imageIn[idxIn], 0.0f);
}

void cuArraysCopyToBatchWithOffsetR2C(cuArrays<float> *image1, const int lda1, cuArrays<float2> *image2,
	const int *offsetH, const int* offsetW, cudaStream_t stream)
{
	const int nthreads = 16;
	dim3 blockSize(nthreads, nthreads, 1);
	dim3 gridSize(IDIVUP(image2->height,nthreads), IDIVUP(image2->width,nthreads), image2->count);
    //fprintf(stderr, "copy tile to batch, %d %d\n", lda1, image2->count);
	cuArraysCopyToBatchWithOffsetR2C_kernel<<<gridSize,blockSize, 0 , stream>>> (
		image1->devData, lda1,
		image2->devData, image2->height, image2->width, image2->count,
		offsetH, offsetW);
	getLastCudaError("cuArraysCopyToBatchWithOffsetR2C_kernel");
}

//copy a chunk into a series of chips
__global__ void cuArraysCopyC2R_kernel(const float2 *imageIn, const int inNX, const int inNY,
    float *imageOut, const int outNX, const int outNY,
    const int nImagesX, const int nImagesY,
	const int strideX, const int strideY, const float factor)
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
	//printf( "%d\n", idxOut);
}

//tested
void cuArraysCopyC2R(cuArrays<float2> *image1, cuArrays<float> *image2,
	int strideH, int strideW, cudaStream_t stream)
{
	const int nthreads = 16;
	dim3 blockSize(nthreads, nthreads, 1);
	dim3 gridSize(IDIVUP(image2->height,nthreads), IDIVUP(image2->width,nthreads), image2->count);
	float factor = 1.0f/image1->size; //the FFT factor
	cuArraysCopyC2R_kernel<<<gridSize,blockSize, 0 , stream>>> (
		image1->devData, image1->height, image1->width,
		image2->devData, image2->height, image2->width,
		image2->countH, image2->countW,
		strideH, strideW, factor);
	getLastCudaError("cuda Error: cuArraysCopyC2R_kernel");
}

__global__ void cuArraysCopyExtractVaryingOffset(const float *imageIn, const int inNX, const int inNY,
     float *imageOut, const int outNX, const int outNY, const int nImages,
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

/* copy a tile of images to another image, with starting pixels offsets
 * param[in] imageIn inut images
 * param[out] imageOut output images of dimension nImages*outNX*outNY
 */
void cuArraysCopyExtract(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut, cuArrays<int2> *offsets, cudaStream_t stream)
{
	//assert(imagesIn->height >= imagesOut && inNY >= outNY);
	const int nthreads = 16;
	dim3 threadsperblock(nthreads, nthreads,1);
	dim3 blockspergrid(IDIVUP(imagesOut->height,nthreads), IDIVUP(imagesOut->width,nthreads), imagesOut->count);
	cuArraysCopyExtractVaryingOffset<<<blockspergrid, threadsperblock,0, stream>>>(imagesIn->devData, imagesIn->height, imagesIn->width,
	    imagesOut->devData, imagesOut->height, imagesOut->width, imagesOut->count, offsets->devData);
	getLastCudaError("cuArraysCopyExtract error");
}


__global__ void cuArraysCopyExtractVaryingOffset_C2C(const float2 *imageIn, const int inNX, const int inNY,
     float2 *imageOut, const int outNX, const int outNY, const int nImages,
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

void cuArraysCopyExtract(cuArrays<float2> *imagesIn, cuArrays<float2> *imagesOut, cuArrays<int2> *offsets, cudaStream_t stream)
{
	//assert(imagesIn->height >= imagesOut && inNY >= outNY);
	const int nthreads = 16;
	dim3 threadsperblock(nthreads, nthreads,1);
	dim3 blockspergrid(IDIVUP(imagesOut->height,nthreads), IDIVUP(imagesOut->width,nthreads), imagesOut->count);
	cuArraysCopyExtractVaryingOffset_C2C<<<blockspergrid, threadsperblock,0, stream>>>(imagesIn->devData, imagesIn->height, imagesIn->width,
	    imagesOut->devData, imagesOut->height, imagesOut->width, imagesOut->count, offsets->devData);
	getLastCudaError("cuArraysCopyExtractC2C error");

}

// correlation surface extraction (Minyan Zhong)
__global__ void cuArraysCopyExtractVaryingOffsetCorr(const float *imageIn, const int inNX, const int inNY,
     float *imageOut, const int outNX, const int outNY, int *imageValid, const int nImages,
     const int2 *maxloc)
{

        int idxImage = blockIdx.z;

        // One thread per out point. Find the coordinates within the current image.
        int outx = threadIdx.x + blockDim.x*blockIdx.x;
        int outy = threadIdx.y + blockDim.y*blockIdx.y;

        // Find the correponding input.
        int inx = outx + maxloc[idxImage].x - outNX/2;
        int iny = outy + maxloc[idxImage].y - outNY/2;

        if (outx < outNX && outy < outNY)
        {
                // Find the location in full array.
                int idxOut = ( blockIdx.z * outNX + outx ) * outNY + outy;

                int idxIn = ( blockIdx.z * inNX + inx ) * inNY + iny;

            if (inx>=0 && iny>=0 && inx<inNX && iny<inNY) {

                    imageOut[idxOut] = imageIn[idxIn];
                    imageValid[idxOut] = 1;
                }
            else {
                    imageOut[idxOut] = 0.0f;
                    imageValid[idxOut] = 0;
            }
        }
}

/* copy a tile of images to another image, with starting pixels offsets accouting for boundary
 * param[in] imageIn inut images
 * param[out] imageOut output images of dimension nImages*outNX*outNY
 */
void cuArraysCopyExtractCorr(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut, cuArrays<int> *imagesValid, cuArrays<int2> *maxloc, cudaStream_t stream)
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



__global__ void cuArraysCopyExtractFixedOffset(const float *imageIn, const int inNX, const int inNY,
     float *imageOut, const int outNX, const int outNY, const int nImages,
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

/* copy a tile of images to another image, with starting pixels offsets
 * param[in] imageIn inut images
 * param[out] imageOut output images of dimension nImages*outNX*outNY
 */
void cuArraysCopyExtract(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut, int2 offset, cudaStream_t stream)
{
	//assert(imagesIn->height >= imagesOut && inNY >= outNY);
	const int nthreads = 16;
	dim3 threadsperblock(nthreads, nthreads,1);
	dim3 blockspergrid(IDIVUP(imagesOut->height,nthreads), IDIVUP(imagesOut->width,nthreads), imagesOut->count);
	cuArraysCopyExtractFixedOffset<<<blockspergrid, threadsperblock,0, stream>>>(imagesIn->devData, imagesIn->height, imagesIn->width,
	    imagesOut->devData, imagesOut->height, imagesOut->width, imagesOut->count, offset.x, offset.y);
	getLastCudaError("cuArraysCopyExtract error");
}

//

__global__ void cuArraysCopyExtract_C2C_FixedOffset(const float2 *imageIn, const int inNX, const int inNY,
     float2 *imageOut, const int outNX, const int outNY, const int nImages,
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


void cuArraysCopyExtract(cuArrays<float2> *imagesIn, cuArrays<float2> *imagesOut, int2 offset, cudaStream_t stream)
{
	//assert(imagesIn->height >= imagesOut && inNY >= outNY);
	const int nthreads = NTHREADS2D;
	dim3 threadsperblock(nthreads, nthreads,1);
	dim3 blockspergrid(IDIVUP(imagesOut->height,nthreads), IDIVUP(imagesOut->width,nthreads), imagesOut->count);
    //std::cout << "debug copyExtract" << imagesOut->width << imagesOut->height << "\n";
    //imagesIn->debuginfo(stream);
    //imagesOut->debuginfo(stream);
	cuArraysCopyExtract_C2C_FixedOffset<<<blockspergrid, threadsperblock,0, stream>>>
        (imagesIn->devData, imagesIn->height, imagesIn->width,
	    imagesOut->devData, imagesOut->height, imagesOut->width, imagesOut->count, offset.x, offset.y);
	getLastCudaError("cuArraysCopyExtractC2C error");
}
//

// float3
__global__ void cuArraysCopyExtract_C2C_FixedOffset(const float3 *imageIn, const int inNX, const int inNY,
     float3 *imageOut, const int outNX, const int outNY, const int nImages,
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


void cuArraysCopyExtract(cuArrays<float3> *imagesIn, cuArrays<float3> *imagesOut, int2 offset, cudaStream_t stream)
{
	//assert(imagesIn->height >= imagesOut && inNY >= outNY);
	const int nthreads = NTHREADS2D;
	dim3 threadsperblock(nthreads, nthreads,1);
	dim3 blockspergrid(IDIVUP(imagesOut->height,nthreads), IDIVUP(imagesOut->width,nthreads), imagesOut->count);
    //std::cout << "debug copyExtract" << imagesOut->width << imagesOut->height << "\n";
    //imagesIn->debuginfo(stream);
    //imagesOut->debuginfo(stream);
	cuArraysCopyExtract_C2C_FixedOffset<<<blockspergrid, threadsperblock,0, stream>>>
        (imagesIn->devData, imagesIn->height, imagesIn->width,
	    imagesOut->devData, imagesOut->height, imagesOut->width, imagesOut->count, offset.x, offset.y);
	getLastCudaError("cuArraysCopyExtractFloat3 error");
}

//


__global__ void cuArraysCopyExtract_C2R_FixedOffset(const float2 *imageIn, const int inNX, const int inNY,
     float *imageOut, const int outNX, const int outNY, const int nImages,
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



void cuArraysCopyExtract(cuArrays<float2> *imagesIn, cuArrays<float> *imagesOut, int2 offset, cudaStream_t stream)
{
	//assert(imagesIn->height >= imagesOut && inNY >= outNY);
	const int nthreads = NTHREADS2D;
	dim3 threadsperblock(nthreads, nthreads,1);
	dim3 blockspergrid(IDIVUP(imagesOut->height,nthreads), IDIVUP(imagesOut->width,nthreads), imagesOut->count);
	cuArraysCopyExtract_C2R_FixedOffset<<<blockspergrid, threadsperblock,0, stream>>>
        (imagesIn->devData, imagesIn->height, imagesIn->width,
	    imagesOut->devData, imagesOut->height, imagesOut->width, imagesOut->count, offset.x, offset.y);
	getLastCudaError("cuArraysCopyExtractC2C error");
}
//

__global__ void cuArraysCopyInsert_kernel(const float2* imageIn, const int inNX, const int inNY,
   float2* imageOut, const int outNY, const int offsetX, const int offsetY)
{
	int inx = threadIdx.x + blockDim.x*blockIdx.x;
	int iny = threadIdx.y + blockDim.y*blockIdx.y;
	if(inx < inNX && iny < inNY) {
		int idxOut = IDX2R(inx+offsetX, iny+offsetY, outNY);
		int idxIn = IDX2R(inx, iny, inNY);
		imageOut[idxOut] = make_float2(imageIn[idxIn].x, imageIn[idxIn].y);
	}
}


void cuArraysCopyInsert(cuArrays<float2> *imageIn, cuArrays<float2> *imageOut, int offsetX, int offsetY, cudaStream_t stream)
{
	const int nthreads = 16;
	dim3 threadsperblock(nthreads, nthreads);
	dim3 blockspergrid(IDIVUP(imageIn->height,nthreads), IDIVUP(imageIn->width,nthreads));
	cuArraysCopyInsert_kernel<<<blockspergrid, threadsperblock,0, stream>>>(imageIn->devData, imageIn->height, imageIn->width,
	       imageOut->devData,  imageOut->width, offsetX, offsetY);
	getLastCudaError("cuArraysCopyInsert error");
}
//
// float3
__global__ void cuArraysCopyInsert_kernel(const float3* imageIn, const int inNX, const int inNY,
   float3* imageOut, const int outNY, const int offsetX, const int offsetY)
{
	int inx = threadIdx.x + blockDim.x*blockIdx.x;
	int iny = threadIdx.y + blockDim.y*blockIdx.y;
	if(inx < inNX && iny < inNY) {
		int idxOut = IDX2R(inx+offsetX, iny+offsetY, outNY);
		int idxIn = IDX2R(inx, iny, inNY);
		imageOut[idxOut] = make_float3(imageIn[idxIn].x, imageIn[idxIn].y, imageIn[idxIn].z);
	}
}

void cuArraysCopyInsert(cuArrays<float3> *imageIn, cuArrays<float3> *imageOut, int offsetX, int offsetY, cudaStream_t stream)
{
	const int nthreads = 16;
	dim3 threadsperblock(nthreads, nthreads);
	dim3 blockspergrid(IDIVUP(imageIn->height,nthreads), IDIVUP(imageIn->width,nthreads));
	cuArraysCopyInsert_kernel<<<blockspergrid, threadsperblock,0, stream>>>(imageIn->devData, imageIn->height, imageIn->width,
	       imageOut->devData,  imageOut->width, offsetX, offsetY);
	getLastCudaError("cuArraysCopyInsert error");
}

//

__global__ void cuArraysCopyInsert_kernel(const float* imageIn, const int inNX, const int inNY,
   float* imageOut, const int outNY, const int offsetX, const int offsetY)
{
	int inx = threadIdx.x + blockDim.x*blockIdx.x;
	int iny = threadIdx.y + blockDim.y*blockIdx.y;
	if(inx < inNX && iny < inNY) {
		int idxOut = IDX2R(inx+offsetX, iny+offsetY, outNY);
		int idxIn = IDX2R(inx, iny, inNY);
		imageOut[idxOut] = imageIn[idxIn];
	}
}


void cuArraysCopyInsert(cuArrays<float> *imageIn, cuArrays<float> *imageOut, int offsetX, int offsetY, cudaStream_t stream)
{
	const int nthreads = 16;
	dim3 threadsperblock(nthreads, nthreads);
	dim3 blockspergrid(IDIVUP(imageIn->height,nthreads), IDIVUP(imageIn->width,nthreads));
	cuArraysCopyInsert_kernel<<<blockspergrid, threadsperblock,0, stream>>>(imageIn->devData, imageIn->height, imageIn->width,
	       imageOut->devData,  imageOut->width, offsetX, offsetY);
	getLastCudaError("cuArraysCopyInsert Float error");
}

//

__global__ void cuArraysCopyInsert_kernel(const int* imageIn, const int inNX, const int inNY,
   int* imageOut, const int outNY, const int offsetX, const int offsetY)
{
	int inx = threadIdx.x + blockDim.x*blockIdx.x;
	int iny = threadIdx.y + blockDim.y*blockIdx.y;
	if(inx < inNX && iny < inNY) {
		int idxOut = IDX2R(inx+offsetX, iny+offsetY, outNY);
		int idxIn = IDX2R(inx, iny, inNY);
		imageOut[idxOut] = imageIn[idxIn];
	}
}


void cuArraysCopyInsert(cuArrays<int> *imageIn, cuArrays<int> *imageOut, int offsetX, int offsetY, cudaStream_t stream)
{
	const int nthreads = 16;
	dim3 threadsperblock(nthreads, nthreads);
	dim3 blockspergrid(IDIVUP(imageIn->height,nthreads), IDIVUP(imageIn->width,nthreads));
	cuArraysCopyInsert_kernel<<<blockspergrid, threadsperblock,0, stream>>>(imageIn->devData, imageIn->height, imageIn->width,
	       imageOut->devData,  imageOut->width, offsetX, offsetY);
	getLastCudaError("cuArraysCopyInsert Integer error");
}
//


__global__ void cuArraysCopyInversePadded_kernel(float *imageIn, int inNX, int inNY, int sizeIn,
    float *imageOut, int outNX, int outNY, int sizeOut, int nImages)
{
	int outx = threadIdx.x + blockDim.x*blockIdx.x;
	int outy = threadIdx.y + blockDim.y*blockIdx.y;

	if(outx < outNX && outy < outNY)
	{
		int idxImage = blockIdx.z;
		int idxOut = IDX2R(outx, outy, outNY)+idxImage*sizeOut;
		if(outx < inNX && outy <inNY) {
			int idxIn = IDX2R(inNX-outx-1, inNY-outy-1, inNY)+idxImage*sizeIn;
			imageOut[idxOut] = imageIn[idxIn];
		}
		else
		{	imageOut[idxOut] = 0.0f; }
	}
}

void cuArraysCopyInversePadded(cuArrays<float> *imageIn, cuArrays<float> *imageOut,cudaStream_t stream)
{
	const int nthreads = 16;
	int nImages = imageIn->count;
	dim3 blockSize(nthreads, nthreads,1);
	dim3 gridSize(IDIVUP(imageOut->height,nthreads), IDIVUP(imageOut->width,nthreads), nImages);
	cuArraysCopyInversePadded_kernel<<<gridSize, blockSize, 0, stream>>>(imageIn->devData, imageIn->height, imageIn->width, imageIn->size,
	   imageOut->devData, imageOut->height, imageOut->width, imageOut->size, nImages);
	 getLastCudaError("cuArraysCopyInversePadded error");
}


__global__ void cuArraysCopyPadded_R2R_kernel(float *imageIn, int inNX, int inNY, int sizeIn,
    float *imageOut, int outNX, int outNY, int sizeOut, int nImages)
{
	int outx = threadIdx.x + blockDim.x*blockIdx.x;
	int outy = threadIdx.y + blockDim.y*blockIdx.y;

	if(outx < outNX && outy < outNY)
	{
		int idxImage = blockIdx.z;
		int idxOut = IDX2R(outx, outy, outNY)+idxImage*sizeOut;
		if(outx < inNX && outy <inNY) {
			int idxIn = IDX2R(outx, outy, inNY)+idxImage*sizeIn;
			imageOut[idxOut] = imageIn[idxIn];
		}
		else
		{	imageOut[idxOut] = 0.0f; }
	}
}

void cuArraysCopyPadded(cuArrays<float> *imageIn, cuArrays<float> *imageOut,cudaStream_t stream)
{
	const int nthreads = 16;
	int nImages = imageIn->count;
	dim3 blockSize(nthreads, nthreads,1);
	dim3 gridSize(IDIVUP(imageOut->height,nthreads), IDIVUP(imageOut->width,nthreads), nImages);
	cuArraysCopyPadded_R2R_kernel<<<gridSize, blockSize, 0, stream>>>(imageIn->devData, imageIn->height, imageIn->width, imageIn->size,
	   imageOut->devData, imageOut->height, imageOut->width, imageOut->size, nImages);
    getLastCudaError("cuArraysCopyPaddedR2R error");
}

__global__ void cuArraysCopyPadded_C2C_kernel(float2 *imageIn, int inNX, int inNY, int sizeIn,
    float2 *imageOut, int outNX, int outNY, int sizeOut, int nImages)
{
	int outx = threadIdx.x + blockDim.x*blockIdx.x;
	int outy = threadIdx.y + blockDim.y*blockIdx.y;

	if(outx < outNX && outy < outNY)
	{
		int idxImage = blockIdx.z;
		int idxOut = IDX2R(outx, outy, outNY)+idxImage*sizeOut;
		if(outx < inNX && outy <inNY) {
			int idxIn = IDX2R(outx, outy, inNY)+idxImage*sizeIn;
			imageOut[idxOut] = imageIn[idxIn];
		}
		else{
            imageOut[idxOut] = make_float2(0.0f, 0.0f);
        }
	}
}


void cuArraysCopyPadded(cuArrays<float2> *imageIn, cuArrays<float2> *imageOut,cudaStream_t stream)
{
	const int nthreads = NTHREADS2D;
	int nImages = imageIn->count;
	dim3 blockSize(nthreads, nthreads,1);
	dim3 gridSize(IDIVUP(imageOut->height,nthreads), IDIVUP(imageOut->width,nthreads), nImages);
	cuArraysCopyPadded_C2C_kernel<<<gridSize, blockSize, 0, stream>>>
        (imageIn->devData, imageIn->height, imageIn->width, imageIn->size,
	    imageOut->devData, imageOut->height, imageOut->width, imageOut->size, nImages);
	 getLastCudaError("cuArraysCopyInversePadded error");
}

__global__ void cuArraysCopyPadded_R2C_kernel(float *imageIn, int inNX, int inNY, int sizeIn,
    float2 *imageOut, int outNX, int outNY, int sizeOut, int nImages)
{
	int outx = threadIdx.x + blockDim.x*blockIdx.x;
	int outy = threadIdx.y + blockDim.y*blockIdx.y;

	if(outx < outNX && outy < outNY)
	{
		int idxImage = blockIdx.z;
		int idxOut = IDX2R(outx, outy, outNY)+idxImage*sizeOut;
		if(outx < inNX && outy <inNY) {
			int idxIn = IDX2R(outx, outy, inNY)+idxImage*sizeIn;
			imageOut[idxOut] = make_float2(imageIn[idxIn], 0.0f);
		}
		else{
            imageOut[idxOut] = make_float2(0.0f, 0.0f);
        }
	}
}


void cuArraysCopyPadded(cuArrays<float> *imageIn, cuArrays<float2> *imageOut,cudaStream_t stream)
{
	const int nthreads = NTHREADS2D;
	int nImages = imageIn->count;
	dim3 blockSize(nthreads, nthreads,1);
	dim3 gridSize(IDIVUP(imageOut->height,nthreads), IDIVUP(imageOut->width,nthreads), nImages);
	cuArraysCopyPadded_R2C_kernel<<<gridSize, blockSize, 0, stream>>>
        (imageIn->devData, imageIn->height, imageIn->width, imageIn->size,
	    imageOut->devData, imageOut->height, imageOut->width, imageOut->size, nImages);
	 getLastCudaError("cuArraysCopyPadded error");
}


__global__ void cuArraysSetConstant_kernel(float *image, int size, float value)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;

	if(idx < size)
	{
		image[idx] = value;
	}
}

void cuArraysSetConstant(cuArrays<float> *imageIn, float value, cudaStream_t stream)
{
	const int nthreads = 256;
	int size = imageIn->getSize();

	cuArraysSetConstant_kernel<<<IDIVUP(size, nthreads), nthreads, 0, stream>>>
        (imageIn->devData, imageIn->size, value);
	 getLastCudaError("cuArraysCopyPadded error");
}
