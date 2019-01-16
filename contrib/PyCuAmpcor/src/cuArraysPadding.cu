/*
 * cuArraysPadding.cu
 * Padding Utitilies for oversampling
 */

#include "cuAmpcorUtil.h"
#include "float2.h"

//padding zeros in the middle, move quads to corners  
//for raw chunk data oversampling
__global__ void cuArraysPadding_kernel(
	const float2 *image1, const int height1, const int width1,
	float2 *image2, const int height2, const int width2)
{
	int tx = threadIdx.x + blockDim.x*blockIdx.x;
	int ty = threadIdx.y + blockDim.y*blockIdx.y;  
	if(tx < height1/2 && ty < width1/2)
	{
		int tx1 = height1 - 1 - tx;
		int ty1 = width1 -1 -ty;
		int tx2 = height2 -1 -tx;
		int ty2 = width2 -1 -ty;  
		
		//printf("%d %d %d\n", tx, height1, height2); 
		 
		image2[IDX2R(tx, ty, width2)] = image1[IDX2R(tx, ty, width1)];
		image2[IDX2R(tx2, ty, width2)] = image1[IDX2R(tx1, ty, width1)];
		image2[IDX2R(tx, ty2, width2)] = image1[IDX2R(tx, ty1, width1)];
		image2[IDX2R(tx2, ty2, width2)] = image1[IDX2R(tx1, ty1, width1)];
		
	}
}
//tested 
void cuArraysPadding(cuArrays<float2> *image1, cuArrays<float2> *image2, cudaStream_t stream)
{
	int ThreadsPerBlock = NTHREADS2D;
	int BlockPerGridx = IDIVUP (image1->height/2, ThreadsPerBlock);
	int BlockPerGridy = IDIVUP (image1->width/2, ThreadsPerBlock);
	dim3 dimBlock(ThreadsPerBlock, ThreadsPerBlock);
	dim3 dimGrid(BlockPerGridx, BlockPerGridy);
	checkCudaErrors(cudaMemsetAsync(image2->devData, 0, image2->getByteSize(),stream));
	cuArraysPadding_kernel<<<dimGrid, dimBlock, 0, stream>>>(
		image1->devData, image1->height, image1->width,
		image2->devData, image2->height, image2->width);
	getLastCudaError("cuArraysPadding_kernel");
} 

inline __device__ float2 cmplxMul(float2 c, float a)
{
	return make_float2(c.x*a, c.y*a);
}

//padding for zoomIned correlation oversampling/interpolation 
__global__ void cuArraysPaddingMany_kernel(
	const float2 *image1, const int height1, const int width1, const int size1,
	float2 *image2, const int height2, const int width2, const int size2, const float factor )
{
	int tx = threadIdx.x + blockDim.x*blockIdx.x;
	int ty = threadIdx.y + blockDim.y*blockIdx.y;  
	if(tx < height1/2 && ty < width1/2)
	{  
		
		int tx1 = height1 - 1 - tx;
		int ty1 = width1 -1 -ty;
		int tx2 = height2 -1 -tx;
		int ty2 = width2 -1 -ty;  
		
		int stride1 = blockIdx.z*size1;
		int stride2 = blockIdx.z*size2;
		//printf("%d %d %d\n", tx, height1, height2); 
		
		image2[IDX2R(tx,  ty,  width2)+stride2] = image1[IDX2R(tx,  ty,  width1)+stride1]*factor;
		image2[IDX2R(tx2, ty,  width2)+stride2] = cmplxMul(image1[IDX2R(tx1, ty,  width1)+stride1], factor);
		image2[IDX2R(tx,  ty2, width2)+stride2] = cmplxMul(image1[IDX2R(tx,  ty1, width1)+stride1], factor);
		image2[IDX2R(tx2, ty2, width2)+stride2] = cmplxMul(image1[IDX2R(tx1, ty1, width1)+stride1], factor);
	}
}

void cuArraysPaddingMany(cuArrays<float2> *image1, cuArrays<float2> *image2, cudaStream_t stream)
{
	int ThreadsPerBlock = NTHREADS2D;
	int BlockPerGridx = IDIVUP (image1->height/2, ThreadsPerBlock);
	int BlockPerGridy = IDIVUP (image1->width/2, ThreadsPerBlock);
	dim3 dimBlock(ThreadsPerBlock, ThreadsPerBlock, 1);
	dim3 dimGrid(BlockPerGridx, BlockPerGridy, image1->count);
	
	checkCudaErrors(cudaMemsetAsync(image2->devData, 0, image2->getByteSize(),stream));
	float factor = 1.0f/image1->size;
	cuArraysPaddingMany_kernel<<<dimGrid, dimBlock, 0, stream>>>(
		image1->devData, image1->height, image1->width, image1->size,
		image2->devData, image2->height, image2->width, image2->size, factor);
	getLastCudaError("cuArraysPadding_kernel");
} 


// convert float to float2(complex)
__global__ void cuArraysR2C_kernel(float *image1, float2 *image2, int size)
{
	int idx =  threadIdx.x + blockDim.x*blockIdx.x;
	if(idx < size)
	{
		image2[idx].x = image1[idx];
		image2[idx].y =  0.0f;
	}
}

//tested
void cuArraysR2C(cuArrays<float> *image1, cuArrays<float2> *image2, cudaStream_t stream)
{
	int size = image1->getSize();
	cuArraysR2C_kernel<<<IDIVUP(size, NTHREADS), NTHREADS, 0, stream>>>(image1->devData, image2->devData, size);
	getLastCudaError("cuArraysR2C");
}


// take real part of float2 to float
__global__ void cuArraysC2R_kernel(float2 *image1, float *image2, int size)
{
	int idx =  threadIdx.x + blockDim.x*blockIdx.x;
	if(idx < size)
	{
		image2[idx] = image1[idx].x;
	}
}

//tested
void cuArraysC2R(cuArrays<float2> *image1, cuArrays<float> *image2, cudaStream_t stream)
{
	int size = image1->getSize();
	cuArraysC2R_kernel<<<IDIVUP(size, NTHREADS), NTHREADS, 0, stream>>>(image1->devData, image2->devData, size);
	getLastCudaError("cuArraysC2R");
}

// take real part of float2 to float
__global__ void cuArraysAbs_kernel(float2 *image1, float *image2, int size)
{
	int idx =  threadIdx.x + blockDim.x*blockIdx.x;
	if(idx < size)
	{
		image2[idx] = complexAbs(image1[idx]);
	}
}

//tested
void cuArraysAbs(cuArrays<float2> *image1, cuArrays<float> *image2, cudaStream_t stream)
{
	int size = image1->getSize();
	cuArraysAbs_kernel<<<IDIVUP(size, NTHREADS), NTHREADS, 0, stream>>>(image1->devData, image2->devData, size);
	getLastCudaError("cuArraysAbs_kernel");
}








