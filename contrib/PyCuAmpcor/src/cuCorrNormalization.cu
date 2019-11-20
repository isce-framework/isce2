/* 
 * cuCorrNormalization.cu
 * various utilities related to normalization of images
 * including calculating mean, subtract mean, ....
 * 
 */
 
#include "cuAmpcorUtil.h" 
#include <cfloat>
 
template <const int Nthreads>
__device__ float sumReduceBlock(float sum, volatile float *shmem)
{
    const int tid = threadIdx.x;
    shmem[tid] = sum;
    __syncthreads();
    
    if (Nthreads >=1024) { if (tid < 512) { shmem[tid] += shmem[tid + 512]; } __syncthreads(); }
    if (Nthreads >= 512) { if (tid < 256) { shmem[tid] += shmem[tid + 256]; } __syncthreads(); }
    if (Nthreads >= 256) { if (tid < 128) { shmem[tid] += shmem[tid + 128]; } __syncthreads(); }
    if (Nthreads >= 128) { if (tid <  64) { shmem[tid] += shmem[tid +  64]; } __syncthreads(); }
    if (tid < 32)
    {
        shmem[tid] += shmem[tid + 32]; 
        shmem[tid] += shmem[tid + 16];
        shmem[tid] += shmem[tid +  8];
        shmem[tid] += shmem[tid +  4];
        shmem[tid] += shmem[tid +  2];
        shmem[tid] += shmem[tid +  1]; 
    }
    
    __syncthreads();
    return shmem[0];
}
 
/* subtracts mean value from the images */
template<const int Nthreads>
__global__ void cuArraysMean_kernel(float *images, float *image_sum, int imageSize, float invSize, int nImages)
{
    __shared__ float shmem[Nthreads];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    if (bid >= nImages) return;
    
    const int       imageIdx = bid;
    const int imageOffset = imageIdx * imageSize;
    float   *imageD = images + imageOffset;
    
    float sum  = 0.0f;
    for (int i = tid; i < imageSize; i += Nthreads)
            sum += imageD[i];
    sum = sumReduceBlock<Nthreads>(sum, shmem);
    
    const float mean = sum * invSize;
    if(tid ==0) image_sum[bid] = mean;
} 

void cuArraysMeanValue(cuArrays<float> *images, cuArrays<float> *mean, cudaStream_t stream)
{
	const dim3 grid(images->count, 1, 1);
	//const int Nthreads=512;
	const int imageSize = images->width*images->height;
	const float invSize = 1.0f/imageSize;
    
	cuArraysMean_kernel<512> <<<grid,512,0,stream>>>(images->devData, mean->devData, imageSize, invSize, images->count);
	getLastCudaError("cuArraysMeanValue kernel error\n");
}



/* subtracts mean value from the images */
template<const int Nthreads>
__global__ void cuArraysSubtractMean_kernel(float *images, int imageSize, float invSize, int nImages)
{
    __shared__ float shmem[Nthreads];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    if (bid >= nImages) return;
    
    const int       imageIdx = bid;
    const int imageOffset = imageIdx * imageSize;
    float   *imageD = images + imageOffset;
    
    float sum  = 0.0f;
    for (int i = tid; i < imageSize; i += Nthreads)
            sum += imageD[i];
    sum = sumReduceBlock<Nthreads>(sum, shmem);
    
    const float mean = sum * invSize;
    
    for (int i = tid; i < imageSize; i += Nthreads)
            imageD[i] -= mean;
} 

void cuArraysSubtractMean(cuArrays<float> *images, cudaStream_t stream)
{
	const dim3 grid(images->count, 1, 1);
	//const int Nthreads=512;
	const int imageSize = images->width*images->height;
	const float invSize = 1.0f/imageSize;
    
	cuArraysSubtractMean_kernel<512> <<<grid,512,0,stream>>>(images->devData, imageSize, invSize, images->count);
	getLastCudaError("cuArraysSubtractMean kernel error\n");
}


// Summation on extracted correlation surface (Minyan)
template<const int Nthreads>
__global__ void cuArraysSumCorr_kernel(float *images, int *imagesValid, float *imagesSum, int *imagesValidCount, int imageSize, int nImages)
{
    __shared__ float shmem[Nthreads];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (bid >= nImages) return;

    const int imageIdx = bid;
    const int imageOffset = imageIdx * imageSize;
    float*    imageD = images + imageOffset;
    int*      imageValidD = imagesValid + imageOffset;

    float sum  = 0.0f;
    int count = 0;

    for (int i = tid; i < imageSize; i += Nthreads) {
            sum += imageD[i] * imageD[i];
            count += imageValidD[i]; 
    }   

    sum = sumReduceBlock<Nthreads>(sum, shmem);
    count = sumReduceBlock<Nthreads>(count, shmem);

    if(tid ==0) {
        imagesSum[bid] = sum;
        imagesValidCount[bid] = count;
    }
}

void cuArraysSumCorr(cuArrays<float> *images, cuArrays<int> *imagesValid, cuArrays<float> *imagesSum, cuArrays<int> *imagesValidCount, cudaStream_t stream)
{
const dim3 grid(images->count, 1, 1);
//const int Nthreads=512;
const int imageSize = images->width*images->height;

cuArraysSumCorr_kernel<512> <<<grid,512,0,stream>>>(images->devData, imagesValid->devData,
imagesSum->devData, imagesValidCount->devData, imageSize, images->count);

getLastCudaError("cuArraysSumValueCorr kernel error\n");

}

// end of summation on extracted correlation surface (Minyan)



/* intra-block inclusive prefix sum */
template<int Nthreads2>
__device__ void inclusive_prefix_sum(float sum, volatile float *shmem)
{
    const int tid = threadIdx.x;
    shmem[tid] = sum;
    __syncthreads();

#pragma unroll
    for (int i = 0; i < Nthreads2; i++)
    {
        const int offset = 1 << i;
        if (tid >= offset) sum += shmem[tid - offset]; 
        __syncthreads();
        shmem[tid] = sum;
        __syncthreads();
    }
}


template<const int Nthreads2>
__device__ float2 partialSums(const float v, volatile float* shmem, const int stride)
{
    const int tid = threadIdx.x;
    
    volatile float *shMem  = shmem + 1;
    volatile float *shMem2 = shMem + 1 + (1 << Nthreads2);
    
    inclusive_prefix_sum<Nthreads2>(v,   shMem);
    inclusive_prefix_sum<Nthreads2>(v*v, shMem2);
    const float Sum  = shMem [tid-1 + stride] - shMem [tid-1];
    const float Sum2 = shMem2[tid-1 + stride] - shMem2[tid-1];
    //__syncthreads();
    
    return make_float2(Sum, Sum2);
} 


template<const int Nthreads2>
__global__ void cuCorrNormalize_kernel(
	int nImages, 
    const float *templateIn, int templateNX, int templateNY, int templateSize,
    const float *imageIn, int imageNX, int imageNY, int imageSize, 
    float *resultOut, int resultNX, int resultNY, int resultSize,
    float templateCoeff)
{
    const int Nthreads = 1<<Nthreads2; 
    __shared__ float shmem[Nthreads*3];
    
    const int tid = threadIdx.x;
    const int imageIdx = blockIdx.z; 
    if (imageIdx >= nImages) return;
    
    //if(tid ==0 ) printf("debug corrNorm, %d %d %d %d %d %d %d %d %d\n", templateNX, templateNY, templateSize,
    //imageNX, imageNY, imageSize,     resultNX, resultNY, resultSize);
    
    const int    imageOffset = imageIdx *    imageSize;
    const int templateOffset = imageIdx * templateSize;
    const int   resultOffset = imageIdx *   resultSize;
    
    const float *   imageD =    imageIn  +    imageOffset;
    const float *templateD = templateIn  + templateOffset;
    float *  resultD =   resultOut +   resultOffset;
    
    /*template sum squar */
    
    float templateSum = 0.0f;
    
    for(uint i=tid; i<templateSize; i+=Nthreads)
    {
        templateSum += templateD[i];
    }
    templateSum = sumReduceBlock<Nthreads>(templateSum, shmem);
    __syncthreads();
    
    float templateSum2 = 0.0f;
    for (int i = tid; i < templateSize; i += Nthreads)
        {
            const float t = templateD[i];
            templateSum2 += t*t;
        }
    templateSum2 = sumReduceBlock<Nthreads>(templateSum2, shmem);
    __syncthreads();

    //if(tid ==0) printf("template sum %d %g %g \n", imageIdx, templateSum, templateSum2);
    /*********/

    shmem[tid] = shmem[tid + Nthreads] = shmem[tid + 2*Nthreads] = 0.0f;
    __syncthreads();
    
    float imageSum  = 0.0f;
    float imageSum2 = 0.0f;
    int iaddr = 0;
    const int windowSize = templateNX*imageNY;
    while (iaddr < windowSize)
    {
        const float2 res = partialSums<Nthreads2>(imageD[iaddr + tid], shmem, templateNY);
        imageSum  += res.x;
        imageSum2 += res.y;
        iaddr     += imageNY;
    }
    
    if (tid < resultNY)
    {
        //if(blockIdx.z ==0) printf("image sum %d %g %g \n", tid, imageSum*templateCoeff, sqrtf(imageSum2*templateCoeff));
        
        const float norm2 = (imageSum2 - imageSum*imageSum*templateCoeff)*templateSum2;
        resultD[tid] *= rsqrtf(norm2 + FLT_EPSILON);
    } 
    
    /*********/
    
    while (iaddr < imageSize)
    {
        const float2 res1 = partialSums<Nthreads2>(imageD[iaddr-windowSize + tid], shmem, templateNY);
        const float2 res2 = partialSums<Nthreads2>(imageD[iaddr            + tid], shmem, templateNY);
        imageSum  += res2.x - res1.x;
        imageSum2 += res2.y - res1.y;
        iaddr     += imageNY;
        
        if (tid < resultNY)
        {
            const int         ix = iaddr/imageNY;
            const int       addr = (ix-templateNX)*resultNY;
            
            //printf("test norm %d %d %d %d %f\n", tid, ix, addr, addr+tid, resultD[addr + tid]);
            
            const float    norm2 = (imageSum2 - imageSum*imageSum*templateCoeff)*templateSum2;
            resultD[addr + tid] *= rsqrtf(norm2 + FLT_EPSILON);
        }
    }
}  

void cuCorrNormalize(cuArrays<float> *templates, cuArrays<float> *images, cuArrays<float> *results, cudaStream_t stream)
{
	const int nImages = images->count; 
	const int imageNY = images->width;
	const dim3 grid(1, 1, nImages);
	const float invTemplateSize = 1.0f/templates->size;
    //printf("test normalize %d %g\n", templates->size, invTemplateSize);
	if      (imageNY <=   64) cuCorrNormalize_kernel< 6><<<grid,  64, 0, stream>>>(nImages, 
		templates->devData, templates->height, templates->width, templates->size, 
		images->devData, images->height, images->width, images->size,
		results->devData, results->height, results->width, results->size, 
		invTemplateSize);
    else if (imageNY <=  128) cuCorrNormalize_kernel< 7><<<grid, 128, 0, stream>>>(nImages, 
		templates->devData, templates->height, templates->width, templates->size, 
		images->devData, images->height, images->width, images->size,
		results->devData, results->height, results->width, results->size, 
		invTemplateSize);
    else if (imageNY <=  256) cuCorrNormalize_kernel< 8><<<grid, 256, 0, stream>>>(nImages, 
		templates->devData, templates->height, templates->width, templates->size, 
		images->devData, images->height, images->width, images->size,
		results->devData, results->height, results->width, results->size, 
		invTemplateSize);
    else if (imageNY <=  512) cuCorrNormalize_kernel< 9><<<grid, 512, 0, stream>>>(nImages, 
		templates->devData, templates->height, templates->width, templates->size, 
		images->devData, images->height, images->width, images->size,
		results->devData, results->height, results->width, results->size, 
		invTemplateSize);
    else if (imageNY <= 1024) cuCorrNormalize_kernel<10><<<grid,1024, 0, stream>>>(nImages, 
		templates->devData, templates->height, templates->width, templates->size, 
		images->devData, images->height, images->width, images->size,
		results->devData, results->height, results->width, results->size, 
		invTemplateSize);
    else 
    {
        fprintf(stderr, "The image size along across direction %d should be smaller than 1024.\n", imageNY);
        assert(0);
    }
    getLastCudaError("cuCorrNormalize kernel error\n");
	
}


