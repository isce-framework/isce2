/*
 * cuDeramp.cu 
 * Derampling a batch of 2D complex images with GPU
 * 
 * Method 1: use Fortran code algorithm
 * Method 2: use phase gradient
 * Method 0 or else: no deramping
 * 
 * v1.0 2/1/2017, Lijun Zhu
 */
 
#include "cuArrays.h" 
#include "float2.h" 
#include <cfloat>
#include "cudaError.h"
#include "cudaUtil.h"
#include "cuAmpcorUtil.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
// note by Lijun
// cuda does not have a good support on volatile vector struct, e.g. float2
// I have to use regular float type for shared memory (volatile)

inline static __device__ void copyToShared(volatile float *s, const int i, const float2 x, const int block) 
{ s[i] = x.x; s[i+block] = x.y; }

inline static __device__ void copyFromShared(float2 &x, volatile float *s, const int i, const int block) 
{ x.x = s[i]; x.y = s[i+block]; }


inline static __device__ void addInShared(volatile float *s, const int i, const int j, const int block) 
{ s[i] += s[i+j]; s[i+block] += s[i+j+block];}


__device__ void debugPhase(float2 c1, float2 c2)
{
    float2 cp = complexMulConj(c1, c2);
    float phase = atan2f(cp.y, cp.x);
} 


template <const int nthreads>
__device__ float sumReduceBlock(float sum, volatile float *shmem)
{
    const int tid = threadIdx.x;
    shmem[tid] = sum;
    __syncthreads();
    
    if (nthreads >=1024) { if (tid < 512) { shmem[tid] = sum = sum + shmem[tid + 512]; } __syncthreads(); }
    if (nthreads >= 512) { if (tid < 256) { shmem[tid] = sum = sum + shmem[tid + 256]; } __syncthreads(); }
    if (nthreads >= 256) { if (tid < 128) { shmem[tid] = sum = sum + shmem[tid + 128]; } __syncthreads(); }
    if (nthreads >= 128) { if (tid <  64) { shmem[tid] = sum = sum + shmem[tid +  64]; } __syncthreads(); }
    if (tid < 32)
    {
        shmem[tid] = sum = sum + shmem[tid + 32];
        shmem[tid] = sum = sum + shmem[tid + 16]; 
        shmem[tid] = sum = sum + shmem[tid +  8];
        shmem[tid] = sum = sum + shmem[tid +  4];
        shmem[tid] = sum = sum + shmem[tid +  2];
        shmem[tid] = sum = sum + shmem[tid +  1]; 
    }
    
    __syncthreads();
    return shmem[0];
}


template<const int nthreads>
__global__ void cuDerampMethod2_kernel(float2 *images, const int imageNX, const int imageNY, 
const int imageSize, const int nImages, const float normCoefX, const float normCoefY)
{
    
    __shared__ float shmem[nthreads];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;    
    //printf("bid %d\n", bid);
    float2  *imageD = images + bid*imageSize;
    
    int pixelIdx, pixelIdxX, pixelIdxY;
    
    float phaseDiffY  = 0.0f;
    for (int i = tid; i < imageSize; i += nthreads) {
        pixelIdx = i;
        pixelIdxY = pixelIdx % imageNY; 
        if(pixelIdxY < imageNY -1) 
        {
            phaseDiffY += complexArg(complexMulConj(imageD[pixelIdx], imageD[pixelIdx+1]));
        } 
    }       
    
    phaseDiffY=sumReduceBlock<nthreads>(phaseDiffY, shmem);
    phaseDiffY*=normCoefY;
    
    float phaseDiffX  = 0.0f;
    for (int i = tid; i < imageSize; i += nthreads) {
        pixelIdx = i;
        pixelIdxX = pixelIdx / imageNY; 
        if(pixelIdxX < imageNX -1) 
        {
            phaseDiffX += complexArg(complexMulConj(imageD[pixelIdx], imageD[pixelIdx+imageNY]));
        } 
    }      
    phaseDiffX=sumReduceBlock<nthreads>(phaseDiffX, shmem);
    phaseDiffX*=normCoefX; 
    
    for (int i = tid; i < imageSize; i += nthreads)
    { 
        int pixelIdx = i;
        pixelIdxX = pixelIdx/imageNY;
        pixelIdxY = pixelIdx%imageNY;
        float phase = pixelIdxX*phaseDiffX + pixelIdxY*phaseDiffY;
        imageD[pixelIdx] *= make_float2(cosf(phase), sinf(phase));
    }
}


void cuDerampMethod2(cuArrays<float2> *images, cudaStream_t stream)
{
    const dim3 grid(images->count);
    const int nthreads=512;
    const int imageSize = images->width*images->height;
    const float normCoefY = 1.0f/((images->width-1)*images->height);
    const float normCoefX = 1.0f/((images->height-1)*images->width);
    cuDerampMethod2_kernel<nthreads> <<<grid, 512,0,stream>>>
        (images->devData, images->height, images->width, imageSize, images->count, normCoefX, normCoefY);
    getLastCudaError("cuDerampMethod2 kernel error\n");
}



template <const int nthreads>
__device__ void complexSumReduceBlock(float2& sum, volatile float *shmem)
{
    const int tid = threadIdx.x;
    copyToShared(shmem, tid, sum, nthreads);
    __syncthreads();
    
    if (nthreads >=1024) { if (tid < 512) { addInShared(shmem, tid, 512, nthreads); } __syncthreads(); }
    if (nthreads >= 512) { if (tid < 256) { addInShared(shmem, tid, 256, nthreads); } __syncthreads(); }
    if (nthreads >= 256) { if (tid < 128) { addInShared(shmem, tid, 128, nthreads); } __syncthreads(); }
    if (nthreads >= 128) { if (tid <  64) { addInShared(shmem, tid,  64, nthreads); } __syncthreads(); }
    if (tid < 32)
    {	
        addInShared(shmem, tid, 32, nthreads);
        addInShared(shmem, tid, 16, nthreads);
        addInShared(shmem, tid,  8, nthreads);
        addInShared(shmem, tid,  4, nthreads);
        addInShared(shmem, tid,  2, nthreads);
        addInShared(shmem, tid,  1, nthreads); 
    }
    __syncthreads();
    copyFromShared(sum, shmem, 0, nthreads);
}


// block id is the image index
// thread id ranges all pixels in one image
template<const int nthreads>
__global__ void cuDerampMethod1_kernel(float2 *images, const int imageNX, int const imageNY, 
    const int imageSize, const int nImages, const float normCoef)
{
    __shared__ float shmem[2*nthreads];
    int pixelIdx, pixelIdxX, pixelIdxY;
    
    const int bid = blockIdx.x;    
    if(bid >= nImages) return;
    float2 *image = images+ bid*imageSize;
    const int tid = threadIdx.x;  
    float2 phaseDiffY  = make_float2(0.0f, 0.0f);
    for (int i = tid; i < imageSize; i += nthreads) {
        pixelIdxY = i % imageNY;
        if(pixelIdxY < imageNY -1) {
            pixelIdx = i;
            float2 cprod = complexMulConj( image[pixelIdx], image[pixelIdx+1]);   
            phaseDiffY += cprod;
        } 
    }       
    complexSumReduceBlock<nthreads>(phaseDiffY, shmem);
    //phaseDiffY *= normCoef;
    float phaseY=atan2f(phaseDiffY.y, phaseDiffY.x);
    //__syncthreads();

    float2 phaseDiffX  = make_float2(0.0f, 0.0f);
    for (int i = tid; i < imageSize; i += nthreads)  {
        pixelIdxX = i / imageNY; 
        if(pixelIdxX < imageNX -1) {
            pixelIdx = i;
            float2 cprod = complexMulConj(image[i], image[i+imageNY]);
            phaseDiffX += cprod;
        }
    }   
    
    complexSumReduceBlock<nthreads>(phaseDiffX, shmem);
   
    //phaseDiffX *= normCoef;
    float phaseX = atan2f(phaseDiffX.y, phaseDiffX.x);  //+FLT_EPSILON
     
    for (int i = tid; i < imageSize; i += nthreads)
    { 
        pixelIdxX = i%imageNY;
        pixelIdxY = i/imageNY;
        float phase = pixelIdxX*phaseX + pixelIdxY*phaseY;
        float2 phase_factor = make_float2(cosf(phase), sinf(phase));
        image[i] *= phase_factor;
    }     
}


void cuDerampMethod1(cuArrays<float2> *images, cudaStream_t stream)
{
    
    const dim3 grid(images->count);
    //int nthreads;
    const int imageSize = images->width*images->height;
    const float invSize = 1.0f/imageSize;

    if(imageSize <=64) {
        cuDerampMethod1_kernel<64> <<<grid, 64, 0, stream>>>
        (images->devData, images->height, images->width, 
        imageSize, images->count, invSize); }
     else if(imageSize <=128) {
        cuDerampMethod1_kernel<128> <<<grid, 128, 0, stream>>>
        (images->devData, images->height, images->width, 
        imageSize, images->count, invSize); }   
     else if(imageSize <=256) {
        cuDerampMethod1_kernel<256> <<<grid, 256, 0, stream>>>
        (images->devData, images->height, images->width, 
        imageSize, images->count, invSize); }  
    else  {
        cuDerampMethod1_kernel<512> <<<grid, 512, 0, stream>>>
        (images->devData, images->height, images->width, 
        imageSize, images->count, invSize); }  
    
    getLastCudaError("cuDerampMethod1 kernel error\n");

    
}



/*
static inline double complexAbs (double2 a)
{
    double r = sqrt(a.x*a.x + a.y*a.y);
    return r;
}*/



void cpuDerampMethod3(cuArrays<float2> *imagesD, cudaStream_t stream)
{
    float2 *images = (float2 *) malloc(imagesD->getByteSize());
    float2 phaseDiffX, phaseDiffY;
    int idxPixel;

    
    cudaMemcpyAsync(images, imagesD->devData, imagesD->getByteSize(), cudaMemcpyDeviceToHost, stream);
    
    int count = imagesD->count;
    int height = imagesD->height;
    int width = imagesD->width;
    float2 cprod; 
    float phaseX, phaseY;
    for (int icount = 0; icount < count; icount ++)
    {
        phaseDiffY = make_float2(0.0f, 0.0f);
        for (int i=0; i<height; i++)
        {
            for(int j=0; j<width-1; j++) 
            {
                idxPixel = icount*width*height + i*width + j;
                cprod = complexMulConj(images[idxPixel], images[idxPixel+1]);
                phaseDiffY.x  += (cprod.x);
                phaseDiffY.y  += (cprod.y);
            }         
        }
        //phaseDiffY /=  height*(width-1);
         if (complexAbs(phaseDiffY) < 1.e-5) {
            phaseY = 0.0;
        }
        else {  
            phaseY = atan2(phaseDiffY.y, phaseDiffY.x);
        }
            
        phaseDiffX = make_float2(0.0f, 0.0f);
        for (int j=0; j<width; j++)
        {
            for(int i=0; i<height-1; i++) {
                idxPixel = icount*width*height + i*width + j;
                cprod = complexMulConj(images[idxPixel], images[idxPixel+width]);
                phaseDiffX.x  += (cprod.x);
                phaseDiffX.y  += (cprod.y);;
            }
        }
        //phaseDiffX /=  (height-1)*width;
        if (complexAbs(phaseDiffX) < 1.e-5) {
            phaseX = 0.0;
        }
        else {  
            phaseX = atan2(phaseDiffX.y, phaseDiffX.x);
        }
        
        //printf("cpu deramp %d (%g,%g) (%g,%g)\n", icount, phaseDiffX.x, phaseDiffX.y, phaseDiffY.x, phaseDiffY.y);
        
        /*
        std::setprecision(12);
        std::cout << "cpu " << icount << " " << 
            std::setprecision(std::numeric_limits<long double>::digits10 + 1) << phaseX << 
            " " << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << phaseY << std::endl;
         std::cout << "cpu "  << phaseDiffX.x << " " << phaseDiffX.y << std::endl;
        std::cout << "cpu "  << phaseDiffY.x << " " << phaseDiffY.y << std::endl;
        */
        for(int i=0; i<height; i++)
        {
            for(int j=0; j<width; j++)
            {
                idxPixel = icount*width*height + i*width + j;
                float phase = phaseX*i + phaseY*j;
                images[idxPixel]*=make_float2(cos(phase), sin(phase));
            }
        }
    }
    cudaMemcpyAsync(imagesD->devData, images, imagesD->getByteSize(), cudaMemcpyHostToDevice, stream);
    free(images);
}
        
void cuDeramp(int method, cuArrays<float2> *images, cudaStream_t stream)
{
    // methods 2-3 are for test purposes only, removed for release
    // note method 0 is designed for TOPSAR: not only deramping is skipped,
    //    the amplitude is taken before oversampling
    switch(method) {
    //case 3:
    //    cpuDerampMethod3(images, stream);
    case 1:
        cuDerampMethod1(images, stream);
        break;
    //case 2:
    //    cuDerampMethod2(images, stream);
        break;
    default:
        break;
    }
}
