/* 
 * cuCorrTimetime.cu 
 * correlation between two sets of images in time domain
 */

#include "cuAmpcorUtil.h"

template<const int nthreads, const int NPT>
__global__ void cuArraysCorrTime_kernel(
	const int nImages, 
	const float *templateIn, const int templateNY, const int templateNX, const int templateSize,   
	const float *imageIn, const int imageNY, const int imageNX, const int imageSize,  
    float *resultOut, const int resultNY, const int resultNX, const int resultSize)
{
    __shared__ float shmem[nthreads*(1+NPT)];
    const int tid = threadIdx.x;
    const int bid =  blockIdx.x;
    const int  yc =  blockIdx.y*NPT;
    
    const int       imageIdx = bid;
    const int    imageOffset = imageIdx *    imageSize;
    const int templateOffset = imageIdx * templateSize;
    const int   resultOffset = imageIdx *   resultSize;
    
    const float *   imageD =    imageIn  +    imageOffset + tid;
    const float *templateD = templateIn  + templateOffset + tid;
     float *  resultD =   resultOut +   resultOffset;
    
    const int q  = min(nthreads/resultNX, 4);
    const int nt = nthreads/q;
    const int ty = threadIdx.x / nt;
    const int tx = threadIdx.x - nt * ty;
    
    const int templateNXq = templateNX/q;
    const int jbeg = templateNXq * ty;
    const int jend = ty+1 >= q ? templateNX : templateNXq + jbeg;
    
    float *shTemplate = shmem;
    float *shImage    = shmem + nthreads;
    float *shImage1   = shImage + tx;
    
    float corrCoeff[NPT];
    for (int k = 0; k < NPT; k++)
        corrCoeff[k] = 0.0f;
    
    int iaddr = yc*imageNX;
    

    float img[NPT];
    for (int k = 0; k < NPT-1; k++, iaddr += imageNX)
        img[k] = imageD[iaddr]; 
    for (int taddr = 0; taddr < templateSize; taddr += templateNX, iaddr += imageNX)
    {
        shTemplate[tid] = templateD[taddr];
        img     [NPT-1] =    imageD[iaddr];
        for (int k = 0; k < NPT; k++)
            shImage[tid + nthreads*k] = img[k];
        for (int k = 0; k < NPT-1; k++)
            img[k] = img[k+1];
        __syncthreads();
        
        if (tx < resultNX && ty < q)
        {
#pragma unroll 8  
            for (int j = jbeg; j < jend; j++)
                for (int k = 0; k < NPT; k++)
                    corrCoeff[k] += shTemplate[j]*shImage1[j + nthreads*k];
        }
        __syncthreads();
    }

    for (int k = 0; k < NPT; k++)
        shmem[tid + nthreads*k] = corrCoeff[k];
    __syncthreads();
    
    for (int j = tx + nt; j < nthreads; j += nt)
        for (int k = 0; k < NPT; k++)
            corrCoeff[k] += shmem[j + nthreads*k];
    __syncthreads();
    
    if (tid < resultNX)
    {
        int raddr = yc*resultNX + tid;
        for (int k = 0; k < NPT; k++, raddr += resultNX)
            if (raddr < resultSize)
                resultD[raddr] = corrCoeff[k];
    }
}


void cuCorrTimeDomain(cuArrays<float> *templates,
			   cuArrays<float> *images,
			   cuArrays<float> *results,
			   cudaStream_t stream)
{
    /* compute correlation matrix */
    const int nImages = images->count;
    const int imageNX = images->width;
    const int NPT = 8;
    
    
    const dim3 grid(nImages, (results->width-1)/NPT+1, 1);
    //fprintf(stderr, "corrTimeDomain %d %d %d\n", imageNX, templates->height, results->height);
    if      (imageNX <=   64) cuArraysCorrTime_kernel<  64,NPT><<<grid,  64, 0, stream>>>(nImages, 
		templates->devData, templates->height, templates->width, templates->size, 
		images->devData, images->height, images->width, images->size,
		results->devData, results->height, results->width, results->size);
    else if (imageNX <=  128) cuArraysCorrTime_kernel< 128,NPT><<<grid, 128, 0, stream>>>(nImages, 
		templates->devData, templates->height, templates->width, templates->size, 
		images->devData, images->height, images->width, images->size,
		results->devData, results->height, results->width, results->size);
    else if (imageNX <=  192) cuArraysCorrTime_kernel< 192,NPT><<<grid, 192, 0, stream>>>(nImages, 
		templates->devData, templates->height, templates->width, templates->size, 
		images->devData, images->height, images->width, images->size,
		results->devData, results->height, results->width, results->size);
    else if (imageNX <=  256) cuArraysCorrTime_kernel< 256,NPT><<<grid, 256, 0, stream>>>(nImages, 
		templates->devData, templates->height, templates->width, templates->size, 
		images->devData, images->height, images->width, images->size,
		results->devData, results->height, results->width, results->size);
    else if (imageNX <=  384) cuArraysCorrTime_kernel< 384,NPT><<<grid, 384, 0, stream>>>(nImages, 
		templates->devData, templates->height, templates->width, templates->size, 
		images->devData, images->height, images->width, images->size,
		results->devData, results->height, results->width, results->size);
    else if (imageNX <=  512) cuArraysCorrTime_kernel< 512,NPT><<<grid, 512, 0, stream>>>(nImages, 
		templates->devData, templates->height, templates->width, templates->size, 
		images->devData, images->height, images->width, images->size,
		results->devData, results->height, results->width, results->size);
    else if (imageNX <=  640) cuArraysCorrTime_kernel< 640,NPT><<<grid, 640, 0, stream>>>(nImages, 
		templates->devData, templates->height, templates->width, templates->size, 
		images->devData, images->height, images->width, images->size,
		results->devData, results->height, results->width, results->size);
    else if (imageNX <=  768) cuArraysCorrTime_kernel< 768,NPT><<<grid, 768, 0, stream>>>(nImages, 
		templates->devData, templates->height, templates->width, templates->size, 
		images->devData, images->height, images->width, images->size,
		results->devData, results->height, results->width, results->size);
    else if (imageNX <=  896) cuArraysCorrTime_kernel< 896,NPT><<<grid, 896, 0, stream>>>(nImages, 
		templates->devData, templates->height, templates->width, templates->size, 
		images->devData, images->height, images->width, images->size,
		results->devData, results->height, results->width, results->size);
    else if (imageNX <= 1024) cuArraysCorrTime_kernel<1024,NPT><<<grid,1024, 0, stream>>>(nImages, 
		templates->devData, templates->height, templates->width, templates->size, 
		images->devData, images->height, images->width, images->size,
		results->devData, results->height, results->width, results->size);
    else assert(0);
	getLastCudaError("cuArraysCorrTime error");
}
