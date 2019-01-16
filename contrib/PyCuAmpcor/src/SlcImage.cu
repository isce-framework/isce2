#include "SlcImage.h"
#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <cuComplex.h>
#include <assert.h>
#include <cublas_v2.h>
#include "cudaError.h"
#include <errno.h>
#include <unistd.h>

SlcImage::SlcImage() {
    fileid = -1;
    is_mapped = 0;
    is_opened = 0;
    height = 0;
    width = 0;
}
  
 
SlcImage::SlcImage(std::string fn, size_t h, size_t w) {
    filename = fn;
    width = w;
    height = h;
    is_mapped = 0;
    is_opened = 0;
    openFile();
    buffersize = filesize;
    offset = 0l; 
    openFile();
    setupMmap();
}

SlcImage::SlcImage(std::string fn, size_t h, size_t w, size_t bsize) {
    filename = fn;
    width = w;
    height = h;
    is_mapped = 0;
    is_opened = 0;
    buffersize = bsize*(1l<<30); //1G as a unit
    offset = 0l;
    openFile();
    //std::cout << "buffer and file sizes" << buffersize << " " << filesize << std::endl;
    setupMmap();
}

void SlcImage::setBufferSize(size_t sizeInG)
{
    buffersize = sizeInG*(1l<<30);
}

void SlcImage::openFile()
{
    if(!is_opened){
        fileid = open(filename.c_str(), O_RDONLY, 0);
        if(fileid == -1) 
            {
            fprintf(stderr, "Error opening file %s\n", filename.c_str());
            exit(EXIT_FAILURE);
        }
    }
    struct stat st;
    stat(filename.c_str(), &st);
    filesize = st.st_size;
    //lseek(fileid,filesize-1,SEEK_SET);
    is_opened = 1;
}

void SlcImage::closeFile()
{
    if(is_opened)
        {
        close(fileid);
        is_opened = 0;
    }
}
/*
  void SlcImage::setupMmap()
{
    if(!is_mapped) {
        float2 *fmmap = (float2 *)mmap(NULL, filesize, PROT_READ, MAP_SHARED, fileid, 0);
        assert (fmmap != MAP_FAILED);
        mmapPtr =  fmmap;
        is_mapped = 1;
    }
}*/

void SlcImage::setupMmap()
{

    if(is_opened) {
        if(!is_mapped) {
            void * fmmap;
            if((fmmap=mmap((caddr_t)0, buffersize, PROT_READ, MAP_SHARED, fileid, offset)) == MAP_FAILED)
            {
                fprintf(stderr, "mmap error: %d %d\n", fileid, errno);
                exit(1);
            }	       
            mmapPtr = (float2 *)fmmap;
            is_mapped = 1;
        }
    }
    else {
        fprintf(stderr, "error! file is not opened");
        exit(1);}
    //fprintf(stderr, "debug mmap setup %ld, %ld\n", offset, buffersize);
    //fprintf(stderr, "starting mmap pixel %f %f\n", mmapPtr[0].x, mmapPtr[0].y);
}

void SlcImage::mUnMap()
{
    if(is_mapped) {
        if(munmap((void *)mmapPtr, buffersize) == -1)
        {
            fprintf(stderr, "munmap error: %d\n", fileid);
        } 
        is_mapped = 0; 
    }
}


/// load a tile of data h_tile x w_tile from CPU (mmap) to GPU
/// @param dArray pointer for array in device memory
/// @param h_offset Down/Height offset
/// @param w_offset Across/Width offset
/// @param h_tile Down/Height tile size
/// @param w_tile Across/Width tile size
/// @param stream CUDA stream for copying
void SlcImage::loadToDevice(float2 *dArray, size_t h_offset, size_t w_offset, size_t h_tile, size_t w_tile, cudaStream_t stream)
{
    size_t tileStartAddress = (h_offset*width + w_offset)*sizeof(float2); 
    size_t tileLastAddress = tileStartAddress + (h_tile*width + w_tile)*sizeof(float2); 
    size_t pagesize = getpagesize();
     
    if(tileStartAddress  < offset || tileLastAddress > offset + buffersize )
    {
        size_t temp = tileStartAddress/pagesize;
        offset = temp*pagesize;
        mUnMap();
        setupMmap(); 
    }
    
    float2 *startPtr = mmapPtr ;
    startPtr += (tileStartAddress - offset)/sizeof(float2);
    
    // @note 
    // We assume down/across directions as rows/cols. Therefore, SLC mmap and device array are both row major. 
    // cuBlas assumes both source and target arrays are column major. 
    // To use cublasSetMatrix, we need to switch w_tile/h_tile for rows/cols  
    // checkCudaErrors(cublasSetMatrixAsync(w_tile, h_tile, sizeof(float2), startPtr, width, dArray, w_tile, stream)); 
    
    checkCudaErrors(cudaMemcpy2DAsync(dArray, w_tile*sizeof(float2), startPtr, width*sizeof(float2), 
                                      w_tile*sizeof(float2), h_tile, cudaMemcpyHostToDevice,stream)); 
}

SlcImage::~SlcImage()
{
    mUnMap();
    closeFile();
}
  	  

void SlcImage::testData()
{
    float2 *test;
    test =(float2 *)malloc(10*sizeof(float2));
    mempcpy(test, mmapPtr+1000000l, 10*sizeof(float2));
    for(int i=0; i<10; i++)
        std::cout << test[i].x << " " << test[i].y << ",";
    std::cout << std::endl;
}
