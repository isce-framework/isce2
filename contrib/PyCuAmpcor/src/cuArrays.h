/*
 * cuArrays.h
 * Header file for declaring a group of images
 * 
 * Lijun Zhu
 * Seismo Lab, Caltech
 * V1.0 11/29/2016
 */

#ifndef __CUIMAGES_H
#define __CUIMAGES_H

#include <cuda.h>
#include <driver_types.h>

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>



template <typename T>
class cuArrays{
public:
    int height; // x, row, down, length, azimuth, along the track 
    int width;  // y, col, across, range, along the sight 
    int size;   // chip size, heigh*width 
    int countW; // number of images along width direction
    int countH; // number of images along height direction
    int count;  // countW*countH, number of images
    T* devData; // pointer to data in device (gpu) memory 
    T* hostData;
    
    bool is_allocated;
    bool is_allocatedHost;
    
    cuArrays() : width(0), height(0), size(0), countW(0), countH(0), count(0), 
        is_allocated(0), is_allocatedHost(0), 
        devData(0), hostData(0) {}
        
    // single image
    cuArrays(size_t h, size_t w) : width(w), height(h), countH(1), countW(1), count(1),
        is_allocated(0), is_allocatedHost(0), 
        devData(0), hostData(0)
    {
        size = w*h;
    }
		
    // 
    cuArrays(size_t h, size_t w, size_t n) : width(w), height(h), countH(1), countW(n), count(n),
        is_allocated(0), is_allocatedHost(0), 
        devData(0), hostData(0) 
    {
        size = w*h;
    }
		
    cuArrays(size_t h, size_t w, size_t ch, size_t cw) : width(w), height(h), countW(cw), countH(ch),
        is_allocated(0), is_allocatedHost(0), 
        devData(0), hostData(0) 
    {
        size = w*h;
        count = countH*countW;
    }		
    
    void allocate();
    void allocateHost();
    void deallocate();
    void deallocateHost();

    void copyToHost(cudaStream_t stream);
    void copyToDevice(cudaStream_t stream);
	
    size_t getSize()
    {
        return size*count;
    }
    
    long getByteSize()
    {
        return width*height*count*sizeof(T);
    }
		
    ~cuArrays() 
    {
        if(is_allocated)
            deallocate();
        if(is_allocatedHost)
            deallocateHost();
    }
    
    void setZero(cudaStream_t stream);
    void debuginfo(cudaStream_t stream) ;
    void debuginfo(cudaStream_t stream, float factor);
    void outputToFile(std::string fn, cudaStream_t stream);
    void outputHostToFile(std::string fn);

};

#endif //__CUIMAGES_H	
