// -*- c++ -*- 
#ifndef __SLCIMAGE_H
#define __SLCIMAGE_H

#include <cublas_v2.h>
#include <string>

class SlcImage{
private:
    std::string filename;
    int fileid;
    size_t filesize;
    size_t height;
    size_t width;
    
    bool is_mapped;
    bool is_opened;
    float2* mmapPtr;  
    size_t buffersize;
    size_t offset;
    
public:  
    SlcImage();
    
    SlcImage(std::string fn, size_t h, size_t w);
    SlcImage(std::string fn, size_t h, size_t w, size_t bsize);
    void openFile();
    void closeFile();
    void setupMmap();
    void mUnMap();
    void setBufferSize(size_t size);
    
    float2* getmmapPtr()
    {
        return(mmapPtr);
    }
    
    size_t getFileSize()
    {
        return (filesize);
    }
    
    size_t getHeight() {
        return (height);
    }
    
    size_t getWidth()
    {
        return (width);
    }
    
    bool getMmapStatus() 
    {
        return(is_mapped);
    }
    
    //tested
    void loadToDevice(float2 *dArray, size_t h_offset, size_t w_offset, size_t h_tile, size_t w_tile, cudaStream_t stream);
    ~SlcImage();
    void testData();
    
};

#endif //__SLCIMAGE_H
