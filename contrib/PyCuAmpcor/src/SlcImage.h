// -*- c++ -*-
// file slcimage.h
// a mmap loader for slc images

#ifndef __SLCIMAGE_H
#define __SLCIMAGE_H

#include <string>
#include <cuda_runtime.h>

class SlcImage{
public:
    // disable default constructor
    SlcImage()=delete;
    // constructor
    SlcImage(const std::string& fn, size_t image_height, size_t image_width, size_t pixel_size, size_t buffersize);
    // interface
    void loadToDevice(void* dArray, size_t h_offset, size_t w_offset, size_t h_tile, size_t w_tile, cudaStream_t stream);
    // destructor
    ~SlcImage();

private:
    int fd;
    size_t file_size;
    size_t pixel_size;
    size_t height;
    size_t width;

    void* mapped_data;
    size_t page_size;
    size_t mapped_offset;
    size_t mapped_size;
    size_t max_map_size;

    void remapIfNeeded(size_t required_start, size_t required_end);
};

#endif //__SLCIMAGE_H
