#include "SlcImage.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <cuComplex.h>
#include <assert.h>
#include <cublas_v2.h>
#include "cudaError.h"
#include <iostream>
#include <stdexcept>

SlcImage::SlcImage(const std::string& filepath, size_t img_height, size_t img_width, size_t pixel_size, size_t buffer_size)
    : width(img_width), height(img_height), pixel_size(pixel_size), fd(-1), mapped_data(nullptr),
      mapped_offset(0), mapped_size(0)
{
    file_size = width * height * pixel_size;
    max_map_size = buffer_size*1024*1024*1024;
    page_size = sysconf(_SC_PAGE_SIZE);  // Get system page size

    // Open the file
    fd = open(filepath.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
}

void SlcImage::remapIfNeeded(size_t required_start, size_t required_end)
{

    if(required_start < mapped_offset || required_end > mapped_offset + mapped_size)
    {
        // out of range, remap
        // unmap first, if neccesary
        if(mapped_data!=nullptr)
            munmap(mapped_data, mapped_size);
        // align new mapping offset
        // round to the page size
        mapped_offset = (required_start/page_size)*page_size;
        // compute the mapped size
        mapped_size = file_size - mapped_offset;
        // not to exceed the buffer size
        if (mapped_size > max_map_size) {
            mapped_size = max_map_size;
        }
        // remap
        mapped_data = mmap(nullptr, mapped_size, PROT_READ, MAP_PRIVATE, fd, mapped_offset);
        if (mapped_data == MAP_FAILED) {
            throw std::runtime_error("Failed to mmap file at offset " + std::to_string(mapped_offset));
        }
    }
    // else - in range, do nothing
}


/// load a tile of data h_tile x w_tile from CPU (mmap) to GPU
/// @param dArray pointer for array in device memory
/// @param h_offset Down/Height offset
/// @param w_offset Across/Width offset
/// @param h_tile Down/Height tile size
/// @param w_tile Across/Width tile size
/// @param stream CUDA stream for copying
void SlcImage::loadToDevice(void *dArray, size_t h_offset, size_t w_offset, size_t h_tile, size_t w_tile, cudaStream_t stream)
{
    size_t tileStartAddress = (h_offset*width + w_offset)*pixel_size;
    size_t tileLastAddress = tileStartAddress + (h_tile*width + w_tile)*pixel_size;
     
    remapIfNeeded(tileStartAddress, tileLastAddress);
    
    char *startPtr = (char *)mapped_data ;
    startPtr += tileStartAddress - mapped_offset;
    
    // @note 
    // We assume down/across directions as rows/cols. Therefore, SLC mmap and device array both use row major.
    // cuBlas assumes both source and target arrays are column major. 
    // To use cublasSetMatrix, we need to switch w_tile/h_tile for rows/cols  
    // checkCudaErrors(cublasSetMatrixAsync(w_tile, h_tile, pixelsize, startPtr, width, dArray, w_tile, stream)); 
    
    checkCudaErrors(cudaMemcpy2DAsync(dArray, w_tile*pixel_size, startPtr, width*pixel_size,
                                      w_tile*pixel_size, h_tile, cudaMemcpyHostToDevice,stream));
}

SlcImage::~SlcImage()
{
    if (mapped_data!=nullptr) {
        munmap(mapped_data, mapped_size);
    }
    if (fd != -1) {
        close(fd);
    }
}
  	  
// end of file
