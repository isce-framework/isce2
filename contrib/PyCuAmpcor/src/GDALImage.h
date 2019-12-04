// -*- c++ -*-
/**
 * \brief Class for an image described GDAL vrt
 *
 * only complex (pixelOffset=8) or real(pixelOffset=4) images are supported, such as SLC and single-precision TIFF
 */

#ifndef __GDALIMAGE_H
#define __GDALIMAGE_H

#include <cublas_v2.h>
#include <string>
#include <gdal/gdal_priv.h>
#include <gdal/cpl_conv.h>

class GDALImage{

public:
    using size_t = std::size_t;

private:
    size_t _fileSize;
    int _height;
    int _width;

    // buffer pointer
    void * _memPtr = NULL;

    int _pixelSize; //in bytes

    int _isComplex;

    size_t _bufferSize;
    int _useMmap;

    GDALDataType _dataType;
    CPLVirtualMem * _poBandVirtualMem = NULL;
    GDALDataset * _poDataset = NULL;
    GDALRasterBand * _poBand = NULL;

public:
    GDALImage() = delete;
    GDALImage(std::string fn, int band=1, int cacheSizeInGB=0, int useMmap=1);

    void * getmemPtr()
    {
        return(_memPtr);
    }

    size_t getFileSize()
    {
        return (_fileSize);
    }

    size_t getHeight() {
        return (_height);
    }

    size_t getWidth()
    {
        return (_width);
    }

    int getPixelSize()
    {
        return _pixelSize;
    }

    bool isComplex()
    {
        return _isComplex;
    }

    void loadToDevice(void *dArray, size_t h_offset, size_t w_offset, size_t h_tile, size_t w_tile, cudaStream_t stream);
    ~GDALImage();

};

#endif //__GDALIMAGE_H
