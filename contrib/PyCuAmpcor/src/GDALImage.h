/**
 * @file GDALImage.h
 * @brief Interface with GDAL vrt driver
 *
 * To read image file with the GDAL vrt driver, including SLC, GeoTIFF images
 * @warning Only single precision images are supported: complex(pixelOffset=8) or real(pixelOffset=4).
 * @warning Only single band file is currently supported.
 */

// code guard
#ifndef __GDALIMAGE_H
#define __GDALIMAGE_H

// dependencies
#include <string>
#include <gdal_priv.h>
#include <cpl_conv.h>


class GDALImage{
public:
    // specify the types
    using size_t = std::size_t;

private:
    int _height;      ///< image height
    int _width;       ///< image width

    void * _memPtr = NULL; ///< pointer to buffer

    int _pixelSize; ///< pixel size in bytes

    int _isComplex; ///< whether the image is complex

    size_t _bufferSize; ///< buffer size
    int _useMmap;   ///< whether to use memory map

    // GDAL temporary objects
    GDALDataType _dataType;
    CPLVirtualMem * _poBandVirtualMem = NULL;
    GDALDataset * _poDataset = NULL;
    GDALRasterBand * _poBand = NULL;

public:
    //disable default constructor
    GDALImage() = delete;
    // constructor
    GDALImage(std::string fn, int band=1, int cacheSizeInGB=0, int useMmap=1);
    // destructor
    ~GDALImage();

    // get class properties
    void * getmemPtr()
    {
        return(_memPtr);
    }

    int getHeight() {
        return (_height);
    }

    int getWidth()
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

    // load data from cpu buffer to gpu
    void loadToDevice(void *dArray, size_t h_offset, size_t w_offset, size_t h_tile, size_t w_tile, cudaStream_t stream);

};

#endif //__GDALIMAGE_H
// end of file
