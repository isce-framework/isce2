#include "GDALImage.h"
#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <assert.h>
#include <cublas_v2.h>
#include "cudaError.h"
#include <errno.h>
#include <unistd.h>


/**
 * \brief Constructor
 *
 * @param filename a std::string with the raster image file name
 */

GDALImage::GDALImage(std::string filename, int band, int cacheSizeInGB, int useMmap)
   : _useMmap(useMmap)
{
    // open the file as dataset
    _poDataset = (GDALDataset *) GDALOpen(filename.c_str(), GA_ReadOnly );
    // if something is wrong, throw an exception
    // GDAL reports the error message
    if(!_poDataset)
        throw;

    // check the band info
    int count = _poDataset->GetRasterCount();
    if(band > count)
    {
        std::cout << "The desired band " << band << " is greated than " << count << " bands available";
        throw;
    }

    // get the desired band
    _poBand = _poDataset->GetRasterBand(band);
    if(!_poBand)
        throw;

     // get the width(x), and height(y)
    _width = _poBand->GetXSize();
    _height = _poBand->GetYSize();

    _dataType = _poBand->GetRasterDataType();
    // determine the image type
    _isComplex = GDALDataTypeIsComplex(_dataType);
    // determine the pixel size in bytes
    _pixelSize = GDALGetDataTypeSize(_dataType);

    _bufferSize = 1024*1024*cacheSizeInGB;

    // checking whether using memory map
    if(_useMmap) {

       char **papszOptions = NULL;
        // if cacheSizeInGB = 0, use default
        // else set the option
        if(cacheSizeInGB > 0)
            papszOptions = CSLSetNameValue( papszOptions,
                "CACHE_SIZE",
		        std::to_string(_bufferSize).c_str());

        // space between two lines
	    GIntBig pnLineSpace;
        // set up the virtual mem buffer
        _poBandVirtualMem =  GDALGetVirtualMemAuto(
            static_cast<GDALRasterBandH>(_poBand),
		    GF_Read,
		    &_pixelSize,
		    &pnLineSpace,
		    papszOptions);

        // check it
        if(!_poBandVirtualMem)
            throw;

        // get the starting pointer
        _memPtr = CPLVirtualMemGetAddr(_poBandVirtualMem);
    }
    else { // use a buffer
        checkCudaErrors(cudaMallocHost((void **)&_memPtr, _bufferSize));
    }

    // make sure memPtr is not Null
    if (!_memPtr)
        throw;

    // all done
}


/// load a tile of data h_tile x w_tile from CPU (mmap) to GPU
/// @param dArray pointer for array in device memory
/// @param h_offset Down/Height offset
/// @param w_offset Across/Width offset
/// @param h_tile Down/Height tile size
/// @param w_tile Across/Width tile size
/// @param stream CUDA stream for copying
void GDALImage::loadToDevice(void *dArray, size_t h_offset, size_t w_offset, size_t h_tile, size_t w_tile, cudaStream_t stream)
{
    size_t tileStartOffset = (h_offset*_width + w_offset)*_pixelSize;

    char * startPtr = (char *)_memPtr ;
    startPtr += tileStartOffset;

    // @note
    // We assume down/across directions as rows/cols. Therefore, SLC mmap and device array are both row major.
    // cuBlas assumes both source and target arrays are column major.
    // To use cublasSetMatrix, we need to switch w_tile/h_tile for rows/cols
    // checkCudaErrors(cublasSetMatrixAsync(w_tile, h_tile, sizeof(float2), startPtr, width, dArray, w_tile, stream));
    if (_useMmap)
        checkCudaErrors(cudaMemcpy2DAsync(dArray, w_tile*_pixelSize, startPtr, _width*_pixelSize,
                                      w_tile*_pixelSize, h_tile, cudaMemcpyHostToDevice,stream));
    else {
        // get the total tile size in bytes
        size_t tileSize = h_tile*w_tile*_pixelSize;
        // if the size is bigger than existing buffer, reallocate
        if (tileSize > _bufferSize) {
            // maybe we need to make it to fit the pagesize
            _bufferSize = tileSize;
            checkCudaErrors(cudaFree(_memPtr));
            checkCudaErrors(cudaMallocHost((void **)&_memPtr, _bufferSize));
        }
        // copy from file to buffer
        CPLErr err = _poBand->RasterIO(GF_Read, //eRWFlag
            w_offset, h_offset,  //nXOff, nYOff
            w_tile, h_tile,  // nXSize, nYSize
            _memPtr, // pData
            w_tile*h_tile, 1, // nBufXSize, nBufYSize
            _dataType, //eBufType
            0, 0, //nPixelSpace, nLineSpace in pData
            NULL //psExtraArg extra resampling callback
            );

        if(err != CE_None)
            throw;
        // copy from buffer to gpu
        checkCudaErrors(cudaMemcpyAsync(dArray, _memPtr, tileSize, cudaMemcpyHostToDevice, stream));
    }
}

GDALImage::~GDALImage()
{
    // free the virtual memory
    CPLVirtualMemFree(_poBandVirtualMem),
    // free the GDAL Dataset, close the file
    delete _poDataset;
}

// end of file
