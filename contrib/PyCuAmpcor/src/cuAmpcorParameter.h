/**
 * cuAmpcorParameter.h
 * Header file for Ampcor Parameter Class
 *
 * Author: Lijun Zhu @ Seismo Lab, Caltech
 * March 2017
 */

#ifndef __CUAMPCORPARAMETER_H
#define __CUAMPCORPARAMETER_H

#include <string>

/// Class container for all parameters
///
/// @note
/// The dimension/direction names used are:
/// The inner-most dimension: x, row, height, down, azimuth, along the track.
/// The outer-most dimension: y, column, width, across, range, along the sight.
/// C/C++/Python use row-major indexing: a[i][j] -> a[i*WIDTH+j]
/// FORTRAN/BLAS/CUBLAS use column-major indexing: a[i][j]->a[i+j*LENGTH]

/// @note
/// Common procedures to use cuAmpcorParameter
/// 1. Create an instance of cuAmpcorParameter: param = new cuAmpcorParameter()
/// 2. Provide/set constant parameters, including numberWindows such as : param->numberWindowDown = 100
/// 3. Call setupParameters() to determine related parameters and allocate starting pixels for each window: param->setupParameters()
/// 4. Provide/set Reference window starting pixel(s), and gross offset(s): param->setStartPixels(referenceStartDown, referenceStartAcross, grossOffsetDown, grossOffsetAcross)
/// 4a. Optionally, check the range of windows is within the SLC image range: param->checkPixelInImageRange()
/// Steps 1, 3, 4 are mandatory. If step 2 is missing, default values will be used


class cuAmpcorParameter{
public:
    int algorithm;      /// Cross-correlation algorithm: 0=freq domain (default) 1=time domain
    int deviceID;       /// Targeted GPU device ID: use -1 to auto select
    int nStreams;       /// Number of streams to asynchonize data transfers and compute kernels
    int derampMethod;   /// Method for deramping 0=None, 1=average, 2=phase gradient

    // chip or window size for raw data
    int windowSizeHeightRaw;        /// Template window height (original size)
    int windowSizeWidthRaw;         /// Template window width (original size)
    int searchWindowSizeHeightRaw;  /// Search window height (original size)
    int searchWindowSizeWidthRaw;   /// Search window width (orignal size)

    int halfSearchRangeDownRaw;   ///(searchWindowSizeHeightRaw-windowSizeHeightRaw)/2
    int halfSearchRangeAcrossRaw;    ///(searchWindowSizeWidthRaw-windowSizeWidthRaw)/2
   	// search range is (-halfSearchRangeRaw, halfSearchRangeRaw)

    int searchWindowSizeHeightRawZoomIn;
    int searchWindowSizeWidthRawZoomIn;

    int corrRawZoomInHeight;  // window to estimate snr
    int corrRawZoomInWidth;

    // chip or window size after oversampling
    int rawDataOversamplingFactor;  /// Raw data overampling factor (from original size to oversampled size)
    int windowSizeHeight;           /// Template window length (oversampled size)
    int windowSizeWidth;            /// Template window width (original size)
    int searchWindowSizeHeight;     /// Search window height (oversampled size)
    int searchWindowSizeWidth;      /// Search window width (oversampled size)

    // strides between chips/windows
    int skipSampleDownRaw;   /// Skip size between neighboring windows in Down direction (original size)
    int skipSampleAcrossRaw; /// Skip size between neighboring windows in across direction (original size)
    //int skipSampleDown;      /// Skip size between neighboring windows in Down direction (oversampled size)
    //int skipSampleAcross;    /// Skip size between neighboring windows in Across direction (oversampled size)

    // Zoom in region near location of max correlation
    int zoomWindowSize;      /// Zoom-in window size in correlation surface (same for down and across directions)
    int halfZoomWindowSizeRaw; /// = half of zoomWindowSize/rawDataOversamplingFactor



    int oversamplingFactor;  /// Oversampling factor for interpolating correlation surface
    int oversamplingMethod;  /// 0 = fft (default)  1 = sinc


    float thresholdSNR;      /// Threshold of Signal noise ratio to remove noisy data

    //reference image
    std::string referenceImageName;    /// reference SLC image name
    int imageDataType1;             /// reference image data type, 2=cfloat=complex=float2 1=float
    int referenceImageHeight;          /// reference image height
    int referenceImageWidth;           /// reference image width

    //secondary image
    std::string secondaryImageName;     /// secondary SLC image name
    int imageDataType2;             /// secondary image data type, 2=cfloat=complex=float2 1=float
    int secondaryImageHeight;           /// secondary image height
    int secondaryImageWidth;            /// secondary image width

    // total number of chips/windows
    int numberWindowDown;           /// number of total windows (down)
    int numberWindowAcross;         /// number of total windows (across)
    int numberWindows; 				/// numberWindowDown*numberWindowAcross

    // number of chips/windows in a batch/chunk
    int numberWindowDownInChunk;    /// number of windows processed in a chunk (down)
    int numberWindowAcrossInChunk;  /// number of windows processed in a chunk (across)
    int numberWindowsInChunk; 		/// numberWindowDownInChunk*numberWindowAcrossInChunk
    int numberChunkDown;            /// number of chunks (down)
    int numberChunkAcross;          /// number of chunks (across)
    int numberChunks;

    int useMmap;                    /// whether to use mmap 0=not 1=yes (default = 0)
    int mmapSizeInGB;               /// size for mmap buffer(useMmap=1) or a cpu memory buffer (useMmap=0)

    int referenceStartPixelDown0;
    int referenceStartPixelAcross0;
    int *referenceStartPixelDown;  /// reference starting pixels for each window (down)
    int *referenceStartPixelAcross;/// reference starting pixels for each window (across)
    int *secondaryStartPixelDown;   /// secondary starting pixels for each window (down)
    int *secondaryStartPixelAcross; /// secondary starting pixels for each window (across)
    int grossOffsetDown0;
    int grossOffsetAcross0;
    int *grossOffsetDown;		/// Gross offsets between reference and secondary windows (down) : secondaryStartPixel - referenceStartPixel
    int *grossOffsetAcross;     /// Gross offsets between reference and secondary windows (across)

    int *referenceChunkStartPixelDown;
    int *referenceChunkStartPixelAcross;
    int *secondaryChunkStartPixelDown;
    int *secondaryChunkStartPixelAcross;
    int *referenceChunkHeight;
    int *referenceChunkWidth;
    int *secondaryChunkHeight;
    int *secondaryChunkWidth;
    int maxReferenceChunkHeight, maxReferenceChunkWidth;
    int maxSecondaryChunkHeight, maxSecondaryChunkWidth;

    std::string grossOffsetImageName;
    std::string offsetImageName;    /// Output Offset fields filename
    std::string snrImageName;       /// Output SNR filename
    std::string covImageName;

    cuAmpcorParameter();  /// Class constructor and default parameters setter
    ~cuAmpcorParameter(); /// Class descontructor

    void allocateArrays();      /// Allocate various arrays after the number of Windows is given
    void deallocateArrays();    /// Deallocate arrays on exit


    /// Three methods to set reference/secondary starting pixels and gross offsets from input reference start pixel(s) and gross offset(s)
	/// 1 (int *, int *, int *, int *): varying reference start pixels and gross offsets
	/// 2 (int, int, int *, int *): fixed reference start pixel (first window) and varying gross offsets
	/// 3 (int, int, int, int): fixed reference start pixel(first window) and fixed gross offsets
    void setStartPixels(int*, int*, int*, int*);
    void setStartPixels(int, int, int*, int*);
    void setStartPixels(int, int, int, int);
    void setChunkStartPixels();
    void checkPixelInImageRange(); /// check whether
    void setupParameters();     /// Process other parameters after Python Input

};

#endif
