//
// Author: Joshua Cohen
// Copyright 2016
//

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <cufftXt.h>
#include <math.h>
#include <complex.h>
#include <stdio.h>
#include <sys/time.h>

// Safe ternary operator implementation of min()
#define min(a,b) \
            ({ __typeof__ (a) _a = (a); \
               __typeof__ (b) _b = (b); \
               _a < _b ? _a : _b;})

// Safe ternary operator implementation of max()
#define max(a,b) \
            ({ __typeof__ (a) _a = (a); \
               __typeof__ (b) _b = (b); \
               _a > _b ? _a : _b;})

// Safe ternary operator implementation of abs()
#define abs(a) \
            ({ __typeof__ (a) _a = (a); \
               _a >= 0 ? _a : -1*_a;})

#define pow2(a) powf((a), 2.)

// Row-major inline conversion from 2D index to 1D index (takes i,j and line-width w)
#define IDX1D(i,j,w) (((i)*(w))+(j))

#define THREAD_PER_IMG_BLOCK 64
#define THREAD_PER_PIX_BLOCK 64

// ---------------- STRUCTS -------------------

// Data for first kernel
struct StepZeroData {
    cuFloatComplex *refBlocks;      // Input from Ampcor.cpp
    cuFloatComplex *schBlocks;      // Input from Ampcor.cpp
    cuFloatComplex *padRefChips;    // Block array 0
    cuFloatComplex *padSchWins;     // Block array 1
    int *locationAcrossArr;         // Output point array
    int *locationDownArr;           // Output point array
    int *globalX;                   // Input from Ampcor.cpp
    int *globalY;                   // Input from Ampcor.cpp
};

// Data for second kernel
struct StepOneData {
    cuFloatComplex *padRefChips;    // Block array 0
    cuFloatComplex *padSchWins;     // Block array 1
    float *schSums;                 // Block array 2
    float *schSumSqs;               // Block array 3
    float *refNorms;                // Point array 3
};

// Data for third kernel
struct StepTwoData {
    cuFloatComplex *padRefChips;    // Block array 0
    cuFloatComplex *corrWins;       // Block array 1 (renamed)
    cuFloatComplex *zoomWins;       // Block array 4
    float *schSums;                 // Block array 2
    float *schSumSqs;               // Block array 3
    float *cov1Arr;                 // Output point array
    float *cov2Arr;                 // Output point array
    float *cov3Arr;                 // Output point array
    float *snrArr;                  // Output point array
    float *refNorms;                // Point array 3
    int *roughPeakRowArr;           // Point array 0
    int *roughPeakColArr;           // Point array 1
    bool *flagArr;                  // Point array 2
};

// Data for fourth kernel
struct StepThreeData {
    cuFloatComplex *zoomWins;       // Block array 4
    float *locationAcrossOffsetArr; // Output point array
    float *locationDownOffsetArr;   // Output point array
    float *cov1Arr;                 // Output point array
    float *cov2Arr;                 // Output point array
    float *cov3Arr;                 // Output point array
    float *snrArr;                  // Output point array
    int *locationAcrossArr;         // Output point array
    int *locationDownArr;           // Output point array
    int *roughPeakRowArr;           // Point array 0
    int *roughPeakColArr;           // Point array 1
    bool *flagArr;                  // Point array 2
};

// Constant memory for the device (store precalculated constants)
__constant__ float inf[3];
__constant__ int ini[21];


__device__ inline void deramp(cuFloatComplex *img, int length, int width, cuFloatComplex *padImg) {
    /*
    *   Deramp an image block and copy to the padded window. Used before first FFT-spread operation
    *   Data usage: 56 bytes (4 complex/pointer, 6 int/float)
    */

    cuFloatComplex cx_phaseDown, cx_phaseAcross, temp;
    float rl_phaseDown, rl_phaseAcross;
    int i,j;

    // Init to 0.
    cx_phaseDown = make_cuFloatComplex(0.,0.);
    cx_phaseAcross = make_cuFloatComplex(0.,0.);
    rl_phaseDown = 0.;
    rl_phaseAcross = 0.;

    // Accumulate phase across and phase down. For phase across, sum adds the value of the pixel multiplied by the complex
    // conjugate of the pixel in the next row (same column). For phase down, sum adds the value of the pixel multiplied by
    // complex conjugate of the pixel in the next column (same row). Note that across/down refer to original image, and
    // since blocks are transposed, these "directions" switch.
    for (i=0; i<(length-1); i++) {
        for (j=0; j<width; j++) {
            cx_phaseAcross = cuCaddf(cx_phaseAcross, cuCmulf(img[IDX1D(i,j,width)], cuConjf(img[IDX1D(i+1,j,width)])));
        }
    }
    for (i=0; i<length; i++) {
        for (j=0; j<(width-1); j++) {
            cx_phaseDown = cuCaddf(cx_phaseDown, cuCmulf(img[IDX1D(i,j,width)], cuConjf(img[IDX1D(i,j+1,width)])));
        }
    }

    // Transform complex phases to the real domain (note for cuFloatComplex, cx.x == cx.real and cx.y == cx.imag)
    if (cuCabsf(cx_phaseDown) != 0.) rl_phaseDown = atan2(cx_phaseDown.y, cx_phaseDown.x);
    if (cuCabsf(cx_phaseAcross) != 0.) rl_phaseAcross = atan2(cx_phaseAcross.y, cx_phaseAcross.x);

    // For each pixel in the image block...
    for (i=0; i<length; i++) {
        for (j=0; j<width; j++) {
            // Calculate scaling factor for a given pixel location
            temp = make_cuFloatComplex(cos((rl_phaseAcross*(i+1))+(rl_phaseDown*(j+1))), sin((rl_phaseAcross*(i+1))+(rl_phaseDown*(j+1))));
            // Apply scaling factor to the pixel to deramp it and copies the result to the corresponding location within the padded window
            padImg[IDX1D(i,j,ini[7])] = cuCmulf(img[IDX1D(i,j,width)], temp);
        }
    }
}

// First kernel
__global__ void prepBlocks(struct StepZeroData s0Data) {
    /*
    *   Get offset locations, deramp the ref/sch blocks, and copy them into the larger padded blocks
    *   Data usage: 164 bytes (12 pointers, 3 ints, 1 deramp() call) - only 1 deramp() call factored
    *               since they're consecutive
    */

    // Calculate "reference" thread number (analogous in this usage to the overall image block number)
    int imgBlock = (blockDim.x * blockIdx.x) + threadIdx.x;
    
    // Make sure the image block exists (since we have threads that may be empty due to trying to keep warp-size % THREAD_PER_BLOCK == 0)
    if (imgBlock < ini[9]) {

        // Maintain local copy of pointer to particular block since the reference array is linearly-contiguous
        cuFloatComplex *refChip = &(s0Data.refBlocks[IDX1D(imgBlock,0,ini[0]*ini[1])]);        // Non-zero data in ini[0] x ini[1]
        cuFloatComplex *schWin = &(s0Data.schBlocks[IDX1D(imgBlock,0,ini[2]*ini[3])]);         // Non-zero data in ini[2] x ini[3]
        cuFloatComplex *padRefChip = &(s0Data.padRefChips[IDX1D(imgBlock,0,ini[6]*ini[7])]);   // Non-zero data in ini[0] x ini[1]
        cuFloatComplex *padSchWin = &(s0Data.padSchWins[IDX1D(imgBlock,0,ini[6]*ini[7])]);     // Non-zero data in ini[2] x ini[3]

        // Calculate offset locations. Note that across/down refer to the original image, which is transposed diagonally (i.e.
        // across is relative to this block's length, and down is relative to its width).
        s0Data.locationAcrossArr[imgBlock] = s0Data.globalX[imgBlock] + ((ini[0] - 1) / 2);
        s0Data.locationDownArr[imgBlock] = s0Data.globalY[imgBlock] + ((ini[1] - 1) / 2);

        // Deramp the reference/search image blocks. Also copies results into padded windows (since the originals don't need to be altered)
        deramp(refChip, ini[0], ini[1], padRefChip);
        deramp(schWin, ini[2], ini[3], padSchWin);
    }
}

__global__ void spreadPaddedBlock(cuFloatComplex *arr) {
    // Blocks are ini[6]*ini[7] each (ini[9] blocks total)

    int pix = (blockDim.x * blockIdx.x) + threadIdx.x;

    // Make sure this is a real pixel (end of last block will possibly have empty threads)
    if (pix < (ini[9]*ini[6]*ini[7])) {
        
        int imgBlock, row, col, newOffset;

        imgBlock = pix / (ini[6] * ini[7]);
        row = (pix / ini[7]) - (imgBlock * ini[6]);
        col = (pix % ini[7]);

        if ((row < (ini[6]/2)) && (col < (ini[7]/2))) {
            if (row >= (ini[6]/4)) row = row + (ini[6]/2);
            if (col >= (ini[7]/4)) col = col + (ini[7]/2);
            newOffset = (imgBlock * ini[6] * ini[7]) + IDX1D(row,col,ini[7]);

            // Set element in spread location to element at [pix]
            arr[newOffset].x = arr[pix].x;
            arr[newOffset].y = arr[pix].y;

            // If the element was spread, set the element at [pix] to 0
            if (pix != newOffset) {
                arr[pix].x = 0.;
                arr[pix].y = 0.;
            }
        }
    }
}

__device__ void refNormMagCB(void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPointer) {

    int block, row, col;

    block = offset / (ini[6] * ini[7]);
    row = (offset / ini[7]) - (block * ini[6]);
    col = offset % ini[7];

    ((cuFloatComplex*)dataOut)[offset].x = cuCabsf(element) / ((ini[6] * ini[7]) / 4.);
    if ((row >= (2*ini[0])) || (col >= (2*ini[1]))) ((cuFloatComplex*)dataOut)[offset].x = 0.;
    ((cuFloatComplex*)dataOut)[offset].y = 0.;
}

__device__ void schNormMagCB(void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPointer) {

    int block, row, col;

    block = offset / (ini[6] * ini[7]);
    row = (offset / ini[7]) - (block * ini[6]);
    col = offset % ini[7];

    ((cuFloatComplex*)dataOut)[offset].x = cuCabsf(element) / ((ini[6] * ini[7]) / 4.);
    if ((row >= (2*ini[2])) || (col >= (2*ini[3]))) ((cuFloatComplex*)dataOut)[offset].x = 0.;
    ((cuFloatComplex*)dataOut)[offset].y = 0.;
}

__device__ cufftCallbackStoreC d_refNormMagPtr = refNormMagCB;
__device__ cufftCallbackStoreC d_schNormMagPtr = schNormMagCB;

// Second kernel
__global__ void accumulate(struct StepOneData s1Data) {
    /*
    *   Calculate and remove the mean values from the ref/sch image blocks, accumulate the sum and sum-squared arrays
    *   for the sch image block.
    *   Data usage: 100 bytes (9 pointers, 7 floats/ints) - does not factor cuBLAS calls
    */

    // Reference thread number (also the reference image block number)
    int block = (blockDim.x * blockIdx.x) + threadIdx.x;
    
    // Make sure we're operating on an existing block of data (and not in an empty thread)
    if (block < ini[9]) {

        // Again, maintain local pointer to image-block-specific set of data
        cuFloatComplex *padRefChip = &(s1Data.padRefChips[IDX1D(block,0,ini[6]*ini[7])]);  // Non-zero data in (2*ini[0]) x (2*ini[1]) (mostly, depends on spread interpolation)
        cuFloatComplex *padSchWin = &(s1Data.padSchWins[IDX1D(block,0,ini[6]*ini[7])]);    // Non-zero data in (2*ini[2]) x (2*ini[3])
        //float *schSum = &(s1Data.schSums[IDX1D(block,0,(2*ini[2]+1)*(2*ini[3]+1))]);                 // Non-zero data in (2*ini[2]) x (2*ini[3])
        //float *schSumSq = &(s1Data.schSumSqs[IDX1D(block,0,(2*ini[2]+1)*(2*ini[3]+1))]);             // Non-zero data in (2*ini[2]) x (2*ini[3])
        float refMean, schMean, refChipSum, schWinSum;
        int i, j;
        cublasHandle_t handle;  // Pointer to cuBLAS library context
        
        // Bind cuBLAS library context pointer to working environment
        cublasCreate(&handle);
        // Use cuBLAS to calculate the sum of the complex array (where every element is the magnitude after the callbacks)
        cublasScasum(handle, ini[6]*ini[7], padRefChip, 1, &refChipSum);
        // Divide sum by number of real elements, not by the size of the matrices (since they're 0-padded), to get mean values
        refMean = refChipSum / (4*ini[0]*ini[1]);
      
        // Subtract the mean from its respective image block (ignore imag() value since they're zeroed out in callbacks)
        refChipSum = 0.;
        for (i=0; i<(2*ini[0]); i++) {
            for (j=0; j<(2*ini[1]); j++) {
                padRefChip[IDX1D(i,j,ini[7])].x = padRefChip[IDX1D(i,j,ini[7])].x - refMean;
                refChipSum = refChipSum + pow2(padRefChip[IDX1D(i,j,ini[7])].x); // Need this for later
            }
        }
        // Save the norm for the next kernel
        s1Data.refNorms[block] = sqrtf(refChipSum);
       
        // Get the sum of the other array
        cublasScasum(handle, ini[6]*ini[7], padSchWin, 1, &schWinSum);
        // Matching call to unbind the cuBLAS library context
        cublasDestroy(handle);
        // Get the mean
        schMean = schWinSum / (4*ini[2]*ini[3]);
      
        // Subtract the mean from the image block
        for (i=0; i<(2*ini[2]); i++) {
            for (j=0; j<(2*ini[3]); j++) {
                padSchWin[IDX1D(i,j,ini[7])].x = padSchWin[IDX1D(i,j,ini[7])].x - schMean;
            }
        }
        /*
        // Fill in sum window
        for (i=0; i<(2*ini[2]); i++) {
            for (j=0; j<(2*ini[3]); j++) {
                schSum[IDX1D(i+1,j+1,2*ini[3]+1)] = padSchWin[IDX1D(i,j,ini[7])].x + schSum[IDX1D(i,j+1,2*ini[3]+1)] + 
                                                    schSum[IDX1D(i+1,j,2*ini[3]+1)] - schSum[IDX1D(i,j,2*ini[3]+1)];
            }
        }
        
        // Fill in sum-squared window
        for (i=0; i<(2*ini[2]); i++) {
            for (j=0; j<(2*ini[3]); j++) {
                schSumSq[IDX1D(i+1,j+1,2*ini[3]+1)] = pow2(padSchWin[IDX1D(i,j,ini[7])].x) + schSumSq[IDX1D(i,j+1,2*ini[3]+1)] +
                                                    schSumSq[IDX1D(i+1,j,2*ini[3]+1)] - schSumSq[IDX1D(i,j,2*ini[3]+1)];
            }
        }
        */
    }
}

// ******** DEBUG *********
__global__ void accumulateSum(struct StepOneData s1Data) {

    int block = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (block < ini[9]) {
        cuFloatComplex *padSchWin = &(s1Data.padSchWins[IDX1D(block,0,ini[6]*ini[7])]);
        float *schSum = &(s1Data.schSums[IDX1D(block,0,(2*ini[2]+1)*(2*ini[3]+1))]);
        int i,j;

        for (i=0; i<(2*ini[2]); i++) {
            for (j=0; j<(2*ini[3]); j++) {
                schSum[IDX1D(i+1,j+1,2*ini[3]+1)] = padSchWin[IDX1D(i,j,ini[7])].x + schSum[IDX1D(i,j+1,2*ini[3]+1)] +
                                                    schSum[IDX1D(i+1,j,2*ini[3]+1)] - schSum[IDX1D(i,j,2*ini[3]+1)];
            }
        }
    }
}

__global__ void accumulateSumSq(struct StepOneData s1Data) {
    
    int block = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (block < ini[9]) {
        cuFloatComplex *padSchWin = &(s1Data.padSchWins[IDX1D(block,0,ini[6]*ini[7])]);
        float *schSumSq = &(s1Data.schSumSqs[IDX1D(block,0,(2*ini[2]+1)*(2*ini[3]+1))]);
        int i,j;

        for (i=0; i<(2*ini[2]); i++) {
            for (j=0; j<(2*ini[3]); j++) {
                schSumSq[IDX1D(i+1,j+1,2*ini[3]+1)] = pow2(padSchWin[IDX1D(i,j,ini[7])].x) + schSumSq[IDX1D(i,j+1,2*ini[3]+1)] +
                                                        schSumSq[IDX1D(i+1,j,2*ini[3]+1)] - schSumSq[IDX1D(i,j,2*ini[3]+1)];
            }
        }
    }
}
// *******************

// Callback to call the element-by-element multiplication of refBlock and schBlock' (i.e. complex-conjugate of schBlock)
__device__ void conjMultCB(void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPointer) {
    /*
    *   Multiply the complex conjugate of the return element from the forward FFT of the sch image block by the corresponding
    *   element from the forward FFT of the ref image block.
    *   callerInfo  -   Pointer to a user-defined input passed in when setting up the callback (in this case points to the
    *                   padded ref image block)
    */

    ((cuFloatComplex*)dataOut)[offset] = cuCmulf(element, cuConjf(((cuFloatComplex*)callerInfo)[offset]));
}

// Create a device-side pointer to the above callback
__device__ cufftCallbackStoreC d_conjMultPtr = conjMultCB;

__global__ void fftShiftCorr(cuFloatComplex *arr) {
    
    int pix = (blockDim.x * blockIdx.x) + threadIdx.x;

    // Make sure this is a real pixel (end of last block will possibly have empty threads)
    if (pix < (ini[9]*ini[6]*ini[7])) {

        int row, col, cullWinWidth, cullWinHeight; // imgBlock, newRow, newCol, newOffset

        //imgBlock = pix / (ini[6] * ini[7]);         // Array index
        row = int(pix / ini[7]) % ini[6]; // Row relative to image block
        col = (pix % ini[7]);                       // Col relative to image block
        cullWinWidth = 2 * (ini[2] - ini[0]);
        cullWinHeight = 2 * (ini[3] - ini[1]);

        if ((row < cullWinWidth) && (col < cullWinHeight)) {
            arr[pix].x = cuCabsf(arr[pix]) / float(ini[6] * ini[7]);
            arr[pix].y = 0.;
        } else {
            arr[pix].x = 0.;
            arr[pix].y = 0.;
        }
    }
}

// Third kernel
__global__ void calcRough(struct StepTwoData s2Data) {
    /*
    *   Normalize the correlation surface window using the sum and sum-squared accumulators, calculate the indices of the rough peak
    *   of the correlation surface, calculate the covariances and SNR around the peak, and copy the area-of-interest around the peak
    *   into the zoomed-in surface.
    *   Data usage: 212 bytes (17 pointers, 19 ints) - does not factor cuBLAS calls
    */

    // Reference thread/image block index
    int block = (blockDim.x * blockIdx.x) + threadIdx.x;

    // Make sure we're operating on an existing block
    if (block < ini[9]) {

        // Maintain a local pointer to the particular image block being handled by the thread
        // This is actually the same pointer as padSchWins just renamed for clarity
        cuFloatComplex *corrWin = &(s2Data.corrWins[IDX1D(block,0,ini[6]*ini[7])]);            // Non-zero data in ini[6] x ini[7]
        cuFloatComplex *zoomWin = &(s2Data.zoomWins[IDX1D(block,0,4*ini[8]*ini[8])]);          // Non-zero data in ini[8] x ini[8]
        float *schSum = &(s2Data.schSums[IDX1D(block,0,(2*ini[2]+1)*(2*ini[3]+1))]);                     // Non-zero data in (2*ini[2]) x (2*ini[3])
        float *schSumSq = &(s2Data.schSumSqs[IDX1D(block,0,(2*ini[2]+1)*(2*ini[3]+1))]);                 // Non-zero data in (2*ini[2]) x (2*ini[3])
        float eSum, e2Sum, vertVar, horzVar, diagVar, noiseSq, noiseFr, u, snrNorm, snr;
        int i, j, idx, peakRow, peakCol, count, widthMargin, heightMargin; // sumRow, sumCol
        cublasHandle_t handle;  // Pointer to cuBLAS library context

        // Normalize the correlation surface using the sum and sum-squared accumulators (energies).The margins here are 2 times the original search
        // search margins (since we've upsampled by a factor of 2), however if the peak row/col are found to be within half the zoom window size of
        // the edges of the normalized window, flag it as a bad point since the zoom window would need at least one point outside of the 
        // normalizable surface.
        widthMargin = ini[2] - ini[0];
        heightMargin = ini[3] - ini[1];

        // We only want to look near the correlation peak within the search margins, the rest of the surface is zeroed out
        for (i=0; i<(2*widthMargin); i++) {
            for (j=0; j<(2*heightMargin); j++) {
                eSum = schSum[IDX1D(i+(2*ini[0]),j+(2*ini[1]),2*ini[3]+1)] - schSum[IDX1D(i,j+(2*ini[1]),2*ini[3]+1)] - 
                        schSum[IDX1D(i+(2*ini[0]),j,2*ini[3]+1)] + schSum[IDX1D(i,j,2*ini[3]+1)];
                e2Sum = schSumSq[IDX1D(i+(2*ini[0]),j+(2*ini[1]),2*ini[3]+1)] - schSumSq[IDX1D(i,j+(2*ini[1]),2*ini[3]+1)] -
                            schSumSq[IDX1D(i+(2*ini[0]),j,2*ini[3]+1)] + schSumSq[IDX1D(i,j,2*ini[3]+1)];
                // Normalize
                corrWin[IDX1D(i,j,ini[7])].x = corrWin[IDX1D(i,j,ini[7])].x / (sqrt(e2Sum - (pow2(abs(eSum)) / (4.*ini[0]*ini[1]))) / s2Data.refNorms[block]);
            }
        }
        
        // Bind cuBLAS library context pointer to working environment
        cublasCreate(&handle);
        // Find row/col of max value in the window (rough offsets)
        int chk = cublasIcamax(handle, ini[6]*ini[7], corrWin, 1, &idx);
        // Note that cuBLAS is 1-based indexing, so subtract that off in result
        peakRow = ((idx-1) / ini[7]);
        peakCol = ((idx-1) % ini[7]);
        // Matching call to unbind the handle from the library context
        cublasDestroy(handle);
        
        // cuBLAS seems to fail for currently-unknown reasons on certain configurations (BAD_ALLOC errors that don't make sense), so in case that happens
        // switch to a linear index search. There's minimal performance impact for using this hybrid style of max-element searching
        if (chk != 0) {
            idx = 0;
            for (i=0; i<(ini[6]*ini[7]); i++) {
                idx = ((corrWin[i].x > corrWin[idx].x) ? i : idx);
            }
            peakRow = idx / ini[7];
            peakCol = idx % ini[7];
        }

        // Remove centering factor (half of the size of the correlation surface) and remove margin offset (2 * original search margin since
        // we upsampled the data by a factor of 2)
        s2Data.roughPeakRowArr[block] = peakRow - widthMargin;
        s2Data.roughPeakColArr[block] = peakCol - heightMargin;
        
        // Calculate covariances (incompatible with multi-looked data at the moment) and SNR
        // Initialize to "BAD_VALUE" equivalents for covariances/SNR, and 0. for the rest
        s2Data.cov1Arr[block] = 99.;
        s2Data.cov2Arr[block] = 99.;
        s2Data.cov3Arr[block] = 0.;
        s2Data.snrArr[block] = 9999.99999;
        s2Data.flagArr[block] = true;
        vertVar = 0.;
        horzVar = 0.;
        diagVar = 0.;
        noiseSq = 0.;
        noiseFr = 0.;
        u = 0.;
        snrNorm = 0.;
        snr = 0.;
        count = 0.;
      
        // Covariances are only valid if the ref image block is not located on the edge of the sch win block
        // NOTE: Should we modify the boundaries of this? Theoretically I'd imagine there's a point at which
        //       the peak is outside a reasonable search area...
        if ((peakRow >= (ini[8]/2)) && (peakRow < ((2*widthMargin)-(ini[8]/2))) && (peakCol >= (ini[8]/2)) && (peakCol < ((2*heightMargin)-(ini[8]/2)))) {
            
            // Calculate the horizontal, vertical, and diagonal base variance components
            vertVar = (2 * corrWin[IDX1D(peakRow,peakCol,ini[7])].x) - corrWin[IDX1D(peakRow-1,peakCol,ini[7])].x - corrWin[IDX1D(peakRow+1,peakCol,ini[7])].x;
            horzVar = (2 * corrWin[IDX1D(peakRow,peakCol,ini[7])].x) - corrWin[IDX1D(peakRow,peakCol-1,ini[7])].x - corrWin[IDX1D(peakRow,peakCol+1,ini[7])].x;
            diagVar = ((corrWin[IDX1D(peakRow+1,peakCol+1,ini[7])].x + corrWin[IDX1D(peakRow-1,peakCol-1,ini[7])].x) - 
                        (corrWin[IDX1D(peakRow+1,peakCol-1,ini[7])].x + corrWin[IDX1D(peakRow-1,peakCol+1,ini[7])].x)) / 4.;
            
            // Adjust variances to scale by number of valid data points (in the original ref image block)
            vertVar = vertVar * ini[0] * ini[1];
            horzVar = horzVar * ini[0] * ini[1];
            diagVar = diagVar * ini[0] * ini[1];

            // Calculate noise factors
            noiseSq = 2. * max(1.-corrWin[IDX1D(peakRow,peakCol,ini[7])].x, 0.);
            noiseFr = .5 * ini[0] * ini[1] * pow2(noiseSq / 2.);
            
            // Calculate base covariance parameter
            u = pow2(diagVar) - (vertVar * horzVar);
            
            // u == 0. implies that the correlation surface is too smooth to get accurate covariance values
            if (u != 0.) {
                
                // Use base variance factors and the covariance parameter to get final covariance values
                s2Data.cov1Arr[block] = ((noiseFr * (pow2(horzVar) + pow2(diagVar))) - (noiseSq * u * horzVar)) / pow2(u);
                s2Data.cov2Arr[block] = ((noiseFr * (pow2(vertVar) + pow2(diagVar))) - (noiseSq * u * vertVar)) / pow2(u);
                s2Data.cov3Arr[block] = ((noiseSq * u * diagVar) - (noiseFr * diagVar * (vertVar + horzVar))) / pow2(u);
            }

            // Accumulate a window of (max) 18 x 18 values around the rough peak
            for (i=max(peakRow-9,0); i<min(peakRow+9,2*widthMargin); i++) {
                for (j=max(peakCol-9,0); j<min(peakCol+9,2*heightMargin); j++) {
                    count = count + 1;
                    snrNorm = snrNorm + pow2(corrWin[IDX1D(i,j,ini[7])].x);
                }
            }
            // Average the accumulated values less the peak value itself to get the approximate normalization magnitude
            snrNorm = (snrNorm - pow2(corrWin[IDX1D(peakRow,peakCol,ini[7])].x)) / (count - 1);
            // Find the SNR as a measure of the magnitude of the peak value relative to the window of the surface around it
            snr = pow2(corrWin[IDX1D(peakRow,peakCol,ini[7])].x) / max(snrNorm, float(1.e-10));
            // Cap the SNR by a max value
            s2Data.snrArr[block] = min(snr, float(9999.99999));

            // Only flag the data as good if not an edge case, SNR is over min threshold, and cov[0] & cov[1] are under max threshold
            if ((s2Data.snrArr[block] > inf[0]) && (s2Data.cov1Arr[block] < inf[1]) && (s2Data.cov2Arr[block] < inf[1])) s2Data.flagArr[block] = false;

            // Copy area of interest around the peak to zoom window
            for (i=0; i<ini[8]; i++) {
                for (j=0; j<ini[8]; j++) {
                    zoomWin[IDX1D(i,j,2*ini[8])] = corrWin[IDX1D(peakRow-(ini[8]/2)+i, peakCol-(ini[8]/2)+j, ini[7])];
                }
            }
        }
    }
}

__global__ void spreadZoomBlock(cuFloatComplex *arr) {
    // Blocks are 4*ini[8]*ini[8] each (ini[9] blocks total)

    int pix = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (pix < (ini[9]*4*ini[8]*ini[8])) {

        int imgBlock, row, col, newOffset;

        imgBlock = pix / (4 * ini[8] * ini[8]);
        row = (pix / (2*ini[8])) - (imgBlock * 2*ini[8]);
        col = (pix % (2*ini[8]));

        if ((row < ini[8]) && (col < ini[8])) {
            if (row >= (ini[8]/2)) row = row + ini[8];
            if (col >= (ini[8]/2)) col = col + ini[8];
            newOffset = (imgBlock * 4 * ini[8] * ini[8]) + IDX1D(row,col,2*ini[8]);

            arr[newOffset].x = arr[pix].x;
            arr[newOffset].y = arr[pix].y;

            if (pix != newOffset) {
                arr[pix].x = 0.;
                arr[pix].y = 0.;
            }
        }
    }
}

__device__ void zoomNormMagCB(void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPointer) {

    ((cuFloatComplex*)dataOut)[offset].x = cuCabsf(element) / (ini[8] * ini[8]);
    ((cuFloatComplex*)dataOut)[offset].y = 0.;
}

__device__ cufftCallbackStoreC d_zoomNormMagPtr = zoomNormMagCB;

// Fourth kernel
__global__ void calcFine(struct StepThreeData s3Data) {
    /*
    *   Find the fine approximation of the correlation surface peak location using the indices of the peak value of the
    *   FFT-spread correlation surface around the area-of-interest found in the third kernel.
    *   Data usage: 72 bytes (7 pointers, 4 ints) - does not factor cuBLAS calls
    */

    // Reference thread/image block index
    int block = (blockDim.x * blockIdx.x) + threadIdx.x;

    // Make sure we're operating on an existing image block
    if (block < ini[9]) {

        // Maintain a local pointer to the image block
        cuFloatComplex *zoomWin = &(s3Data.zoomWins[IDX1D(block,0,4*ini[8]*ini[8])]);  // Non-zero data in (2*ini[8]) x (2*ini[8])
        float mx;
        int idx, finePeakRow, finePeakCol;

        mx = 0.;
        for (idx=0; idx<(4*ini[8]*ini[8]); idx++) {
            if (zoomWin[idx].x > mx) {
                mx = zoomWin[idx].x;
                finePeakRow = idx / (2 * ini[8]);
                finePeakCol = idx % (2 * ini[8]);
            }
        }

        // Remove centering factor from the row/col
        finePeakRow = finePeakRow - ini[8];
        finePeakCol = finePeakCol - ini[8];

        // Estimate full offsets using rough and fine offsets calculated in the third and fourth kernels
        s3Data.locationAcrossOffsetArr[block] = ini[4] + (((2. * s3Data.roughPeakRowArr[block]) + finePeakRow) / 4.);
        s3Data.locationDownOffsetArr[block] = ini[5] + (((2. * s3Data.roughPeakColArr[block]) + finePeakCol) / 4.);
        
        // Wipe results if block is flagged at the end of the third kernel
        if (s3Data.flagArr[block]) {
            s3Data.locationAcrossOffsetArr[block] = 0.;
            s3Data.locationDownOffsetArr[block] = 0.;
            s3Data.locationAcrossArr[block] = 0;
            s3Data.locationDownArr[block] = 0;
            s3Data.cov1Arr[block] = 99.;
            s3Data.cov2Arr[block] = 99.;
            s3Data.cov3Arr[block] = 0.;
            s3Data.snrArr[block] = 9999.99999;
        }
    }
}

// --------------- CPU HELPER FUNCTIONS -----------------

double cpuSecond() {
    /*
    *   Timing function for kernels/subprocesses. Returns time value in seconds.
    */
    
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return (double(tp.tv_sec) + double(tp.tv_usec)*1.e-6);
}

int nBlocksPossible(int *inps) {
    /*
    * Calculates, given the input constants, the maximum reasonable number of block pairs to run (i.e. ini[9]). 
    * Assumes worst-case usage of local variables, which would be the third kernel (212 bytes + malloc).
    */
    
    // Assume that we can reasonably work with a certain # of bytes (1e10 == 10 GB for K40, 3.3e9 == 3.3 GB for K520)
    size_t NB_MAX = 1e10;

    // Calculate the amount of memory that needs to be available to malloc on the device. For Imcor, the worst-case memory
    // usage is during the third kernel.
    size_t nb_malloc = (2*inps[6]*inps[7]*sizeof(cuFloatComplex)) + (8*inps[2]*inps[3]*sizeof(float)) +                 // 2 x padWin + 2 x sum/sumSq
                        (4*inps[8]*inps[8]*sizeof(cuFloatComplex)) + (4*sizeof(int)) + (4*sizeof(float)) + sizeof(bool); // zoomWin + point arrs (4 int, 4 float, 1 bool)
   
    // Calculate the amount of memory that needs to be available for kernel-local memory. For Imcor, the worst-case memory
    // usage is during the third kernel, so this value is fixed to what the kernel uses locally (known).
    size_t nb_kernel = (17*sizeof(void*)) + (12*sizeof(float)) + (7*sizeof(int));   // 212 bytes on most systems
    
    // Let's say for safety's sake that every block needs an extra MB. So we'll add a MB and round up to a whole # of MB per block pair
    size_t nb_total = (int(float(nb_malloc + nb_kernel) / 1.e6) + 2) * 1e6; // # bytes per block pair

    printf("Single block-pair will use a maximum of %d MB.\n", int(nb_total/1e6)); // Info for user to see (should be roughly 6-7 MB at worst for defaults)

    return int(NB_MAX / nb_total);
}

void checkKernelErrors() {
    /*
    *   Synchronizes the host and device after a kernel call, checks for synchronous and asynchronous device errors, and prints
    *   the relevant error message if applicable.
    */

    cudaError_t errSync = cudaGetLastError();       // Gets any errors that occur after launching the kernel
    cudaError_t errAsync = cudaDeviceSynchronize(); // Holds the host from running until the kernel is finished, and gets any errors
                                                    // that occur on device synchronization

    // Display any errors that occurred
    if (errSync != cudaSuccess) {
        printf("\nSync kernel error: %s\n", cudaGetErrorString(errSync));
    } if (errAsync != cudaSuccess) {
        printf("\nAsync kernel error: %s\n", cudaGetErrorString(errAsync));
    }
}

// --------------------- C FUNCTIONS (CONTROLLER) -----------------------

void runGPUAmpcor(float *h_inpts_flt, int *h_inpts_int, void **refBlocks, void **schBlocks, int *globalX, int *globalY, int **retArrs_int, float **retArrs_flt) {
    /*
    *   This is the GPU code's equivalent of a "main()" function. This is called from the primary C++ Ampcor code, which passes in the relevant
    *   input arrays and parameters as necessary, as well as the array of pointers to write the output data to after each run.
    *
    *   Input Constants:
    *   h_inpts_flt[0] = snrThresh
    *   h_inpts_flt[1] = covThresh
    *   h_inpts_int[0] = refChipWidth
    *   h_inpts_int[1] = refChipHeight
    *   h_inpts_int[2] = schWinWidth
    *   h_inpts_int[3] = schWinHeight
    *   h_inpts_int[4] = acrossGrossOffset
    *   h_inpts_int[5] = downGrossOffset
    *   h_inpts_int[6] = padWinWidth
    *   h_inpts_int[7] = padWinHeight
    *   h_inpts_int[8] = zoomWinSize
    *   h_inpts_int[9] = nBlocks
    */

    // Since we need to convert the complex<float> arrays in C++ to complex float arrays in C, we have to take
    // advantage of C allowing for blind void-to-xxx pointer casting (fine here because it's internal)
    cuFloatComplex **h_refBlocks = (cuFloatComplex **)refBlocks;
    cuFloatComplex **h_schBlocks = (cuFloatComplex **)schBlocks;

    // To avoid adding an extra layer of complexity to each kernel, the input/output arrays in the 
    // CUDA code will be linearly contiguous in 1D. Each kernel will handle selecting the right starting
    // point in the reference array to use as the kernel's "copy" of the array (does not actually copy data)
    cuFloatComplex *d_refBlocks, *d_schBlocks;

    // Device output arrays
    float *d_locAcOffArr, *d_locDnOffArr, *d_snrArr, *d_cov1Arr, *d_cov2Arr, *d_cov3Arr;
    int *d_locAcArr, *d_locDnArr, *d_gX, *d_gY;

    // Device scratch-work arrays
    cuFloatComplex *d_padRefChips, *d_padSchWins, *d_zoomWins;
    float *d_schSums, *d_schSumSqs, *d_refNorms;
    int *d_roughPeakColArr, *d_roughPeakRowArr;
    bool *d_flagArr;

    // Timing variables
    double startRun, endRun, startProcess, endProcess;

    // Structs to collect and organize the various array pointers needed by each kernel
    struct StepZeroData s0data;
    struct StepOneData s1data;
    struct StepTwoData s2data;
    struct StepThreeData s3data;

    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
                                                                    Step 1: Set up
     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

    printf("\n  ------------------ INITIALIZING GPU IMCOR ------------------\n");    
    cudaSetDevice(0);   // Targets first (and currently only) GPU device connected to be the one to run the code on

    startRun = cpuSecond(); // Start timing entire GPU run

    printf("    Allocating initial memory... ");
    
    int nRefPixels = h_inpts_int[0] * h_inpts_int[1];       // Number of pixels per ref block
    int nSchPixels = h_inpts_int[2] * h_inpts_int[3];       // Number of pixels per sch block
    int nSumPixels = ((2*h_inpts_int[2])+1) * ((2*h_inpts_int[3])+1); // Number of pixels per sum/sumsq block
    int nPadPixels = h_inpts_int[6] * h_inpts_int[7];       // Number of pixels per padded window block
    int nZoomPixels = 4 * h_inpts_int[8] * h_inpts_int[8];  // Number of pixels per zoom window block
   
    size_t nb_ref = nRefPixels * sizeof(cuFloatComplex);    // Number of bytes per ref block
    size_t nb_sch = nSchPixels * sizeof(cuFloatComplex);    // Number of bytes per sch block
    size_t nb_sum = nSumPixels * sizeof(float);             // Number of bytes per sum/sumsq block
    size_t nb_pad = nPadPixels * sizeof(cuFloatComplex);    // Number of bytes per padded window block
    size_t nb_zoom = nZoomPixels * sizeof(cuFloatComplex);  // Number of bytes per zoom window block
    size_t nb_fltArr = h_inpts_int[9] * sizeof(float);      // Number of bytes for float-type point array
    size_t nb_intArr = h_inpts_int[9] * sizeof(int);        // Number of bytes for int-type point array
    size_t nb_boolArr = h_inpts_int[9] * sizeof(bool);      // Number of bytes for bool-type point array

    // Malloc arrays needed for first kernel on device
    cudaMalloc((cuFloatComplex**)&d_refBlocks, (h_inpts_int[9]*nb_ref));
    cudaMalloc((cuFloatComplex**)&d_schBlocks, (h_inpts_int[9]*nb_sch));
    cudaMalloc((int**)&d_gX, nb_intArr);
    cudaMalloc((int**)&d_gY, nb_intArr);

    printf("Done.\n    Copying data to GPU... ");

    startProcess = cpuSecond(); // Start timing the first memory copy

    // Use pointer logic to copy in the ref/sch blocks to one big array (contiguous)
    // since inputs are arrays of image blocks
    int i;
    for (i=0; i<h_inpts_int[9]; i++) {
        cudaMemcpy((d_refBlocks + (i*nRefPixels)), h_refBlocks[i], nb_ref, cudaMemcpyHostToDevice);
        cudaMemcpy((d_schBlocks + (i*nSchPixels)), h_schBlocks[i], nb_sch, cudaMemcpyHostToDevice);
    }

    // Copy x/y sub-image locations to device
    cudaMemcpy(d_gX, globalX, nb_intArr, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gY, globalY, nb_intArr, cudaMemcpyHostToDevice);
    
    // Copy constants to device
    cudaMemcpyToSymbol(inf, h_inpts_flt, (2*sizeof(float)));
    cudaMemcpyToSymbol(ini, h_inpts_int, (10*sizeof(int)));

    endProcess = cpuSecond();

    printf("Done. (%f s.)\n", (endProcess-startProcess));

    // Set up thread grid/blocks for the block-by-block operations
    dim3 block(THREAD_PER_IMG_BLOCK);
    dim3 grid(int((h_inpts_int[9] + (THREAD_PER_IMG_BLOCK-1)) / THREAD_PER_IMG_BLOCK));    // == ceil(nPairs / THREAD_PER_IMG_BLOCK) , preserves warp sizing
    if ((grid.x * THREAD_PER_IMG_BLOCK) > h_inpts_int[9]) 
        printf("    (NOTE: There will be %d 'empty' threads in the last thread block).\n", ((grid.x*THREAD_PER_IMG_BLOCK)-h_inpts_int[9]));

    // Set up thread grid/blocks for the pixel-by-pixel operations on the padded windows
    dim3 block2(THREAD_PER_PIX_BLOCK);
    dim3 grid2(int(((h_inpts_int[9]*h_inpts_int[6]*h_inpts_int[7]) + (THREAD_PER_PIX_BLOCK-1)) / THREAD_PER_PIX_BLOCK));

    // Set up thread grid/blocks for the pixel-by-pixel operations on the zoom windows
    dim3 block3(THREAD_PER_PIX_BLOCK);
    dim3 grid3(int(((h_inpts_int[9]*4*h_inpts_int[8]*h_inpts_int[8]) + (THREAD_PER_PIX_BLOCK-1)) / THREAD_PER_PIX_BLOCK)); 

    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
                                                            Step 2: Run first kernel
     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

    printf("    Starting GPU Imcor...\n    Stage 1: Pre-process blocks...      ");
    fflush(stdout);

    startProcess = cpuSecond(); // Start timing the first kernel execution

    // Malloc new memory needed specifically for the kernel (and potentially beyond)
    cudaMalloc((cuFloatComplex**)&d_padRefChips, (h_inpts_int[9]*nb_pad));
    cudaMalloc((cuFloatComplex**)&d_padSchWins, (h_inpts_int[9]*nb_pad));
    cudaMalloc((int**)&d_locAcArr, nb_intArr);
    cudaMalloc((int**)&d_locDnArr, nb_intArr);
    
    // Set padded windows to 0
    cudaMemset(d_padRefChips, 0, (h_inpts_int[9]*nb_pad));
    cudaMemset(d_padSchWins, 0, (h_inpts_int[9]*nb_pad));

    // Store pointers to device memory malloc'ed since we can pass the structs in
    // by value (which will just copy the pointers over)
    s0data.refBlocks = d_refBlocks;
    s0data.schBlocks = d_schBlocks;
    s0data.padRefChips = d_padRefChips;
    s0data.padSchWins = d_padSchWins;
    s0data.locationAcrossArr = d_locAcArr;
    s0data.locationDownArr = d_locDnArr;
    s0data.globalX = d_gX;
    s0data.globalY = d_gY;

    // Run first kernel
    prepBlocks <<<grid, block>>>(s0data);
    checkKernelErrors();

    endProcess = cpuSecond(); // Stop timing the first kernel execution
    printf("Done. (%f s.)\n", (endProcess-startProcess));

    // Clean as you go!
    cudaFree(s0data.refBlocks);
    cudaFree(s0data.schBlocks);
    cudaFree(s0data.globalX);
    cudaFree(s0data.globalY);

    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
                                                            Step 3: Run first FFT-spread
     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

    printf("    Stage 2: FFT-spread blocks...       ");
    fflush(stdout);

    startProcess = cpuSecond(); // Start timing the FFT-spread

    // Create batched plans to run multiple 2D FFTs
    cufftHandle fwd_Plan, inv_Plan;
    
    // Dimensions of the areas to FFT over within the primary padded window (so we don't 
    // FFT the entire window before spreading). h_inpts_int[6/7]/2 is half the size of the
    // padded windows (need both FFT-spreads to have the same frequency sampling, doesn't
    // affect numerical output of the oversampled data). Note the column-major ordering
    // to be compatible with cuFFT's layout
    int npts[2] = {h_inpts_int[7]/2, h_inpts_int[6]/2};
    int inv_npts[2] = {h_inpts_int[7], h_inpts_int[6]};

    // Set batched plans to use advanced data layouts (so we can work in-place with the array blocks)
    cufftPlanMany(&fwd_Plan, 2, npts, inv_npts, 1, h_inpts_int[6]*h_inpts_int[7],
                                      inv_npts, 1, h_inpts_int[6]*h_inpts_int[7],
                                      CUFFT_C2C, h_inpts_int[9]);

    // The inverse FFTs don't need advanced layouts since the entire padded windows will (initially) have data
    cufftPlanMany(&inv_Plan, 2, inv_npts, NULL, 1, h_inpts_int[6]*h_inpts_int[7],
                                          NULL, 1, h_inpts_int[6]*h_inpts_int[7],
                                          CUFFT_C2C, h_inpts_int[9]);

    // Run the forward FFTs (spreads out the data in-place in the padded ref/sch blocks using the callback tied to the plan)
    cufftExecC2C(fwd_Plan, (cufftComplex *)s0data.padRefChips, (cufftComplex *)s0data.padRefChips, CUFFT_FORWARD);
    cufftExecC2C(fwd_Plan, (cufftComplex *)s0data.padSchWins, (cufftComplex *)s0data.padSchWins, CUFFT_FORWARD);
    cufftDestroy(fwd_Plan); // Cleanup!

    spreadPaddedBlock <<<grid2, block2>>>(s0data.padRefChips);
    checkKernelErrors();
    spreadPaddedBlock <<<grid2, block2>>>(s0data.padSchWins);
    checkKernelErrors();

    // Run the inverse FFTs
    cufftCallbackStoreC h_refNormMagPtr, h_schNormMagPtr;

    cudaMemcpyFromSymbol(&h_refNormMagPtr, d_refNormMagPtr, sizeof(h_refNormMagPtr)); // Copy the device pointer to host
    cudaMemcpyFromSymbol(&h_schNormMagPtr, d_schNormMagPtr, sizeof(h_schNormMagPtr));
    
    cufftXtSetCallback(inv_Plan, (void **)&h_refNormMagPtr, CUFFT_CB_ST_COMPLEX, NULL); // Bind the first callback to the plan
    cufftExecC2C(inv_Plan, (cufftComplex *)s0data.padRefChips, (cufftComplex *)s0data.padRefChips, CUFFT_INVERSE);
    
    cufftXtClearCallback(inv_Plan, CUFFT_CB_ST_COMPLEX); // Unbind the first callback from the plan
    cufftXtSetCallback(inv_Plan, (void **)&h_schNormMagPtr, CUFFT_CB_ST_COMPLEX, NULL); // Bind the second callback to the plan
    cufftExecC2C(inv_Plan, (cufftComplex *)s0data.padSchWins, (cufftComplex *)s0data.padSchWins, CUFFT_INVERSE);
    cufftDestroy(inv_Plan); // Cleanup!
    
    endProcess = cpuSecond(); // Stop timing the FFT-spread
    
    printf("Done. (%f s.)\n", (endProcess-startProcess));

    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
                                                            Step 4: Run second kernel
     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

    printf("    Stage 3: Accumulate block sums...   ");
    fflush(stdout);

    startProcess = cpuSecond(); // Start timing the second kernel

    // Malloc new memory needed
    cudaMalloc((float**)&d_schSums, (h_inpts_int[9]*nb_sum));
    cudaMalloc((float**)&d_schSumSqs, (h_inpts_int[9]*nb_sum));
    cudaMalloc((float**)&d_refNorms, nb_fltArr);
    cudaMemset(d_schSums, 0, (h_inpts_int[9]*nb_sum));
    cudaMemset(d_schSumSqs, 0, (h_inpts_int[9]*nb_sum));

    // Copy device pointers to local host structs
    s1data.padRefChips = d_padRefChips;
    s1data.padSchWins = d_padSchWins;
    s1data.schSums = d_schSums;
    s1data.schSumSqs = d_schSumSqs;
    s1data.refNorms = d_refNorms;

    // Run the second kernel
    accumulate <<<grid, block>>>(s1data);
    checkKernelErrors();
    // ********** DEBUG ************
    struct StepOneData s1sdata, s1ssdata;
    
    s1sdata.padSchWins = d_padSchWins;
    s1sdata.schSums = d_schSums;
    accumulateSum <<<grid, block>>>(s1sdata);
    checkKernelErrors();

    s1ssdata.padSchWins = d_padSchWins;
    s1ssdata.schSumSqs = d_schSumSqs;
    accumulateSumSq <<<grid, block>>>(s1ssdata);
    checkKernelErrors();
    // *****************************

    endProcess = cpuSecond(); // Stop timing the second kernel
    printf("Done. (%f s.)\n", (endProcess-startProcess));

    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
                                                    Step 5: Cross-multiply the ref and sch blocks
     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

    printf("    Stage 4: Cross-multiply blocks...   ");
    fflush(stdout);

    startProcess = cpuSecond(); // Start timing the FFT cross-multiply

    // Set batched plans (we don't need to use advanced data layout here as we're 
    // operating over the whole windows). Also reuse dimension tuples from earlier
    cufftPlanMany(&fwd_Plan, 2, inv_npts, NULL, 1, h_inpts_int[6]*h_inpts_int[7],
                                          NULL, 1, h_inpts_int[6]*h_inpts_int[7],
                                          CUFFT_C2C, h_inpts_int[9]);

    // Run the forward FFT on the ref win
    cufftExecC2C(fwd_Plan, (cufftComplex *)s1data.padRefChips, (cufftComplex *)s1data.padRefChips, CUFFT_FORWARD);
   
    cufftCallbackStoreC h_conjMultPtr;
    cudaMemcpyFromSymbol(&h_conjMultPtr, d_conjMultPtr, sizeof(h_conjMultPtr)); // Copy the device pointer to host
    cufftXtSetCallback(fwd_Plan, (void **)&h_conjMultPtr, CUFFT_CB_ST_COMPLEX, (void **)&d_padRefChips); // Bind the callback to the plan

    // Run the forward FFT on the sch win, running the complex-conj cross-mul after the FFT
    cufftExecC2C(fwd_Plan, (cufftComplex *)s1data.padSchWins, (cufftComplex *)s1data.padSchWins, CUFFT_FORWARD);
    // Clear the callback from the plan so we can use it again
    cufftXtClearCallback(fwd_Plan, CUFFT_CB_ST_COMPLEX);

    // Run the inverse FFTs (runs the fft-shift on the sch iFFT)
    cufftExecC2C(fwd_Plan, (cufftComplex *)s1data.padRefChips, (cufftComplex *)s1data.padRefChips, CUFFT_INVERSE);
    cufftExecC2C(fwd_Plan, (cufftComplex *)s1data.padSchWins, (cufftComplex *)s1data.padSchWins, CUFFT_INVERSE);
    cufftDestroy(fwd_Plan); // Cleanup!

    // FFT-shift the correlation surface
    fftShiftCorr <<<grid2, block2>>>(s1data.padSchWins);
    checkKernelErrors();

    endProcess = cpuSecond(); // Stop timing the FFT cross-multiply

    printf("Done. (%f s.)\n", (endProcess-startProcess));

    // Clean as you go!
    cudaFree(s1data.padRefChips);

    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
                            Step 6: Fill normalized correlation surface and calculate rough offsets, covariances, and SNR
     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

    printf("    Stage 5: Calculate rough offsets... ");
    fflush(stdout);

    startProcess = cpuSecond(); // Start timing the third kernel

    // Malloc new memory needed
    cudaMalloc((cuFloatComplex**)&d_zoomWins, (h_inpts_int[9]*nb_zoom));
    cudaMalloc((int**)&d_roughPeakColArr, nb_intArr);
    cudaMalloc((int**)&d_roughPeakRowArr, nb_intArr);
    cudaMalloc((bool**)&d_flagArr, nb_boolArr);
    cudaMalloc((float**)&d_snrArr, nb_fltArr);
    cudaMalloc((float**)&d_cov1Arr, nb_fltArr);
    cudaMalloc((float**)&d_cov2Arr, nb_fltArr);
    cudaMalloc((float**)&d_cov3Arr, nb_fltArr);

    // Zero out zoom windows
    cudaMemset(d_zoomWins, 0, h_inpts_int[9]*nb_zoom);

    // Store device pointers in local host struct
    s2data.corrWins = d_padSchWins;
    s2data.zoomWins = d_zoomWins;
    s2data.schSums = d_schSums;
    s2data.schSumSqs = d_schSumSqs;
    s2data.refNorms = d_refNorms;
    s2data.roughPeakColArr = d_roughPeakColArr;
    s2data.roughPeakRowArr = d_roughPeakRowArr;
    s2data.cov1Arr = d_cov1Arr;
    s2data.cov2Arr = d_cov2Arr;
    s2data.cov3Arr = d_cov3Arr;
    s2data.snrArr = d_snrArr;
    s2data.flagArr = d_flagArr;
    
    // Run the third kernel
    calcRough <<<grid, block>>>(s2data);
    checkKernelErrors();
    
    endProcess = cpuSecond(); // Stop timing the third kernel
    printf("Done. (%f s.)\n", (endProcess-startProcess));

    // Clean as you go!
    cudaFree(s2data.corrWins);
    cudaFree(s2data.schSums);
    cudaFree(s2data.schSumSqs);
    cudaFree(s2data.refNorms);

    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
                                                            Step 7: Run second FFT-spread
     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

    printf("    Stage 6: FFT-spread block...        ");
    fflush(stdout);

    startProcess = cpuSecond();

    // Dimensions of the areas to FFT over within the primary padded window (so we don't
    // FFT the entire window before spreading)
    int zoomN[2] = {h_inpts_int[8], h_inpts_int[8]};
    int inv_zoomN[2] = {2*h_inpts_int[8], 2*h_inpts_int[8]};

    // Set batched plans to use advanced data layouts (so we can work in-place with the array blocks), just on FFT.
    // Reuse older plan handles for cleanliness
    cufftPlanMany(&fwd_Plan, 2, zoomN, inv_zoomN, 1, 4*h_inpts_int[8]*h_inpts_int[8],
                                       inv_zoomN, 1, 4*h_inpts_int[8]*h_inpts_int[8],
                                       CUFFT_C2C, h_inpts_int[9]);

    cufftPlanMany(&inv_Plan, 2, inv_zoomN, NULL, 1, 4*h_inpts_int[8]*h_inpts_int[8],
                                           NULL, 1, 4*h_inpts_int[8]*h_inpts_int[8],
                                           CUFFT_C2C, h_inpts_int[9]);

    // Run the forward FFTs (spreads out the data in-place in the padded ref/sch blocks using the callback tied to the plan)
    cufftExecC2C(fwd_Plan, (cufftComplex *)s2data.zoomWins, (cufftComplex *)s2data.zoomWins, CUFFT_FORWARD);
    cufftDestroy(fwd_Plan); // Cleanup!

    spreadZoomBlock <<<grid3, block3>>>(s2data.zoomWins);
    checkKernelErrors();

    cufftCallbackStoreC h_zoomNormMagPtr;
    
    // Copy the device pointer to host
    cudaMemcpyFromSymbol(&h_zoomNormMagPtr, d_zoomNormMagPtr, sizeof(h_zoomNormMagPtr));
    // Bind the callback to the plan 
    cufftXtSetCallback(inv_Plan, (void **)&h_zoomNormMagPtr, CUFFT_CB_ST_COMPLEX, NULL);
    // Run the inverse FFTs (the data was spread out at the end of the prior forward FFTs)
    cufftExecC2C(inv_Plan, (cufftComplex *)s2data.zoomWins, (cufftComplex *)s2data.zoomWins, CUFFT_INVERSE);
    cufftDestroy(inv_Plan); // Cleanup!

    endProcess = cpuSecond(); // Stop timing the FFT-spread
    printf("Done. (%f s.)\n", (endProcess-startProcess));

    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
                                            Step 8: Calculate fine offsets and store results as necessary
     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

    printf("    Stage 7: Calculate fine offsets...  ");
    fflush(stdout);

    startProcess = cpuSecond(); // Start timing the fourth kernel

    // Malloc new memory needed
    cudaMalloc((float**)&d_locAcOffArr, nb_fltArr);
    cudaMalloc((float**)&d_locDnOffArr, nb_fltArr);

    // Copy device pointers to local host struct
    s3data.zoomWins = d_zoomWins;
    s3data.locationAcrossOffsetArr = d_locAcOffArr;
    s3data.locationDownOffsetArr = d_locDnOffArr;
    s3data.roughPeakColArr = d_roughPeakColArr;
    s3data.roughPeakRowArr = d_roughPeakRowArr;
    s3data.flagArr = d_flagArr;
    s3data.locationAcrossArr = d_locAcArr;
    s3data.locationDownArr = d_locDnArr;
    s3data.cov1Arr = d_cov1Arr;
    s3data.cov2Arr = d_cov2Arr;
    s3data.cov3Arr = d_cov3Arr;
    s3data.snrArr = d_snrArr;

    // Run fourth kernel
    calcFine <<<grid, block>>>(s3data);
    checkKernelErrors();

    endProcess = cpuSecond(); // Stop timing the fourth kernel
    printf("Done. (%f s.)\n", (endProcess-startProcess));

    // Clean as you go!
    cudaFree(s3data.zoomWins);
    cudaFree(s3data.roughPeakColArr);
    cudaFree(s3data.roughPeakRowArr);
    cudaFree(s3data.flagArr);

    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
                                                            Step 9: Clean up
     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

    printf("    Copying memory back to host...      ");
    fflush(stdout);

    startProcess = cpuSecond(); // Start timing second memory copy

    // Copy outputs from device to host
    cudaMemcpy(retArrs_int[0], d_locAcArr, nb_intArr, cudaMemcpyDeviceToHost);
    cudaMemcpy(retArrs_int[1], d_locDnArr, nb_intArr, cudaMemcpyDeviceToHost);
    cudaMemcpy(retArrs_flt[0], d_locAcOffArr, nb_fltArr, cudaMemcpyDeviceToHost);
    cudaMemcpy(retArrs_flt[1], d_locDnOffArr, nb_fltArr, cudaMemcpyDeviceToHost);
    cudaMemcpy(retArrs_flt[2], d_snrArr, nb_fltArr, cudaMemcpyDeviceToHost);
    cudaMemcpy(retArrs_flt[3], d_cov1Arr, nb_fltArr, cudaMemcpyDeviceToHost);
    cudaMemcpy(retArrs_flt[4], d_cov2Arr, nb_fltArr, cudaMemcpyDeviceToHost);
    cudaMemcpy(retArrs_flt[5], d_cov3Arr, nb_fltArr, cudaMemcpyDeviceToHost);

    endProcess = cpuSecond(); // Stop timing second memory copy
    endRun = cpuSecond(); // Stop timing GPU run

    printf("Done. (%f s.)\n", (endProcess-startProcess));
    printf("    Finished GPU Imcor in %f s.\n", (endRun-startRun));
    printf("    Cleaning device memory and returning to main Topo function...\n");

    // Free up output memory on device
    cudaFree(d_locAcArr);
    cudaFree(d_locDnArr);
    cudaFree(d_locAcOffArr);
    cudaFree(d_locDnOffArr);
    cudaFree(d_snrArr);
    cudaFree(d_cov1Arr);
    cudaFree(d_cov2Arr);
    cudaFree(d_cov3Arr);
    cudaDeviceReset();  // Not 100% needed, but makes sure that next GPU run is done with a clean device

    printf("\n  ------------------ EXITING GPU AMPCOR ------------------\n\n");
}
