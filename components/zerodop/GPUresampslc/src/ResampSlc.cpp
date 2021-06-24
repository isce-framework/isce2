//
// Author: Joshua Cohen
// Copyright 2017
//
// Note the algorithm has been updated to both tile the input image processing, as well as switch
// from column-major Fortran ordering to row-major C++ ordering. For the purposes of this algorithm,
// the "image" refers to the full input or output image, whereas the "tile" refers to a block of
// between 1 and LINES_PER_TILE output image lines.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <omp.h>
#include <vector>
#include "DataAccessor.h"
#include "Constants.h"
#include "Poly2d.h"
#include "ResampMethods.h"
#include "ResampSlc.h"
#ifdef GPU_ACC_ENABLED
#include "GPUresamp.h"
#endif

#define LINES_PER_TILE 1000

using std::complex;
using std::max;
using std::min;
using std::vector;

ResampSlc::ResampSlc() {
    rgCarrier = new Poly2d();
    azCarrier = new Poly2d();
    rgOffsetsPoly = new Poly2d();
    azOffsetsPoly = new Poly2d();
    dopplerPoly = new Poly2d();
    slcInAccessor = 0;
    slcOutAccessor = 0;
    residRgAccessor = 0;
    residAzAccessor = 0;
    usr_enable_gpu = true;
}

ResampSlc::ResampSlc(const ResampSlc &rsmp) {
    rgCarrier = new Poly2d(*rsmp.rgCarrier);
    azCarrier = new Poly2d(*rsmp.azCarrier);
    rgOffsetsPoly = new Poly2d(*rsmp.rgOffsetsPoly);
    azOffsetsPoly = new Poly2d(*rsmp.azOffsetsPoly);
    dopplerPoly = new Poly2d(*rsmp.dopplerPoly);
    slcInAccessor = rsmp.slcInAccessor;
    slcOutAccessor = rsmp.slcOutAccessor;
    residRgAccessor = rsmp.residRgAccessor;
    residAzAccessor = rsmp.residAzAccessor;
    usr_enable_gpu = rsmp.usr_enable_gpu;
}

ResampSlc::~ResampSlc() {
    clearPolys();
}

void ResampSlc::setRgCarrier(Poly2d *poly) {
    if (rgCarrier != NULL) delete rgCarrier;
    rgCarrier = poly;
}

void ResampSlc::setAzCarrier(Poly2d *poly) {
    if (azCarrier != NULL) delete azCarrier;
    azCarrier = poly;
}

void ResampSlc::setRgOffsets(Poly2d *poly) {
    if (rgOffsetsPoly != NULL) delete rgOffsetsPoly;
    rgOffsetsPoly = poly;
}

void ResampSlc::setAzOffsets(Poly2d *poly) {
    if (azOffsetsPoly != NULL) delete azOffsetsPoly;
    azOffsetsPoly = poly;
}

void ResampSlc::setDoppler(Poly2d *poly) {
    if (dopplerPoly != NULL) delete dopplerPoly;
    dopplerPoly = poly;
}

// * * * * * * * * *    NOTE: THESE SHOULD BE USED WITH EXTREME PREJUDICE    * * * * * * * * *
Poly2d* ResampSlc::releaseRgCarrier() {
    Poly2d *tmp = rgCarrier;
    rgCarrier = NULL;
    return tmp;
}

Poly2d* ResampSlc::releaseAzCarrier() {
    Poly2d *tmp = azCarrier;
    azCarrier = NULL;
    return tmp;
}

Poly2d* ResampSlc::releaseRgOffsets() {
    Poly2d *tmp = rgOffsetsPoly;
    rgOffsetsPoly = NULL;
    return tmp;
}

Poly2d* ResampSlc::releaseAzOffsets() {
    Poly2d *tmp = azOffsetsPoly;
    azOffsetsPoly = NULL;
    return tmp;
}

Poly2d* ResampSlc::releaseDoppler() {
    Poly2d *tmp = dopplerPoly;
    dopplerPoly = NULL;
    return tmp;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

void ResampSlc::clearPolys() {
    if (rgCarrier != NULL) delete rgCarrier;
    if (azCarrier != NULL) delete azCarrier;
    if (rgOffsetsPoly != NULL) delete rgOffsetsPoly;
    if (azOffsetsPoly != NULL) delete azOffsetsPoly;
    if (dopplerPoly != NULL) delete dopplerPoly;
}

void ResampSlc::resetPolys() {
    clearPolys();
    rgCarrier = new Poly2d();
    azCarrier = new Poly2d();
    rgOffsetsPoly = new Poly2d();
    azOffsetsPoly = new Poly2d();
    dopplerPoly = new Poly2d();
}

void copyPolyToArr(Poly2d *poly, vector<double> &destArr) {
    // Len of destArr is at least 7
    destArr[0] = poly->azimuthOrder;
    destArr[1] = poly->rangeOrder;
    destArr[2] = poly->azimuthMean;
    destArr[3] = poly->rangeMean;
    destArr[4] = poly->azimuthNorm;
    destArr[5] = poly->rangeNorm;
    for (int i=0; i<((destArr[0]+1)*(destArr[1]+1)); i++) destArr[6+i] = poly->coeffs[i];
}


// wrapper for calling cpu or gpu methods
void ResampSlc::resamp()
{
    #ifndef GPU_ACC_ENABLED
    usr_enable_gpu = false;
    #endif

    if (usr_enable_gpu) {
        _resamp_gpu();
    }
    else {
        _resamp_cpu();
    }
}

// not checked
void ResampSlc::_resamp_cpu() {

    vector<double> residAz(outWidth,0.), residRg(outWidth,0.);
    double ro, ao, ph, dop, fracr, fraca, t0, k, kk;

    vector<vector<complex<float> > > chip(SINC_ONE, vector<complex<float> >(SINC_ONE));
    vector<complex<float> > imgIn(0); // Linearizing the image so it's easier to pass around
    vector<complex<float> > imgOut(outWidth,complex<float>(0.,0.));
    complex<float> cval;

    int chipi, chipj, nTiles, lastLines, firstImageRow, lastImageRow, firstTileRow;
    int imgLine, nRowsInTile, nRowsInBlock;

    ResampMethods rMethods;

    DataAccessor *slcInAccObj = (DataAccessor*)slcInAccessor;
    DataAccessor *slcOutAccObj = (DataAccessor*)slcOutAccessor;
    DataAccessor *residRgAccObj, *residAzAccObj;
    if (residRgAccessor != 0) residRgAccObj = (DataAccessor*)residRgAccessor;
    else residRgAccObj = NULL;
    if (residAzAccessor != 0) residAzAccObj = (DataAccessor*)residAzAccessor;
    else residAzAccObj = NULL;


    // Moving this here so we don't waste any time
    if (!isComplex) {
        printf("Real data interpolation not implemented yet.\n");
        return;
    }

    t0 = omp_get_wtime();

    printf("\n << Resample one image to another image coordinates >> \n\n");
    printf("Input Image Dimensions:  %6d lines, %6d pixels\n\n", inLength, inWidth);
    printf("Output Image Dimensions: %6d lines, %6d pixels\n\n", outLength, outWidth);

    printf("Number of threads: %d\n", omp_get_max_threads());
    printf("Complex data interpolation\n");

    rMethods.prepareMethods(SINC_METHOD);

    printf("Azimuth Carrier Poly\n");
    azCarrier->printPoly();
    printf("Range Carrier Poly\n");
    rgCarrier->printPoly();
    printf("Range Offsets Poly\n");
    rgOffsetsPoly->printPoly();
    printf("Azimuth Offsets Poly\n");
    azOffsetsPoly->printPoly();

    // Determine number of tiles needed to process image
    nTiles = outLength / LINES_PER_TILE;
    lastLines = outLength - (nTiles * LINES_PER_TILE);
    printf("Resampling in %d tile(s) of %d line(s)", nTiles, LINES_PER_TILE);
    if (lastLines > 0) {
        printf(", with a final tile containing %d line(s)", lastLines);
    }
    printf("\n");

    // For each full tile of LINES_PER_TILE lines...
    for (int tile=0; tile<nTiles; tile++) {

        firstTileRow = tile * LINES_PER_TILE; // Equivalent to line number in output image

        // To determine the amount of data we need to read into imgIn, we need to figure out the minimum and maximum row indices
        // of the original input image that get accessed by each tile. These indices are bound by the values of the azimuth
        // polynomial offsets as well as the residual offsets. Since we don't want to evaluate the offsets of LINES_PER_TILE*
        // inWidth points before the actual processing, we can stick to the GDAL convention of checking ~40 pixels on each edge.
        // Since we're reading in entire lines, we only have to check the top/bottom 40 lines of the image (no need to check the
        // edges). So we eval the top 40 lines worth of pixel offsets and find the largest negative row shift (the minimum row
        // index from the reference input image). Then we eval the bottom 40 lines worth of pixel offsets and find the largest
        // positive row shift (the maximum row index). This gives us the number of lines to read, as well as the firstImageRow
        // reference offset for the tile. Note that firstImageRow/lastImageRow are row indices relative to the reference input image.

        printf("Reading in image data for tile %d\n", tile);

        firstImageRow = outLength - 1;                                                  // Initialize to last image row
        for (int i=firstTileRow; i<(firstTileRow+40); i++) {                            // Iterate over first 40 lines of tile (bounded by total # of rows in tile)
            if (residAzAccessor != 0) residAzAccObj->getLine((char *)&residAz[0], i);   // Read in azimuth residual if it exists
            for (int j=0; j<outWidth; j++) {                                            // Iterate over the width of the tile
                ao = azOffsetsPoly->eval(i+1, j+1) + residAz[j];                        // Evaluate net azimuth offset of each pixel in row
                //imgLine = int(i + ao + 1) - SINC_HALF;                              // Calculate corresponding minimum line idx of input image
                imgLine = int(i+ao) - SINC_HALF;
                firstImageRow = min(firstImageRow, imgLine);                            // Set the first input image line idx to the smallest value
            }
        }
        firstImageRow = max(firstImageRow, 0);                                          // firstImageRow now has the lowest image row called in the tile processing

        lastImageRow = 0;                                                                       // Initialize to first image row
        for (int i=(firstTileRow+LINES_PER_TILE-40); i<(firstTileRow+LINES_PER_TILE); i++) {   // Iterate over last 40 lines of tile
            if (residAzAccessor != 0) residAzAccObj->getLine((char *)&residAz[0], i);           // Read in azimuth residual
            for (int j=0; j<outWidth; j++) {                                                    // Iterate over the width of the tile
                ao = azOffsetsPoly->eval(i+1, j+1) + residAz[j];                                 // Evaluate net azimuth offset of each pixel in row
                //imgLine = int(i + ao + 1) + SINC_LEN - SINC_HALF;                           // Calculate corresponding maximum line idx of input image
                                                                                                // (note includes the SINC_LEN added later)
                imgLine = int(i+ao) + SINC_HALF;
                lastImageRow = max(lastImageRow, imgLine);                                      // Set last input image line idx to the largest value
            }
        }
        lastImageRow = min(lastImageRow, inLength-1);       // lastImageRow now has the highest image row called in the tile processing

        nRowsInBlock = lastImageRow - firstImageRow + 1;    // Number of rows in imgIn (NOT TILE)

        // Resize the image tile to the necessary number of lines if necessary using value-initialization resizing (automatically resizes/initializes new rows)
        if (imgIn.size() < size_t(nRowsInBlock*inWidth)) imgIn.resize(nRowsInBlock*inWidth);
        for (int i=0; i<nRowsInBlock; i++) {                                    // Read in nRowsInBlock lines of data from the input image to the image block
            slcInAccObj->getLine((char *)&(imgIn[IDX1D(i,0,inWidth)]), firstImageRow+i);      // Sets imgIn[0] == reference_image[firstImageRow]

            // Remove the carriers using OpenMP acceleration
            #pragma omp parallel for private(ph)
            for (int j=0; j<inWidth; j++) {
                ph = modulo_f(rgCarrier->eval(firstImageRow+i+1,j+1) + azCarrier->eval(firstImageRow+i+1,j+1), 2.*M_PI);    // Evaluate the pixel's carrier
                imgIn[IDX1D(i,j,inWidth)] = imgIn[IDX1D(i,j,inWidth)] * complex<float>(cos(ph), -sin(ph));                  // Remove the carrier
            }
        }

        // Loop over lines
        printf("Interpolating tile %d\n", tile);


        // Interpolation of the complex image. Note that we don't need to make very many changes to the original code in this loop
        // since the i-index should numerically match the original i-index
        for (int i=firstTileRow; i<(firstTileRow+LINES_PER_TILE); i++) {
            // GetLineSequential is fine here, we don't need specific lines, just continue grabbing them
            if (residAzAccessor != 0) residAzAccObj->getLineSequential((char *)&residAz[0]);
            if (residRgAccessor != 0) residRgAccObj->getLineSequential((char *)&residRg[0]);

            #pragma omp parallel for private(ro,ao,fracr,fraca,ph,cval,dop,chipi,chipj,k,kk) \
                                     firstprivate(chip)
            for (int j=0; j<outWidth; j++) {

                ao = azOffsetsPoly->eval(i+1,j+1) + residAz[j];
                ro = rgOffsetsPoly->eval(i+1,j+1) + residRg[j];

                fraca = modf(i+ao, &k);
                if ((k < SINC_HALF) || (k >= (inLength-SINC_HALF))) continue;

                fracr = modf(j+ro, &kk);
                if ((kk < SINC_HALF) || (kk >= (inWidth-SINC_HALF))) continue;

                dop = dopplerPoly->eval(i+1,j+1);

                // Data chip without the carriers
                for (int ii=0; ii<SINC_ONE; ii++) {
                    // Subtracting off firstImageRow removes the offset from the first row in the reference
                    // image to the first row actually contained in imgIn
                    chipi = k - firstImageRow + ii - SINC_HALF;
                    cval = complex<float>(cos((ii-4.)*dop), -sin((ii-4.)*dop));
                    for (int jj=0; jj<SINC_ONE; jj++) {
                        chipj = kk + jj - SINC_HALF;
                        // Take out doppler in azimuth
                        chip[ii][jj] = imgIn[IDX1D(chipi,chipj,inWidth)] * cval;
                    }
                }

                // Doppler to be added back. Simultaneously evaluate carrier that needs to be added back after interpolation
                ph = (dop * fraca) + rgCarrier->eval(i+ao,j+ro) + azCarrier->eval(i+ao,j+ro);

                // Flatten the carrier if the user wants to
                if (flatten) {
                    ph = ph + ((4. * (M_PI / wvl)) * ((r0 - refr0) + (j * (slr - refslr)) + (ro * slr))) +
                            ((4. * M_PI * (refr0 + (j * refslr))) * ((1. / refwvl) - (1. / wvl)));
                }

                ph = modulo_f(ph, 2.*M_PI);

                cval = rMethods.interpolate_cx(chip,(SINC_HALF+1),(SINC_HALF+1),fraca,fracr,SINC_ONE,SINC_ONE,SINC_METHOD);

                imgOut[j] = cval * complex<float>(cos(ph), sin(ph));
            }
            slcOutAccObj->setLineSequential((char *)&imgOut[0]);
        }
    }

    // And if there is a final partial tile...
    if (lastLines > 0) {

        firstTileRow = nTiles * LINES_PER_TILE;
        nRowsInTile = outLength - firstTileRow ; // NOT EQUIVALENT TO NUMBER OF ROWS IN IMAGE BLOCK

        printf("Reading in image data for final partial tile\n");

        firstImageRow = outLength - 1;
        for (int i=firstTileRow; i<min(firstTileRow+40,outLength); i++) { // Make sure if nRowsInTile < 40 to not read too many lines
            if (residAzAccessor != 0) residAzAccObj->getLine((char *)&residAz[0], i);
            for (int j=0; j<outWidth; j++) {
                ao = azOffsetsPoly->eval(i+1, j+1) + residAz[j];
                imgLine = int(i+ao) - SINC_HALF;
                firstImageRow = min(firstImageRow, imgLine);
            }
        }
        firstImageRow = max(firstImageRow, 0);

        lastImageRow = 0;
        for (int i=max(firstTileRow,firstTileRow+nRowsInTile-40); i<(firstTileRow+nRowsInTile); i++) { // Make sure if nRowsInTile < 40 to not read too many lines
            if (residAzAccessor != 0) residAzAccObj->getLine((char *)&residAz[0], i);
            for (int j=0; j<outWidth; j++) {
                ao = azOffsetsPoly->eval(i+1, j+1) + residAz[j];
                imgLine = int(i+ao) + SINC_HALF;
                lastImageRow = max(lastImageRow, imgLine);
            }
        }
        lastImageRow = min(lastImageRow, inLength-1);

        nRowsInBlock = lastImageRow - firstImageRow + 1;

        if (imgIn.size() < size_t(nRowsInBlock*inWidth)) imgIn.resize(nRowsInBlock*inWidth);
        for (int i=0; i<nRowsInBlock; i++) {
            slcInAccObj->getLine((char *)&(imgIn[IDX1D(i,0,inWidth)]), firstImageRow+i);

            #pragma omp parallel for private(ph)
            for (int j=0; j<inWidth; j++) {
                ph = modulo_f(rgCarrier->eval(firstImageRow+i+1,j+1) + azCarrier->eval(firstImageRow+i+1,j+1), 2.*M_PI);
                imgIn[IDX1D(i,j,inWidth)] = imgIn[IDX1D(i,j,inWidth)] * complex<float>(cos(ph), -sin(ph));
            }
        }

        printf("Interpolating final partial tile\n");


        for (int i=firstTileRow; i<(firstTileRow+nRowsInTile); i++) {

            if (residAzAccessor != 0) residAzAccObj->getLineSequential((char *)&residAz[0]);
            if (residRgAccessor != 0) residRgAccObj->getLineSequential((char *)&residRg[0]);

            #pragma omp parallel for private(ro,ao,fracr,fraca,ph,cval,dop,chipi,chipj,k,kk) \
                                     firstprivate(chip)
            for (int j=0; j<outWidth; j++) {

                ro = rgOffsetsPoly->eval(i+1,j+1) + residRg[j];
                ao = azOffsetsPoly->eval(i+1,j+1) + residAz[j];

                fraca = modf(i+ao, &k);
                if ((k < SINC_HALF) || (k >= (inLength-SINC_HALF))) continue;

                fracr = modf(j+ro, &kk);
                if ((kk < SINC_HALF) || (kk >= (inWidth-SINC_HALF))) continue;

                dop = dopplerPoly->eval(i+1,j+1);

                // Data chip without the carriers
                for (int ii=0; ii<SINC_ONE; ii++) {
                    // Subtracting off firstImageRow removes the offset from the first row in the reference
                    // image to the first row actually contained in imgIn
                    chipi = k - firstImageRow + ii - SINC_HALF;
                    cval = complex<float>(cos((ii-4.)*dop), -sin((ii-4.)*dop));
                    for (int jj=0; jj<SINC_ONE; jj++) {
                        chipj = kk + jj - SINC_HALF;
                        // Take out doppler in azimuth
                        chip[ii][jj] = imgIn[IDX1D(chipi,chipj,inWidth)] * cval;
                    }
                }

                // Doppler to be added back. Simultaneously evaluate carrier that needs to be added back after interpolation
                ph = (dop * fraca) + rgCarrier->eval(i+ao,j+ro) + azCarrier->eval(i+ao,j+ro);

                // Flatten the carrier if the user wants to
                if (flatten) {
                    ph = ph + ((4. * (M_PI / wvl)) * ((r0 - refr0) + (j * (slr - refslr)) + (ro * slr))) +
                            ((4. * M_PI * (refr0 + (j * refslr))) * ((1. / refwvl) - (1. / wvl)));
                }

                ph = modulo_f(ph, 2.*M_PI);

                cval = rMethods.interpolate_cx(chip,(SINC_HALF+1),(SINC_HALF+1),fraca,fracr,SINC_ONE,SINC_ONE,SINC_METHOD);

                imgOut[j] = cval * complex<float>(cos(ph), sin(ph));

            }
            slcOutAccObj->setLineSequential((char *)&imgOut[0]);
        }
    }
    printf("Elapsed time: %f\n", (omp_get_wtime()-t0));
}

void ResampSlc::_resamp_gpu()
{
    vector<complex<float> > imgIn(inLength*inWidth);
    vector<complex<float> > imgOut(outLength*outWidth);
    vector<float> residAz(outLength*outWidth), residRg(outLength*outWidth);

    ResampMethods rMethods;

    DataAccessor *slcInAccObj = (DataAccessor*)slcInAccessor;
    DataAccessor *slcOutAccObj = (DataAccessor*)slcOutAccessor;

    DataAccessor *residRgAccObj, *residAzAccObj;
    if (residRgAccessor != 0) residRgAccObj = (DataAccessor*)residRgAccessor;
    else residRgAccObj = NULL;
    if (residAzAccessor != 0) residAzAccObj = (DataAccessor*)residAzAccessor;
    else residAzAccObj = NULL;

    // Moving this here so we don't waste any time
    if (!isComplex) {
        printf("Real data interpolation not implemented yet.\n");
        return;
    }

    double t0 = omp_get_wtime();

    printf("\n << Resample one image to another image coordinates >> \n\n");
    printf("Input Image Dimensions:  %6d lines, %6d pixels\n\n", inLength, inWidth);
    printf("Output Image Dimensions: %6d lines, %6d pixels\n\n", outLength, outWidth);

    printf("Complex data interpolation\n");

    rMethods.prepareMethods(SINC_METHOD);

    printf("Azimuth Carrier Poly\n");
    azCarrier->printPoly();
    printf("Range Carrier Poly\n");
    rgCarrier->printPoly();
    printf("Range Offsets Poly\n");
    rgOffsetsPoly->printPoly();
    printf("Azimuth Offsets Poly\n");
    azOffsetsPoly->printPoly();
    printf("Doppler Poly\n");
    dopplerPoly->printPoly();


    printf("Reading in image data ... \n");
    // read the whole input SLC image
    for (int i=0; i<inLength; i++) {
        slcInAccObj->getLineSequential((char *)&imgIn[i*inWidth]);
    }
    // read the residAz if providied
    if (residAzAccessor != 0) {
        for (int i=0; i<outLength; i++)
            residAzAccObj->getLineSequential((char *)&residAz[i*outWidth]);
    }
    if (residRgAccessor != 0) {
        for (int i=0; i<outLength; i++)
            residRgAccObj->getLineSequential((char *)&residRg[i*outWidth]);
    }

    // set up and copy the Poly objects
    vector<double> azOffPolyArr(((azOffsetsPoly->azimuthOrder+1)*(azOffsetsPoly->rangeOrder+1))+6);
    vector<double> rgOffPolyArr(((rgOffsetsPoly->azimuthOrder+1)*(rgOffsetsPoly->rangeOrder+1))+6);
    vector<double> dopPolyArr(((dopplerPoly->azimuthOrder+1)*(dopplerPoly->rangeOrder+1))+6);
    vector<double> azCarPolyArr(((azCarrier->azimuthOrder+1)*(azCarrier->rangeOrder+1))+6);
    vector<double> rgCarPolyArr(((rgCarrier->azimuthOrder+1)*(rgCarrier->rangeOrder+1))+6);

    copyPolyToArr(azOffsetsPoly, azOffPolyArr);     // arrs are: [azord, rgord, azmean, rgmean, aznorm, rgnorm, coeffs...]
    copyPolyToArr(rgOffsetsPoly, rgOffPolyArr);
    copyPolyToArr(dopplerPoly, dopPolyArr);
    copyPolyToArr(azCarrier, azCarPolyArr);
    copyPolyToArr(rgCarrier, rgCarPolyArr);

    double gpu_inputs_d[6];
    int gpu_inputs_i[8];

    gpu_inputs_d[0] = wvl;
    gpu_inputs_d[1] = refwvl;
    gpu_inputs_d[2] = r0;
    gpu_inputs_d[3] = refr0;
    gpu_inputs_d[4] = slr;
    gpu_inputs_d[5] = refslr;

    gpu_inputs_i[0] = inLength;
    gpu_inputs_i[1] = inWidth;
    gpu_inputs_i[2] = outWidth;

    int firstImageRow = 0;
    int firstTileRow = 0;
    int nRowsInBlock = outLength;

    gpu_inputs_i[3] = firstImageRow;
    gpu_inputs_i[4] = firstTileRow;
    gpu_inputs_i[5] = nRowsInBlock;
    gpu_inputs_i[6] = outLength; //LINES_PER_TILE;
    gpu_inputs_i[7] = int(flatten);

    // call gpu routine
    runGPUResamp(gpu_inputs_d, gpu_inputs_i, (void*)&imgIn[0], (void*)&imgOut[0], &residAz[0], &residRg[0],
                            &azOffPolyArr[0], &rgOffPolyArr[0], &dopPolyArr[0], &azCarPolyArr[0], &rgCarPolyArr[0],
                            &(rMethods.fintp[0]));

    // write the output file
   for (int i=0; i<outLength; i++) slcOutAccObj->setLineSequential((char *)&imgOut[i*outWidth]);
   printf("Elapsed time: %f\n", (omp_get_wtime()-t0));
   // all done
}
