//
// Author: Joshua Cohen
// Copyright 2016
//

#define _USE_MATH_DEFINES
#include "AmpcorMethods.h"
#include "Constants.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <ctime>
#include <vector>

using std::abs;
using std::complex;
using std::conj;
using std::fill;
using std::imag;
using std::max;
using std::real;
using std::vector;


void AmpcorMethods::fill_sinc(int &interpLength, float &delay, std::vector<float> &interp) {

    // This routine computes the sinc interpolation coefficients needed by the processor
    // for various range and azimuth interpolations. NOTE this includes the sinc_coef
    // function as it could easily be embedded.
    
    double weight, sinFact, filtFact, weightHeight, offset, idx;

    // number of coefficients

    interpLength = round(relfiltlen / beta); // Range migration interpolation kernel length
    filtercoef = interpLength * decfactor;
    weightHeight = (1. - pedestal) / 2.;
    offset = (filtercoef - 1.) / 2.;

    for (int i=0; i<filtercoef; i++) {
        idx = i - offset;
        weight = (1. - weightHeight) + (weightHeight * cos((M_PI * idx) / offset));
        sinFact = (idx * beta) / decfactor;
        if (sinFact != 0.) filtFact = sin(M_PI * sinFact) / (M_PI * sinFact);
        else filtFact = 1.;
        if (hasWeight == 1) filter[i+1] = filtFact * weight;
        else filter[i+1] = filtFact;
    }

    delay = interpLength / 2.0; // Range migration filter delay

    for (int i=0; i<interpLength; i++) {
        for (int j=0; j<decfactor; j++) {
            interp[i + (j*interpLength)] = filter[j + (i*decfactor)]; // interpolation kernel values
        }
    }
}

void AmpcorMethods::startInnerClock() {
    innerStart = clock();
}

void AmpcorMethods::startOuterClock() {
    outerStart = clock();
}

double AmpcorMethods::getInnerClock() {
    return ((clock() - innerStart) / double(CLOCKS_PER_SEC));
}

double AmpcorMethods::getOuterClock() {
    return ((clock() - outerStart) / double(CLOCKS_PER_SEC));
}

void AmpcorMethods::correlate(vector<vector<float> > &refChip, vector<vector<float> > &schImg, int refChipWidth, int refChipHeight, 
                                int schWinWidth, int schWinHeight, int nLookAcross, int nLookDown, float &corrPeak,
                                vector<float> &covs, vector<vector<float> >&corrSurface, int &peakRow, int &peakCol,
                                vector<int> &isEdge, int flg, bool dbg) {

    // This routine will do amplitude correlation on two specified input files

    vector<vector<float> > refBlock(schWinWidth, vector<float>(schWinHeight,0.)), schBlock(schWinWidth, vector<float>(schWinHeight,0.));
    vector<vector<float> > schBlockSum(schWinWidth+1, vector<float>(schWinHeight+1)), schBlockSumSq(schWinWidth+1, vector<float>(schWinHeight+1));
    vector<vector<float> > corr(schWinWidth+1, vector<float>(schWinHeight+1)), normCorr(schWinWidth+1, vector<float>(schWinHeight+1));
    
    float corrSum, refSum, refSumSq, normFact, vertVar, horzVar, diagVar, u;
    float noiseSq, noiseFr, refMean, schMean, refStdDev, schStdDev;
    int refNormHeight, refNormWidth, schNormHeight, schNormWidth, refNormArea, schNormArea;
    int refCount, schCount, refMeanCount, schMeanCount;
    
    if (dbg) {
        printf("\n Debug Statements ** Inputs **\n");
        printf("refChip(1,1),schImg(1,1) = %f, %f\n", refChip[0][0], schImg[0][0]);
        printf("refChip(width,height),schImg(width,height) = %f, %f\n", refChip[refChipWidth-1][refChipHeight-1], 
                                                                        schImg[schWinWidth-1][schWinHeight-1]);
        printf("refChipWidth and refChipHeight = %d, %d\n", refChipWidth, refChipHeight);
        printf("schWinWidth and schWinHeight = %d, %d\n", schWinWidth, schWinHeight);
        printf("corrPeak = %f\n", corrPeak);
        printf("peakRow and peakCol = %d, %d\n", peakRow, peakCol);
        printf("isEdge and flg = %d, %d, %d\n", isEdge[0], isEdge[1], flg);
    }

    // Avoid "uninitialized" errors on debug printing
    refMean = 0.;
    refStdDev = 0.;
    schMean = 0.;
    schStdDev = 0.;
    noiseSq = 0.;
    //

    isEdge[0] = 0;
    isEdge[1] = 0;
    refNormHeight = refChipHeight / nLookDown;
    refNormWidth = refChipWidth / nLookAcross;
    schNormHeight = schWinHeight / nLookDown;
    schNormWidth = schWinWidth / nLookAcross;
    refNormArea = refNormHeight * refNormWidth;
    schNormArea = schNormHeight * schNormWidth;
    covs[0] = 0.;
    covs[1] = 0.;
    covs[2] = 0.;
    
    // compute mean and standard deviations on blocks
    refMeanCount = 0;
    schMeanCount = 0;
    refSum = 0.;
    refSumSq = 0.;

    fill(corrSurface.begin(), corrSurface.end(), vector<float>(corrSurface[0].size(),0.));

    for (int x=0; x<schNormWidth; x++) {
        for (int y=0; y<schNormHeight; y++) {
            refCount = 0;
            schCount = 0;
            if ((nLookAcross != 1) || (nLookDown != 1)) {
                for (int xx=(x*nLookAcross); xx<((x+1)*nLookAcross); xx++) {
                    for (int yy=(y*nLookDown); yy<((y+1)*nLookDown); yy++) {
                        if ((xx < refChipWidth) && (yy < refChipHeight)) {
                            if (refChip[xx][yy] != 0.) {
                                refCount++;
                                refBlock[x][y] = refBlock[x][y] + refChip[xx][yy];
                            }
                        }
                        if (schImg[xx][yy] != 0.) {
                            schCount++;
                            schBlock[x][y] = schBlock[x][y] + schImg[xx][yy];
                        }
                    }
                }
                if (refCount != 0) {
                    refMeanCount++;
                    refBlock[x][y] = refBlock[x][y] / refCount;
                    refSum = refSum + refBlock[x][y];
                    refSumSq = refSumSq + pow(refBlock[x][y],2);
                }
                if (schCount != 0) {
                    schBlock[x][y] = schBlock[x][y] / schCount;
                    schMeanCount++;
                }
            } else {
                schBlock[x][y] = schImg[x][y];
                if ((x < refChipWidth) && (y < refChipHeight)) {
                    refBlock[x][y] = refChip[x][y];
                    if (refBlock[x][y] != 0) {
                        refMeanCount++;
                        refSum = refSum + refBlock[x][y];
                        refSumSq = refSumSq + pow(refBlock[x][y],2);
                    }
                }
                if (schBlock[x][y] != 0) schMeanCount++;
            } // no averaging
        }
    }

    if (refMeanCount != 0) {
        refMean = refSum / refMeanCount;
        refStdDev = sqrt((refSumSq / refMeanCount) - pow(refMean,2));
    } else {
        refMean = 0.;
    }

    if ((refMeanCount >= (.9 * refNormArea)) && (schMeanCount >= (.9 * schNormArea))) { // have enough real estate
        
        //fill(schBlockSum[0].begin(), schBlockSum[0].end(), 0.);
        //fill(schBlockSum[1].begin(), schBlockSum[1].end(), 0.);
        //fill(schBlockSumSq[0].begin(), schBlockSumSq[0].end(), 0.);
        //fill(schBlockSumSq[1].begin(), schBlockSumSq[1].end(), 0.);
    
        for (int y=0; y<schNormHeight; y++) {
            schBlockSum[0][y] = 0.;
            schBlockSum[1][y] = 0.;
            schBlockSumSq[0][y] = 0.;
            schBlockSumSq[1][y] = 0.;
            for (int x=0; x<refNormWidth; x++) {
                schBlockSum[1][y] = schBlockSum[1][y] + schBlock[x][y];
                schBlockSumSq[1][y] = schBlockSumSq[1][y] + pow(schBlock[x][y],2);
            }
            for (int x=2; x<=(schNormWidth-refNormWidth+1); x++) {
                schBlockSum[x][y] = schBlockSum[x-1][y] - schBlock[x-2][y] + schBlock[x+refNormWidth-2][y];
                schBlockSumSq[x][y] = schBlockSumSq[x-1][y] - pow(schBlock[x-2][y],2) + pow(schBlock[x+refNormWidth-2][y],2);
            }
        }

        for (int x=0; x<=(schNormWidth-refNormWidth); x++) {
            schBlockSum[x][0] = 0.;
            schBlockSumSq[x][0] = 0.;
            for (int y=0; y<refNormHeight; y++) {
                schBlockSum[x][0] = schBlockSum[x][0] + schBlockSum[x+1][y];
                schBlockSumSq[x][0] = schBlockSumSq[x][0] + schBlockSumSq[x+1][y];
            }
            for (int y=1; y<=(schNormHeight-refNormHeight); y++) {
                schBlockSum[x][y] = schBlockSum[x][y-1] - schBlockSum[x+1][y-1] + schBlockSum[x+1][y+refNormHeight-1];
                schBlockSumSq[x][y] = schBlockSumSq[x][y-1] - schBlockSumSq[x+1][y-1] + schBlockSumSq[x+1][y+refNormHeight-1];
            }
        }

        peakRow = 0;
        peakCol = 0;
        corrPeak = -9.e27;

        for (int m=0; m<=(schNormWidth-refNormWidth); m++) {
            for (int n=0; n<=(schNormHeight-refNormHeight); n++) {
                corrSum = 0.;
                for (int i=0; i<refNormWidth; i++) {
                    for (int j=0; j<refNormHeight; j++) {
                        corrSum = corrSum + (refBlock[i][j] * schBlock[i+m][j+n]);
                    }
                }
                corr[m][n] = corrSum - (refMean * schBlockSum[m][n]);
                normFact = refStdDev * sqrt((schBlockSumSq[m][n] * refNormArea) - pow(schBlockSum[m][n],2));
                if (normFact > 0.) normCorr[m][n] = corr[m][n] / normFact;
                else normCorr[m][n] = 0.;
                corrSurface[m][n] = normCorr[m][n];
                if (corrPeak < normCorr[m][n]) {
                    corrPeak = normCorr[m][n];
                    peakRow = m;
                    peakCol = n;
                    //if ((m == 8) && (n == 8)) printf("%d %d %f %f %f\n", peakRow, peakCol, corrPeak, corr[m][n], normFact);
                    //printf("%d %d - %f %f\n",m,n,schBlockSum[m][n],schBlockSumSq[m][n]);
                }
                //if ((m == 16) && (n == 4)) printf("16 4 %f %f %d %f\n", refStdDev, schBlockSumSq[16][4], refNormArea, schBlockSum[16][4]);
            }
        }

        // compute the curvature of the correlation surface to estimate the goodness of the match

        if (corrPeak > 0.) {
            int x = peakRow;
            int y = peakCol;
            if ((y == 0) || (y == (schNormHeight - refNormHeight))) isEdge[0] = 1;
            if ((x == 0) || (x == (schNormWidth - refNormWidth))) isEdge[1] = 1;
            schMean = schBlockSum[x][y] / refNormArea;
            schStdDev = sqrt((schBlockSumSq[x][y] / refNormArea) - pow(schMean,2));
            flg = 0;

            if (x == 0) {
                if (y == 0) {
                    vertVar = -(normCorr[x+1][y] + normCorr[x+1][y] - (2 * normCorr[x][y])) / pow(nLookAcross,2);
                    horzVar = -(normCorr[x][y+1] + normCorr[x][y+1] - (2 * normCorr[x][y])) / pow(nLookDown,2);
                    diagVar = 0.;
                    vertVar = vertVar / 4; // added empirically
                    horzVar = horzVar / 4;
                    diagVar = diagVar / 4;
                    corrPeak = corrPeak / 4;
                } else if (y == (schNormHeight - refNormHeight)) {
                    vertVar = -(normCorr[x+1][y] + normCorr[x+1][y] - (2 * normCorr[x][y])) / pow(nLookAcross,2);
                    horzVar = -(normCorr[x][y-1] + normCorr[x][y-1] - (2 * normCorr[x][y])) / pow(nLookDown,2);
                    diagVar= 0;
                    vertVar = vertVar / 4; // added empirically
                    horzVar = horzVar / 4;
                    diagVar = diagVar / 4;
                    corrPeak = corrPeak / 4;
                } else {
                    vertVar = -(normCorr[x+1][y] + normCorr[x+1][y] - (2 * normCorr[x][y])) / pow(nLookAcross,2);
                    horzVar = -(normCorr[x][y+1] + normCorr[x][y-1] - (2 * normCorr[x][y])) / pow(nLookDown,2);
                    diagVar = (2 * (normCorr[x+1][y+1] - normCorr[x+1][y-1])) / (4 * nLookAcross * nLookDown);
                    vertVar = vertVar / 2; // added empirically
                    horzVar = horzVar / 2;
                    diagVar = diagVar / 2;
                    corrPeak = corrPeak / 2;
                }
            } else if (x == (schNormWidth - refNormWidth)) {
                if (y == 0) {
                    vertVar = -(normCorr[x-1][y] + normCorr[x-1][y] - (2 * normCorr[x][y])) / pow(nLookAcross,2);
                    horzVar = -(normCorr[x][y+1] + normCorr[x][y+1] - (2 * normCorr[x][y])) / pow(nLookDown,2);
                    diagVar = 0;
                    vertVar = vertVar / 4; // added empirically
                    horzVar = horzVar / 4;
                    diagVar = diagVar / 4;
                    corrPeak = corrPeak / 4;
                } else if (y == (schNormHeight - refNormHeight)) {
                    vertVar = -(normCorr[x-1][y] + normCorr[x-1][y] - (2 * normCorr[x][y])) / pow(nLookAcross,2);
                    horzVar = -(normCorr[x][y-1] + normCorr[x][y-1] - (2 * normCorr[x][y])) / pow(nLookDown,2);
                    diagVar = 0;
                    vertVar = vertVar / 4; // added empirically
                    horzVar = horzVar / 4;
                    diagVar = diagVar / 4;
                    corrPeak = corrPeak / 4;
                } else {
                    vertVar = -(normCorr[x-1][y] + normCorr[x-1][y] - (2 * normCorr[x][y])) / pow(nLookAcross,2);
                    horzVar = -(normCorr[x][y+1] + normCorr[x][y-1] - (2 * normCorr[x][y])) / pow(nLookDown,2);
                    diagVar = (2 * (normCorr[x-1][y-1] - normCorr[x-1][y+1])) / (4 * nLookAcross * nLookDown);
                    vertVar = vertVar / 2; // added empirically
                    horzVar = horzVar / 2;
                    diagVar = diagVar / 2;
                    corrPeak = corrPeak / 2;
                }
            } else {
                if (y == 0) {
                    vertVar = -(normCorr[x+1][y] + normCorr[x-1][y] - (2 * normCorr[x][y])) / pow(nLookAcross,2);
                    horzVar = -(normCorr[x][y+1] + normCorr[x][y+1] - (2 * normCorr[x][y])) / pow(nLookDown,2);
                    diagVar = (2 * (normCorr[x+1][y+1] - normCorr[x-1][y+1])) / (4 * nLookAcross * nLookDown);
                    vertVar = vertVar / 2; // added empirically
                    horzVar = horzVar / 2;
                    diagVar = diagVar / 2;
                    corrPeak = corrPeak / 2;
                } else if (y == (schNormHeight - refNormHeight)) {
                    vertVar = -(normCorr[x+1][y] + normCorr[x-1][y] - (2 * normCorr[x][y])) / pow(nLookAcross,2);
                    horzVar = -(normCorr[x][y-1] + normCorr[x][y-1] - (2 * normCorr[x][y])) / pow(nLookDown,2);
                    diagVar = (2 * (normCorr[x-1][y-1] - normCorr[x+1][y-1])) / (4 * nLookAcross * nLookDown);
                    vertVar = vertVar / 2; // added empirically
                    horzVar = horzVar / 2;
                    diagVar = diagVar / 2;
                    corrPeak = corrPeak / 2;
                } else {
                    vertVar = -(normCorr[x+1][y] + normCorr[x-1][y] - (2 * normCorr[x][y])) / pow(nLookAcross,2);
                    horzVar = -(normCorr[x][y+1] + normCorr[x][y-1] - (2 * normCorr[x][y])) / pow(nLookDown,2);
                    diagVar = (normCorr[x+1][y+1] + normCorr[x-1][y-1] - normCorr[x+1][y-1] - normCorr[x-1][y+1]) / (4 * nLookAcross * nLookDown);
                }
            }

            noiseSq = max(1.-corrPeak, 0.);
            vertVar = vertVar * refNormArea;
            horzVar = horzVar * refNormArea;
            diagVar = diagVar * refNormArea;
            noiseFr = pow(noiseSq, 2);
            noiseSq = noiseSq * 2;
            noiseFr = noiseFr * .5 * refNormArea;
            u = pow(diagVar,2) - (vertVar * horzVar);

            if (u == 0.) {
                covs[0] = 99.;
                covs[1] = 99.;
                covs[2] = 0.;
                flg = 1;
            } else {
                covs[0] = ((-noiseSq * u * horzVar) + (noiseFr * (pow(horzVar,2) + pow(diagVar,2)))) / pow(u,2);
                covs[1] = ((-noiseSq * u * vertVar) + (noiseFr * (pow(vertVar,2) + pow(diagVar,2)))) / pow(u,2);
                covs[2] = (((noiseSq * u) - (noiseFr * (vertVar + horzVar))) * diagVar) / pow(u,2);
            }

            if (covs[2] != 0) {

                u = sqrt(pow(covs[0] + covs[1],2) - (4. * ((covs[0] * covs[1]) - pow(covs[2],2))));
                if ((covs[0] - ((covs[0] + covs[1] + u) / 2.)) == 0.) printf("e vector 1 error\n");
                if ((covs[0] - ((covs[0] + covs[1] - u) / 2.)) == 0.) printf("e vector 2 error\n");
            }
        } else {
            flg = 1;
            printf("correlation error\n");
        }
    } else {
        flg = 1;
    }
    
    if (dbg) {
        printf("\nExit values\n");
        printf("refChipWidth and refChipHeight = %d, %d\n", refChipWidth, refChipHeight);
        printf("schWinWidth and schWinHeight = %d, %d\n", schWinWidth, schWinHeight);
        printf("refMean and refStdDev = %f, %f\n", refMean, refStdDev);
        printf("schMean and schStdDev = %f, %f\n", schMean, schStdDev);
        printf("corrPeak and noise = %f, %f\n", corrPeak, sqrt(noiseSq/2));
        printf("covs = %f %f %f\n", covs[0], covs[1], covs[2]);
        printf("isEdge and flg = %d %d, %d\n", isEdge[0], isEdge[1], flg);
    }
}

void AmpcorMethods::derampc(vector<complex<float> > &img, int height, int width) {

    // NOTE: In original Fortran code, img is 1D in main module, but reshaped to 2D, so the accessors were changed
    //       back below to 1D accessing
    complex<float> cx_phaseDown(0.,0.), cx_phaseAcross(0.,0.);
    float rl_phaseDown, rl_phaseAcross;

    for (int i=0; i<(height-1); i++) { // alt. i<=(height-1) in original code
        for (int j=0; j<width; j++) {
            cx_phaseAcross = cx_phaseAcross + (img[(i*width)+j] * conj(img[((i+1)*width)+j]));
        }
    }

    for (int i=0; i<height; i++) {
        for (int j=0; j<(width-1); j++) { // alt. j<=(width-1) in original code
            cx_phaseDown = cx_phaseDown + (img[(i*width)+j] * conj(img[(i*width)+j+1]));
        }
    }

    if (abs(cx_phaseDown) == 0) rl_phaseDown = 0.;
    else rl_phaseDown = atan2(cx_phaseDown.imag(), cx_phaseDown.real());

    if (abs(cx_phaseAcross) == 0) rl_phaseAcross = 0.;
    else rl_phaseAcross = atan2(cx_phaseAcross.imag(), cx_phaseAcross.real());

    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            img[(i*width)+j] = img[(i*width)+j] * complex<float>(cos((rl_phaseAcross * (i + 1)) + (rl_phaseDown * (j + 1))), 
                                                                 sin((rl_phaseAcross * (i + 1)) + (rl_phaseDown * (j + 1))));
        }
    }
}

void AmpcorMethods::fourn2d(vector<complex<float> > &img, vector<int> &nPoints, int fftDir) {
    
    vector<complex<float> > d(16384);

    for (int i=0; i<nPoints[1]; i++) aFFT.fft1d(nPoints[0], &img[nPoints[0]*i], -fftDir);

    for (int i=0; i<nPoints[0]; i++) {
        for (int j=0; j<nPoints[1]; j++) d[j] = img[i+(nPoints[0]*j)];

        aFFT.fft1d(nPoints[1], &d[0], -fftDir);

        for (int j=0; j<nPoints[1]; j++) {
            if (-fftDir == 1) d[j] = d[j] * float(nPoints[0]) * float(nPoints[1]);
            img[i+(nPoints[0]*j)] = d[j];
        }
    }
}

