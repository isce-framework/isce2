//
// Author: Joshua Cohen
// Copyright 2016
//

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fftw3.h>
#include <vector>
#include "AmpcorFFT.h"
#include "Constants.h"

using std::complex;
using std::vector;


// Since the original Fortran code passed addresses of different points in the full imgArr, it's easier to have this function
// take a complex* instead of a vector<complex> so that the calling functions can just pass in &arr[offset]
void AmpcorFFT::fft1d(int nPoints, complex<float> *imgArr, int fftDir) {

    if (firstTime) {
        const char *wisdomFilename = std::getenv("WISDOM_FILE");
        if (wisdomFilename == NULL) printf("WARNING: No wisdom file specified in environment. Skipping wisdom-loading...\n");
        else {
            FILE *fp = fopen(wisdomFilename, "r");
            if (fftwf_import_wisdom_from_file(fp) == 0) { // Loads wisdom file inline on success
                printf("ERROR: Cannot read specified wisdom file - %s\nStopping...\n", wisdomFilename);
                fclose(fp);
                exit(0);
            }
            fclose(fp);
        }
        firstTime = false;
    }

    double pow_of_two = log(nPoints) / log(2.);
    if ((pow_of_two != int(pow_of_two)) || (pow_of_two < 2) || (pow_of_two > 16)) {
        printf("ERROR: FFTW length of %d unsupported. Will not execute.\n", nPoints);
        return;
    }
    // Makes everything below way cleaner and easier
    int idx = int(pow_of_two);

    if (fftDir == 0) {
        if (planFlagCreate[idx] == 0) {
            // Note that the pointer to the vector is NOT the pointer to the array. Anytime you
            // need the pointer to the actual array itself, pass in &array[0] instead of array!
            // Also, from the FFTW doc, the array needs to be cast to fftw's 'fftw_complex' type
            // using the method shown below
            planf[idx] = fftwf_plan_dft_1d(nPoints, reinterpret_cast<fftwf_complex*>(&inArr[0]), reinterpret_cast<fftwf_complex*>(&inArr[0]), FFTW_FORWARD, FFTW_MEASURE);
            plani[idx] = fftwf_plan_dft_1d(nPoints, reinterpret_cast<fftwf_complex*>(&inArr[0]), reinterpret_cast<fftwf_complex*>(&inArr[0]), FFTW_BACKWARD, FFTW_MEASURE);
            planFlagCreate[idx] = 1;
        }
    }
    else if (fftDir == -1) fftwf_execute_dft(planf[idx], reinterpret_cast<fftwf_complex*>(imgArr), reinterpret_cast<fftwf_complex*>(imgArr));
    else if (fftDir == 1) fftwf_execute_dft(plani[idx], reinterpret_cast<fftwf_complex*>(imgArr), reinterpret_cast<fftwf_complex*>(imgArr));
    else if (fftDir == 2) {
        if (planFlagDestroy[idx] == 0) {
            planFlagDestroy[idx] = 1;
            planFlagCreate[idx] = 0;
            fftwf_destroy_plan(planf[idx]);
            fftwf_destroy_plan(plani[idx]);
        }
    }
    else printf("ERROR: Unspecified 'dir' flag (received '%d'), FFTW will not execute.\n", fftDir);
}

