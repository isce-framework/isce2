//
// Author: Joshua Cohen
// Copyright 2017
//

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <omp.h>
#include <pthread.h>
#include "DataAccessor.h"
#include "Constants.h"
#include "Ellipsoid.h"
#include "LinAlg.h"
#include "Orbit.h"
#include "Poly1d.h"
#include "Geo2rdr.h"
#ifdef GPU_ACC_ENABLED // Check to see if scons discovered gpu-capable system
#include "GPUgeo.h"
#endif

using std::abs;

pthread_mutex_t m; // Global mutex lock

struct writeData {
    void **accessors;
    double *rg;
    double *az;
    double *rgoff;
    double *azoff;
    bool rgFlag;
    bool azFlag;
    bool rgOffFlag;
    bool azOffFlag;
    int nLines;
    int width;
    bool firstWrite;
};

void *writeToFile(void *inputData) {
    pthread_mutex_lock(&m);
    struct writeData data;
    data.accessors = ((struct writeData *)inputData)->accessors;
    data.rg = ((struct writeData *)inputData)->rg;
    data.az = ((struct writeData *)inputData)->az;
    data.rgoff = ((struct writeData *)inputData)->rgoff;
    data.azoff = ((struct writeData *)inputData)->azoff;
    data.rgFlag = ((struct writeData *)inputData)->rgFlag;
    data.azFlag = ((struct writeData *)inputData)->azFlag;
    data.rgOffFlag = ((struct writeData *)inputData)->rgOffFlag;
    data.azOffFlag = ((struct writeData *)inputData)->azOffFlag;
    data.nLines = ((struct writeData *)inputData)->nLines;
    data.width = ((struct writeData *)inputData)->width;
    data.firstWrite = ((struct writeData *)inputData)->firstWrite;

    if (!data.firstWrite) {
        for (int i=0; i<data.nLines; i++) {
            size_t offset = i * size_t(data.width);
            if (data.rgFlag) ((DataAccessor *)data.accessors[0])->setLineSequential((char *)&data.rg[offset]);
            if (data.azFlag) ((DataAccessor *)data.accessors[1])->setLineSequential((char *)&data.az[offset]);
            if (data.rgOffFlag) ((DataAccessor *)data.accessors[2])->setLineSequential((char *)&data.rgoff[offset]);
            if (data.azOffFlag) ((DataAccessor *)data.accessors[3])->setLineSequential((char *)&data.azoff[offset]);
        }
        free(data.rg); // These free the data from the run that was just completed
        free(data.az); // Note that after each run, this function is the ONLY one that retains
        free(data.rgoff); // these pointers
        free(data.azoff);
    }
    pthread_mutex_unlock(&m);
    pthread_exit(NULL);
}

// Initializes the internal orbit stateVector memory (called from Python-level)
void Geo2rdr::createOrbit() {
    orb.setOrbit(orbit_nvecs,orbit_basis);
}

// Initializes the internal poly1d coefficients memory (called from Python-level)
void Geo2rdr::createPoly() {
    dop.setPoly(poly_order,poly_mean,poly_norm);
}

Geo2rdr::Geo2rdr() {
    usr_enable_gpu = true; // Default to enabling the GPU acceleration (GPU_ACC_ENABLED is an env var set by scons based on capability)
}

void Geo2rdr::geo2rdr() {

    double *lat, *lon, *dem, *rgm, *azt, *rgoff, *azoff;
    double xyz_mid[3], vel_mid[3], llh[3], xyz[3], satx[3], satv[3], dr[3];
    double tend, tline, tprev, rngend, rngpix, tmid, temp, dtaz, dmrg, fdop, fdopder, fnprime;

    double timer_start;

    int *distance;
    int stat, cnt, pixel, line, conv, numOutsideImage;

    bool isOutside;

    DataAccessor *latAccObj = (DataAccessor*)latAccessor;
    DataAccessor *lonAccObj = (DataAccessor*)lonAccessor;
    DataAccessor *hgtAccObj = (DataAccessor*)hgtAccessor;
    DataAccessor *azAccObj = (DataAccessor*)azAccessor;
    DataAccessor *rgAccObj = (DataAccessor*)rgAccessor;
    DataAccessor *azOffAccObj = (DataAccessor*)azOffAccessor;
    DataAccessor *rgOffAccObj = (DataAccessor*)rgOffAccessor;

    Ellipsoid elp(major, eccentricitySquared);
    LinAlg linalg;
    Poly1d fdvsrng, fddotvsrng; // Empty constructor, will be modified later

    #ifndef GPU_ACC_ENABLED // If scons didnt find a CUDA-compatible system, force-disable the GPU code
    usr_enable_gpu = false;
    #endif

    if (orbitMethod == HERMITE_METHOD) {
        if (orb.nVectors < 4) {
            printf("Error in Geo2rdr::geo2rdr - Need at least 4 state vectors for using hermite polynomial interpolation.\n");
            exit(1);
        }
    } else if (orbitMethod == SCH_METHOD) {
        if (orb.nVectors < 4) {
            printf("Error in Geo2rdr::geo2rdr - Need at least 4 state vectors for using SCH interpolation.\n");
            exit(1);
        }
    } else if (orbitMethod == LEGENDRE_METHOD) {
        if (orb.nVectors < 9) {
            printf("Error in Geo2rdr::geo2rdr - Need at least 9 state vectors for using legendre polynomial interpolation.\n");
            exit(1);
        }
    } else {
        printf("Error in Geo2rdr::geo2rdr - Undefined orbit interpolation method.\n");
        exit(1);
    }

    // OpenMP replacement for clock() (clock reports cumulative thread time, not single thread
    // time, so clock() on 4 threads would report 4 x the true runtime)
    timer_start = omp_get_wtime();
    cnt = 0;
    printf("Geo2rdr executing on %d threads...\n",  omp_get_max_threads());

    dtaz = nAzLooks / prf;
    tend = tstart + ((imgLength - 1) * dtaz);
    tmid = 0.5 * (tstart + tend);

    printf("Starting Acquisition time: %f\n", tstart);
    printf("Stop Acquisition time: %f\n", tend);
    printf("Azimuth line spacing in secs: %f\n", dtaz);

    dmrg = nRngLooks * drho;
    rngend = rngstart + ((imgWidth - 1) * dmrg);

    printf("Near Range in m: %f\n", rngstart);
    printf("Far  Range in m: %f\n", rngend);
    printf("Range sample spacing in m: %f\n", dmrg);
    printf("Radar Image Length: %d\n", imgLength);
    printf("Radar Image Width: %d\n", imgWidth);
    printf("Reading DEM...\n");
    printf("Geocoded Lines:   %d\n", demLength);
    printf("Geocoded Samples: %d\n", demWidth);

    // setPoly() resets the internal values of a Poly1d without destruct/construct
    fdvsrng.setPoly(dop.order, rngstart+(dop.mean*drho), dop.norm*drho);
    for (int i=0; i<=dop.order; i++) fdvsrng.setCoeff(i, (prf * dop.getCoeff(i)));

    if (fdvsrng.order == 0) {
        fddotvsrng.setPoly(0,0.,1.);
        fddotvsrng.setCoeff(0, 0.);
    } else {
        fddotvsrng.setPoly(fdvsrng.order-1, fdvsrng.mean, fdvsrng.norm);
        for (int i=1; i<=dop.order; i++) {
            temp = (i * fdvsrng.getCoeff(i)) / fdvsrng.norm;
            fddotvsrng.setCoeff(i-1, temp);
        }
    }

    printf("Dopplers: %f %f\n", fdvsrng.eval(rngstart), fdvsrng.eval(rngend));

    tline = tmid;
    stat = orb.interpolateOrbit(tline, xyz_mid, vel_mid, orbitMethod);

    if (stat != 0) {
        printf("Cannot interpolate orbits at the center of scene.\n");
        exit(1);
    }

    numOutsideImage = 0;
    conv = 0;

    if (usr_enable_gpu) { // GPU-enabled ; will only be true if GPU_ACC_ENABLED is defined and if the user doesn't disable this flag
        #ifdef GPU_ACC_ENABLED // Doesn't compile the GPU code if scons didnt find CUDA-compatible libraries, etc
        double gpu_inputs_d[9];
        int gpu_inputs_i[3];

        gpu_inputs_i[0] = demLength;
        gpu_inputs_i[1] = demWidth;
        gpu_inputs_i[2] = int(bistatic);

        gpu_inputs_d[0] = major;
        gpu_inputs_d[1] = eccentricitySquared;
        gpu_inputs_d[2] = tstart;
        gpu_inputs_d[3] = tend;
        gpu_inputs_d[4] = wvl;
        gpu_inputs_d[5] = rngstart;
        gpu_inputs_d[6] = rngend;
        gpu_inputs_d[7] = dmrg;
        gpu_inputs_d[8] = dtaz;

        printf("\nCopying Orbit and Poly1d data to compatible arrays...\n");

        int gpu_orbNvec = orb.nVectors;
        double *gpu_orbSvs = new double[7*gpu_orbNvec];
        for (int i=0; i<gpu_orbNvec; i++) {
            gpu_orbSvs[(7*i)] = orb.UTCtime[i];
            gpu_orbSvs[(7*i)+1] = orb.position[(3*i)];
            gpu_orbSvs[(7*i)+2] = orb.position[(3*i)+1];
            gpu_orbSvs[(7*i)+3] = orb.position[(3*i)+2];
            gpu_orbSvs[(7*i)+4] = orb.velocity[(3*i)];
            gpu_orbSvs[(7*i)+5] = orb.velocity[(3*i)+1];
            gpu_orbSvs[(7*i)+6] = orb.velocity[(3*i)+2];
        }

        int gpu_polyOrd = dop.order;
        double gpu_polyMean = rngstart + (dop.mean * drho); // This is what fdvsrng gets as its mean, so it's easier to pass this than extra vars
        double gpu_polyNorm = dop.norm * drho; // As above
        double *gpu_polyCoef = new double[gpu_polyOrd+1];
        for (int i=0; i<=gpu_polyOrd; i++) gpu_polyCoef[i] = dop.coeffs[i];

        printf("Calculating relevant GPU parameters...\n");

        double *outputArrays[4]; // These are passed into CUDA
        double *writeArrays[4]; // These are passed into the async write-to-file
        DataAccessor *accObjs[4] = {rgAccObj, azAccObj, rgOffAccObj, azOffAccObj};
        bool rgFlag = bool(rgAccessor > 0);
        bool azFlag = bool(azAccessor > 0);
        bool rgOffFlag = bool(rgOffAccessor > 0);
        bool azOffFlag = bool(azOffAccessor > 0);

        // Create pthread data and initialize dummy thread
        pthread_t writeThread;
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
        void *thread_stat;
        struct writeData wd;
        wd.accessors = (void**)accObjs;
        wd.rg = outputArrays[0]; // Don't contain data/valid pointers yet
        wd.az = outputArrays[1];
        wd.rgoff = outputArrays[2];
        wd.azoff = outputArrays[3];
        wd.rgFlag = rgFlag;
        wd.azFlag = azFlag;
        wd.rgOffFlag = rgOffFlag;
        wd.azOffFlag = azOffFlag;
        wd.nLines = 0;
        wd.width = demWidth;
        wd.firstWrite = true; // Flag to ignore write instructions
        pthread_create(&writeThread, &attr, writeToFile, (void*)&wd); // Fires empty thread

        size_t totalPixels = demLength * demWidth;
        // adjust the lines per run by the available gpu memory
        int linesPerRun = std::min(demLength, nLinesPossible(demLength, demWidth));
        // ! To best parallelize the computation, use the max available gpu memory is the best option
        // ! the following adjustment is not needed
        // adjust further by the max pixels per run, prefavorbly as a user configurable parameter
        // temp set as 2^20
        // size_t maxPixPerRun = 1 << 20;
        // size_t pixPerRun = std::min((size_t)linesPerRun*demWidth, maxPixPerRun);
        // linesPerRun = pixPerRun/demWidth *demWidth;

        // recalculate run info
        size_t pixPerRun = linesPerRun * demWidth;
        int nRuns = demLength / linesPerRun;
        int remPix = totalPixels - (nRuns * pixPerRun);
        int remLines = remPix / demWidth;

        printf("NOTE: GPU will process image in %d runs of %d lines", nRuns, linesPerRun);
        if (remPix > 0) printf(" (with %d lines in a final partial block)", remLines);
        printf("\n");

        lat = new double[pixPerRun];
        lon = new double[pixPerRun];
        dem = new double[pixPerRun];
        size_t nb_pixels = pixPerRun * sizeof(double);

        printf("\n\n  ------------------ INITIALIZING GPU GEO2RDR ------------------\n\n");

        for (int i=0; i<nRuns; i++) {
            printf("    Loading relevant geometry product data...\n");
            for (int j=0; j<linesPerRun; j++) latAccObj->getLineSequential((char *)(lat+(j*demWidth))); // Yay pointer magic
            for (int j=0; j<linesPerRun; j++) lonAccObj->getLineSequential((char *)(lon+(j*demWidth)));
            for (int j=0; j<linesPerRun; j++) hgtAccObj->getLineSequential((char *)(dem+(j*demWidth)));

            outputArrays[0] = (double *)malloc(nb_pixels); // h_rg
            outputArrays[1] = (double *)malloc(nb_pixels); // h_az
            outputArrays[2] = (double *)malloc(nb_pixels); // h_rgoff
            outputArrays[3] = (double *)malloc(nb_pixels); // h_azoff

            runGPUGeo(i, pixPerRun, gpu_inputs_d, gpu_inputs_i, lat, lon, dem,
                        gpu_orbNvec, gpu_orbSvs, gpu_polyOrd, gpu_polyMean, gpu_polyNorm,
                        gpu_polyCoef, prf, outputArrays);
            for (int j=0; j<4; j++) writeArrays[j] = outputArrays[j]; // Copying pointers
            if (i != 0) printf("  Waiting for previous asynchronous write-out to finish...\n");
            pthread_attr_destroy(&attr);
            pthread_join(writeThread, &thread_stat); // Waits for async thread to finish

            printf("  Writing run %d out asynchronously to image files...\n", i);
            wd.accessors = (void**)accObjs;
            wd.rg = writeArrays[0];
            wd.az = writeArrays[1];
            wd.rgoff = writeArrays[2];
            wd.azoff = writeArrays[3];
            wd.rgFlag = rgFlag;
            wd.azFlag = azFlag;
            wd.rgOffFlag = rgOffFlag;
            wd.azOffFlag = azOffFlag;
            wd.nLines = linesPerRun;
            wd.width = demWidth;
            wd.firstWrite = false;
            pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
            pthread_create(&writeThread, &attr, writeToFile, (void*)&wd); // Set up and fire async write thread
        }

        if (remPix > 0) { // If we have a final partial run
            nb_pixels = remPix * sizeof(double);
            outputArrays[0] = (double *)malloc(nb_pixels);
            outputArrays[1] = (double *)malloc(nb_pixels);
            outputArrays[2] = (double *)malloc(nb_pixels);
            outputArrays[3] = (double *)malloc(nb_pixels);

            printf("    Loading relevant geometry product data...\n");
            for (int i=0; i<remLines; i++) {
                latAccObj->getLineSequential((char *)(lat+(i*demWidth)));
                lonAccObj->getLineSequential((char *)(lon+(i*demWidth)));
                hgtAccObj->getLineSequential((char *)(dem+(i*demWidth)));
            }

            for (int i=0; i<4; i++) writeArrays[i] = outputArrays[i];
            runGPUGeo((-1*linesPerRun*nRuns), remPix, gpu_inputs_d, gpu_inputs_i, lat, lon, dem,
                        gpu_orbNvec, gpu_orbSvs, gpu_polyOrd, gpu_polyMean, gpu_polyNorm,
                        gpu_polyCoef, prf, outputArrays); // Iter now stores number of lines processed
            printf("  Waiting for previous asynchronous write-out to finish...\n");
            pthread_attr_destroy(&attr);
            pthread_join(writeThread, &thread_stat);

            printf("  Writing remaining %d lines out asynchronously to image files...\n", remLines);
            wd.accessors = (void**)accObjs;
            wd.rg = writeArrays[0];
            wd.az = writeArrays[1];
            wd.rgoff = writeArrays[2];
            wd.azoff = writeArrays[3];
            wd.rgFlag = rgFlag;
            wd.azFlag = azFlag;
            wd.rgOffFlag = rgOffFlag;
            wd.azOffFlag = azOffFlag;
            wd.nLines = remLines;
            wd.width = demWidth;
            wd.firstWrite = false;
            pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
            pthread_create(&writeThread, &attr, writeToFile, (void*)&wd);
        }
        pthread_attr_destroy(&attr);
        pthread_join(writeThread, &thread_stat);
        printf("  Finished writing to files!\n");

        printf("\n  ------------------ EXITING GPU GEO2RDR ------------------\n\n");
        printf("Finished!\n");
        printf("Elapsed time = %f seconds\n", (omp_get_wtime()-timer_start));

        delete[] lat;
        delete[] lon;
        delete[] dem;
        delete[] gpu_orbSvs;
        delete[] gpu_polyCoef;
        #endif
    } else { // Standard code
        lat = new double[demWidth];
        lon = new double[demWidth];
        dem = new double[demWidth];
        rgm = new double[demWidth];
        azt = new double[demWidth];
        rgoff = new double[demWidth];
        azoff = new double[demWidth];
        distance = new int[demWidth];
        for (line=0; line<demLength; line++) {
            pixel = latAccObj->getLineSequential((char *)lat);
            pixel = lonAccObj->getLineSequential((char *)lon);
            pixel = hgtAccObj->getLineSequential((char *)dem);

            if ((line%1000) == 0) printf("Processing line: %d %d\n", line, numOutsideImage);

            #pragma omp parallel for private(pixel, rngpix, tline, tprev, stat, fnprime, fdop, \
                                             fdopder, isOutside, xyz, llh, satx, satv, dr) \
                                     reduction(+:numOutsideImage,conv,cnt)
            for (pixel=0; pixel<demWidth; pixel++) {

                isOutside = false; // Flag to determine if point is outside image

                llh[0] = lat[pixel] * (M_PI / 180.);
                llh[1] = lon[pixel] * (M_PI / 180.);
                llh[2] = dem[pixel];

                elp.latlon(xyz,llh,LLH_2_XYZ);

                tline = tmid;
                for (int i=0; i<3; i++) {
                    satx[i] = xyz_mid[i];
                    satv[i] = vel_mid[i];
                }

                // Actual iterations
                for (int loop=0; loop<51; loop++) {
                    tprev = tline;

                    for (int i=0; i<3; i++) dr[i] = xyz[i] - satx[i];
                    rngpix = linalg.norm(dr);

                    fdop = .5 * wvl * fdvsrng.eval(rngpix);
                    fdopder = .5 * wvl * fddotvsrng.eval(rngpix);
                    fnprime = (((fdop / rngpix) + fdopder) * linalg.dot(dr,satv)) - linalg.dot(satv,satv);
                    tline = tline - ((linalg.dot(dr,satv) - (fdop * rngpix)) / fnprime);
                    stat = orb.interpolateOrbit(tline, satx, satv, orbitMethod);

                    if (stat != 0) {
                        tline = BAD_VALUE;
                        rngpix = BAD_VALUE;
                        break; // Interpolation determined bad point
                    }
                    if (abs(tline - tprev) < 5.e-9) {
                        conv = conv + 1;
                        break; // Point converged
                    }
                }

                if ((tline < tstart) || (tline > tend)) isOutside = true;

                for (int i=0; i<3; i++) dr[i] = xyz[i] - satx[i];
                rngpix = linalg.norm(dr);

                if ((rngpix < rngstart) || (rngpix > rngend)) isOutside = true;

                if (bistatic) { // Not an available feature yet...
                    tline = tline + ((2. * rngpix) / SPEED_OF_LIGHT);

                    if ((tline < tstart) || (tline > tend)) isOutside = true;

                    stat = orb.interpolateOrbit(tline, satx, satv, orbitMethod);

                    if (stat != 0) isOutside = true;

                    for (int i=0; i<3; i++) dr[i] = xyz[i] - satx[i];
                    rngpix = linalg.norm(dr);

                    if ((rngpix < rngstart) || (rngpix > rngend)) isOutside = true;
                }

               if (!isOutside) { // Found a valid point inside the image
                    cnt = cnt + 1;
                    rgm[pixel] = rngpix;
                    azt[pixel] = tline;
                    rgoff[pixel] = ((rngpix - rngstart) / dmrg) - double(pixel);
                    azoff[pixel] = ((tline - tstart) / dtaz) - double(line);
                    distance[pixel] = tline - tprev;
                } else { // Point is outside the image
                    numOutsideImage = numOutsideImage + 1;
                    rgm[pixel] = BAD_VALUE; // This either-or is better here than filling the
                    azt[pixel] = BAD_VALUE; // whole array first
                    rgoff[pixel] = BAD_VALUE;
                    azoff[pixel] = BAD_VALUE;
                    distance[pixel] = BAD_VALUE;
                }
            } // end omp parallel for

            if (azAccessor > 0) azAccObj->setLineSequential((char*)azt);
            if (rgAccessor > 0) rgAccObj->setLineSequential((char*)rgm);
            if (azOffAccessor > 0) azOffAccObj->setLineSequential((char*)azoff);
            if (rgOffAccessor > 0) rgOffAccObj->setLineSequential((char*)rgoff);
        }

        printf("Number of pixels outside the image: %d\n", numOutsideImage);
        printf("Number of pixels with valid data:   %d\n", cnt);
        printf("Number of pixels that converged:    %d\n", conv);

        // Yay memory management!
        delete[] dem;
        delete[] lat;
        delete[] lon;
        delete[] rgm;
        delete[] azt;
        delete[] rgoff;
        delete[] azoff;
        delete[] distance;

        printf("Elapsed time = %f seconds\n", (omp_get_wtime()-timer_start));
    }
}
