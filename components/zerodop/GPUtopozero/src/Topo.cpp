//
// Author: Joshua Cohen
// Copyright 2016
//
// This code is adapted from the original Fortran topozero.f90 code. All relevant or associated
// structs/methods are contained in this same src/ folder as well (all adapted from the old
// Fortran code). The code was validated in full against the original Fortran code with a
// COSMO SkyMed test set and produced the same exact outputs.
//
// Note: There are a few blocks of code commented out currently (including some variables). These
//       sections calculate things that will be used in future SWOT processing, but to mildly
//       reduce runtime and some overhead they will stay commented out until use.
//
// Note 2: Most include statements in these source files are relatively-pathed. For the most part
//         the files are in a standard main/src/ - main/include/ format. The only exception is
//         the DataAccessor.h header. Please note that moving files around in this structure
//         must be reflected by the header paths (this *will* change before full release to be
//         built and linked authomatically without needing the pathing).

// update: updated to use long for some integers associated with file size to support large images.
//         Cunren Liang, 26-MAR-2018

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <future>
#include <omp.h>
#include <pthread.h>
#include <vector>
#include "DataAccessor.h"
#include "Constants.h"
#include "Ellipsoid.h"
#include "LinAlg.h"
#include "Peg.h"
#include "PegTrans.h"
#include "TopoMethods.h"
#include "Topo.h"
#include "gpuTopo.h"
using std::abs;
using std::vector;

#ifdef GPU_ACC_ENABLED
    #define RUN_GPU_TOPO 1
#else
    #define RUN_GPU_TOPO 0
#endif

pthread_mutex_t m;

struct writeData {
    void **accessors;
    //double **imgArrs;
    double *lat;
    double *lon;
    double *z;
    double *inc;
    double *los;
    bool incFlag;
    bool losFlag;
    int nLines;
    int width;
    bool firstWrite;
};

void *writeToFile(void *inputData) {
    pthread_mutex_lock(&m);
    struct writeData data;
    data.accessors = ((struct writeData *)inputData)->accessors;
    //data.imgArrs = ((struct writeData *)inputData)->imgArrs;
    data.lat = ((struct writeData *)inputData)->lat;
    data.lon = ((struct writeData *)inputData)->lon;
    data.z = ((struct writeData *)inputData)->z;
    data.inc = ((struct writeData *)inputData)->inc;
    data.los = ((struct writeData *)inputData)->los;
    data.incFlag = ((struct writeData *)inputData)->incFlag;
    data.losFlag = ((struct writeData *)inputData)->losFlag;
    data.nLines = ((struct writeData *)inputData)->nLines;
    data.width = ((struct writeData *)inputData)->width;
    data.firstWrite = ((struct writeData *)inputData)->firstWrite;

    if (!data.firstWrite) {
        for (int i=0; i<data.nLines; i++) {
            size_t offset = i * size_t(data.width);
            ((DataAccessor *)data.accessors[0])->setLineSequential((char *)&data.lat[offset]);
            ((DataAccessor *)data.accessors[1])->setLineSequential((char *)&data.lon[offset]);
            ((DataAccessor *)data.accessors[2])->setLineSequential((char *)&data.z[offset]);
            if (data.incFlag) ((DataAccessor *)data.accessors[3])->setLineSequential((char *)&data.inc[2*offset]);
            if (data.losFlag) ((DataAccessor *)data.accessors[4])->setLineSequential((char *)&data.los[2*offset]);
        }
        free(data.lat);
        free(data.lon);
        free(data.z);
        free(data.inc);
        free(data.los);
    }
    pthread_mutex_unlock(&m);
    pthread_exit(NULL);
}

void Topo::createOrbit() {
    // Assumes that the state vars orbit_nvecs/orbit_basis have been set
    orb.setOrbit(orbit_nvecs,orbit_basis);
}
/*
void Topo::writeToFile(void **accessors, double **imgArrs, bool incFlag, bool losFlag, int nLines, int width, bool firstWrite) {
    if (!firstWrite) {
        for (int i=0; i<nLines; i++) {
            printf("  Writing line %d of %d...\r", i+1, nLines);
            size_t offset = i * size_t(width);
            ((DataAccessor *)accessors[0])->setLineSequential((char *)&imgArrs[0][offset]);
            ((DataAccessor *)accessors[1])->setLineSequential((char *)&imgArrs[1][offset]);
            ((DataAccessor *)accessors[2])->setLineSequential((char *)&imgArrs[2][offset]);
            if (incFlag) ((DataAccessor *)accessors[3])->setLineSequential((char *)&imgArrs[3][2*offset]);
            if (losFlag) ((DataAccessor *)accessors[4])->setLineSequential((char *)&imgArrs[4][2*offset]);
        }
        printf("  Finished writing %d lines.\n  Freeing memory...\n", nLines);
        free(imgArrs[0]);
        free(imgArrs[1]);
        free(imgArrs[2]);
        free(imgArrs[3]);
        free(imgArrs[4]);
        printf("  Done.\n");
    }
}
*/
void Topo::topo() {
    vector<vector<double> > enumat(3,vector<double>(3)), xyz2enu(3,vector<double>(3));

    vector<double> sch(3), xyz(3), llh(3), delta(3), llh_prev(3), xyz_prev(3);
    vector<double> xyzsat(3), velsat(3), llhsat(3), enu(3), that(3), chat(3);
    vector<double> nhat(3), vhat(3), hgts(2);

    double ctrackmin,ctrackmax,dctrack,tline,rng,dopfact;
    double height,rcurv,vmag,aa,bb; //,hnadir;
    double beta,alpha,gamm,costheta,sintheta,cosalpha;
    double fraclat,fraclon,enunorm;
    // Vars for cropped DEM
    double umin_lon,umax_lon,umin_lat,umax_lat,ufirstlat,ufirstlon;
    double min_lat, min_lon, max_lat, max_lon;

    float demlat,demlon,demmax;

    int stat,totalconv,owidth,pixel,i_type,idemlat,idemlon; //,nearrangeflag;
    // Vars for cropped DEM
    int udemwidth,udemlength,ustartx,uendx,ustarty,uendy;

    // Data accessor objects
    DataAccessor *demAccObj = (DataAccessor*)demAccessor;
    DataAccessor *dopAccObj = (DataAccessor*)dopAccessor;
    DataAccessor *slrngAccObj = (DataAccessor*)slrngAccessor;
    DataAccessor *latAccObj = (DataAccessor*)latAccessor;
    DataAccessor *lonAccObj = (DataAccessor*)lonAccessor;
    DataAccessor *heightAccObj = (DataAccessor*)heightAccessor;
    DataAccessor *losAccObj = (DataAccessor*)losAccessor;
    DataAccessor *incAccObj = (DataAccessor*)incAccessor;
    DataAccessor *maskAccObj = (DataAccessor*)maskAccessor;
    // Local geometry-type objects
    Ellipsoid elp;
    Peg peg;
    PegTrans ptm;
    TopoMethods tzMethods;
    LinAlg linalg;

    // Set up DEM interpolation method
    if ((dem_method != SINC_METHOD) && (dem_method != BILINEAR_METHOD) &&
            (dem_method != BICUBIC_METHOD) && (dem_method != NEAREST_METHOD) &&
            (dem_method != AKIMA_METHOD) && (dem_method != BIQUINTIC_METHOD)) {
        printf("Error in Topo::topo - Undefined interpolation method.\n");
        exit(1);
    }
    tzMethods.prepareMethods(dem_method);

    // Set up Ellipsoid object
    elp.a = major;
    elp.e2 = eccentricitySquared;

    // Set up orbit interpolation method
    if (orbit_method == HERMITE_METHOD) {
        if (orb.nVectors < 4) {
            printf("Error in Topo::topo - Need at least 4 state vectors for using hermite polynomial interpolation.\n");
            exit(1);
        }
    } else if (orbit_method == SCH_METHOD) {
        if (orb.nVectors < 4) {
            printf("Error in Topo::topo - Need at least 4 state vectors for using SCH interpolation.\n");
            exit(1);
        }
    } else if (orbit_method == LEGENDRE_METHOD) {
        if (orb.nVectors < 9) {
            printf("Error in Topo::topo - Need at least 9 state vectors for using legendre polynomial interpolation.\n");
            exit(1);
        }
    } else {
        printf("Error in Topo::topo - Undefined orbit interpolation method.\n");
        exit(1);
    }

    owidth = (2 * width) + 1;
    totalconv = 0;
    height = 0.0;
    min_lat = 10000.0;
    max_lat = -10000.0;
    min_lon = 10000.0;
    max_lon = -10000.0;

    printf("Max threads used: %d\n", omp_get_max_threads());
    if ((slrngAccessor == 0) && (r0 == 0.0)) {
        printf("Error in Topo::topo - Both the slant range accessor and starting range are zero.\n");
        exit(1);
    }

    vector<double> lat(width), lon(width), z(width), zsch(width), rho(width), dopline(width), converge(width);
    vector<float> distance(width), losang(2*width), incang(2*width);
    vector<double> omask(0), orng(0), ctrack(0), oview(0), mask(0); // Initialize (so no scoping errors), resize only if needed
    if (maskAccessor > 0) {
        omask.resize(owidth);
        orng.resize(owidth);
        ctrack.resize(owidth);
        oview.resize(owidth);
        mask.resize(width);
    }

    //nearrangeflag = 0;
    hgts[0] = MIN_H;
    hgts[1] = MAX_H;

    // Few extra steps to let std::vector interface with getLine
    double *raw_line = new double[width];
    dopAccObj->getLine((char *)raw_line,0);
    dopline.assign(raw_line,(raw_line+width));
    slrngAccObj->getLine((char *)raw_line,0);
    rho.assign(raw_line,(raw_line+width));
    delete[] raw_line; // Manage data VERY carefully!

    // First line
    for (int line=0; line<2; line++) {
        tline = t0 + (line * Nazlooks * ((length - 1.0) / prf));
        stat = orb.interpolateOrbit(tline,xyzsat,velsat,orbit_method);
        if (stat != 0) {
            printf("Error in Topo::topo - Error getting state vector for bounds computation.\n");
            exit(1);
        }
        vmag = linalg.norm(velsat);
        linalg.unitvec(velsat,vhat);
        elp.latlon(xyzsat,llhsat,XYZ_2_LLH);
        height = llhsat[2];
        elp.tcnbasis(xyzsat,velsat,that,chat,nhat);

        peg.lat = llhsat[0];
        peg.lon = llhsat[1];
        peg.hdg = peghdg;
        ptm.radar_to_xyz(elp,peg);
        rcurv = ptm.radcur;

        for (int ind=0; ind<2; ind++) {
            pixel = ind * (width - 1);
            rng = rho[pixel];
            dopfact = (0.5 * wvl * (dopline[pixel] / vmag)) * rng;

            for (int iter=0; iter<2; iter++) {
                // SWOT-specific near range check
                // If slant range vector doesn't hit ground, pick nadir point
                if (rng <= (llhsat[2] - hgts[iter] + 1.0)) {
                    for (int idx=0; idx<3; idx++) llh[idx] = llhsat[idx];
                    //printf("Possible near nadir imaging.\n");
                    //nearrangeflag = 1;
                } else {
                    zsch[pixel] = hgts[iter];
                    aa = height + rcurv;
                    bb = rcurv + zsch[pixel];
                    costheta = 0.5 * ((aa / rng) + (rng / aa) - ((bb / aa) * (bb / rng)));
                    sintheta = sqrt(1.0 - (costheta * costheta));
                    gamm = costheta * rng;
                    alpha = (dopfact - (gamm * linalg.dot(nhat,vhat))) / linalg.dot(vhat,that);
                    beta = -ilrl * sqrt((rng * rng * sintheta * sintheta) - (alpha * alpha));
                    for (int idx=0; idx<3; idx++) delta[idx] = (gamm * nhat[idx]) + (alpha * that[idx]) + (beta *  chat[idx]);
                    for (int idx=0; idx<3; idx++) xyz[idx] = xyzsat[idx] + delta[idx];
                    elp.latlon(xyz,llh,XYZ_2_LLH);
                }
                min_lat = min(min_lat, (llh[0]*(180./M_PI)));
                max_lat = max(max_lat, (llh[0]*(180./M_PI)));
                min_lon = min(min_lon, (llh[1]*(180./M_PI)));
                max_lon = max(max_lon, (llh[1]*(180./M_PI)));
            }
        }
    }

    // Account for margins
    min_lon = min_lon - MARGIN;
    max_lon = max_lon + MARGIN;
    min_lat = min_lat - MARGIN;
    max_lat = max_lat + MARGIN;

    printf("DEM parameters:\n");
    printf("Dimensions: %d %d\n", idemwidth, idemlength);
    printf("Top Left: %g %g\n", firstlon, firstlat);
    printf("Spacing: %g %g\n", deltalon, deltalat);
    printf("Lon: %g %g\n", firstlon, (firstlon+(idemwidth-1)*deltalon));
    printf("Lat: %g %g\n\n", (firstlat+((idemlength-1)*deltalat)), firstlat);
    printf("Estimated DEM bounds needed for global height range:\n");
    printf("Lon: %g %g\n", min_lon, max_lon);
    printf("Lat: %g %g\n", min_lat, max_lat);

    // Compare with what has been provided as input
    umin_lon = max(min_lon, firstlon);
    umax_lon = min(max_lon, (firstlon+((idemwidth-1)*deltalon)));
    umax_lat = min(max_lat, firstlat);
    umin_lat = max(min_lat, (firstlat+((idemlength-1)*deltalat)));
    if (min_lon < firstlon)
        printf("Warning: West limit may be insufficient for global height range.\n");
    if (max_lon > (firstlon+((idemwidth-1)*deltalon)))
        printf("Warning: East limit may be insufficient for global height range.\n");
    if (max_lat > firstlat)
        printf("Warning: North limit may be insufficient for global height range.\n");
    if (min_lat < (firstlat+((idemlength-1)*deltalat)))
        printf("Warning: South limit may be insufficient for global height range.\n");

    // Usable part of the DEM limits
    ustartx = int((umin_lon - firstlon) / deltalon);
    uendx = int(((umax_lon - firstlon) / deltalon) + 0.5);
    ustarty = int((umax_lat - firstlat) / deltalat);
    uendy = int(((umin_lat - firstlat) / deltalat) + 0.5);
    if (ustartx < 1) ustartx = 1;
    if (uendx > idemwidth) uendx = idemwidth;
    if (ustarty < 1) ustarty = 1;
    if (uendy > idemlength) ustarty = idemlength;

    ufirstlon = firstlon + (deltalon * (ustartx));
    ufirstlat = firstlat + (deltalat * (ustarty));
    udemwidth = uendx - ustartx + 1;
    udemlength = uendy - ustarty + 1;

    printf("\nActual DEM bounds used:\n");
    printf("Dimensions: %d %d\n", udemwidth, udemlength);
    printf("Top Left: %g %g\n", ufirstlon, ufirstlat);
    printf("Spacing: %g %g\n", deltalon, deltalat);
    printf("Lon: %g %g\n", ufirstlon, (ufirstlon+(deltalon*(udemwidth-1))));
    printf("Lat: %g %g\n", (ufirstlat+(deltalat*(udemlength-1))), ufirstlat);
    printf("Lines: %d %d\n", ustarty, uendy);
    printf("Pixels: %d %d\n", ustartx, uendx);

    vector<vector<float> > dem(udemwidth,vector<float>(udemlength));
    vector<float> demline(idemwidth);
    float *raw_line_f = new float[idemwidth];
    // Safest way to copy in the DEM using the same std::vector-getLine interface
    // Read the useful part of the DEM
    for (int j=0; j<udemlength; j++) {
        demAccObj->getLine((char *)raw_line_f,(j + ustarty));
        demline.assign(raw_line_f,(raw_line_f + idemwidth));
        for (int ii=0; ii<udemwidth; ii++) dem[ii][j] = demline[(ustartx+ii)];
    }
    delete[] raw_line_f;

    //demmax = maxval(dem);
    //Note this is an O(N) operation...not efficient at all, but there's no easy equivalent...
    //where's spaghetti sort when you need it?
    demmax = -10000.0;
    for (int i=0; i<udemwidth; i++) {
        for (int j=0; j<udemlength; j++) {
            if (dem[i][j] > demmax) demmax = dem[i][j];
        }
    }
    printf("Max DEM height: %g\n", demmax);
    printf("Primary iterations: %d\n", numiter);
    printf("Secondary iterations: %d\n", extraiter);
    printf("Distance threshold: %g\n", thresh);

    height = 0.0;
    min_lat = 10000.0;
    max_lat = -10000.0;
    min_lon = 10000.0;
    max_lon = -10000.0;

    raw_line = new double[width];

    if (RUN_GPU_TOPO) {
        double gpu_inputs_d[14];
        int gpu_inputs_i[7];

        gpu_inputs_d[0] = t0;
        gpu_inputs_d[1] = prf;
        gpu_inputs_d[2] = elp.a;
        gpu_inputs_d[3] = elp.e2;
        gpu_inputs_d[4] = peg.lat;
        gpu_inputs_d[5] = peg.lon;
        gpu_inputs_d[6] = peg.hdg;
        gpu_inputs_d[7] = ufirstlat;
        gpu_inputs_d[8] = ufirstlon;
        gpu_inputs_d[9] = deltalat;
        gpu_inputs_d[10] = deltalon;
        gpu_inputs_d[11] = wvl;
        gpu_inputs_d[12] = ilrl;
        gpu_inputs_d[13] = thresh;

        gpu_inputs_i[0] = Nazlooks;
        gpu_inputs_i[1] = width;
        gpu_inputs_i[2] = udemlength;
        gpu_inputs_i[3] = udemwidth;
        gpu_inputs_i[4] = numiter;
        gpu_inputs_i[5] = extraiter;
        gpu_inputs_i[6] = length;

        printf("\n\nCopying Orbit and DEM data to compatible arrays...\n");

        float *gpu_dem = new float[size_t(udemlength)*udemwidth];
        for (int i=0; i<udemwidth; i++) {
            for (int j=0; j<udemlength; j++) {
                 gpu_dem[(i*udemlength)+j] = dem[i][j];
            }
        }

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

        printf("Calculating relevant GPU parameters...\n");

        double *outputArrays[5];
        double *writeArrays[5];

        // Set up for asynchronous writing-to-file
        DataAccessor *accObjs[5] = {latAccObj, lonAccObj, heightAccObj, incAccObj, losAccObj};
        bool incFlag = bool(incAccessor > 0);
        bool losFlag = bool(losAccessor > 0);
        //std::future<void> result = std::async(std::launch::async, &Topo::writeToFile, this, (void **)accObjs, outputArrays, incFlag, losFlag, 0, width, true);

        // Create pthread data and initialize dummy thread
        pthread_t writeThread;
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
        void *thread_stat;
        struct writeData wd;
        wd.accessors = (void**)accObjs;
        //wd.imgArrs = outputArrays;
        wd.lat = outputArrays[0];
        wd.lon = outputArrays[1];
        wd.z = outputArrays[2];
        wd.inc = outputArrays[3];
        wd.los = outputArrays[4];
        wd.incFlag = incFlag;
        wd.losFlag = losFlag;
        wd.nLines = 0;
        wd.width = width;
        wd.firstWrite = true;
        pthread_create(&writeThread, &attr, writeToFile, (void*)&wd);

        // Calculate number of and size of blocks

        // free GPU memory available
        size_t num_GPU_bytes = getDeviceFreeMem();
        // use 100Mb as a rounding unit , may be adjusted
        size_t memoryRoundingUnit = 1024ULL * 1024ULL * 100;
        // memory to be used for each pixel in bytes, with 9 double elements per pixel
        size_t pixelBytes = sizeof(double) * 10;
        // memory overhead for other shared parameters, in terms of memoryRoundUnit, or 200M
        size_t memoryOverhead = 2;

        // adjust the available free memory by rounding down
        num_GPU_bytes = (num_GPU_bytes/memoryRoundingUnit - memoryOverhead) * memoryRoundingUnit;

        // calculate the max pixels allowed in a batch (block)
        size_t pixPerImg = num_GPU_bytes / pixelBytes;
        assert(pixPerImg > 0);

        // ! To best parallelize the computation, use the max available gpu memory is the best option
        // ! the following adjustment is not needed
        // set a upper limit on the size of the block
        // preferably offered as an input parameter
        // 2^24 is about 1.2G Memory
        // size_t maxPixPerImg = 1 << 24;
        // pixPerImg = std::min(pixPerImg, maxPixPerImg);

        // the max lines in a batch, and will be used for each run
        int linesPerImg = pixPerImg / width;
        assert(linesPerImg >0);
        // now reassign the value for pixels in a batch
        pixPerImg = linesPerImg * width;

        // total number of pixels in SLC
        size_t totalPixels = (size_t)length * width;

        // total of blocks needed to process the whole image
        int nBlocks = length / linesPerImg;

        // check whether there are remnant lines
        int remLines = length - nBlocks*linesPerImg;
        size_t remPix = remLines * width;

        printf("NOTE: GPU will process image in %d blocks of %d lines", nBlocks, linesPerImg);
        if (remPix > 0) printf(" (with %d lines in a final partial block)", remLines);
        printf("\n");

        double *gpu_rho = new double[linesPerImg * width];
        double *gpu_dopline = new double[linesPerImg * width];
        size_t nb_pixels = pixPerImg * sizeof(double);

        printf("\n\n  ------------------ INITIALIZING GPU TOPO ------------------\n\n");

        // Call GPU kernel on blocks
        for (int i=0; i<nBlocks; i++) { // Iterate over full blocks
            printf("    Loading slantrange and doppler data...\n");
            for (int j=0; j<linesPerImg; j++) {
                slrngAccObj->getLineSequential((char *)raw_line);
                for (int k=0; k<width; k++) gpu_rho[(j*width)+k] = raw_line[k];
                dopAccObj->getLineSequential((char *)raw_line);
                for (int k=0; k<width; k++) gpu_dopline[(j*width)+k] = raw_line[k];
            }

            outputArrays[0] = (double *)malloc(nb_pixels); // h_lat
            outputArrays[1] = (double *)malloc(nb_pixels); // h_lon
            outputArrays[2] = (double *)malloc(nb_pixels); // h_z
            outputArrays[3] = (double *)malloc(2 * nb_pixels); // h_incang
            outputArrays[4] = (double *)malloc(2 * nb_pixels); // h_losang

            runGPUTopo(i, pixPerImg, gpu_inputs_d, gpu_inputs_i, gpu_dem, gpu_rho, gpu_dopline, gpu_orbNvec, gpu_orbSvs, outputArrays);
            for (int j=0; j<5; j++) writeArrays[j] = outputArrays[j];
            if (i != 0) printf("  Waiting for previous block-write to finish...\n"); // First block will never need to wait
            pthread_attr_destroy(&attr); // Reset joinable attr before waiting for thread to finish
            pthread_join(writeThread, &thread_stat); // Wait for write thread to finish
            printf("  Writing block %d out (asynchronously) to image files...\n", i);
            wd.accessors = (void**)accObjs; // Reset write data
            //wd.imgArrs = writeArrays;
            wd.lat = writeArrays[0];
            wd.lon = writeArrays[1];
            wd.z = writeArrays[2];
            wd.inc = writeArrays[3];
            wd.los = writeArrays[4];
            wd.incFlag = incFlag;
            wd.losFlag = losFlag;
            wd.nLines = linesPerImg;
            wd.width = width;
            wd.firstWrite = false;
            pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); // Set joinable attr
            pthread_create(&writeThread, &attr, writeToFile, (void*)&wd); // Spawn background write thread
            //writeToFile((void**)accObjs, outputArrays, incFlag, losFlag, linesPerImg, width, false);
        }

        // If there are lines that weren't processed (i.e. a non-full block)...
        if (remPix > 0) {

            nb_pixels = remPix * sizeof(double);
            outputArrays[0] = (double *)malloc(nb_pixels);
            outputArrays[1] = (double *)malloc(nb_pixels);
            outputArrays[2] = (double *)malloc(nb_pixels);
            outputArrays[3] = (double *)malloc(2 * nb_pixels);
            outputArrays[4] = (double *)malloc(2 * nb_pixels);

            printf("    Loading slantrange and doppler data...\n");
            for (int i=0; i<remLines; i++) {
                slrngAccObj->getLineSequential((char *)raw_line);
                for (int j=0; j<width; j++) gpu_rho[(i*width)+j] = raw_line[j];
                dopAccObj->getLineSequential((char *)raw_line);
                for (int j=0; j<width; j++) gpu_dopline[(i*width)+j] = raw_line[j];
            }
            for (int i=0; i<5; i++) writeArrays[i] = outputArrays[i];
            runGPUTopo((-1*pixPerImg*nBlocks), remPix, gpu_inputs_d, gpu_inputs_i, gpu_dem, gpu_rho, gpu_dopline, gpu_orbNvec, gpu_orbSvs, outputArrays);
            printf("  Waiting for previous block-write to finish...\n");
            pthread_attr_destroy(&attr);
            pthread_join(writeThread, &thread_stat);
            printf("  Writing remaining %d lines out (asynchronously) to image files...\n", remLines);
            wd.accessors = (void**)accObjs;
            //wd.imgArrs = outputArrays;
            wd.lat = writeArrays[0];
            wd.lon = writeArrays[1];
            wd.z = writeArrays[2];
            wd.inc = writeArrays[3];
            wd.los = writeArrays[4];
            wd.incFlag = incFlag;
            wd.losFlag = losFlag;
            wd.nLines = remLines;
            wd.width = width;
            wd.firstWrite = false;
            pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
            pthread_create(&writeThread, &attr, writeToFile, (void*)&wd);
            //writeToFile((void**)accObjs, outputArrays, incFlag, losFlag, remLines, width, false);
        }
        pthread_attr_destroy(&attr);
        pthread_join(writeThread, &thread_stat);
        printf("  Finished writing to files!\n");

        printf("\n  ------------------ EXITING GPU TOPO ------------------\n\n");
        printf("Finished!\n");

        delete[] raw_line;
        delete[] gpu_dem;
        delete[] gpu_rho;
        delete[] gpu_dopline;
        delete[] gpu_orbSvs;
    } else {

        // For each line
        for (int line=0; line<length; line++) {
            // Set up the geometry
            // Step 1: Get satellite position
            // Get time
            tline = t0 + (Nazlooks * (line / prf));

            // Get state vector
            stat = orb.interpolateOrbit(tline,xyzsat,velsat,orbit_method);
            if (stat != 0) {
                printf("Error in Topo::topo - Error getting state vector for bounds computation.\n");
                exit(1);
            }
            linalg.unitvec(velsat,vhat); // vhat - unit vector along velocity
            vmag = linalg.norm(velsat); // vmag - magnitude of the velocity

            // Step 2: Get local radius of curvature along heading
            // Convert satellite position to lat-lon
            elp.latlon(xyzsat,llhsat,XYZ_2_LLH);
            height = llhsat[2];

            // Step 3: Get TCN basis using satellite basis
            elp.tcnbasis(xyzsat,velsat,that,chat,nhat); // that - along local tangent to the planet
                                                        // chat - along the cross track direction
                                                        // nhat - along the local normal

            // Step 4: Get Doppler information for the line
            // For native doppler, this corresponds to doppler polynomial
            // For zero doppler, its a constant zero polynomial
            dopAccObj->getLineSequential((char *)raw_line);
            dopline.assign(raw_line,(raw_line + width));

            // Get the slant range
            slrngAccObj->getLineSequential((char *)raw_line);
            rho.assign(raw_line,(raw_line + width));

            // Step 4: Set up SCH basis right below the satellite
            peg.lat = llhsat[0];
            peg.lon = llhsat[1];
            peg.hdg = peghdg;
            //hnadir = 0.0;
            ptm.radar_to_xyz(elp,peg);
            rcurv = ptm.radcur;
            for (int idx=0; idx<width; idx++) {
                converge[idx] = 0;
                z[idx] = 0.0;
                zsch[idx] = 0.0;
            }
            if ((line % 1000) == 0) {
                printf("Processing line: %d %g\n", line, vmag);
                printf("Dopplers: %g %g %g\n", dopline[0], dopline[(width / 2) - 1], dopline[(width - 1)]);
            }

            // Initialize lat/lon to middle of input DEM
            for (int idx=0; idx<width; idx++) {
                lat[idx] = ufirstlat + (0.5 * deltalat * udemlength);
                lon[idx] = ufirstlon + (0.5 * deltalon * udemwidth);
            }
            //  SWOT-specific near range check (currently not implemented)
            //  Computing nadir height
            //if (nearrangeflag != 0) {
            //    demlat = (((llhsat[0] * (180. / M_PI))) - ufirstlat) / deltalat) + 1;
            //    demlon = (((llhsat[1] * (180. / M_PI)) - ufirstlon) / deltalon) + 1;
            //    if (demlat < 1) demlat = 1;
            //    if (demlat > (udemlength - 1)) demlat = udemlength - 1;
            //    if (demlon < 1) demlon = 1;
            //    if (demlon > (udemwidth - 1)) demlon = udemwidth - 1;

            //    idemlat = int(demlat);
            //    idemlon = int(demlon);
            //    fraclat = demlat - idemlat;
            //    fraclon = demlon - idemlon;
            //    hnadir = tzMethods.interpolate(dem,idemlon,idemlat,fraclon,fraclat,udemwidth,udemlength,dem_method);
            //}

            // Start the iterations
            for (int iter=0; iter<=(numiter+extraiter); iter++) {
                #pragma omp parallel for private(pixel,beta,alpha,gamm,idemlat,idemlon,fraclat,fraclon,\
                                                 demlat,demlon,aa,bb,rng,costheta,sintheta,dopfact) \
                                         firstprivate(sch,llh,xyz,llh_prev,xyz_prev,delta) \
                                         reduction(+:totalconv)  // Optimized atomic accumulation of totalconv
                for (pixel=0; pixel<width; pixel++) {
                    rng = rho[pixel];
                    dopfact = (0.5 * wvl * (dopline[pixel] / vmag)) * rng;

                    // If pixel hasn't converged
                    if (converge[pixel] == 0) {

                        // Use previous llh in degrees and meters
                        llh_prev[0] = lat[pixel] / (180. / M_PI);
                        llh_prev[1] = lon[pixel] / (180. / M_PI);
                        llh_prev[2] = z[pixel];

                        // Solve for new position at height zsch
                        aa = height + rcurv;
                        bb = rcurv + zsch[pixel];

                        // Normalize reasonably to avoid overflow
                        costheta = 0.5 * ((aa / rng) + (rng / aa) - ((bb / aa) * (bb / rng)));
                        sintheta = sqrt(1.0 - (costheta * costheta));

                        // Vector from satellite to point on ground can be written as
                        //   vec(dr) = alpha * vec(that) + beta * vec(chat) + gamma *
                        //              vec(nhat)
                        gamm = costheta * rng;
                        alpha = (dopfact - (gamm * linalg.dot(nhat,vhat))) / linalg.dot(vhat,that);
                        beta = -ilrl * sqrt((rng * rng * sintheta * sintheta) - (alpha * alpha));

                        // xyz position of target
                        for (int idx=0; idx<3; idx++) delta[idx] = (gamm * nhat[idx]) + (alpha * that[idx]) + (beta * chat[idx]);
                        for (int idx=0; idx<3; idx++) xyz[idx] = xyzsat[idx] + delta[idx];
                        elp.latlon(xyz,llh,XYZ_2_LLH);

                        // Convert lat, lon, hgt to xyz coordinates
                        lat[pixel] = llh[0] * (180. / M_PI);
                        lon[pixel] = llh[1] * (180. / M_PI);
                        demlat = ((lat[pixel] - ufirstlat) / deltalat) + 1;
                        demlon = ((lon[pixel] - ufirstlon) / deltalon) + 1;
                        if (demlat < 1) demlat = 1;
                        if (demlat > (udemlength-1)) demlat = udemlength - 1;
                        if (demlon < 1) demlon = 1;
                        if (demlon > (udemwidth-1)) demlon = udemwidth - 1;
                        idemlat = int(demlat);
                        idemlon = int(demlon);
                        fraclat = demlat - idemlat;
                        fraclon = demlon - idemlon;
                        z[pixel] = tzMethods.interpolate(dem,idemlon,idemlat,fraclon,fraclat,udemwidth,udemlength,dem_method);
                        if (z[pixel] < -500.0) z[pixel] = -500.0;

                        // Given llh, where h = z(pixel, line) in WGS84, get the SCH height
                        llh[0] = lat[pixel] / (180. / M_PI);
                        llh[1] = lon[pixel] / (180. / M_PI);
                        llh[2] = z[pixel];
                        elp.latlon(xyz,llh,LLH_2_XYZ);
                        ptm.convert_sch_to_xyz(sch,xyz,XYZ_2_SCH);
                        zsch[pixel] = sch[2];

                        // Absolute distance
                        distance[pixel] = sqrt(pow((xyz[0]-xyzsat[0]),2)+pow((xyz[1]-xyzsat[1]),2) + pow((xyz[2]-xyzsat[2]),2)) - rng;
                        if (abs(distance[pixel]) <= thresh) {
                            zsch[pixel] = sch[2];
                            converge[pixel] = 1;
                            totalconv = totalconv + 1;
                        } else if (iter > numiter) {
                            elp.latlon(xyz_prev,llh_prev,LLH_2_XYZ);
                            for (int idx=0; idx<3; idx++) xyz[idx] = 0.5 * (xyz_prev[idx] + xyz[idx]);

                            // Repopulate lat, lon, z
                            elp.latlon(xyz,llh,XYZ_2_LLH);
                            lat[pixel] = llh[0] * (180. / M_PI);
                            lon[pixel] = llh[1] * (180. / M_PI);
                            z[pixel] = llh[2];
                            ptm.convert_sch_to_xyz(sch,xyz,XYZ_2_SCH);
                            zsch[pixel] = sch[2];

                            // Absolute distance
                            distance[pixel] = sqrt(pow((xyz[0]-xyzsat[0]),2)+pow((xyz[1]-xyzsat[1]),2) + pow((xyz[2]-xyzsat[2]),2)) - rng;
                        }
                    }
                }
                //end OMP for loop
            }

            // Final computation.
            // The output points are exactly at range pixel
            // Distance from the satellite
            #pragma omp parallel for private(pixel,cosalpha,rng,aa,bb,alpha,beta,gamm,costheta,sintheta,dopfact,\
                                             demlat,demlon,idemlat,idemlon,fraclat,fraclon,enunorm) \
                                     firstprivate(xyz,llh,delta,enumat,xyz2enu,enu)
            for (pixel=0; pixel<width; pixel++) {
                rng = rho[pixel];
                dopfact = (0.5 * wvl * (dopline[pixel] / vmag)) * rng;

                // Solve for new position at height zsch
                aa = height + rcurv;
                bb = rcurv + zsch[pixel];
                costheta = 0.5 * ((aa / rng) + (rng / aa) - ((bb / aa) * (bb / rng)));
                sintheta = sqrt(1.0 - (costheta * costheta));
                gamm = costheta * rng;
                alpha = (dopfact - (gamm * linalg.dot(nhat,vhat))) / linalg.dot(vhat,that);
                beta = -ilrl * sqrt((rng * rng * sintheta * sintheta) - (alpha * alpha));

                // xyz position of target
                for (int idx=0; idx<3; idx++) delta[idx] = (gamm * nhat[idx]) + (alpha * that[idx]) + (beta * chat[idx]);
                for (int idx=0; idx<3; idx++) xyz[idx] = xyzsat[idx] + delta[idx];
                elp.latlon(xyz,llh,XYZ_2_LLH);

                // Copy into output arrays
                lat[pixel] = llh[0] * (180. / M_PI);
                lon[pixel] = llh[1] * (180. / M_PI);
                z[pixel] = llh[2];
                distance[pixel] = sqrt(pow((xyz[0]-xyzsat[0]),2)+pow((xyz[1]-xyzsat[1]),2) + pow((xyz[2]-xyzsat[2]),2)) - rng;

                // Computation in ENU coordinates around target
                linalg.enubasis(llh[0],llh[1],enumat);
                linalg.tranmat(enumat,xyz2enu);
                linalg.matvec(xyz2enu,delta,enu);
                cosalpha = abs(enu[2]) / linalg.norm(enu);

                // LOS vectors
                losang[(2*pixel)] = acos(cosalpha) * (180. / M_PI);
                losang[((2*pixel)+1)] = (atan2(-enu[1],-enu[0]) - (0.5*M_PI)) * (180. / M_PI);
                incang[(2*pixel)] = acos(costheta) * (180. / M_PI);

                // ctrack gets stored in zsch
                zsch[pixel] = rng * sintheta;

                // Get local incidence angle
                demlat = ((lat[pixel] - ufirstlat) / deltalat) + 1;
                demlon = ((lon[pixel] - ufirstlon) / deltalon) + 1;
                if (demlat < 2) demlat = 2;
                if (demlat > (udemlength-1)) demlat = udemlength - 1;
                if (demlon < 2) demlon = 2;
                if (demlon > (udemwidth-1)) demlon = udemwidth - 1;
                idemlat = int(demlat);
                idemlon = int(demlon);
                fraclat = demlat - idemlat;
                fraclon = demlon - idemlon;
                gamm = lat[pixel] / (180. / M_PI);

                // Slopex
                aa = tzMethods.interpolate(dem,(idemlon-1),idemlat,fraclon,fraclat,udemwidth,udemlength,dem_method);
                bb = tzMethods.interpolate(dem,(idemlon+1),idemlat,fraclon,fraclat,udemwidth,udemlength,dem_method);
                alpha = ((bb - aa) * (180. / M_PI)) / (2.0 * elp.reast(gamm) * deltalon);

                // Slopey
                aa = tzMethods.interpolate(dem,idemlon,(idemlat-1),fraclon,fraclat,udemwidth,udemlength,dem_method);
                bb = tzMethods.interpolate(dem,idemlon,(idemlat+1),fraclon,fraclat,udemwidth,udemlength,dem_method);
                beta = ((bb - aa) * (180. / M_PI)) / (2.0 * elp.rnorth(gamm) * deltalat);
                enunorm = linalg.norm(enu);
                for (int idx=0; idx<3; idx++) enu[idx] = enu[idx] / enunorm;
                costheta = ((enu[0] * alpha) + (enu[1] * beta) - enu[2]) / sqrt(1.0 + (alpha * alpha) + (beta * beta));
                incang[((2*pixel)+1)] = acos(costheta) * (180. / M_PI);
            }
            //end OMP for loop

            double mnlat,mxlat,mnlon,mxlon;
            mnlat = mnlon = 10000.0;
            mxlat = mxlon = -10000.0;
            for (int ii=0; ii<width; ii++) {
                if (lat[ii] < mnlat) mnlat = lat[ii];
                if (lat[ii] > mxlat) mxlat = lat[ii];
                if (lon[ii] < mnlon) mnlon = lon[ii];
                if (lon[ii] > mxlon) mxlon = lon[ii];
            }
            min_lat = min(mnlat, min_lat);
            max_lat = max(mxlat, max_lat);
            min_lon = min(mnlon, min_lon);
            max_lon = max(mxlon, max_lon);

            latAccObj->setLineSequential((char *)&lat[0]);
            lonAccObj->setLineSequential((char *)&lon[0]);
            heightAccObj->setLineSequential((char *)&z[0]);
            if (losAccessor > 0) losAccObj->setLineSequential((char *)&losang[0]);
            if (incAccessor > 0) incAccObj->setLineSequential((char *)&incang[0]);

            if (maskAccessor > 0) {
                double mnzsch,mxzsch;
                mnzsch = 10000.0;
                mxzsch = -10000.0;
                for (int ii=0; ii<width; ii++) {
                    if (zsch[ii] < mnzsch) mnzsch = zsch[ii];
                    if (zsch[ii] > mxzsch) mxzsch = zsch[ii];
                }
                ctrackmin = mnzsch - demmax;
                ctrackmax = mxzsch + demmax;
                dctrack = (ctrackmax - ctrackmin) / (owidth - 1.0);

                // Sort lat/lon by ctrack
                linalg.insertionSort(zsch,width);
                linalg.insertionSort(lat,width);
                linalg.insertionSort(lon,width);

                #pragma omp parallel for private(pixel,aa,bb,i_type,demlat,demlon,\
                                                 idemlat,idemlon,fraclat,fraclon) \
                                         firstprivate(llh,xyz)
                for (pixel=0; pixel<owidth; pixel++) {
                    aa = ctrackmin + (pixel * dctrack);
                    ctrack[pixel] = aa;
                    i_type = linalg.binarySearch(zsch,0,(width-1),aa);

                    // Simple bi-linear interpolation
                    fraclat = (aa - zsch[i_type]) / (zsch[(i_type+1)] - zsch[i_type]);
                    demlat = lat[i_type] + (fraclat * (lat[(i_type+1)] - lat[i_type]));
                    demlon = lon[i_type] + (fraclat * (lon[(i_type+1)] - lon[i_type]));
                    llh[0] = demlat / (180. / M_PI);
                    llh[1] = demlon / (180. / M_PI);
                    demlat = ((demlat - ufirstlat) / deltalat) + 1;
                    demlon = ((demlon - ufirstlon) / deltalon) + 1;
                    if (demlat < 2) demlat = 2;
                    if (demlat > (udemlength-1)) demlat = udemlength - 1;
                    if (demlon < 2) demlon = 2;
                    if (demlon > (udemwidth-1)) demlon = udemwidth - 1;
                    idemlat = int(demlat);
                    idemlon = int(demlon);
                    fraclat = demlat - idemlat;
                    fraclon = demlon - idemlon;
                    llh[2] = tzMethods.interpolate(dem,idemlon,idemlat,fraclon,fraclat,udemwidth,udemlength,dem_method);
                    elp.latlon(xyz,llh,LLH_2_XYZ);
                    for (int idx=0; idx<3; idx++) xyz[idx] = xyz[idx] - xyzsat[idx];
                    bb = linalg.norm(xyz);
                    orng[pixel] = bb;
                    aa = abs((nhat[0] * xyz[0]) + (nhat[1]  * xyz[1]) + (nhat[2] * xyz[2]));
                    oview[pixel] = acos(aa / bb) * (180. / M_PI);
                }
                //end OMP for loop

                // Again sort in terms of slant range
                linalg.insertionSort(orng,owidth);
                linalg.insertionSort(ctrack,owidth);
                linalg.insertionSort(oview,owidth);
                for (int idx=0; idx<width; idx++) mask[idx] = 0;
                for (int idx=0; idx<owidth; idx++) omask[idx] = 0;
                aa = incang[0];
                for (pixel=1; pixel<width; pixel++) {
                    bb = incang[(2*pixel)];
                    if (bb <= aa) mask[pixel] = 1;
                    else aa = bb;
                }
                aa = incang[(2*width)-2];
                for (pixel=(width-2); pixel>=0; pixel--) {
                    bb = incang[(2*pixel)];
                    if (bb >= aa) mask[pixel] = 1;
                    else aa = bb;
                }
                aa = ctrack[0];
                for (pixel=1; pixel<width; pixel++) {
                    bb = ctrack[pixel];
                    if ((bb <= aa) && (omask[pixel] < 2)) omask[pixel] = omask[pixel] + 2;
                    else aa = bb;
                }
                aa = ctrack[(owidth-1)];
                for (pixel=(owidth-2); pixel>=0; pixel--) {
                    bb = ctrack[pixel];
                    if ((bb >= aa) && (omask[pixel] < 2)) omask[pixel] = omask[pixel] + 2;
                    else aa = bb;
                }
                for (pixel=0; pixel<owidth; pixel++) {
                    if (omask[pixel] > 0) {
                        idemlat = linalg.binarySearch(rho,0,(width-1),orng[pixel]);
                        if (mask[idemlat] < omask[pixel]) mask[idemlat] = mask[idemlat] + omask[pixel];
                    }
                }
                maskAccObj->setLineSequential((char *)&mask[0]);
            }
        }
        delete[] raw_line;

        printf("Total convergence: %d out of %d.\n", totalconv, (width * length));
    }
}

