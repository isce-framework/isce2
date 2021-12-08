//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cuda_runtime.h>
#include <cassert>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>

#define max(a,b) \
            ({ __typeof__ (a) _a = (a); \
               __typeof__ (b) _b = (b); \
               _a > _b ? _a : _b;})

#define min(a,b) \
            ({ __typeof__ (a) _a = (a); \
               __typeof__ (b) _b = (b); \
               _a < _b ? _a : _b;})

#define SPEED_OF_LIGHT 299792458.
#define BAD_VALUE -999999.
#define THRD_PER_RUN 128

struct InputImageArrs {
    double *lat;
    double *lon;
    double *dem;
};

struct OutputImageArrs {
    double *azt;
    double *rgm;
    double *azoff;
    double *rgoff;
};

struct stateVector {
    double t;
    double px;
    double py;
    double pz;
    double vx;
    double vy;
    double vz;
};

struct Orbit {
    int nVec;
    struct stateVector *svs;
};

struct Ellipsoid {
    double a;
    double e2;
};

struct Poly1d {
    int order;
    double mean;
    double norm;
    double *coeffs;
};

__constant__ double d_inpts_double[9];
__constant__ int d_inpts_int[3];

// Mem usage: 27 doubles (216 bytes) per call
__device__ int interpolateOrbit(struct Orbit *orb, double t, double *xyz, double *vel) {
    double h[4], hdot[4], f0[4], f1[4], g0[4], g1[4];
    double sum = 0.0;
    int i;
    int v0 = -1;

    if ((t < orb->svs[0].t) || (t > orb->svs[orb->nVec-1].t)) return 1;
    for (i=0; i<orb->nVec; i++) {
        if ((orb->svs[i].t >= t) && (v0 == -1)) {
            v0 = min(max((i-2),0),(orb->nVec-4));
        }
    }

    f1[0] = t - orb->svs[v0].t;
    f1[1] = t - orb->svs[v0+1].t;
    f1[2] = t - orb->svs[v0+2].t;
    f1[3] = t - orb->svs[v0+3].t;

    sum = (1.0 / (orb->svs[v0].t - orb->svs[v0+1].t)) + (1.0 / (orb->svs[v0].t - orb->svs[v0+2].t)) + (1.0 / (orb->svs[v0].t - orb->svs[v0+3].t));
    f0[0] = 1.0 - (2.0 * (t - orb->svs[v0].t) * sum);
    sum = (1.0 / (orb->svs[v0+1].t - orb->svs[v0].t)) + (1.0 / (orb->svs[v0+1].t - orb->svs[v0+2].t)) + (1.0 / (orb->svs[v0+1].t - orb->svs[v0+3].t));
    f0[1] = 1.0 - (2.0 * (t - orb->svs[v0+1].t) * sum);
    sum = (1.0 / (orb->svs[v0+2].t - orb->svs[v0].t)) + (1.0 / (orb->svs[v0+2].t - orb->svs[v0+1].t)) + (1.0 / (orb->svs[v0+2].t - orb->svs[v0+3].t));
    f0[2] = 1.0 - (2.0 * (t - orb->svs[v0+2].t) * sum);
    sum = (1.0 / (orb->svs[v0+3].t - orb->svs[v0].t)) + (1.0 / (orb->svs[v0+3].t - orb->svs[v0+1].t)) + (1.0 / (orb->svs[v0+3].t - orb->svs[v0+2].t));
    f0[3] = 1.0 - (2.0 * (t - orb->svs[v0+3].t) * sum);

    h[0] = ((t - orb->svs[v0+1].t) / (orb->svs[v0].t - orb->svs[v0+1].t)) * ((t - orb->svs[v0+2].t) / (orb->svs[v0].t - orb->svs[v0+2].t)) *
                ((t - orb->svs[v0+3].t) / (orb->svs[v0].t - orb->svs[v0+3].t));
    h[1] = ((t - orb->svs[v0].t) / (orb->svs[v0+1].t - orb->svs[v0].t)) * ((t - orb->svs[v0+2].t) / (orb->svs[v0+1].t - orb->svs[v0+2].t)) *
                ((t - orb->svs[v0+3].t) / (orb->svs[v0+1].t - orb->svs[v0+3].t));
    h[2] = ((t - orb->svs[v0].t) / (orb->svs[v0+2].t - orb->svs[v0].t)) * ((t - orb->svs[v0+1].t) / (orb->svs[v0+2].t - orb->svs[v0+1].t)) *
                ((t - orb->svs[v0+3].t) / (orb->svs[v0+2].t - orb->svs[v0+3].t));
    h[3] = ((t - orb->svs[v0].t) / (orb->svs[v0+3].t - orb->svs[v0].t)) * ((t - orb->svs[v0+1].t) / (orb->svs[v0+3].t - orb->svs[v0+1].t)) *
                ((t - orb->svs[v0+2].t) / (orb->svs[v0+3].t - orb->svs[v0+2].t));

    sum = (((t - orb->svs[v0+2].t) / (orb->svs[v0].t - orb->svs[v0+2].t)) * ((t - orb->svs[v0+3].t) / (orb->svs[v0].t - orb->svs[v0+3].t))) *
            (1.0 / (orb->svs[v0].t - orb->svs[v0+1].t));
    sum += (((t - orb->svs[v0+1].t) / (orb->svs[v0].t - orb->svs[v0+1].t)) * ((t - orb->svs[v0+3].t) / (orb->svs[v0].t - orb->svs[v0+3].t))) *
            (1.0 / (orb->svs[v0].t - orb->svs[v0+2].t));
    sum += (((t - orb->svs[v0+1].t) / (orb->svs[v0].t - orb->svs[v0+1].t)) * ((t - orb->svs[v0+2].t) / (orb->svs[v0].t - orb->svs[v0+2].t))) *
            (1.0 / (orb->svs[v0].t - orb->svs[v0+3].t));
    hdot[0] = sum;

    sum = (((t - orb->svs[v0+2].t) / (orb->svs[v0+1].t - orb->svs[v0+2].t)) * ((t - orb->svs[v0+3].t) / (orb->svs[v0+1].t - orb->svs[v0+3].t))) *
            (1.0 / (orb->svs[v0+1].t - orb->svs[v0].t));
    sum += (((t - orb->svs[v0].t) / (orb->svs[v0+1].t - orb->svs[v0].t)) * ((t - orb->svs[v0+3].t) / (orb->svs[v0+1].t - orb->svs[v0+3].t))) *
            (1.0 / (orb->svs[v0+1].t - orb->svs[v0+2].t));
    sum += (((t - orb->svs[v0].t) / (orb->svs[v0+1].t - orb->svs[v0].t)) * ((t - orb->svs[v0+2].t) / (orb->svs[v0+1].t - orb->svs[v0+2].t))) *
            (1.0 / (orb->svs[v0+1].t - orb->svs[v0+3].t));
    hdot[1] = sum;

    sum = (((t - orb->svs[v0+1].t) / (orb->svs[v0+2].t - orb->svs[v0+1].t)) * ((t - orb->svs[v0+3].t) / (orb->svs[v0+2].t - orb->svs[v0+3].t))) *
            (1.0 / (orb->svs[v0+2].t - orb->svs[v0].t));
    sum += (((t - orb->svs[v0].t) / (orb->svs[v0+2].t - orb->svs[v0].t)) * ((t - orb->svs[v0+3].t) / (orb->svs[v0+2].t - orb->svs[v0+3].t))) *
            (1.0 / (orb->svs[v0+2].t - orb->svs[v0+1].t));
    sum += (((t - orb->svs[v0].t) / (orb->svs[v0+2].t - orb->svs[v0].t)) * ((t - orb->svs[v0+1].t) / (orb->svs[v0+2].t - orb->svs[v0+1].t))) *
            (1.0 / (orb->svs[v0+2].t - orb->svs[v0+3].t));
    hdot[2] = sum;

    sum = (((t - orb->svs[v0+1].t) / (orb->svs[v0+3].t - orb->svs[v0+1].t)) * ((t - orb->svs[v0+2].t) / (orb->svs[v0+3].t - orb->svs[v0+2].t))) *
            (1.0 / (orb->svs[v0+3].t - orb->svs[v0].t));
    sum += (((t - orb->svs[v0].t) / (orb->svs[v0+3].t - orb->svs[v0].t)) * ((t - orb->svs[v0+2].t) / (orb->svs[v0+3].t - orb->svs[v0+2].t))) *
            (1.0 / (orb->svs[v0+3].t - orb->svs[v0+1].t));
    sum += (((t - orb->svs[v0].t) / (orb->svs[v0+3].t - orb->svs[v0].t)) * ((t - orb->svs[v0+1].t) / (orb->svs[v0+3].t - orb->svs[v0+1].t))) *
            (1.0 / (orb->svs[v0+3].t - orb->svs[v0+2].t));
    hdot[3] = sum;

    g1[0] = h[0] + (2.0 * (t - orb->svs[v0].t) * hdot[0]);
    g1[1] = h[1] + (2.0 * (t - orb->svs[v0+1].t) * hdot[1]);
    g1[2] = h[2] + (2.0 * (t - orb->svs[v0+2].t) * hdot[2]);
    g1[3] = h[3] + (2.0 * (t - orb->svs[v0+3].t) * hdot[3]);

    sum = (1.0 / (orb->svs[v0].t - orb->svs[v0+1].t)) + (1.0 / (orb->svs[v0].t - orb->svs[v0+2].t)) + (1.0 / (orb->svs[v0].t - orb->svs[v0+3].t));
    g0[0] = 2.0 * ((f0[0] * hdot[0]) - (h[0] * sum));
    sum = (1.0 / (orb->svs[v0+1].t - orb->svs[v0].t)) + (1.0 / (orb->svs[v0+1].t - orb->svs[v0+2].t)) + (1.0 / (orb->svs[v0+1].t - orb->svs[v0+3].t));
    g0[1] = 2.0 * ((f0[1] * hdot[1]) - (h[1] * sum));
    sum = (1.0 / (orb->svs[v0+2].t - orb->svs[v0].t)) + (1.0 / (orb->svs[v0+2].t - orb->svs[v0+1].t)) + (1.0 / (orb->svs[v0+2].t - orb->svs[v0+3].t));
    g0[2] = 2.0 * ((f0[2] * hdot[2]) - (h[2] * sum));
    sum = (1.0 / (orb->svs[v0+3].t - orb->svs[v0].t)) + (1.0 / (orb->svs[v0+3].t - orb->svs[v0+1].t)) + (1.0 / (orb->svs[v0+3].t - orb->svs[v0+2].t));
    g0[3] = 2.0 * ((f0[3] * hdot[3]) - (h[3] * sum));

    xyz[0] = (((orb->svs[v0].px * f0[0]) + (orb->svs[v0].vx * f1[0])) * h[0] * h[0]) + (((orb->svs[v0+1].px * f0[1]) + (orb->svs[v0+1].vx * f1[1])) * h[1] * h[1]) +
                (((orb->svs[v0+2].px * f0[2]) + (orb->svs[v0+2].vx * f1[2])) * h[2] * h[2]) + (((orb->svs[v0+3].px * f0[3]) + (orb->svs[v0+3].vx * f1[3])) * h[3] * h[3]);
    xyz[1] = (((orb->svs[v0].py * f0[0]) + (orb->svs[v0].vy * f1[0])) * h[0] * h[0]) + (((orb->svs[v0+1].py * f0[1]) + (orb->svs[v0+1].vy * f1[1])) * h[1] * h[1]) +
                (((orb->svs[v0+2].py * f0[2]) + (orb->svs[v0+2].vy * f1[2])) * h[2] * h[2]) + (((orb->svs[v0+3].py * f0[3]) + (orb->svs[v0+3].vy * f1[3])) * h[3] * h[3]);
    xyz[2] = (((orb->svs[v0].pz * f0[0]) + (orb->svs[v0].vz * f1[0])) * h[0] * h[0]) + (((orb->svs[v0+1].pz * f0[1]) + (orb->svs[v0+1].vz * f1[1])) * h[1] * h[1]) +
                (((orb->svs[v0+2].pz * f0[2]) + (orb->svs[v0+2].vz * f1[2])) * h[2] * h[2]) + (((orb->svs[v0+3].pz * f0[3]) + (orb->svs[v0+3].vz * f1[3])) * h[3] * h[3]);

    vel[0] = (((orb->svs[v0].px * g0[0]) + (orb->svs[v0].vx * g1[0])) * h[0]) + (((orb->svs[v0+1].px * g0[1]) + (orb->svs[v0+1].vx * g1[1])) * h[1]) +
                (((orb->svs[v0+2].px * g0[2]) + (orb->svs[v0+2].vx * g1[2])) * h[2]) + (((orb->svs[v0+3].px * g0[3]) + (orb->svs[v0+3].vx * g1[3])) * h[3]);
    vel[1] = (((orb->svs[v0].py * g0[0]) + (orb->svs[v0].vy * g1[0])) * h[0]) + (((orb->svs[v0+1].py * g0[1]) + (orb->svs[v0+1].vy * g1[1])) * h[1]) +
                (((orb->svs[v0+2].py * g0[2]) + (orb->svs[v0+2].vy * g1[2])) * h[2]) + (((orb->svs[v0+3].py * g0[3]) + (orb->svs[v0+3].vy * g1[3])) * h[3]);
    vel[2] = (((orb->svs[v0].pz * g0[0]) + (orb->svs[v0].vz * g1[0])) * h[0]) + (((orb->svs[v0+1].pz * g0[1]) + (orb->svs[v0+1].vz * g1[1])) * h[1]) +
                (((orb->svs[v0+2].pz * g0[2]) + (orb->svs[v0+2].vz * g1[2])) * h[2]) + (((orb->svs[v0+3].pz * g0[3]) + (orb->svs[v0+3].vz * g1[3])) * h[3]);

    return 0; // Successful interpolation
}

// 8 bytes per call
__device__ void llh2xyz(struct Ellipsoid *elp, double *xyz, double *llh) {
    double re;
    re = elp->a / sqrt(1.0 - (elp->e2 * pow(sin(llh[0]),2)));
    xyz[0] = (re + llh[2]) * cos(llh[0]) * cos(llh[1]);
    xyz[1] = (re + llh[2]) * cos(llh[0]) * sin(llh[1]);
    xyz[2] = ((re * (1.0 - elp->e2)) + llh[2]) * sin(llh[0]);
}

// 36 bytes per call
__device__ double evalPoly(struct Poly1d *poly, double xin) {
    double val, xval, scalex;
    int i;
    val = 0.;
    scalex = 1.;
    xval = (xin - poly->mean) / poly->norm;
    for (i=0; i<=poly->order; i++,scalex*=xval) val += scalex * poly->coeffs[i];
    return val;
}

// 0 bytes per call
__device__ double dot(double *a, double *b) {
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
}

__global__ void runGeo(struct Orbit orb, struct Poly1d fdvsrng, struct Poly1d fddotvsrng, struct OutputImageArrs outImgArrs, struct InputImageArrs inImgArrs,
                        int NPIXELS, int OFFSET_LINE) {
    int pixel = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (pixel < NPIXELS) { // The number of pixels in a run changes based on if it's a full run or a partial run
        /* * * * * * * * * * * * * * * * * * * * * * * * * * * * *
         *   Input mapping
         *
         * int[0] = demLength
         * int[1] = demWidth
         * int[2] = bistatic
         *
         * double[0] = major
         * double[1] = eccentricitySquared
         * double[2] = tstart
         * double[3] = tend
         * double[4] = wvl
         * double[5] = rngstart
         * double[6] = rngend
         * double[7] = dmrg
         * double[8] = dtaz
         * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

        double xyz[3], llh[3], satx[3], satv[3], dr[3];
        double rngpix, tline, tprev, fnprime, fdop, fdopder;
        int stat, i, j;
        bool isOutside, runIter;

        struct Ellipsoid elp;
        elp.a = d_inpts_double[0];
        elp.e2 = d_inpts_double[1];

        isOutside = false;
        runIter = true;
        llh[0] = inImgArrs.lat[pixel] * (M_PI / 180.);
        llh[1] = inImgArrs.lon[pixel] * (M_PI / 180.);
        llh[2] = inImgArrs.dem[pixel];

        llh2xyz(&elp,xyz,llh);

        tline = .5 * (d_inpts_double[2] + d_inpts_double[3]);
        stat = interpolateOrbit(&orb, tline, satx, satv); // Originally we got xyz_mid and vel_mid, then copied into satx/satv,
                                                          // but since these are all independent here it's fine
        if (stat != 0) isOutside = true; // Should exit, but this is next-best thing...

        for (i=0; i<51; i++) { // The whole "51 iterations" thing is messing with my coding OCD...
            if (runIter) { // Instead of breaking the loop
                tprev = tline;
                for (j=0; j<3; j++) dr[j] = xyz[j] - satx[j];
                rngpix = sqrt(pow(dr[0],2) + pow(dr[1],2) + pow(dr[2],2)); // No need to add the norm function (useless one-line)
                fdop = .5 * d_inpts_double[4] * evalPoly(&fdvsrng, rngpix);
                fdopder = .5 * d_inpts_double[4] * evalPoly(&fddotvsrng, rngpix);
                fnprime = (((fdop / rngpix) + fdopder) * dot(dr,satv)) - dot(satv,satv);
                tline = tline - ((dot(dr,satv) - (fdop * rngpix)) / fnprime);
                stat = interpolateOrbit(&orb, tline, satx, satv);
                if (stat != 0) {
                    tline = BAD_VALUE;
                    rngpix = BAD_VALUE;
                    runIter = false;
                }
                if (fabs(tline - tprev) < 5.e-9) runIter = false;
            }
        }

        if ((tline < d_inpts_double[2]) || (tline > d_inpts_double[3])) isOutside = true;
        rngpix = sqrt(pow((xyz[0]-satx[0]),2) + pow((xyz[1]-satx[1]),2) + pow((xyz[2]-satx[2]),2));
        if ((rngpix < d_inpts_double[5]) || (rngpix > d_inpts_double[6])) isOutside = true;
        if (d_inpts_int[2] == 1) { // Bistatic (won't be true for awhile, not currently implemented)
            tline = tline + ((2. * rngpix) / SPEED_OF_LIGHT);
            if ((tline < d_inpts_double[2]) || (tline > d_inpts_double[3])) isOutside = true;
            stat = interpolateOrbit(&orb, tline, satx, satv);
            if (stat != 0) isOutside = true;
            rngpix = sqrt(pow((xyz[0]-satx[0]),2) + pow((xyz[1]-satx[1]),2) + pow((xyz[2]-satx[2]),2));
            if ((rngpix < d_inpts_double[5]) || (rngpix > d_inpts_double[6])) isOutside = true;
        }

        if (!isOutside) {
            outImgArrs.rgm[pixel] = rngpix;
            outImgArrs.azt[pixel] = tline;
            outImgArrs.rgoff[pixel] = ((rngpix - d_inpts_double[5]) / d_inpts_double[7]) - double(int(pixel%d_inpts_int[1]));
            outImgArrs.azoff[pixel] = ((tline - d_inpts_double[2]) / d_inpts_double[8]) - double(int(pixel/d_inpts_int[1])+OFFSET_LINE);
        } else {
            outImgArrs.rgm[pixel] = BAD_VALUE;
            outImgArrs.azt[pixel] = BAD_VALUE;
            outImgArrs.rgoff[pixel] = BAD_VALUE;
            outImgArrs.azoff[pixel] = BAD_VALUE;
        }
    }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int nLinesPossible(int length, int width) {
    // 332 bytes per runGeo call (let's say 500 bytes for safety)
    // Device needs 7 * pixPerRun * sizeof(double) bytes malloc'ed
    // (56 * pixPerRun) - # bytes malloc'd on device
    // (500 * pixPerRun) - # bytes used by sum of all runGeo calls
    size_t freeByte, totalByte;
    int linesPerRun;
    cudaMemGetInfo(&freeByte, &totalByte);
    printf("Available free gpu memory in bytes %ld\n", freeByte);
    // use 100Mb as a rounding unit , may be adjusted
    size_t memoryRoundingUnit = 1024ULL * 1024ULL * 100;
    // use 2*memoryRoundingUnit as an overhead for safety
    freeByte = (freeByte / memoryRoundingUnit -2) * memoryRoundingUnit;
    assert(freeByte >0);
    // printf("GPU Memory to be used %ld\n", freeByte);
    // printf("Device has roughly %.4f GB of memory, ", double(totalByte)/1.e9);
    // determine the allowed max lines per run, 556 is per pixel memory usage (estimated)
    linesPerRun = freeByte / (7*sizeof(double) * width);
    assert(linesPerRun>0);
    printf("and can process roughly %d lines (each with %d pixels) per run.\n", linesPerRun, width);
    return linesPerRun;
}

void setOrbit(struct Orbit *orb) {
    orb->svs = (struct stateVector *)malloc(orb->nVec * sizeof(struct stateVector));
}

void freeOrbit(struct Orbit *orb) {
    free(orb->svs);
}

void setPoly1d(struct Poly1d *poly) {
    poly->coeffs = (double *)malloc((poly->order+1) * sizeof(double));
}

void freePoly1d(struct Poly1d *poly) {
    free(poly->coeffs);
}

void runGPUGeo(int iter, int numPix, double *h_inpts_dbl, int *h_inpts_int, double *h_lat, double *h_lon, double *h_dem, int h_orbNvec, double *h_orbSvs,
                int h_polyOrd, double h_polyMean, double h_polyNorm, double *h_polyCoeffs, double h_polyPRF, double **accArr) {

    double iStartCpy, iStartRun, iEndRun, iEndCpy;
    int i;

    struct stateVector *d_svs;
    double *d_fdPolyCoeffs, *d_fddotPolyCoeffs, *d_lat, *d_lon, *d_dem, *d_azt, *d_rgm, *d_azoff, *d_rgoff;

    struct InputImageArrs inImgArrs;
    struct OutputImageArrs outImgArrs;
    struct Orbit orb;
    struct Poly1d fdvsrng, fddotvsrng;

    cudaSetDevice(0);

    printf("    Allocating memory...\n");

    size_t nb_pixels = numPix * sizeof(double);

    orb.nVec = h_orbNvec;
    setOrbit(&orb); // Malloc memory for orbit on host (sizeof(stateVector)*nvec doubles)
    for (i=0; i<h_orbNvec; i++) {
        orb.svs[i].t = h_orbSvs[7*i];
        orb.svs[i].px = h_orbSvs[(7*i)+1];
        orb.svs[i].py = h_orbSvs[(7*i)+2];
        orb.svs[i].pz = h_orbSvs[(7*i)+3];
        orb.svs[i].vx = h_orbSvs[(7*i)+4];
        orb.svs[i].vy = h_orbSvs[(7*i)+5];
        orb.svs[i].vz = h_orbSvs[(7*i)+6];
    }
    fdvsrng.order = h_polyOrd;
    fdvsrng.mean = h_polyMean;
    fdvsrng.norm = h_polyNorm;
    setPoly1d(&fdvsrng); // Malloc memory for fdvsrng Poly1d on host (order+1 doubles)
    for (i=0; i<=h_polyOrd; i++) fdvsrng.coeffs[i] = h_polyPRF * h_polyCoeffs[i];
    if (h_polyOrd == 0) {
        fddotvsrng.order = 0;
        fddotvsrng.mean = 0.;
        fddotvsrng.norm = 1.;
        setPoly1d(&fddotvsrng); // Malloc memory for fddotvsrng Poly1d on host
        fddotvsrng.coeffs[0] = 0.;
    } else {
        fddotvsrng.order = h_polyOrd-1;
        fddotvsrng.mean = fdvsrng.mean;
        fddotvsrng.norm = fdvsrng.norm;
        setPoly1d(&fddotvsrng); // As above
        for (i=1; i<=h_polyOrd; i++) fddotvsrng.coeffs[i-1] = (i * fdvsrng.coeffs[i]) / fdvsrng.norm;
    }
    cudaMalloc((void**)&d_svs, (orb.nVec*sizeof(struct stateVector)));
    cudaMalloc((double**)&d_fdPolyCoeffs, ((fdvsrng.order+1)*sizeof(double)));
    cudaMalloc((double**)&d_fddotPolyCoeffs, ((fddotvsrng.order+1)*sizeof(double)));
    cudaMalloc((double**)&d_lat, nb_pixels);
    cudaMalloc((double**)&d_lon, nb_pixels);
    cudaMalloc((double**)&d_dem, nb_pixels);
    cudaMalloc((double**)&d_azt, nb_pixels);
    cudaMalloc((double**)&d_rgm, nb_pixels);
    cudaMalloc((double**)&d_azoff, nb_pixels);
    cudaMalloc((double**)&d_rgoff, nb_pixels);

    printf("    Done.\n    Copying data to GPU...\n");

    iStartCpy = cpuSecond();
    cudaMemcpy(d_svs, orb.svs, (orb.nVec*sizeof(struct stateVector)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fdPolyCoeffs, fdvsrng.coeffs, ((fdvsrng.order+1)*sizeof(double)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fddotPolyCoeffs, fddotvsrng.coeffs, ((fddotvsrng.order+1)*sizeof(double)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lat, h_lat, nb_pixels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lon, h_lon, nb_pixels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dem, h_dem, nb_pixels, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_inpts_double, h_inpts_dbl, (9*sizeof(double)));
    cudaMemcpyToSymbol(d_inpts_int, h_inpts_int, (3*sizeof(int)));

    freeOrbit(&orb); // Since the data for these is already on the GPU, we need the space again
    freePoly1d(&fdvsrng);
    freePoly1d(&fddotvsrng);
    orb.svs = d_svs; // Magic of the logic - we pass the objects in by value, but the variable
    fdvsrng.coeffs = d_fdPolyCoeffs; // size components (svs/coeffs) are malloc'ed on the GPU,
    fddotvsrng.coeffs = d_fddotPolyCoeffs; // so we can just have the objects store device ptrs
    inImgArrs.lat = d_lat;
    inImgArrs.lon = d_lon;
    inImgArrs.dem = d_dem;
    outImgArrs.azt = d_azt;
    outImgArrs.rgm = d_rgm;
    outImgArrs.azoff = d_azoff;
    outImgArrs.rgoff = d_rgoff;

    dim3 block(THRD_PER_RUN);
    dim3 grid((numPix + (THRD_PER_RUN - 1)) / THRD_PER_RUN);
    if ((grid.x * THRD_PER_RUN) > numPix) printf("    (NOTE: There will be %d 'empty' threads).\n", ((grid.x*THRD_PER_RUN)-numPix));

    if (iter > -1) printf("    Starting GPU Geo2rdr for run %d...\n", iter);
    else printf("    Starting GPU Geo2rdr for remaining lines...\n");

    iStartRun = cpuSecond();
    if (iter > -1) runGeo <<<grid, block>>>(orb, fdvsrng, fddotvsrng, outImgArrs, inImgArrs, numPix, int((iter*numPix)/h_inpts_int[1]));
    else runGeo <<<grid, block>>>(orb, fdvsrng, fddotvsrng, outImgArrs, inImgArrs, numPix, (-1*iter)); // This time iter is -1*nRuns*linesPerRun (i.e. a final partial block run)

    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) {
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    } if (errAsync != cudaSuccess) {
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
    }

    iEndRun = cpuSecond();
    if (iter > -1) printf("    GPU finished run %d in %f s.\n", iter, (iEndRun-iStartRun));
    else printf("    GPU finished remaining lines in %f s.\n", (iEndRun-iStartRun));

    printf("    Copying memory back to host...\n");

    cudaMemcpy(accArr[0], outImgArrs.rgm, nb_pixels, cudaMemcpyDeviceToHost);
    cudaMemcpy(accArr[1], outImgArrs.azt, nb_pixels, cudaMemcpyDeviceToHost);
    cudaMemcpy(accArr[2], outImgArrs.rgoff, nb_pixels, cudaMemcpyDeviceToHost);
    cudaMemcpy(accArr[3], outImgArrs.azoff, nb_pixels, cudaMemcpyDeviceToHost);

    iEndCpy = cpuSecond();
    if (iter > -1) printf("    GPU finished run %d (with memory copies) in %f s.\n", iter, (iEndCpy-iStartCpy));
    else printf("    GPU finished remaining lines (with memory copies) in %f s.\n", (iEndCpy-iStartCpy));

    printf("    Cleaning device memory and returning to main Geo2rdr function...\n");
    cudaFree(d_svs);
    cudaFree(d_fdPolyCoeffs);
    cudaFree(d_fddotPolyCoeffs);
    cudaFree(d_lat);
    cudaFree(d_lon);
    cudaFree(d_dem);
    cudaFree(d_azt);
    cudaFree(d_rgm);
    cudaFree(d_azoff);
    cudaFree(d_rgoff);
    cudaDeviceReset();
}
