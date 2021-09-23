//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>

#define SINC_SUB 8192
#define SINC_LEN 8
#define SINC_HALF (SINC_LEN/2)
#define SINC_ONE (SINC_LEN+1)


#define IDX1D(i,j,w) (((i)*(w))+(j))
#define modulo_f(a,b) fmod(fmod(a,b)+(b),(b))


struct InputData {
    cuFloatComplex *imgIn;
    cuFloatComplex *imgOut;
    float *residAz;
    float *residRg;
    double *azOffPoly;
    double *rgOffPoly;
    double *dopPoly;
    double *azCarrierPoly;
    double *rgCarrierPoly;
    float *fintp;
};

__constant__ double ind[6];
__constant__ int ini[8];

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//              GPU Helper Functions
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// Data usage: 8 floats/pointers, 2 ints   --      72 bytes/call
__device__ double evalPolyAt(double *polyArr, double azi, double rng) {
    // C-style eval method of Poly2d (adjusted to work with the array-format Poly2d where:
    //  polyArr[0] = azimuthOrder
    //  polyArr[1] = rangeOrder
    //  polyArr[2] = azimuthMean
    //  polyArr[3] = rangeMean
    //  polyArr[4] = azimuthNorm
    //  polyArr[5] = rangeNorm
    //  polyArr[6...] = coeffs (len ([0]+1)*([1]+1))
    // Therefore we can guarantee that polyArr has at least 7 elements, and intuitively stores its own length using the orders

    double val, scalex, scaley, xval, yval;
    int i, j;
    val = 0.;
    scaley = 1.;
    xval = (rng - polyArr[3]) / polyArr[5];
    yval = (azi - polyArr[2]) / polyArr[4];
    for (i=0; i<=polyArr[0]; i++,scaley*=yval) {
        scalex = 1.;
        for (j=0; j<=polyArr[1]; j++,scalex*=xval) {
            val += scalex * scaley * polyArr[IDX1D(i,j,int(polyArr[1])+1)+6];
        }
    }
    return val;
}

__global__ void removeCarrier(struct InputData inData) {
    // remove the carriers from input slc
    // thread id, as the pixel index for the input image
    int pix = blockDim.x * blockIdx.x + threadIdx.x;
    // check the thread range
    // ini[0] - inLength
    // ini[1] - inWidth
    if(pix >= ini[0]*ini[1])
        return;

    // get pixel location along azimuth/range
    int idxi = pix/ini[1];
    int idxj = pix%ini[1];

    // the poly uses fortran 1-indexing
    double r_i = idxi +1;
    double r_j = idxj +1;
    // get the phase shift due to carriers
    double ph =  evalPolyAt(inData.rgCarrierPoly, r_i, r_j) +
        evalPolyAt(inData.azCarrierPoly, r_i, r_j);
    ph = modulo_f(ph, 2.*M_PI);
    // remove the phase shift from the data
    cuFloatComplex cval = cuCmulf(inData.imgIn[pix], make_cuFloatComplex(cosf(ph), -sinf(ph)));
    // assign the new value
    inData.imgIn[pix] = cval;
    // all done
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//              GPU Main Kernel
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// Data Usage: 15 pointers/floats, 5 ints, 1 bool      --      144 bytes/call (assuming 1 bool ==> 1 int)
//             Add call to sinfc_interp (100 bytes/call) --      244 bytes/call (for funsies let's assume ~250 bytes/call)
// NOTE: We ignore calls to evalPolyAt sinfce they have less
//       data usage and therefore do not really matter for
//       max data usage
__global__ void GPUResamp(struct InputData inData) {
    // Main GPU ResampSlc kernel, slightly modified from original algorithm to save significant space

    int pix = blockDim.x * blockIdx.x + threadIdx.x;

    // check within outWidth*LINES_PER_TILE
    if (pix >= (ini[2] * ini[6]))
        return;

    // index along row/azimuth
    int idxi = (pix / ini[2]) + ini[4];
    // index along width/range
    int idxj = (pix % ini[2]);

    // offset
    // note that the polys use 1-indexing in Fortran code
    double ao = evalPolyAt(inData.azOffPoly, idxi+1, idxj+1) + inData.residAz[pix];
    double ro = evalPolyAt(inData.rgOffPoly, idxi+1, idxj+1) + inData.residRg[pix];

    // azimuth coordinate
    int ka = floor(idxi + ao);
    double fraca = idxi + ao - ka;
    // range coordinate
    int kr = floor(idxj + ro);
    double fracr = idxj + ro - kr;
    // check whether the pixel is out of the interpolation region
    if ((ka < SINC_HALF) || ( ka >= (ini[0]-SINC_HALF))
        || (kr < SINC_HALF) || (kr >= (ini[1]-SINC_HALF)))
    {
        // out of range
        inData.imgOut[pix] = make_cuFloatComplex(0., 0.);
        return;
    }

    // in range, continue

    // evaluate the doppler phase at the secondary coordinate
    double dop = evalPolyAt(inData.dopPoly, idxi+1+ao, idxj+1+ro);

    // phase corrections to be added later
    double ph = (dop * fraca) + evalPolyAt(inData.rgCarrierPoly, idxi+1+ao, idxj+1+ro) +
        evalPolyAt(inData.azCarrierPoly, idxi+1+ao, idxj+1+ro);

    // if flatten
    if (ini[7] == 1)
        ph = ph + ((4.*(M_PI/ind[0]))*((ind[2]-ind[3])+(idxj*(ind[4]-ind[5]))+(ro*ind[4])))
            +((4.*M_PI*(ind[3]+(idxj*ind[5])))*((1./ind[1])-(1./ind[0])));

    ph = modulo_f(ph, 2.*M_PI);

    // temp variable to keep track of the interpolated value
    cuFloatComplex cval = make_cuFloatComplex(0.,0.);
    // get the indices in the sinfc_coef of the fractional parts
    int ifraca = int(fraca*SINC_SUB);
    int ifracr = int(fracr*SINC_SUB);

    // weight for sinfc interp coefficients
    float weightsum = 0.;

    // iterate over the interpolation zone, e.g.  [-3, 4] x [-3, 4] for SINC_LEN = 8
    for (int i=-SINC_HALF+1; i<=SINC_HALF; i++) {
        cuFloatComplex cdop = make_cuFloatComplex(cosf(i*dop), -sinf(i*dop));
        for (int j=-SINC_HALF+1; j<=SINC_HALF; j++) {
            float weight = inData.fintp[IDX1D(ifraca,SINC_HALF-i,SINC_LEN)]
                            *inData.fintp[IDX1D(ifracr,SINC_HALF-j,SINC_LEN)];
            // correct the doppler phase here
            cuFloatComplex cin = cuCmulf(inData.imgIn[IDX1D(i+ka,j+kr,ini[1])], cdop);
            cval = cuCaddf(cval, make_cuFloatComplex(cuCrealf(cin)*weight, cuCimagf(cin)*weight));
            weightsum += weight;
        }
    }
    // normalize
    cval = make_cuFloatComplex(cuCrealf(cval)/weightsum, cuCimagf(cval)/weightsum);
    // phase correction
    cval = cuCmulf(cval, make_cuFloatComplex(cosf(ph), sinf(ph)));
    // assign and return
    inData.imgOut[pix] = cval;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//              CPU Helper Functions
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

double cpuSecond() {

    struct timeval tp;
    gettimeofday(&tp,NULL);
    return (double(tp.tv_sec) + double(tp.tv_usec)*1.e-6);
}

void checkKernelErrors() {

    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();

    if (errSync != cudaSuccess) printf("\nSync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess) printf("\nAsync kernel error: %s\n", cudaGetErrorString(errAsync));
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//              Main CPU Function
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

void runGPUResamp(double *h_inpts_dbl, int *h_inpts_int, void *imgIn, void *imgOut,
                    float *residAz, float *residRg, double *azOffPoly, double *rgOffPoly,
                    double *dopPoly, double *azCarrierPoly, double *rgCarrierPoly, float *fintp)
{
    /* * * * * * * * * * * * * * * * * * * *
     * Input mapping -
     *
     *  Double 0 - wvl
     *  Double 1 - refwvl
     *  Double 2 - r0
     *  Double 3 - refr0
     *  Double 4 - slr
     *  Double 5 - refslr
     *
     *  Int 0 - inLength
     *  Int 1 - inWidth
     *  Int 2 - outWidth
     *  Int 3 - firstImageRow
     *  Int 4 - firstTileRow
     *  Int 5 - nRowsInBlock
     *  Int 6 - LINES_PER_TILE
     *  Int 7 - flatten
     *
     * * * * * * * * * * * * * * * * * * * */

    // Casting input/output images to native cuFloatComplex type from complex<float>
    cuFloatComplex *h_imgIn = (cuFloatComplex *)imgIn;
    cuFloatComplex *h_imgOut = (cuFloatComplex *)imgOut;

    // Create handles for device copies of inputs
    cuFloatComplex *d_imgIn, *d_imgOut;
    float *d_residAz, *d_residRg;
    double *d_azOffPoly, *d_rgOffPoly, *d_dopPoly, *d_azCarrierPoly, *d_rgCarrierPoly;
    float *d_fintp;

    double startRun, endRun, startKernel, endKernel;

    struct InputData inData;


    printf("\n  Initializing GPU ResampSlc\n");
    cudaSetDevice(0);

    startRun = cpuSecond();

    printf("    Allocating initial memory... ");
    fflush(stdout);

    int nInPix = h_inpts_int[5] * h_inpts_int[1];
    int nOutPix = h_inpts_int[6] * h_inpts_int[2];
    int nResidAzPix = 0;
    if (residAz != 0) nResidAzPix = h_inpts_int[6] * h_inpts_int[2];
    int nResidRgPix = 0;
    if (residRg != 0) nResidRgPix = h_inpts_int[6] * h_inpts_int[2];
    int nAzOffPix = ((azOffPoly[0]+1) * (azOffPoly[1]+1)) + 6; // [0] and [1] of the Poly2d arrays hold the az/rg orders
    int nRgOffPix = ((rgOffPoly[0]+1) * (rgOffPoly[1]+1)) + 6;
    int nDopPix = ((dopPoly[0]+1) * (dopPoly[1]+1)) + 6;
    int nAzCarryPix = ((azCarrierPoly[0]+1) * (azCarrierPoly[1]+1)) + 6;
    int nRgCarryPix = ((rgCarrierPoly[0]+1) * (rgCarrierPoly[1]+1)) + 6;

    size_t nb_in = nInPix * sizeof(cuFloatComplex);
    size_t nb_out = nOutPix * sizeof(cuFloatComplex);
    size_t nb_rsdAz = nResidAzPix * sizeof(float);
    size_t nb_rsdRg = nResidRgPix * sizeof(float);
    size_t nb_azOff = nAzOffPix * sizeof(double);
    size_t nb_rgOff = nRgOffPix * sizeof(double);
    size_t nb_dop = nDopPix * sizeof(double);
    size_t nb_azCarry = nAzCarryPix * sizeof(double);
    size_t nb_rgCarry = nRgCarryPix * sizeof(double);

    cudaMalloc((cuFloatComplex**)&d_imgIn, nb_in);
    cudaMalloc((cuFloatComplex**)&d_imgOut, nb_out);
    if (residAz != 0) cudaMalloc((float**)&d_residAz, nb_rsdAz);
    if (residRg != 0) cudaMalloc((float**)&d_residRg, nb_rsdRg);
    cudaMalloc((double**)&d_azOffPoly, nb_azOff);
    cudaMalloc((double**)&d_rgOffPoly, nb_rgOff);
    cudaMalloc((double**)&d_dopPoly, nb_dop);
    cudaMalloc((double**)&d_azCarrierPoly, nb_azCarry);
    cudaMalloc((double**)&d_rgCarrierPoly, nb_rgCarry);
    cudaMalloc((float**)&d_fintp, (SINC_LEN*SINC_SUB*sizeof(float)));

    printf("Done.\n    Copying data to GPU... ");
    fflush(stdout);

    startKernel = cpuSecond();

    cudaMemcpy(d_imgIn, h_imgIn, nb_in, cudaMemcpyHostToDevice);
    if (residAz != 0) cudaMemcpy(d_residAz, residAz, nb_rsdAz, cudaMemcpyHostToDevice);
    if (residRg != 0) cudaMemcpy(d_residRg, residRg, nb_rsdRg, cudaMemcpyHostToDevice);
    cudaMemcpy(d_azOffPoly, azOffPoly, nb_azOff, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rgOffPoly, rgOffPoly, nb_rgOff, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dopPoly, dopPoly, nb_dop, cudaMemcpyHostToDevice);
    cudaMemcpy(d_azCarrierPoly, azCarrierPoly, nb_azCarry, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rgCarrierPoly, rgCarrierPoly, nb_rgCarry, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fintp, fintp, (SINC_LEN*SINC_SUB*sizeof(float)), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(ind, h_inpts_dbl, (6*sizeof(double)));
    cudaMemcpyToSymbol(ini, h_inpts_int, (8*sizeof(int)));

    cudaMemset(d_imgOut, 0, nb_out);

    endKernel = cpuSecond();

    printf("Done. (%f s.)\n", (endKernel-startKernel));


    printf("    Running GPU ResampSlc... ");
    fflush(stdout);

    startKernel = cpuSecond();

    inData.imgIn = d_imgIn;
    inData.imgOut = d_imgOut;
    inData.residAz = 0;
    if (residAz != 0) inData.residAz = d_residAz;
    inData.residRg = 0;
    if (residRg != 0) inData.residRg = d_residRg;
    inData.azOffPoly = d_azOffPoly;
    inData.rgOffPoly = d_rgOffPoly;
    inData.dopPoly = d_dopPoly;
    inData.azCarrierPoly = d_azCarrierPoly;
    inData.rgCarrierPoly = d_rgCarrierPoly;
    inData.fintp = d_fintp;

   // remove carriers from the input image
    int threads = 1024;
    int blocks = (nInPix + threads-1) / threads;
    removeCarrier<<<blocks, threads>>>(inData);
    checkKernelErrors();
    // resample
    blocks = (nOutPix + threads -1) / threads;
    GPUResamp <<<blocks, threads>>>(inData);
    checkKernelErrors();

    endKernel = cpuSecond();

    printf("Done. (%f s.)\n", (endKernel-startKernel));

    printf("    Copying memory back to host... ");
    fflush(stdout);

    startKernel = cpuSecond();

    cudaMemcpy(h_imgOut, d_imgOut, nb_out, cudaMemcpyDeviceToHost);

    endKernel = cpuSecond();
    endRun = cpuSecond();

    printf("Done. (%f s.)\n", (endKernel-startKernel));
    printf("    Finished GPU ResampSlc in %f s.\n", (endRun-startRun));
    printf("    Cleaning device memory and returning to main ResampSlc function...\n");

    cudaFree(d_imgIn);
    cudaFree(d_imgOut);
    if (residAz != 0) cudaFree(d_residAz);
    if (residRg != 0) cudaFree(d_residRg);
    cudaFree(d_azOffPoly);
    cudaFree(d_rgOffPoly);
    cudaFree(d_dopPoly);
    cudaFree(d_azCarrierPoly);
    cudaFree(d_rgCarrierPoly);
    cudaFree(d_fintp);
    cudaDeviceReset();

    printf("  Exiting GPU ResampSlc\n\n");
}
