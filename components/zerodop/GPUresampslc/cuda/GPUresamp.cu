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
    double *residAz;
    double *residRg;
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

// Data usage: 8 doubles/pointers, 2 ints   --      72 bytes/call
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

// Data usage: 4 doubles, 1 int     --      36 bytes/call
__device__ cuFloatComplex fintp(int idx) {
    // Replaces storing ~4MB and copying in/out of constant storage by calculating an element of fintp on the fly
    
    double weight, filtFact, sinFact, i;
    i = int(int(idx / SINC_LEN) + ((idx % SINC_LEN) * (SINC_SUB / 2)) - 1) - 16383.5;
    weight = .5 + (.5 * cos((M_PI * i) / 16383.5));
    sinFact = i * (.75 / (SINC_SUB / 2));
    filtFact = (sin(M_PI * sinFact) / (M_PI * sinFact)) * weight; // No need to check if sinFact is 0 since SINC_SUB != 0 and i-16383.5 != 0 as i is an int
    return make_cuFloatComplex(filtFact*weight, 0.);
}

// Data usage: 6 double/complex/pointers, 4 ints    --      64 bytes/call
//             Add call to fintp (36 bytes)         --      100 bytes/call
__device__ cuFloatComplex sinc_interp(cuFloatComplex *chip, double fraca, double fracr, double dop, float *fintp) {
    // This is a modified/hardwired form of sinc_eval_2d from Interpolator. We eliminate a couple of checks that we know will pass, and
    // adjusted the algorithm to account for modifications to the main kernel below. Primarily three things are of interest:
    //      1. Chip is actually a pointer to the top-left of the chip location in the main image block, so chip's 'width' actually is
    //         ini[1], not SINC_ONE.
    //      2. We account for removing doppler effects using cval and taking in dop. In the older version of the main kernel, this
    //         value was calculated and multiplied as the data was copied into the smaller chip. Here we calculate it on the fly using
    //         the same index for 'ii' as the row index of the chip being operated on (so in this case since it happens from the bottom
    //         up, we use SINC_LEN-i instead of i).

    cuFloatComplex ret = make_cuFloatComplex(0.,0.);
    cuFloatComplex cval;
    int ifracx, ifracy, i, j;

    ifracx = min(max(0,int(fraca*SINC_SUB)), SINC_SUB-1);
    ifracy = min(max(0,int(fracr*SINC_SUB)), SINC_SUB-1);
    for (i=0; i<SINC_LEN; i++) {
        cval = make_cuFloatComplex(cos((SINC_LEN-i-4.)*dop), -sin((SINC_LEN-i-4.)*dop));
        for (j=0; j<SINC_LEN; j++) {
            ret = cuCaddf(ret, 
                    cuCmulf(
                        cuCmulf(chip[IDX1D(SINC_LEN-i,SINC_LEN-j,ini[1])], cval), 
                        make_cuFloatComplex(fintp[IDX1D(ifracx,i,SINC_LEN)]*fintp[IDX1D(ifracy,j,SINC_LEN)], 0.)
                    )
                  );
        }
    }

    return ret;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//              GPU Main Kernel
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// Data Usage: 15 pointers/doubles, 5 ints, 1 bool      --      144 bytes/call (assuming 1 bool ==> 1 int)
//             Add call to sinc_interp (100 bytes/call) --      244 bytes/call (for funsies let's assume ~250 bytes/call)
// NOTE: We ignore calls to evalPolyAt since they have less
//       data usage and therefore do not really matter for
//       max data usage
__global__ void GPUResamp(struct InputData inData) {
    // Main GPU ResampSlc kernel, slightly modified from original algorithm to save significant space

    int pix = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (pix < (ini[2] * ini[6])) {
        //cuFloatComplex chip[SINC_ONE*SINC_ONE];
        cuFloatComplex cval;
        double ao, ro, fracr, fraca, ph, dop;
        int k, kk, idxi, idxj; //chipi, chipj, ii, jj;
        bool flag;

        flag = false;
        idxi = (pix / ini[2]) + ini[4];
        idxj = (pix % ini[2]);

        ao = evalPolyAt(inData.azOffPoly, idxi+1, idxj+1);
        ro = evalPolyAt(inData.rgOffPoly, idxi+1, idxj+1);

        k = idxi + ao;
        fraca = idxi + ao - k;
        if ((k < SINC_HALF) || (k >= (ini[0]-SINC_HALF))) flag = true;
        
        if (!flag) {
            kk = idxj + ro;
            fracr = idxj + ro - kk;
            if ((kk < SINC_HALF) || (kk >= (ini[1]-SINC_HALF))) flag = true;

            if (!flag) {
                dop = evalPolyAt(inData.dopPoly, idxi+1, idxj+1);

                /*
                for (ii=0; ii<SINC_ONE; ii++) {
                    chipi = k - ini[3] + ii - SINC_HALF;
                    cval = make_cuFloatComplex(cos((ii-4.)*dop), -sin((ii-4.)*dop));
                    for (jj=0; jj<SINC_ONE; jj++) {
                        chipj = kk + jj - SINC_HALF;
                        chip[IDX1D(ii,jj,SINC_ONE)] = cuCmulf(inData.imgIn[IDX1D(chipi,chipj,ini[1])], cval);
                    }
                }

                top = k - ini[3] - SINC_HALF;
                left = kk - SINC_HALF;
                topLeft = IDX1D(top,left,ini[1]);
                */

                ph = (dop * fraca) + evalPolyAt(inData.rgCarrierPoly, idxi+ao, idxj+ro) + evalPolyAt(inData.azCarrierPoly, idxi+ao, idxj+ro);

                if (ini[7] == 1)
                    ph = ph + ((4.*(M_PI/ind[0]))*((ind[2]-ind[3])+(idxj*(ind[4]-ind[5]))+(ro*ind[4])))+((4.*M_PI*(ind[3]+(idxj*ind[5])))*((1./ind[1])-(1./ind[0])));

                ph = modulo_f(ph, 2.*M_PI);
                // NOTE: This has been modified to pass the pointer to the "top left" location of what used to be the 'chip' copy of data from imgIn. This
                //       saves allocating 81 extra floats per pixel (since chip is SINC_ONE*SINC_ONE), and instead uses logic to determine the offsets. The
                //       logic simply takes the minimum values of chipi and chipj above and adds the IDX1D index to the pointer. This means that sinc_interp
                //       will pass this pointer as "chip" and it will still point to the correct values (since chip is just a 2D window subset of imgIn). We
                //       Also have to account for 'cval' by modifying the sinc_interp function to calculate 'cval' dynamically (no extra cost computationally).
                //       This should actually speed up the kernel as well as significantly reduce redundant data usage.
                cval = sinc_interp((inData.imgIn + IDX1D(k-ini[3]-SINC_HALF,kk-SINC_HALF,ini[1])), fraca, fracr, dop, inData.fintp);
                inData.imgOut[pix] = cuCmulf(cval, make_cuFloatComplex(cos(ph), sin(ph)));
            }
        }
    }
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

void runGPUResamp(double *h_inpts_dbl, int *h_inpts_int, void *imgIn, void *imgOut, double *residAz, double *residRg, double *azOffPoly, double *rgOffPoly,
                    double *dopPoly, double *azCarrierPoly, double *rgCarrierPoly, float *fintp) {
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
    double *d_residAz, *d_residRg, *d_azOffPoly, *d_rgOffPoly, *d_dopPoly, *d_azCarrierPoly, *d_rgCarrierPoly;
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
    size_t nb_rsdAz = nResidAzPix * sizeof(double);
    size_t nb_rsdRg = nResidRgPix * sizeof(double);
    size_t nb_azOff = nAzOffPix * sizeof(double);
    size_t nb_rgOff = nRgOffPix * sizeof(double);
    size_t nb_dop = nDopPix * sizeof(double);
    size_t nb_azCarry = nAzCarryPix * sizeof(double);
    size_t nb_rgCarry = nRgCarryPix * sizeof(double);
    
    cudaMalloc((cuFloatComplex**)&d_imgIn, nb_in);
    cudaMalloc((cuFloatComplex**)&d_imgOut, nb_out);
    if (residAz != 0) cudaMalloc((double**)&d_residAz, nb_rsdAz);
    if (residRg != 0) cudaMalloc((double**)&d_residRg, nb_rsdRg);
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
    
    dim3 block(32);
    dim3 grid(int((nInPix + 31) / 32));
    if ((grid.x * 32) > nInPix) printf("    (DEBUG: There will be %d 'empty' threads in the last thread block).\n", ((grid.x*32)-nInPix));

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

    GPUResamp <<<grid, block>>>(inData);
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
