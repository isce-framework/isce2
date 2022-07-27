//
// Author: Joshua Cohen
// Copyright 2017
//

#include <complex>
#include <cstdio>
#include <vector>
#include "Interpolator.h"
#include "Constants.h"
#include "ResampMethods.h"

using std::complex;
using std::vector;

ResampMethods::ResampMethods() {
    return;
}

void ResampMethods::prepareMethods(int method) {
    if (method == SINC_METHOD) {
        vector<double> filter(SINC_SUB*SINC_LEN);
        double ssum;
        int intplength, filtercoef;
        Interpolator interp;

        printf("Initializing Sinc interpolator\n");

        interp.sinc_coef(1.,SINC_LEN,SINC_SUB,0.,1,intplength,filtercoef,filter);

        // note also the type conversion
        fintp.resize(SINC_SUB*SINC_LEN);
        for (int i=0; i<SINC_LEN; i++) {
            for (int j=0; j<SINC_SUB; j++) {
                fintp[i+(j*SINC_LEN)] = filter[j+(i*SINC_SUB)];
            }
        }
        f_delay = SINC_LEN / 2.;
    } else {
        printf("Error: Other interpolation methods for ResampSlc not implemented yet.\n");
    }
    /*
    else if (method == BILINEAR_METHOD) {
        printf("Initializing Bilinear interpolator\n");
        f_delay = 2.;

    } else if (method == BICUBIC_METHOD) {
        printf("Initializing Bicubic interpolator\n");
        f_delay = 3.;

    } else if (method == NEAREST_METHOD) {
        printf("Initializing Nearest Neighbor interpolator\n");
        f_delay = 2.;

    } else if (method == AKIMA_METHOD) {
        printf("Initializing Akima interpolator\n");
        f_delay = 2.;

    } else if (method == BIQUINTIC_METHOD) {
        printf("Initializing Biquintic interpolator\n");
        f_delay = 3.;

    } else {
        printf("Error in ResampMethods::prepareMethods - Unknown interpolation method (received %d)\n", method);
        exit(1);
    }
    */
}

complex<float> ResampMethods::interpolate_cx(vector<vector<complex<float> > > &ifg, int x, int y, double fx, double fy, int nx, int ny, int method) {
    int xx, yy;
    Interpolator interp;

    if (method != SINC_METHOD) {
        printf("Error in ResampMethods::interpolate_cx - invalid interpolation method; interpolate_cx only performs a sinc interpolation currently\n");
        return complex<float>(0.,0.);
    }

    if ((x < SINC_HALF) || (x > (nx-SINC_HALF))) return complex<float>(0.,0.);
    if ((y < SINC_HALF) || (y > (ny-SINC_HALF))) return complex<float>(0.,0.);

    xx = x + SINC_HALF - 1;
    yy = y + SINC_HALF - 1;

    return interp.sinc_eval_2d(ifg,fintp,SINC_SUB,SINC_LEN,xx,yy,fx,fy,nx,ny);
}

