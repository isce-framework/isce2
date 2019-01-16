//
// Author: Joshua Cohen
// Copyright 2016
//
// For later integration/release this should probably be renamed to something more appropriate...
// (to-do: either rename the module to DemInterp or wrap these into UniformInterp)
// Actually, should this be wrapped into the UniformInterp struct? It basically is just a fancy
// wrapper for those methods anyway...

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "AkimaLib.h"
#include "Constants.h"
#include "TopoMethods.h"
#include "UniformInterp.h"
using std::vector;

// Default constructor
TopoMethods::TopoMethods() :
    fintp(0) {
    f_delay = 0.0;
}

// Copy constructor
TopoMethods::TopoMethods(const TopoMethods &tm) {
    fintp = tm.fintp;   // Uses vector's copy constructor
    f_delay = tm.f_delay;
}

void TopoMethods::prepareMethods(int method) {
    int intplength,filtercoef;
    
    if (method == SINC_METHOD) {
        printf("Initializing Sinc interpolator...\n");
        vector<double> r_filter(SINC_SUB*SINC_LEN+1);
        fintp.resize(SINC_SUB*SINC_LEN);
        
        UniformInterp uinterp;
        uinterp.sinc_coef(1.0,(1.0*SINC_LEN),SINC_SUB,0.0,1,intplength,filtercoef,r_filter);

        for (int i=0; i<SINC_LEN; i++) {
            for (int j=0; j<SINC_SUB; j++) {
                fintp[i+(j*SINC_LEN)] = r_filter[j+(i*SINC_SUB)];
            }
        }
        f_delay = SINC_LEN / 2.0;
    } else if (method == BILINEAR_METHOD) {
        printf("Initializing Bilinear interpolator...\n");
        f_delay = 2.0;
    } else if (method == BICUBIC_METHOD) {
        printf("Initializing Bicubic interpolator...\n");
        f_delay = 3.0;
    } else if (method == NEAREST_METHOD) {
        printf("Initializing Nearest Neighbor interpolator...\n");
        f_delay = 2.0;
    } else if (method == AKIMA_METHOD) {
        printf("Initializing Akima interpolator...\n");
        f_delay = 2.0;
    } else if (method == BIQUINTIC_METHOD) {
        printf("Initializing Biquintic interpolator...\n");
        f_delay = 3.0;
    } else {
        printf("Error in TopoMethods::prepareMethods - Unknown method type.\n");
        exit(1);
    }
}

// Single DEM interpolator wrapper function
float TopoMethods::interpolate(vector<vector<float> > &dem, int i_x, int i_y, double f_x, double f_y, int nx, int ny, int method) {
    if (method == 0) return intp_sinc(dem,i_x,i_y,f_x,f_y,nx,ny);
    if (method == 1) return intp_bilinear(dem,i_x,i_y,f_x,f_y,nx,ny);
    if (method == 2) return intp_bicubic(dem,i_x,i_y,f_x,f_y,nx,ny);
    if (method == 3) return intp_nearest(dem,i_x,i_y,f_x,f_y,nx,ny);
    if (method == 4) return intp_akima(dem,i_x,i_y,f_x,f_y,nx,ny);
    if (method == 5) return intp_biquintic(dem,i_x,i_y,f_x,f_y,nx,ny);
    else {
        printf("Error in TopoMethods::interpolate - Invalid interpolation method (%d)\n",method);
        exit(1);
    }
    return 0.0; // Never hit, but needed to satisfy compiler
}

float TopoMethods::intp_sinc(vector<vector<float> > &dem, int i_x, int i_y, double f_x, double f_y, int nx, int ny) {
    float ret;
    int i_xx,i_yy;

    if ((i_x < 4) || (i_x > (nx-3))) {
        ret = BADVALUE;
        return ret;
    }
    if ((i_y < 4) || (i_y > (ny-3))) {
        ret = BADVALUE;
        return ret;
    }
    i_xx = i_x + (SINC_LEN / 2);
    i_yy = i_y + (SINC_LEN / 2);

    UniformInterp uinterp;
    ret = uinterp.sinc_eval_2d<float,float,float>(dem,fintp,SINC_SUB,SINC_LEN,i_xx,i_yy,f_x,f_y,nx,ny);
    return ret;
}

float TopoMethods::intp_bilinear(vector<vector<float> > &dem, int i_x, int i_y, double f_x, double f_y, int nx, int ny) {
    double dx,dy,temp;
    float ret;

    dx = i_x + f_x;
    dy = i_y + f_y;

    if ((i_x < 1) || (i_x >= nx)) {
        ret = BADVALUE;
        return ret;
    }
    if ((i_y < 1) || (i_y >= ny)) {
        ret = BADVALUE;
        return ret;
    }

    UniformInterp uinterp;
    temp = uinterp.bilinear<double,float>(dy,dx,dem); // Explicit template call is a little safer (and compiler has trouble
                                                      // identifying this one)
    ret = float(temp); // Not entirely sure why it's being down-cast to float from double, but keeping with the original code
    return ret;
}

float TopoMethods::intp_bicubic(vector<vector<float> > &dem, int i_x, int i_y, double f_x, double f_y, int nx, int ny) {
    double dx,dy,temp;
    float ret;

    dx = i_x + f_x;
    dy = i_y + f_y;

    if ((i_x < 2) || (i_x >= (nx-1))) {
        ret = BADVALUE;
        return ret;
    }
    if ((i_y < 2) || (i_y >= (ny-1))) {
        ret = BADVALUE;
        return ret;
    }

    UniformInterp uinterp;
    temp = uinterp.bicubic<double,float>(dy,dx,dem);
    ret = float(temp);
    return ret;
}

float TopoMethods::intp_biquintic(vector<vector<float> > &dem, int i_x, int i_y, double f_x, double f_y, int nx, int ny) {
    double dx,dy;
    float ret;

    dx = i_x + f_x;
    dy = i_y + f_y;

    if ((i_x < 3) || (i_x >= (nx-2))) {
        ret = BADVALUE;
        return ret;
    }
    if ((i_y < 3) || (i_y >= (ny-2))) {
        ret = BADVALUE;
        return ret;
    }
    
    UniformInterp uinterp;
    ret = uinterp.interp2DSpline(6,ny,nx,dem,dy,dx);
    return ret;
}

float TopoMethods::intp_nearest(vector<vector<float> > &dem, int i_x, int i_y, double f_x, double f_y, int nx, int ny) {
    float ret;
    int dx,dy;

    dx = round(i_x + f_x);
    dy = round(i_y + f_y);

    if ((dx < 1) || (dx > nx)) {
        ret = BADVALUE;
        return ret;
    }
    if ((dy < 1) || (dy > ny)) {
        ret = BADVALUE;
        return ret;
    }
    ret = dem[dx-1][dy-1];
    return ret;
}

float TopoMethods::intp_akima(vector<vector<float> > &dem, int i_x, int i_y, double f_x, double f_y, int nx, int ny) {
    double dx,dy,temp;
    vector<double> poly(AKI_NSYS);
    float ret;

    dx = i_x + f_x;
    dy = i_y + f_y;

    if ((i_x < 1) || (i_x >= (nx-1))) {
        ret = BADVALUE;
        return ret;
    }
    if ((i_y < 1) || (i_y >= (ny-1))) {
        ret = BADVALUE;
        return ret;
    }

    AkimaLib aklib;
    aklib.polyfitAkima(nx,ny,dem,i_x,i_y,poly);
    temp = aklib.polyvalAkima(i_x,i_y,dx,dy,poly);
    ret = float(temp);
    return ret;
}

