//
// Author: Joshua Cohen
// Copyright 2016
//

// update: updated to use long for some integers associated with file size to support large images.
//         Cunren Liang, 26-MAR-2018


#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>

#define THRD_PER_BLOCK 96   // Number of threads per block (should always %32==0)

// --------------- STRUCTS ------------------

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

struct OutputImgArrs {
    double *lat;
    double *lon;
    double *z;
    //double *zsch;
    double *losang;
    double *incang;
};

struct InputImgArrs {
    double *rho;
    double *dopline;
    float *DEM;
};

struct Ellipsoid {
    double a;
    double e2;
};

struct Peg {
    double lat;
    double lon;
    double hdg;
};

struct PegTrans {
    double mat[3][3];
    double ov[3];
    double radcur;
};

// Constant memory is ideal for const input values
__constant__ double d_inpts_dbl[14];
__constant__ int d_inpts_int[7];

// --------------- GPU HELPER FUNCTIONS ----------------

__device__ int interpolateOrbit(struct Orbit *orb, double t, double *xyz, double *vel) { //, int method) {
    double h[4], hdot[4], f0[4], f1[4], g0[4], g1[4];
    double sum = 0.0;
    int v0 = -1;
   
    if ((t < orb->svs[0].t) || (t > orb->svs[orb->nVec-1].t)) return 1;
    for (int i=0; i<orb->nVec; i++) {
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

    sum = ((t - orb->svs[v0+2].t) / (orb->svs[v0].t - orb->svs[v0+2].t)) * ((t - orb->svs[v0+3].t) / (orb->svs[v0].t - orb->svs[v0+3].t)) * 
            (1.0 / (orb->svs[v0].t - orb->svs[v0+1].t));
    sum += ((t - orb->svs[v0+1].t) / (orb->svs[v0].t - orb->svs[v0+1].t)) * ((t - orb->svs[v0+3].t) / (orb->svs[v0].t - orb->svs[v0+3].t)) * 
            (1.0 / (orb->svs[v0].t - orb->svs[v0+2].t));
    sum += ((t - orb->svs[v0+1].t) / (orb->svs[v0].t - orb->svs[v0+1].t)) * ((t - orb->svs[v0+2].t) / (orb->svs[v0].t - orb->svs[v0+2].t)) * 
            (1.0 / (orb->svs[v0].t - orb->svs[v0+3].t));
    hdot[0] = sum;

    sum = ((t - orb->svs[v0+2].t) / (orb->svs[v0+1].t - orb->svs[v0+2].t)) * ((t - orb->svs[v0+3].t) / (orb->svs[v0+1].t - orb->svs[v0+3].t)) *
            (1.0 / (orb->svs[v0+1].t - orb->svs[v0].t));
    sum += ((t - orb->svs[v0].t) / (orb->svs[v0+1].t - orb->svs[v0].t)) * ((t - orb->svs[v0+3].t) / (orb->svs[v0+1].t - orb->svs[v0+3].t)) * 
            (1.0 / (orb->svs[v0+1].t - orb->svs[v0+2].t));
    sum += ((t - orb->svs[v0].t) / (orb->svs[v0+1].t - orb->svs[v0].t)) * ((t - orb->svs[v0+2].t) / (orb->svs[v0+1].t - orb->svs[v0+2].t)) * 
            (1.0 / (orb->svs[v0+1].t - orb->svs[v0+3].t));
    hdot[1] = sum;

    sum = ((t - orb->svs[v0+1].t) / (orb->svs[v0+2].t - orb->svs[v0+1].t)) * ((t - orb->svs[v0+3].t) / (orb->svs[v0+2].t - orb->svs[v0+3].t)) * 
            (1.0 / (orb->svs[v0+2].t - orb->svs[v0].t));
    sum += ((t - orb->svs[v0].t) / (orb->svs[v0+2].t - orb->svs[v0].t)) * ((t - orb->svs[v0+3].t) / (orb->svs[v0+2].t - orb->svs[v0+3].t)) * 
            (1.0 / (orb->svs[v0+2].t - orb->svs[v0+1].t));
    sum += ((t - orb->svs[v0].t) / (orb->svs[v0+2].t - orb->svs[v0].t)) * ((t - orb->svs[v0+1].t) / (orb->svs[v0+2].t - orb->svs[v0+1].t)) * 
            (1.0 / (orb->svs[v0+2].t - orb->svs[v0+3].t));
    hdot[2] = sum;

    sum = ((t - orb->svs[v0+1].t) / (orb->svs[v0+3].t - orb->svs[v0+1].t)) * ((t - orb->svs[v0+2].t) / (orb->svs[v0+3].t - orb->svs[v0+2].t)) * 
            (1.0 / (orb->svs[v0+3].t - orb->svs[v0].t));
    sum += ((t - orb->svs[v0].t) / (orb->svs[v0+3].t - orb->svs[v0].t)) * ((t - orb->svs[v0+2].t) / (orb->svs[v0+3].t - orb->svs[v0+2].t)) * 
            (1.0 / (orb->svs[v0+3].t - orb->svs[v0+1].t));
    sum += ((t - orb->svs[v0].t) / (orb->svs[v0+3].t - orb->svs[v0].t)) * ((t - orb->svs[v0+1].t) / (orb->svs[v0+3].t - orb->svs[v0+1].t)) * 
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

    return 0;
}

__device__ void initSpline(double *A, double *R, double *Q) {
    Q[0] = 0.0;
    R[0] = 0.0;
    Q[1] = -0.5 / ((Q[0] / 2.) + 2.);
    R[1] = ((3. * (A[2] - (2. * A[1]) + A[0])) - (R[0] / 2.)) / ((Q[0] / 2.) + 2.);
    Q[2] = -0.5 / ((Q[1] / 2.) + 2.);
    R[2] = ((3. * (A[3] - (2. * A[2]) + A[1])) - (R[1] / 2.)) / ((Q[1] / 2.) + 2.);
    Q[3] = -0.5 / ((Q[2] / 2.) + 2.);
    R[3] = ((3. * (A[4] - (2. * A[3]) + A[2])) - (R[2] / 2.)) / ((Q[2] / 2.) + 2.);
    Q[4] = -0.5 / ((Q[3] / 2.) + 2.);
    R[4] = ((3. * (A[5] - (2. * A[4]) + A[3])) - (R[3] / 2.)) / ((Q[3] / 2.) + 2.);
    R[5] = 0.0;
    R[4] = (Q[4] * R[5]) + R[4];
    R[3] = (Q[3] * R[4]) + R[3];
    R[2] = (Q[2] * R[3]) + R[2];
    R[1] = (Q[1] * R[2]) + R[1];
}

// Note we're actually passing in the "length" variable, but width makes more sense in the algorithm
__device__ void spline(int indi, int j0, int width, double *A, float *DEM) {
    int indj;
    indj = min((j0+1),width);
    A[0] = DEM[((indi-1)*width)+(indj-1)];
    indj = min((j0+2),width);
    A[1] = DEM[((indi-1)*width)+(indj-1)];
    indj = min((j0+3),width);
    A[2] = DEM[((indi-1)*width)+(indj-1)];
    indj = min((j0+4),width);
    A[3] = DEM[((indi-1)*width)+(indj-1)];
    indj = min((j0+5),width);
    A[4] = DEM[((indi-1)*width)+(indj-1)];
    indj = min((j0+6),width);
    A[5] = DEM[((indi-1)*width)+(indj-1)];
}

__device__ double interpolateDEM(float *DEM, double lon, double lat, int width, int length) {
    bool out_of_bounds = ((int(lat) < 3) || (int(lat) >= (length-2)) || (int(lon) < 3) || (int(lon) >= (width-2)));
    if (out_of_bounds) return -500.0;

    double A[6], R[6], Q[6], HC[6];
    double t0, t1;
    int indi, i0, j0;

    i0 = int(lon) - 2;
    j0 = int(lat) - 2;
    
    indi = min((i0+1), width); // bound by out_of_bounds, so this isn't a concern
    spline(indi, j0, length, A, DEM);
    initSpline(A,R,Q);
    t0 = A[2] - A[1] - (R[1] / 3.) - (R[2] / 6.);
    t1 = (lat - j0 - 2.) * ((R[1] / 2.) + ((lat - j0 - 2.) * ((R[2] - R[1]) / 6.)));
    HC[0] = A[1] + ((lat - j0 - 2.) * (t0 + t1));

    indi = min((i0+2), width);
    spline(indi, j0, length, A, DEM);
    initSpline(A,R,Q);
    t0 = A[2] - A[1] - (R[1] / 3.) - (R[2] / 6.);
    t1 = (lat - j0 - 2.) * ((R[1] / 2.) + ((lat - j0 - 2.) * ((R[2] - R[1]) / 6.)));
    HC[1] = A[1] + ((lat - j0 - 2.) * (t0 + t1));

    indi = min((i0+3), width);
    spline(indi, j0, length, A, DEM);
    initSpline(A,R,Q);
    t0 = A[2] - A[1] - (R[1] / 3.) - (R[2] / 6.);
    t1 = (lat - j0 - 2.) * ((R[1] / 2.) + ((lat - j0 - 2.) * ((R[2] - R[1]) / 6.)));
    HC[2] = A[1] + ((lat - j0 - 2.) * (t0 + t1));

    indi = min((i0+4), width);
    spline(indi, j0, length, A, DEM);
    initSpline(A,R,Q);
    t0 = A[2] - A[1] - (R[1] / 3.) - (R[2] / 6.);
    t1 = (lat - j0 - 2.) * ((R[1] / 2.) + ((lat - j0 - 2.) * ((R[2] - R[1]) / 6.)));
    HC[3] = A[1] + ((lat - j0 - 2.) * (t0 + t1));

    indi = min((i0+5), width);
    spline(indi, j0, length, A, DEM);
    initSpline(A,R,Q);
    t0 = A[2] - A[1] - (R[1] / 3.) - (R[2] / 6.);
    t1 = (lat - j0 - 2.) * ((R[1] / 2.) + ((lat - j0 - 2.) * ((R[2] - R[1]) / 6.)));
    HC[4] = A[1] + ((lat - j0 - 2.) * (t0 + t1));

    indi = min((i0+6), width);
    spline(indi, j0, length, A, DEM);
    initSpline(A,R,Q);
    t0 = A[2] - A[1] - (R[1] / 3.) - (R[2] / 6.);
    t1 = (lat - j0 - 2.) * ((R[1] / 2.) + ((lat - j0 - 2.) * ((R[2] - R[1]) / 6.)));
    HC[5] = A[1] + ((lat - j0 - 2.) * (t0 + t1));

    initSpline(HC,R,Q);
    t0 = HC[2] - HC[1] - (R[1] / 3.) - (R[2] / 6.);
    t1 = (lon - i0 - 2.) * ((R[1] / 2.) + ((lon - i0 - 2.) * ((R[2] - R[1]) / 6.)));
    return HC[1] + ((lon - i0 - 2.) * (t0 + t1));
}

__device__ void unitvec(double *v, double *vhat) {
    double mag = norm(3,v);
    vhat[0] = v[0] / mag;
    vhat[1] = v[1] / mag;
    vhat[2] = v[2] / mag;
}

__device__ void cross(double *u, double *v, double *w) {
    w[0] = (u[1] * v[2]) - (u[2] * v[1]);
    w[1] = (u[2] * v[0]) - (u[0] * v[2]);
    w[2] = (u[0] * v[1]) - (u[1] * v[0]);
}

__device__ double dot(double *u, double *v) {
    return ((u[0]*v[0]) + (u[1]*v[1]) + (u[2]*v[2]));
}

__device__ void xyz2llh(double *xyz, double *llh, struct Ellipsoid *elp) {
    double d,k,p,q,r,rv,s,t,u,w;
    p = (pow(xyz[0],2) + pow(xyz[1],2)) / pow(elp->a,2);
    q = ((1.0 - elp->e2) * pow(xyz[2],2)) / pow(elp->a,2);
    r = (p + q - pow(elp->e2,2)) / 6.0;
    s = (pow(elp->e2,2) * p * q) / (4.0 * pow(r,3));
    t = cbrt(1.0 + s + sqrt(s * (2.0 + s)));
    //t = pow((1.0 + s + sqrt(s * (2.0 + s))),(1./3.));
    u = r * (1.0 + t + (1.0 / t));
    rv = sqrt(pow(u,2) + (pow(elp->e2,2) * q));
    w = (elp->e2 * (u + rv - q)) / (2.0 * rv);
    k = sqrt(u + rv + pow(w,2)) - w;
    d = (k * sqrt(pow(xyz[0],2) + pow(xyz[1],2))) / (k + elp->e2);
    llh[0] = atan2(xyz[2],d);
    llh[1] = atan2(xyz[1],xyz[0]);
    llh[2] = ((k + elp->e2 - 1.0) * sqrt(pow(d,2) + pow(xyz[2],2))) / k;
}

__device__ void llh2xyz(double *xyz, double *llh, struct Ellipsoid *elp) {
    double re;
    re = elp->a / sqrt(1.0 - (elp->e2 * pow(sin(llh[0]),2)));
    xyz[0] = (re + llh[2]) * cos(llh[0]) * cos(llh[1]);
    xyz[1] = (re + llh[2]) * cos(llh[0]) * sin(llh[1]);
    xyz[2] = ((re * (1.0 - elp->e2)) + llh[2]) * sin(llh[0]);
}

__device__ void tcnbasis(double *pos, double *vel, double *t, double *c, double *n, struct Ellipsoid *elp) {
    double llh[3], temp[3];
    xyz2llh(pos,llh,elp);
    n[0] = -cos(llh[0]) * cos(llh[1]);
    n[1] = -cos(llh[0]) * sin(llh[1]);
    n[2] = -sin(llh[0]);
    cross(n,vel,temp);
    unitvec(temp,c);
    cross(c,n,temp);
    unitvec(temp,t);
}

__device__ void radar2xyz(struct Peg *peg, struct Ellipsoid *elp, struct PegTrans *ptm) {
    double llh[3], temp[3];
    double re, rn;
    ptm->mat[0][0] = cos(peg->lat) * cos(peg->lon);
    ptm->mat[0][1] = (-sin(peg->hdg) * sin(peg->lon)) - (sin(peg->lat) * cos(peg->lon) * cos(peg->hdg));
    ptm->mat[0][2] = (sin(peg->lon) * cos(peg->hdg)) - (sin(peg->lat) * cos(peg->lon) * sin(peg->hdg));
    ptm->mat[1][0] = cos(peg->lat) * sin(peg->lon);
    ptm->mat[1][1] = (cos(peg->lon) * sin(peg->hdg)) - (sin(peg->lat) * sin(peg->lon) * cos(peg->hdg));
    ptm->mat[1][2] = (-cos(peg->lon) * cos(peg->hdg)) - (sin(peg->lat) * sin(peg->lon) * sin(peg->hdg));
    ptm->mat[2][0] = sin(peg->lat);
    ptm->mat[2][1] = cos(peg->lat) * cos(peg->hdg);
    ptm->mat[2][2] = cos(peg->lat) * sin(peg->hdg);
    
    re = elp->a / sqrt(1.0 - (elp->e2 * pow(sin(peg->lat),2)));
    rn = (elp->a * (1.0 - elp->e2)) / pow((1.0 - (elp->e2 * pow(sin(peg->lat),2))),1.5);
    ptm->radcur = (re * rn) / ((re * pow(cos(peg->hdg),2)) + (rn * pow(sin(peg->hdg),2)));

    llh[0] = peg->lat;
    llh[1] = peg->lon;
    llh[2] = 0.0;
    llh2xyz(temp,llh,elp);
   
    ptm->ov[0] = temp[0] - (ptm->radcur * cos(peg->lat) * cos(peg->lon));
    ptm->ov[1] = temp[1] - (ptm->radcur * cos(peg->lat) * sin(peg->lon));
    ptm->ov[2] = temp[2] - (ptm->radcur * sin(peg->lat));
}

__device__ void xyz2sch(double *schv, double *xyzv, struct PegTrans *ptm, struct Ellipsoid *elp) {
    double schvt[3], llh[3];
    double tempa, tempe2;
    schvt[0] = xyzv[0] - ptm->ov[0];
    schvt[1] = xyzv[1] - ptm->ov[1];
    schvt[2] = xyzv[2] - ptm->ov[2];
    schv[0] = (ptm->mat[0][0] * schvt[0]) + (ptm->mat[1][0] * schvt[1]) + (ptm->mat[2][0] * schvt[2]); // Switched from using ptm->matinv
    schv[1] = (ptm->mat[0][1] * schvt[0]) + (ptm->mat[1][1] * schvt[1]) + (ptm->mat[2][1] * schvt[2]);
    schv[2] = (ptm->mat[0][2] * schvt[0]) + (ptm->mat[1][2] * schvt[1]) + (ptm->mat[2][2] * schvt[2]);
    tempa = elp->a;
    tempe2 = elp->e2;
    elp->a = ptm->radcur;
    elp->e2 = 0.;
    xyz2llh(schv,llh,elp);
    elp->a = tempa;
    elp->e2 = tempe2;
    schv[0] = ptm->radcur * llh[1];
    schv[1] = ptm->radcur * llh[0];
    schv[2] = llh[2];
}

// --------------- CUDA FUNCTIONS ------------------

__global__ void runTopo(struct Orbit orbit, struct OutputImgArrs outImgArrs, struct InputImgArrs inImgArrs, long NPIXELS, long OFFSET) {
    long pixel = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (pixel < NPIXELS) { // Make sure we're not operating on a non-existent pixel
        
        double enumat[3][3];
        double xyzsat[3], velsat[3], llhsat[3], vhat[3], that[3], chat[3], nhat[3];
        double llh[3], llh_prev[3], xyz[3], xyz_prev[3], sch[3], enu[3], delta[3];
        double line, tline, vmag, height, dopfact, costheta, sintheta, alpha, beta;
        double demlat, demlon, cosalpha, aa, bb, enunorm;
        int iter;
        // Because the arrays get read from AND written to, use thread-specific vars until final assignment
        double thrd_z, thrd_zsch, thrd_lat, thrd_lon, thrd_distance, thrd_losang0, thrd_losang1;
        double thrd_incang0, thrd_incang1;
        int thrd_converge;
    
        struct Ellipsoid elp;
        struct Peg peg;
        struct PegTrans ptm;
    
        /* * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        *   double t0 = inpts_dbl[0];
        *   double prf = inpts_dbl[1];
        */
        elp.a = d_inpts_dbl[2];
        elp.e2 = d_inpts_dbl[3];
        peg.lat = d_inpts_dbl[4];
        peg.lon = d_inpts_dbl[5];
        peg.hdg = d_inpts_dbl[6];
        /*
        *   double ufirstlat = inpts_dbl[7];
        *   double ufirstlon = inpts_dbl[8];
        *   double deltalat = inpts_dbl[9];
        *   double deltalon = inpts_dbl[10];
        *   double wvl = inpts_dbl[11];
        *   double ilrl = inpts_dbl[12];
        *   double thresh = inpts_dbl[13];
        *
        *   int NazLooks = inpts_int[0];
        *   int width = inpts_int[1];
        *   int udemlength = inpts_int[2];
        *   int udemwidth = inpts_int[3];
        *   int numiter = inpts_int[4];
        *   int extraiter = inpts_int[5];
        *   int length = inpts_int[6];      NOT USED IN THIS KERNEL
        * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
    
        line = (pixel + OFFSET) / d_inpts_int[1];
        tline = d_inpts_dbl[0] + (d_inpts_int[0] * (line / d_inpts_dbl[1]));
        if (interpolateOrbit(&orbit,tline,xyzsat,velsat) != 0) {
            printf("Error getting state vector for bounds computation\n");
            //exit(1);
        }
        unitvec(velsat,vhat);
        vmag = norm(3,velsat);
        xyz2llh(xyzsat,llhsat,&elp);
        height = llhsat[2];
        tcnbasis(xyzsat,velsat,that,chat,nhat,&elp);
        peg.lat = llhsat[0];
        peg.lon = llhsat[1];
        radar2xyz(&peg,&elp,&ptm);
   
        thrd_converge = 0;
        thrd_z = 0.0;
        thrd_zsch = 0.0;
        thrd_lat = d_inpts_dbl[7] + (0.5 * d_inpts_dbl[9] * d_inpts_int[2]);
        thrd_lon = d_inpts_dbl[8] + (0.5 * d_inpts_dbl[10] * d_inpts_int[3]);
    
        dopfact = (0.5 * d_inpts_dbl[11] * (inImgArrs.dopline[pixel] / vmag)) * inImgArrs.rho[pixel];
    
        // START THE ITERATIONS
        for (iter=0; iter<=(d_inpts_int[4]+d_inpts_int[5]); iter++) {
            if (thrd_converge == 0) { // Designing this way helps prevent thread divergence as much as possible
                llh_prev[0] = thrd_lat / (180. / M_PI);
                llh_prev[1] = thrd_lon / (180. / M_PI);
                llh_prev[2] = thrd_z;
               
                costheta = 0.5 * (((height + ptm.radcur) / inImgArrs.rho[pixel]) + (inImgArrs.rho[pixel] / (height + ptm.radcur)) - 
                                    (((ptm.radcur + thrd_zsch) / (height + ptm.radcur)) * ((ptm.radcur + thrd_zsch) / inImgArrs.rho[pixel])));
                sintheta = sqrt(1.0 - pow(costheta,2));
                alpha = (dopfact - (costheta * inImgArrs.rho[pixel] * dot(nhat,vhat))) / dot(vhat,that);
                beta = -d_inpts_dbl[12] * sqrt((pow(inImgArrs.rho[pixel],2) * pow(sintheta,2)) - pow(alpha,2));
   
                delta[0] = (costheta * inImgArrs.rho[pixel] * nhat[0]) + (alpha * that[0]) + (beta * chat[0]);
                delta[1] = (costheta * inImgArrs.rho[pixel] * nhat[1]) + (alpha * that[1]) + (beta * chat[1]);
                delta[2] = (costheta * inImgArrs.rho[pixel] * nhat[2]) + (alpha * that[2]) + (beta * chat[2]);

                xyz[0] = xyzsat[0] + delta[0];
                xyz[1] = xyzsat[1] + delta[1];
                xyz[2] = xyzsat[2] + delta[2];
                xyz2llh(xyz,llh,&elp);
   
                thrd_lat = llh[0] * (180. / M_PI);
                thrd_lon = llh[1] * (180. / M_PI);
                demlat = ((thrd_lat - d_inpts_dbl[7]) / d_inpts_dbl[9]) + 1;
                demlat = fmax(demlat,1.);
                demlat = fmin(demlat,(d_inpts_int[2]-1.));
                demlon = ((thrd_lon - d_inpts_dbl[8]) / d_inpts_dbl[10]) + 1;
                demlon = fmax(demlon,1.);
                demlon = fmin(demlon,(d_inpts_int[3]-1.));
                thrd_z = interpolateDEM(inImgArrs.DEM,demlon,demlat,d_inpts_int[3],d_inpts_int[2]);
                thrd_z = fmax(thrd_z,-500.);
   
                llh[0] = thrd_lat / (180. / M_PI);
                llh[1] = thrd_lon / (180. / M_PI);
                llh[2] = thrd_z;
                llh2xyz(xyz,llh,&elp);
                xyz2sch(sch,xyz,&ptm,&elp);
                thrd_zsch = sch[2];

                thrd_distance = sqrt(pow((xyz[0]-xyzsat[0]),2) + pow((xyz[1]-xyzsat[1]),2) + pow((xyz[2]-xyzsat[2]),2)) - inImgArrs.rho[pixel];
                thrd_converge = (fabs(thrd_distance) <= d_inpts_dbl[13]);

                if ((thrd_converge == 0) && (iter > d_inpts_int[4])) { // Yay avoiding thread divergence!
                    llh2xyz(xyz_prev,llh_prev,&elp);
                    xyz[0] = 0.5 * (xyz_prev[0] + xyz[0]);
                    xyz[1] = 0.5 * (xyz_prev[1] + xyz[1]);
                    xyz[2] = 0.5 * (xyz_prev[2] + xyz[2]);
                    xyz2llh(xyz,llh,&elp);
                    thrd_lat = llh[0] * (180. / M_PI);
                    thrd_lon = llh[1] * (180. / M_PI);
                    thrd_z = llh[2];
                    xyz2sch(sch,xyz,&ptm,&elp);
                    thrd_zsch = sch[2];
                    thrd_distance = sqrt(pow((xyz[0]-xyzsat[0]),2) + pow((xyz[1]-xyzsat[1]),2) + pow((xyz[2]-xyzsat[2]),2)) - inImgArrs.rho[pixel];
                }
            }
        }
   
        // Final computation
        costheta = 0.5 * (((height + ptm.radcur) / inImgArrs.rho[pixel]) + (inImgArrs.rho[pixel] / (height + ptm.radcur)) -
                            (((ptm.radcur + thrd_zsch) / (height + ptm.radcur)) * ((ptm.radcur + thrd_zsch) / inImgArrs.rho[pixel])));
        sintheta = sqrt(1.0 - pow(costheta,2));
        alpha = (dopfact - (costheta * inImgArrs.rho[pixel] * dot(nhat,vhat))) / dot(vhat,that);
        beta = -d_inpts_dbl[12] * sqrt((pow(inImgArrs.rho[pixel],2) * pow(sintheta,2)) - pow(alpha,2));
        
        delta[0] = (costheta * inImgArrs.rho[pixel] * nhat[0]) + (alpha * that[0]) + (beta * chat[0]);
        delta[1] = (costheta * inImgArrs.rho[pixel] * nhat[1]) + (alpha * that[1]) + (beta * chat[1]);
        delta[2] = (costheta * inImgArrs.rho[pixel] * nhat[2]) + (alpha * that[2]) + (beta * chat[2]);
        
        xyz[0] = xyzsat[0] + delta[0];
        xyz[1] = xyzsat[1] + delta[1];
        xyz[2] = xyzsat[2] + delta[2];
        xyz2llh(xyz,llh,&elp);
    
        thrd_lat = llh[0] * (180. / M_PI);
        thrd_lon = llh[1] * (180. / M_PI);
        thrd_z = llh[2];
        thrd_distance = sqrt(pow((xyz[0]-xyzsat[0]),2) + pow((xyz[1]-xyzsat[1]),2) + pow((xyz[2]-xyzsat[2]),2)) - inImgArrs.rho[pixel];

        // Expanded from Linalg::enubasis/Linalg::tranmat
        enumat[0][0] = -sin(llh[1]);
        enumat[1][0] = -sin(llh[0]) * cos(llh[1]);
        enumat[2][0] = cos(llh[0]) * cos(llh[1]);
        enumat[0][1] = cos(llh[1]);
        enumat[1][1] = -sin(llh[0]) * sin(llh[1]);
        enumat[2][1] = cos(llh[0]) * sin(llh[1]);
        enumat[0][2] = 0.0;
        enumat[1][2] = cos(llh[0]);
        enumat[2][2] = sin(llh[0]);
   
        // Expanded from Linalg::matvec
        enu[0] = (enumat[0][0] * delta[0]) + (enumat[0][1] * delta[1]) + (enumat[0][2] * delta[2]);
        enu[1] = (enumat[1][0] * delta[0]) + (enumat[1][1] * delta[1]) + (enumat[1][2] * delta[2]);
        enu[2] = (enumat[2][0] * delta[0]) + (enumat[2][1] * delta[1]) + (enumat[2][2] * delta[2]);
        
        cosalpha = fabs(enu[2]) / norm(3,enu);
        thrd_losang0 = acos(cosalpha) * (180. / M_PI);
        thrd_losang1 = (atan2(-enu[1],-enu[0]) - (0.5*M_PI)) * (180. / M_PI);
        thrd_incang0 = acos(costheta) * (180. / M_PI);
        thrd_zsch = inImgArrs.rho[pixel] * sintheta;
    
        demlat = ((thrd_lat - d_inpts_dbl[7]) / d_inpts_dbl[9]) + 1;
        demlat = fmax(demlat,2.);
        demlat = fmin(demlat,(d_inpts_int[2]-1.));
        demlon = ((thrd_lon - d_inpts_dbl[8]) / d_inpts_dbl[10]) + 1;
        demlon = fmax(demlon,2.);
        demlon = fmin(demlon,(d_inpts_int[3]-1.));
    
        aa = interpolateDEM(inImgArrs.DEM,(demlon-1.),demlat,d_inpts_int[3],d_inpts_int[2]);
        bb = interpolateDEM(inImgArrs.DEM,(demlon+1.),demlat,d_inpts_int[3],d_inpts_int[2]);
        alpha = ((bb - aa) * (180. / M_PI)) / (2.0 * (elp.a / sqrt(1.0 - (elp.e2 * pow(sin(thrd_lat / (180. / M_PI)),2)))) * d_inpts_dbl[10]);
    
        aa = interpolateDEM(inImgArrs.DEM,demlon,(demlat-1.),d_inpts_int[3],d_inpts_int[2]);
        bb = interpolateDEM(inImgArrs.DEM,demlon,(demlat+1.),d_inpts_int[3],d_inpts_int[2]);
        beta = ((bb - aa) * (180. / M_PI)) / (2.0 * ((elp.a * (1.0 - elp.e2)) / pow((1.0 - (elp.e2 * pow(sin(thrd_lat / (180. / M_PI)),2))),1.5)) * d_inpts_dbl[9]);
    
        enunorm = norm(3,enu);
        enu[0] = enu[0] / enunorm;
        enu[1] = enu[1] / enunorm;
        enu[2] = enu[2] / enunorm;
        costheta = ((enu[0] * alpha) + (enu[1] * beta) - enu[2]) / sqrt(1.0 + pow(alpha,2) + pow(beta,2));
        thrd_incang1 = acos(costheta) * (180. / M_PI);
    
        // Leave out masking stuff for now (though it's doable)
    
        // Finally write to reference arrays
        outImgArrs.lat[pixel] = thrd_lat;
        outImgArrs.lon[pixel] = thrd_lon;
        outImgArrs.z[pixel] = thrd_z;
        //outImgArrs.zsch[pixel] = thrd_zsch;
        outImgArrs.losang[2*pixel] = thrd_losang0;
        outImgArrs.losang[(2*pixel)+1] = thrd_losang1;
        outImgArrs.incang[2*pixel] = thrd_incang0;
        outImgArrs.incang[(2*pixel)+1] = thrd_incang1;
    }
}

// --------------- CPU HELPER FUNCTIONS -----------------

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void setOrbit(struct Orbit *orb) {
    orb->svs = (struct stateVector *)malloc(orb->nVec * sizeof(struct stateVector));
}

void freeOrbit(struct Orbit *orb) {
    free(orb->svs);
}

size_t getDeviceMem() {
    size_t freeByte, totalByte;
    cudaMemGetInfo(&freeByte, &totalByte);
    totalByte = (totalByte / 1e9) * 1e9; // Round down to nearest GB
    return totalByte;
}

// --------------- C FUNCTIONS ----------------

void runGPUTopo(long nBlock, long numPix, double *h_inpts_dbl, int *h_inpts_int, float *h_DEM, double *h_rho, double *h_dopline, int h_orbNvec, double *h_orbSvs, double **accArr) {

    //double *h_lat, *h_lon, *h_z, *h_incang, *h_losang; // , *h_zsch;
    double iStartCpy, iStartRun, iEndRun, iEndCpy;
    int i;

    struct stateVector *d_svs;
    double *d_rho, *d_dopline, *d_lat, *d_lon, *d_z, *d_incang, *d_losang; // , *d_zsch;
    float *d_DEM;

    struct InputImgArrs inImgArrs;
    struct OutputImgArrs outImgArrs;
    struct Orbit orbit;

    cudaSetDevice(0);

    printf("    Allocating host and general GPU memory...\n");
    
    size_t nb_pixels = numPix * sizeof(double);    // size of rho/dopline/lat/lon/z/zsch/incang/losang
    size_t nb_DEM = h_inpts_int[3] * h_inpts_int[2] * sizeof(float);    // size of DEM
    
    /*
    h_lat = (double *)malloc(nb_pixels);
    h_lon = (double *)malloc(nb_pixels);
    h_z = (double *)malloc(nb_pixels);
    //h_zsch = (double *)malloc(nb_pixels);
    h_incang = (double *)malloc(2 * nb_pixels);
    h_losang = (double *)malloc(2 * nb_pixels);
    */

    orbit.nVec = h_orbNvec;
    setOrbit(&orbit);
    for (i=0; i<h_orbNvec; i++) {
        orbit.svs[i].t = h_orbSvs[7*i];
        orbit.svs[i].px = h_orbSvs[(7*i)+1];
        orbit.svs[i].py = h_orbSvs[(7*i)+2];
        orbit.svs[i].pz = h_orbSvs[(7*i)+3];
        orbit.svs[i].vx = h_orbSvs[(7*i)+4];
        orbit.svs[i].vy = h_orbSvs[(7*i)+5];
        orbit.svs[i].vz = h_orbSvs[(7*i)+6];
    }
    cudaMalloc((void**)&d_svs, (orbit.nVec*sizeof(struct stateVector)));
    cudaMalloc((double**)&d_rho, nb_pixels);
    cudaMalloc((double**)&d_dopline, nb_pixels);
    cudaMalloc((float**)&d_DEM, nb_DEM);

    printf("    Copying general memory to GPU...\n");

    iStartCpy = cpuSecond();
    cudaMemcpy(d_svs, orbit.svs, (orbit.nVec*sizeof(struct stateVector)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho, h_rho, nb_pixels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dopline, h_dopline, nb_pixels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_DEM, h_DEM, nb_DEM, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_inpts_dbl, h_inpts_dbl, (14*sizeof(double)));
    cudaMemcpyToSymbol(d_inpts_int, h_inpts_int, (7*sizeof(int)));
    freeOrbit(&orbit);
    
    orbit.svs = d_svs;
    inImgArrs.DEM = d_DEM;
    inImgArrs.rho = d_rho;
    inImgArrs.dopline = d_dopline;

    printf("    Allocating block memory (%d pixels per image)...\n", numPix);
    
    cudaMalloc((double**)&d_lat, nb_pixels);
    cudaMalloc((double**)&d_lon, nb_pixels);
    cudaMalloc((double**)&d_z, nb_pixels);
    //cudaMalloc((double**)&d_zsch, nb_pixels);
    cudaMalloc((double**)&d_incang, (2*nb_pixels));
    cudaMalloc((double**)&d_losang, (2*nb_pixels));
    
    outImgArrs.lat = d_lat;
    outImgArrs.lon = d_lon;
    outImgArrs.z = d_z;
    outImgArrs.incang = d_incang;
    outImgArrs.losang = d_losang;
    //outImgArrs.zsch = d_zsch;

    dim3 block(THRD_PER_BLOCK);
    dim3 grid((numPix + (THRD_PER_BLOCK - 1)) / THRD_PER_BLOCK); // == ceil(numPix / THRD_PER_BLOCK), preserves warp sizing
    if ((grid.x * THRD_PER_BLOCK) > numPix) printf("    (NOTE: There will be %d 'empty' threads per image block).\n", ((grid.x*THRD_PER_BLOCK)-numPix));

    if (nBlock > -1) printf("    Starting GPU Topo for block %d...\n", nBlock);
    else printf("    Starting GPU Topo for remaining lines...\n");

    iStartRun = cpuSecond();
    if (nBlock > -1) runTopo <<<grid, block>>>(orbit, outImgArrs, inImgArrs, numPix, (nBlock*numPix));
    else {
        long offset = abs(nBlock);
        runTopo <<<grid, block>>>(orbit, outImgArrs, inImgArrs, numPix, offset);
    }

    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize(); // Double-duty of also waiting for the Topo algorithm to finish
    if (errSync != cudaSuccess) {
        printf("    Sync kernel error: %s\n", cudaGetErrorString(errSync));
    } if (errAsync != cudaSuccess) {
        printf("    Async kernel error: %s\n", cudaGetErrorString(errAsync));
    }

    iEndRun = cpuSecond();
    if (nBlock > -1) printf("    GPU finished block %d in %f s.\n", nBlock, (iEndRun-iStartRun));
    else printf("    GPU finished remaining lines in %f s.\n", (iEndRun-iStartRun));
    
    printf("    Copying memory back to host...\n");

    cudaMemcpy(accArr[0], outImgArrs.lat, nb_pixels, cudaMemcpyDeviceToHost); // Copy memory from device to host with offset
    cudaMemcpy(accArr[1], outImgArrs.lon, nb_pixels, cudaMemcpyDeviceToHost);
    cudaMemcpy(accArr[2], outImgArrs.z, nb_pixels, cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_zsch, outImgArrs.zsch, nb_pixels, cudaMemcpyDeviceToHost);
    cudaMemcpy(accArr[3], outImgArrs.incang, (2*nb_pixels), cudaMemcpyDeviceToHost);
    cudaMemcpy(accArr[4], outImgArrs.losang, (2*nb_pixels), cudaMemcpyDeviceToHost);

    iEndCpy = cpuSecond();
    if (nBlock > -1) printf("    GPU finished block %d (with memory copies) in %f s.\n", nBlock, (iEndCpy-iStartCpy));
    else printf("    GPU finished remaining lines (with memory copies) in %f s.\n", (iEndCpy-iStartCpy));

    printf("    Cleaning device memory and returning to main Topo function...\n");
    cudaFree(d_svs);
    cudaFree(d_rho);
    cudaFree(d_dopline);
    cudaFree(d_lat);
    cudaFree(d_lon);
    cudaFree(d_z);
    //cudaFree(d_zsch);
    cudaFree(d_incang);
    cudaFree(d_losang);
    cudaFree(d_DEM);
    cudaDeviceReset();

    /*
    accArr[0] = h_lat;
    accArr[1] = h_lon;
    accArr[2] = h_z;
    accArr[3] = h_incang;
    accArr[4] = h_losang;
    */
    //accArr[5] = h_zsch;   // Won't be used until we add the masking stuff
}

