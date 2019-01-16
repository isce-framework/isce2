#include "image_sio.h"
#include "siocomplex.h"
#include <math.h>

fcomplex Cmul(fcomplex x, fcomplex y)
{
    fcomplex z;
    z.r = x.r*y.r - x.i*y.i;
    z.i = x.i*y.r + x.r*y.i;
    return z;
}

fcomplex Cexp(float theta)
{
    fcomplex z;
    z.r = cos(theta);
    z.i = sin(theta);
    return z;
}

fcomplex Conjg(fcomplex z)
{
    fcomplex x;
    x.r = z.r;
    x.i = -z.i;
    return x;
}

fcomplex RCmul(float a, fcomplex z)
{
    fcomplex x;
    x.r = a*z.r;
    x.i = a*z.i;
    return x;
}

fcomplex Cadd(fcomplex x, fcomplex y)
{
    fcomplex z;
    z.r = x.r + y.r;
    z.i = x.i + y.i;
    return z;
}

float Cabs(fcomplex z)
{
    return hypot(z.r, z.i);
}
