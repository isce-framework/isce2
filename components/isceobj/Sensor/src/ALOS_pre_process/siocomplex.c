#include "image_sio.h"
#include "siocomplex.h"
#include <math.h>

fcomplex_sio Cmul(fcomplex_sio x, fcomplex_sio y)
{
    fcomplex_sio z;
    z.r = x.r*y.r - x.i*y.i;
    z.i = x.i*y.r + x.r*y.i;
    return z;
}

fcomplex_sio Cexp(float theta)
{
    fcomplex_sio z;
    z.r = cos(theta);
    z.i = sin(theta);
    return z;
}

fcomplex_sio Conjg(fcomplex_sio z)
{
    fcomplex_sio x;
    x.r = z.r;
    x.i = -z.i;
    return x;
}

fcomplex_sio RCmul(float a, fcomplex_sio z)
{
    fcomplex_sio x;
    x.r = a*z.r;
    x.i = a*z.i;
    return x;
}

fcomplex_sio Cadd(fcomplex_sio x, fcomplex_sio y)
{
    fcomplex_sio z;
    z.r = x.r + y.r;
    z.i = x.i + y.i;
    return z;
}

float Cabs(fcomplex_sio z)
{
    return hypot(z.r, z.i);
}
