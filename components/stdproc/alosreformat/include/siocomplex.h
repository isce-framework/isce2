#ifndef _COMPLEX_H
#define _COMPLEX_H

fcomplex Cmul(fcomplex x, fcomplex y);
fcomplex Cexp(float theta);
fcomplex Conjg(fcomplex z);
fcomplex RCmul(float a, fcomplex z);
fcomplex Cadd(fcomplex x, fcomplex y);
float Cabs(fcomplex z);

#endif /* _COMPLEX_H */
