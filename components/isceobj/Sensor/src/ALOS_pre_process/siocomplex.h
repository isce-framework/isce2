#ifndef _COMPLEX_H
#define _COMPLEX_H

fcomplex_sio Cmul(fcomplex_sio x, fcomplex_sio y);
fcomplex_sio Cexp(float theta);
fcomplex_sio Conjg(fcomplex_sio z);
fcomplex_sio RCmul(float a, fcomplex_sio z);
fcomplex_sio Cadd(fcomplex_sio x, fcomplex_sio y);
float Cabs(fcomplex_sio z);

#endif /* _COMPLEX_H */
