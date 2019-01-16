#include <stdio.h>

void orbithermite_(double *x, double *v,double *t,double *time,double *xx,double *vv);

int
orbitHermite_C(double *x, double *v, double *t, double * ptime, double *xx, double *vv)
{

    // x and v are in row major order, which is OK since the matrices expected by orbithermite_() are the transpose
    // of what you would expect
	// xx and vv are the outputs
	orbithermite_(x,v,t,ptime,xx,vv);
	return 0;
}
