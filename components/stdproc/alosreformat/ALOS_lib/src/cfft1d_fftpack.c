#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
/************************************************************************
* cfft1d is a subroutine used to call and initialize perflib Fortran FFT *
* routines.                                                             *
************************************************************************/
/************************************************************************
* Creator: David T. Sandwell	(Scripps Institution of Oceanography    *
* Date   : 12/27/96                                                     *
************************************************************************/
 
void die(char *, char*);
void cffti(int, float *);
void cfftf(int, complex float *, float *);
void cfftb(int, complex float *, float *);

void cfft1d(int np, complex float *c, int dir)
{

	static float *work;
	static int nold = 0;
	int i,n;

/* Initialize work array with sines and cosines to save CPU time later 
   This is done when the length of the FFT has changed or when dir == 0. */
        n = np;

	if((n != nold) || (dir == 0)){
	  if(nold != 0) free((char *) work);
	  if((work = (float *) malloc((4*n+30)*sizeof(float))) == NULL) die("Sorry, can't allocate mem","");

	  cffti(np, work);

	  nold = n;
	}

/* Do forward transform with NO normalization.  Forward is exp(+i*k*x) */

	if (dir == -1) cfftf(np, c, work); 

/* Do inverse transform with normalization.  Inverse is exp(-i*k*x) */

	if (dir == 1){
	  	cfftb(np, c, work);
          	for (i=0; i<np; i++) c[i] = c[i]/(1.0*np);
		}
}
