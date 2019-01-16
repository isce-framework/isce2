/************************************************************************
* cfft1d is a subroutine used to call and initialize FFT routines from  *
* fftpack.c   The calls are almost identical to the old Sun perflib     *
************************************************************************/
/************************************************************************
* Creator: David T. Sandwell	(Scripps Institution of Oceanography    *
* Date   : 12/27/96                                                     *
* Date   : 09/15/07 re-worked by Rob Mellors                            *
* Date   : 10/16/07 re-worked by David Sandwells  to use pointers       *
************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "image_sio.h"
#include "siocomplex.h"
#include "lib_functions.h"
 

/*----------------------------------------------------------------------------*/
void cfft1d_(int *np, fcomplex *c, int *dir)
{

	static float *work;
	static int nold = 0;
	int i,n;

/* Initialize work array with sines and cosines to save CPU time later 
   This is done when the length of the FFT has changed or when *dir == 0. */
	
	n = *np;

	if((n != nold) || (*dir == 0)){
	  if(nold != 0) free((char *) work);
	  if((work = (float *) malloc((4*n+30)*sizeof(float))) == NULL) die("Sorry, can't allocate mem","");

	  cffti(n, work);

	  nold = n;
	}

/* Do forward transform with NO normalization.  Forward is exp(+i*k*x) */

	if (*dir == -1) cfftf(n, c, work); 

/* Do inverse transform with normalization.  Inverse is exp(-i*k*x) */

	if (*dir == 1){
	  	cfftb(n, c, work);
          	for (i=0; i<n; i++) {
			c[i].i = c[i].i/(1.0*n);
			c[i].r = c[i].r/(1.0*n);
			}
		}
}
