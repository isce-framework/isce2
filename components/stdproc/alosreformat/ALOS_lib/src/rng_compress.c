/************************************************************************
* rng_compress reduces the bandwidth of an array by 2 times by          *
*	low-pass filtering and decimating the wavenumber space          *
************************************************************************/
/************************************************************************
* Creator: David T. SandwellScripps Institution of Oceanography)	*
* Date   : 06/21/07							*
************************************************************************/
/************************************************************************
* Modification History							*
* 									*
* Date									*
************************************************************************/ 

#include"image_sio.h"
#include"siocomplex.h"
#include"lib_functions.h"

void rng_compress(fcomplex * cin, int nffti,fcomplex * cout, int nffto)
{
	int i, dir, n4;
        n4 = nffti/4;

/* do the forward fft */
        dir = -1;
	cfft1d_(&nffti,cin,&dir);

/* then move the input to the output 1 to 1 and 4 to 2 */
		
	for(i=0;i<n4;i++){
		cout[i].r=cin[i].r;
		cout[i].i=cin[i].i;
		cout[i+n4].r=cin[i+3*n4].r;
		cout[i+n4].i=cin[i+3*n4].i;
	}

/* now inverse fft */
	
        dir = 1;
	cfft1d_(&nffto,cout,&dir);
}
