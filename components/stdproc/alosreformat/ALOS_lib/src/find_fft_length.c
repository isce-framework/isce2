#include"stdio.h"
#include"stdlib.h"
#include"math.h"
#include"image_sio.h"
#include"lib_functions.h"
/*-----------------------------------------------------------------------*/
int	find_fft_length(int	n)
{
int	nfft;

	nfft = 2;

	while (nfft < n) nfft = 2*nfft;

	if (debug) fprintf(stderr,"find_fft_length:\n...data length n %d nfft %d \n\n",n,nfft);

	return(nfft);
}
/*-----------------------------------------------------------------------*/
