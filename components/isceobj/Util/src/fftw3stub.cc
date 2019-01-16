/* Stub to provide Fortran interface to FFTW when it has been
   configured with different name mangeling than ROI
*/

#if defined(FFTW) || defined(HAVE_FFTW)
/*     NOTE: Above condition must match same test in cfft1d_JPL.F
 */

#include "config.h"

#include <fftw3.h>

typedef float R;
#define CONCAT(prefix, name) prefix ## name
#define X(name) CONCAT(fftwf_, name)
typedef R C[2];


/*
#if defined(F77_FUNC)
#  define F77(a, A) F77_FUNC(a, A)
#endif
*/

/* ifort default name mangling a ## _ */
/* #define F77(a, A) a ## _ */
#define F77(a, A) FC_FUNC_(a, A)
#ifdef __cplusplus
extern "C" /* prevent C++ name mangling */
#endif


void F77(sfftw_plan_dft_1d, SFFTW_PLAN_DFT_1D)(X(plan) *p, int *n, C *in, C *out,
                                   int *sign, int *flags)
{
     *p = X(plan_dft_1d)(*n, in, out, *sign, *flags);
}


void F77(sfftw_execute_dft, SFFTW_EXECUTE_DFT)(X(plan) * const p, C *in, C *out){
  X(execute_dft)(*p, in, out);
}

/* end #if defined(FFTW) || defined(HAVE_FFTW) */
#endif
