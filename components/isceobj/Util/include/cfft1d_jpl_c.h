/*
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 *                        NASA Jet Propulsion Laboratory
 *                      California Institute of Technology
 *                      (C) 2004-2005  All Rights Reserved
 *
 * <LicenseText>
 *
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

#if !defined(MROIPAC_CFFT1D_JPL_H)
#define MROIPAC_CFFT1D_JPL_H


/* ---------------- symbol mappings ---------------- */


#if defined(NEEDS_F77_TRANSLATION)

#if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
#define cfft1d_jpl cfft1d_jpl_

#elif defined(F77EXTERNS_NOTRAILINGBAR)
#define cfft1d_jpl cfft1d_jpl

#elif defined(F77EXTERNS_EXTRATRAILINGBAR)
#define cfft1d_jpl cfft1d_jpl__

#elif defined(F77EXTERNS_UPPERCASE_NOTRAILINGBAR)
#define cfft1d_jpl CFFT1D_JPL

#elif defined(F77EXTERNS_COMPAQ_F90)
#define cfft1d_jpl cfft1d_jpl_

#else /* no defined F77EXTERNS */
#error Unknown translation for FORTRAN external symbols
#endif /* if defined F77EXTERNS */

#endif /* NEEDS_F77_TRANSLATION */

/* ------------------------------------------------- */

#ifdef __cplusplus
extern "C"
{
#endif
  void cfft1d_jpl(int *n, float *c, int *dir);
#ifdef __cplusplus
}
#endif


/* ----------------  FFTW library ---------------- */

#ifdef WITH_FFTW
#include <fftw3.h>

/*  symbol mappings for external FFTW library */

#if defined(FFTW_NEEDS_F77_TRANSLATION)

#if defined(FFTW_F77EXTERNS_LOWERCASE_TRAILINGBAR)
#define sfftw_plan_dft_1d_f sfftw_plan_dft_1d_
#define sfftw_execute_dft_f sfftw_execute_dft_

#elif defined(FFTW_F77EXTERNS_NOTRAILINGBAR)
#define sfftw_plan_dft_1d_f sfftw_plan_dft_1d
#define sfftw_execute_dft_f sfftw_execute_dft

#elif defined(FFTW_F77EXTERNS_EXTRATRAILINGBAR)
#define sfftw_plan_dft_1d_f sfftw_plan_dft_1d__
#define sfftw_execute_dft_f sfftw_execute_dft__

#elif defined(FFTW_F77EXTERNS_UPPERCASE_NOTRAILINGBAR)
#define sfftw_plan_dft_1d_f SFFTW_PLAN_DFT_1D
#define sfftw_execute_dft_f SFFTW_EXECUTE_DFT

#elif defined(FFTW_F77EXTERNS_COMPAQ_F90)
#define sfftw_plan_dft_1d_f sfftw_plan_dft_1d_
#define sfftw_execute_dft_f sfftw_execute_dft_

#else /* no defined F77EXTERNS */
#error Unknown translation for FORTRAN external symbols
#endif /* if defined F77EXTERNS */

#endif /* FFTW_NEEDS_F77_TRANSLATION */

#endif /* WITH_FFTW */


/* ----------------  HPUX FFT library ---------------- */

#ifdef WITH_HPUX_FFT

/*  symbol mappings for external HPUX FFT library */

#if defined(HPUX_FFT_NEEDS_F77_TRANSLATION)

#if defined(HPUX_FFT_F77EXTERNS_LOWERCASE_TRAILINGBAR)
#define c1dfft_f c1dfft_

#elif defined(HPUX_FFT_F77EXTERNS_NOTRAILINGBAR)
#define c1dfft_f c1dfft

#elif defined(HPUX_FFT_F77EXTERNS_EXTRATRAILINGBAR)
#define c1dfft_f c1dfft__

#elif defined(HPUX_FFT_F77EXTERNS_UPPERCASE_NOTRAILINGBAR)
#define c1dfft_f C1DFFT

#elif defined(HPUX_FFT_F77EXTERNS_COMPAQ_F90)
#define c1dfft_f c1dfft_

#else /* no defined F77EXTERNS */
#error Unknown translation for FORTRAN external symbols
#endif /* if defined F77EXTERNS */

#endif /* HPUX_FFT_NEEDS_F77_TRANSLATION */

#endif /* WITH_HPUX_FFT */


/* ----------------  Irix FFT library ---------------- */

#ifdef WITH_IRIX_FFT
#include <fft.h>

/*  symbol mappings for external IRIX FFT library */

#if defined(IRIX_FFT_NEEDS_F77_TRANSLATION)

#if defined(IRIX_FFT_F77EXTERNS_LOWERCASE_TRAILINGBAR)
#define cfft1di_f cfft1di_
#define cfft1d_f cfft1d_

#elif defined(IRIX_FFT_F77EXTERNS_NOTRAILINGBAR)
#define cfft1di_f cfft1di
#define cfft1d_f cfft1d

#elif defined(IRIX_FFT_F77EXTERNS_EXTRATRAILINGBAR)
#define cfft1di_f cfft1di__
#define cfft1d_f cfft1d__

#elif defined(IRIX_FFT_F77EXTERNS_UPPERCASE_NOTRAILINGBAR)
#define cfft1di_f CFFT1DI
#define cfft1d_f CFFT1D

#elif defined(IRIX_FFT_F77EXTERNS_COMPAQ_F90)
#define cfft1di_f cfft1di_
#define cfft1d_f cfft1d_

#else /* no defined F77EXTERNS */
#error Unknown translation for FORTRAN external symbols
#endif /* if defined F77EXTERNS */

#endif /* IRIX_FFT_NEEDS_F77_TRANSLATION */

#endif /* WITH_IRIX_FFT */


/* ---------------- SunOS FFT library ---------------- */

#ifdef WITH_SUNOS_FFT

/*  symbol mappings for external SUNOS FFT library */

#if defined(SUNOS_FFT_NEEDS_F77_TRANSLATION)

#if defined(SUNOS_FFT_F77EXTERNS_LOWERCASE_TRAILINGBAR)
#define cfft1d_sun_f cfft1d_sun_

#elif defined(SUNOS_FFT_F77EXTERNS_NOTRAILINGBAR)
#define cfft1d_sun_f cfft1d_sun

#elif defined(SUNOS_FFT_F77EXTERNS_EXTRATRAILINGBAR)
#define cfft1d_sun_f cfft1d_sun__

#elif defined(SUNOS_FFT_F77EXTERNS_UPPERCASE_NOTRAILINGBAR)
#define cfft1d_sun_f CFFT1D_SUN

#elif defined(SUNOS_FFT_F77EXTERNS_COMPAQ_F90)
#define cfft1d_sun_f cfft1d_sun_

#else /* no defined F77EXTERNS */
#error Unknown translation for FORTRAN external symbols
#endif /* if defined F77EXTERNS */

#endif /* SUNOS_FFT_NEEDS_F77_TRANSLATION */

#endif /* WITH_SUNOS_FFT */


/* ---------------- JPL FFT library ---------------- */

/* #### CURRENTLY NOT IN USE HERE #### */

#ifdef WITH_JPL_FFT

/* Since the JPL FFT Fortran routines will be Pyre-compiled, the symbol mappings
 * will match that of those defined using Pyre's <portinfo> constructs.
 * Therefore, NEEDS_F77_TRANSLATION and the F77EXTERNS_ will be defined for
 * us by the include of <portinfo>.
 */
//#include <portinfo>

#if defined(NEEDS_F77_TRANSLATION)

#if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)

#elif defined(F77EXTERNS_NOTRAILINGBAR)

#elif defined(F77EXTERNS_EXTRATRAILINGBAR)

#elif defined(F77EXTERNS_UPPERCASE_NOTRAILINGBAR)

#elif defined(F77EXTERNS_COMPAQ_F90)

#else /* no defined F77EXTERNS */
#error Unknown translation for FORTRAN external symbols
#endif /* if defined F77EXTERNS */

#endif /* NEEDS_F77_TRANSLATION */

#endif /* WITH_JPL_FFT */




#endif /* MROIPAC_CFFT1D_JPL_H */
