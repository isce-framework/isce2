#include	<unistd.h>
#include	<sys/time.h>
#include	<sys/times.h>
#include	<sys/resource.h>

#if defined(NEEDS_F77_TRANSLATION)

#if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)

#define secondo secondo_
#define wc_second wc_second_
#define us_second us_second_

#elif defined(F77EXTERNS_NOTRAILINGBAR)

#define secondo secondo
#define wc_second wc_second
#define us_second us_second

#elif defined(F77EXTERNS_EXTRATRAILINGBAR)

#define secondo secondo__
#define wc_second wc_second__
#define us_second us_second__

#elif defined(F77EXTERNS_UPPERCASE_NOTRAILINGBAR)

#define secondo SECONDO
#define wc_second WC_SECOND
#define us_second US_SECOND

#elif defined(F77EXTERNS_COMPAQ_F90)

#define secondo secondo_
#define wc_second wc_second_
#define us_second us_second_


#else
#error Unknown translation for FORTRAN external symbols
#endif

#endif


/* The same code is used for both C and Fortran entry points.
 */
#define	WC_GUTS								\
									\
  static int     first = 1;						\
  static double  t0;							\
  struct timeval s_val;							\
									\
  gettimeofday(&s_val,0);						\
  if (first) {								\
    t0    = (double) s_val.tv_sec + 0.000001*s_val.tv_usec;		\
    first = 0;								\
    return (0.0);							\
  }									\
  return ((double) s_val.tv_sec + 0.000001*s_val.tv_usec - t0);
  
/* Returns the current value of the wall clock timer.
 * Fortran or C entry point.
 */
double
wc_second()

{
  WC_GUTS;
}
  
#define	US_GUTS								\
									\
  static int	first = 1;						\
  static double	t0;							\
  struct rusage	ru;							\
  double	tu, ts;							\
									\
  getrusage(RUSAGE_SELF,&ru);						\
  if (first) {								\
    t0    = ru.ru_utime.tv_sec + 1.0e-6*ru.ru_utime.tv_usec		\
          + ru.ru_stime.tv_sec + 1.0e-6*ru.ru_stime.tv_usec;		\
    first = 0;								\
    return (0.0);							\
  }									\
									\
  tu = ru.ru_utime.tv_sec + 1.0e-6*ru.ru_utime.tv_usec;			\
  ts = ru.ru_stime.tv_sec + 1.0e-6*ru.ru_stime.tv_usec;			\
									\
  return (tu + ts - t0);

/* Returns the current value of the user+system timer.  Fortran or C entry point.
 */
double
us_second()

{
  US_GUTS;
}

/* Returns the current value of the wall clock timer, or
 * user+system timer depending on the valueof tmode: 
 * less than zero the wall-clock timer, and greater than zero
 * user+system time.
 * If/when called from C, tmode must be passed by reference.
 */
double
secondo(

int	*ptmode)

{
  int	tmode = *ptmode;

  if (tmode > 0) {
    US_GUTS;
  } else if (tmode < 0) {
    WC_GUTS;
  } else if (tmode == 0) {
    printf("Invalid tmode.\n");
    return(0.0);
  }
  /* XXBUG - bswift 11/4/04  'if (tmode == 0)' should be removed to prevent compiler warning about no return for non-void function */

}
