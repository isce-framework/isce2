#include <portinfo.h>

#if defined(NEEDS_F77_TRANSLATION)

#if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)

#define inventoryValChar inventoryValChar_
#define inventoryValNum inventoryValNum_
#define inventoryValNum2 inventoryValNum2_
#define inventoryValNum3 inventoryValNum3_
#define inventoryValNum4 inventoryValNum4_
#define inventoryValArray inventoryValArray_

#elif defined(F77EXTERNS_NOTRAILINGBAR)

#define inventoryValChar inventoryValChar
#define inventoryValNum inventoryValNum
#define inventoryValNum2 inventoryValNum2
#define inventoryValNum3 inventoryValNum3
#define inventoryValNum4 inventoryValNum4
#define inventoryValArray inventoryValArray

#elif defined(F77EXTERNS_EXTRATRAILINGBAR)

#define inventoryValChar inventoryValChar__
#define inventoryValNum inventoryValNum__
#define inventoryValNum2 inventoryValNum2__
#define inventoryValNum3 inventoryValNum3__
#define inventoryValNum4 inventoryValNum4__
#define inventoryValArray inventoryValArray__

#elif defined(F77EXTERNS_UPPERCASE_NOTRAILINGBAR)

#define inventoryValChar INVENTORYVALCHAR
#define inventoryValNum INVENTORYVALNUM
#define inventoryValNum2 INVENTORYVALNUM2
#define inventoryValNum3 INVENTORYVALNUM3
#define inventoryValNum4 INVENTORYVALNUM4
#define inventoryValArray INVENTORYVALARRAY

#elif defined(F77EXTERNS_COMPAQ_F90)

// symbols that contain underbars get two underbars at the end
// symbols that do not contain underbars get one underbar at the end
// this applies to the FORTRAN external, not the local macro alias!!!

#define inventoryValChar inventoryValChar_
#define inventoryValNum inventoryValNum_
#define inventoryValNum2 inventoryValNum2_
#define inventoryValNum3 inventoryValNum3_
#define inventoryValNum4 inventoryValNum4_
#define inventoryValArray inventoryValArray_

#else
#error Unknown translation for FORTRAN external symbols
#endif

#endif
