
#ifndef driverCCFortTrans_h
#define driverCCFortTrans_h

        #if defined(NEEDS_F77_TRANSLATION)

                #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
                        #define testImageSetGet_f testimagesetget_
                #else
                        #error Unknown traslation for FORTRAN external symbols
                #endif

        #endif

#endif //driverCCFortTrans_h
