#ifndef dopplerFortTrans_h
#define dopplerFortTrans_h

#if defined(NEEDS_F77_TRANSLATION)
        #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
                #define doppler_f doppler_
                #define setStartLine_f setstartline_
                #define setSamples_f setsamples_
                #define setLines_f setlines_
                #define get_r_fd_f get_r_fd_
                #define allocate_r_fd_f allocate_r_fd_
                #define deallocate_r_fd_f deallocate_r_fd_
        #else
                #error Unknown translation for FORTRAN external symbols
        #endif
#endif
#endif //dopplerFortTrans_h
