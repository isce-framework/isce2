#ifndef dopiqFortTrans_h
#define dopiqFortTrans_h

#if defined(NEEDS_F77_TRANSLATION)
        #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
                #define dopiq_f dopiq_
                #define setLineLength_f setlinelength_
                #define setLineHeaderLength_f setlineheaderlength_
                #define setLastSample_f setlastsample_
                #define setStartLine_f setstartline_
                #define setNumberOfLines_f setnumberoflines_
                #define setMean_f setmean_
                #define setPRF_f setprf_
                #define getAcc_f get_acc_
                #define allocate_acc_f allocate_acc_
                #define deallocate_acc_f deallocate_acc_
        #else
                #error Unknown translation for FORTRAN external symbols
        #endif
#endif
#endif //dopiqFortTrans_h
