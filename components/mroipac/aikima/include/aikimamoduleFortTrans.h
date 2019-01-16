#ifndef aikimamoduleFortTrans_h
#define aikimamoduleFortTrans_h

    #if defined(NEEDS_F77_TRANSLATION)

        #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)

            #define aikima_f aikima_
            #define setWidth_f setwidth_
            #define setLength_f setlength_
            #define setFirstPixelAcross_f setfirstpixelacross_
            #define setLastPixelAcross_f setlastpixelacross_
            #define setFirstLineDown_f setfirstlinedown_
            #define setLastLineDown_f setlastlinedown_
            #define setBlockSize_f setblocksize_
            #define setPadSize_f setpadsize_
            #define setNumberPtsPartial_f setnumberptspartial_
            #define setPrintFlag_f setprintflag_
            #define setThreshold_f setthreshold_

        #else
            #error Unknown translation for FORTRAN external symbols
        #endif

    #endif

#endif //aikimamoduleFortTrans_h
